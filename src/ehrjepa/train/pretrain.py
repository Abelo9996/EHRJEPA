"""The self-supervised pretraining loop.

``python -m ehrjepa.train.pretrain --config configs/pretrain_small.yaml [--override k=v ...]``

One step is: draw a batch of subject windows, sample context/target masks, run the
JEPA forward, add SIGReg, backward, clip, step, and (in ``ema`` target mode) move
the target encoder. Everything else here is bookkeeping: a cosine schedule with
linear warmup, gradient accumulation, resumable checkpoints, and per-step logging
to CSV and TensorBoard.

``objective.kind: ar`` swaps the middle of that sentence and nothing else. The
model becomes :class:`~ehrjepa.models.ar.EHRAR` -- the same embedding and encoder,
the encoder causal, a tied next-code head instead of a predictor -- and the step
becomes forward, cross-entropy, backward. Data, sampler, optimizer, schedule,
checkpoint format, diagnostics and logging path are the ones above, which is what
makes a matched-compute comparison between the two objectives mean anything.

The metrics worth watching are not the loss. A JEPA loss can fall to zero by
collapse, so every ``log_every`` steps the loop also records, on the current
batch's encoder outputs: the mean per-dimension standard deviation, the effective
rank (``exp`` of the entropy of the normalised singular values), and the gap
between the mean cosine of predictions with their own targets and with shuffled
targets. The gap is the part that cannot be faked -- a collapsed model scores
identically on matched and mismatched targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ehrjepa.data.cache import read_meta
from ehrjepa.data.dataset import EventSequenceDataset, collate_events
from ehrjepa.data.masking import sample_masks
from ehrjepa.models.ar import EHRAR
from ehrjepa.models.jepa import EHRJEPA, JEPAOutput, ema_momentum
from ehrjepa.objectives.ar import ARObjective
from ehrjepa.objectives.loss import JEPAObjective, collapse_diagnostics
from ehrjepa.train.config import PretrainConfig, load_config
from ehrjepa.utils.runtime import (
    autocast_for,
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    seed_everything,
)

__all__ = ["LOG_COLUMNS", "Trainer", "cosine_lr", "main"]

#: CSV columns, in order. TensorBoard gets the same scalars under ``train/``.
#: A run writes every column; the ones its objective does not produce stay empty
#: (``pred_loss``/``sigreg_*`` for ``ar``, ``ce``/``top1``/``top10`` for ``jepa``),
#: so one reader handles both kinds of ``metrics.csv``.
LOG_COLUMNS = (
    "step",
    "loss",
    "pred_loss",
    "sigreg_tokens",
    "sigreg_cls",
    "ce",
    "top1",
    "top10",
    "lr",
    "grad_norm",
    "tokens_per_s",
    "mean_std",
    "effective_rank",
    "cos_real",
    "cos_shuffled",
    "cos_gap",
    "ema_momentum",
    "peak_memory_mb",
    "elapsed_s",
)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int, min_ratio: float) -> float:
    """Linear warmup for ``warmup`` steps, then cosine decay to ``min_ratio * base_lr``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    scale = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * scale


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Decay matrices, do not decay biases, norms or embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "emb" in name or name.endswith("cls_token") or "mask_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class InfiniteBatchSampler:
    """A never-ending stream of index batches that is a pure function of the seed.

    Epoch ``e`` is ``randperm(n)`` under a generator seeded from ``(seed, e)``, cut
    into ``batch_size`` chunks with the ragged tail dropped. Because the stream is
    positional rather than stateful, resuming at global batch ``k`` is exactly
    "start at ``k``" -- no need to replay, and no dependence on the global RNG,
    which is what makes ``resume`` reproduce an uninterrupted run.
    """

    def __init__(self, n_items: int, batch_size: int, seed: int, start_batch: int = 0) -> None:
        if n_items < batch_size:
            raise ValueError(f"dataset has {n_items} items, fewer than batch_size {batch_size}")
        self.n_items = n_items
        self.batch_size = batch_size
        self.seed = seed
        self.start_batch = start_batch
        self.per_epoch = n_items // batch_size

    def permutation(self, epoch: int) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        return torch.randperm(self.n_items, generator=generator)

    def __iter__(self) -> Iterator[list[int]]:
        index = self.start_batch
        epoch = -1
        order: Tensor = torch.empty(0, dtype=torch.long)
        while True:
            current, offset = divmod(index, self.per_epoch)
            if current != epoch:
                epoch, order = current, self.permutation(current)
            lo = offset * self.batch_size
            yield order[lo : lo + self.batch_size].tolist()
            index += 1


class Trainer:
    """Owns the model, the data, the optimizer and the run directory."""

    def __init__(self, config: PretrainConfig) -> None:
        self.config = config
        run = config.run
        seed_everything(run.seed)
        self.device = resolve_device(run.device)
        self.out_dir = Path(run.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.meta = read_meta(config.data.cache_dir)
        self.dataset = EventSequenceDataset(
            config.data.cache_dir,
            config.data.split,
            max_len=config.data.max_len,
            min_len=config.data.min_len,
            sampling=config.data.sampling,
            seed=run.seed,
        )
        self.model_config = config.model_config(int(self.meta["vocab_size"]))
        self.kind = config.objective.kind
        self.model: nn.Module
        if self.kind == "ar":
            if not self.model_config.causal:
                # The AR objective is only defined against causal attention, and
                # a bidirectional encoder would read the answer off its input.
                print("[note] objective.kind=ar implies model.causal=true; enabling it", flush=True)
                self.model_config.causal = True
            self.model = EHRAR(self.model_config).to(self.device)
            self.objective: nn.Module = ARObjective().to(self.device)
        else:
            self.model = EHRJEPA(self.model_config).to(self.device)
            self.objective = JEPAObjective(config.objective).to(self.device)
        self.optimizer = torch.optim.AdamW(
            param_groups(self.model, config.optim.weight_decay),
            lr=config.optim.lr,
            betas=tuple(config.optim.betas),
            eps=config.optim.eps,
        )
        # Masks are drawn on the CPU from a dedicated generator, so the mask
        # stream is reproducible independently of model-side randomness.
        self.mask_generator = torch.Generator().manual_seed(run.seed + 1)
        # Diagnostics get their own stream so logging cannot shift training.
        self.diag_generator = torch.Generator().manual_seed(run.seed + 2)
        self.step = 0
        self._csv_path = self.out_dir / "metrics.csv"
        self._writer = None
        if run.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(str(self.out_dir / "tb"))
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[warn] tensorboard unavailable: {exc}")
        (self.out_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, default=str) + "\n"
        )
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(LOG_COLUMNS)
        self._peak_memory = 0

    # ------------------------------------------------------------------ #

    def data_loader(self, start_batch: int) -> DataLoader:
        """A loader whose batch stream starts at global batch ``start_batch``."""
        config = self.config
        sampler = InfiniteBatchSampler(
            len(self.dataset), config.run.batch_size, config.run.seed, start_batch
        )
        loader_kwargs: dict[str, object] = {}
        if config.data.num_workers > 0 and config.data.prefetch_factor:
            loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
            loader_kwargs["persistent_workers"] = True
        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            collate_fn=collate_events,
            num_workers=config.data.num_workers,
            # Its own generator: a DataLoader iterator draws a base seed for its
            # workers on construction, and taking that from the global stream
            # would shift every later draw by one on resume.
            generator=torch.Generator().manual_seed(config.run.seed + 3),
            **loader_kwargs,  # type: ignore[arg-type]
        )

    def to_device(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def state_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "model_config": vars(self.model_config),
            "vocab": {
                "vocab_size": self.meta["vocab_size"],
                "tokenizer_version": self.meta["tokenizer_version"],
                "cache_dir": str(self.config.data.cache_dir),
                "fit": self.meta.get("fit", {}),
            },
            # Everything stochastic in a step: the mask stream, the window
            # sampler, and the global generator SIGReg's directions come from.
            "rng": {
                "torch": torch.get_rng_state(),
                "mask": self.mask_generator.get_state(),
                "diag": self.diag_generator.get_state(),
                "dataset": self.dataset.rng_state(),
                "device": (
                    torch.mps.get_rng_state()
                    if self.device.type == "mps"
                    else torch.cuda.get_rng_state()
                    if self.device.type == "cuda"
                    else None
                ),
            },
        }

    def save_checkpoint(self, name: str | None = None) -> Path:
        path = self.out_dir / (name or f"step_{self.step:07d}.pt")
        torch.save(self.state_dict(), path)
        latest = self.out_dir / "latest.pt"
        torch.save(self.state_dict(), latest)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore weights, optimizer, step *and* every RNG stream.

        With ``num_workers=0`` this makes a resumed run bit-identical to the
        uninterrupted one: the batch stream is positional (see
        :class:`InfiniteBatchSampler`) and the three generators are restored
        exactly where they stopped.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
        rng = state.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            self.mask_generator.set_state(rng["mask"].cpu().to(torch.uint8))
            self.diag_generator.set_state(rng["diag"].cpu().to(torch.uint8))
            self.dataset.set_rng_state(rng["dataset"])
            if rng.get("device") is not None:
                device_state = rng["device"].cpu().to(torch.uint8)
                if self.device.type == "mps":
                    torch.mps.set_rng_state(device_state)
                elif self.device.type == "cuda":
                    torch.cuda.set_rng_state(device_state)
        else:  # pragma: no cover - checkpoints from before RNG capture
            self.mask_generator.manual_seed(self.config.run.seed + 1 + self.step)

    # ------------------------------------------------------------------ #

    def _forward(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], object]:
        if self.kind == "ar":
            output = self.model(batch)
            return dict(self.objective(output.logits, output.targets)), output
        context_mask, target_mask = sample_masks(
            batch["attention_mask"].cpu(),
            p_future=self.config.masking.p_future,
            generator=self.mask_generator,
            **self.config.masking.kwargs(),
        )
        context_mask = context_mask.to(self.device)
        target_mask = target_mask.to(self.device)
        output = self.model(batch, context_mask, target_mask)
        losses = self.objective(output)
        return losses, output

    def _diagnostics(self, output: object) -> dict[str, float]:
        """Collapse diagnostics for whichever forward just ran.

        The AR head has no predicted/target latent pair, so ``cos_*`` is reported
        as zero there; ``mean_std`` and ``effective_rank`` are computed on the
        encoder's token outputs exactly as for JEPA.
        """
        if isinstance(output, JEPAOutput):
            tokens, mask = output.context_tokens.detach(), output.context_mask
            predictions, targets = output.predictions.detach(), output.targets
        else:
            tokens, mask = output.tokens.detach(), output.valid_mask  # type: ignore[attr-defined]
            predictions = targets = tokens.new_zeros((0, tokens.shape[-1]))
        return collapse_diagnostics(
            tokens, mask, predictions, targets, generator=self.diag_generator
        )

    def train(self) -> dict[str, float]:
        cfg = self.config
        run = cfg.run
        reset_peak_memory(self.device)
        self.model.train()
        accum = max(1, cfg.optim.accum_steps)
        batches = iter(self.data_loader(self.step * accum))
        started = time.perf_counter()
        window_tokens = 0
        window_started = started
        last: dict[str, float] = {}

        while self.step < run.steps:
            lr = cosine_lr(
                self.step, run.steps, cfg.optim.lr, cfg.optim.warmup_steps, cfg.optim.min_lr_ratio
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            output: object | None = None
            for _ in range(accum):
                batch = self.to_device(next(batches))
                with autocast_for(self.device, run.precision):
                    losses, output = self._forward(batch)
                (losses["loss"] / accum).backward()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) / accum
                window_tokens += int(batch["attention_mask"].sum())

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
                if cfg.optim.grad_clip > 0
                else torch.zeros(())
            )
            self.optimizer.step()
            momentum = float("nan")
            if self.model.uses_ema:
                momentum = ema_momentum(
                    self.step, run.steps, self.model_config.ema_start, self.model_config.ema_end
                )
                self.model.update_ema(momentum)
            self.step += 1
            self._peak_memory = max(self._peak_memory, peak_memory_bytes(self.device))

            log_now = self.step % run.log_every == 0 or self.step == run.steps
            if log_now and output is not None:
                now = time.perf_counter()
                diagnostics = self._diagnostics(output)
                row = {
                    "step": self.step,
                    **totals,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_per_s": window_tokens / max(now - window_started, 1e-9),
                    **diagnostics,
                    "ema_momentum": momentum,
                    "peak_memory_mb": self._peak_memory / 2**20,
                    "elapsed_s": now - started,
                }
                self._log(row)
                last = {k: float(v) for k, v in row.items()}
                window_tokens = 0
                window_started = time.perf_counter()

            if run.ckpt_every > 0 and self.step % run.ckpt_every == 0:
                self.save_checkpoint()
            if run.max_seconds > 0 and time.perf_counter() - started > run.max_seconds:
                print(f"[stop] wall-clock budget reached at step {self.step}")
                break

        self.save_checkpoint("final.pt")
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        return last

    def _log(self, row: Mapping[str, float]) -> None:
        with self._csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([row.get(name, "") for name in LOG_COLUMNS])
        if self._writer is not None:
            for name, value in row.items():
                if name != "step" and isinstance(value, (int, float)) and math.isfinite(value):
                    self._writer.add_scalar(f"train/{name}", value, self.step)
        if self.kind == "ar":
            body = "ce {ce:.4f} top1 {top1:.3f} top10 {top10:.3f}"
        else:
            body = "pred {pred_loss:.4f} sig_tok {sigreg_tokens:.4f} sig_cls {sigreg_cls:.4f}"
        print(
            (
                "step {step:>6} loss {loss:.4f} " + body + " rank {effective_rank:.1f} "
                "std {mean_std:.3f} cosgap {cos_gap:+.4f} tok/s {tokens_per_s:,.0f}"
            ).format(**row),
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ehrjepa.train.pretrain",
        description="Self-supervised JEPA pretraining on an EHR event cache.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-path overrides, e.g. --override optim.lr=1e-4 run.steps=50",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, args.override)
    trainer = Trainer(config)
    counts = trainer.model.n_parameters()
    parts = " / ".join(
        f"{name} {counts[name]:,}"
        for name in ("embedding", "encoder", "predictor", "head")
        if name in counts
    )
    print(
        f"device={trainer.device} objective={trainer.kind} vocab={trainer.meta['vocab_size']} "
        f"subjects={len(trainer.dataset)} (dropped {trainer.dataset.n_dropped}) "
        f"params={counts['trainable']:,} ({parts})",
        flush=True,
    )
    if config.run.resume:
        trainer.load_checkpoint(config.run.resume)
        print(f"resumed from {config.run.resume} at step {trainer.step}", flush=True)
    final = trainer.train()
    print(json.dumps({"final": final, "params": counts}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
