"""End-to-end fine-tuning of a pretrained encoder on one downstream task.

A frozen linear probe (:mod:`ehrjepa.eval.probe`) answers "what is already in
the representation". The ICU literature does not report that number: it reports
what the encoder scores after the whole stack is trained on the task, against a
gradient-boosting model on tabular features and against the same architecture
trained from scratch. Those are different questions, and a checkpoint can win
one and lose the other -- a probe rewards a representation that is already
linearly separable, a fine-tune rewards one that is a good initialisation. This
module is the second question, so the two can be read side by side rather than
one standing in for the other.

Everything that is not the training is held identical to the probe:

* the encoder is rebuilt by :func:`ehrjepa.eval.probe.load_encoder`, so an
  ``ar``, ``nextlatent``, ``window`` or ``lm`` checkpoint loads its own class,
  and ``random_init`` copies the architecture and discards the weights;
* the window is the last ``max_len`` events **strictly before** the anchor, read
  through :class:`ehrjepa.eval.history.HistoryReader` -- the one leakage barrier
  this repository has;
* the readout is :func:`ehrjepa.eval.probe.pool` under the same ``features``
  (``auto`` resolves per checkpoint: ``last`` for causal, ``mean`` for
  bidirectional) and the same ``layer``;
* predictions come back for exactly the rows of the shared task frame the probe
  is scored on, in the same order, so :mod:`ehrjepa.eval.metrics`' paired
  bootstrap compares two models on identical subjects.

**The head** is ``LayerNorm -> Linear(width, 1)`` on the pooled row and nothing
else. A deeper head would make "the encoder is a good initialisation"
unfalsifiable: enough MLP on top of an untrained encoder fits any of these tasks
to some degree, and the control (``ft_random``) would stop being a control.

**The optimisation** is fixed, not searched: AdamW, ``2e-5`` on the encoder,
``1e-3`` on the head, ``1e-4`` on LoRA adapters when the encoder is a pretrained
language model, linear warmup over the first 5% of the planned steps then cosine
decay, gradient clipping at 1.0, ``batch_size`` (64) windows per step, at most
``max_epochs``
(5) passes with early stopping on tuning-split AUROC at patience 2, and the best
epoch's weights restored before the held-out pass. Two discriminative learning
rates because the head starts from noise and the encoder starts from a
pretrained optimum; the LoRA rate is lower again because a rank-8 adapter is the
only thing moving inside a 0.5B trunk. None of these is tuned per cell -- the
tuning split chooses *when to stop* and nothing else, which is the one
hyperparameter every ICU fine-tuning paper reports.

**Only the modules the forward pass touches are trained.** For the from-scratch
stack that is ``embed`` and ``encoder``; for the LM stack it is exactly the
parameters that stack's own construction left trainable -- LoRA, the projection,
its LayerNorm, the CLS row -- and *not* its code table, which is a tokenizer's
next-code output projection rather than an input embedding. The predictor, the
pretraining heads and any EMA target copy are frozen either way, because they
are not on the path from a window to a logit and counting them as trainable
would misstate what was fine-tuned.

**Multi-anchor tasks are not special-cased.** ``sepsis_6h`` draws up to eight
anchors per stay; every train anchor is a training row and every held-out anchor
is a scored row, exactly as in the probe. Rows within a stay are therefore not
independent -- in the loss as well as in the bootstrap -- which is a property of
the task, identical across every model in a run.

``random_init=True`` is the "train from scratch" arm: the same architecture, the
same head, the same schedule, from initialisation. A checkpoint whose fine-tune
does not beat it did not contribute an initialisation worth having.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

from ehrjepa.data.dataset import collate_events
from ehrjepa.eval.history import HistoryReader, anchor_minutes
from ehrjepa.eval.metrics import auroc
from ehrjepa.eval.probe import PROBE_LAYERS, load_encoder, n_features, pool
from ehrjepa.models.ar import EHRAR
from ehrjepa.models.jepa import EHRJEPA
from ehrjepa.train.pretrain import cosine_lr
from ehrjepa.utils.runtime import autocast_for, resolve_device

__all__ = [
    "BATCH_SIZE",
    "ENCODER_LR",
    "FineTuneModel",
    "FineTuned",
    "HEAD_LR",
    "LORA_LR",
    "MAX_EPOCHS",
    "PATIENCE",
    "build_model",
    "epoch_order",
    "fine_tune",
    "stop_early",
]

log = logging.getLogger(__name__)

#: AdamW learning rate for the pretrained encoder.
ENCODER_LR = 2e-5
#: AdamW learning rate for the freshly initialised task head.
HEAD_LR = 1e-3
#: AdamW learning rate for LoRA adapters (``model.encoder: lm`` checkpoints).
LORA_LR = 1e-4
#: Weight decay, on matrices only (biases, norms and embeddings get none), the
#: same split :func:`ehrjepa.train.pretrain.param_groups` makes.
WEIGHT_DECAY = 0.01
#: Fraction of the planned steps spent in linear warmup.
WARMUP_FRACTION = 0.05
#: Floor of the cosine decay, as a fraction of the base rate.
MIN_LR_RATIO = 0.01
#: Gradient-norm clip, as in pretraining.
GRAD_CLIP = 1.0
#: Windows per optimizer step. Lowered per cell for an encoder that does not fit
#: -- 64 windows of 160 events through a 0.5B LM trunk exceeds an 8 GB card --
#: which is a memory bound rather than a hyperparameter, so a cell that changed
#: it records it (``ehrjepa.eval.run`` ``--ft-batch``, ``ft_batch:`` in a grid).
BATCH_SIZE = 64
#: Passes over the train anchors, before early stopping.
MAX_EPOCHS = 5
#: Epochs without a new best tuning AUROC before the run stops.
PATIENCE = 2


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class FineTuneModel(nn.Module):
    """A pretrained event encoder plus a linear task head, trained end to end.

    ``forward`` takes a collated batch (:func:`ehrjepa.data.dataset.collate_events`)
    and returns one logit per window. The pooled row is cast to float32 before
    the head so the loss is computed in float32 under bf16 autocast.
    """

    def __init__(
        self,
        backbone: EHRJEPA | EHRAR,
        features: str,
        layer: str = "final",
    ) -> None:
        super().__init__()
        if layer not in PROBE_LAYERS:
            raise ValueError(f"layer must be one of {PROBE_LAYERS}, got {layer!r}")
        self.backbone = backbone
        self.features = features
        self.layer = layer
        # Freeze everything the forward pass does not reach, keeping whatever the
        # backbone's own construction decided for the two modules it does: an LM
        # encoder has a frozen trunk and trainable adapters, and that split must
        # survive being handed to a fine-tune.
        wanted = {name: param.requires_grad for name, param in backbone.named_parameters()}
        # ``model.encoder: lm`` serialises events to text, so its "embedding" is
        # a tokenizer whose one parameter -- the code table -- is the *output*
        # projection the next-code head ties to and is never read on the path
        # from a window to a logit. Recognised by the tokenizer it carries
        # rather than by class, so this module needs no import from the
        # optional ``lm`` extra.
        serialised = hasattr(backbone.embed, "tokenizer")
        for name, param in backbone.named_parameters():
            module = name.split(".", 1)[0]
            reached = module == "encoder" or (module == "embed" and not serialised)
            param.requires_grad_(wanted[name] and reached)
        width = n_features(backbone.config.dim, features)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(self.head.bias)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        tokens = self.backbone.embed_batch(batch)
        encoded = self.backbone.encoder(
            tokens, batch["attention_mask"], return_penultimate=self.layer == "penultimate"
        )
        cls = encoded.cls if self.layer == "final" else encoded.cls_penultimate
        hidden = encoded.tokens if self.layer == "final" else encoded.tokens_penultimate
        rows = pool(hidden, cls, batch["attention_mask"], self.features)
        return self.head(self.norm(rows.float()))[:, 0]


def build_model(
    checkpoint: str | Path | None,
    *,
    features: str,
    layer: str = "final",
    random_init: bool = False,
    seed: int = 0,
    vocab_size: int | None = None,
) -> tuple[FineTuneModel, int]:
    """``(model, max_len)``: the checkpoint's own architecture plus a task head.

    ``random_init=True`` keeps the architecture and discards the weights, which
    is the from-scratch control every ICU paper reports alongside its
    pretrained arm.
    """
    backbone, max_len = load_encoder(
        checkpoint, vocab_size=vocab_size, random_init=random_init, seed=seed
    )
    return FineTuneModel(backbone, features, layer), max_len


def param_groups(model: FineTuneModel) -> list[dict[str, object]]:
    """AdamW groups: head, LoRA and encoder rates, each split by weight decay."""
    buckets: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("norm.", "head.")):
            lr = HEAD_LR
        elif "lora_" in name:
            lr = LORA_LR
        else:
            lr = ENCODER_LR
        decay = (
            param.ndim >= 2
            and "emb" not in name
            and not name.endswith("cls_token")
            and "mask_token" not in name
        )
        buckets.setdefault((lr, decay), []).append(param)
    return [
        {
            "params": params,
            "lr": lr,
            "base_lr": lr,
            "weight_decay": WEIGHT_DECAY if decay else 0.0,
        }
        for (lr, decay), params in sorted(buckets.items())
    ]


# --------------------------------------------------------------------------- #
# Batching and epoch order
# --------------------------------------------------------------------------- #


class _Windows:
    """Collated pre-anchor windows for arbitrary rows of a task frame.

    Holds the anchor frame's ``(subject_id, split, anchor_time)`` columns as
    arrays so a batch is named by row indices into the frame -- which is what
    both the training order and the split index arrays are.
    """

    def __init__(self, reader: HistoryReader, anchors: pl.DataFrame) -> None:
        self.reader = reader
        self.subjects = anchors["subject_id"].to_numpy()
        self.splits = anchors["split"].to_list()
        self.minutes = anchor_minutes(anchors["anchor_time"])

    def batch(self, rows: np.ndarray, device: torch.device) -> dict[str, Tensor]:
        items = [
            self.reader.dataset(self.splits[int(row)]).windows_at(
                int(self.subjects[int(row)]), int(self.minutes[int(row)])
            )
            for row in rows
        ]
        return {k: v.to(device) for k, v in collate_events(items).items()}


def epoch_order(labels: np.ndarray, epoch: int, seed: int, balanced: bool = False) -> np.ndarray:
    """Row order for one epoch: a permutation, or a class-balanced draw.

    ``balanced=True`` draws the same number of rows *with replacement*, half
    from each class, so an epoch is the same number of gradient steps either way
    and "epoch" keeps one meaning across the two settings. It is off by default:
    BCE on the natural prevalence is what the probe's logistic regression sees
    (``class_weight=None``), and re-weighting changes the calibration of the
    scores that :mod:`ehrjepa.eval.metrics` reports Brier and slope on. A
    single-class train split falls back to the permutation.
    """
    rng = np.random.default_rng([seed, epoch])
    if not balanced:
        return rng.permutation(labels.size)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if positives.size == 0 or negatives.size == 0:
        return rng.permutation(labels.size)
    half = labels.size // 2
    draw = np.concatenate(
        [
            rng.choice(positives, size=half, replace=True),
            rng.choice(negatives, size=labels.size - half, replace=True),
        ]
    )
    return rng.permutation(draw)


def stop_early(history: Sequence[float], patience: int = PATIENCE) -> bool:
    """Whether the last ``patience`` epochs all failed to beat everything before.

    A pure function of the tuning curve so the rule is testable without a GPU,
    a cache or a checkpoint. ``nan`` scores (a single-class tuning split, which
    :func:`ehrjepa.eval.metrics.auroc` reports as ``nan``) never count as an
    improvement, so a run whose tuning AUROC is undefined stops at ``patience``
    rather than spending every epoch.
    """
    if patience <= 0 or len(history) <= patience:
        return False
    best = max(history[:-patience])
    return not any(score > best for score in history[-patience:])


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclass
class FineTuned:
    """The outcome of one fine-tuning run, in the shape a results row wants.

    ``scores`` are held-out probabilities in the row order of the evaluation
    split's index array -- the same order the probe returns -- so
    :mod:`ehrjepa.eval.run` writes them into the shared ``predictions.parquet``
    unchanged and ``metrics``/``report`` need to know nothing about
    fine-tuning.
    """

    scores: np.ndarray
    tuning_auroc: float
    epochs: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    n_features: int = 0
    wall_seconds: float = 0.0

    def record(self) -> dict:
        """The per-model JSON block, keyed like a fitted probe's."""
        return {
            "params": dict(self.params),
            "grid": list(self.epochs),
            "tuning_auroc": self.tuning_auroc,
            "n_features": self.n_features,
            "finetune_seconds": round(self.wall_seconds, 2),
        }


def fine_tune(
    checkpoint: str | Path | None,
    cache_dir: str | Path,
    anchors: pl.DataFrame,
    index: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    eval_split: str = "held_out",
    random_init: bool = False,
    features: str = "mean",
    layer: str = "final",
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    balanced: bool = False,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
    seed: int = 0,
    vocab_size: int | None = None,
    name: str = "ft",
) -> FineTuned:
    """Train ``checkpoint`` end to end on ``index["train"]``, score ``index[eval_split]``.

    ``index`` is the split-to-row-indices mapping :mod:`ehrjepa.eval.run` builds
    once per task, and ``labels`` the task frame's label column, so every model
    in a run -- baselines, probes, fine-tunes -- is fit and scored on exactly
    the same rows.
    """
    started = time.perf_counter()
    torch.manual_seed(seed)
    model, max_len = build_model(
        checkpoint,
        features=features,
        layer=layer,
        random_init=random_init,
        seed=seed,
        vocab_size=vocab_size,
    )
    dev = resolve_device(device or "auto")
    model = model.to(dev)
    windows = _Windows(HistoryReader(cache_dir, max_len=max_len), anchors)

    train_rows = np.asarray(index["train"])
    tune_rows = np.asarray(index["tuning"])
    eval_rows = np.asarray(index[eval_split])
    y_train = labels[train_rows]
    y_tune = labels[tune_rows]

    steps_per_epoch = max(1, -(-train_rows.size // batch_size))
    total_steps = steps_per_epoch * max_epochs
    warmup = max(1, round(WARMUP_FRACTION * total_steps))
    groups = param_groups(model)
    optimizer = torch.optim.AdamW(groups, lr=ENCODER_LR, betas=(0.9, 0.999), eps=1e-8)

    best_score = -np.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    curve: list[float] = []
    epochs: list[dict] = []
    step = 0

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = epoch_order(y_train, epoch, seed, balanced)
        losses: list[float] = []
        for start in range(0, order.size, batch_size):
            rows = train_rows[order[start : start + batch_size]]
            batch = windows.batch(rows, dev)
            target = torch.as_tensor(labels[rows], dtype=torch.float32, device=dev)
            scale = cosine_lr(step, total_steps, 1.0, warmup, MIN_LR_RATIO)
            for group in optimizer.param_groups:
                group["lr"] = float(group["base_lr"]) * scale
            with autocast_for(dev, "auto"):
                logits = model(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1

        tuning = auroc(y_tune, _predict(model, windows, tune_rows, dev, batch_size))
        curve.append(tuning)
        seconds = time.perf_counter() - epoch_started
        epochs.append(
            {
                "epoch": epoch,
                "tuning_auroc": tuning,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "seconds": round(seconds, 2),
            }
        )
        log.info(
            "%s: epoch %d/%d tuning auroc %.4f loss %.4f (%.1fs)",
            name,
            epoch,
            max_epochs,
            tuning,
            epochs[-1]["train_loss"],
            seconds,
        )
        if np.isfinite(tuning) and tuning > best_score:
            best_score, best_epoch = tuning, epoch
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        if stop_early(curve, patience):
            log.info("%s: stopping after epoch %d (patience %d)", name, epoch, patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = _predict(model, windows, eval_rows, dev, batch_size)
    wall = time.perf_counter() - started
    log.info("%s: %d epoch(s) in %.1fs, best epoch %d", name, len(epochs), wall, best_epoch)
    return FineTuned(
        scores=scores,
        tuning_auroc=float(best_score) if np.isfinite(best_score) else float("nan"),
        epochs=epochs,
        params={
            "epochs_run": len(epochs),
            "best_epoch": best_epoch,
            "max_epochs": max_epochs,
            "patience": patience,
            "balanced": bool(balanced),
            "batch_size": batch_size,
            "max_len": max_len,
            "encoder_lr": ENCODER_LR,
            "head_lr": HEAD_LR,
            "lora_lr": LORA_LR,
            "warmup_steps": warmup,
            "total_steps": total_steps,
            "trainable": model.n_trainable(),
            "random_init": bool(random_init),
            "device": dev.type,
        },
        n_features=n_features(model.backbone.config.dim, features),
        wall_seconds=wall,
    )


@torch.no_grad()
def _predict(
    model: FineTuneModel,
    windows: _Windows,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Probabilities for ``rows``, in ``rows`` order, with the model in eval mode."""
    was_training = model.training
    model.eval()
    out = np.zeros(rows.size, dtype=np.float64)
    for start in range(0, rows.size, batch_size):
        chunk = rows[start : start + batch_size]
        batch = windows.batch(chunk, device)
        with autocast_for(device, "auto"):
            logits = model(batch)
        out[start : start + chunk.size] = torch.sigmoid(logits.float()).cpu().numpy()
    model.train(was_training)
    return out
