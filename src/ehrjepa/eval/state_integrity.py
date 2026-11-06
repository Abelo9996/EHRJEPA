"""Controlled action-echo and calendar-horizon audit; no training or clinical claims.

Compare identical weights with legacy vs observation-only targets. Perturb action
content while keeping observations fixed, then perturb observations as a positive
control against constant targets. Default inputs are TRAIN crops, so exploratory
debugging does not consume the held-out test set. Reports contain aggregates only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ehrjepa.data.actions import DEFAULT_ADMISSION_PREFIXES, code_flags
from ehrjepa.data.dataset import EventSequenceDataset, collate_events
from ehrjepa.data.tokenize import Vocabulary
from ehrjepa.eval.probe import load_encoder
from ehrjepa.models.jepa import EHRJEPAConfig
from ehrjepa.models.state import EHRStateModel


def perturb_content(batch: dict, model: EHRStateModel, *, actions: bool) -> dict:
    """Change code/value content, preserving token roles, positions and times."""
    changed = {key: value.clone() for key, value in batch.items()}
    role = model.action_table if actions else ~model.action_table
    ids = role.nonzero(as_tuple=True)[0]
    ids = ids[ids >= 4]  # exclude special tokens
    if ids.numel() == 0:
        return changed
    select = (model.action_table[batch["code_id"]] == actions) & batch["attention_mask"].bool()
    rank = torch.searchsorted(ids, batch["code_id"][select])
    changed["code_id"][select] = ids[(rank + 1) % len(ids)]
    changed["value_bin"][select] = (changed["value_bin"][select] % 10) + 1
    changed["value_z"][select] = -changed["value_z"][select] + 0.5
    return changed


@torch.no_grad()
def target_integrity(model: EHRStateModel, batches: list[dict]) -> dict:
    """Paired target changes under fixed weights and identical observation streams."""
    model = copy.deepcopy(model).eval()
    views = {}
    for view in ("all", "observations"):
        model.config.state_target_view = view
        count, action_delta, obs_delta, action_max = 0, 0.0, 0.0, 0.0
        action_tokens = 0
        for batch in batches:
            obs = ~model.action_table[batch["code_id"]] & batch["attention_mask"].bool()
            action_tokens += int(
                (model.action_table[batch["code_id"]] & batch["attention_mask"].bool()).sum()
            )
            windows, n = model.windows(batch, batch["attention_mask"].bool())
            width = max(1, int(n.max()))
            # Measure every window with observations, including those that are
            # not scorable transitions. Presence is independent of target values.
            present = torch.zeros(obs.shape[0], width, device=obs.device)
            present.scatter_add_(1, windows.clamp(min=0), obs.float())
            present = present > 0
            original = model.build_pairs(batch).targets
            altered_actions = model.build_pairs(perturb_content(batch, model, actions=True)).targets
            altered_obs = model.build_pairs(perturb_content(batch, model, actions=False)).targets
            assert original is not None and altered_actions is not None and altered_obs is not None
            base = original[present].float()
            if base.numel() == 0:
                continue
            da = altered_actions[present].float() - base
            do = altered_obs[present].float() - base
            scale = base.norm(dim=-1).clamp(min=1e-8)
            action_delta += float((da.norm(dim=-1) / scale).sum())
            obs_delta += float((do.norm(dim=-1) / scale).sum())
            action_max = max(action_max, float(da.abs().max()))
            count += len(base)
        views[view] = {
            "sampled_observation_windows": count,
            "sampled_action_tokens": action_tokens,
            "mean_relative_target_change_actions": action_delta / count if count else None,
            "max_absolute_target_change_actions": action_max if count else None,
            "mean_relative_target_change_observations": obs_delta / count if count else None,
        }
    fixed = views["observations"]
    enough = fixed["sampled_observation_windows"] > 0 and fixed["sampled_action_tokens"] > 0
    passed = (
        enough
        and fixed["max_absolute_target_change_actions"] <= 1e-6
        and fixed["mean_relative_target_change_observations"] > 1e-5
    )
    return {
        "status": "passed" if passed else "failed" if enough else "inconclusive",
        "views": views,
    }


@torch.no_grad()
def calendar_horizons(model: EHRStateModel, batches: list[dict]) -> dict:
    """Check whether occupied-window offsets equal calendar-bin offsets."""
    if model.config.state_anchor != "calendar":
        return {"status": "not_applicable", "reason": "calendar anchors only"}
    counts = {
        str(k): {"pairs": 0, "calendar_gap_not_k": 0, "max_calendar_gap": None}
        for k in model.config.state_horizons
    }
    for batch in batches:
        pairs = model.build_pairs(batch, compute_targets=False)
        window, n = model.windows(batch, pairs.valid)
        width = max(1, int(n.max()))
        bins = torch.div(
            batch["time_min"], model.config.state_window_minutes, rounding_mode="floor"
        )
        big = torch.iinfo(torch.int64).max
        first = torch.full((len(bins), width), big, dtype=torch.int64, device=bins.device)
        first.scatter_reduce_(
            1, window.clamp(min=0), torch.where(pairs.valid, bins, big), reduce="amin"
        )
        for slot, k in enumerate(model.config.state_horizons):
            r, t = pairs.rows[slot], pairs.cols[slot]
            gap = first[r, pairs.target_cols[slot]] - first[r, t]
            row = counts[str(k)]
            row["pairs"] += len(gap)
            row["calendar_gap_not_k"] += int((gap != k).sum())
            if gap.numel():
                row["max_calendar_gap"] = max(row["max_calendar_gap"] or 0, int(gap.max()))
    total = sum(row["pairs"] for row in counts.values())
    mismatches = sum(row["calendar_gap_not_k"] for row in counts.values())
    return {
        "status": "inconclusive" if total == 0 else "mismatch" if mismatches else "passed",
        "window_minutes": model.config.state_window_minutes,
        "per_horizon": counts,
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--codes-file", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "tuning", "held_out"], default="train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.batches, args.batch_size, args.max_len) < 1:
        parser.error("batches, batch-size and max-len must be positive")
    torch.set_num_threads(1)
    vocab = Vocabulary.read(args.cache / "vocab.parquet")
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        if args.checkpoint:
            model, max_len = load_encoder(args.checkpoint)
            if not isinstance(model, EHRStateModel):
                raise ValueError("checkpoint must be an EHRStateModel")
            if model.config.vocab_size != len(vocab):
                raise ValueError("checkpoint and cache vocabulary sizes differ")
        else:
            max_len = args.max_len
            model = EHRStateModel(
                EHRJEPAConfig(
                    vocab_size=len(vocab),
                    dim=32,
                    depth=2,
                    heads=4,
                    mlp="gelu",
                    mlp_ratio=2.0,
                    dropout=0.0,
                    n_freq=8,
                    causal=True,
                    build_predictor=False,
                    target_mode="ema",
                    state_window_minutes=240,
                )
            )
            model.set_code_tables(
                torch.from_numpy(code_flags(vocab.codes, codes_file=args.codes_file)),
                torch.from_numpy(code_flags(vocab.codes, DEFAULT_ADMISSION_PREFIXES)),
            )
        model = model.cpu().eval()
        if model.config.state_anchor != "calendar":
            raise ValueError("this paired audit currently requires calendar-anchored checkpoints")
        dataset = EventSequenceDataset(
            args.cache, args.split, max_len=max_len, sampling="random_window", seed=seed
        )
        generator = torch.Generator().manual_seed(seed)
        batches = []
        for _ in range(args.batches):
            indices = torch.randint(
                0, len(dataset), (args.batch_size,), generator=generator
            ).tolist()
            batches.append(collate_events([dataset[i] for i in indices]))
        occupied_model = copy.deepcopy(model)
        occupied_model.config.state_horizon_mode = "occupied"
        calendar_model = copy.deepcopy(model)
        calendar_model.config.state_horizon_mode = "calendar"
        results.append(
            {
                "seed": seed,
                "target_integrity": target_integrity(model, batches),
                "calendar_horizons": calendar_horizons(occupied_model, batches),
                "exact_calendar_horizons": calendar_horizons(calendar_model, batches),
            }
        )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "state_target_integrity_v1",
        "weights": "checkpoint" if args.checkpoint else "random_initialization_not_trained",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": _hash(args.checkpoint) if args.checkpoint else None,
        "cache": str(args.cache),
        "split": args.split,
        "cache_meta_sha256": _hash(args.cache / "meta.json"),
        "vocab_sha256": _hash(args.cache / "vocab.parquet"),
        "codes_file_sha256": _hash(args.codes_file) if args.codes_file else None,
        "audit_code_sha256": _hash(Path(__file__)),
        "model_code_sha256": _hash(Path(__file__).parents[1] / "models" / "state.py"),
        "batches_per_seed": args.batches,
        "batch_size": args.batch_size,
        "max_len": max_len,
        "results": results,
        "interpretation": "Structural target dependence only. Not a trained clinical benefit, "
        "not an intervention effect, and not a cross-paper finding. "
        "Seeds resample crops and (unless using a checkpoint) random weights. "
        "The exact-calendar arm skips unobserved target bins; no imputation is performed.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in results:
        print(json.dumps({"seed": row["seed"], **row["target_integrity"]}))
    return (
        0
        if all(
            row["target_integrity"]["status"] == "passed"
            and row["exact_calendar_horizons"]["status"] == "passed"
            for row in results
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
