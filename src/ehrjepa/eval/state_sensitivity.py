"""Does the state predictor use the actions? An eval-time check on held-out crops.

::

    python -m ehrjepa.eval.state_sensitivity \\
        --checkpoint runs/.../final.pt --cache data/cache/mimic-iv

For every scored pair ``(t, t+k)`` in ``--batches`` random crops of ``--split``,
the prediction under the true intervening actions is compared with the
prediction under a permutation of the action inputs across all pairs of the
same horizon in the batch (a different patient's or window's actions):

``rel_delta``   ``||f(s,a) - f(s,a_perm)|| / ||f(s,a)||``, mean over pairs.
``loss_true``   smooth L1 of ``f(s,a)`` against the layer-normed target state.
``loss_perm``   the same under permuted actions.
``win_rate``    fraction of pairs whose true-action prediction is closer (L2)
                to the target than the permuted-action one.

An ``action_blind`` checkpoint is identically zero / equal here by construction,
which is the control the ``state_ac`` row is read against.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from ehrjepa.data.dataset import EventSequenceDataset, collate_events
from ehrjepa.eval.probe import load_encoder
from ehrjepa.models.state import EHRStateModel
from ehrjepa.utils.runtime import resolve_device

__all__ = ["action_sensitivity", "main"]


@torch.no_grad()
def action_sensitivity(
    model: EHRStateModel,
    batches: Sequence[dict[str, torch.Tensor]],
    seed: int = 0,
) -> dict[str, object]:
    """Per-horizon and pooled sensitivity statistics over ``batches``."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    sums: dict[int, dict[str, float]] = {}
    for batch in batches:
        pairs = model.build_pairs(batch, compute_targets=True)
        assert pairs.targets is not None
        for slot, k in enumerate(model.config.state_horizons):
            r, t = pairs.rows[slot], pairs.cols[slot]
            n = int(r.numel())
            if n < 2:
                continue
            state = pairs.states[r, t]
            action = model.action_vector(pairs.act_mean[slot], pairs.act_count[slot])
            perm = torch.randperm(n, generator=generator).to(action.device)
            true = model.predict(state, action, slot, pairs.log_dt[slot]).float()
            other = model.predict(state, action[perm], slot, pairs.log_dt[slot]).float()
            target = pairs.targets[r, t + k].float()
            normed = F.layer_norm(target, (target.shape[-1],))
            loss_true = F.smooth_l1_loss(true, normed, reduction="none").mean(dim=-1)
            loss_perm = F.smooth_l1_loss(other, normed, reduction="none").mean(dim=-1)
            dist_true = (true - normed).norm(dim=-1)
            dist_perm = (other - normed).norm(dim=-1)
            rel = (true - other).norm(dim=-1) / true.norm(dim=-1).clamp(min=1e-8)
            acc = sums.setdefault(
                k, {"n": 0, "rel_delta": 0.0, "loss_true": 0.0, "loss_perm": 0.0, "wins": 0.0}
            )
            acc["n"] += n
            acc["rel_delta"] += float(rel.sum())
            acc["loss_true"] += float(loss_true.sum())
            acc["loss_perm"] += float(loss_perm.sum())
            # A tie (identical action inputs, e.g. two empty windows) is half a win.
            acc["wins"] += float((dist_true < dist_perm).float().sum())
            acc["wins"] += 0.5 * float((dist_true == dist_perm).float().sum())
    per_k = {}
    total = {"n": 0, "rel_delta": 0.0, "loss_true": 0.0, "loss_perm": 0.0, "wins": 0.0}
    for k, acc in sorted(sums.items()):
        n = max(1, acc["n"])
        per_k[f"k{k}"] = {
            "pairs": int(acc["n"]),
            "rel_delta": acc["rel_delta"] / n,
            "loss_true": acc["loss_true"] / n,
            "loss_perm": acc["loss_perm"] / n,
            "gain": (acc["loss_perm"] - acc["loss_true"]) / n,
            "win_rate": acc["wins"] / n,
        }
        for key in total:
            total[key] += acc[key]
    n = max(1, total["n"])
    pooled = {
        "pairs": int(total["n"]),
        "rel_delta": total["rel_delta"] / n,
        "loss_true": total["loss_true"] / n,
        "loss_perm": total["loss_perm"] / n,
        "gain": (total["loss_perm"] - total["loss_true"]) / n,
        "win_rate": total["wins"] / n,
    }
    return {"actions_on": bool(model.config.state_actions), "pooled": pooled, "per_horizon": per_k}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ehrjepa.eval.state_sensitivity")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--split", default="held_out")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    model, max_len = load_encoder(args.checkpoint)
    if not isinstance(model, EHRStateModel):
        raise SystemExit(f"{args.checkpoint} is not an objective.kind=state checkpoint")
    device = resolve_device(args.device)
    model = model.to(device)
    dataset = EventSequenceDataset(
        args.cache, args.split, max_len=max_len, sampling="random_window", seed=args.seed
    )
    generator = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(args.batches):
        rows = torch.randint(0, len(dataset), (args.batch_size,), generator=generator).tolist()
        batch = collate_events([dataset[i] for i in rows])
        batches.append({k: v.to(device) for k, v in batch.items()})
    result = action_sensitivity(model, batches, seed=args.seed)
    result.update({"checkpoint": str(args.checkpoint), "split": args.split})
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
