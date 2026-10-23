"""How much does the count-GBM baseline lean on near-synonyms of a new diagnosis?

The ``new_dx_365d/*`` tasks already drop every subject whose history holds one of
the outcome's own codes (the ``condition`` predicate in
``configs/tasks/predicates.yaml``, e.g. ``ICD9CM/585`` / ``ICD10CM/N18`` for CKD).
What they do not drop is a history that documents the same condition under a code
outside those prefixes -- hypertensive CKD (``I12``), diabetic kidney disease
(``E11.2x``), dialysis status (``Z99.2``) -- in which case a "new" CKD diagnosis is
partly a coding-consistency label, and a model that counts codes gets it cheaply.

For each new-dx task this reports, on the cohort the eval harness builds:

* the share of train positives vs negatives whose history holds a near-synonym
  (model-free);
* held-out AUROC of an XGBoost model on the eval's own count features, with and
  without the near-synonym columns (same hyper-parameters both times -- the
  middle setting of ``baselines.GBM_GRID``), and the near-synonyms' share of the
  model's total gain;
* the top features by gain.

Run from the repo root, in two processes (``ehrjepa.eval`` imports torch, and
torch and xgboost must not share a process -- see ``ehrjepa.eval._gbm_worker``)::

    python scripts/gbm_leak_check.py build OUT [SOURCE]
    python scripts/gbm_leak_check.py fit OUT
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

#: Codes (dots removed) that document each condition but are NOT in its
#: exclusion predicate. Prefix matches against the vocabulary.
NEAR_SYNONYMS = {
    "ckd": r"ICD10CM/(I12|I13|E1[0-3]2|N08|N25|R80|Z49|Z940|Z992)|ICD9CM/(2504|403|404|586|V420|V451|V56)",
    "diabetes": r"ICD10CM/(E08|E09|E13|O24|R73|Z794|Z7984)|ICD9CM/(249|6480|7902|V5867)",
    "heart_failure": (
        r"ICD10CM/(I0981|I110|I130|I132|I42|I9713)"
        r"|ICD9CM/(39891|40201|40211|40291|404[019][13]|425)"
    ),
    "copd": r"ICD10CM/(J47|J961|Z9981)|ICD9CM/(490|494|V462)",
}
BLOCK_NAMES = ("cnt30d", "cnt365d", "cnt_all", "last_z", "n_vals")
ALL_TIME_BLOCK = 2
GBM_PARAMS = {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300}
SPLITS = ("train", "tuning", "held_out")


def build(out: Path, source: str = "mimic-iv") -> None:
    import polars as pl
    from scipy import sparse

    from ehrjepa.data.tokenize import Vocabulary
    from ehrjepa.eval.baselines import count_matrix, prune_columns
    from ehrjepa.eval.history import HistoryReader
    from ehrjepa.eval.tasks import load_task

    cache = Path("data/cache") / source
    vocab = Vocabulary.read(cache / "vocab.parquet")
    n_vocab = len(vocab)
    reader = HistoryReader(cache, max_len=None)
    normalized = [code.replace(".", "") for code in vocab.codes]
    out.mkdir(parents=True, exist_ok=True)
    for short, pattern in NEAR_SYNONYMS.items():
        rx = re.compile(pattern)
        near_ids = np.array([i for i, c in enumerate(normalized) if rx.match(c)], dtype=np.int64)
        near_set = set(near_ids.tolist())
        frame = load_task(Path("data/tasks") / source, f"new_dx_365d/{short}")
        mats, ys = {}, {}
        for split in SPLITS:
            part = frame.filter(pl.col("split") == split)
            mats[split] = count_matrix(reader, part, n_vocab)
            ys[split] = part["label"].to_numpy().astype(np.int8)
        keep = prune_columns(mats["train"])
        for split in SPLITS:
            sparse.save_npz(out / f"{short}_{split}.npz", mats[split][:, keep].tocsr())
        np.savez(out / f"{short}_y.npz", **ys)

        y = ys["train"]
        if near_ids.size:
            all_time = mats["train"][:, ALL_TIME_BLOCK * n_vocab + near_ids]
            has_near = np.asarray(all_time.sum(axis=1)).ravel() > 0
        else:
            has_near = np.zeros(y.size, dtype=bool)
        meta = {
            "task": short,
            "n_vocab_near": int(near_ids.size),
            "near_codes": [vocab.codes[i] for i in near_ids.tolist()],
            "columns": [
                {"code": vocab.codes[c], "block": BLOCK_NAMES[b], "near": c in near_set}
                for c, b in zip((keep % n_vocab).tolist(), (keep // n_vocab).tolist(), strict=True)
            ],
            "train_n": int(y.size),
            "train_pos": int(y.sum()),
            "has_near_rate_pos": float(has_near[y == 1].mean()) if y.sum() else float("nan"),
            "has_near_rate_neg": float(has_near[y == 0].mean()),
        }
        (out / f"{short}_meta.json").write_text(json.dumps(meta))
        print(
            f"[build] {short}: train {y.size} ({int(y.sum())} pos), {keep.size} columns kept, "
            f"{near_ids.size} near-synonym codes in vocab; history has one: "
            f"pos {meta['has_near_rate_pos']:.3f} / neg {meta['has_near_rate_neg']:.3f}",
            flush=True,
        )


def fit(out: Path) -> None:
    assert "torch" not in sys.modules, "fit must run in a process that never imported torch"
    import xgboost as xgb
    from scipy import sparse
    from sklearn.metrics import roc_auc_score

    report = []
    for short in NEAR_SYNONYMS:
        meta = json.loads((out / f"{short}_meta.json").read_text())
        x = {s: sparse.load_npz(out / f"{short}_{s}.npz") for s in SPLITS}
        y = np.load(out / f"{short}_y.npz")
        near = np.array([c["near"] for c in meta["columns"]], dtype=bool)

        def held_out_auroc(cols: np.ndarray) -> tuple[object, float]:
            model = xgb.XGBClassifier(
                tree_method="hist", random_state=0, eval_metric="logloss", n_jobs=4, **GBM_PARAMS
            )
            model.fit(x["train"][:, cols], y["train"])
            return model, float(roc_auc_score(y["held_out"], model.predict_proba(x["held_out"][:, cols])[:, 1]))

        model, auc_all = held_out_auroc(np.arange(near.size))
        _, auc_without = held_out_auroc(np.flatnonzero(~near))
        gain = np.asarray(model.feature_importances_)
        top = [
            {**meta["columns"][i], "gain": round(float(gain[i]), 4)}
            for i in np.argsort(gain)[::-1][:15]
        ]
        row = {
            "task": f"new_dx_365d/{short}",
            "auroc_all": round(auc_all, 4),
            "auroc_without_near": round(auc_without, 4),
            "delta": round(auc_without - auc_all, 4),
            "near_gain_share": round(float(gain[near].sum()), 4),
            "near_columns": int(near.sum()),
            **{k: meta[k] for k in ("train_n", "train_pos", "has_near_rate_pos", "has_near_rate_neg")},
            "top": top,
        }
        report.append(row)
        print(
            f"\n=== {row['task']} ===  held-out AUROC {auc_all:.4f} -> {auc_without:.4f} without "
            f"near-synonyms ({row['delta']:+.4f}); near-synonym gain share {row['near_gain_share']:.3f}; "
            f"history has one: pos {meta['has_near_rate_pos']:.3f} vs neg {meta['has_near_rate_neg']:.3f}"
        )
        for t in top:
            print(f"  {t['gain']:.4f}  {'NEAR ' if t['near'] else '     '}{t['block']:<8} {t['code']}")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    mode, target = sys.argv[1], Path(sys.argv[2])
    if mode == "build":
        build(target, *(sys.argv[3:4]))
    elif mode == "fit":
        fit(target)
    else:
        raise SystemExit(f"unknown mode {mode!r}: use build or fit")
