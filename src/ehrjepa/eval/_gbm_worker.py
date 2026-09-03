"""Fit the gradient-boosting baseline in a process that never imports torch.

See :mod:`ehrjepa.eval` for why. Invoked as
``python -m ehrjepa.eval._gbm_worker <dir>``; the directory holds
``x_train.npz``, ``x_tune.npz``, ``x_predict.npz``, ``y.npz`` and ``grid.json``,
and gains ``result.json`` and ``p_predict.npy``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from ehrjepa.eval.metrics import auroc


def main(argv: list[str]) -> int:
    import xgboost as xgb

    work = Path(argv[0])
    x_train = sparse.load_npz(work / "x_train.npz")
    x_tune = sparse.load_npz(work / "x_tune.npz")
    x_predict = sparse.load_npz(work / "x_predict.npz")
    labels = np.load(work / "y.npz")
    y_train, y_tune = labels["train"], labels["tune"]
    payload = json.loads((work / "grid.json").read_text())
    grid, seed = payload["grid"], payload["seed"]

    best = None
    history = []
    for params in grid:
        model = xgb.XGBClassifier(
            tree_method="hist", random_state=seed, eval_metric="logloss", **params
        )
        model.fit(x_train, y_train)
        score = auroc(y_tune, model.predict_proba(x_tune)[:, 1])
        history.append({**params, "tuning_auroc": score})
        if best is None or (np.isfinite(score) and score > best[0]):
            best = (score, model, dict(params))
    assert best is not None
    np.save(work / "p_predict.npy", best[1].predict_proba(x_predict)[:, 1])
    (work / "result.json").write_text(
        json.dumps({"tuning_auroc": best[0], "params": best[2], "grid": history})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    raise SystemExit(main(sys.argv[1:]))
