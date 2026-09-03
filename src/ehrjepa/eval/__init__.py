"""Evaluation: downstream tasks, probes, and baselines.

Everything used to judge a pretrained encoder, kept strictly separate from
pretraining so that no label ever reaches the SSL loop. The label-leakage failure
of the archived pipeline in ``legacy/`` is the specific thing this design is
meant to prevent, so the leakage barrier is one function --
:meth:`ehrjepa.data.dataset.EventSequenceDataset.windows_at`, reached only
through :class:`ehrjepa.eval.history.HistoryReader` -- rather than a convention.

=========================== ================================================
:mod:`~ehrjepa.eval.tasks`     anchors and labels, ACES over a seeded anchor rule
:mod:`~ehrjepa.eval.history`   pre-anchor history out of the tensor cache
:mod:`~ehrjepa.eval.baselines` count features, logistic regression, gradient boosting
:mod:`~ehrjepa.eval.probe`     frozen-encoder embeddings, probes, few-shot
:mod:`~ehrjepa.eval.metrics`   AUROC/AUPRC/Brier/calibration, bootstrap CIs
:mod:`~ehrjepa.eval.report`    markdown rendering of a results dict
:mod:`~ehrjepa.eval.run`       the CLI that ties the above together
=========================== ================================================

Every model in a run is fit and scored on the *same* task frame -- the same
subjects, the same anchor times, the same labels -- which is what makes the
paired bootstrap in :func:`ehrjepa.eval.metrics.paired_bootstrap` a comparison
rather than a coincidence.

**xgboost does not share a process with torch here.** Both ship their own
OpenMP runtime, and on macOS arm64 whichever loads second corrupts the first:
``import torch`` then ``import xgboost`` segfaults in xgboost's ``fit``, and the
reverse order segfaults later in numpy. Neither ordering survives a run that
needs both, so :func:`ehrjepa.eval.baselines.fit_gbm` writes its matrices to a
temporary directory and fits in a subprocess that never imports torch. This
module therefore imports neither, and must keep importing neither.
"""
