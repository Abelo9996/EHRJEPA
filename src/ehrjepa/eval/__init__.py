"""Evaluation: downstream tasks, probes, and baselines.

This subpackage will hold everything used to judge a pretrained encoder, kept
strictly separate from pretraining so that no label ever reaches the SSL loop.
Planned contents: adapters for MEDS-DEV / ACES task definitions, which specify
prediction times and label windows declaratively so cohorts are reproducible and
comparable to published numbers, plus EHRSHOT's task suite; a featurizer that
freezes the encoder and emits one representation per (subject, prediction time)
strictly from events before the prediction time; linear and logistic probes and
few-shot variants over those features; baselines to compare against (count-based
logistic regression, gradient boosting, a supervised-from-scratch transformer);
and metric computation with subject-level splits and bootstrap confidence
intervals. The label-leakage failure of the archived pipeline in ``legacy/`` is
the specific thing this design is meant to prevent.
"""
