"""EHRJEPA: a joint-embedding predictive architecture for longitudinal EHR.

EHRJEPA learns patient representations by predicting, in latent space, the
embeddings of masked future spans of a patient's clinical event stream. Records
are consumed in MEDS format: one token per clinical event, whose embedding fuses
a learned code embedding with a continuous-value embedding, positioned by
continuous time rather than by ordinal index. A context encoder reads the
observed events, a latent predictor is conditioned on the target span's time
positions, and an EMA target encoder supplies the prediction targets. Collapse is
prevented by a SIGReg-style distributional regularizer (LeJEPA, arXiv:2511.08544)
rather than by architectural asymmetry alone.

Subpackages: :mod:`ehrjepa.data` (MEDS ETL, tokenization, datasets),
:mod:`ehrjepa.models` (encoder, predictor), :mod:`ehrjepa.objectives` (JEPA loss,
SIGReg), :mod:`ehrjepa.train` (loop, config), :mod:`ehrjepa.eval` (probes, tasks,
baselines), and :mod:`ehrjepa.utils` (shared helpers).

The project is under active rebuild; no pretrained weights or benchmark results
exist yet.
"""

__all__ = ["__version__"]

__version__ = "0.0.0"
