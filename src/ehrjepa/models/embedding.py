"""The event token embedding: what happened, how much, and when.

One MEDS event is one token, and the token is a sum of five terms:

``code_emb[code_id]``
    The clinical fact itself. ``PAD`` is a real row but is never attended to.
``value_bin_emb[value_bin]``
    The per-code decile, ``0`` meaning "this event carries no number". The bin is
    coarse and robust; it is what the model can rely on when a code's value
    distribution is lumpy.
``value_mlp(fourier(value_z))``, *gated to zero when* ``value_bin == 0``
    The fine-grained residual within the bin. The gate matters: ``value_z`` is
    stored as ``0.0`` for value-less events, and ``0.0`` is also a perfectly
    ordinary z-score, so without the gate "no value" and "exactly average value"
    would collide.
``age_mlp(fourier(age))`` and ``delta_mlp(fourier(log_delta))``
    When, twice over: absolute position in the patient's life, and the log-gap
    since the previous event. Sequence order is carried separately by RoPE in the
    encoder; these two carry the wall-clock, which is irregular by nature.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ehrjepa.data.tokenize import N_VALUE_BINS, PAD_ID
from ehrjepa.models.layers import ScalarEncoder

__all__ = ["EventEmbedding"]


class EventEmbedding(nn.Module):
    """Sum the code, value and time channels of an event into one ``dim``-vector.

    Parameters
    ----------
    vocab_size:
        Rows in the code embedding, i.e. ``len(Vocabulary)``.
    dim:
        Model width.
    n_freq:
        Fourier frequencies per continuous scalar.
    dropout:
        Applied once, to the summed token.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_freq: int = 16,
        dropout: float = 0.0,
        scalar_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.code_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.value_bin_emb = nn.Embedding(N_VALUE_BINS + 1, dim)
        self.value_enc = ScalarEncoder(
            dim, n_freq, hidden=scalar_hidden, min_freq=1e-2, max_freq=1e1
        )
        self.age_enc = ScalarEncoder(dim, n_freq, hidden=scalar_hidden, min_freq=1e-2, max_freq=1e1)
        self.delta_enc = ScalarEncoder(
            dim, n_freq, hidden=scalar_hidden, min_freq=1e-2, max_freq=1e1
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.code_emb.weight, std=0.02)
        nn.init.normal_(self.value_bin_emb.weight, std=0.02)
        with torch.no_grad():
            self.code_emb.weight[PAD_ID].zero_()

    def time_features(self, age: Tensor, log_delta: Tensor) -> Tensor:
        """The two *time-only* channels, which is all the predictor is allowed to see."""
        return self.age_enc(age) + self.delta_enc(log_delta)

    def forward(
        self,
        code_id: Tensor,
        value_bin: Tensor,
        value_z: Tensor,
        age: Tensor,
        log_delta: Tensor,
    ) -> Tensor:
        """All inputs are ``(batch, seq)``; the result is ``(batch, seq, dim)``."""
        gate = (value_bin != 0).to(value_z.dtype).unsqueeze(-1)
        token = (
            self.code_emb(code_id)
            + self.value_bin_emb(value_bin)
            + gate * self.value_enc(value_z)
            + self.time_features(age, log_delta)
        )
        return self.dropout(token)
