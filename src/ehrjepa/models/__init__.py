"""Model layer: the event encoder and the latent predictor.

This subpackage will hold the two networks that make up a JEPA. The encoder is a
transformer over event tokens whose input embedding sums a code embedding, a
continuous-value embedding (a learned projection of the per-code normalized
``numeric_value``, gated off for events that carry none), and a continuous-time
position encoding computed from the event timestamp rather than from its ordinal
position, so that irregular sampling and long gaps are represented faithfully.
The predictor is a narrower transformer that maps context representations plus
the target span's time positions to predicted latents for that span; it is
deliberately weaker than the encoder so representation quality is not offloaded
into it. An EMA copy of the encoder serves as the target encoder. Also planned
here: parameter-group helpers for weight decay, checkpoint save/load, and small
named configurations (tiny/small/base) for scaling studies.
"""

from ehrjepa.models.embedding import EventEmbedding
from ehrjepa.models.encoder import Encoder, EncoderOutput
from ehrjepa.models.jepa import EHRJEPA, EHRJEPAConfig, JEPAOutput, ema_momentum
from ehrjepa.models.layers import (
    FourierFeatures,
    RotaryEmbedding,
    ScalarEncoder,
    TransformerBlock,
)
from ehrjepa.models.predictor import Predictor

__all__ = [
    "EHRJEPA",
    "EHRJEPAConfig",
    "Encoder",
    "EncoderOutput",
    "EventEmbedding",
    "FourierFeatures",
    "JEPAOutput",
    "Predictor",
    "RotaryEmbedding",
    "ScalarEncoder",
    "TransformerBlock",
    "ema_momentum",
]
