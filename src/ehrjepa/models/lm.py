"""``model.encoder: lm`` -- the event encoder is a pretrained language model.

Everything else in this repository learns what a code *is* from co-occurrence: a
row of ``N(0, 0.02)`` per code, 30,000 of them on claims data, 47 on
PhysioNet-2019, and the encoder's job is to discover from the data that
``VAR//HR`` is a heart rate and that 92 of them is unremarkable. A pretrained
language model already knows both. This module is the arm of the experiment that
asks what that knowledge is worth once the numbers are the content of the record.

**The serialisation.** One event becomes one short text span::

    heart rate 92 (+1h);

-- the code's plain-English phrase from :mod:`ehrjepa.data.code_text`, the
numeric value, and the gap since the previous event. Three notes on it:

*The number is the real one, not the z-score.* The cache stores ``value_z``, a
per-code z-score of the (optionally ``log1p``-ed) value; the same cache's
``quantizer.parquet`` stores the ``mean``/``std``/``use_log1p`` that produced it,
so the raw value is recovered here and printed to three significant digits. "92"
is a quantity a language model has an opinion about; "+0.43" is not. The round
trip is exact except at the ``z_clip`` tails, where ``value_z`` was clipped to
+-5 and the printed number is the clipped one -- which is the value the
from-scratch arm sees too, so the two arms are shown the same information.

*Value-less events print no number.* ``value_bin == 0`` is the cache's encoding of
"this event carries no measurement", and ``value_z`` is stored as ``0.0`` there --
an ordinary z-score. Printing it would tell the model that every admission had an
exactly average something.

*The gap is text, and it is the only clock.* There is no RoPE-on-wall-clock and
no Fourier time channel here: ``(+1h)`` is what the model gets, and
``target.time_features: false`` drops it, exactly as it drops the Fourier terms
on the from-scratch side.

**Where the event representation comes from.** The spans of a window are
concatenated in order and tokenised, and event ``i``'s representation is the last
hidden state of the *last token of its span* -- which, because the separator is
part of the span, is the separator token. Under causal attention that position
has read events ``0..i`` and nothing later, which is the same contract
:class:`~ehrjepa.models.encoder.Encoder` honours with ``causal=True`` and the
same one the ``ar`` and ``nextlatent`` objectives require. The
``lm_hidden -> dim`` projection and its LayerNorm are trainable and are what let
every downstream part -- the per-horizon MLP heads, the next-code head tied to
this stack's own small code table, ``last`` probe pooling -- run unchanged at the
repository's usual ``dim``.

**What is trained.** The base model is frozen. LoRA adapters (rank
``model.lm_lora_r``, on ``model.lm_lora_targets``), the projection, its
LayerNorm, the constant CLS row, and the small code table that the next-code head
ties to. Gradient checkpointing is on by default: the 8 GB card this is aimed at
cannot hold 24 layers of activations over 2,000 tokens otherwise.

**The token cap.** ``model.lm_max_tokens`` (2,048) bounds the tokens per window.
A window over the cap loses its *oldest* events -- the truncation is from the
left -- so the last valid event, which is what ``last`` pooling reads and what
the causal targets are anchored on, is never the one dropped. Dropped events
carry no hidden state, so they are reported in :attr:`LMTokens.kept` and
:func:`~ehrjepa.models.jepa.effective_valid` removes them from every mask
downstream. The fraction kept is logged as ``lm_events_kept``; at
``data.max_len`` small enough for the cap it is 1.0 and the mechanism is inert.

Install the optional extra first::

    pip install -e '.[lm]'
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ehrjepa.data.code_text import DescriptionTables, describe, load_tables
from ehrjepa.data.tokenize import PAD_ID, Vocabulary, load_quantizer, split_code
from ehrjepa.models.jepa import EHRJEPAConfig

__all__ = [
    "EVENT_SEPARATOR",
    "LMEncoder",
    "LMEventTokenizer",
    "LMTokens",
    "code_phrases",
    "event_texts",
    "load_base_model",
    "load_tokenizer",
    "value_scales",
]

#: Appended to every event's span. Part of the span, so the position whose hidden
#: state represents the event is this token: an explicit "the event ends here"
#: marker rather than whichever word happened to come last.
EVENT_SEPARATOR = "; "

#: Code families whose descriptions need the downloaded tables
#: :func:`ehrjepa.data.code_text.load_tables` reads. A vocabulary with none of
#: them -- both PhysioNet caches -- skips that read, which is a 75 MB pair of FDA
#: text files it would otherwise parse to answer nothing.
_TABLE_FAMILIES = frozenset({"ICD9CM", "ICD9PROC", "HCPCS", "NDC", "DRG"})

_EMPTY_TABLES = DescriptionTables(
    icd9_dx={},
    icd9_sg={},
    icd9_category=None,
    hcpcs={},
    hcpcs_section={},
    drg={},
    ndc_product={},
    ndc_labeler={},
)


def _family(code: str) -> str:
    parts = split_code(code)
    return parts[0].upper() if parts else code.upper()


def code_phrases(cache_dir: str | Path) -> list[str]:
    """One plain-English phrase per ``code_id``, in id order.

    Straight from :func:`ehrjepa.data.code_text.describe`, which is the same
    text ``model.code_init: text`` embeds -- so "the LM was told what the code
    means" and "the code embedding was initialized from what the code means" are
    the same words, and the difference between those two arms is the mechanism
    rather than the vocabulary.
    """
    vocab = Vocabulary.read(Path(cache_dir) / "vocab.parquet")
    needs_tables = any(_family(code) in _TABLE_FAMILIES for code in vocab.codes)
    tables = load_tables() if needs_tables else _EMPTY_TABLES
    return [describe(code, tables).text for code in vocab.codes]


def value_scales(cache_dir: str | Path, vocab_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """``(mean, std, use_log1p)`` per ``code_id``, for inverting ``value_z``.

    Read from the cache's ``quantizer.parquet``, which carries one row per code,
    and returned as three ``(vocab_size,)`` tensors so they can be registered as
    buffers: after that the checkpoint carries its own de-normalisation table and
    a machine that evaluates the checkpoint does not need the quantizer file.
    """
    frame = load_quantizer(Path(cache_dir) / "quantizer.parquet")
    mean = torch.zeros(vocab_size, dtype=torch.float32)
    std = torch.zeros(vocab_size, dtype=torch.float32)
    log1p = torch.zeros(vocab_size, dtype=torch.bool)
    ids = frame["code_id"].to_numpy()
    inside = (ids >= 0) & (ids < vocab_size)
    index = torch.from_numpy(ids[inside].astype(np.int64))
    mean[index] = torch.from_numpy(frame["mean"].to_numpy()[inside].astype(np.float32))
    std[index] = torch.from_numpy(frame["std"].to_numpy()[inside].astype(np.float32))
    log1p[index] = torch.from_numpy(frame["use_log1p"].to_numpy()[inside].astype(bool))
    return mean, std, log1p


def _significant(value: float) -> str:
    """Three significant digits, without an exponent for the ordinary range."""
    if not np.isfinite(value):
        return "?"
    text = f"{value:.3g}"
    return text.replace("e+0", "e").replace("e-0", "e-")


def event_texts(
    phrases: Sequence[str],
    code_id: np.ndarray,
    value_bin: np.ndarray,
    raw_value: np.ndarray,
    delta_hours: np.ndarray,
    valid: np.ndarray,
    use_time: bool = True,
) -> list[list[str]]:
    """The span of every valid event, ``[batch][position]``, ``""`` where invalid.

    Pure and array-in/array-out so the serialisation can be tested without a
    tokenizer, a model, or a cache.
    """
    out: list[list[str]] = []
    for row in range(code_id.shape[0]):
        spans: list[str] = []
        for col in range(code_id.shape[1]):
            if not valid[row, col]:
                spans.append("")
                continue
            text = phrases[int(code_id[row, col])]
            if value_bin[row, col] != 0:
                text = f"{text} {_significant(float(raw_value[row, col]))}"
            if use_time:
                text = f"{text} (+{_significant(float(delta_hours[row, col]))}h)"
            spans.append(text + EVENT_SEPARATOR)
        out.append(spans)
    return out


class LMTokens(dict):
    """What the tokenizer stage hands the encoder stage.

    A ``dict`` subclass for the same reason
    :class:`~ehrjepa.models.encoder.EncoderOutput` is one -- it survives
    ``torch.utils.checkpoint`` -- with two additions that let it stand where a
    ``(B, L, dim)`` tensor stood before: ``detach``, which the shared-target path
    calls on the online tokens, and ``kept``, which
    :func:`~ehrjepa.models.jepa.effective_valid` looks for.

    ``input_ids``/``attn``
        ``(B, T)``: the window's tokens, right-padded.
    ``span_end``
        ``(B, L)``: for each event, the index in ``0..T-1`` of the last token of
        its span. ``0`` for events with no span (padding, or truncated away),
        whose row is zeroed rather than read.
    ``kept``
        ``(B, L)`` bool: events that made it into ``input_ids``.
    """

    def detach(self) -> LMTokens:
        """The token ids carry no gradient, so this is the identity."""
        return self

    @property
    def input_ids(self) -> Tensor:
        return self["input_ids"]

    @property
    def attn(self) -> Tensor:
        return self["attn"]

    @property
    def span_end(self) -> Tensor:
        return self["span_end"]

    @property
    def kept(self) -> Tensor:
        return self["kept"]


class LMEventTokenizer(nn.Module):
    """Serialise a batch of windows to text and tokenise it.

    Stands where :class:`~ehrjepa.models.embedding.EventEmbedding` stands: same
    call signature, and ``code_emb`` is still here -- a ``(vocab_size, dim)``
    table -- because the next-code head ties to it. Nothing else reads it: the LM
    has its own subword embeddings and this table is only ever the *output*
    projection of the code softmax, which is why it is small (47 x 256 on
    PhysioNet-2019) and why the "tied embeddings" arithmetic that dominates the
    from-scratch model's parameter count does not apply here.
    """

    def __init__(self, config: EHRJEPAConfig) -> None:
        super().__init__()
        if not config.lm_cache_dir:  # pragma: no cover - EHRJEPAConfig checks this
            raise ValueError("model.encoder='lm' needs model.lm_cache_dir")
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size
        # Parity with EventEmbedding, which the trainer and the target stack read.
        self.time_dropout = 0.0
        self.code_emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=PAD_ID)
        nn.init.normal_(self.code_emb.weight, std=0.02)
        with torch.no_grad():
            self.code_emb.weight[PAD_ID].zero_()
        self.phrases = code_phrases(config.lm_cache_dir)
        if len(self.phrases) != config.vocab_size:
            raise ValueError(
                f"model.lm_cache_dir={config.lm_cache_dir} has a "
                f"{len(self.phrases)}-code vocabulary, but this model's is {config.vocab_size}"
            )
        mean, std, log1p = value_scales(config.lm_cache_dir, config.vocab_size)
        self.register_buffer("value_mean", mean)
        self.register_buffer("value_std", std)
        self.register_buffer("value_log1p", log1p)
        self.tokenizer = load_tokenizer(config)
        self.max_tokens = int(config.lm_max_tokens)
        #: Fraction of the last batch's valid events that survived the token cap.
        #: Read by the trainer as ``lm_events_kept``; a diagnostic, not a
        #: gradient path.
        self.last_kept_frac: float = 1.0

    # ------------------------------------------------------------------ #

    def raw_values(self, code_id: Tensor, value_z: Tensor) -> Tensor:
        """Undo the cache's per-code normalisation: ``mean + z * std``, then ``expm1``."""
        mean = self.value_mean[code_id]
        std = self.value_std[code_id]
        raw = mean + value_z.to(mean.dtype) * std
        return torch.where(self.value_log1p[code_id], torch.expm1(raw), raw)

    def forward(
        self,
        code_id: Tensor,
        value_bin: Tensor,
        value_z: Tensor,
        age: Tensor,
        log_delta: Tensor,
        use_time: bool = True,
    ) -> LMTokens:
        """``(B, L)`` event tensors in, one :class:`LMTokens` out.

        ``age`` is accepted for signature parity and not used: the serialisation
        carries the gap between events, which is what the from-scratch stack's
        ``log_delta`` channel carries, and a patient's absolute age is already an
        ``AGE`` event of its own in both PhysioNet vocabularies.
        """
        device = code_id.device
        valid = (code_id != PAD_ID).cpu().numpy()
        codes = code_id.cpu().numpy()
        bins = value_bin.cpu().numpy()
        raw = self.raw_values(code_id, value_z).float().cpu().numpy()
        # ``log_delta`` is ``log1p`` of the hours since the previous event.
        hours = torch.expm1(log_delta.float()).clamp(min=0.0).cpu().numpy()
        spans = event_texts(self.phrases, codes, bins, raw, hours, valid, use_time=use_time)

        flat = [text for row in spans for text in row if text]
        lengths = np.zeros(valid.shape, dtype=np.int64)
        if flat:
            encoded = self.tokenizer(flat, add_special_tokens=False)["input_ids"]
            pieces: list[list[int]] = []
            cursor = 0
            for row in range(valid.shape[0]):
                ids_row: list[list[int]] = []
                for col in range(valid.shape[1]):
                    if not spans[row][col]:
                        ids_row.append([])
                        continue
                    ids_row.append(list(encoded[cursor]))
                    lengths[row, col] = len(encoded[cursor])
                    cursor += 1
                pieces.append(ids_row)
        else:  # pragma: no cover - an entirely padded batch cannot be collated
            pieces = [[[] for _ in range(valid.shape[1])] for _ in range(valid.shape[0])]

        # Tokens from position ``i`` to the end of the window. Every valid event
        # is at least one token, so this is non-increasing in ``i`` and the
        # comparison below keeps a contiguous *suffix*: the oldest events go
        # first, which is the whole point of truncating from the left.
        tail = np.cumsum(lengths[:, ::-1], axis=1)[:, ::-1]
        kept = valid & (tail <= self.max_tokens)
        n_valid = int(valid.sum())
        self.last_kept_frac = float(kept.sum() / n_valid) if n_valid else 1.0

        rows_ids: list[list[int]] = []
        span_end = np.zeros(valid.shape, dtype=np.int64)
        for row in range(valid.shape[0]):
            ids: list[int] = []
            for col in range(valid.shape[1]):
                if not kept[row, col]:
                    continue
                ids.extend(pieces[row][col])
                span_end[row, col] = len(ids) - 1
            rows_ids.append(ids)

        width = max(1, max(len(ids) for ids in rows_ids))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        ids_array = np.full((valid.shape[0], width), int(pad_id), dtype=np.int64)
        attn = np.zeros((valid.shape[0], width), dtype=np.int64)
        for row, ids in enumerate(rows_ids):
            ids_array[row, : len(ids)] = ids
            attn[row, : len(ids)] = 1
        return LMTokens(
            input_ids=torch.from_numpy(ids_array).to(device),
            attn=torch.from_numpy(attn).to(device),
            span_end=torch.from_numpy(span_end).to(device),
            kept=torch.from_numpy(kept).to(device),
        )


def load_tokenizer(config: EHRJEPAConfig):  # noqa: ANN201 - a HF tokenizer
    """The Hugging Face tokenizer named by ``lm_tokenizer`` or ``lm_name``.

    A module-level function rather than an inline call so a test can replace it
    with a small local tokenizer and never touch the network.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.lm_tokenizer or config.lm_name)


def load_base_model(config: EHRJEPAConfig) -> nn.Module:
    """The frozen pretrained trunk, in bf16.

    ``AutoModel``, not ``AutoModelForCausalLM``: only hidden states are read, and
    the vocabulary projection would be a 150k-wide matmul per token for an output
    nothing consumes. Replaceable by a test for the same reason
    :func:`load_tokenizer` is.
    """
    from transformers import AutoModel

    return AutoModel.from_pretrained(config.lm_name, dtype=torch.bfloat16)


class LMEncoder(nn.Module):
    """A frozen pretrained causal LM with LoRA, read out one row per event.

    Stands where :class:`~ehrjepa.models.encoder.Encoder` stands: called as
    ``encoder(tokens, valid_mask, return_penultimate=...)`` and returning an
    :class:`~ehrjepa.models.encoder.EncoderOutput` with ``cls`` and ``tokens`` at
    the model's own ``dim``.

    ``cls`` is a learned constant. That is not a shortcut: under causal attention
    a prefix summary row cannot exist without leaking the future back through it,
    which is exactly why :class:`~ehrjepa.models.encoder.Encoder` produces a
    constant CLS in its ``causal=True`` mode and why
    :mod:`ehrjepa.eval.probe` resolves ``auto`` to ``last`` for causal
    checkpoints. Keeping the row means every consumer of ``EncoderOutput`` -- the
    SIGReg term, the probe's ``cls_mean`` option -- still finds it.
    """

    def __init__(self, config: EHRJEPAConfig) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.config = config
        self.dim = config.dim
        #: Read by :func:`ehrjepa.eval.probe.default_features` via the checkpoint
        #: config, and true by construction here.
        self.causal = True
        base = load_base_model(config)
        self.hidden_size = int(base.config.hidden_size)
        targets = [name.strip() for name in config.lm_lora_targets.split(",") if name.strip()]
        if not targets:
            raise ValueError("model.lm_lora_targets must name at least one module")
        self.lm = get_peft_model(
            base,
            LoraConfig(
                r=config.lm_lora_r,
                lora_alpha=config.lm_lora_alpha,
                # Zero on purpose: with no dropout anywhere in the trunk the
                # forward pass is a deterministic function of the token ids,
                # which is what lets the shared-target path reuse the online
                # pass instead of running the base model twice per step.
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
            ),
        )
        for name, param in self.lm.named_parameters():
            param.requires_grad_("lora_" in name)
        # A bf16 trunk leaves the adapters in bf16 too, and AdamW on bf16
        # parameters loses the small updates a rank-8 adapter is made of.
        for param in self.lm.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        if config.lm_grad_checkpointing:
            base.gradient_checkpointing_enable()
            # Without this the first activation entering the checkpointed region
            # has no ``requires_grad`` -- the embedding table is frozen -- and
            # every recomputed block is silently dropped from the graph.
            base.enable_input_require_grads()
        self.proj = nn.Linear(self.hidden_size, config.dim)
        self.norm = nn.LayerNorm(config.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, config.dim))
        nn.init.trunc_normal_(self.proj.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def n_trainable(self) -> int:
        """Parameters this stack actually updates -- adapters, projection, CLS."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _rows(self, hidden: Tensor, span_end: Tensor, mask: Tensor) -> Tensor:
        """Gather one hidden state per event, project to ``dim``, zero the rest."""
        width = hidden.shape[-1]
        index = span_end.clamp(min=0)[:, :, None].expand(-1, -1, width)
        gathered = hidden.gather(1, index)
        rows = self.norm(self.proj(gathered.to(self.proj.weight.dtype)))
        return rows * mask[:, :, None].to(rows.dtype)

    def forward(
        self, tokens: LMTokens, valid_mask: Tensor | None = None, return_penultimate: bool = False
    ):  # noqa: ANN201 - EncoderOutput, imported lazily to avoid a cycle
        from ehrjepa.models.encoder import EncoderOutput

        mask = tokens.kept.bool()
        if valid_mask is not None:
            mask = mask & valid_mask.bool()
        out = self.lm(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attn,
            output_hidden_states=return_penultimate,
            use_cache=False,
        )
        cls = self.cls_token.expand(tokens.input_ids.shape[0], -1)
        result = EncoderOutput(
            cls=cls, tokens=self._rows(out.last_hidden_state, tokens.span_end, mask)
        )
        if return_penultimate:
            # The projection was fitted on the final layer, so a penultimate
            # readout through it is a different quantity than the from-scratch
            # stack's penultimate residual stream. Offered because the flag
            # exists; the grids read ``final``.
            states = out.hidden_states
            result["cls_penultimate"] = cls
            result["tokens_penultimate"] = self._rows(states[-2], tokens.span_end, mask)
        return result
