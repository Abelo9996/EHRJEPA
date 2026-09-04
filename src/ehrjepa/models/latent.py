"""Two *causal* latent-prediction models: dense next-latent, and window-pooled latent.

Both exist because of one measurement. On the phase-5 pilot grids, masked-span
latent JEPA -- every target/masking/SIGReg variant tried -- moved the linear
probes not at all over its own untrained control, while next-code AR moved them
by about five AUROC points and auxiliary code reconstruction by about half that.
The two properties AR has and masked-span JEPA does not are **supervision
density** (a term at every position, not at the 10-30% of positions a mask
covers) and a **discrete target**. These two models put the first property into a
latent objective, in the two places it can go:

:class:`EHRNextLatent` (``objective.kind: nextlatent``)
    LeNEPA-style. The encoder runs causally, and at *every* position ``i`` a small
    MLP maps ``h_i`` to the target encoder's latent at ``i + k``, for each ``k``
    in ``objective.horizons``. One term per position per horizon: as dense as the
    AR loss, and predicting a representation rather than a code id.

:class:`EHRWindowLatent` (``objective.kind: window``)
    The clinical shape of the same idea. At each of ``K`` sampled anchors, predict
    the *mean-pooled* target latent of everything that happens in the next 30 or
    365 days. Far fewer terms, but each one is a summary of a horizon a downstream
    task actually asks about, rather than of the single next event.

**What the context summary is.** Both take it from a causal encoder pass: the
representation at position ``i`` (``nextlatent``) or ``a - 1`` (``window``) has
attended to events ``0..i`` / ``0..a-1`` and to nothing later, so "strictly before
the anchor" is enforced by the attention mask rather than by a second forward
pass over a truncated window. That is a deliberate simplification for
``window``: the alternative -- re-running a bidirectional encoder per anchor over
``events[:a]`` -- is ``K`` times the compute for a context that differs only in
being bidirectional. RoPE sees the true window offsets either way.

**What the target is.** ``EHRJEPA.window_targets``: the EMA copy when
``model.target_mode`` is ``ema``, the online encoder under stop-gradient when it
is ``shared``, with ``target.time_features`` honoured. The target encoder inherits
``causal``, so ``z_j`` summarises ``0..j`` -- which means ``z_{i+1}`` contains a
prefix ``h_i`` already knows. The predictor can therefore score well by copying,
and ``cos_gap`` against shuffled targets is the diagnostic that says whether it
did more than that. ``target.span_only`` has no meaning without a target span and
is rejected at construction rather than silently ignored.

**What is skipped.** ``nextlatent`` scores position ``i`` at horizon ``k`` only
when ``i`` and ``i + k`` are both real events, so a window of length ``L``
contributes ``L - k`` terms at most. ``window`` skips an anchor/horizon pair when
the horizon runs past the last event in the window (the future is not observed,
so an empty or truncated pool would be indistinguishable from a quiet one) or
when no event falls inside it (there is nothing to pool). Both counts come back
in ``extras`` and are logged as ``skipped_frac``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ehrjepa.data.tokenize import PAD_ID
from ehrjepa.models.encoder import _init_linear
from ehrjepa.models.jepa import EHRJEPA, EHRJEPAConfig, JEPAOutput
from ehrjepa.models.layers import ScalarEncoder
from ehrjepa.objectives.ar import next_code_targets

__all__ = ["LATENT_MODELS", "EHRNextLatent", "EHRWindowLatent", "LatentPredictor"]

#: Minutes per day, for turning ``objective.window_horizons`` into ``time_min``.
MINUTES_PER_DAY = 1440.0


class LatentPredictor(nn.Module):
    """``Linear -> GELU -> Linear`` at the encoder's own width.

    Deliberately not the transformer :class:`~ehrjepa.models.predictor.Predictor`:
    there is no set of masked positions to attend over here, only one context
    vector per prediction, so a per-position MLP is the whole of what the
    architecture needs. Keeping it at ``dim`` rather than wider is the usual JEPA
    precaution -- representation quality must not be offloadable into the head.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.apply(_init_linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _CausalLatent(EHRJEPA):
    """Shared construction for the two causal latent models.

    Subclasses :class:`~ehrjepa.models.jepa.EHRJEPA` for its embedding, encoder,
    EMA target stack and reconstruction head, and turns off ``build_predictor``
    so the transformer predictor those objectives do not use is never allocated.
    """

    def __init__(self, config: EHRJEPAConfig) -> None:
        if not config.causal:
            raise ValueError(f"{type(self).__name__} needs model.causal=true")
        if config.target_span_only:
            raise ValueError(f"{type(self).__name__} has no target span; unset target.span_only")
        if config.build_predictor:
            raise ValueError(f"{type(self).__name__} needs model.build_predictor=false")
        super().__init__(config)

    def n_parameters(self) -> dict[str, int]:
        """As :meth:`EHRJEPA.n_parameters`, with the MLP heads under ``predictor``."""
        counts = super().n_parameters()
        counts["predictor"] = sum(p.numel() for p in self.heads.parameters())
        recon = self.recon_head
        counts["head"] = 0 if recon is None else sum(p.numel() for p in recon.parameters())
        return counts

    @property
    def heads(self) -> nn.ModuleList:  # pragma: no cover - overridden
        raise NotImplementedError


class EHRNextLatent(_CausalLatent):
    """Dense causal next-latent prediction, one MLP head per horizon.

    ``objective.horizons`` (``[1]``, or e.g. ``[1, 4, 16]``) names the step
    offsets. Each gets its own :class:`LatentPredictor`; a shared head with a
    horizon embedding would have been the other option, and the separate heads
    are chosen because they make "horizon 4 learned nothing" a readable statement
    about one module rather than about one row of an embedding table.

    With ``predictor.mask_token_time`` on, the head's input is
    ``h_i + f(age_{i+k}) + f(log_delta_{i+k})`` -- the *time* of the event being
    predicted, and nothing else about it. Off, it is ``h_i`` alone and the model
    is asked to predict the next latent unconditionally on when it happens.
    """

    def __init__(self, config: EHRJEPAConfig) -> None:
        super().__init__(config)
        if not config.horizons or any(k < 1 for k in config.horizons):
            raise ValueError(f"objective.horizons must be positive, got {config.horizons}")
        self.pred_heads = nn.ModuleList(LatentPredictor(config.dim) for _ in config.horizons)
        if config.share_time_encoders:
            self.pred_age_enc: nn.Module = self.embed.age_enc
            self.pred_delta_enc: nn.Module = self.embed.delta_enc
        else:
            self.pred_age_enc = ScalarEncoder(config.dim, config.n_freq)
            self.pred_delta_enc = ScalarEncoder(config.dim, config.n_freq)

    @property
    def heads(self) -> nn.ModuleList:
        return self.pred_heads

    def forward(  # type: ignore[override]
        self, batch: Mapping[str, Tensor], compute_targets: bool = True
    ) -> JEPAOutput:
        """Predict ``z_{i+k}`` from ``h_i`` at every valid ``i``, for every horizon.

        ``compute_targets=False`` (set by the trainer when
        ``objective.lambda_pred`` is 0) skips the target encoder entirely and
        returns a zero placeholder, exactly as the masked-span model does; with
        ``lambda_recon`` positive that configuration *is* the AR objective, run
        through this class's encoder.
        """
        valid = batch["attention_mask"].bool()
        tokens = self.embed_batch(batch)
        context = self.encoder(tokens, valid)
        hidden = context.tokens
        targets = self.window_targets(batch, tokens) if compute_targets else None

        length = hidden.shape[1]
        predictions: list[Tensor] = []
        target_rows: list[Tensor] = []
        sizes: list[int] = []
        rows: list[Tensor] = []
        cols: list[Tensor] = []
        for head, step in zip(self.pred_heads, self.config.horizons, strict=True):
            if step >= length:
                sizes.append(0)
                continue
            keep = valid[:, :-step] & valid[:, step:]
            x = hidden[:, :-step]
            if self.config.mask_token_time:
                age = self.pred_age_enc(batch["age"][:, step:])
                delta = self.pred_delta_enc(batch["log_delta"][:, step:])
                x = x + age + delta
            predicted = head(x[keep])
            predictions.append(predicted)
            target_rows.append(
                targets[:, step:][keep] if targets is not None else torch.zeros_like(predicted)
            )
            row, col = keep.nonzero(as_tuple=True)
            rows.append(row)
            cols.append(col)
            sizes.append(int(predicted.shape[0]))

        extras: dict[str, Tensor] = {
            "horizon_sizes": torch.tensor(sizes, dtype=torch.long, device=hidden.device)
        }
        if self.recon_head is not None:
            # The AR term, verbatim: the encoder's own outputs, not the predicted
            # latents, so `lambda_pred: 0` with `lambda_recon: 1` is the next-code
            # objective and nothing else.
            code_targets = next_code_targets(batch["code_id"], valid)
            scored = code_targets != PAD_ID
            extras["recon_hidden"] = hidden[scored]
            extras["recon_code_id"] = code_targets[scored]
        return JEPAOutput(
            predictions=_cat(predictions, hidden),
            targets=_cat(target_rows, hidden),
            target_index=(_cat_index(rows, hidden), _cat_index(cols, hidden)),
            context_tokens=hidden,
            context_mask=valid,
            cls=context.cls,
            extras=extras,
        )


class EHRWindowLatent(_CausalLatent):
    """Predict the mean-pooled latent of the next ``H`` days, at ``K`` anchors.

    One :class:`LatentPredictor` shared across horizons, with a learned horizon
    embedding summed into its input -- the opposite choice from
    :class:`EHRNextLatent`, and made because the horizons here are a *scale*
    (30 days, 365 days) rather than a set of separate tasks, so sharing the map
    and varying one additive vector is the parameterisation that says so.

    The pooled target for anchor ``a`` at horizon ``H`` is the mean of the target
    encoder's outputs over events whose ``time_min`` lies in
    ``(t_a, t_a + H days]`` -- strictly after the anchor event, inclusive of the
    boundary. Everything is read from the same window; nothing outside it is
    available, which is what the skip rule below is about.
    """

    def __init__(self, config: EHRJEPAConfig) -> None:
        super().__init__(config)
        if not config.window_horizons or any(h <= 0 for h in config.window_horizons):
            raise ValueError(
                f"objective.window_horizons must be positive days, got {config.window_horizons}"
            )
        self.window_head = LatentPredictor(config.dim)
        self.horizon_emb = nn.Embedding(len(config.window_horizons), config.dim)
        nn.init.normal_(self.horizon_emb.weight, std=0.02)

    @property
    def heads(self) -> nn.ModuleList:
        return nn.ModuleList([self.window_head, self.horizon_emb])

    def forward(  # type: ignore[override]
        self,
        batch: Mapping[str, Tensor],
        anchors: Tensor,
        anchor_mask: Tensor,
        compute_targets: bool = True,
    ) -> JEPAOutput:
        """``anchors``/``anchor_mask`` are ``(B, K)`` from
        :func:`~ehrjepa.data.masking.sample_anchors`.

        An anchor/horizon pair is dropped when the horizon end falls past the last
        event of the window (unobserved future) or when no event falls inside it
        (nothing to pool). ``extras`` carries the two counts and the offered
        total; the objective turns them into ``skipped_frac``.
        """
        if "time_min" not in batch:
            raise KeyError("the window objective needs `time_min`; collate the full feature set")
        valid = batch["attention_mask"].bool()
        tokens = self.embed_batch(batch)
        context = self.encoder(tokens, valid)
        hidden = context.tokens
        targets = self.window_targets(batch, tokens) if compute_targets else None

        dim = hidden.shape[-1]
        # int64 throughout: ``time_min`` is minutes since epoch, which passes
        # 2^24 in 2001, so float32 cannot represent every value and "strictly
        # after the anchor" would silently include or drop events a minute apart.
        times = batch["time_min"].to(torch.int64)  # (B, L)
        floor = torch.full_like(times, torch.iinfo(torch.int64).min)
        last_time = torch.where(valid, times, floor).max(dim=1).values
        anchor_mask = anchor_mask.bool() & (anchors >= 1)
        index = (anchors - 1).clamp(min=0)
        summary = hidden.gather(1, index[:, :, None].expand(-1, -1, dim))  # (B, K, dim)
        anchor_time = times.gather(1, anchors)  # (B, K)

        predictions: list[Tensor] = []
        target_rows: list[Tensor] = []
        code_rows: list[Tensor] = []
        sizes: list[int] = []
        rows: list[Tensor] = []
        cols: list[Tensor] = []
        offered = unobserved = empty = 0
        for slot, days in enumerate(self.config.window_horizons):
            end = anchor_time + round(days * MINUTES_PER_DAY)
            inside = (
                valid[:, None, :]
                & (times[:, None, :] > anchor_time[:, :, None])
                & (times[:, None, :] <= end[:, :, None])
            )  # (B, K, L)
            counts = inside.sum(dim=-1)
            observed = end <= last_time[:, None]
            keep = anchor_mask & observed & (counts > 0)
            offered += int(anchor_mask.sum())
            unobserved += int((anchor_mask & ~observed).sum())
            empty += int((anchor_mask & observed & (counts == 0)).sum())

            predicted = self.window_head(summary[keep] + self.horizon_emb.weight[slot])
            predictions.append(predicted)
            if targets is not None:
                weights = inside.to(targets.dtype) / counts.clamp(min=1)[..., None].to(
                    targets.dtype
                )
                pooled = torch.einsum("bkl,bld->bkd", weights, targets)
                target_rows.append(pooled[keep])
            else:
                target_rows.append(torch.zeros_like(predicted))
            if self.recon_head is not None:
                codes = batch["code_id"][:, None, :].expand(-1, anchors.shape[1], -1)
                code_rows.append(torch.where(inside, codes, PAD_ID)[keep])
            row, col = keep.nonzero(as_tuple=True)
            rows.append(row)
            cols.append(anchors[keep])
            sizes.append(int(predicted.shape[0]))

        extras: dict[str, Tensor] = {
            "horizon_sizes": torch.tensor(sizes, dtype=torch.long, device=hidden.device),
            "anchors_offered": torch.tensor(float(offered), device=hidden.device),
            "anchors_unobserved": torch.tensor(float(unobserved), device=hidden.device),
            "anchors_empty": torch.tensor(float(empty), device=hidden.device),
        }
        if code_rows:
            window_codes = _cat(code_rows, hidden).long()
            extras["window_codes"] = window_codes
            extras["positives_per_anchor"] = _distinct_per_row(window_codes)
        return JEPAOutput(
            predictions=_cat(predictions, hidden),
            targets=_cat(target_rows, hidden),
            target_index=(_cat_index(rows, hidden), _cat_index(cols, hidden)),
            context_tokens=hidden,
            context_mask=valid,
            cls=context.cls,
            extras=extras,
        )


def _cat(parts: list[Tensor], like: Tensor) -> Tensor:
    """Concatenate rows, or an empty ``(0, dim)`` of the right dtype and device."""
    if not parts:
        return like.new_zeros((0, like.shape[-1]))
    return torch.cat(parts, dim=0)


def _cat_index(parts: list[Tensor], like: Tensor) -> Tensor:
    if not parts:
        return torch.zeros(0, dtype=torch.long, device=like.device)
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _distinct_per_row(codes: Tensor) -> Tensor:
    """Mean number of distinct non-``PAD`` code ids per row of an ``(N, L)`` tensor.

    This is the count of positive labels the multi-label head is asked for, which
    is not the number of *events* in the horizon: a code repeated three times in
    thirty days is one positive.
    """
    if codes.numel() == 0:
        return torch.zeros((), device=codes.device)
    ordered, _ = codes.sort(dim=1)
    fresh = torch.ones_like(ordered, dtype=torch.bool)
    fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    return (fresh & (ordered != PAD_ID)).sum(dim=1).float().mean()


#: ``objective.kind`` -> class, for the trainer and the probe's ``load_encoder``.
LATENT_MODELS: dict[str, type[_CausalLatent]] = {
    "nextlatent": EHRNextLatent,
    "window": EHRWindowLatent,
}
