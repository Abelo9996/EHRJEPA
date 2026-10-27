"""Action-conditioned patient-state prediction (``objective.kind: state``).

The token stream is cut into time windows (:func:`ehrjepa.data.actions.assign_windows`)
and every token is an *action* (medication, infusion, procedure) or an
*observation* (everything else), by a per-code table built from the vocabulary.
Then, with the same causal encoder the ``ar`` and ``nextlatent`` runs use:

``s_t``
    The patient state of window ``t``: the causal encoder's outputs over the
    *observation* tokens of window ``t``, pooled (``last`` observation token,
    ``mean``, or learned single-query ``attn``). Every pooled row has attended to
    tokens at or before itself, so ``s_t`` is a function of windows ``<= t`` only.
    Because the encoder is causal, ``s_t`` already summarises the whole history
    ``s_{<=t}``; the predictor reads ``s_t`` alone rather than re-attending over
    earlier states (a deliberate simplification -- the history is in it).
``a_{t..t+k-1}``
    The actions of windows ``t .. t+k-1``: mean of their *content* embeddings
    (code + value, no time terms) plus an encoding of ``log1p`` of their count.
    Content embeddings, not encoder outputs: an encoder output at an action in
    window ``t+1`` has attended to that window's observations, which are the
    future being predicted. Actions in window ``t+k`` itself are not given.
``ŝ_{t+k} = f(s_t, a_{t..t+k-1}, k, Δt)``
    An MLP over ``[s_t ; act ; horizon embedding (+ time-to-target encoding)]``,
    for ``k`` in ``objective.state_horizons`` (counted in non-empty windows).
    ``Δt`` -- hours from the last event of window ``t`` to the first of window
    ``t+k`` -- is given when ``predictor.mask_token_time`` is on, exactly as the
    hybrid's heads are given the time of the event they predict.
target
    The same pooling of the *target* stack (EMA copy under ``model.target_mode:
    ema``) over window ``t+k``'s observation tokens, under stop-gradient, scored
    by smooth L1 against its layer norm (the repository's ``jepa_loss``).

``objective.state_actions: false`` is the ``action_blind`` ablation: the action
vector is replaced by zeros, everything else -- parameters, pairs, targets --
identical. A pair ``(t, t+k)`` is scored when both windows have a state; the
last window of a crop is never a target unless the crop ends at the subject's
last event (``at_end`` in the batch), because the crop may have cut it short.

The next-code AR loss rides along exactly as in :class:`EHRNextLatent`
(``lambda_recon``), so ``loss = lambda_pred * state + lambda_recon * AR`` --
the same two weights, in the same places, as the hybrid.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ehrjepa.data.actions import WINDOW_ANCHORS, assign_windows
from ehrjepa.data.tokenize import PAD_ID
from ehrjepa.models.encoder import _init_linear
from ehrjepa.models.jepa import EHRJEPAConfig, JEPAOutput, effective_valid
from ehrjepa.models.latent import _cat, _cat_index, _CausalLatent
from ehrjepa.models.layers import ScalarEncoder
from ehrjepa.objectives.ar import next_code_targets

__all__ = ["STATE_POOLS", "EHRStateModel", "StatePairs", "StatePool"]

STATE_POOLS = ("last", "mean", "attn")


class StatePool(nn.Module):
    """Pool token rows into one row per window, over a selected subset of tokens.

    ``attn`` is a single learned query over layer-normed rows, softmaxed within
    each window; ``last`` and ``mean`` have no parameters.
    """

    def __init__(self, dim: int, kind: str) -> None:
        super().__init__()
        if kind not in STATE_POOLS:
            raise ValueError(f"state_pool must be one of {STATE_POOLS}, got {kind!r}")
        self.kind = kind
        if kind == "attn":
            self.norm = nn.LayerNorm(dim)
            self.query = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.query, std=0.02)

    def forward(self, hidden: Tensor, window: Tensor, select: Tensor, n_windows: int) -> Tensor:
        """``(B, W, D)`` from ``hidden (B, L, D)``; empty windows pool to zero."""
        b, length, dim = hidden.shape
        width = max(1, n_windows)
        rows, cols = select.nonzero(as_tuple=True)
        slot = rows * width + window[rows, cols]
        out = hidden.new_zeros(b * width, dim)
        if rows.numel() == 0:
            return out.view(b, width, dim)
        if self.kind == "last":
            last = torch.full((b * width,), -1, dtype=torch.long, device=hidden.device)
            last = last.scatter_reduce(0, slot, cols, reduce="amax", include_self=True)
            filled = last >= 0
            owner = torch.arange(b * width, device=hidden.device) // width
            out[filled] = hidden[owner[filled], last[filled]]
            return out.view(b, width, dim)
        picked = hidden[rows, cols]
        if self.kind == "mean":
            out = out.index_add(0, slot, picked)
            count = torch.zeros(b * width, device=hidden.device)
            count = count.index_add(0, slot, torch.ones(slot.shape[0], device=hidden.device))
            return (out / count.clamp(min=1)[:, None].to(out.dtype)).view(b, width, dim)
        score = (self.norm(picked).float() @ self.query.float()) / math.sqrt(dim)
        peak = torch.full((b * width,), -float("inf"), device=hidden.device)
        peak = peak.scatter_reduce(0, slot, score.detach(), reduce="amax", include_self=True)
        weight = torch.exp(score - peak[slot])
        denom = torch.zeros(b * width, device=hidden.device).index_add(0, slot, weight)
        weight = (weight / denom[slot]).to(hidden.dtype)
        out = out.index_add(0, slot, picked * weight[:, None])
        return out.view(b, width, dim)


@dataclass
class StatePairs:
    """Everything one batch's state prediction needs, before the predictor runs."""

    hidden: Tensor  # (B, L, D) online encoder outputs
    cls: Tensor  # (B, D) online CLS row (SIGReg reads it, as for the hybrid)
    valid: Tensor  # (B, L) bool
    targets: Tensor | None  # (B, W, D) pooled target states, stop-gradient
    states: Tensor  # (B, W, D) pooled online states
    rows: list[Tensor]  # per horizon: batch index of each pair
    cols: list[Tensor]  # per horizon: window index t of each pair
    act_mean: list[Tensor]  # per horizon: (N, D) mean action content embedding
    act_count: list[Tensor]  # per horizon: (N,) actions in windows t..t+k-1
    log_dt: list[Tensor]  # per horizon: (N,) log1p hours from window t to t+k
    action_frac: Tensor  # scalar: share of valid tokens that are actions


class EHRStateModel(_CausalLatent):
    """Causal encoder + window-state pooling + action-conditioned state predictor."""

    def __init__(self, config: EHRJEPAConfig) -> None:
        if config.encoder != "scratch":
            raise ValueError("objective.kind=state needs model.encoder=scratch")
        if config.state_anchor not in WINDOW_ANCHORS:
            raise ValueError(
                f"state_anchor must be one of {WINDOW_ANCHORS}, got {config.state_anchor!r}"
            )
        if not config.state_horizons or any(k < 1 for k in config.state_horizons):
            raise ValueError(f"state_horizons must be positive, got {config.state_horizons}")
        if config.state_window_minutes < 1:
            raise ValueError("state_window_minutes must be >= 1")
        super().__init__(config)
        dim = config.dim
        self.state_pool = StatePool(dim, config.state_pool)
        self.target_pool: StatePool | None = None
        if self.uses_target_copy:
            self.target_pool = copy.deepcopy(self.state_pool).requires_grad_(False)
        self.act_proj = nn.Linear(dim, dim)
        self.act_count_enc = ScalarEncoder(dim, config.n_freq)
        self.horizon_emb = nn.Embedding(len(config.state_horizons), dim)
        nn.init.normal_(self.horizon_emb.weight, std=0.02)
        self.dt_enc = ScalarEncoder(dim, config.n_freq)
        self.state_head = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.state_head.apply(_init_linear)
        _init_linear(self.act_proj)
        # Per-code tables, filled from the vocabulary by the trainer
        # (:meth:`set_code_tables`) and saved in the ``state_dict``, so an eval
        # machine rebuilds them from the checkpoint without the action config.
        self.register_buffer("action_table", torch.zeros(config.vocab_size, dtype=torch.bool))
        self.register_buffer("admission_table", torch.zeros(config.vocab_size, dtype=torch.bool))

    @property
    def heads(self) -> nn.ModuleList:
        return nn.ModuleList(
            [
                self.state_pool,
                self.act_proj,
                self.act_count_enc,
                self.horizon_emb,
                self.dt_enc,
                self.state_head,
            ]
        )

    def set_code_tables(self, action: Tensor, admission: Tensor | None = None) -> None:
        """Install the per-code action (and admission) flags, ``(vocab_size,)`` bool."""
        with torch.no_grad():
            self.action_table.copy_(torch.as_tensor(action, dtype=torch.bool))
            if admission is not None:
                self.admission_table.copy_(torch.as_tensor(admission, dtype=torch.bool))

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        super().update_ema(momentum)
        if self.uses_ema and self.target_pool is not None:
            pairs = zip(self.state_pool.parameters(), self.target_pool.parameters(), strict=True)
            for p_online, p_target in pairs:
                p_target.mul_(momentum).add_(p_online.detach(), alpha=1.0 - momentum)

    # ------------------------------------------------------------------ #

    def windows(self, batch: Mapping[str, Tensor], valid: Tensor) -> tuple[Tensor, Tensor]:
        """``(window, n_windows)`` for the batch under this model's window rule."""
        admission = None
        if self.config.state_anchor == "admission":
            admission = self.admission_table[batch["code_id"]]
        return assign_windows(batch["time_min"], valid, self.config.state_window_minutes, admission)

    def build_pairs(self, batch: Mapping[str, Tensor], compute_targets: bool = True) -> StatePairs:
        if "time_min" not in batch:
            raise KeyError("the state objective needs `time_min`; collate the full feature set")
        tokens = self.embed_batch(batch)
        valid = effective_valid(tokens, batch["attention_mask"])
        encoded = self.encoder(tokens, valid)
        hidden = encoded.tokens
        targets_tok = self.window_targets(batch, tokens) if compute_targets else None

        window, n_windows = self.windows(batch, valid)
        width = max(1, int(n_windows.max()))
        is_action = self.action_table[batch["code_id"]] & valid
        is_obs = valid & ~is_action
        states = self.state_pool(hidden, window, is_obs, width)
        targets = None
        if targets_tok is not None:
            pool = self.target_pool if self.target_pool is not None else self.state_pool
            with torch.no_grad():
                targets = pool(targets_tok, window, is_obs, width).detach()

        b, length = valid.shape
        device = hidden.device
        flat = torch.arange(b, device=device)[:, None] * width + window.clamp(min=0)
        obs_count = torch.zeros(b * width, device=device).index_add(
            0, flat[is_obs], torch.ones(int(is_obs.sum()), device=device)
        )
        has_state = (obs_count > 0).view(b, width)
        times = batch["time_min"].to(torch.int64)
        big = torch.iinfo(torch.int64).max
        first = torch.full((b * width,), big, dtype=torch.int64, device=device)
        first = first.scatter_reduce(0, flat[valid], times[valid], reduce="amin").view(b, width)
        last = torch.full((b * width,), -big, dtype=torch.int64, device=device)
        last = last.scatter_reduce(0, flat[valid], times[valid], reduce="amax").view(b, width)

        # Action content embeddings: code + value, no time terms, summed per window.
        content = self.embed(
            batch["code_id"],
            batch["value_bin"],
            batch["value_z"],
            batch["age"],
            batch["log_delta"],
            use_time=False,
        )
        dim = content.shape[-1]
        act_sum = content.new_zeros(b * width, dim).index_add(
            0, flat[is_action], content[is_action]
        )
        act_n = torch.zeros(b * width, device=device).index_add(
            0, flat[is_action], torch.ones(int(is_action.sum()), device=device)
        )
        zero_row = content.new_zeros(b, 1, dim, dtype=torch.float32)
        prefix = torch.cat((zero_row, act_sum.view(b, width, dim).float().cumsum(dim=1)), dim=1)
        prefix_n = torch.cat((act_n.new_zeros(b, 1), act_n.view(b, width).cumsum(dim=1)), dim=1)

        index = torch.arange(width, device=device)[None, :]
        at_end = batch.get("at_end")
        at_end = (
            torch.zeros(b, dtype=torch.bool, device=device) if at_end is None else at_end.bool()
        )
        target_ok = has_state & ((index < (n_windows[:, None] - 1)) | at_end[:, None])

        rows: list[Tensor] = []
        cols: list[Tensor] = []
        act_mean: list[Tensor] = []
        act_count: list[Tensor] = []
        log_dt: list[Tensor] = []
        for k in self.config.state_horizons:
            if k >= width:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                rows.append(empty)
                cols.append(empty)
                act_mean.append(content.new_zeros(0, dim))
                act_count.append(torch.zeros(0, device=device))
                log_dt.append(torch.zeros(0, device=device))
                continue
            keep = has_state[:, :-k] & target_ok[:, k:]
            r, t = keep.nonzero(as_tuple=True)
            total = prefix[r, t + k] - prefix[r, t]
            count = prefix_n[r, t + k] - prefix_n[r, t]
            rows.append(r)
            cols.append(t)
            act_mean.append((total / count.clamp(min=1)[:, None]).to(content.dtype))
            act_count.append(count)
            gap = (first[r, t + k] - last[r, t]).clamp(min=0).float() / 60.0
            log_dt.append(torch.log1p(gap))
        frac = is_action.sum().float() / valid.sum().clamp(min=1).float()
        return StatePairs(
            hidden=hidden,
            cls=encoded.cls,
            valid=valid,
            targets=targets,
            states=states,
            rows=rows,
            cols=cols,
            act_mean=act_mean,
            act_count=act_count,
            log_dt=log_dt,
            action_frac=frac.detach(),
        )

    def action_vector(self, act_mean: Tensor, act_count: Tensor) -> Tensor:
        """The predictor's action input; zeros under ``state_actions: false``."""
        vec = self.act_proj(act_mean) + self.act_count_enc(torch.log1p(act_count))
        if not self.config.state_actions:
            return torch.zeros_like(vec)
        return vec

    def predict(self, state: Tensor, action: Tensor, slot: int, log_dt: Tensor) -> Tensor:
        """``ŝ_{t+k}`` from ``s_t``, the action vector, horizon slot and time gap."""
        cond = self.horizon_emb.weight[slot].expand_as(state)
        if self.config.mask_token_time:
            cond = cond + self.dt_enc(log_dt)
        return self.state_head(torch.cat((state, action, cond.to(state.dtype)), dim=-1))

    def forward(  # type: ignore[override]
        self, batch: Mapping[str, Tensor], compute_targets: bool = True
    ) -> JEPAOutput:
        pairs = self.build_pairs(batch, compute_targets=compute_targets)
        hidden, valid = pairs.hidden, pairs.valid
        predictions: list[Tensor] = []
        target_rows: list[Tensor] = []
        sizes: list[int] = []
        gains: list[Tensor] = []
        deltas: list[Tensor] = []
        for slot, k in enumerate(self.config.state_horizons):
            r, t = pairs.rows[slot], pairs.cols[slot]
            state = pairs.states[r, t]
            action = self.action_vector(pairs.act_mean[slot], pairs.act_count[slot])
            predicted = self.predict(state, action, slot, pairs.log_dt[slot])
            target = (
                pairs.targets[r, t + k]
                if pairs.targets is not None
                else torch.zeros_like(predicted)
            )
            predictions.append(predicted)
            target_rows.append(target)
            sizes.append(int(predicted.shape[0]))
            n = int(predicted.shape[0])
            if self.config.state_actions and n >= 2 and pairs.targets is not None:
                with torch.no_grad():
                    # Actions rolled half the batch away -- a different patient's
                    # (or a far window's) actions, deterministically, so logging
                    # never draws from the training RNG.
                    rolled = self.predict(
                        state, action.roll(max(1, n // 2), dims=0), slot, pairs.log_dt[slot]
                    )
                    normed = F.layer_norm(target.float(), (target.shape[-1],))
                    true_loss = F.smooth_l1_loss(predicted.float(), normed)
                    gains.append(F.smooth_l1_loss(rolled.float(), normed) - true_loss)
                    deltas.append(
                        (predicted.float() - rolled.float()).norm(dim=-1).mean()
                        / predicted.float().norm(dim=-1).mean().clamp(min=1e-8)
                    )

        nan = torch.tensor(float("nan"), device=hidden.device)
        extras: dict[str, Tensor] = {
            "horizon_sizes": torch.tensor(sizes, dtype=torch.long, device=hidden.device),
            "diag_state_pairs": torch.tensor(float(sum(sizes)), device=hidden.device),
            "diag_action_frac": pairs.action_frac,
            "diag_act_gain": torch.stack(gains).mean() if gains else nan,
            "diag_act_delta": torch.stack(deltas).mean() if deltas else nan,
        }
        if self.recon_head is not None:
            code_targets = next_code_targets(batch["code_id"], valid)
            scored = code_targets != PAD_ID
            extras["recon_hidden"] = hidden[scored]
            extras["recon_code_id"] = code_targets[scored]
            if self.recon_value_head is not None:
                shifted = torch.zeros_like(batch["value_bin"])
                shifted[:, :-1] = batch["value_bin"][:, 1:]
                extras["recon_value_bin"] = shifted[scored]
        rows = list(pairs.rows)
        cols = [t + k for t, k in zip(pairs.cols, self.config.state_horizons, strict=True)]
        return JEPAOutput(
            predictions=_cat(predictions, hidden),
            targets=_cat(target_rows, hidden),
            target_index=(_cat_index(rows, hidden), _cat_index(cols, hidden)),
            context_tokens=hidden,
            context_mask=valid,
            cls=pairs.cls,
            extras=extras,
        )
