"""The action-conditioned state model, the MLM baseline, and their plumbing.

Pinned here:

1. Windowing: calendar and admission windows, ties grouped by time, padding -1.
2. The action/observation split by MEDS code prefix, pattern and codes file.
3. No future leakage across windows: ``s_t`` (online) and the target state of
   window ``t`` do not move when anything after window ``t`` changes, and the
   prediction for ``(t, t+k)`` does not move when observations after ``t`` or
   actions at or after ``t+k`` change -- while it *does* move with the actions
   in ``t..t+k-1``, unless the model is ``action_blind``.
4. The EMA target: no gradient, and the attention pooler moves with the encoder.
5. The loss is finite (including a batch with no scorable pair), backward runs.
6. The trainer runs both kinds end to end, and the probe reloads the checkpoint
   with its action table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ehrjepa.data.actions import (
    DEFAULT_ACTION_PATTERNS,
    DEFAULT_ACTION_PREFIXES,
    DEFAULT_ADMISSION_PREFIXES,
    assign_windows,
    code_flags,
    window_stats,
)
from ehrjepa.data.dataset import EventSequenceDataset, collate_events
from ehrjepa.data.tokenize import MASK_ID
from ehrjepa.eval import probe
from ehrjepa.eval.finetune import build_model
from ehrjepa.eval.state_sensitivity import action_sensitivity
from ehrjepa.models.jepa import EHRJEPAConfig
from ehrjepa.models.mlm import EHRMLM, mlm_inputs
from ehrjepa.models.state import EHRStateModel
from ehrjepa.objectives.ar import ar_loss_chunked
from ehrjepa.objectives.latent import LatentObjective
from ehrjepa.objectives.loss import ObjectiveConfig
from ehrjepa.train.config import load_config
from ehrjepa.train.pretrain import LOG_COLUMNS, Trainer

REPO = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO / "configs" / "pretrain_debug.yaml"
DEMO_CACHE = REPO / "data" / "cache" / "mimic-demo"
ITEMS = REPO / "configs" / "actions" / "mimic_icu_action_items.txt"

requires_cache = pytest.mark.skipif(
    not (DEMO_CACHE / "meta.json").exists(), reason="mimic-demo cache is not built"
)

VOCAB = 64
ACTION_LO = 40  # codes 40..63 are actions in the synthetic vocabulary
DAY = 1440
#: Within-window offsets (minutes) of the five events of each window; two ties.
OFFSETS = (0, 60, 60, 120, 600)
#: Which of the five slots are actions.
ACTION_SLOTS = (1, 3)
N_WINDOWS = 6


def _config(**overrides) -> EHRJEPAConfig:
    values = dict(
        vocab_size=VOCAB,
        dim=32,
        depth=2,
        heads=4,
        mlp="gelu",
        mlp_ratio=2.0,
        dropout=0.0,
        n_freq=8,
        causal=True,
        build_predictor=False,
        target_mode="shared",
        recon_head=True,
        state_horizons=[1, 2, 4],
        state_window_minutes=DAY,
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _model(**overrides) -> EHRStateModel:
    torch.manual_seed(0)
    model = EHRStateModel(_config(**overrides))
    table = torch.zeros(VOCAB, dtype=torch.bool)
    table[ACTION_LO:] = True
    model.set_code_tables(table)
    return model.eval()


def _batch(batch: int = 2) -> dict[str, torch.Tensor]:
    """``N_WINDOWS`` calendar days of five events each; slots 1 and 3 are actions."""
    g = torch.Generator().manual_seed(3)
    times, codes = [], []
    for w in range(N_WINDOWS):
        for slot, off in enumerate(OFFSETS):
            times.append(w * DAY + off)
            lo, hi = (ACTION_LO, VOCAB) if slot in ACTION_SLOTS else (4, ACTION_LO)
            codes.append(int(torch.randint(lo, hi, (1,), generator=g)))
    length = len(times)
    return {
        "code_id": torch.tensor(codes).repeat(batch, 1),
        "value_bin": torch.randint(0, 11, (batch, length), generator=g),
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 90,
        "log_delta": torch.rand(batch, length, generator=g) * 3,
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
        "time_min": torch.tensor(times).repeat(batch, 1),
        "at_end": torch.ones(batch, dtype=torch.bool),
    }


def _positions(window: int, kind: str) -> list[int]:
    """Token positions of ``kind`` (``obs``/``act``/``all``) in ``window``."""
    base = window * len(OFFSETS)
    out = []
    for slot in range(len(OFFSETS)):
        is_act = slot in ACTION_SLOTS
        if kind == "all" or (kind == "act") == is_act:
            out.append(base + slot)
    return out


def _tamper(batch: dict, positions: list[int], action: bool, seed: int = 11) -> dict:
    out = {k: v.clone() for k, v in batch.items()}
    g = torch.Generator().manual_seed(seed)
    lo, hi = (ACTION_LO, VOCAB) if action else (4, ACTION_LO)
    idx = torch.tensor(positions)
    out["code_id"][:, idx] = torch.randint(
        lo, hi, (out["code_id"].shape[0], len(positions)), generator=g
    )
    out["value_bin"][:, idx] = torch.randint(
        0, 11, (out["code_id"].shape[0], len(positions)), generator=g
    )
    out["value_z"][:, idx] = torch.randn(out["code_id"].shape[0], len(positions), generator=g)
    return out


def _prediction(model: EHRStateModel, batch: dict, t: int, k: int) -> torch.Tensor:
    """Row 0's prediction for the pair ``(t, t + k)``."""
    with torch.no_grad():
        out = model(batch)
    sizes = out.extras["horizon_sizes"].tolist()
    slot = model.config.state_horizons.index(k)
    start = sum(sizes[:slot])
    rows, cols = out.target_index
    rows, cols = rows[start : start + sizes[slot]], cols[start : start + sizes[slot]]
    hit = ((rows == 0) & (cols == t + k)).nonzero(as_tuple=True)[0]
    assert hit.numel() == 1, (t, k)
    return out.predictions[start + int(hit)]


# --------------------------------------------------------------------------- #
# Windowing and the action split
# --------------------------------------------------------------------------- #


def test_calendar_windows_group_by_day_keep_ties_together_and_mark_padding() -> None:
    times = torch.tensor([[0, 10, 10, DAY - 1, DAY, DAY, 5 * DAY + 3, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    window, n = assign_windows(times, valid, DAY)
    assert window.tolist() == [[0, 0, 0, 0, 1, 1, 2, -1, -1]]
    assert n.tolist() == [3]
    # A crop that starts mid-history lands every event in the same day bucket,
    # so the windows are a function of time, not of where the crop began.
    shifted, _ = assign_windows(times[:, 3:7], valid[:, 3:7], DAY)
    assert shifted.tolist() == [[0, 1, 1, 2]]
    hours, n4 = assign_windows(times, valid, 4 * 60)
    assert n4.tolist() == [4]
    assert hours[0, :3].tolist() == [0, 0, 0]


def test_admission_windows_count_from_the_admission_including_simultaneous_events() -> None:
    # Day 0 outpatient; admission at 20:00 on day 2, with a lab sorted *before*
    # the admission code at the same minute; the stay then crosses midnight.
    t_adm = 2 * DAY + 1200
    times = torch.tensor([[0, 5, t_adm, t_adm, t_adm + 60, t_adm + 300, t_adm + 1800]])
    admission = torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    valid = torch.ones_like(admission)
    window, n = assign_windows(times, valid, DAY, admission)
    assert window.tolist() == [[0, 0, 1, 1, 1, 1, 2]]
    assert n.tolist() == [3]
    # Calendar windows split the first admission day at midnight instead.
    calendar, _ = assign_windows(times, valid, DAY)
    assert calendar.tolist() == [[0, 0, 1, 1, 1, 2, 3]]


def test_mimic_action_split_by_prefix_pattern_and_codes_file(tmp_path: Path) -> None:
    codes = [
        "[PAD]",
        "NDC/00338004904",
        "MIMIC_IV_Drug/Heparin",
        "ICD10PCS/0BH17EZ",
        "ICD9Proc/96.04",
        "HCPCS/99223",
        "MIMIC_IV_ITEM/225158/Continuous Med",  # inputevents
        "MIMIC_IV_ITEM/225792",  # procedureevents (ventilation) -- only via the file
        "MIMIC_IV_ITEM/220045",  # chartevents heart rate
        "MIMIC_IV_LABITEM/50912",
        "MIMIC_IV_OMR/BMI (kg/m2)",  # two slashes, not inputevents
        "ICD10CM/I50.9",
        "MIMIC_IV_Admission/EW EMER.",
        "MIMIC_IV_Admission_Location/TRANSFER FROM HOSPITAL",
    ]
    flags = code_flags(codes, DEFAULT_ACTION_PREFIXES, DEFAULT_ACTION_PATTERNS)
    assert flags.tolist() == [False] + [True] * 6 + [False] * 7
    listed = tmp_path / "items.txt"
    listed.write_text("# comment\nMIMIC_IV_ITEM/225792\n\n")
    with_file = code_flags(codes, DEFAULT_ACTION_PREFIXES, DEFAULT_ACTION_PATTERNS, listed)
    assert with_file[7] and int(with_file.sum()) == 7
    admission = code_flags(codes, DEFAULT_ADMISSION_PREFIXES)
    assert admission.nonzero()[0].tolist() == [12]
    shipped = ITEMS.read_text().splitlines()
    assert "MIMIC_IV_ITEM/225792" in shipped  # invasive ventilation
    assert "MIMIC_IV_ITEM/220045" not in shipped  # heart rate stays an observation


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pool", ["last", "mean", "attn"])
def test_states_and_targets_at_t_do_not_see_tokens_after_window_t(pool: str) -> None:
    model = _model(state_pool=pool, target_mode="ema")
    batch = _batch()
    t = 2
    later = [p for w in range(t + 1, N_WINDOWS) for p in _positions(w, "all")]
    tampered = _tamper(_tamper(batch, later, action=False), _positions(t + 1, "act"), True)
    with torch.no_grad():
        before = model.build_pairs(batch)
        after = model.build_pairs(tampered)
    assert torch.equal(before.states[:, : t + 1], after.states[:, : t + 1])
    assert before.targets is not None and after.targets is not None
    assert torch.equal(before.targets[:, : t + 1], after.targets[:, : t + 1])
    # Control: the tampering did reach the later windows.
    assert not torch.equal(before.states[:, t + 1 :], after.states[:, t + 1 :])


@pytest.mark.parametrize("pool", ["last", "attn"])
def test_prediction_sees_actions_in_between_but_no_future_observation(pool: str) -> None:
    model = _model(state_pool=pool)
    batch = _batch()
    t, k = 1, 2
    base = _prediction(model, batch, t, k)
    # Every observation after window t, and every action at or after t + k.
    future_obs = [p for w in range(t + 1, N_WINDOWS) for p in _positions(w, "obs")]
    future_act = [p for w in range(t + k, N_WINDOWS) for p in _positions(w, "act")]
    tampered = _tamper(_tamper(batch, future_obs, action=False), future_act, action=True, seed=5)
    assert torch.equal(base, _prediction(model, tampered, t, k))
    # The actions of windows t .. t+k-1 are the conditioning input: they move it.
    between = [p for w in range(t, t + k) for p in _positions(w, "act")]
    moved = _prediction(model, _tamper(batch, between, action=True, seed=6), t, k)
    assert not torch.allclose(base, moved)


def test_action_blind_predictions_ignore_the_actions_in_between() -> None:
    model = _model(state_actions=False)
    batch = _batch()
    t, k = 1, 2
    base = _prediction(model, batch, t, k)
    # Only actions in windows t+1 .. t+k-1 (those in t also feed s_t through the
    # encoder, which the blind variant keeps): the predictor input is zeroed.
    between = [p for w in range(t + 1, t + k) for p in _positions(w, "act")]
    assert torch.equal(base, _prediction(model, _tamper(batch, between, action=True), t, k))


def test_the_last_window_of_a_cut_crop_is_never_a_target() -> None:
    model = _model()
    batch = _batch()
    with torch.no_grad():
        full = model(batch)
        batch["at_end"] = torch.zeros(2, dtype=torch.bool)
        cut = model(batch)
    assert int((full.target_index[1] == N_WINDOWS - 1).sum()) > 0
    assert int((cut.target_index[1] == N_WINDOWS - 1).sum()) == 0
    # Horizon arithmetic: with all six windows usable, k pairs per row = 6 - k.
    assert full.extras["horizon_sizes"].tolist() == [2 * 5, 2 * 4, 2 * 2]


# --------------------------------------------------------------------------- #
# EMA, loss
# --------------------------------------------------------------------------- #


def test_ema_targets_carry_no_gradient_and_the_attention_pool_follows_the_online_one() -> None:
    model = _model(state_pool="attn", target_mode="ema").train()
    assert model.target_pool is not None
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_pool.parameters())
    out = model(_batch())
    assert not out.targets.requires_grad
    assert out.predictions.requires_grad
    before = model.target_pool.query.detach().clone()
    with torch.no_grad():
        model.state_pool.query.add_(1.0)
    model.update_ema(0.75)
    expected = 0.75 * before + 0.25 * model.state_pool.query.detach()
    assert torch.allclose(model.target_pool.query, expected)


@pytest.mark.parametrize("actions", [True, False])
def test_state_loss_is_finite_with_ar_and_sigreg_and_backward_runs(actions: bool) -> None:
    model = _model(target_mode="ema", state_actions=actions).train()
    objective = LatentObjective(
        ObjectiveConfig(kind="state", lambda_recon=0.1, lambda_sigreg=0.05, sigreg_directions=16),
        recon_head=model.recon_head,
    )
    out = model(_batch())
    losses = objective(out)
    assert all(torch.isfinite(losses[k]) for k in ("loss", "pred_loss", "recon_loss", "ce"))
    assert float(losses["state_pairs"]) == 2 * (5 + 4 + 2)
    assert 0.35 < float(losses["action_frac"]) < 0.45
    assert torch.isfinite(losses["act_gain"]) == actions
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # AR term is the AR loss, verbatim (same head, same next-code targets).
    ar = ar_loss_chunked(model.recon_head, out.extras["recon_hidden"], out.extras["recon_code_id"])
    assert torch.allclose(losses["recon_loss"], ar["loss"].detach())


def test_a_crop_with_a_single_window_scores_no_pair_and_stays_finite() -> None:
    model = _model().train()
    batch = _batch()
    batch["time_min"] = torch.zeros_like(batch["time_min"]) + 7  # one window
    objective = LatentObjective(ObjectiveConfig(kind="state", lambda_recon=0.1), model.recon_head)
    losses = objective(model(batch))
    assert float(losses["state_pairs"]) == 0
    assert torch.isfinite(losses["loss"]) and float(losses["pred_loss"]) == 0.0
    losses["loss"].backward()


def test_sensitivity_diagnostic_is_zero_for_the_blind_model() -> None:
    batches = [_batch(4)]
    blind = action_sensitivity(_model(state_actions=False), batches)
    assert blind["pooled"]["rel_delta"] == 0.0 and blind["pooled"]["gain"] == 0.0
    assert blind["pooled"]["win_rate"] == 0.5
    seeing = action_sensitivity(_model(), batches)
    assert seeing["pooled"]["pairs"] == blind["pooled"]["pairs"] > 0


# --------------------------------------------------------------------------- #
# MLM
# --------------------------------------------------------------------------- #


def test_mlm_inputs_mask_at_the_rate_blank_values_and_cover_every_row() -> None:
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(4, VOCAB, (64, 200), generator=g)
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[0, 1:] = False  # a one-event row still gets its selection
    inputs, selected, blank = mlm_inputs(codes, valid, 0.15, VOCAB, generator=g)
    assert bool(selected[0, 0]) and int(selected[0].sum()) == 1
    rate = float(selected[1:].float().mean())
    assert 0.13 < rate < 0.17
    assert not bool((selected & ~valid).any())
    masked = inputs == MASK_ID
    assert 0.75 < float(masked[selected].float().mean()) < 0.85
    assert torch.equal(inputs[~selected], codes[~selected])
    assert bool((blank <= selected).all()) and bool(masked[selected].le(blank[selected]).all())


def test_mlm_is_bidirectional_and_scores_only_the_selected_codes() -> None:
    torch.manual_seed(0)
    model = EHRMLM(_config(causal=False, recon_head=False)).eval()
    batch = _batch()
    inputs = mlm_inputs(
        batch["code_id"], batch["attention_mask"], 0.3, VOCAB, torch.Generator().manual_seed(1)
    )
    out = model(batch, *inputs)
    assert torch.equal(out.targets, batch["code_id"][inputs[1]])
    tampered = _tamper(batch, [len(OFFSETS) * N_WINDOWS - 1], action=False)
    first = model(tampered, *inputs).tokens[:, 0]
    assert not torch.allclose(first, out.tokens[:, 0]), "position 0 sees the future"
    loss = ar_loss_chunked(model.head, out.hidden, out.targets)["loss"]
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="causal=false"):
        EHRMLM(_config(causal=True))


# --------------------------------------------------------------------------- #
# End to end on the demo cache
# --------------------------------------------------------------------------- #


def test_dataset_items_say_whether_they_reach_the_subjects_last_event() -> None:
    if not (DEMO_CACHE / "meta.json").exists():
        pytest.skip("mimic-demo cache is not built")
    full = EventSequenceDataset(DEMO_CACHE, "train", max_len=10**6, sampling="full")
    assert bool(full[0]["at_end"])
    # Half the subjects fit whole (always at the end), half are cropped.
    median = int(np.median(full._length))
    crops = EventSequenceDataset(DEMO_CACHE, "train", max_len=median, sampling="random_window")
    seen = set()
    for i in range(len(crops)):
        for _ in range(3):
            item = crops[i]
            end = int(crops._offset[i] + crops._length[i])
            assert bool(item["at_end"]) == (int(item["start"] + item["length"]) == end)
            seen.add(bool(item["at_end"]))
    assert seen == {True, False}
    assert "at_end" in collate_events([crops[0], crops[1]])


@requires_cache
@pytest.mark.parametrize("kind", ["state", "mlm"])
def test_trainer_runs_and_the_probe_reloads_the_checkpoint(tmp_path: Path, kind: str) -> None:
    overrides = [
        f"objective.kind={kind}",
        "run.steps=4",
        "run.log_every=2",
        f"run.out_dir={tmp_path / kind}",
        "data.max_len=128",
    ]
    if kind == "state":
        overrides += [
            "model.causal=true",
            "model.target_mode=ema",
            "objective.lambda_recon=0.1",
            "objective.state_pool=attn",
            "objective.state_window_hours=4",
            f"objective.action_codes_file={ITEMS}",
        ]
    else:
        overrides += ["model.causal=false"]
    trainer = Trainer(load_config(DEBUG_CONFIG, overrides))
    final = trainer.train()
    assert np.isfinite(final["loss"])
    header = (tmp_path / kind / "metrics.csv").read_text().splitlines()[0].split(",")
    assert header == list(LOG_COLUMNS)
    model, _ = probe.load_encoder(tmp_path / kind / "final.pt")
    if kind == "state":
        assert isinstance(model, EHRStateModel)
        assert torch.equal(model.action_table, trainer.model.action_table.cpu())
        assert int(model.action_table.sum()) > 100
        rows = (tmp_path / kind / "metrics.csv").read_text().splitlines()[1:]
        column = list(LOG_COLUMNS).index("state_pairs")
        assert all(row.split(",")[column] != "" for row in rows)
    else:
        assert isinstance(model, EHRMLM) and not model.config.causal
    tuned, _ = build_model(tmp_path / kind / "final.pt", features="last")
    batch = collate_events([trainer.dataset[0], trainer.dataset[1]])
    assert tuned(batch).shape == (2,)


@requires_cache
def test_window_stats_report_the_action_share_on_the_demo_cache() -> None:
    stats = window_stats(DEMO_CACHE, window_hours=[24.0], samples=50, codes_file=ITEMS)
    assert 0.05 < stats["action_token_share"] < 0.10
    assert stats["windows"]["24h"]["windows_per_crop_mean"] >= 1.0


def test_the_gate_grid_plans_every_cell_at_one_budget() -> None:
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import ablate

    grid = ablate.load_grid(REPO / "configs" / "grids" / "mimic_state_gate.yaml")
    assert grid.eval_modes == ("probe", "finetune") and grid.source == "mimic-iv"
    entries = {e["run"]: e for e in ablate.plan(grid)}
    assert list(entries) == [
        "hybrid_s0",
        "ar_s0",
        "state_ac_s0",
        "state_blind_s0",
        "mlm_s0",
        "state_ac_s1",
        "state_blind_s1",
        "mlm_s1",
        "hybrid_s1",
        "ar_s1",
    ]
    assert entries["hybrid_s0"]["reuse"] and entries["ar_s0"]["reuse"]
    trained = [e for e in entries.values() if not e["reuse"]]
    assert {e["tokens"] for e in trained} == {30_518 * 64 * 512}
    kinds = {name: e["objective"] for name, e in entries.items() if not e["reuse"]}
    assert kinds["state_ac_s0"] == kinds["state_blind_s1"] == "state"
    assert kinds["mlm_s0"] == "mlm" and kinds["hybrid_s1"] == "nextlatent"
    blind = load_config(REPO / grid.base, grid.runs[3].override_strings())
    seeing = load_config(REPO / grid.base, grid.runs[2].override_strings())
    assert blind.objective.state_actions is False and seeing.objective.state_actions is True
    # The pair differs in that one switch and nothing else.
    a, b = blind.to_dict(), seeing.to_dict()
    a["objective"].pop("state_actions"), b["objective"].pop("state_actions")
    assert a == b
    assert Path(REPO / seeing.objective.action_codes_file).exists()
