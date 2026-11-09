# Next experiments

Updated 2026-11-06. This supersedes the October 30 gate-first plan (preserved in
git at `459b31e`). The owner wants iterative research aimed at a useful, novel,
citable contribution. Follow [AUTORESEARCH.md](AUTORESEARCH.md); the original
late-January 2027 paper target remains a planning target, not a reason to relax
the validity checks.

## Completed locally

- [Diagnosis timing audit](experiments/diagnosis-timing/README.md): demo reproduces
  27/33 positive labels at the current stay's discharge. Full MIMIC unverified.
- [State integrity cycles](experiments/state-target-integrity/README.md): action
  content affects legacy targets; observation-only targets are invariant in the
  paired controls. Occupied-window offsets misstate elapsed calendar horizons;
  exact-calendar lookup fixes this without allocating empty periods.
- Both changes are explicit config ablations. Legacy checkpoint defaults remain
  intact, and resuming under different target/horizon semantics is rejected.
- These are local structural results, not trained performance or causal results.

## Ordered work

| Priority | Experiment/action | Exit criterion | Current dependency |
| --- | --- | --- | --- |
| 1 | Audit PC's actual `mimic-iv` paper label cohort, then `mimic-iv-full` separately | Aggregate all/split timing counts, zero input/label inconsistencies, explicit cohort identities | SSH `gpu` currently times out; CPU only |
| 2 | Audit saved state checkpoint with `ehrjepa.eval.state_integrity` on training crops | Quantify target dependence after training; distinguish direct content from constant-target controls | PC checkpoint access; CPU feasible but runtime unmeasured |
| 3 | Version corrected downstream task definitions | Clinical decision point, index-stay exclusions, observation/censoring rules and cohort accounting tested; original labels preserved | Full-data audit and task design review |
| 4 | Fix prediction-time state/action semantics and timestamp handling | No action-plan echo in online pre-action state; no post-baseline action/count/time shortcuts; invariance tests and meaningful observation targets | Local engineering plus raw-data verification |
| 5 | Freeze a small controlled training experiment | Target-view × explicit-action-input arms on identical calendar pairs; matched weights, seeds, compute accounting; specified observation-forecast and downstream outcomes | 1–4; owner releases GPU |
| 6 | Run short training pilot, inspect tuning results, keep/revise/reject | Finite losses, no collapse, enough exact-time pairs; effect distinguished from persistence/action-only controls | 5; measured cap before launch |
| 7 | Confirm on frozen held-out protocol | Three pretraining seeds for headline learned results, subject/seed uncertainty, all arms reported, tuned tabular and released-model comparators | 6 and frozen valid tasks |
| 8 | Test portability of the finding and release artifact | At least one prior implementation or second site; if effect is only our bug, drop general methodological claims | 7; access/licences as applicable |

The exact PC audit commands are in the diagnosis audit README. Deploy only the
required files and preserve PC-local ETL changes. Copy back aggregates, never
credentialed source records.

## Constraints and parallel owner actions

- The old `mimic_state_gate.yaml` is a historical experiment, not a corrected
  gate. Do not relaunch it under new semantics or recycle its run directories.
  GPU remains assigned to the other paper until the owner explicitly releases it.
- When a new GPU run is authorized, verify availability and keep-alive state;
  use the established `MemoryHigh=16G` cap. No reboot/WSL shutdown as part of an
  experiment. Do not overwrite a running service's script.
- Owner handles EHRSHOT DUA/INSPECT access, outreach follow-ups, and submission.
  Emails were already sent on October 27; do not resend them.
- Deconfound hybrid vs AR (SIGReg, code-loss weight, compute); expand full-held-out
  scoring only with validated labels. Larger models and a trial benchmark remain
  lower-priority hypotheses until these checks succeed.
- Keep the paper's result tables explicitly provisional. Do not claim first
  hybrid/action-conditioned EHR model, proven causal treatment effects, or a
  generic first leakage benchmark. Novelty is rechecked before each major pivot.

## Resume

Use the task-specific `.parallel-code/checkpoints/.../CURRENT.md` for live state,
then the [experiment ledger](experiments/autoresearch-ledger.jsonl). The shared
`~/claude-sync/projects/EHRJEPA/HANDOFF.md` is supporting context. Historical
training estimates and old plans are not launch instructions.
