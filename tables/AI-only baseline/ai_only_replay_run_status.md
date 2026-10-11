# AI-only replay run status

Date: 2026-08-28

## Requested analysis

The requested strict baseline is:

- AI-only, K=5: five candidate targets available, true target hidden, no neural command/history.
- Candidate-set sweep: AI-only, K=1..5, with K=1 interpreted as oracle/goal-revealed autonomy.

## What was run

The GitHub AI paper analysis runner was executed in no-replay-input mode:

```powershell
python "navbcidecode\bcidecode\scripts\Github AI paper\scripts\run_ai_only_baseline.py"
```

This successfully generated the replay-row specification/template files used by the paper pipeline.

## Current result status

No numeric AI-only replay result has been produced yet from the available files.

The available `_aitrials.pkl` / Unity log data contain recorded online shared-control streams. Their `aiVelocities` records include nonzero `Input` values, so using the recorded `Output` values as "AI-only" would retain the actual neural command/history and would not answer the reviewer question.

The replay scripts found in this checkout replay neural decoder outputs and spike perturbations. They do not expose an RT-V2 controller entry point that can rerun each trial with neural input removed and candidate set size K varied.

## Files generated

- `ai_only_baseline_analysis_note.md`: design note for the requested analysis.
- `ai_only_candidate_replay_rows_template.csv`: expected schema for true RT-V2 AI-only replay rows.

## Needed to compute the numeric result

Provide or locate the RT-V2 controller/replay entry point that can accept:

- trial geometry and candidate targets,
- `K=1..5` candidate-set configuration,
- hidden true target for K=5,
- zero/absent neural command input,
- trial outcome scoring against the reached target.

Once those replay rows exist, run:

```powershell
python "navbcidecode\bcidecode\scripts\Github AI paper\scripts\run_ai_only_baseline.py" --replay-csv <path-to-replay-rows.csv>
```

Expected numeric outputs are:

- `ai_only_candidate_replay_rows_normalized.csv`
- `ai_only_baseline_summary.csv`
- `ai_only_reached_target_distribution.csv`
- `ai_only_k5_critical_comparison.csv`
- `ai_only_candidate_set_success.svg`
