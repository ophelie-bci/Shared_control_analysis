# Shared Control Analysis

This repository contains the public analysis code, selected regenerated figure
panels, and summary tables for the manuscript:

**Stabilization-Responsiveness Trade-offs in Continuous Shared-Control for
Invasive Brain-Computer Interfaces**

## Data Not Included

Raw trial data are distributed separately through the dataset DOI:
[https://doi.org/10.48804/7KGSQS](https://doi.org/10.48804/7KGSQS).

Download and extract the dataset, then set `base_dir` in `config.yaml` or pass
`--base-dir` to `run_all.py`. The dataset root should contain the released
`Monkey 1/` and `Monkey 2/` folders. Some reviewer follow-up analyses also use
the manuscript-source monkey mapping stored in `config.yaml`.

Example:

```yaml
base_dir: ""
```

For local work, either edit `config.yaml` or keep machine-specific paths in an
untracked `config.local.yaml`.

## Repository Contents

```text
.
|-- config.yaml          # Public configuration template
|-- run_all.py           # Driver for implemented analyses
|-- requirements.txt     # Tested Python package versions
|-- ARTIFACTS.md         # Detailed script-to-output map
|-- scripts/             # Figure/table generation and audit scripts
|-- src/                 # Shared analysis helpers
|-- Variables/           # Cached derived summary for failure-mode plots
|-- figures/             # Selected manuscript/reviewer figure outputs
`-- tables/              # Selected summary tables and reports
```

`analysis_results/`, `tables/raw_chunks/`, Python bytecode, raw neural files,
and local logs are internal/generated artifacts and are intentionally ignored.

## Environment Setup

The analyses were tested with Python 3.9.13. Create and activate an environment,
then install the pinned requirements:

```bash
pip install -r requirements.txt
```

The package list is intentionally conservative and was not upgraded during
public-release cleanup.

## Rebuilding Artifacts

After setting `base_dir`, run:

```bash
python run_all.py
```

or pass the dataset location explicitly:

```bash
python run_all.py --base-dir /path/to/AI_paper_data
```

`run_all.py` rebuilds the implemented manuscript and reviewer artifacts and
organizes selected outputs into `figures/` and `tables/`. Use the `--skip-*`
options shown by `python run_all.py --help` to run subsets.

## Figure And Analysis Map

| Manuscript item | Main script(s) | Main output location |
| --- | --- | --- |
| Figure 1: task success rates | `scripts/run_success_rate.py` | `figures/Figure 1/Success rate/`, `tables/success_rate_*.csv` |
| Figure 2: fixed-obstacle example | `scripts/make_figure2_fixed_obstacle_example.py` | `figures/Figure 2/` |
| Figure 3: task success-rate summary | `scripts/make_figure3_success_rates.py`, `scripts/redraw_figure3_from_cached_tables.py` | `figures/Figure 3/`, `tables/Figure 3C original session audit/`, `tables/Figure 3C stratified/` |
| Figure 4: baseline-performance/gain analysis | `scripts/run_success_rate.py`, `scripts/run_gain_curve_pooling_audit.py` | `figures/Figure 4/AI gain with baseline/`, `tables/Gain curve pooling audit/` |
| Figure 5: failure modes and decoded choice state | `scripts/run_failure_mode_analysis.py` | `figures/Figure 5/Failure modes/`, `figures/Figure 5/Success by choice state/` |
| Figure 6: prior-confidence and Respawn analyses | `scripts/run_prior_confidence_index_analysis.py`, `scripts/run_respawn_reorientation_lag.py`, `scripts/make_temporal_prior_reset_example.py` | `figures/Figure 6/`, `tables/respawn_*` |
| Extended Data Figure 1: trajectory supplement | `scripts/make_extended_data_figure1_trajectory_supplement.py` | `figures/Extended Data Figure 1/` |
| Extended Data Figure 2: monkey/task gain-curve extension | `scripts/make_extended_data_figure5_gain_curve.py` | `figures/Extended Data Figure 2/Gain curve extension/`, `tables/Extended Data Table 5/` |
| Extended Data Figure 3 and reviewer controls | `scripts/run_obstacle_session_metrics.py`, `scripts/run_same_alpha_assistance_analysis.py`, `scripts/run_time_resolved_bci_influence.py`, `scripts/run_decoded_command_condition_comparison.py`, `scripts/run_neural_target_cross_condition_generalization.py`, `scripts/run_shared_control_adaptation_analysis.py` | `figures/Extended Data Figure 3/`, corresponding `tables/` subfolders |
| Hidden-prior six-month follow-up | `scripts/run_hidden_prior_followup_gain.py`, `scripts/run_bci_only_learning_followup.py`, `scripts/run_followup_reactive_obstacle_analysis.py` | `figures/Hidden prior follow-up/`, `figures/Follow-up reactive obstacle/`, matching `tables/` folders |
| Strict AI-only baseline / candidate-set sweep | `scripts/run_ai_only_baseline.py` | `tables/AI-only baseline/`; optional figure if replay rows are supplied |

For a more detailed script-by-script output map, see `ARTIFACTS.md`.

## Outputs And Intermediates

Manuscript-facing outputs are the selected files under `figures/` and the small
summary/report tables under `tables/`. Large sample-level CSVs are generated
intermediates and are not intended for GitHub; they can be regenerated from the
dataset by their corresponding scripts.

Audit and verification scripts are retained in `scripts/` because they document
reviewer-driven checks, but their generated outputs belong in ignored
`analysis_results/`.

## Cached Derived Input

Failure-mode panels are regenerated from `Variables/grand_summary.pkl` by
default. This pickle is a derived summary cache used by
`scripts/run_failure_mode_analysis.py`. The raw-data recomputation path is
guarded because legacy placeholder bodies remain in that script; using
`--recompute-failure-modes` delegates to the verified raw classifier in
`scripts/verify_figure5_failure_modes.py`. Do not remove
`Variables/grand_summary.pkl` unless the raw recomputation path has been
validated end to end against the public dataset.

The strict AI-only baseline requires separate autonomous-controller replay rows.
If no replay CSV is supplied, the code writes an analysis note and replay-row
template rather than fabricating an AI-only result.
