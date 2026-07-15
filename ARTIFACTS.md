# Manuscript Artifact Map

This folder is intended to regenerate the analysis artifacts for the AI-assisted
BCI navigation manuscript.

Run everything that is currently implemented with:

```powershell
venv/Scripts/python.exe "navbcidecode/bcidecode/scripts/Github AI paper/run_all.py"
```

Install dependencies with `pip install -r requirements.txt` first. Before
running, edit `config.yaml` and set `base_dir` to the local root folder that
contains the released `Monkey 1/` and `Monkey 2/` data directories. The driver
reads `config.yaml` for `base_dir` unless `--base-dir` is supplied.

## Implemented Outputs

| Artifact group | Script/function | Outputs |
| --- | --- | --- |
| Figure 2 Fixed Obstacle example trial | `scripts/make_figure2_fixed_obstacle_example.py` | `figures/Figure 2/figure2_fixed_obstacle_Monkey2_navdecodingsphereaiobstacle_Loki_20241223_1106_E_trial004.*` |
| Figure 5 temporal prior reset example | `scripts/make_temporal_prior_reset_example.py` | `figures/Figure 5/Temporal prior reset/temporal_prior_reset_example_*.png/.svg/.pdf`, plus provenance text |
| Extended Data Figure 1 trajectory supplement | `scripts/make_extended_data_figure1_trajectory_supplement.py` | `figures/Extended Data Figure 1/extended_data_figure1_trajectory_supplement.svg` with the metric table embedded, plus provenance text |
| AI ON vs AI OFF success rate and gain | `scripts/run_success_rate.py` | Success-rate SVGs are staged under `figures/Figure 1/Success rate/`; the three selected AI-gain SVGs are staged under `figures/Figure 3/AI gain with baseline/`; tables go to `tables/success_rate_session_summary.csv` and `tables/success_rate_target_gain_summary.csv` |
| Prior-confidence / entropy around task events | `scripts/run_prior_confidence_index_analysis.py` | Appearing-obstacle and Monkey 1 respawn SVGs are staged under `figures/Figure 5/Prior confidence index/`; obstacle SVGs are staged under `figures/Extended Data Figure 2/Prior confidence index/` |
| Failure modes and success by decoded choice state | `scripts/run_failure_mode_analysis.py` | Regenerated from `Variables/grand_summary.pkl` into failure-mode and choice-state figure folders |
| Reset-prior session summary | `scripts/run_reset_prior_analysis.py` | Console summary only |
| Extended Data tables | `scripts/make_extended_data_tables.py` | `tables/Extended Data Table 1/extended_data_table_1_success_rate_by_task_excel_semicolon.csv`; `tables/Extended Data Table 2/extended_data_table_2_additive_multiplicative_aic_excel_semicolon.csv` |
| Figure rebuild status | manual audit table | `tables/figure_rebuild_status.csv` |

## Known Gap

`scripts/run_failure_mode_analysis.py` currently contains placeholder bodies in
the raw-data recomputation path. The safe default is therefore to regenerate the
manuscript failure-mode figures from the cached `Variables/grand_summary.pkl`.

Attempting:

```powershell
venv/Scripts/python.exe "navbcidecode/bcidecode/scripts/Github AI paper/run_all.py" --recompute-failure-modes
```

will fail loudly until those placeholder bodies are restored. This prevents a
valid cached manuscript summary from being overwritten by an empty recomputation.
