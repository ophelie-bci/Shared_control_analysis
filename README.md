# Shared Control Analysis

This repository contains the analysis scripts, regenerated figure panels, and
summary tables for the manuscript:

**Stabilization-Responsiveness Trade-offs in Continuous Shared-Control for
Invasive Brain-Computer Interfaces**

The raw dataset is distributed separately through the dataset DOI:
[https://doi.org/10.48804/7KGSQS](https://doi.org/10.48804/7KGSQS).

## Repository Contents

```text
.
├── config.yaml          # Local path to the downloaded dataset
├── run_all.py           # Driver script for the implemented analyses
├── requirements.txt     # Python dependencies used for regeneration
├── scripts/             # Figure/table generation scripts
├── src/                 # Shared analysis helpers
├── Variables/           # Cached summary object used for failure-mode plots
├── figures/             # Regenerated manuscript figure panels
└── tables/              # Regenerated manuscript tables
```

The included `figures/` and `tables/` folders contain only the manuscript
artifacts selected for this public release.

## Data Setup

Download and extract the dataset from the DOI above. Then edit `config.yaml` so
that `base_dir` points to the folder containing the released `Monkey 1/` and
`Monkey 2/` directories.

Example:

```yaml
base_dir: "D:/path/to/AI_paper_data"
```

The committed `config.yaml` intentionally leaves this value blank so each user
can fill in their local dataset path.

## Environment Setup

Create and activate a Python environment, then install the requirements:

```bash
pip install -r requirements.txt
```

The scripts were prepared with Python scientific packages listed in
`requirements.txt`. They use only files inside this repository plus the dataset
folder supplied through `config.yaml`.

## Rebuilding Artifacts

After setting `base_dir`, run:

```bash
python run_all.py
```

You can also pass the dataset path directly:

```bash
python run_all.py --base-dir /path/to/AI_paper_data
```

By default, `run_all.py` rebuilds all implemented figure and table outputs and
then organizes them into the manuscript folders.

## Included Outputs

### Figures

- `figures/Figure 1/` - success-rate panels
- `figures/Figure 2/` - fixed-obstacle example trial
- `figures/Figure 3/` - AI-gain panels and performance-dependent AI gains
- `figures/Figure 4/` - failure modes and success by decoded choice state
- `figures/Figure 5/` - prior-confidence and temporal-prior-reset panels
- `figures/Extended Data Figure 1/` - trajectory supplement
- `figures/Extended Data Figure 2/` - obstacle prior-confidence panels

### Tables

- `tables/Extended Data Table 1/extended_data_table_1_success_rate_by_task_excel_semicolon.csv`
- `tables/Extended Data Table 2/extended_data_table_2_additive_multiplicative_aic_excel_semicolon.csv`

The table files are semicolon-separated CSV files for Excel compatibility.

## Notes

Failure-mode figure panels are regenerated from the cached
`Variables/grand_summary.pkl` by default. The raw-data recomputation path is
guarded because placeholder bodies remain in that script; using
`--recompute-failure-modes` will fail loudly until that path is restored.

For a script-by-script artifact map, see `ARTIFACTS.md`.
