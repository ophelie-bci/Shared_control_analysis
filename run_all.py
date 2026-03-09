"""
run_all.py

Run all core analysis scripts for the AI-assisted intracortical BCI navigation study.

This pipeline sequentially executes:

- run_success_rate.py
    Session-level success-rate analysis for AI ON vs AI OFF conditions

- run_failure_mode_analysis.py
    Trajectory, failure-mode, and ambiguity analysis across experiments

- run_prior_confidence_index_analysis.py
    Prior-confidence / entropy-based analyses aligned to task events

- run_reset_prior_analysis.py
    Reset-prior session summaries and success-rate extraction from reset-prior files

Before running, make sure:
- config.yaml contains the correct base_dir
- each script's default monkey / experiment settings are appropriate
- required input files are present in the expected data folders
"""

from scripts import (
    run_success_rate,
    run_failure_mode_analysis,
    run_prior_confidence_index_analysis,
    run_reset_prior_analysis,
)

run_success_rate.main()
run_failure_mode_analysis.main()
run_prior_confidence_index_analysis.main()
run_reset_prior_analysis.main()