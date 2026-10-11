"""Redraw Figure 3 A-C.

Panels A/B use the original Figure 3 success-rate data path. Panel C uses the
cached reviewer target-session table for the displayed target averages and the
full target-session model curve.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load import load_files
from src.trial_utils import categorize_trials_by_target_and_ai
from src.success_rate_metrics import calculate_accuracy_per_target
from src.figure3_plots import plot_gain_with_labels, plot_summary_scatter


TABLE = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_target_session_rows.csv"


def _mean_ignore_nan(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return np.nan if values.size == 0 else float(values.mean())


def _load_original_panel_rows(base_dir: str, monkey: str, experiment: str) -> tuple[dict, dict]:
    all_trials, correct_trials_all, incorrect_trials_all, *_ = load_files(
        experiment,
        monkey,
        base_dir,
        nb_trials=None,
    )
    if all_trials == []:
        raise RuntimeError(f"No loaded trials for {monkey} / {experiment}.")
    per_target_gains = {}
    cumulative_correct_ai_off = {}
    cumulative_incorrect_ai_off = {}

    for correct_trials, incorrect_trials in zip(correct_trials_all, incorrect_trials_all):
        correct_ai_on = categorize_trials_by_target_and_ai(experiment, correct_trials, ai_condition=1.0)
        correct_ai_off = categorize_trials_by_target_and_ai(experiment, correct_trials, ai_condition=0.0)
        incorrect_ai_on = categorize_trials_by_target_and_ai(experiment, incorrect_trials, ai_condition=1.0)
        incorrect_ai_off = categorize_trials_by_target_and_ai(experiment, incorrect_trials, ai_condition=0.0)

        acc_ai_on, _, _ = calculate_accuracy_per_target(correct_ai_on, incorrect_ai_on)
        acc_ai_off, _, _ = calculate_accuracy_per_target(correct_ai_off, incorrect_ai_off)
        for target in set(acc_ai_on) | set(acc_ai_off):
            per_target_gains.setdefault(target, []).append(acc_ai_on.get(target, np.nan) - acc_ai_off.get(target, np.nan))

        for source, destination in [
            (correct_ai_off, cumulative_correct_ai_off),
            (incorrect_ai_off, cumulative_incorrect_ai_off),
        ]:
            for target, target_trials in source.items():
                destination[target] = destination.get(target, []) + target_trials

    bci_only_by_target, _, _ = calculate_accuracy_per_target(
        cumulative_correct_ai_off,
        cumulative_incorrect_ai_off,
    )
    return per_target_gains, bci_only_by_target


def _plot_gain_panel(base_dir: str, monkey: str, experiment: str) -> None:
    per_target_gains, bci_only_by_target = _load_original_panel_rows(base_dir, monkey, experiment)
    plot_gain_with_labels(per_target_gains, bci_only_by_target, monkey, experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw Figure 3/4 shared-control gain panels.")
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Figure output directory under figures/. Defaults to AI_PAPER_FIGURE_DIR or Figure 3.",
    )
    args = parser.parse_args()
    if args.figure_dir:
        os.environ["AI_PAPER_FIGURE_DIR"] = args.figure_dir
    os.chdir(PROJECT_ROOT)
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        base_dir = yaml.safe_load(f)["base_dir"]
    _plot_gain_panel(base_dir, "Monkey 1", "AI Appearing Obstacle 2")
    _plot_gain_panel(base_dir, "Monkey 2", "AI Appearing Obstacle")
    plot_summary_scatter([], str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
