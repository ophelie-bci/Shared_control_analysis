import os
import sys
import argparse
import numpy as np
import pandas as pd

# Allow imports from the project root when this script is run from /scripts.
# This makes modules in /src importable without installing the package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plots import (
    plot_alpha_vs_time_from_obstacle_appearance,
    plot_individual_trials,
)


def run_prior_confidence_analysis(monkeys, experiments, base_dir):
    """
    Run prior-confidence analyses for a set of monkeys and experiments.

    Parameters
    ----------
    monkeys : list of str
        Monkey identifiers (for example ["Monkey 1", "Monkey 2"]).
    experiments : list of str
        Experiment names to analyze.
    base_dir : str
        Root directory containing the experiment data.

    Notes
    -----
    This function currently performs two main actions for each monkey × experiment:
    1. Computes and plots prior-confidence dynamics relative to task events.
    2. Plots example individual trials for qualitative visualization.
    """
    for experiment in experiments:
        print(f"\n[INFO] Processing experiment: {experiment}")

        for monkey in monkeys:
            print(f"\n--- Analyzing {monkey} ---")

            # -----------------------------------------------------------------
            # Prior-confidence analysis aligned to the task event
            # -----------------------------------------------------------------
            # This function computes the prior-confidence index (alpha) from
            # entropy-related signals and plots it as a function of time from
            # obstacle appearance, target jump, or trial start depending on task.
            #
            # Key options used here:
            # - stride=4: keep every 4th sample to reduce density/noise
            # - t_window: analysis window around the event in milliseconds
            # - only_answer=1: restrict analysis to successful trials only
            # - show_per_trial=False: show only population summary, not each trial
            result = plot_alpha_vs_time_from_obstacle_appearance(
                monkey,
                experiment,
                base_dir,
                file_path=None,              # If None, analyze all matching files for this condition
                stride=4,                    # Keep every k-th sample
                min_dt=None,                 # Optional temporal thinning threshold
                min_dist=None,               # Optional spatial thinning threshold
                t_window=(-2000.0, 3000.0),  # Analysis window relative to event [ms]
                nbins=24,                    # Number of common bins across trials
                min_alpha=0.2,               # Lower bound after normalization
                only_answer=1,               # 1: only correct trials; None: all trials
                task=None,                   # Use experiment name unless manually overridden
                show_per_trial=False,        # Do not overlay individual trial traces
            )

            # -----------------------------------------------------------------
            # Example single-trial plots
            # -----------------------------------------------------------------
            # This is mainly for qualitative visualization of shared-control
            # behavior on individual trials.
            plot_individual_trials(monkey, experiment, base_dir)

            print("[INFO] Done.")


def main():
    """
    Parse command-line arguments and run the prior-confidence analysis pipeline.

    If --base_dir is not provided, the script loads it from config.yaml located
    in the project root.
    """
    parser = argparse.ArgumentParser(
        description="Run prior-confidence analysis for AI-assisted BCI navigation experiments."
    )

    parser.add_argument(
        "--monkeys",
        nargs="+",
        default=["Monkey 1", "Monkey 2"],
        help="List of monkeys to analyze.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "AI Obstacle",
            "AI Appearing Obstacle",
            "AI Appearing Obstacle 2",
            "AI Respawn",   
        ],
        help="List of experiment names to analyze.",
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to the root data directory. If omitted, load from config.yaml.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load base_dir from config.yaml if not provided on the command line
    # -------------------------------------------------------------------------
    if args.base_dir is None:
        import yaml
        import pathlib

        config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        args.base_dir = config.get("base_dir")

        if args.base_dir is None:
            raise ValueError("base_dir must be specified either via --base_dir or in config.yaml")

    run_prior_confidence_analysis(args.monkeys, args.experiments, args.base_dir)


if __name__ == "__main__":
    main()
