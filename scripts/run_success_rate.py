import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle

# Allow imports from the project root when this script is run from /scripts.
# This makes modules in /src importable without installing the package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load import load_files
from src.trial_utils import (
    categorize_trials_by_target_and_ai,
    compute_target_stats,
    collect_summary_points,
)
from src.success_rate_metrics import (
    calculate_accuracy_per_target,
    compute_success_rate_stats,
)
from src.plots import (
    plot_success_rate_per_session,
    plot_gain_with_labels,
    plot_summary_scatter,
)
from src.stats_tests import compute_chance_level, paired_success_test
from src.tables import SuccessRateSummary
from src.AI_metrics import compute_additive_vs_multiplicative_effect

# Global summary table object used across all monkey × experiment analyses.
summary_table = SuccessRateSummary()


def run_success_rate(monkeys, experiments, base_dir):
    """
    Run success-rate analysis for AI ON vs AI OFF conditions.

    For each monkey and experiment, this function:
    1. Loads trial data across sessions.
    2. Splits trials by correctness and AI condition.
    3. Computes per-session overall success rates.
    4. Computes per-target accuracies and AI gains.
    5. Runs paired statistical tests across sessions.
    6. Generates summary plots.

    Parameters
    ----------
    monkeys : list of str
        Monkey identifiers to analyze.
    experiments : list of str
        Experiment names to analyze.
    base_dir : str
        Root directory containing the dataset.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from collections import defaultdict

    # Store pooled summary points across all monkey × experiment pairs.
    # These are later used for the cross-condition summary scatter plot.
    all_summary_points = []

    # Store overall session-level summaries for optional downstream use.
    summary_success_rates = []

    # ---------------------------------------------------------------------
    # Main analysis loop over experiment × monkey
    # ---------------------------------------------------------------------
    for experiment in experiments:
        print(f"\n[INFO] Processing experiment: {experiment}")

        for monkey in monkeys:
            print(f"\n--- Analyzing {monkey} ---")

            # Load all sessions for this monkey and experiment.
            # The loader returns:
            # - all_trials
            # - correct_trials_all
            # - incorrect_trials_all
            # - and several additional outputs not used here
            try:
                all_trials, correct_trials_all, incorrect_trials_all, _, _, _, _, _ = load_files(
                    experiment, monkey, base_dir, nb_trials=None
                )
            except FileNotFoundError as e:
                print(f"[ERROR] Data not found for {monkey} in {experiment}: {e}")
                continue

            # Skip conditions with no data.
            if all_trials == []:
                continue

            # Per-session overall success rates for AI ON and AI OFF.
            success_rate_ai_on_all = []
            success_rate_ai_off_all = []

            # Cumulative pooled trial collections across sessions.
            # These are used to compute pooled accuracies across all sessions.
            cumulative_correct_ai_on = {}
            cumulative_incorrect_ai_on = {}
            cumulative_correct_ai_off = {}
            cumulative_incorrect_ai_off = {}

            # Store per-session AI gain per target.
            # Format: {target: [gain_session_1, gain_session_2, ...]}
            per_target_gains = defaultdict(list)

            # -----------------------------------------------------------------
            # Session loop
            # -----------------------------------------------------------------
            for session_idx, (correct_trials, incorrect_trials) in enumerate(
                zip(correct_trials_all, incorrect_trials_all)
            ):
                print(f"[Session {session_idx}]")
                print(f"All trials: {len(correct_trials) + len(incorrect_trials)}")

                # Split correct and incorrect trials by AI condition.
                correct_ai_on = categorize_trials_by_target_and_ai(
                    experiment, correct_trials, ai_condition=1.0
                )
                correct_ai_off = categorize_trials_by_target_and_ai(
                    experiment, correct_trials, ai_condition=0.0
                )
                incorrect_ai_on = categorize_trials_by_target_and_ai(
                    experiment, incorrect_trials, ai_condition=1.0
                )
                incorrect_ai_off = categorize_trials_by_target_and_ai(
                    experiment, incorrect_trials, ai_condition=0.0
                )

                # Simple consistency print: total number of trials across all
                # target × correctness × AI-condition groupings.
                total = sum(
                    len(v)
                    for d in [correct_ai_on, correct_ai_off, incorrect_ai_on, incorrect_ai_off]
                    for v in d.values()
                )
                print(f"Total number of values: {total}")

                # Compute per-target accuracies and overall session success rate
                # separately for AI ON and AI OFF.
                acc_ai_on, _, overall_ai_on = calculate_accuracy_per_target(
                    correct_ai_on, incorrect_ai_on
                )
                acc_ai_off, _, overall_ai_off = calculate_accuracy_per_target(
                    correct_ai_off, incorrect_ai_off
                )

                # -------------------------------------------------------------
                # Accumulate pooled trials across sessions
                # -------------------------------------------------------------
                # This preserves the original behavior where pooled accuracies are
                # computed from combined trial counts rather than averages of session means.
                def accumulate_trials(source, destination):
                    for target, trials in source.items():
                        destination[target] = destination.get(target, []) + trials

                accumulate_trials(correct_ai_on, cumulative_correct_ai_on)
                accumulate_trials(incorrect_ai_on, cumulative_incorrect_ai_on)
                accumulate_trials(correct_ai_off, cumulative_correct_ai_off)
                accumulate_trials(incorrect_ai_off, cumulative_incorrect_ai_off)

                # Save per-session overall success rates.
                success_rate_ai_on_all.append(overall_ai_on)
                success_rate_ai_off_all.append(overall_ai_off)

                # -------------------------------------------------------------
                # Per-session AI gain per target
                # -------------------------------------------------------------
                # Gain is defined as:
                #   success_rate(AI ON) - success_rate(AI OFF)
                # computed separately for each target in this session.
                all_tgts = set(acc_ai_on) | set(acc_ai_off)
                for tgt in all_tgts:
                    g = acc_ai_on.get(tgt, np.nan) - acc_ai_off.get(tgt, np.nan)
                    per_target_gains[tgt].append(g)

                # Optional target-level printouts for quick inspection.
                stats_on = compute_target_stats(correct_ai_on, incorrect_ai_on)
                stats_off = compute_target_stats(correct_ai_off, incorrect_ai_off)
                print("AI ON:", stats_on)
                print("AI OFF:", stats_off)

            # -----------------------------------------------------------------
            # Pooled target-level accuracies across all sessions
            # -----------------------------------------------------------------
            acc_ai_on, _, _ = calculate_accuracy_per_target(
                cumulative_correct_ai_on, cumulative_incorrect_ai_on
            )
            acc_ai_off, _, _ = calculate_accuracy_per_target(
                cumulative_correct_ai_off, cumulative_incorrect_ai_off
            )

            # Quantify whether the AI effect behaves more like an additive or
            # multiplicative improvement across sessions.
            compute_additive_vs_multiplicative_effect(
                success_rate_ai_on_all,
                success_rate_ai_off_all
            )

            # Collect pooled/summary points for the across-condition scatter plot.
            collect_summary_points(
                summary_list=all_summary_points,
                monkey=monkey,
                experiment=experiment,
                per_target_gains=per_target_gains,  # sessionwise gains per target
                acc_ai_off=acc_ai_off               # pooled baseline per target
            )

            # -----------------------------------------------------------------
            # Plot target-level AI gain with 95% CI across sessions
            # -----------------------------------------------------------------
            if len(per_target_gains) > 0:
                plot_gain_with_labels(
                    per_target_gains=per_target_gains,  # dict[target] -> list of gains in [0, 1]
                    acc_ai_off=acc_ai_off,             # dict[target] -> baseline success rate
                    monkey=monkey,
                    experiment=experiment,
                    label_fmt="{:.0f}%",               # label as percentage
                    label_pos="auto"                  # automatic label placement
                )

            # chance-level computation
            chance_level, observed, null = compute_chance_level(correct_trials_all, incorrect_trials_all)


            # Plot session-by-session success rate for AI ON and AI OFF.
            plot_success_rate_per_session(
                success_rate_ai_on_all,
                success_rate_ai_off_all,
                monkey=monkey,
                experiment=experiment,
                chance_level=chance_level
            )

            # -----------------------------------------------------------------
            # Session-level paired statistics and summary metrics
            # -----------------------------------------------------------------
            if success_rate_ai_on_all and success_rate_ai_off_all:
                # Paired significance test across sessions.
                # By default this uses a two-sided Wilcoxon signed-rank test.
                test = paired_success_test(
                    success_rate_ai_on_all,
                    success_rate_ai_off_all,
                    alternative="two-sided"
                )

                print(f"\n[Paired across-session test: Wilcoxon signed-rank]")
                print(
                    f"n_sessions_total={test['n_total']}, "
                    f"zeros_dropped={test['n_zero']}, "
                    f"n_used={test['n_used']}"
                )
                print(
                    f"Δ=ON−OFF: median={test['median_delta']:.3f}, "
                    f"mean={test['mean_delta']:.3f}"
                )
                if np.isfinite(test["p"]):
                    print(f"W={test['W']:.3f}, p={test['p']:.4g} (alternative='two-sided')")

                # -------------------------------------------------------------
                # Learning curve across sessions
                # -------------------------------------------------------------
                # Assess monotonic change in performance across session number
                # using Spearman correlation.
                session_nums = np.arange(len(success_rate_ai_on_all))

                if len(session_nums) >= 3:
                    rho_on, p_on = stats.spearmanr(
                        session_nums,
                        success_rate_ai_on_all,
                        nan_policy="omit"
                    )
                    rho_off, p_off = stats.spearmanr(
                        session_nums,
                        success_rate_ai_off_all,
                        nan_policy="omit"
                    )

                    print(f"[Learning curve Spearman] {experiment} – {monkey}")
                    print(f"  AI ON : rho={rho_on:.3f}, p={p_on:.3g}, n={len(success_rate_ai_on_all)}")
                    print(f"  AI OFF: rho={rho_off:.3f}, p={p_off:.3g}, n={len(success_rate_ai_off_all)}")
                else:
                    print(
                        f"[Learning curve Spearman] {experiment} – {monkey}: "
                        f"not enough sessions (n={len(session_nums)})"
                    )

                # Mean and standard deviation across session-level success rates.
                mean_on = np.nanmean(success_rate_ai_on_all)
                std_on = np.nanstd(success_rate_ai_on_all, ddof=1) if len(success_rate_ai_on_all) > 1 else np.nan

                mean_off = np.nanmean(success_rate_ai_off_all)
                std_off = np.nanstd(success_rate_ai_off_all, ddof=1) if len(success_rate_ai_off_all) > 1 else np.nan

                # -------------------------------------------------------------
                # Pooled success rate across all sessions
                # -------------------------------------------------------------
                # This is count-based pooling, not the average of session means.
                total_correct_on = sum(len(v) for v in cumulative_correct_ai_on.values())
                total_incorrect_on = sum(len(v) for v in cumulative_incorrect_ai_on.values())
                pooled_on = (
                    total_correct_on / (total_correct_on + total_incorrect_on)
                    if (total_correct_on + total_incorrect_on) > 0 else np.nan
                )

                total_correct_off = sum(len(v) for v in cumulative_correct_ai_off.values())
                total_incorrect_off = sum(len(v) for v in cumulative_incorrect_ai_off.values())
                pooled_off = (
                    total_correct_off / (total_correct_off + total_incorrect_off)
                    if (total_correct_off + total_incorrect_off) > 0 else np.nan
                )

                print(f"\n>>> {experiment} – {monkey}")
                print(
                    f"AI ON :  mean={mean_on:.3f}, std={std_on:.3f}, "
                    f"pooled={pooled_on:.3f}, n_sessions={len(success_rate_ai_on_all)}"
                )
                print(
                    f"AI OFF:  mean={mean_off:.3f}, std={std_off:.3f}, "
                    f"pooled={pooled_off:.3f}, n_sessions={len(success_rate_ai_off_all)}"
                )

                # Store summary values for optional later use.
                summary_success_rates.append({
                    "experiment": experiment,
                    "monkey": monkey,
                    "mean_on": mean_on,
                    "std_on": std_on,
                    "mean_off": mean_off,
                    "std_off": std_off,
                    "pooled_on": pooled_on,
                    "pooled_off": pooled_off,
                    "n_sessions": len(success_rate_ai_on_all)
                })

    # Finalize and export the summary table if implemented in SuccessRateSummary.
    summary_table.finalize()

    # Plot global summary: AI gain versus baseline performance.
    plot_summary_scatter(
        all_summary_points,
        base_dir,
        title="AI gain is maximal at intermediate baseline"
    )


def main():
    """
    Parse command-line arguments and run the success-rate analysis pipeline.

    If --base_dir is not provided, the script loads it from config.yaml
    located in the project root.
    """
    parser = argparse.ArgumentParser(
        description="Run success-rate analysis for AI vs non-AI BCI trials."
    )

    parser.add_argument(
        '--monkeys',
        nargs='+',
        default=['Monkey 1', 'Monkey 2'],
        help='List of monkeys to analyze'
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        default=["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2", "AI Respawn"],
        help='List of experiment names'
    )

    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='Path to the root data directory'
    )

    args = parser.parse_args()

    # Load base_dir from config.yaml if not provided on the command line.
    if args.base_dir is None:
        import yaml
        import pathlib

        config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        args.base_dir = config.get("base_dir")
        if args.base_dir is None:
            raise ValueError("base_dir must be specified via CLI or in config.yaml")

    run_success_rate(args.monkeys, args.experiments, args.base_dir)


if __name__ == "__main__":
    main()