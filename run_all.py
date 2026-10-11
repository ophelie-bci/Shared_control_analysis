"""
Run manuscript figure/table scripts for the AI-assisted BCI navigation paper.

This driver is intentionally conservative:
- implemented analyses are run from raw data;
- failure-mode manuscript plots are regenerated from the cached grand summary by
  default, because the recomputation script still contains placeholder bodies.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
    "AI Respawn",
]


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_base_dir(cli_base_dir: Optional[str]) -> str:
    if cli_base_dir:
        return cli_base_dir

    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def run_step(name: str, func, failures: list[dict], keep_going: bool) -> None:
    print(f"\n=== {name} ===")
    try:
        func()
    except Exception as exc:
        failures.append({"step": name, "error": str(exc)})
        print(f"[FAILED] {name}: {exc}")
        traceback.print_exc()
        if not keep_going:
            raise


def run_failure_mode_plots(recompute: bool, monkeys: list[str], experiments: list[str], base_dir: str) -> None:
    from scripts import run_failure_mode_analysis as failure
    from src.trial_utils import load_grand_summary, save_grand_summary

    if recompute:
        from scripts.verify_figure5_failure_modes import recompute_grand_summary

        grand_summary = recompute_grand_summary(base_dir, monkeys, experiments)
        save_grand_summary(grand_summary, experiments)
    else:
        grand_summary = load_grand_summary()

    failure.plot_success_by_choice_state_global_final(grand_summary, behavioral_only=False)
    failure.plot_failure_modes_all(grand_summary, save_prefix="failure_modes_rebinned")
    failure.plot_failure_modes_global(grand_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild manuscript artifacts for the Github AI paper folder.")
    parser.add_argument("--base-dir", default=None, help="Root data directory. Defaults to config.yaml.")
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--skip-success-rate", action="store_true")
    parser.add_argument("--skip-decoder-electrode-summary", action="store_true")
    parser.add_argument("--skip-collision-corrected-success", action="store_true")
    parser.add_argument("--skip-figure2", action="store_true")
    parser.add_argument("--skip-extended-data-figure1", action="store_true")
    parser.add_argument("--skip-temporal-prior-reset-example", action="store_true")
    parser.add_argument("--skip-respawn-reorientation-lag", action="store_true")
    parser.add_argument("--skip-respawn-bci-reorientation-by-outcome", action="store_true")
    parser.add_argument("--skip-respawn-bci-reorientation-sensitivity", action="store_true")
    parser.add_argument("--skip-obstacle-session-metrics", action="store_true")
    parser.add_argument("--skip-obstacle-exclusion-audit", action="store_true")
    parser.add_argument("--skip-same-alpha-assistance", action="store_true")
    parser.add_argument("--skip-time-resolved-bci-influence", action="store_true")
    parser.add_argument("--skip-decoded-command-comparison", action="store_true")
    parser.add_argument("--skip-neural-target-generalization", action="store_true")
    parser.add_argument("--skip-shared-control-adaptation", action="store_true")
    parser.add_argument("--skip-hidden-prior-followup-gain", action="store_true")
    parser.add_argument("--skip-bci-only-learning-followup", action="store_true")
    parser.add_argument("--skip-followup-reactive-obstacle", action="store_true")
    parser.add_argument(
        "--hidden-prior-source-monkey",
        default="Monkey 3",
        help=(
            "Raw-data monkey folder for the Monkey 1 hidden-prior follow-up "
            "trial-level model. The session-level comparison uses the cached "
            "manuscript Monkey 1 session table when available."
        ),
    )
    parser.add_argument("--skip-ai-only-baseline", action="store_true")
    parser.add_argument(
        "--ai-only-replay-csv",
        default=None,
        help=(
            "Optional CSV of strict AI-only/shared-control replay outcomes for "
            "the K=1..5 candidate-set-size analysis."
        ),
    )
    parser.add_argument("--skip-prior-confidence", action="store_true")
    parser.add_argument("--skip-reset-prior", action="store_true")
    parser.add_argument("--skip-failure-modes", action="store_true")
    parser.add_argument("--skip-extended-data-tables", action="store_true")
    parser.add_argument(
        "--recompute-failure-modes",
        action="store_true",
        help="Recompute failure-mode grand summary from raw data. Currently guarded until placeholders are restored.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed analysis step.")
    return parser.parse_args()


def main() -> int:
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings(
        "ignore",
        message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
        category=UserWarning,
    )
    _ensure_import_path()

    args = parse_args()
    base_dir = load_base_dir(args.base_dir)
    failures: list[dict] = []
    keep_going = not args.stop_on_error

    if not args.skip_success_rate:
        from scripts import run_success_rate

        run_step(
            "success-rate figures and summary tables",
            lambda: run_success_rate.run_success_rate(args.monkeys, args.experiments, base_dir),
            failures,
            keep_going,
        )

    if not args.skip_decoder_electrode_summary:
        from scripts import run_decoder_electrode_summary

        run_step(
            "decoder electrode summary",
            lambda: run_decoder_electrode_summary.run_decoder_electrode_summary(
                base_dir,
                args.monkeys,
                args.experiments,
            ),
            failures,
            keep_going,
        )

    if not args.skip_collision_corrected_success:
        from scripts import run_collision_corrected_success

        obstacle_experiments = [
            exp for exp in args.experiments
            if exp in run_collision_corrected_success.EXPERIMENT_LABELS
        ]
        run_step(
            "collision-corrected success counterfactual",
            lambda: run_collision_corrected_success.run_collision_corrected_success(
                args.monkeys,
                obstacle_experiments,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_figure2:
        from scripts import make_figure2_fixed_obstacle_example

        run_step(
            "figure 2 fixed-obstacle example",
            lambda: make_figure2_fixed_obstacle_example.main(["--base-dir", base_dir]),
            failures,
            keep_going,
        )

    if not args.skip_extended_data_figure1:
        from scripts import make_extended_data_figure1_trajectory_supplement

        run_step(
            "extended data figure 1 trajectory supplement",
            lambda: make_extended_data_figure1_trajectory_supplement.main(["--base-dir", base_dir]),
            failures,
            keep_going,
        )

    if not args.skip_temporal_prior_reset_example:
        from scripts import make_temporal_prior_reset_example

        run_step(
            "temporal prior reset example",
            lambda: make_temporal_prior_reset_example.main(["--base-dir", base_dir]),
            failures,
            keep_going,
        )

    if not args.skip_respawn_reorientation_lag:
        from scripts import run_respawn_reorientation_lag

        run_step(
            "respawn reorientation-lag analysis",
            lambda: run_respawn_reorientation_lag.run_respawn_reorientation_lag(
                args.monkeys,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_respawn_bci_reorientation_by_outcome:
        from scripts import run_respawn_bci_reorientation_by_outcome

        run_step(
            "respawn decoded BCI reorientation by outcome",
            lambda: run_respawn_bci_reorientation_by_outcome.run_respawn_bci_reorientation_by_outcome(
                args.monkeys,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_respawn_bci_reorientation_sensitivity:
        from scripts import run_respawn_bci_reorientation_sensitivity

        run_step(
            "respawn decoded BCI reorientation cumulative and sensitivity analysis",
            lambda: run_respawn_bci_reorientation_sensitivity.run(
                args.monkeys,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_obstacle_session_metrics:
        from scripts import run_obstacle_session_metrics

        obstacle_experiments = [
            exp for exp in args.experiments
            if exp in run_obstacle_session_metrics.EXPERIMENT_LABELS
        ]
        run_step(
            "obstacle session-level trajectory and override metrics",
            lambda: run_obstacle_session_metrics.run_obstacle_session_metrics(
                args.monkeys,
                obstacle_experiments,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_obstacle_exclusion_audit:
        from scripts import run_obstacle_exclusion_audit

        obstacle_experiments = [
            exp for exp in args.experiments
            if exp in run_obstacle_exclusion_audit.DEFAULT_EXPERIMENTS
        ]
        run_step(
            "near-obstacle decoded-choice exclusion audit",
            lambda: run_obstacle_exclusion_audit.run_obstacle_exclusion_audit(
                args.monkeys,
                obstacle_experiments,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_same_alpha_assistance:
        from scripts import run_same_alpha_assistance_analysis

        run_step(
            "same-alpha assistance magnitude analysis",
            lambda: run_same_alpha_assistance_analysis.run_analysis(
                base_dir,
                args.monkeys,
                args.experiments,
            ),
            failures,
            keep_going,
        )

    if not args.skip_time_resolved_bci_influence:
        from scripts import run_time_resolved_bci_influence

        obstacle_experiments = [
            exp for exp in args.experiments
            if exp in run_time_resolved_bci_influence.EXPERIMENT_LABELS
        ]
        run_step(
            "time-resolved BCI influence on executed command",
            lambda: run_time_resolved_bci_influence.run_time_resolved_bci_influence(
                args.monkeys,
                obstacle_experiments,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_decoded_command_comparison:
        from scripts import run_decoded_command_condition_comparison

        obstacle_experiments = [
            exp for exp in args.experiments
            if exp in run_decoded_command_condition_comparison.EXPERIMENT_LABELS
        ]
        run_step(
            "decoded-command comparison between BCI-only and shared-control",
            lambda: run_decoded_command_condition_comparison.run_analysis(
                base_dir,
                args.monkeys,
                obstacle_experiments,
            ),
            failures,
            keep_going,
        )

    if not args.skip_neural_target_generalization:
        from scripts import run_neural_target_cross_condition_generalization

        obstacle_experiments = [
            exp for exp in args.experiments
            if exp in run_neural_target_cross_condition_generalization.EXPERIMENT_LABELS
        ]
        run_step(
            "neural target cross-condition generalization",
            lambda: run_neural_target_cross_condition_generalization.run_analysis(
                base_dir,
                args.monkeys,
                obstacle_experiments,
            ),
            failures,
            keep_going,
        )

    if not args.skip_shared_control_adaptation:
        from scripts import run_shared_control_adaptation_analysis

        run_step(
            "shared-control adaptation analysis",
            lambda: run_shared_control_adaptation_analysis.run_analysis(),
            failures,
            keep_going,
        )

    if not args.skip_hidden_prior_followup_gain:
        from scripts import run_hidden_prior_followup_gain

        run_step(
            "hidden-prior follow-up shared-control gain analysis",
            lambda: run_hidden_prior_followup_gain.run_analysis(
                base_dir,
                args.hidden_prior_source_monkey,
            ),
            failures,
            keep_going,
        )

    if not args.skip_bci_only_learning_followup:
        from scripts import run_bci_only_learning_followup

        run_step(
            "BCI-only learning across original and follow-up appearing-obstacle sessions",
            lambda: run_bci_only_learning_followup.run_analysis(
                base_dir,
                "Monkey 1",
            ),
            failures,
            keep_going,
        )

    if not args.skip_followup_reactive_obstacle:
        from scripts import run_followup_reactive_obstacle_analysis

        run_step(
            "follow-up reactive obstacle analysis",
            lambda: run_followup_reactive_obstacle_analysis.run_analysis(
                base_dir,
                args.hidden_prior_source_monkey,
            ),
            failures,
            keep_going,
        )

    if not args.skip_ai_only_baseline:
        from scripts import run_ai_only_baseline

        run_step(
            "AI-only baseline and candidate-set-size analysis",
            lambda: run_ai_only_baseline.run_ai_only_baseline(args.ai_only_replay_csv),
            failures,
            keep_going,
        )

    if not args.skip_failure_modes:
        failure_experiments = [exp for exp in args.experiments if exp != "AI Respawn"]
        run_step(
            "failure-mode figures",
            lambda: run_failure_mode_plots(
                args.recompute_failure_modes,
                args.monkeys,
                failure_experiments,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_prior_confidence:
        from scripts import run_prior_confidence_index_analysis

        run_step(
            "prior-confidence figures",
            lambda: run_prior_confidence_index_analysis.run_prior_confidence_analysis(
                args.monkeys,
                args.experiments,
                base_dir,
            ),
            failures,
            keep_going,
        )

    if not args.skip_reset_prior:
        from scripts import run_reset_prior_analysis

        run_step(
            "reset-prior summary",
            lambda: run_reset_prior_analysis.main(base_dir),
            failures,
            keep_going,
        )

    if not args.skip_extended_data_tables:
        from scripts import make_extended_data_tables

        run_step(
            "extended data tables",
            lambda: make_extended_data_tables.main(["--base-dir", base_dir]),
            failures,
            keep_going,
        )

    from scripts import organize_artifacts

    run_step(
        "organize figure and table artifacts",
        lambda: organize_artifacts.main(),
        failures,
        keep_going,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
