"""Run the strict AI-only baseline and candidate-set-size summaries.

This script evaluates replay outputs. If no replay CSV is present, it writes the
analysis specification and an input template without fabricating performance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_only_baseline import (  # noqa: E402
    load_replay_rows,
    plot_candidate_set_success,
    summarize_replay_rows,
    write_analysis_note,
    write_template,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"


def run_ai_only_baseline(
    replay_csv: str | Path | None = None,
    output_tables_dir: str | Path = TABLES_DIR / "AI-only baseline",
    output_figures_dir: str | Path = FIGURES_DIR / "Extended Data Figure 3" / "AI-only baseline",
) -> dict[str, Path | None]:
    """Evaluate strict AI-only replay rows and write manuscript artifacts."""
    output_tables_dir = Path(output_tables_dir)
    output_figures_dir = Path(output_figures_dir)
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    note_path = output_tables_dir / "ai_only_baseline_analysis_note.md"
    template_path = output_tables_dir / "ai_only_candidate_replay_rows_template.csv"
    write_analysis_note(note_path)
    write_template(template_path)

    replay_path = Path(replay_csv) if replay_csv else output_tables_dir / "ai_only_candidate_replay_rows.csv"
    if not replay_path.exists():
        print(
            "[INFO] No AI-only replay CSV found. Wrote the analysis note and "
            f"input template to {output_tables_dir}."
        )
        return {
            "analysis_note": note_path,
            "template": template_path,
            "normalized_rows": None,
            "summary": None,
            "reached_distribution": None,
            "shared_minus_ai_only": None,
            "figure": None,
        }

    rows = load_replay_rows(replay_path)
    summary, reached, critical_k5, paired = summarize_replay_rows(rows)

    normalized_path = output_tables_dir / "ai_only_candidate_replay_rows_normalized.csv"
    summary_path = output_tables_dir / "ai_only_baseline_summary.csv"
    reached_path = output_tables_dir / "ai_only_reached_target_distribution.csv"
    critical_k5_path = output_tables_dir / "ai_only_k5_critical_comparison.csv"
    paired_path = output_tables_dir / "shared_minus_ai_only_by_k.csv"
    figure_path = output_figures_dir / "ai_only_candidate_set_success.svg"

    rows.to_csv(normalized_path, index=False)
    summary.to_csv(summary_path, index=False)
    reached.to_csv(reached_path, index=False)
    critical_k5.to_csv(critical_k5_path, index=False)
    if not paired.empty:
        paired.to_csv(paired_path, index=False)
    plot_candidate_set_success(summary, figure_path)

    print(f"[INFO] Wrote AI-only baseline summaries to {output_tables_dir}")
    print(f"[INFO] Wrote AI-only candidate-set-size figure to {figure_path}")
    return {
        "analysis_note": note_path,
        "template": template_path,
        "normalized_rows": normalized_path,
        "summary": summary_path,
        "reached_distribution": reached_path,
        "critical_k5": critical_k5_path,
        "shared_minus_ai_only": paired_path,
        "figure": figure_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate strict AI-only and shared-control replay rows over K=1..5 candidate sets."
    )
    parser.add_argument(
        "--replay-csv",
        default=None,
        help=(
            "CSV of replay outcomes. Defaults to "
            "tables/AI-only baseline/ai_only_candidate_replay_rows.csv if present."
        ),
    )
    parser.add_argument(
        "--output-tables-dir",
        default=str(TABLES_DIR / "AI-only baseline"),
        help="Directory for AI-only baseline tables and analysis notes.",
    )
    parser.add_argument(
        "--output-figures-dir",
        default=str(FIGURES_DIR / "Extended Data Figure 3" / "AI-only baseline"),
        help="Directory for the candidate-set-size success figure.",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> dict[str, Path | None]:
    args = parse_args() if argv is None else parse_args_from_list(argv)
    return run_ai_only_baseline(
        replay_csv=args.replay_csv,
        output_tables_dir=args.output_tables_dir,
        output_figures_dir=args.output_figures_dir,
    )


def parse_args_from_list(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate strict AI-only and shared-control replay rows over K=1..5 candidate sets."
    )
    parser.add_argument("--replay-csv", default=None)
    parser.add_argument("--output-tables-dir", default=str(TABLES_DIR / "AI-only baseline"))
    parser.add_argument("--output-figures-dir", default=str(FIGURES_DIR / "Extended Data Figure 3" / "AI-only baseline"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
