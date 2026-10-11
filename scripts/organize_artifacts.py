"""Organize generated manuscript artifacts into figure/table folders."""

from __future__ import annotations

import shutil
import stat
import time
import filecmp
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"


FIGURE_DIR_TARGETS = [
    ("Figure 2", "Figure 2"),
    ("Extended Data Figure 1", "Extended Data Figure 1"),
    ("Failure modes (execution grouped) - all monkeys x all tasks", "Figure 5/Failure modes"),
    ("Execution failure modes (all monkeys x all tasks)", "Figure 5/Failure modes"),
    ("Success by choice state (global)", "Figure 5/Success by choice state"),
]

FIGURE3_AI_GAIN_FILES = {
    "AI_gain_labels_Monkey 1_AI Appearing Obstacle 2.svg",
    "AI_gain_labels_Monkey 2_AI Appearing Obstacle.svg",
    "Performance-dependent AI gains.svg",
}

FIGURE5_PRIOR_CONFIDENCE_FILES = {
    "alpha_vs_time_FROM_EVENT_Monkey 1_AI Appearing Obstacle.svg",
    "alpha_vs_time_FROM_EVENT_Monkey 1_AI Respawn.svg",
    "alpha_vs_time_FROM_EVENT_Monkey 2_AI Appearing Obstacle.svg",
}

EXTENDED_DATA_FIGURE2_FILES = {
    "alpha_vs_time_FROM_START_Monkey 1_AI Obstacle.svg",
    "alpha_vs_time_FROM_START_Monkey 2_AI Obstacle.svg",
}

TABLE_TARGETS = [
    (
        "extended_data_table_1_success_rate_by_task_excel_semicolon.csv",
        "Extended Data Table 1",
    ),
    ("ai_only_baseline_analysis_note.md", "AI-only baseline"),
    ("ai_only_candidate_replay_rows_template.csv", "AI-only baseline"),
    ("ai_only_replay_run_status.md", "AI-only baseline"),
    ("example_aitrials_pickle_field_inspection.md", "AI-only baseline"),
    ("example_ai_log_field_inspection.md", "AI-only baseline"),
    ("ai_only_candidate_replay_rows.csv", "AI-only baseline"),
    ("ai_only_candidate_replay_rows_normalized.csv", "AI-only baseline"),
    ("ai_only_baseline_summary.csv", "AI-only baseline"),
    ("ai_only_reached_target_distribution.csv", "AI-only baseline"),
    ("ai_only_k5_critical_comparison.csv", "AI-only baseline"),
    ("shared_minus_ai_only_by_k.csv", "AI-only baseline"),
    ("hidden_prior_followup_trial_rows.csv", "Hidden prior follow-up"),
    ("hidden_prior_followup_session_gains.csv", "Hidden prior follow-up"),
    ("hidden_prior_followup_gain_summary.csv", "Hidden prior follow-up"),
    ("hidden_prior_followup_gain_comparison.csv", "Hidden prior follow-up"),
    ("hidden_prior_followup_trial_level_interaction.csv", "Hidden prior follow-up"),
    ("hidden_prior_followup_trial_success_summary.csv", "Hidden prior follow-up"),
    ("hidden_prior_followup_report.md", "Hidden prior follow-up"),
    ("appearing_obstacle_log_coordinate_audit.csv", "Hidden prior follow-up"),
    ("appearing_obstacle_log_coordinate_audit_summary.csv", "Hidden prior follow-up"),
    ("appearing_obstacle_log_coordinate_audit.md", "Hidden prior follow-up"),
    ("followup_reactive_obstacle_trial_metrics.csv", "Follow-up reactive obstacle"),
    ("followup_reactive_obstacle_session_metrics.csv", "Follow-up reactive obstacle"),
    ("followup_reactive_obstacle_summary.csv", "Follow-up reactive obstacle"),
    ("followup_reactive_obstacle_trial_models.csv", "Follow-up reactive obstacle"),
    ("followup_reactive_obstacle_report.md", "Follow-up reactive obstacle"),
    ("gain_curve_target_session_rows.csv", "Gain curve pooling audit"),
    ("gain_curve_baseline_bin_counts.csv", "Gain curve pooling audit"),
    ("gain_curve_baseline_distribution.csv", "Gain curve pooling audit"),
    ("gain_curve_nested_model_comparisons.csv", "Gain curve pooling audit"),
    ("gain_curve_cross_validated_mse.csv", "Gain curve pooling audit"),
    ("gain_curve_binomial_logit_robustness.csv", "Gain curve pooling audit"),
    ("gain_curve_separate_quadratic_fits.csv", "Gain curve pooling audit"),
    ("gain_curve_pooling_audit_report.md", "Gain curve pooling audit"),
    ("figure3c_original_session_target_audit.csv", "Figure 3C original session audit"),
    ("figure3c_original_plotted_target_points.csv", "Figure 3C original session audit"),
    ("figure3c_original_experiment_summary.csv", "Figure 3C original session audit"),
    ("figure3c_original_session_audit_report.md", "Figure 3C original session audit"),
    ("figure3c_stratified_points.csv", "Figure 3C stratified"),
    ("figure3c_stratified_quadratic_fits.csv", "Figure 3C stratified"),
    ("figure3c_stratified_plot_report.md", "Figure 3C stratified"),
]
ALLOWED_TABLE_FOLDERS = {folder for _, folder in TABLE_TARGETS}
ALLOWED_TABLE_FILES = {filename for filename, _ in TABLE_TARGETS}


def _safe_unlink(path: Path) -> bool:
    for attempt in range(5):
        try:
            path.chmod(stat.S_IWRITE)
            path.unlink()
            return True
        except PermissionError:
            if attempt == 4:
                print(f"[WARN] Could not remove locked file: {path}")
                return False
            time.sleep(0.25)


def _move_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        _safe_unlink(src)
        return dst
    if dst.exists():
        if not _safe_unlink(dst):
            shutil.copy2(str(src), str(dst))
            _safe_unlink(src)
            return dst
    shutil.copy2(str(src), str(dst))
    _safe_unlink(src)
    return dst


def _move_files_from_dir(src_dir: Path, dst_dir: Path) -> list[Path]:
    moved = []
    if not src_dir.exists():
        return moved
    for path in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(src_dir)
        moved.append(_move_file(path, dst_dir / rel))
    return moved


def _figure_file_target(path: Path) -> Path | None:
    name = path.name
    if name.startswith("success_rate_per_session_"):
        return FIGURES_DIR / "Figure 1" / "Success rate" / name
    if name.startswith("figure2_fixed_obstacle_"):
        return FIGURES_DIR / "Figure 2" / name
    if name in FIGURE3_AI_GAIN_FILES:
        return FIGURES_DIR / "Figure 4" / "AI gain with baseline" / name
    if name in {"execution_failure_modes__global.svg", "failure_modes_rebinned__global.svg"}:
        return FIGURES_DIR / "Figure 5" / "Failure modes" / name
    if name == "global_success_by_state_success_by_state.svg":
        return FIGURES_DIR / "Figure 5" / "Success by choice state" / name
    if name in FIGURE5_PRIOR_CONFIDENCE_FILES:
        return FIGURES_DIR / "Figure 6" / "Prior confidence index" / name
    if name.startswith("temporal_prior_reset_example_"):
        return FIGURES_DIR / "Figure 6" / "Temporal prior reset" / name
    if name in EXTENDED_DATA_FIGURE2_FILES:
        return FIGURES_DIR / "Extended Data Figure 2" / "Prior confidence index" / name
    return None


def organize_figures_by_file() -> list[Path]:
    moved = []
    if not FIGURES_DIR.exists():
        return moved
    for path in sorted(p for p in FIGURES_DIR.rglob("*") if p.is_file()):
        target = _figure_file_target(path)
        if target is not None:
            moved.append(_move_file(path, target))
    return moved


def discard_nonmanuscript_figure_outputs() -> None:
    """Remove generated generic outputs that are not assigned to a manuscript figure."""
    discard_dirs = [
        FIGURES_DIR / "AI gain with baseline",
        FIGURES_DIR / "success_rate",
    ]
    for discard_dir in discard_dirs:
        if not discard_dir.exists():
            continue
        for path in sorted(p for p in discard_dir.rglob("*") if p.is_file()):
            _safe_unlink(path)


def organize_figures() -> list[Path]:
    moved = []
    for source_name, target_name in FIGURE_DIR_TARGETS:
        src_dir = FIGURES_DIR / source_name
        dst_dir = FIGURES_DIR / target_name
        moved.extend(_move_files_from_dir(src_dir, dst_dir))
    moved.extend(organize_figures_by_file())
    discard_nonmanuscript_figure_outputs()
    return moved


def _table_target(path: Path) -> Path | None:
    name = path.name
    for filename, folder in TABLE_TARGETS:
        if name == filename:
            return TABLES_DIR / folder / name
    return None


def organize_tables() -> list[Path]:
    moved = []
    for path in sorted(TABLES_DIR.glob("*")):
        if not path.is_file():
            continue
        target = _table_target(path)
        if target is not None:
            moved.append(_move_file(path, target))
        else:
            _safe_unlink(path)
    for folder in sorted(p for p in TABLES_DIR.iterdir() if p.is_dir()):
        if folder.name not in ALLOWED_TABLE_FOLDERS:
            for path in sorted(p for p in folder.rglob("*") if p.is_file()):
                _safe_unlink(path)
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            if path.name not in ALLOWED_TABLE_FILES:
                _safe_unlink(path)
    return moved


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> list[Path]:
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    moved = []
    moved.extend(organize_figures())
    moved.extend(organize_tables())
    prune_empty_dirs(FIGURES_DIR)
    prune_empty_dirs(TABLES_DIR)
    for path in moved:
        print(path)
    return moved


if __name__ == "__main__":
    main()
