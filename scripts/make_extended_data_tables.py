"""
Build manuscript-ready Extended Data tables from the AI paper analysis outputs.

The analysis scripts also write compact internal CSVs. This script converts
those into clean table files with explicit columns, units, and AI ON/OFF
comparisons that are easier to inspect and submit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "tables"
BASE_DIR_OVERRIDE: str | None = None
DEFAULT_ANALYSIS_ORDER = [
    ("Monkey 1", "AI Obstacle"),
    ("Monkey 1", "AI Appearing Obstacle"),
    ("Monkey 1", "AI Appearing Obstacle 2"),
    ("Monkey 1", "AI Respawn"),
    ("Monkey 2", "AI Obstacle"),
    ("Monkey 2", "AI Appearing Obstacle"),
]
TASK_LABELS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle (6 months later)",
    "AI Respawn": "Respawn",
}
DEFAULT_AIC_ORDER = [
    ("Monkey 1", "AI Obstacle"),
    ("Monkey 1", "AI Appearing Obstacle"),
    ("Monkey 1", "AI Appearing Obstacle 2"),
    ("Monkey 2", "AI Obstacle"),
    ("Monkey 2", "AI Appearing Obstacle"),
]
AIC_TASK_LABELS = {
    "AI Obstacle": "AI Obstacle",
    "AI Appearing Obstacle": "AI Appearing Obstacle",
    "AI Appearing Obstacle 2": "AI Appearing Obstacle (6 months later)",
}


def _round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_base_dir() -> str:
    if BASE_DIR_OVERRIDE:
        return BASE_DIR_OVERRIDE
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("base_dir:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            return value.replace("\\\\", "\\")
    raise ValueError(f"No base_dir entry found in {config_path}")


def _as_percent(series: pd.Series) -> pd.Series:
    return 100.0 * pd.to_numeric(series, errors="coerce")


def _fmt_mean_sd(mean_fraction: float, sd_fraction: float) -> str:
    if not np.isfinite(mean_fraction) or not np.isfinite(sd_fraction):
        return ""
    return f"{mean_fraction * 100:.0f} +/- {sd_fraction * 100:.0f}"


def _fmt_gain(gain_fraction: float) -> str:
    if not np.isfinite(gain_fraction):
        return ""
    value = gain_fraction * 100.0
    if abs(value - round(value)) < 0.005:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _fmt_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value == 0:
        return "<1e-300"
    exponent = int(np.floor(np.log10(abs(p_value))))
    mantissa = p_value / (10 ** exponent)
    return f"{mantissa:.2f} x 10^{exponent}"


def _exact_chance_level(correct_trials_all: list, calculate_success) -> float:
    """
    Deterministic expectation of the original target-shuffle chance level.

    The legacy helper estimates this by repeatedly permuting the empirical
    target list. The expected value is the average success rate obtained by
    assigning each empirical target to every trial, weighted by target count.
    """
    trials = [trial for session in correct_trials_all for trial in session]
    if not trials:
        return np.nan

    states_data = [
        list(zip(trial.avatarTrajectory["x"], trial.avatarTrajectory["y"], trial.avatarTrajectory["z"]))
        for trial in trials
    ]
    obstacles = [trial.obstaclePosition for trial in trials]
    target_counts: dict[tuple[float, float, float], int] = {}
    target_values: dict[tuple[float, float, float], np.ndarray] = {}
    for trial in trials:
        target = np.asarray(trial.targetPosition, dtype=float)
        key = tuple(float(x) for x in np.round(target, 6))
        target_counts[key] = target_counts.get(key, 0) + 1
        target_values[key] = target

    weighted = 0.0
    total = float(sum(target_counts.values()))
    for key, count in target_counts.items():
        repeated_targets = [target_values[key]] * len(trials)
        weighted += count * calculate_success(states_data, repeated_targets, obstacles)
    return weighted / total


def make_success_rate_table() -> pd.DataFrame:
    """Build Extended Data Table 1 directly from session trial data."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.load import load_files
    from src.trial_utils import categorize_trials_by_target_and_ai
    from src.success_rate_metrics import calculate_accuracy_per_target
    from src.stats_tests import calculate_success, paired_success_test

    base_dir = _read_base_dir()
    rows = []
    for monkey, experiment in DEFAULT_ANALYSIS_ORDER:
        all_trials, correct_trials_all, incorrect_trials_all, *_ = load_files(
            experiment, monkey, base_dir, nb_trials=None
        )
        if not all_trials:
            continue

        success_on = []
        success_off = []
        for correct_trials, incorrect_trials in zip(correct_trials_all, incorrect_trials_all):
            correct_ai_on = categorize_trials_by_target_and_ai(experiment, correct_trials, ai_condition=1.0)
            correct_ai_off = categorize_trials_by_target_and_ai(experiment, correct_trials, ai_condition=0.0)
            incorrect_ai_on = categorize_trials_by_target_and_ai(experiment, incorrect_trials, ai_condition=1.0)
            incorrect_ai_off = categorize_trials_by_target_and_ai(experiment, incorrect_trials, ai_condition=0.0)

            _, _, overall_on = calculate_accuracy_per_target(correct_ai_on, incorrect_ai_on)
            _, _, overall_off = calculate_accuracy_per_target(correct_ai_off, incorrect_ai_off)
            success_on.append(overall_on)
            success_off.append(overall_off)

        success_on = np.asarray(success_on, dtype=float)
        success_off = np.asarray(success_off, dtype=float)
        paired = paired_success_test(success_on, success_off, alternative="two-sided")
        chance_level = _exact_chance_level(correct_trials_all, calculate_success)
        mean_on = float(np.nanmean(success_on))
        sd_on = float(np.nanstd(success_on, ddof=1)) if len(success_on) > 1 else np.nan
        mean_off = float(np.nanmean(success_off))
        sd_off = float(np.nanstd(success_off, ddof=1)) if len(success_off) > 1 else np.nan
        gain = mean_on - mean_off

        rows.append(
            {
                "Monkey": monkey,
                "Task": TASK_LABELS.get(experiment, experiment),
                "Number of sessions": int(len(success_on)),
                "BCI-only (mean +/- SD)": _fmt_mean_sd(mean_off, sd_off),
                "Shared-control (mean +/- SD)": _fmt_mean_sd(mean_on, sd_on),
                "Chance level": int(round(chance_level * 100)),
                "Shared-control Gain (pp)": _fmt_gain(gain),
                "p-value (paired Wilcoxon)": _fmt_p_value(float(paired.get("p", np.nan))),
            }
        )

    return pd.DataFrame(rows)


def make_target_gain_table() -> pd.DataFrame:
    src = _safe_read_csv(TABLES_DIR / "success_rate_target_gain_summary.csv")
    if src.empty:
        return src

    out = src.rename(
        columns={
            "experiment": "task",
            "baseline_off_pct": "ai_off_baseline_success_percent",
            "gain_pp": "ai_gain_percentage_points",
        }
    )
    out.insert(0, "extended_data_table", "Extended Data Table 2")
    return _round_numeric(out)


def _ols_aic(y: np.ndarray, x: np.ndarray, include_intercept: bool) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if len(y) < 3:
        return np.nan

    design = np.column_stack([np.ones_like(x), x]) if include_intercept else x[:, None]
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    rss = float(np.sum(resid**2))
    if rss <= 0:
        rss = np.finfo(float).tiny
    n = len(y)
    k = design.shape[1]
    return float(n * (np.log(2.0 * np.pi) + 1.0 + np.log(rss / n)) + 2.0 * k)


def make_additive_multiplicative_aic_table() -> pd.DataFrame:
    """Build Extended Data Table 2 from per-session success-rate data."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.load import load_files
    from src.trial_utils import categorize_trials_by_target_and_ai
    from src.success_rate_metrics import calculate_accuracy_per_target

    base_dir = _read_base_dir()
    rows = []
    for monkey, experiment in DEFAULT_AIC_ORDER:
        all_trials, correct_trials_all, incorrect_trials_all, *_ = load_files(
            experiment, monkey, base_dir, nb_trials=None
        )
        if not all_trials:
            continue

        success_on = []
        success_off = []
        for correct_trials, incorrect_trials in zip(correct_trials_all, incorrect_trials_all):
            correct_ai_on = categorize_trials_by_target_and_ai(experiment, correct_trials, ai_condition=1.0)
            correct_ai_off = categorize_trials_by_target_and_ai(experiment, correct_trials, ai_condition=0.0)
            incorrect_ai_on = categorize_trials_by_target_and_ai(experiment, incorrect_trials, ai_condition=1.0)
            incorrect_ai_off = categorize_trials_by_target_and_ai(experiment, incorrect_trials, ai_condition=0.0)

            _, _, overall_on = calculate_accuracy_per_target(correct_ai_on, incorrect_ai_on)
            _, _, overall_off = calculate_accuracy_per_target(correct_ai_off, incorrect_ai_off)
            success_on.append(overall_on)
            success_off.append(overall_off)

        aic_add = _ols_aic(np.asarray(success_on), np.asarray(success_off), include_intercept=True)
        aic_mult = _ols_aic(np.asarray(success_on), np.asarray(success_off), include_intercept=False)
        aic_add_display = round(aic_add, 2)
        aic_mult_display = round(aic_mult, 2)
        delta_display = round(aic_mult_display - aic_add_display, 2)
        rows.append(
            {
                "Monkey": monkey,
                "Task": AIC_TASK_LABELS.get(experiment, experiment),
                "AIC (Additive)": aic_add_display,
                "AIC (Multiplicative)": aic_mult_display,
                "Delta AIC (Mult - Add)": delta_display,
                "Preferred Model": "Additive" if delta_display > 0 else "Multiplicative",
            }
        )

    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        print(f"Skipped empty table: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";")
    print(f"Wrote {path} ({len(df)} rows, {len(df.columns)} columns)")


def build_extended_data_tables(base_dir: str | None = None) -> list[Path]:
    global BASE_DIR_OVERRIDE
    BASE_DIR_OVERRIDE = base_dir
    TABLES_DIR.mkdir(exist_ok=True)

    tables = [
        (
            TABLES_DIR / "Extended Data Table 1" / "extended_data_table_1_success_rate_by_task_excel_semicolon.csv",
            make_success_rate_table(),
        ),
        (
            TABLES_DIR / "Extended Data Table 2" / "extended_data_table_2_additive_multiplicative_aic_excel_semicolon.csv",
            make_additive_multiplicative_aic_table(),
        ),
    ]

    written: list[Path] = []
    for path, df in tables:
        write_table(df, path)
        if path.exists():
            written.append(path)

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build numbered Extended Data tables.")
    parser.add_argument("--base-dir", default=None, help="Root data directory. Defaults to config.yaml.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    return build_extended_data_tables(args.base_dir)


if __name__ == "__main__":
    main()
