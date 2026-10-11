"""
Collision-corrected BCI-only success counterfactual.

This analysis asks how much of the shared-control success-rate gain could be
explained solely by eliminating BCI-only obstacle-block/collision failures.
For each session:

    S_collision_corrected = S_BCI_only + P_blocked,BCI_only

where P_blocked,BCI_only is the fraction of BCI-only trials that failed and
touched the obstacle. This is intentionally generous: every BCI-only collision
failure is converted into a success while all other outcomes stay unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLISION_RADIUS_M = 0.45 + 0.25
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
]
EXPERIMENT_LABELS = {
    "AI Obstacle": ("Fixed Obstacle", "Fixed Obstacle"),
    "AI Appearing Obstacle": ("Appearing Obstacle", "Appearing Obstacle"),
    "AI Appearing Obstacle 2": (
        "Appearing Obstacle",
        "Appearing Obstacle (6 months later)",
    ),
}


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_base_dir(cli_base_dir: str | None) -> str | None:
    if cli_base_dir:
        return cli_base_dir

    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return None

    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("base_dir") or None


def _is_ai_condition(trial, condition: float) -> bool:
    try:
        return float(getattr(trial, "aiVelocityFactor", np.nan)) == float(condition)
    except Exception:
        return False


def _trial_has_obstacle(trial, experiment: str) -> bool:
    obstacle = getattr(trial, "obstaclePosition", None)
    if obstacle is None:
        return False
    try:
        if np.any(np.isnan(np.asarray(obstacle, dtype=float))):
            return False
    except Exception:
        return False
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return getattr(trial, "obstacleAppearedTime", None) is not None
    return True


def _has_obstacle_collision(trial, radius_m: float = COLLISION_RADIUS_M) -> bool:
    if not _trial_has_obstacle(trial, ""):
        return False
    traj = getattr(trial, "avatarTrajectory", None)
    if not isinstance(traj, dict):
        return False
    try:
        points = np.column_stack(
            [
                np.asarray(traj["x"], dtype=float),
                np.asarray(traj["z"], dtype=float),
            ]
        )
        obstacle = np.asarray(
            [trial.obstaclePosition[0], trial.obstaclePosition[2]],
            dtype=float,
        )
    except Exception:
        return False
    if points.size == 0:
        return False
    return bool(np.nanmin(np.linalg.norm(points - obstacle, axis=1)) <= radius_m)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _session_row(monkey: str, experiment: str, session_idx: int, correct_trials, incorrect_trials) -> dict:
    task_family, task_label = EXPERIMENT_LABELS[experiment]

    correct_on = [
        t for t in correct_trials
        if _is_ai_condition(t, 1.0) and _trial_has_obstacle(t, experiment)
    ]
    incorrect_on = [
        t for t in incorrect_trials
        if _is_ai_condition(t, 1.0) and _trial_has_obstacle(t, experiment)
    ]
    correct_off = [
        t for t in correct_trials
        if _is_ai_condition(t, 0.0) and _trial_has_obstacle(t, experiment)
    ]
    incorrect_off = [
        t for t in incorrect_trials
        if _is_ai_condition(t, 0.0) and _trial_has_obstacle(t, experiment)
    ]

    n_on = len(correct_on) + len(incorrect_on)
    n_off = len(correct_off) + len(incorrect_off)
    n_blocked_off = sum(_has_obstacle_collision(t) for t in incorrect_off)

    s_shared = _rate(len(correct_on), n_on)
    s_bci = _rate(len(correct_off), n_off)
    p_blocked_bci = _rate(n_blocked_off, n_off)
    s_collision_corrected = (
        min(1.0, s_bci + p_blocked_bci)
        if np.isfinite(s_bci) and np.isfinite(p_blocked_bci)
        else np.nan
    )
    observed_gain = s_shared - s_bci
    remaining_gain = s_shared - s_collision_corrected
    collision_explained_fraction = (
        p_blocked_bci / observed_gain
        if np.isfinite(observed_gain) and observed_gain > 0
        else np.nan
    )

    return {
        "source": "raw_sessions",
        "monkey": monkey,
        "experiment": experiment,
        "task_family": task_family,
        "task_label": task_label,
        "session_index": session_idx,
        "n_shared": n_on,
        "n_shared_success": len(correct_on),
        "n_bci_only": n_off,
        "n_bci_only_success": len(correct_off),
        "n_bci_only_blocked_failures": int(n_blocked_off),
        "s_shared_pct": 100.0 * s_shared,
        "s_bci_only_pct": 100.0 * s_bci,
        "p_blocked_bci_only_pct": 100.0 * p_blocked_bci,
        "s_collision_corrected_pct": 100.0 * s_collision_corrected,
        "observed_gain_pp": 100.0 * observed_gain,
        "remaining_gain_after_collision_correction_pp": 100.0 * remaining_gain,
        "collision_explained_percent_of_observed_gain": 100.0 * collision_explained_fraction,
    }


def compute_session_counterfactuals(
    monkeys: list[str],
    experiments: list[str],
    base_dir: str,
) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    rows = []
    for experiment in experiments:
        if experiment not in EXPERIMENT_LABELS:
            continue
        for monkey in monkeys:
            all_trials, all_correct, all_incorrect, *_ = load_files(experiment, monkey, base_dir)
            if all_trials == []:
                continue
            for session_idx, (correct_trials, incorrect_trials) in enumerate(zip(all_correct, all_incorrect)):
                rows.append(_session_row(monkey, experiment, session_idx, correct_trials, incorrect_trials))
    return pd.DataFrame(rows)


def compute_cached_unit_counterfactuals() -> pd.DataFrame:
    _ensure_import_path()
    from src.trial_utils import load_grand_summary

    grand_summary = load_grand_summary()
    rows = []
    df = pd.DataFrame(grand_summary.get("failure_props_per_unit", []))
    if df.empty:
        return pd.DataFrame()

    for experiment in DEFAULT_EXPERIMENTS:
        if experiment not in EXPERIMENT_LABELS:
            continue
        for monkey in DEFAULT_MONKEYS:
            unit = df[(df["experiment"] == experiment) & (df["monkey"] == monkey)]
            shared = unit[unit["condition"] == "AI"]
            bci = unit[unit["condition"] == "No AI"]
            if shared.empty or bci.empty:
                continue

            shared_row = shared.iloc[0]
            bci_row = bci.iloc[0]
            s_shared = float(shared_row["success"])
            s_bci = float(bci_row["success"])
            p_blocked_bci = float(bci_row.get("stuck_obstacle", np.nan))
            s_collision_corrected = min(1.0, s_bci + p_blocked_bci)
            observed_gain = s_shared - s_bci
            remaining_gain = s_shared - s_collision_corrected
            collision_explained_fraction = (
                p_blocked_bci / observed_gain
                if np.isfinite(observed_gain) and observed_gain > 0
                else np.nan
            )
            task_family, task_label = EXPERIMENT_LABELS[experiment]

            rows.append({
                "source": "cached_grand_summary_units",
                "monkey": monkey,
                "experiment": experiment,
                "task_family": task_family,
                "task_label": task_label,
                "session_index": np.nan,
                "n_shared": int(shared_row["n_trials"]),
                "n_shared_success": int(round(s_shared * float(shared_row["n_trials"]))),
                "n_bci_only": int(bci_row["n_trials"]),
                "n_bci_only_success": int(round(s_bci * float(bci_row["n_trials"]))),
                "n_bci_only_blocked_failures": int(round(p_blocked_bci * float(bci_row["n_trials"]))),
                "s_shared_pct": 100.0 * s_shared,
                "s_bci_only_pct": 100.0 * s_bci,
                "p_blocked_bci_only_pct": 100.0 * p_blocked_bci,
                "s_collision_corrected_pct": 100.0 * s_collision_corrected,
                "observed_gain_pp": 100.0 * observed_gain,
                "remaining_gain_after_collision_correction_pp": 100.0 * remaining_gain,
                "collision_explained_percent_of_observed_gain": 100.0 * collision_explained_fraction,
            })
    return pd.DataFrame(rows)


def summarize_counterfactuals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["source", "task_family"]
    value_cols = [
        "s_shared_pct",
        "s_bci_only_pct",
        "p_blocked_bci_only_pct",
        "s_collision_corrected_pct",
        "observed_gain_pp",
        "remaining_gain_after_collision_correction_pp",
        "collision_explained_percent_of_observed_gain",
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        source, task_family = keys
        row = {
            "source": source,
            "task_family": task_family,
            "n_rows": int(len(group)),
        }
        for col in value_cols:
            values = group[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{col}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan

        r_values = (
            group["remaining_gain_after_collision_correction_pp"]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        row["n_Rs"] = int(len(r_values))
        row["n_Rs_positive"] = int(np.sum(r_values > 0)) if len(r_values) else 0
        row["n_Rs_negative"] = int(np.sum(r_values < 0)) if len(r_values) else 0
        row["n_Rs_zero"] = int(np.sum(r_values == 0)) if len(r_values) else 0

        row["wilcoxon_Rs_greater_than_zero_statistic"] = np.nan
        row["wilcoxon_Rs_greater_than_zero_p"] = np.nan
        row["wilcoxon_Rs_two_sided_statistic"] = np.nan
        row["wilcoxon_Rs_two_sided_p"] = np.nan
        row["binomial_sign_Rs_greater_than_zero_p"] = np.nan

        if source == "raw_sessions" and len(r_values) > 0:
            nonzero = r_values[r_values != 0]
            if len(nonzero) > 0:
                greater = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
                two_sided = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
                sign = stats.binomtest(
                    int(np.sum(nonzero > 0)),
                    n=int(len(nonzero)),
                    p=0.5,
                    alternative="greater",
                )
                row["wilcoxon_Rs_greater_than_zero_statistic"] = float(greater.statistic)
                row["wilcoxon_Rs_greater_than_zero_p"] = float(greater.pvalue)
                row["wilcoxon_Rs_two_sided_statistic"] = float(two_sided.statistic)
                row["wilcoxon_Rs_two_sided_p"] = float(two_sided.pvalue)
                row["binomial_sign_Rs_greater_than_zero_p"] = float(sign.pvalue)

        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x: float) -> str:
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.1f}"


def _fmt_p(x: float) -> str:
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    if float(x) < 1e-3:
        return f"{float(x):.2e}"
    return f"{float(x):.3f}"


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(exist_ok=True)
    details_path = tables_dir / "collision_corrected_success_by_session.csv"
    summary_path = tables_dir / "collision_corrected_success_summary.csv"
    statement_path = tables_dir / "collision_corrected_success_statement.md"

    df.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    lines = [
        "# Collision-corrected Success Counterfactual",
        "",
        (
            "Counterfactual: every BCI-only failed trial that contacted the obstacle "
            "is counted as a success, with all other outcomes unchanged."
        ),
        f"Rows source: {', '.join(sorted(str(x) for x in df['source'].dropna().unique()))}.",
        "",
    ]
    for _, row in summary.iterrows():
        blocked = row["p_blocked_bci_only_pct_mean"]
        observed = row["observed_gain_pp_mean"]
        remaining_mean = row["remaining_gain_after_collision_correction_pp_mean"]
        remaining_median = row["remaining_gain_after_collision_correction_pp_median"]
        explained = row["collision_explained_percent_of_observed_gain_mean"]
        source = row["source"]
        if source == "raw_sessions":
            lines.append(
                f"- {row['task_family']}: across {int(row['n_Rs'])} sessions, "
                f"mean R_s = {_fmt(remaining_mean)} pp and median R_s = "
                f"{_fmt(remaining_median)} pp after collision correction. "
                f"Two-sided Wilcoxon against zero: p = "
                f"{_fmt_p(row['wilcoxon_Rs_two_sided_p'])}; "
                f"directional R_s > 0 p = {_fmt_p(row['wilcoxon_Rs_greater_than_zero_p'])}. "
                f"{int(row['n_Rs_positive'])}/{int(row['n_Rs'])} sessions had R_s > 0. "
                f"Mean collision-removal term = {_fmt(blocked)} pp of the "
                f"{_fmt(observed)} pp observed improvement ({_fmt(explained)}%)."
            )
        else:
            lines.append(
                f"- {row['task_family']} cached pooled sanity check only: "
                f"mean residual after correction = {_fmt(remaining_mean)} pp "
                f"across {int(row['n_rows'])} monkey/task rows. Do not use this "
                "as the final reviewer-response value; rerun with raw base_dir "
                "for session-level inference."
            )
    lines.append("")
    lines.append(
        f"Collision definition: minimum avatar-obstacle distance <= {COLLISION_RADIUS_M:.2f} m."
    )
    statement_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {statement_path}")


def run_collision_corrected_success(
    monkeys: list[str],
    experiments: list[str],
    base_dir: str | None = None,
    use_cached_if_needed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_dir:
        df = compute_session_counterfactuals(monkeys, experiments, base_dir)
        if df.empty and use_cached_if_needed:
            df = compute_cached_unit_counterfactuals()
    elif use_cached_if_needed:
        df = compute_cached_unit_counterfactuals()
    else:
        df = pd.DataFrame()

    if df.empty:
        raise RuntimeError("No obstacle counterfactual rows could be computed.")

    summary = summarize_counterfactuals(df)
    write_outputs(df, summary, PROJECT_ROOT / "tables")
    return df, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run collision-corrected success counterfactual.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument(
        "--no-cached-fallback",
        action="store_true",
        help="Fail instead of using Variables/grand_summary.pkl when no raw base_dir is available.",
    )
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    base_dir = _load_base_dir(args.base_dir)
    run_collision_corrected_success(
        monkeys=args.monkeys,
        experiments=args.experiments,
        base_dir=base_dir,
        use_cached_if_needed=not args.no_cached_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
