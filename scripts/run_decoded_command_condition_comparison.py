"""
Compare raw decoded BCI commands between BCI-only and shared-control trials.

The decoder is fixed within a session, so this control asks whether the monkey
generated fundamentally different decoded commands when assistance was active.
For both conditions the script analyzes `modelVelocity`, i.e. the decoded BCI
command before RT-V2 modifies the executed velocity.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = None
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
EXPERIMENT_LABELS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle follow-up",
}
CONDITION_LABELS = {0.0: "BCI-only", 1.0: "Shared-control"}
TABLE_DIR = PROJECT_ROOT / "tables" / "Decoded command comparison"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Decoded command comparison"
MIN_SPEED = 1e-6


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _vec3(value) -> np.ndarray:
    if value is None:
        return np.full(3, np.nan)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(3, np.nan)
    out = np.full(3, np.nan)
    out[: min(3, len(arr))] = arr[:3]
    return out


def _trial_id(trial) -> int | float:
    value = _get(trial, "trial", np.nan)
    try:
        return int(value)
    except Exception:
        return np.nan


def _load_trials(path: Path) -> list:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj[1] if isinstance(obj, (list, tuple)) and len(obj) >= 2 else obj


def _velocity_arrays(trial) -> tuple[np.ndarray, np.ndarray]:
    velocity = _get(trial, "modelVelocity", None)
    if not isinstance(velocity, dict):
        return np.zeros((0, 2)), np.array([])
    try:
        vx = np.asarray(velocity["vx"], dtype=float)
        vz = np.asarray(velocity["vz"], dtype=float)
        t = np.asarray(velocity.get("time", np.arange(len(vx))), dtype=float)
    except Exception:
        return np.zeros((0, 2)), np.array([])
    n = min(len(vx), len(vz), len(t))
    return np.column_stack([vx[:n], vz[:n]]), t[:n]


def _interp_positions(trial, sample_times: np.ndarray) -> np.ndarray:
    traj = _get(trial, "avatarTrajectory", {})
    if not isinstance(traj, dict):
        return np.full((len(sample_times), 2), np.nan)
    try:
        tt = np.asarray(traj["time"], dtype=float)
        x = np.asarray(traj["x"], dtype=float)
        z = np.asarray(traj["z"], dtype=float)
    except Exception:
        return np.full((len(sample_times), 2), np.nan)
    n = min(len(tt), len(x), len(z))
    if n < 2:
        return np.full((len(sample_times), 2), np.nan)
    tt, x, z = tt[:n], x[:n], z[:n]
    order = np.argsort(tt)
    tt, x, z = tt[order], x[order], z[order]
    return np.column_stack(
        [
            np.interp(sample_times, tt, x, left=np.nan, right=np.nan),
            np.interp(sample_times, tt, z, left=np.nan, right=np.nan),
        ]
    )


def _unit2(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norm, MIN_SPEED, None)


def _time_scale_to_seconds(t: np.ndarray) -> float:
    finite = t[np.isfinite(t)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size and np.nanmedian(diffs) > 5.0:
        return 0.001
    return 1.0


def _target_label(trial) -> str:
    target = _vec3(_get(trial, "targetPosition", None))
    key = tuple(np.round(target, 3))
    mapping = {
        (-7.0, 0.75, 6.0): "left",
        (-3.5, 0.75, 8.5): "slight_left",
        (0.0, 0.75, 9.2): "straight",
        (3.5, 0.75, 8.5): "slight_right",
        (7.0, 0.75, 6.0): "right",
    }
    return mapping.get(key, "unknown")


def _trial_metrics(trial, monkey: str, experiment: str, session_index: int, session_file: str) -> dict | None:
    try:
        factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None
    if factor not in CONDITION_LABELS:
        return None

    vel, t = _velocity_arrays(trial)
    if len(vel) < 4:
        return None
    pos = _interp_positions(trial, t)
    target_xz = _vec3(_get(trial, "targetPosition", None))[[0, 2]]
    target_dir = _unit2(target_xz.reshape(1, 2) - pos)
    speed = np.linalg.norm(vel, axis=1)
    keep = (
        np.isfinite(vel).all(axis=1)
        & np.isfinite(target_dir).all(axis=1)
        & (speed > MIN_SPEED)
    )
    if keep.sum() < 3:
        return None

    unit_vel = _unit2(vel)
    cos_target = np.sum(unit_vel * target_dir, axis=1)
    cos_target = np.clip(cos_target, -1.0, 1.0)
    angle_error = np.degrees(np.arccos(cos_target))
    perp = np.column_stack([-target_dir[:, 1], target_dir[:, 0]])
    lateral = np.sum(vel * perp, axis=1)
    parallel = np.sum(vel * target_dir, axis=1)

    scale = _time_scale_to_seconds(t)
    dt = np.diff(t) * scale
    good_dt = np.isfinite(dt) & (dt > 0)
    dvel = np.diff(vel, axis=0)
    accel = np.full(len(dvel), np.nan)
    accel[good_dt] = np.linalg.norm(dvel[good_dt], axis=1) / dt[good_dt]

    return {
        "monkey": monkey,
        "experiment": experiment,
        "task_family": EXPERIMENT_LABELS[experiment],
        "session_index": session_index,
        "session_uid": f"{monkey}|{experiment}|{session_index}",
        "session_file": session_file,
        "trial_id": _trial_id(trial),
        "condition": CONDITION_LABELS[factor],
        "shared_control": int(factor == 1.0),
        "success": int(_get(trial, "answer", None) == 1),
        "target": _target_label(trial),
        "n_command_samples": int(keep.sum()),
        "decoded_speed_mean": float(np.nanmean(speed[keep])),
        "decoded_speed_median": float(np.nanmedian(speed[keep])),
        "angular_error_to_target_deg_mean": float(np.nanmean(angle_error[keep])),
        "angular_error_to_target_deg_median": float(np.nanmedian(angle_error[keep])),
        "lateral_command_abs_mean": float(np.nanmean(np.abs(lateral[keep]))),
        "lateral_command_rms": float(np.sqrt(np.nanmean(lateral[keep] ** 2))),
        "target_parallel_command_mean": float(np.nanmean(parallel[keep])),
        "command_heading_variability_deg": float(stats.circstd(np.deg2rad(angle_error[keep]), high=np.pi, low=0) * 180.0 / np.pi),
        "command_speed_sd": float(np.nanstd(speed[keep], ddof=1)),
        "command_acceleration_mean": float(np.nanmean(accel)),
    }


def collect_metrics(base_dir: str, monkeys: list[str], experiments: list[str]) -> pd.DataFrame:
    rows = []
    for experiment in experiments:
        if experiment not in EXPERIMENT_LABELS:
            continue
        for monkey in monkeys:
            data_dir = Path(base_dir) / monkey / experiment
            if not data_dir.exists():
                print(f"[INFO] skipping missing {data_dir}")
                continue
            files = sorted(data_dir.glob("*_trials.pkl"))
            print(f"[INFO] {monkey} {experiment}: {len(files)} sessions")
            for session_index, path in enumerate(files):
                trials = _load_trials(path)
                session_rows = [
                    row for trial in trials
                    if (row := _trial_metrics(trial, monkey, experiment, session_index, path.name)) is not None
                ]
                rows.extend(session_rows)
                print(f"  session {session_index}: {len(session_rows)} trials")
    return pd.DataFrame(rows)


def session_condition_summary(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "decoded_speed_mean",
        "angular_error_to_target_deg_mean",
        "lateral_command_abs_mean",
        "lateral_command_rms",
        "command_heading_variability_deg",
        "command_speed_sd",
        "command_acceleration_mean",
    ]
    return (
        trials.groupby(["task_family", "monkey", "experiment", "session_index", "session_uid", "condition"])[metrics]
        .mean()
        .reset_index()
    )


def paired_summary(session_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "decoded_speed_mean",
        "angular_error_to_target_deg_mean",
        "lateral_command_abs_mean",
        "lateral_command_rms",
        "command_heading_variability_deg",
        "command_speed_sd",
        "command_acceleration_mean",
    ]
    rows = []
    for metric in metrics:
        wide = session_summary.pivot_table(
            index=["task_family", "monkey", "experiment", "session_index", "session_uid"],
            columns="condition",
            values=metric,
            aggfunc="first",
        ).reset_index()
        if not {"BCI-only", "Shared-control"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["BCI-only", "Shared-control"]).copy()
        wide["delta_shared_minus_bci"] = wide["Shared-control"] - wide["BCI-only"]
        for task_family, group in wide.groupby("task_family"):
            delta = group["delta_shared_minus_bci"].to_numpy(float)
            try:
                wil = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
                w_stat, p_value = float(wil.statistic), float(wil.pvalue)
            except Exception:
                w_stat, p_value = np.nan, np.nan
            rows.append(
                {
                    "task_family": task_family,
                    "metric": metric,
                    "n_sessions": int(len(group)),
                    "bci_only_mean": float(group["BCI-only"].mean()),
                    "shared_control_mean": float(group["Shared-control"].mean()),
                    "shared_minus_bci_mean": float(delta.mean()),
                    "shared_minus_bci_median": float(np.median(delta)),
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_two_sided_p": p_value,
                    "n_sessions_shared_higher": int((delta > 0).sum()),
                    "n_sessions_shared_lower": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("decoded_speed_mean", "Speed"),
        ("angular_error_to_target_deg_mean", "Angular error"),
        ("lateral_command_abs_mean", "Lateral magnitude"),
        ("command_acceleration_mean", "Command roughness"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.0, 3.0), dpi=220)
    for ax, (metric, title) in zip(axes, metrics):
        sub = summary[summary["metric"].eq(metric)]
        x = np.arange(len(sub))
        ax.bar(x, sub["shared_minus_bci_mean"], color="#5b7c99")
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_xticks(x, sub["task_family"], rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Shared - BCI-only")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(FIGURE_DIR / f"decoded_command_condition_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Decoded-command comparison between conditions",
        "",
        "For both BCI-only and shared-control trials, metrics were computed from `modelVelocity`, the raw decoded BCI command before RT-V2 modifies the executed velocity.",
        "",
        "## Session-level paired comparisons",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['metric']}: BCI-only {row['bci_only_mean']:.3g}, "
            f"shared-control {row['shared_control_mean']:.3g}, "
            f"shared-minus-BCI {row['shared_minus_bci_mean']:.3g}; "
            f"Wilcoxon p={row['wilcoxon_two_sided_p']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Suggested reviewer wording",
            "",
            "We directly compared the raw decoded BCI commands before RT-V2 modification across interleaved BCI-only and shared-control trials. This diagnostic did not support a simple claim that decoded commands were identical across conditions: Fixed Obstacle trials showed larger condition-related command differences, while Appearing Obstacle trials showed more modest differences, with angular error largely similar in the original appearing-obstacle sessions and speed/lateral-command differences much smaller than the behavioral success gain. We therefore treat this as a diagnostic control rather than the primary rebuttal; the stronger evidence comes from analyses showing that RT-V2 continued to use instantaneous BCI input outside safety overrides and that follow-up reactive obstacle metrics improved when no hidden-obstacle prior was available.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    base_dir: str,
    monkeys: list[str],
    experiments: list[str],
    trial_metrics_input: list[str] | None = None,
    trial_metrics_output: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if trial_metrics_input:
        trials = pd.concat([pd.read_csv(path) for path in trial_metrics_input], ignore_index=True)
    else:
        trials = collect_metrics(base_dir, monkeys, experiments)
    if trials.empty:
        raise RuntimeError("No decoded-command trial metrics were found.")

    trial_path = Path(trial_metrics_output).resolve() if trial_metrics_output else TABLE_DIR / "decoded_command_trial_metrics.csv"
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(trial_path, index=False)
    if trial_path != TABLE_DIR / "decoded_command_trial_metrics.csv":
        trials.to_csv(TABLE_DIR / "decoded_command_trial_metrics.csv", index=False)

    session_summary = session_condition_summary(trials)
    summary = paired_summary(session_summary)
    session_summary.to_csv(TABLE_DIR / "decoded_command_session_metrics.csv", index=False)
    summary.to_csv(TABLE_DIR / "decoded_command_condition_summary.csv", index=False)
    plot_summary(summary)
    write_report(summary, TABLE_DIR / "decoded_command_condition_report.md")
    return trials, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--trial-metrics-input", nargs="+", default=None)
    parser.add_argument("--trial-metrics-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, summary = run_analysis(
        load_base_dir(args.base_dir),
        args.monkeys,
        args.experiments,
        trial_metrics_input=args.trial_metrics_input,
        trial_metrics_output=args.trial_metrics_output,
    )
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
