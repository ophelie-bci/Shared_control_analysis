"""
Build Extended Data Figure 1 from released AI Obstacle trial data.

Panel A shows target-wise average trajectories for BCI-only and shared-control
trials. Panel B summarizes trajectory metrics from the cached analysis summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import target_mapping, target_to_obstacle_mapping
from src.load import load_files
from src.trajectory_metrics import compute_mean_squared_jerk, determine_avoidance_side


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "Extended Data Figure 1"
DEFAULT_OUTPUT_STEM = "extended_data_figure1_trajectory_supplement"
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENT = "AI Obstacle"
TARGET_ORDER = ["left", "right", "slight_left", "straight", "slight_right"]
TARGET_POSITIONS_BY_LABEL = {label: np.asarray(pos, float) for pos, label in target_mapping.items()}
TIME_STEP_SECONDS = 0.05
OBSTACLE_HALF_WIDTH = 0.45
AVATAR_RADIUS = 0.25
OBSTACLE_RADIUS = 0.45
OBSTACLE_PROXIMITY_RADIUS = 0.94
_LOAD_CACHE = {}


def load_base_dir(cli_base_dir: str | None = None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["base_dir"]


def _load_obstacle_files(monkey: str, base_dir: str):
    key = (base_dir, DEFAULT_EXPERIMENT, monkey)
    if key not in _LOAD_CACHE:
        _LOAD_CACHE[key] = load_files(DEFAULT_EXPERIMENT, monkey, base_dir)
    return _LOAD_CACHE[key]


def _trial_target_label(trial) -> str | None:
    pos = getattr(trial, "targetPosition", None)
    if pos is None:
        return None
    key = tuple(np.asarray(pos, float).round(2))
    return target_mapping.get(key)


def _trial_condition(trial) -> str | None:
    factor = getattr(trial, "aiVelocityFactor", np.nan)
    if factor == 1:
        return "Shared-control"
    if factor == 0:
        return "BCI-only"
    return None


def _trial_xz(trial) -> np.ndarray | None:
    traj = getattr(trial, "avatarTrajectory", None)
    if not isinstance(traj, dict) or "x" not in traj or "z" not in traj:
        return None
    x = np.asarray(traj["x"], float)
    z = np.asarray(traj["z"], float)
    n = min(len(x), len(z))
    if n < 4:
        return None
    pts = np.column_stack([x[:n], z[:n]])
    keep = np.isfinite(pts).all(axis=1)
    pts = pts[keep]
    return pts if len(pts) >= 4 else None


def _resample_arclength(points: np.ndarray, n_samples: int = 60) -> np.ndarray | None:
    if points is None or len(points) < 4:
        return None
    ds = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    s = np.r_[0.0, np.cumsum(ds)]
    if not np.isfinite(s[-1]) or s[-1] <= 0:
        return None
    s_new = np.linspace(0, s[-1], n_samples)
    return np.column_stack(
        [
            np.interp(s_new, s, points[:, 0]),
            np.interp(s_new, s, points[:, 1]),
        ]
    )


def _entry_index_square(points: np.ndarray, center: tuple[float, float], half_width: float) -> int | None:
    inside = (np.abs(points[:, 0] - center[0]) <= half_width) & (
        np.abs(points[:, 1] - center[1]) <= half_width
    )
    idx = np.flatnonzero(inside)
    return int(idx[0]) if idx.size else None


def _trim_before_entry(points: np.ndarray, target_center: tuple[float, float], obstacle_center: tuple[float, float]) -> np.ndarray:
    candidates = [
        _entry_index_square(points, target_center, 0.60),
        _entry_index_square(points, obstacle_center, 0.45),
    ]
    candidates = [idx for idx in candidates if idx is not None]
    if not candidates:
        return points
    stop = max(min(candidates) - 1, 3)
    return points[: stop + 1]


def collect_trajectory_rows(base_dir: str, monkeys: list[str], experiment: str) -> pd.DataFrame:
    rows = []
    for monkey in monkeys:
        all_trials, all_correct, _all_incorrect, *_ = _load_obstacle_files(monkey, base_dir)
        for session_idx, session_trials in enumerate(all_correct):
            for trial in session_trials:
                label = _trial_target_label(trial)
                condition = _trial_condition(trial)
                points = _trial_xz(trial)
                if label is None or condition is None or points is None:
                    continue
                target = TARGET_POSITIONS_BY_LABEL[label]
                target_center = (float(target[0]), float(target[2]))
                obstacle_center = target_to_obstacle_mapping[(round(target_center[0], 1), round(target_center[1], 1))]
                avoidance_side = determine_avoidance_side(
                    points[:, 0],
                    points[:, 1],
                    obstacle_center[0],
                    obstacle_center[1],
                )
                trimmed = _trim_before_entry(points, target_center, obstacle_center)
                resampled = _resample_arclength(trimmed)
                if resampled is None:
                    continue
                rows.append(
                    {
                        "monkey": monkey,
                        "session_index": session_idx,
                        "trial_id": getattr(trial, "trial", len(rows)),
                        "target_label": label,
                        "condition": condition,
                        "avoidance_side": avoidance_side,
                        "trajectory": resampled,
                    }
                )
    return pd.DataFrame(rows)


def _plot_square(ax, center: tuple[float, float], width: float, color: str, fill: bool = True):
    patch = Rectangle(
        (center[0] - width / 2, center[1] - width / 2),
        width,
        width,
        facecolor=color if fill else "none",
        edgecolor=color,
        linestyle="-" if fill else "--",
        linewidth=1.2,
    )
    ax.add_patch(patch)


def _plot_trajectory_panel(ax, df: pd.DataFrame, monkey: str, target_label: str):
    colors = {"BCI-only": "red", "Shared-control": "green"}
    target = TARGET_POSITIONS_BY_LABEL[target_label]
    target_center = (float(target[0]), float(target[2]))
    obstacle_center = target_to_obstacle_mapping[(round(target_center[0], 1), round(target_center[1], 1))]

    sub = df[(df["monkey"] == monkey) & (df["target_label"] == target_label)]
    for side in ["left", "right"]:
        for condition, color in colors.items():
            trajs = list(
                sub.loc[
                    (sub["condition"] == condition) & (sub["avoidance_side"] == side),
                    "trajectory",
                ]
            )
            if not trajs:
                continue
            stack = np.stack(trajs)
            center = np.nanmean(stack, axis=0)
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
            ax.plot(center[:, 0], center[:, 1], color=color, lw=1.45)
            if len(trajs) >= 3:
                ax.fill_betweenx(center[:, 1], q25[:, 0], q75[:, 0], color=color, alpha=0.08, linewidth=0)

    _plot_square(ax, obstacle_center, width=0.9, color="red", fill=True)
    _plot_square(ax, target_center, width=1.5, color="green", fill=True)
    ax.set_title(f"Target: {target_label}", fontsize=8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.grid(True, alpha=0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p == 0:
        return "p < 1 x 10^-300"
    exponent = int(np.floor(np.log10(abs(p))))
    mantissa = p / (10**exponent)
    return f"p = {mantissa:.2f} x 10^{exponent}"


def _fmt_mean_sem_delta(delta: float, sem: float, unit: str = "") -> str:
    if unit:
        return f"{delta:+.2f} +/- {sem:.2f} {unit}"
    return f"{delta:+.3f} +/- {sem:.3f}"


def _condition_label(trial) -> str | None:
    factor = getattr(trial, "aiVelocityFactor", np.nan)
    if factor == 1:
        return "Shared-control"
    if factor == 0:
        return "BCI-only"
    return None


def _trial_clearance(trial) -> float:
    points = _trial_xz(trial)
    if points is None:
        return np.nan
    obstacle = np.asarray([trial.obstaclePosition[0], trial.obstaclePosition[2]], dtype=float)
    return float(np.nanmin(np.linalg.norm(points - obstacle, axis=1)))


def _trial_obstacle_collision(trial) -> float:
    points = _trial_xz(trial)
    if points is None:
        return np.nan
    obstacle = np.asarray([trial.obstaclePosition[0], trial.obstaclePosition[2]], dtype=float)
    return float(np.nanmin(np.linalg.norm(points - obstacle, axis=1)) <= OBSTACLE_PROXIMITY_RADIUS)


def _trial_time_to_target(trial) -> float:
    traj = getattr(trial, "avatarTrajectory", None)
    if not isinstance(traj, dict) or "time" not in traj:
        points = _trial_xz(trial)
        return float(len(points) * TIME_STEP_SECONDS) if points is not None else np.nan

    time = np.asarray(traj["time"], dtype=float)
    if len(time) < 2:
        return np.nan

    model_velocity = getattr(trial, "modelVelocity", None)
    vz = np.asarray(model_velocity.get("vz", []), dtype=float) if isinstance(model_velocity, dict) else np.asarray([])
    nonzero = np.flatnonzero(vz)
    start_idx = int(nonzero[0]) if nonzero.size else 0
    start_idx = min(start_idx, len(time) - 1)
    return float((time[-1] - time[start_idx]) / 1000.0)


def _trial_log_msj(trial) -> float:
    points = _trial_xz(trial)
    if points is None or len(points) < 4:
        return np.nan
    msj = np.nanmean(
        [
            compute_mean_squared_jerk(points[:, 0], TIME_STEP_SECONDS),
            compute_mean_squared_jerk(points[:, 1], TIME_STEP_SECONDS),
        ]
    )
    norm_msj = msj / len(points)
    return float(np.log10(norm_msj)) if np.isfinite(norm_msj) and norm_msj > 0 else np.nan


def _collect_metric_session_rows(base_dir: str) -> pd.DataFrame:
    rows = []
    for monkey in DEFAULT_MONKEYS:
        _all_trials, all_correct, all_incorrect, *_ = _load_obstacle_files(monkey, base_dir)
        for session_idx, (correct_trials, incorrect_trials) in enumerate(zip(all_correct, all_incorrect)):
            for outcome, trials in [("Successful", correct_trials), ("Unsuccessful", incorrect_trials)]:
                for target_label in TARGET_ORDER:
                    target_trials = [trial for trial in trials if _trial_target_label(trial) == target_label]
                    if not target_trials:
                        continue
                    for condition in ["Shared-control", "BCI-only"]:
                        condition_trials = [trial for trial in target_trials if _condition_label(trial) == condition]
                        if not condition_trials:
                            continue
                        clearances = np.asarray([_trial_clearance(trial) for trial in condition_trials], dtype=float)
                        row = {
                            "monkey": monkey,
                            "session_index": session_idx,
                            "target_label": target_label,
                            "trial_type": outcome,
                            "condition": condition,
                            "n_trials": len(condition_trials),
                            "obstacle_clearance_m": float(np.nanmean(clearances)),
                            "collision_rate_percent": float(
                                100.0 * np.nanmean([_trial_obstacle_collision(trial) for trial in condition_trials])
                            ),
                        }
                        if outcome == "Successful":
                            row["time_to_target_s"] = float(
                                np.nanmean([_trial_time_to_target(trial) for trial in condition_trials])
                            )
                            row["log_msj"] = float(np.nanmean([_trial_log_msj(trial) for trial in condition_trials]))
                        rows.append(row)
    return pd.DataFrame(rows)


def _paired_metric(session_df: pd.DataFrame, metric: str, trial_type: str, unit: str = "") -> tuple[str, str]:
    sub = session_df[session_df["trial_type"] == trial_type]
    pivot = sub.pivot_table(index=["monkey", "session_index", "target_label"], columns="condition", values=metric)
    if not {"Shared-control", "BCI-only"}.issubset(pivot.columns):
        return "n/a", "n/a"
    pivot = pivot[["Shared-control", "BCI-only"]].replace([np.inf, -np.inf], np.nan).dropna()
    if pivot.empty:
        return "n/a", "n/a"
    delta = pivot["Shared-control"] - pivot["BCI-only"]
    try:
        p_value = stats.wilcoxon(
            pivot["Shared-control"],
            pivot["BCI-only"],
            alternative="two-sided",
            zero_method="wilcox",
            mode="auto",
        ).pvalue
    except ValueError:
        p_value = np.nan
    if unit == "percentage points":
        delta_text = f"{delta.mean():+.1f} +/- {delta.sem():.1f} {unit}"
    else:
        delta_text = _fmt_mean_sem_delta(float(delta.mean()), float(delta.sem()), unit)
    return delta_text, _fmt_p(p_value)


def _metric_delta_table(base_dir: str, source_path: Path | None = None) -> pd.DataFrame:
    session_df = _collect_metric_session_rows(base_dir)
    if source_path is not None:
        session_df.to_csv(source_path, index=False, sep=";")

    rows = []
    specs = [
        ("Time-to-target (s)", "Successful", "time_to_target_s", "s"),
        ("Obstacle clearance", "Successful", "obstacle_clearance_m", "m"),
        ("Obstacle clearance", "Unsuccessful", "obstacle_clearance_m", "m"),
        ("Collision rate", "Successful", "collision_rate_percent", "percentage points"),
        ("Collision rate", "Unsuccessful", "collision_rate_percent", "percentage points"),
        ("log(MSJ)", "Successful", "log_msj", ""),
    ]
    for metric_label, trial_type, column, unit in specs:
        delta_text, p_text = _paired_metric(session_df, column, trial_type, unit)
        rows.append(
            {
                "Metric": metric_label,
                "Trial Type": trial_type,
                "Delta (Shared-control - BCI-only)": delta_text,
                "Wilcoxon signed-rank p-value": p_text,
            }
        )
    return pd.DataFrame(rows)


def _add_panel(fig, parent_spec, df: pd.DataFrame, monkey: str, label_x: float):
    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=parent_spec,
        height_ratios=[1.0, 1.0],
        wspace=0.34,
        hspace=0.36,
    )
    axes = [
        fig.add_subplot(inner[0, 0]),
        fig.add_subplot(inner[0, 1]),
        fig.add_subplot(inner[1, 0]),
        fig.add_subplot(inner[1, 1]),
        fig.add_subplot(inner[1, 2]),
    ]
    for ax, target in zip(axes, TARGET_ORDER):
        _plot_trajectory_panel(ax, df, monkey, target)
    fig.text(label_x, 0.93, monkey, ha="center", fontsize=13, weight="bold")
    return axes


def build_extended_data_figure1(base_dir: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    df = collect_trajectory_rows(base_dir, DEFAULT_MONKEYS, DEFAULT_EXPERIMENT)
    if df.empty:
        raise RuntimeError("No trajectory rows were collected for Extended Data Figure 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_df = _metric_delta_table(base_dir)

    fig = plt.figure(figsize=(14, 11))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.48], hspace=0.22, wspace=0.18)
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, :], wspace=0.16)
    _add_panel(fig, top[0, 0], df, DEFAULT_MONKEYS[0], 0.255)
    _add_panel(fig, top[0, 1], df, DEFAULT_MONKEYS[1], 0.745)

    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="BCI-only (mean)"),
        Patch(facecolor="red", alpha=0.08, label="BCI-only (IQR)"),
        Line2D([0], [0], color="green", lw=2, label="Shared-control (mean)"),
        Patch(facecolor="green", alpha=0.08, label="Shared-control (IQR)"),
        Patch(facecolor="green", edgecolor="green", label="Target"),
        Patch(facecolor="red", edgecolor="red", label="Obstacle"),
    ]
    fig.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.5, 0.43), ncol=3, frameon=True, fontsize=8)

    ax_table = fig.add_subplot(outer[1, :])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.22, 0.17, 0.27, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.4)
        else:
            cell.set_linewidth(0.8)

    fig.text(0.02, 0.965, "A", fontsize=12, weight="bold", va="top")
    fig.text(0.02, 0.395, "B", fontsize=12, weight="bold", va="top")

    outputs = []
    path = output_dir / f"{DEFAULT_OUTPUT_STEM}.svg"
    fig.savefig(path, bbox_inches="tight")
    outputs.append(path)
    print(f"Saved Extended Data Figure 1: {path}")
    plt.close(fig)

    provenance = output_dir / f"{DEFAULT_OUTPUT_STEM}_provenance.txt"
    provenance.write_text(
        "\n".join(
            [
                "Data-driven Extended Data Figure 1.",
                "Base data directory: config.yaml base_dir",
                f"Experiment: {DEFAULT_EXPERIMENT}",
                f"Monkeys: {', '.join(DEFAULT_MONKEYS)}",
                "Metric table is embedded in the figure; no standalone Extended Data Figure 1 table files are written.",
                "Metric pairing unit: session x target.",
                f"Collision-rate definition: previous-script obstacle proximity proxy; min distance to obstacle <= {OBSTACLE_PROXIMITY_RADIUS} m",
                "Time-to-target definition: final trajectory time minus first nonzero modelVelocity vz bin.",
                "Smoothness definition: log10(mean squared jerk normalized by number of time points)",
                "P-value definition: two-sided paired Wilcoxon signed-rank test, zero_method='wilcox', mode='auto'.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(provenance)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-driven Extended Data Figure 1.")
    parser.add_argument("--base-dir", default=None, help="Root data directory. Defaults to config.yaml.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Destination figure directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    return build_extended_data_figure1(load_base_dir(args.base_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
