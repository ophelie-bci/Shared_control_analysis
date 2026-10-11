"""
Build representative decoder-direction error examples for Figure 4.

This figure complements the execution-error examples by showing failed trials
where the decoded modelVelocity points to an adjacent target, an incorrect
target, or is globally ambiguous. The examples are selected from raw trials
using the same 60% decoded-choice rule described in the Figure 4 code.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AI_metrics import ai_on_indices_from_trial
from src.constants import target_mapping, target_to_obstacle_mapping
from src.load import load_data, load_files


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "Extended Data Figure 3"
DEFAULT_OUTPUT_STEM = "extended_data_figure3_decoder_direction_examples"
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
TARGET_HALF_WIDTH = 2.10
DRAW_TARGET_WIDTH = 1.50
OBSTACLE_WIDTH = 0.90
AVATAR_RADIUS = 0.25
OBSTACLE_RADIUS = 0.45
STUCK_MARGIN = 0.20
CHOICE_THRESH = 0.60
SPEED_THRESH = 0.10

TARGETS = {
    label: np.asarray(pos, dtype=float)[[0, 2]]
    for pos, label in target_mapping.items()
}
TARGET_ORDER = ["left", "slight_left", "straight", "slight_right", "right"]
KNOWN_GOALS = [tuple(TARGETS[label]) for label in ["right", "slight_right", "straight", "slight_left", "left"]]
GOAL_IDX = {goal: i for i, goal in enumerate(KNOWN_GOALS)}
NEIGHBORS = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
GOAL_TO_LABEL = {tuple(v): k for k, v in TARGETS.items()}


@dataclass(frozen=True)
class TrialRecord:
    panel: str
    monkey: str
    experiment: str
    session_index: int
    session_file: str
    trial_id: int | str
    condition: str
    target_label: str
    choice_state: str
    choice_label: str
    best_goal_label: str
    true_frac: float
    neighbor_frac: float
    other_frac: float
    best_frac: float
    points: np.ndarray
    bci_velocity: np.ndarray
    actual_velocity: np.ndarray
    obstacle_center: tuple[float, float]
    min_obstacle_distance: float
    final_distance_true: float
    max_consecutive_target_frames: int
    max_z_minus_target_z_m: float
    score: float


def load_base_dir(cli_base_dir: str | None = None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        base_dir = (yaml.safe_load(f) or {}).get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def _trial_id(trial) -> int | str:
    value = getattr(trial, "trial", None)
    try:
        return int(value)
    except Exception:
        return "unknown" if value is None else str(value)


def _condition_label(trial) -> str | None:
    factor = getattr(trial, "aiVelocityFactor", np.nan)
    if factor == 1:
        return "Shared-control"
    if factor == 0:
        return "BCI-only"
    return None


def _position_from_ai_record(rec: dict) -> tuple[float, float] | None:
    pos = rec.get("AvatarPosition")
    if pos is None or isinstance(pos, (float, int, np.floating, np.integer)):
        return None
    if isinstance(pos, dict):
        x, z = pos.get("x"), pos.get("z")
    else:
        try:
            if len(pos) < 3:
                return None
            x, z = pos[0], pos[2]
        except TypeError:
            return None
    if x is None or z is None:
        return None
    x, z = float(x), float(z)
    if not np.isfinite(x) or not np.isfinite(z):
        return None
    return x, z


def _ai_log_xz(trial) -> np.ndarray | None:
    recs = getattr(trial, "aiVelocities", None)
    if not recs:
        return None
    points = []
    for rec in recs:
        pos = _position_from_ai_record(rec)
        if pos is not None:
            points.append(pos)
    if len(points) < 4:
        return None
    return np.asarray(points, dtype=float)


def _trial_xz(trial) -> np.ndarray | None:
    ai_points = _ai_log_xz(trial)
    if ai_points is not None:
        return ai_points
    traj = getattr(trial, "avatarTrajectory", None)
    if not isinstance(traj, dict) or "x" not in traj or "z" not in traj:
        return None
    x = np.asarray(traj["x"], dtype=float)
    z = np.asarray(traj["z"], dtype=float)
    n = min(len(x), len(z))
    if n < 4:
        return None
    points = np.column_stack([x[:n], z[:n]])
    keep = np.isfinite(points).all(axis=1)
    points = points[keep]
    return points if len(points) >= 4 else None


def _velocity_xz(trial, field: str) -> np.ndarray:
    velocity = getattr(trial, field, None)
    traj = getattr(trial, "avatarTrajectory", None)
    n_points = 0
    if isinstance(traj, dict) and "x" in traj and "z" in traj:
        n_points = min(len(traj["x"]), len(traj["z"]))
    if not isinstance(velocity, dict) or "vx" not in velocity or "vz" not in velocity or n_points == 0:
        return np.full((n_points, 2), np.nan)
    vx = np.asarray(velocity["vx"], dtype=float)
    vz = np.asarray(velocity["vz"], dtype=float)
    n = min(n_points, len(vx), len(vz))
    out = np.full((n_points, 2), np.nan)
    out[:n, 0] = vx[:n]
    out[:n, 1] = vz[:n]
    return out


def _ai_log_velocity_xz(trial, which: str) -> np.ndarray | None:
    try:
        _ai_idxs, inputs, outputs, _time_info = ai_on_indices_from_trial(trial)
    except Exception:
        return None
    values = inputs if which == "input" else outputs
    if values is None or len(values) == 0:
        return None
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3:
        return None
    traj = getattr(trial, "avatarTrajectory", None)
    n_points = 0
    if isinstance(traj, dict) and "x" in traj and "z" in traj:
        n_points = min(len(traj["x"]), len(traj["z"]))
    n = min(n_points, len(values))
    if n == 0:
        return None
    out = np.full((n_points, 2), np.nan)
    out[:n, 0] = values[:n, 0]
    out[:n, 1] = values[:n, 2]
    return out


def _mean_velocity(velocity: np.ndarray) -> tuple[float, float]:
    if velocity.size == 0:
        return np.nan, np.nan
    speed = np.linalg.norm(velocity, axis=1)
    valid = np.isfinite(speed) & (speed > SPEED_THRESH)
    if not np.any(valid):
        return np.nan, np.nan
    mean = np.nanmean(velocity[valid], axis=0)
    return float(mean[0]), float(mean[1])


def _target_label(trial) -> str | None:
    pos = getattr(trial, "targetPosition", None)
    if pos is None:
        return None
    xz = np.asarray(pos, dtype=float)[[0, 2]]
    label, dist = min(
        ((label, float(np.linalg.norm(xz - center))) for label, center in TARGETS.items()),
        key=lambda item: item[1],
    )
    return label if dist < 0.35 else None


def _obstacle_center(target_label: str, trial) -> tuple[float, float] | None:
    obstacle_pos = getattr(trial, "obstaclePosition", None)
    if obstacle_pos is not None:
        try:
            obstacle = np.asarray([obstacle_pos[0], obstacle_pos[2]], dtype=float)
            if np.isfinite(obstacle).all():
                return float(obstacle[0]), float(obstacle[1])
        except Exception:
            pass
    target = TARGETS[target_label]
    return target_to_obstacle_mapping.get((round(float(target[0]), 1), round(float(target[1]), 1)))


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _target_window_mask(points: np.ndarray, target_label: str) -> np.ndarray:
    target = TARGETS[target_label]
    return (
        (np.abs(points[:, 0] - target[0]) <= TARGET_HALF_WIDTH)
        & (np.abs(points[:, 1] - target[1]) <= TARGET_HALF_WIDTH)
    )


def _is_execution_failure(points: np.ndarray, target_label: str, obstacle_center: tuple[float, float]) -> bool:
    obstacle = np.asarray(obstacle_center, dtype=float)
    inflated_half = OBSTACLE_RADIUS + AVATAR_RADIUS + STUCK_MARGIN
    inside_obstacle = (
        (np.abs(points[:, 0] - obstacle[0]) <= inflated_half)
        & (np.abs(points[:, 1] - obstacle[1]) <= inflated_half)
    )
    blocked = np.any(inside_obstacle) and not bool(np.nanmax(points[:, 1]) > obstacle[1] + inflated_half)
    inside_target = _target_window_mask(points, target_label)
    target_z = float(TARGETS[target_label][1])
    overshoot = float(np.nanmax(points[:, 1])) > target_z and _max_consecutive_true(inside_target) < 10
    insufficient_proximity = not np.any(inside_target) and float(np.nanmax(points[:, 1])) <= target_z
    insufficient_dwell = 0 < _max_consecutive_true(inside_target) < 10
    return bool(blocked or overshoot or insufficient_proximity or insufficient_dwell)


def _choice_state(trial, target_label: str) -> dict | None:
    mv = getattr(trial, "modelVelocity", None)
    traj = getattr(trial, "avatarTrajectory", None)
    if not isinstance(mv, dict) or not isinstance(traj, dict):
        return None
    try:
        vx = np.asarray(mv["vx"], dtype=float)
        vz = np.asarray(mv["vz"], dtype=float)
        xs = np.asarray(traj["x"], dtype=float)
        zs = np.asarray(traj["z"], dtype=float)
    except Exception:
        return None
    n = min(len(vx), len(vz), len(xs), len(zs))
    if n < 5:
        return None
    vx, vz, xs, zs = vx[:n], vz[:n], xs[:n], zs[:n]
    vectors = np.column_stack([vx, vz])
    speed = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(speed) & (speed > SPEED_THRESH)
    if not np.any(valid):
        return None
    vectors = vectors[valid]
    xs = xs[valid]
    zs = zs[valid]
    counts = {goal: 0 for goal in KNOWN_GOALS}
    for vec, x, z in zip(vectors, xs, zs):
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-9:
            continue
        best_goal = None
        best_cos = -np.inf
        for goal in KNOWN_GOALS:
            goal_vec = np.asarray(goal, dtype=float) - np.asarray([x, z], dtype=float)
            goal_norm = np.linalg.norm(goal_vec)
            if goal_norm < 1e-9:
                continue
            cos = float(np.dot(vec, goal_vec) / (vec_norm * goal_norm + 1e-12))
            if cos > best_cos:
                best_cos = cos
                best_goal = goal
        if best_goal is not None and best_cos > 0:
            counts[best_goal] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    fracs = {goal: count / total for goal, count in counts.items() if count > 0}
    true_goal = tuple(TARGETS[target_label])
    best_goal = max(fracs, key=fracs.get)
    best_frac = float(fracs[best_goal])
    true_frac = float(fracs.get(true_goal, 0.0))
    choice_label = "ambiguous_choice" if best_frac < CHOICE_THRESH else (
        "correct_choice" if best_goal == true_goal else "wrong_choice"
    )
    neighbor_frac = 0.0
    other_frac = 0.0
    neighbor_rel = 0.0
    if true_goal in GOAL_IDX:
        neighbor_goals = [KNOWN_GOALS[i] for i in NEIGHBORS[GOAL_IDX[true_goal]]]
        neighbor_frac = float(sum(fracs.get(goal, 0.0) for goal in neighbor_goals))
        other_frac = float(max(0.0, 1.0 - true_frac - neighbor_frac))
        neighbor_rel = neighbor_frac / max(1e-9, 1.0 - true_frac)
    if choice_label == "ambiguous_choice":
        state = "neighbor_choice" if true_goal in GOAL_IDX and neighbor_rel >= 0.8 else "ambiguous_choice"
    else:
        state = choice_label
    return {
        "choice_state": state,
        "choice_label": choice_label,
        "best_goal_label": GOAL_TO_LABEL.get(best_goal, "unknown"),
        "true_frac": true_frac,
        "neighbor_frac": neighbor_frac,
        "other_frac": other_frac,
        "best_frac": best_frac,
    }


def _record_from_trial(
    panel: str,
    trial,
    monkey: str,
    experiment: str,
    session_index: int,
    session_file: str,
    score: float,
) -> TrialRecord | None:
    points = _trial_xz(trial)
    target_label = _target_label(trial)
    condition = _condition_label(trial)
    if points is None or target_label is None or condition is None:
        return None
    obstacle = _obstacle_center(target_label, trial)
    choice = _choice_state(trial, target_label)
    if obstacle is None or choice is None:
        return None
    obstacle_arr = np.asarray(obstacle, dtype=float)
    min_obstacle_distance = float(np.nanmin(np.linalg.norm(points - obstacle_arr, axis=1)))
    final_distance_true = float(np.linalg.norm(points[-1] - TARGETS[target_label]))
    return TrialRecord(
        panel=panel,
        monkey=monkey,
        experiment=experiment,
        session_index=session_index,
        session_file=session_file,
        trial_id=_trial_id(trial),
        condition=condition,
        target_label=target_label,
        choice_state=choice["choice_state"],
        choice_label=choice["choice_label"],
        best_goal_label=choice["best_goal_label"],
        true_frac=choice["true_frac"],
        neighbor_frac=choice["neighbor_frac"],
        other_frac=choice["other_frac"],
        best_frac=choice["best_frac"],
        points=points,
        bci_velocity=(
            _ai_log_velocity_xz(trial, "input")
            if _ai_log_velocity_xz(trial, "input") is not None
            else _velocity_xz(trial, "modelVelocity")
        ),
        actual_velocity=(
            _ai_log_velocity_xz(trial, "output")
            if _ai_log_velocity_xz(trial, "output") is not None
            else _velocity_xz(trial, "avatarVelocity")
        ),
        obstacle_center=(float(obstacle[0]), float(obstacle[1])),
        min_obstacle_distance=min_obstacle_distance,
        final_distance_true=final_distance_true,
        max_consecutive_target_frames=_max_consecutive_true(_target_window_mask(points, target_label)),
        max_z_minus_target_z_m=float(np.nanmax(points[:, 1]) - TARGETS[target_label][1]),
        score=float(score),
    )


def select_examples(base_dir: str, monkeys: list[str], experiments: list[str]) -> list[TrialRecord]:
    wanted = {
        "Adjacent decoded choice": "neighbor_choice",
        "Ambiguous decoded choice": "ambiguous_choice",
        "Incorrect decoded choice": "wrong_choice",
    }
    best: dict[str, TrialRecord] = {}
    for experiment in experiments:
        for monkey in monkeys:
            all_trials, _all_correct, all_incorrect, *_rest, pkl_files, _ai_trials = load_files(
                experiment,
                monkey,
                base_dir,
            )
            if not all_trials:
                continue
            for session_index, incorrect_trials in enumerate(all_incorrect):
                session_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                for trial in incorrect_trials:
                    points = _trial_xz(trial)
                    target_label = _target_label(trial)
                    if points is None or target_label is None:
                        continue
                    obstacle = _obstacle_center(target_label, trial)
                    if obstacle is None or _is_execution_failure(points, target_label, obstacle):
                        continue
                    choice = _choice_state(trial, target_label)
                    if choice is None:
                        continue
                    if _condition_label(trial) != "Shared-control":
                        continue
                    for panel, state in wanted.items():
                        if choice["choice_state"] != state:
                            continue
                        if state == "neighbor_choice":
                            score = choice["neighbor_frac"] - choice["true_frac"]
                        elif state == "ambiguous_choice":
                            score = 1.0 - choice["best_frac"]
                        else:
                            score = choice["best_frac"] - choice["true_frac"]
                        record = _record_from_trial(panel, trial, monkey, experiment, session_index, session_file, score)
                        if record is not None and (panel not in best or record.score > best[panel].score):
                            best[panel] = record
            if len(best) == len(wanted):
                # Usually enough after the first sessions; keep this quick for visual iteration.
                continue
    missing = [panel for panel in wanted if panel not in best]
    if missing:
        raise RuntimeError(f"No decoder-direction example found for: {', '.join(missing)}")
    return [best[panel] for panel in wanted]


def select_examples_from_provenance(base_dir: str, provenance_path: Path) -> list[TrialRecord]:
    rows = pd.read_csv(provenance_path, sep=";").to_dict("records")
    examples: list[TrialRecord] = []
    for row in rows:
        all_trials, *_rest, pkl_files, _ai_trials = load_files(str(row["experiment"]), str(row["monkey"]), base_dir)
        session_file = str(row["session_file"])
        session_index = next(
            (i for i, path in enumerate(pkl_files) if Path(path).name == session_file),
            int(row["session_index"]),
        )
        trials = all_trials[session_index]
        trial_id = int(row["trial_id"])
        match = next((trial for trial in trials if _trial_id(trial) == trial_id), None)
        if match is None:
            raise RuntimeError(f"Could not find trial {trial_id} in {session_file}")
        record = _record_from_trial(
            str(row["panel"]),
            match,
            str(row["monkey"]),
            str(row["experiment"]),
            int(row["session_index"]),
            str(row["session_file"]),
            float(row["selection_score"]),
        )
        if record is None:
            raise RuntimeError(f"Could not rebuild record for {row['panel']}")
        examples.append(record)
    return examples


def _plot_square(ax, center: np.ndarray | tuple[float, float], width: float, **kwargs) -> None:
    center = np.asarray(center, dtype=float)
    ax.add_patch(Rectangle((center[0] - width / 2, center[1] - width / 2), width, width, **kwargs))


def _plot_trial(ax, record: TrialRecord) -> None:
    for label in TARGET_ORDER:
        target = TARGETS[label]
        is_true = label == record.target_label
        is_best = label == record.best_goal_label and record.choice_state != "ambiguous_choice"
        _plot_square(
            ax,
            target,
            TARGET_HALF_WIDTH * 2 if is_true else DRAW_TARGET_WIDTH,
            facecolor="#4daf4a" if is_true else "#f2f2f2",
            edgecolor="none" if is_true else "#a8a8a8",
            linewidth=0.0 if is_true else 0.8,
            alpha=0.12 if is_true else 0.45,
            zorder=0,
        )
        if is_true:
            _plot_square(
                ax,
                target,
                DRAW_TARGET_WIDTH,
                facecolor="none",
                edgecolor="#2ca02c",
                linewidth=1.0,
                alpha=1.0,
                zorder=3,
            )
            ax.scatter(target[0], target[1], s=40, marker="x", color="#2f7d32", lw=1.4, zorder=4)
        else:
            ax.scatter(target[0], target[1], s=26, marker="x", color="#9e9e9e", lw=0.9, zorder=1)
        if is_best:
            _plot_square(
                ax,
                target,
                TARGET_HALF_WIDTH * 2,
                facecolor="none",
                edgecolor="#d95f02",
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
                zorder=1,
            )

    _plot_square(
        ax,
        record.obstacle_center,
        OBSTACLE_WIDTH,
        facecolor="#d95f02",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.25,
        zorder=2,
    )
    ax.add_patch(
        Circle(
            record.obstacle_center,
            OBSTACLE_RADIUS + AVATAR_RADIUS,
            facecolor="none",
            edgecolor="#d95f02",
            linewidth=0.9,
            linestyle=":",
            alpha=0.8,
            zorder=1,
        )
    )

    ax.plot(record.points[:, 0], record.points[:, 1], color="#315b8a", linewidth=1.6, alpha=0.95, zorder=3)
    ax.scatter(record.points[0, 0], record.points[0, 1], s=28, color="#2f7d32", marker="o", zorder=5)
    ax.scatter(record.points[-1, 0], record.points[-1, 1], s=30, color="#b33a3a", marker="o", zorder=5)
    _plot_velocity_arrows(ax, record, show_actual=True)
    if record.panel == "Incorrect decoded choice":
        bci_mean = _mean_velocity(record.bci_velocity)
        actual_mean = _mean_velocity(record.actual_velocity)
        ax.text(
            0.02,
            0.03,
            f"mean BCI v=({bci_mean[0]:.2f}, {bci_mean[1]:.2f})\n"
            f"mean actual v=({actual_mean[0]:.2f}, {actual_mean[1]:.2f})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.5},
            zorder=20,
        )
    title = (
        f"{record.panel}\n"
        f"{record.condition}; true: {record.target_label}; decoded: {record.best_goal_label}\n"
        f"T={record.true_frac:.2f}, Adj={record.neighbor_frac:.2f}, Other={record.other_frac:.2f}"
    )
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-1.0, 14.5)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    ax.grid(True, color="#dddddd", lw=0.45, alpha=0.8)
    ax.tick_params(labelsize=7)


def _plot_velocity_arrows(
    ax,
    record: TrialRecord,
    arrow_every: int = 6,
    arrow_scale: float = 0.62,
    show_actual: bool = False,
) -> None:
    n = min(len(record.points), len(record.bci_velocity), len(record.actual_velocity))
    if n == 0:
        return
    points = record.points[:n]
    bci = record.bci_velocity[:n]
    actual = record.actual_velocity[:n]
    bci_speed = np.linalg.norm(bci, axis=1)
    actual_speed = np.linalg.norm(actual, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(bci_speed)
        & np.isfinite(actual_speed)
        & ((bci_speed > SPEED_THRESH) | (actual_speed > SPEED_THRESH))
    )
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return
    idx = idx[::max(1, int(arrow_every))]

    def unit(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return (v / norm) * arrow_scale

    bci_u = unit(bci[idx])
    actual_u = unit(actual[idx]) * 0.78
    for point, bci_vec, actual_vec in zip(points[idx], bci_u, actual_u):
        if show_actual:
            ax.arrow(
                point[0],
                point[1],
                actual_vec[0],
                actual_vec[1],
                head_width=0.10,
                length_includes_head=True,
                color="#2AA198",
                alpha=0.55,
                zorder=11,
            )
        ax.arrow(
            point[0],
            point[1],
            bci_vec[0],
            bci_vec[1],
            head_width=0.16,
            length_includes_head=True,
            color="black",
            alpha=1.0,
            linewidth=1.15,
            zorder=14,
        )


def _write_provenance(examples: list[TrialRecord], output_dir: Path) -> list[Path]:
    rows = []
    for r in examples:
        rows.append(
            {
                "panel": r.panel,
                "monkey": r.monkey,
                "experiment": r.experiment,
                "session_index": r.session_index,
                "session_file": r.session_file,
                "trial_id": r.trial_id,
                "condition": r.condition,
                "target_label": r.target_label,
                "choice_state": r.choice_state,
                "choice_label": r.choice_label,
                "best_goal_label": r.best_goal_label,
                "true_frac": r.true_frac,
                "neighbor_frac": r.neighbor_frac,
                "other_frac": r.other_frac,
                "best_frac": r.best_frac,
                "mean_bci_vx": _mean_velocity(r.bci_velocity)[0],
                "mean_bci_vz": _mean_velocity(r.bci_velocity)[1],
                "mean_actual_vx": _mean_velocity(r.actual_velocity)[0],
                "mean_actual_vz": _mean_velocity(r.actual_velocity)[1],
                "min_obstacle_distance_m": r.min_obstacle_distance,
                "final_distance_true_target_m": r.final_distance_true,
                "max_consecutive_target_frames": r.max_consecutive_target_frames,
                "max_z_minus_target_z_m": r.max_z_minus_target_z_m,
                "selection_score": r.score,
            }
        )
    csv_path = output_dir / f"{DEFAULT_OUTPUT_STEM}_provenance.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, sep=";")
    txt_path = output_dir / f"{DEFAULT_OUTPUT_STEM}_provenance.txt"
    txt_path.write_text(
        "\n".join(
            [
                "Representative decoder-direction error examples.",
                "Categories illustrate the decoded-choice states that contribute to Figure 4B's decoder-direction error bucket.",
                "Choice rule: each valid modelVelocity frame votes for the goal with maximal positive cosine similarity; a goal is chosen when its fraction is >= 0.60.",
                "Examples are selected from shared-control trials only.",
                "Adjacent decoded choice: ambiguous_choice trials where neighbor mass is >= 80% of non-true mass.",
                "Ambiguous decoded choice: no goal reaches the 0.60 choice threshold and votes are not locally concentrated on adjacent targets.",
                "Incorrect decoded choice: a non-true goal reaches the 0.60 choice threshold.",
                "Black arrows show decoded BCI velocity from the paired AI-log Input stream; teal arrows show executed velocity from the AI-log Output stream.",
                "Execution-like failures are excluded using the same geometric guards as the execution-example figure.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [csv_path, txt_path]


def build_decoder_direction_figure(
    base_dir: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    monkeys: list[str] | None = None,
    experiments: list[str] | None = None,
    from_provenance: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = (
        select_examples_from_provenance(base_dir, from_provenance)
        if from_provenance is not None
        else select_examples(base_dir, monkeys or DEFAULT_MONKEYS, experiments or DEFAULT_EXPERIMENTS)
    )
    fig, axes = plt.subplots(1, len(examples), figsize=(4.35 * len(examples), 4.7), sharex=True, sharey=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, record in zip(axes, examples):
        _plot_trial(ax, record)

    handles = [
        Line2D([0], [0], color="#315b8a", lw=1.6, label="Trajectory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2f7d32", markersize=5, label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#b33a3a", markersize=5, label="End"),
        Patch(facecolor="#4daf4a", edgecolor="none", alpha=0.12, label="True target window"),
        Line2D([0], [0], marker="x", color="#2f7d32", lw=0, markersize=5, label="True target"),
        Patch(facecolor="none", edgecolor="#d95f02", linestyle="--", label="Decoded target"),
        Patch(facecolor="#d95f02", edgecolor="none", alpha=0.25, label="Obstacle"),
        FancyArrowPatch((0, 0), (1.0, 0.0), arrowstyle="->", mutation_scale=14, color="black", label="Decoded BCI velocity"),
        FancyArrowPatch((0, 0), (1.0, 0.0), arrowstyle="->", mutation_scale=14, color="#2AA198", label="Actual velocity"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.22, wspace=0.18)

    outputs = []
    for ext in ("svg", "png", "pdf"):
        path = output_dir / f"{DEFAULT_OUTPUT_STEM}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(path)
        print(f"Saved decoder-direction examples: {path}")
    plt.close(fig)
    outputs.extend(_write_provenance(examples, output_dir))
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build representative decoder-direction error examples.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--from-provenance", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    return build_decoder_direction_figure(
        load_base_dir(args.base_dir),
        Path(args.output_dir),
        monkeys=args.monkeys,
        experiments=args.experiments,
        from_provenance=Path(args.from_provenance) if args.from_provenance else None,
    )


if __name__ == "__main__":
    main()
