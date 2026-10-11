"""Verify Figure 5 failure-mode and decoded-choice summaries from raw trials.

This script is intentionally read-only with respect to source data and
Variables/grand_summary.pkl. It recomputes the Figure 5 quantities from raw
trial pickles, compares them with the cached manuscript summary, and writes
verification outputs under analysis_results/figure5_failure_verification/.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import target_mapping, target_to_obstacle_mapping
from src.load import load_files


OUT_DIR = PROJECT_ROOT / "analysis_results" / "figure5_failure_verification"
FIG_DIR = PROJECT_ROOT / "figures" / "Figure 5" / "Failure verification"
SUMMARY_PATH = PROJECT_ROOT / "Variables" / "grand_summary.pkl"

DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
]

ELIGIBLE_ANSWERS = {1, 3, 5, 6}
EXCLUDED_ANSWERS = {4, 9, 10}
SUCCESS_ANSWER = 1

TARGET_HALF_WIDTH = 2.10
OBSTACLE_RADIUS = 0.45
AVATAR_RADIUS = 0.25
STUCK_MARGIN = 0.02
MIN_OBSTACLE_FRAMES = 10
MIN_DWELL_FRAMES = 10
CHOICE_THRESH = 0.60
SPEED_THRESH = 0.10

BCI_COLOR = "#CC79A7"
SHARED_COLOR = "#009E73"

TARGETS = {
    label: np.asarray(pos, dtype=float)[[0, 2]]
    for pos, label in target_mapping.items()
}
TARGET_ORDER = ["left", "slight_left", "straight", "slight_right", "right"]
KNOWN_GOALS = [tuple(TARGETS[label]) for label in ["right", "slight_right", "straight", "slight_left", "left"]]
GOAL_IDX = {goal: i for i, goal in enumerate(KNOWN_GOALS)}
ARC_NEIGHBORS = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
GOAL_TO_LABEL = {tuple(v): k for k, v in TARGETS.items()}

EXECUTION_REASONS = {
    "stuck_obstacle",
    "overshoot",
    "not_close_to_true_z",
    "not_long_enough",
}
DECODER_REASONS = {
    "wrong_target_choice",
    "neighbor_choice",
    "ambiguous_choice",
    "ai_bci_failure_target",
    "wrong_target_other",
    "ambiguous_choice_unknown",
}
CHOICE_STATES = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]


def neighbor_indices(goal_index: int, definition: str = "arc") -> list[int]:
    if definition == "arc":
        return list(ARC_NEIGHBORS.get(goal_index, []))
    if definition == "two_nearest":
        goals = np.asarray(KNOWN_GOALS, dtype=float)
        dists = np.linalg.norm(goals - goals[goal_index], axis=1)
        order = [int(i) for i in np.argsort(dists) if int(i) != goal_index]
        return order[:2]
    raise ValueError(f"Unknown neighbor definition: {definition}")


def neighbor_definition_table() -> pd.DataFrame:
    rows = []
    for i, goal in enumerate(KNOWN_GOALS):
        label = GOAL_TO_LABEL[goal]
        distances = []
        for j, other in enumerate(KNOWN_GOALS):
            if i == j:
                continue
            distances.append(
                {
                    "target_label": label,
                    "target_goal_index": i,
                    "target_x": goal[0],
                    "target_z": goal[1],
                    "other_label": GOAL_TO_LABEL[other],
                    "other_goal_index": j,
                    "other_x": other[0],
                    "other_z": other[1],
                    "distance_m": float(np.linalg.norm(np.asarray(goal) - np.asarray(other))),
                    "arc_neighbor": int(j in neighbor_indices(i, "arc")),
                    "two_nearest_neighbor": int(j in neighbor_indices(i, "two_nearest")),
                }
            )
        for rank, item in enumerate(sorted(distances, key=lambda row: (row["distance_m"], row["other_goal_index"])), start=1):
            item["distance_rank"] = rank
            rows.append(item)
    return pd.DataFrame(rows)


def load_base_dir(cli_base_dir: str | None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def get_attr(trial, key: str, default=None):
    return trial.get(key, default) if isinstance(trial, dict) else getattr(trial, key, default)


def trial_id(trial) -> int | str:
    value = get_attr(trial, "trial")
    try:
        return int(value)
    except Exception:
        return "unknown" if value is None else str(value)


def condition_label(trial) -> str | None:
    try:
        factor = float(get_attr(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None
    if factor == 1.0:
        return "AI"
    if factor == 0.0:
        return "No AI"
    return None


def included_by_task(experiment: str, trial) -> bool:
    answer = get_attr(trial, "answer", None)
    if answer not in ELIGIBLE_ANSWERS:
        return False
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return get_attr(trial, "obstacleAppearedTime", None) is not None
    return True


def target_label(trial) -> str | None:
    pos = get_attr(trial, "targetPosition")
    if pos is None:
        return None
    try:
        xz = np.asarray(pos, dtype=float)[[0, 2]]
    except Exception:
        return None
    label, dist = min(
        ((label, float(np.linalg.norm(xz - center))) for label, center in TARGETS.items()),
        key=lambda item: item[1],
    )
    return label if dist < 0.35 else None


def trial_xz(trial) -> np.ndarray | None:
    traj = get_attr(trial, "avatarTrajectory")
    if not isinstance(traj, dict) or "x" not in traj or "z" not in traj:
        return None
    x = np.asarray(traj["x"], dtype=float)
    z = np.asarray(traj["z"], dtype=float)
    n = min(len(x), len(z))
    if n < 4:
        return None
    points = np.column_stack([x[:n], z[:n]])
    points = points[np.isfinite(points).all(axis=1)]
    return points if len(points) >= 4 else None


def obstacle_center(label: str, trial) -> tuple[float, float] | None:
    obstacle_pos = get_attr(trial, "obstaclePosition")
    if obstacle_pos is not None:
        try:
            obstacle = np.asarray([obstacle_pos[0], obstacle_pos[2]], dtype=float)
            if np.isfinite(obstacle).all():
                return float(obstacle[0]), float(obstacle[1])
        except Exception:
            pass
    target = TARGETS[label]
    mapped = target_to_obstacle_mapping.get((round(float(target[0]), 1), round(float(target[1]), 1)))
    if mapped is None:
        return None
    return float(mapped[0]), float(mapped[1])


def target_window_mask(points: np.ndarray, label: str) -> np.ndarray:
    target = TARGETS[label]
    return (
        (np.abs(points[:, 0] - target[0]) <= TARGET_HALF_WIDTH)
        & (np.abs(points[:, 1] - target[1]) <= TARGET_HALF_WIDTH)
    )


def max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def classify_behavioral_reason(
    points: np.ndarray,
    label: str,
    obstacle: tuple[float, float] | None,
) -> tuple[str, dict]:
    diagnostics = {
        "target_half_width_m": TARGET_HALF_WIDTH,
        "min_dwell_frames": MIN_DWELL_FRAMES,
        "min_obstacle_frames": MIN_OBSTACLE_FRAMES,
        "obstacle_radius_m": OBSTACLE_RADIUS,
        "avatar_radius_m": AVATAR_RADIUS,
        "stuck_margin_m": STUCK_MARGIN,
    }
    target = TARGETS[label]
    if obstacle is not None:
        obstacle_arr = np.asarray(obstacle, dtype=float)
        inflated_half = OBSTACLE_RADIUS + AVATAR_RADIUS + STUCK_MARGIN
        inside_obstacle = (
            (np.abs(points[:, 0] - obstacle_arr[0]) < inflated_half)
            & (np.abs(points[:, 1] - obstacle_arr[1]) < inflated_half)
        )
        diagnostics["max_consecutive_obstacle_frames"] = max_consecutive(inside_obstacle)
        diagnostics["passed_obstacle"] = bool(np.any(points[:, 1] > obstacle_arr[1] + OBSTACLE_RADIUS))
        if diagnostics["max_consecutive_obstacle_frames"] >= MIN_OBSTACLE_FRAMES and not diagnostics["passed_obstacle"]:
            return "stuck_obstacle", diagnostics

    inside_true = target_window_mask(points, label)
    ever_in_true = bool(np.any(inside_true))
    longest_true = max_consecutive(inside_true)
    target_z = float(target[1])
    diagnostics["ever_in_true_target_window"] = ever_in_true
    diagnostics["max_consecutive_target_frames"] = longest_true
    diagnostics["max_z_minus_target_z_m"] = float(np.nanmax(points[:, 1]) - target_z)

    if ever_in_true:
        enter_idx = int(np.argmax(inside_true))
        if bool(np.any(points[enter_idx:, 1] > target_z + TARGET_HALF_WIDTH)):
            return "overshoot", diagnostics
    if ever_in_true and longest_true < MIN_DWELL_FRAMES:
        return "not_long_enough", diagnostics

    other_dwell = {}
    for other_label, other_target in TARGETS.items():
        if other_label == label:
            continue
        other_mask = (
            (np.abs(points[:, 0] - other_target[0]) <= TARGET_HALF_WIDTH)
            & (np.abs(points[:, 1] - other_target[1]) <= TARGET_HALF_WIDTH)
            & ~inside_true
        )
        other_dwell[other_label] = max_consecutive(other_mask)
    if other_dwell:
        best_other_label = max(other_dwell, key=other_dwell.get)
        diagnostics["wrong_target_label_by_dwell"] = best_other_label
        diagnostics["wrong_target_max_dwell_frames"] = int(other_dwell[best_other_label])
        if other_dwell[best_other_label] >= MIN_DWELL_FRAMES:
            return "wrong_target", diagnostics

    nearest_label, nearest_dist = min(
        ((goal_label, float(np.linalg.norm(points[-1] - goal))) for goal_label, goal in TARGETS.items()),
        key=lambda item: item[1],
    )
    diagnostics["nearest_goal_label_final"] = nearest_label
    diagnostics["nearest_goal_dist_final_m"] = nearest_dist
    if (
        (not ever_in_true)
        and nearest_label == label
        and float(np.nanmax(points[:, 1])) < target_z - TARGET_HALF_WIDTH
    ):
        return "not_close_to_true_z", diagnostics
    return "other", diagnostics


def choice_state(
    trial,
    label: str,
    experiment: str,
    obstacle: tuple[float, float] | None = None,
    obstacle_mask_radius: float | None = 1.5,
    neighbor_definition: str = "arc",
) -> dict:
    mv = get_attr(trial, "modelVelocity")
    traj = get_attr(trial, "avatarTrajectory")
    empty = {
        "choice_state": "no_model",
        "choice_label": "no_model",
        "best_goal_label": "no_model",
        "true_frac": np.nan,
        "neighbor_frac": np.nan,
        "other_frac": np.nan,
        "best_frac": np.nan,
        "n_choice_frames": 0,
        "n_speed_valid_frames": 0,
        "n_obstacle_masked_frames": 0,
        "obstacle_mask_radius_m": obstacle_mask_radius if obstacle_mask_radius is not None else np.nan,
    }
    if not isinstance(mv, dict) or not isinstance(traj, dict):
        return empty
    try:
        vx = np.asarray(mv["vx"], dtype=float)
        vz = np.asarray(mv["vz"], dtype=float)
        xs = np.asarray(traj["x"], dtype=float)
        zs = np.asarray(traj["z"], dtype=float)
    except Exception:
        return empty
    n = min(len(vx), len(vz), len(xs), len(zs))
    if n < 5:
        return empty
    vectors = np.column_stack([vx[:n], vz[:n]])
    xs = xs[:n]
    zs = zs[:n]
    speed = np.linalg.norm(vectors, axis=1)
    speed_valid = np.isfinite(speed) & (speed > SPEED_THRESH)
    valid = speed_valid.copy()
    n_obstacle_masked = 0
    if obstacle is not None and "Obstacle" in str(experiment) and obstacle_mask_radius is not None:
        obstacle_arr = np.asarray(obstacle, dtype=float)
        dist_obstacle = np.sqrt((xs - obstacle_arr[0]) ** 2 + (zs - obstacle_arr[1]) ** 2)
        obstacle_keep = dist_obstacle > float(obstacle_mask_radius)
        n_obstacle_masked = int(np.sum(speed_valid & ~obstacle_keep))
        valid &= obstacle_keep
    if not np.any(valid):
        return empty
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
        if best_goal is not None and best_cos > 0.0:
            counts[best_goal] += 1
    total = sum(counts.values())
    if total == 0:
        return empty

    fracs = {goal: count / total for goal, count in counts.items() if count > 0}
    true_goal = tuple(TARGETS[label])
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
        neighbor_goals = [KNOWN_GOALS[i] for i in neighbor_indices(GOAL_IDX[true_goal], neighbor_definition)]
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
        "n_choice_frames": int(total),
        "n_speed_valid_frames": int(np.sum(speed_valid)),
        "n_obstacle_masked_frames": n_obstacle_masked,
        "obstacle_mask_radius_m": obstacle_mask_radius if obstacle_mask_radius is not None else np.nan,
    }


def failure_reason_from_trial(trial, success: bool, behavioral_reason: str, choice: dict) -> str:
    if success:
        return "correct"
    if behavioral_reason in EXECUTION_REASONS:
        return behavioral_reason
    state = choice["choice_state"]
    label = choice["choice_label"]
    if behavioral_reason == "wrong_target":
        if label == "wrong_choice":
            return "wrong_target_choice"
        if label == "correct_choice":
            return "ai_bci_failure_target"
        if state == "neighbor_choice":
            return "neighbor_choice"
        if state == "ambiguous_choice":
            return "ambiguous_choice"
        return "wrong_target_other"
    if behavioral_reason == "other":
        return "other"
    if label == "no_model":
        return "other"
    return "other"


def collect_raw_trials(base_dir: str, monkeys: list[str], experiments: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    sessions = []
    for experiment in experiments:
        for monkey in monkeys:
            (
                all_trials,
                _all_correct,
                _all_incorrect,
                _all_training,
                _all_channels,
                _nb_channels,
                pkl_files,
                _ai_trials,
            ) = load_files(experiment, monkey, base_dir=base_dir)
            for session_index, trials in enumerate(all_trials):
                session_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                answer_counts = Counter(get_attr(tr, "answer", None) for tr in trials)
                sessions.append(
                    {
                        "monkey": monkey,
                        "experiment": experiment,
                        "session_index": session_index,
                        "session_file": session_file,
                        "n_loaded_trials": len(trials),
                        "n_answer_1": answer_counts.get(1, 0),
                        "n_answer_3": answer_counts.get(3, 0),
                        "n_answer_5": answer_counts.get(5, 0),
                        "n_answer_6": answer_counts.get(6, 0),
                        "n_excluded_answer_4": answer_counts.get(4, 0),
                        "n_excluded_answer_9": answer_counts.get(9, 0),
                        "n_excluded_answer_10": answer_counts.get(10, 0),
                    }
                )
                for trial in trials:
                    ans = get_attr(trial, "answer", None)
                    condition = condition_label(trial)
                    label = target_label(trial)
                    points = trial_xz(trial)
                    included = condition is not None and label is not None and points is not None and included_by_task(experiment, trial)
                    base = {
                        "monkey": monkey,
                        "experiment": experiment,
                        "session_index": session_index,
                        "session_file": session_file,
                        "trial_id": trial_id(trial),
                        "answer": ans,
                        "condition": condition,
                        "target_label": label,
                        "included": int(included),
                        "excluded_answer_code": int(ans in EXCLUDED_ANSWERS),
                    }
                    if not included:
                        rows.append(base)
                        continue
                    success = ans == SUCCESS_ANSWER
                    obstacle = obstacle_center(label, trial)
                    behavioral_reason, diagnostics = classify_behavioral_reason(points, label, obstacle)
                    choice = choice_state(trial, label, experiment, obstacle)
                    choice_unmasked = choice_state(
                        trial,
                        label,
                        experiment,
                        obstacle,
                        obstacle_mask_radius=None,
                    )
                    choice_two_nearest = choice_state(
                        trial,
                        label,
                        experiment,
                        obstacle,
                        neighbor_definition="two_nearest",
                    )
                    reason = failure_reason_from_trial(trial, success, behavioral_reason, choice)
                    reason_unmasked = failure_reason_from_trial(trial, success, behavioral_reason, choice_unmasked)
                    reason_force_no_model_decoder = (
                        "wrong_target_other"
                        if (not success and behavioral_reason == "wrong_target" and choice["choice_label"] == "no_model")
                        else reason
                    )
                    row = {
                        **base,
                        "success": int(success),
                        "execution_reason": "correct" if success else behavioral_reason,
                        "failure_reason": reason,
                        "failure_reason_no_obstacle_choice_mask": reason_unmasked,
                        "failure_reason_force_no_model_decoder_bucket": reason_force_no_model_decoder,
                        "outcome_class": outcome_class(reason, success),
                        "obstacle_x": obstacle[0] if obstacle is not None else np.nan,
                        "obstacle_z": obstacle[1] if obstacle is not None else np.nan,
                        "n_trajectory_points": len(points),
                        **choice,
                        "choice_state_two_nearest": choice_two_nearest["choice_state"],
                        "choice_label_two_nearest": choice_two_nearest["choice_label"],
                        "best_goal_label_two_nearest": choice_two_nearest["best_goal_label"],
                        "true_frac_two_nearest": choice_two_nearest["true_frac"],
                        "neighbor_frac_two_nearest": choice_two_nearest["neighbor_frac"],
                        "other_frac_two_nearest": choice_two_nearest["other_frac"],
                        "best_frac_two_nearest": choice_two_nearest["best_frac"],
                        "n_choice_frames_two_nearest": choice_two_nearest["n_choice_frames"],
                        "choice_state_no_obstacle_mask": choice_unmasked["choice_state"],
                        "choice_label_no_obstacle_mask": choice_unmasked["choice_label"],
                        "best_goal_label_no_obstacle_mask": choice_unmasked["best_goal_label"],
                        "true_frac_no_obstacle_mask": choice_unmasked["true_frac"],
                        "neighbor_frac_no_obstacle_mask": choice_unmasked["neighbor_frac"],
                        "other_frac_no_obstacle_mask": choice_unmasked["other_frac"],
                        "best_frac_no_obstacle_mask": choice_unmasked["best_frac"],
                        "n_choice_frames_no_obstacle_mask": choice_unmasked["n_choice_frames"],
                        **diagnostics,
                    }
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(sessions)


def outcome_class(reason: str, success: bool) -> str:
    if success:
        return "success"
    if reason in EXECUTION_REASONS:
        return "execution_error"
    if reason in DECODER_REASONS:
        return "decoder_direction_error"
    return "other_failure"


def summarize_failure_props(trials: pd.DataFrame) -> pd.DataFrame:
    included = trials[trials["included"].eq(1)].copy()
    records = []
    for (monkey, experiment, condition), sub in included.groupby(["monkey", "experiment", "condition"], sort=False):
        total = int(len(sub))
        counts = sub["failure_reason"].value_counts().to_dict()
        success = int(sub["success"].sum())
        row = {
            "monkey": monkey,
            "experiment": experiment,
            "condition": condition,
            "n_trials": total,
            "success": success / total if total else np.nan,
        }
        for key in [
            "stuck_obstacle",
            "overshoot",
            "not_close_to_true_z",
            "not_long_enough",
            "wrong_target_choice",
            "neighbor_choice",
            "ambiguous_choice",
            "ai_bci_failure_target",
            "wrong_target_other",
            "ambiguous_choice_unknown",
            "other",
        ]:
            row[key] = counts.get(key, 0) / total if total else np.nan
        records.append(row)
    return pd.DataFrame(records)


def summarize_choice_state(trials: pd.DataFrame) -> pd.DataFrame:
    included = trials[trials["included"].eq(1) & trials["choice_state"].isin(CHOICE_STATES)].copy()
    records = []
    for (monkey, experiment, state), sub in included.groupby(["monkey", "experiment", "choice_state"], sort=False):
        off = sub[sub["condition"].eq("No AI")]
        on = sub[sub["condition"].eq("AI")]
        exec_mask = ~sub["failure_reason"].isin(EXECUTION_REASONS)
        off_b = sub[exec_mask & sub["condition"].eq("No AI")]
        on_b = sub[exec_mask & sub["condition"].eq("AI")]
        records.append(
            {
                "monkey": monkey,
                "experiment": experiment,
                "state": state,
                "p_success_off": float(off["success"].mean()) if len(off) else np.nan,
                "p_success_on": float(on["success"].mean()) if len(on) else np.nan,
                "p_success_off_behavioral": float(off_b["success"].mean()) if len(off_b) else np.nan,
                "p_success_on_behavioral": float(on_b["success"].mean()) if len(on_b) else np.nan,
                "n_off": int(len(off)),
                "n_on": int(len(on)),
                "n_off_behavioral": int(len(off_b)),
                "n_on_behavioral": int(len(on_b)),
            }
        )
    existing = pd.DataFrame(records)
    units = included[["monkey", "experiment"]].drop_duplicates()
    complete = []
    for _, unit in units.iterrows():
        for state in CHOICE_STATES:
            match = existing[
                existing["monkey"].eq(unit["monkey"])
                & existing["experiment"].eq(unit["experiment"])
                & existing["state"].eq(state)
            ]
            if match.empty:
                complete.append(
                    {
                        "monkey": unit["monkey"],
                        "experiment": unit["experiment"],
                        "state": state,
                        "p_success_off": np.nan,
                        "p_success_on": np.nan,
                        "p_success_off_behavioral": np.nan,
                        "p_success_on_behavioral": np.nan,
                        "n_off": 0,
                        "n_on": 0,
                        "n_off_behavioral": 0,
                        "n_on_behavioral": 0,
                    }
                )
            else:
                complete.append(match.iloc[0].to_dict())
    return pd.DataFrame(complete)


def summarize_choice_state_variant(trials: pd.DataFrame, state_col: str) -> pd.DataFrame:
    included = trials[trials["included"].eq(1) & trials[state_col].isin(CHOICE_STATES)].copy()
    records = []
    for (monkey, experiment, state), sub in included.groupby(["monkey", "experiment", state_col], sort=False):
        off = sub[sub["condition"].eq("No AI")]
        on = sub[sub["condition"].eq("AI")]
        records.append(
            {
                "monkey": monkey,
                "experiment": experiment,
                "state": state,
                "p_success_off": float(off["success"].mean()) if len(off) else np.nan,
                "p_success_on": float(on["success"].mean()) if len(on) else np.nan,
                "n_off": int(len(off)),
                "n_on": int(len(on)),
            }
        )
    existing = pd.DataFrame(records)
    units = included[["monkey", "experiment"]].drop_duplicates()
    complete = []
    for _, unit in units.iterrows():
        for state in CHOICE_STATES:
            match = existing[
                existing["monkey"].eq(unit["monkey"])
                & existing["experiment"].eq(unit["experiment"])
                & existing["state"].eq(state)
            ]
            if match.empty:
                complete.append(
                    {
                        "monkey": unit["monkey"],
                        "experiment": unit["experiment"],
                        "state": state,
                        "p_success_off": np.nan,
                        "p_success_on": np.nan,
                        "n_off": 0,
                        "n_on": 0,
                    }
                )
            else:
                complete.append(match.iloc[0].to_dict())
    return pd.DataFrame(complete)


def summarize_outcome_accounting(trials: pd.DataFrame) -> pd.DataFrame:
    included = trials[trials["included"].eq(1)].copy()
    rows = []
    for (monkey, experiment, condition), sub in included.groupby(["monkey", "experiment", "condition"], sort=False):
        counts = sub["outcome_class"].value_counts().to_dict()
        total = int(len(sub))
        rows.append(
            {
                "monkey": monkey,
                "experiment": experiment,
                "condition": condition,
                "n_trials": total,
                "n_success": counts.get("success", 0),
                "n_execution_error": counts.get("execution_error", 0),
                "n_decoder_direction_error": counts.get("decoder_direction_error", 0),
                "n_other_failure": counts.get("other_failure", 0),
                "n_accounted": sum(counts.values()),
                "accounting_ok": bool(total == sum(counts.values())),
            }
        )
    return pd.DataFrame(rows)


def export_disagreement_examples(trials: pd.DataFrame, path: Path, max_rows: int = 200) -> pd.DataFrame:
    included = trials[trials["included"].eq(1)].copy()
    if included.empty:
        out = pd.DataFrame()
        out.to_csv(path, index=False)
        return out

    mask = (
        included["choice_state"].ne(included["choice_state_no_obstacle_mask"])
        | included["failure_reason"].ne(included["failure_reason_no_obstacle_choice_mask"])
        | included["failure_reason"].ne(included["failure_reason_force_no_model_decoder_bucket"])
    )
    rows = included[mask].copy()
    if rows.empty:
        out = pd.DataFrame()
        out.to_csv(path, index=False)
        return out

    def rule(row):
        reasons = []
        if row.get("choice_state") != row.get("choice_state_no_obstacle_mask"):
            reasons.append("obstacle_distance_mask_changed_choice_state")
        if row.get("failure_reason") != row.get("failure_reason_no_obstacle_choice_mask"):
            reasons.append("obstacle_distance_mask_changed_failure_reason")
        if row.get("failure_reason") != row.get("failure_reason_force_no_model_decoder_bucket"):
            reasons.append("no_model_residual_other_vs_decoder_bucket")
        return ";".join(reasons)

    rows["disagreement_rule"] = rows.apply(rule, axis=1)
    rows["trial_level_cached_label_available"] = False
    rows["cached_label_limitation"] = (
        "Variables/grand_summary.pkl stores per-unit aggregates, not trial-level cached labels."
    )
    cols = [
        "monkey",
        "experiment",
        "session_index",
        "session_file",
        "trial_id",
        "answer",
        "condition",
        "target_label",
        "success",
        "execution_reason",
        "failure_reason",
        "failure_reason_no_obstacle_choice_mask",
        "failure_reason_force_no_model_decoder_bucket",
        "choice_state",
        "choice_label",
        "choice_state_no_obstacle_mask",
        "choice_label_no_obstacle_mask",
        "best_goal_label",
        "best_goal_label_no_obstacle_mask",
        "true_frac",
        "neighbor_frac",
        "other_frac",
        "best_frac",
        "true_frac_no_obstacle_mask",
        "neighbor_frac_no_obstacle_mask",
        "other_frac_no_obstacle_mask",
        "best_frac_no_obstacle_mask",
        "n_speed_valid_frames",
        "n_obstacle_masked_frames",
        "n_choice_frames",
        "n_choice_frames_no_obstacle_mask",
        "max_consecutive_obstacle_frames",
        "passed_obstacle",
        "ever_in_true_target_window",
        "max_consecutive_target_frames",
        "max_z_minus_target_z_m",
        "target_half_width_m",
        "min_dwell_frames",
        "obstacle_mask_radius_m",
        "disagreement_rule",
        "trial_level_cached_label_available",
        "cached_label_limitation",
    ]
    out = rows[[c for c in cols if c in rows.columns]].head(max_rows)
    out.to_csv(path, index=False)
    return out


def load_cached_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with SUMMARY_PATH.open("rb") as f:
        summary = pickle.load(f)
    return pd.DataFrame(summary["failure_props_per_unit"]), pd.DataFrame(summary["success_state_per_unit"])


def compare_tables(raw: pd.DataFrame, cached: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    merged = raw.merge(cached, on=keys, how="outer", suffixes=("_raw", "_cache"), indicator=True)
    for col in value_cols:
        raw_col = f"{col}_raw"
        cache_col = f"{col}_cache"
        if raw_col in merged and cache_col in merged:
            merged[f"{col}_delta"] = pd.to_numeric(merged[raw_col], errors="coerce") - pd.to_numeric(
                merged[cache_col], errors="coerce"
            )
    return merged


def mean_sem(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else np.nan
    return mean, sem


def paired_test(on: np.ndarray, off: np.ndarray, alternative: str) -> dict:
    on = np.asarray(on, dtype=float)
    off = np.asarray(off, dtype=float)
    ok = np.isfinite(on) & np.isfinite(off)
    on = on[ok]
    off = off[ok]
    diffs = on - off
    out = {
        "n_pairs": int(len(diffs)),
        "n_zero_diffs": int(np.sum(diffs == 0)) if len(diffs) else 0,
        "alternative": alternative,
        "test": "wilcoxon_signed_rank",
        "zero_method": "wilcox",
        "method": "auto",
        "statistic": np.nan,
        "p_value": np.nan,
    }
    if len(diffs) < 2:
        return out
    if np.allclose(on, off, equal_nan=False):
        out["p_value"] = 1.0
        out["test"] = "all_equal"
        return out
    try:
        result = wilcoxon(on, off, alternative=alternative, zero_method="wilcox", mode="auto")
        out["statistic"] = float(result.statistic)
        out["p_value"] = float(result.pvalue)
    except Exception:
        pass
    return out


def stats_for_panels(failure: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    records = []
    panel_specs = {
        "a": {
            "blocked_at_obstacle": ("stuck_obstacle", "less"),
            "target_overshoot": ("overshoot", "less"),
            "insufficient_target_proximity": ("not_close_to_true_z", "less"),
            "insufficient_dwell_time": ("not_long_enough", "less"),
        },
        "b": {
            "execution_errors": (["stuck_obstacle", "overshoot", "not_close_to_true_z", "not_long_enough"], "less"),
            "decoder_direction_errors": (list(DECODER_REASONS), "two-sided"),
            "other_failures": (["other"], "two-sided"),
        },
    }
    for panel, specs in panel_specs.items():
        for category, (cols, alt) in specs.items():
            cols = [cols] if isinstance(cols, str) else cols
            off = []
            on = []
            for (monkey, experiment), sub in failure.groupby(["monkey", "experiment"], sort=False):
                r_off = sub[sub["condition"].eq("No AI")]
                r_on = sub[sub["condition"].eq("AI")]
                if r_off.empty or r_on.empty:
                    continue
                off.append(float(r_off.iloc[0][cols].sum()) * 100.0)
                on.append(float(r_on.iloc[0][cols].sum()) * 100.0)
            m_off, se_off = mean_sem(off)
            m_on, se_on = mean_sem(on)
            test = paired_test(np.asarray(on), np.asarray(off), alt)
            records.append(
                {
                    "panel": panel,
                    "category": category,
                    "mean_bci_only": m_off,
                    "sem_bci_only": se_off,
                    "mean_shared_control": m_on,
                    "sem_shared_control": se_on,
                    **test,
                }
            )
    for state in CHOICE_STATES:
        sub = choice[choice["state"].eq(state)]
        off = pd.to_numeric(sub["p_success_off"], errors="coerce").to_numpy()
        on = pd.to_numeric(sub["p_success_on"], errors="coerce").to_numpy()
        m_off, se_off = mean_sem(off)
        m_on, se_on = mean_sem(on)
        test = paired_test(on, off, "greater")
        records.append(
            {
                "panel": "c",
                "category": state,
                "mean_bci_only": m_off,
                "sem_bci_only": se_off,
                "mean_shared_control": m_on,
                "sem_shared_control": se_on,
                **test,
            }
        )
    return pd.DataFrame(records)


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def horizontal_jitter(n: int, width: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([-0.35 * width], dtype=float)
    offsets = np.arange(n, dtype=float) - (n - 1) / 2.0
    offsets[offsets >= 0] += 0.5
    offsets[offsets < 0] -= 0.5
    return offsets * (width / np.max(np.abs(offsets)))


def paired_dot_panel(ax, labels, vals_off, vals_on, ylabel, ylim, alternatives):
    x = np.arange(len(labels), dtype=float)
    offset = 0.16
    jitter_width = 0.055
    all_values = []
    pvals = []
    for i, label in enumerate(labels):
        off = np.asarray(vals_off[i], dtype=float)
        on = np.asarray(vals_on[i], dtype=float)
        ok = np.isfinite(off) & np.isfinite(on)
        off = off[ok]
        on = on[ok]
        jitter = horizontal_jitter(len(off), jitter_width)
        ax.scatter(
            np.full(len(off), x[i] - offset) + jitter,
            off,
            s=30,
            color=BCI_COLOR,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.88,
            zorder=3,
        )
        ax.scatter(
            np.full(len(on), x[i] + offset) + jitter,
            on,
            s=30,
            color=SHARED_COLOR,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.88,
            zorder=3,
        )
        for xpos, values, color in ((x[i] - offset, off, BCI_COLOR), (x[i] + offset, on, SHARED_COLOR)):
            mean, err = mean_sem(values)
            if np.isfinite(err):
                ax.errorbar([xpos], [mean], yerr=[err], fmt="none", ecolor="black", capsize=4, linewidth=1.1)
            ax.scatter([xpos], [mean], s=82, marker="D", color=color, edgecolor="black", linewidth=0.9, zorder=5)
        all_values.extend(off[np.isfinite(off)])
        all_values.extend(on[np.isfinite(on)])
        pvals.append(paired_test(on, off, alternatives[i])["p_value"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    h = (ylim[1] - ylim[0]) * 0.035
    for i, p in enumerate(pvals):
        stars = p_to_stars(p)
        if not stars:
            continue
        y_base = max(np.nanmax(vals_off[i]), np.nanmax(vals_on[i]))
        y = min(y_base + h, ylim[1] - 3 * h)
        ax.plot([x[i] - offset, x[i] - offset, x[i] + offset, x[i] + offset], [y, y + h, y + h, y], color="black")
        ax.text(x[i], y + h * 1.08, stars, ha="center", va="bottom", fontsize=10)


def make_candidate_figure(failure: pd.DataFrame, choice: pd.DataFrame) -> list[Path]:
    mpl.rcParams["svg.fonttype"] = "none"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 12.2))

    pair_keys = sorted(
        set(zip(failure["monkey"], failure["experiment"]))
    )

    def failure_values(specs):
        vals_off = [[] for _ in specs]
        vals_on = [[] for _ in specs]
        for monkey, experiment in pair_keys:
            sub = failure[failure["monkey"].eq(monkey) & failure["experiment"].eq(experiment)]
            off = sub[sub["condition"].eq("No AI")]
            on = sub[sub["condition"].eq("AI")]
            if off.empty or on.empty:
                continue
            for i, cols in enumerate(specs):
                vals_off[i].append(float(off.iloc[0][cols].sum()) * 100.0)
                vals_on[i].append(float(on.iloc[0][cols].sum()) * 100.0)
        return vals_off, vals_on

    vals_off, vals_on = failure_values([
        ["stuck_obstacle"],
        ["overshoot"],
        ["not_close_to_true_z"],
        ["not_long_enough"],
    ])
    paired_dot_panel(
        axes[0],
        ["Blocked at\nobstacle", "Target\novershoot", "Insufficient\ntarget proximity", "Insufficient\ndwell time"],
        vals_off,
        vals_on,
        "Percentage of all trials (%)",
        (0, max(1.0, max([max(v or [0]) for v in vals_off + vals_on]) * 1.28)),
        ["less", "less", "less", "less"],
    )
    axes[0].set_title("a  Failure categories")

    vals_off, vals_on = failure_values([
        ["stuck_obstacle", "overshoot", "not_close_to_true_z", "not_long_enough"],
        list(DECODER_REASONS),
        ["other"],
    ])
    paired_dot_panel(
        axes[1],
        ["Execution\nerrors", "Decoder-direction\nerrors", "Other\nfailures"],
        vals_off,
        vals_on,
        "Percentage of all trials (%)",
        (0, max(1.0, max([max(v or [0]) for v in vals_off + vals_on]) * 1.28)),
        ["less", "two-sided", "two-sided"],
    )
    axes[1].set_title("b  Overall outcome classes")

    vals_off = []
    vals_on = []
    for state in CHOICE_STATES:
        sub = choice[choice["state"].eq(state)]
        vals_off.append(pd.to_numeric(sub["p_success_off"], errors="coerce").tolist())
        vals_on.append(pd.to_numeric(sub["p_success_on"], errors="coerce").tolist())
    paired_dot_panel(
        axes[2],
        ["Correct\nchoice", "Adjacent\nchoice", "Ambiguous\nchoice", "Incorrect\nchoice"],
        vals_off,
        vals_on,
        "P(success | choice state)",
        (0, 1.05),
        ["greater", "greater", "greater", "greater"],
    )
    axes[2].set_title("c  Success conditioned on decoder-inferred choice state")

    axes[0].scatter([], [], color=BCI_COLOR, edgecolor="black", label="BCI-only")
    axes[0].scatter([], [], color=SHARED_COLOR, edgecolor="black", label="Shared-control")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(h_pad=2.0)

    paths = []
    for suffix in ("svg", "pdf", "png"):
        path = FIG_DIR / f"figure5_failure_verification_candidate.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def compare_to_fig3_sessions(session_rows: pd.DataFrame) -> pd.DataFrame:
    fig3_path = PROJECT_ROOT / "tables" / "Figure 3C original session audit" / "figure3c_original_experiment_summary.csv"
    if not fig3_path.exists():
        return pd.DataFrame()
    fig3 = pd.read_csv(fig3_path)
    figure5 = (
        session_rows.groupby(["monkey", "experiment"], dropna=False)
        .agg(
            n_figure5_session_files=("session_file", "nunique"),
            figure5_session_files=("session_file", lambda x: "; ".join(sorted(set(map(str, x))))),
            n_answer_1=("n_answer_1", "sum"),
            n_answer_3=("n_answer_3", "sum"),
            n_answer_5=("n_answer_5", "sum"),
            n_answer_6=("n_answer_6", "sum"),
            n_excluded_answer_4_9_10=("n_excluded_answer_4", "sum"),
        )
        .reset_index()
    )
    figure5["n_eligible_answer_1_3_5_6"] = (
        figure5["n_answer_1"] + figure5["n_answer_3"] + figure5["n_answer_5"] + figure5["n_answer_6"]
    )
    figure5["n_excluded_answer_4_9_10"] += (
        session_rows.groupby(["monkey", "experiment"])["n_excluded_answer_9"].sum().to_numpy()
        + session_rows.groupby(["monkey", "experiment"])["n_excluded_answer_10"].sum().to_numpy()
    )
    fig3_default = fig3[fig3["audit"].eq("default_run_success_rate_paths")].copy()
    return figure5.merge(
        fig3_default[
            [
                "monkey",
                "experiment",
                "n_available_session_files",
                "min_sessions_with_both_conditions",
                "median_sessions_with_both_conditions",
                "max_sessions_with_both_conditions",
            ]
        ],
        on=["monkey", "experiment"],
        how="left",
    )


def write_report(paths: dict[str, Path], stats_df: pd.DataFrame, failure_cmp: pd.DataFrame, choice_cmp: pd.DataFrame) -> None:
    max_failure_delta = np.nan
    max_choice_delta = np.nan
    failure_delta_cols = [c for c in failure_cmp.columns if c.endswith("_delta")]
    choice_delta_cols = [c for c in choice_cmp.columns if c.endswith("_delta")]
    if failure_delta_cols:
        max_failure_delta = float(np.nanmax(np.abs(failure_cmp[failure_delta_cols].to_numpy(dtype=float))))
    if choice_delta_cols:
        max_choice_delta = float(np.nanmax(np.abs(choice_cmp[choice_delta_cols].to_numpy(dtype=float))))

    lines = [
        "# Figure 5 Failure Verification",
        "",
        "Raw trial data were loaded from `config.yaml` `base_dir`; eligible answer codes were `{1,3,5,6}` and `{4,9,10}` were excluded.",
        "The reference cache `Variables/grand_summary.pkl` was read for comparison only and was not overwritten.",
        "",
        "## Definitions",
        "",
        f"- Blocked at obstacle: trajectory remains for at least {MIN_OBSTACLE_FRAMES} consecutive frames in the obstacle square inflated by obstacle radius {OBSTACLE_RADIUS:g} m, avatar radius {AVATAR_RADIUS:g} m, and margin {STUCK_MARGIN:g} m, and does not pass the obstacle.",
        f"- Target overshoot: failed trial enters the true-target window, then later has z greater than target z + {TARGET_HALF_WIDTH:g} m.",
        f"- Insufficient target proximity: failed trial never enters the true-target window and does not reach target z - {TARGET_HALF_WIDTH:g} m.",
        f"- Insufficient dwell time: failed trial enters the true-target window but remains there for fewer than {MIN_DWELL_FRAMES} consecutive frames.",
        f"- Decoder-inferred choice state: valid `modelVelocity` frames with speed > {SPEED_THRESH:g} vote by maximal positive cosine similarity; a goal is selected at fraction >= {CHOICE_THRESH:g}; adjacent choice is ambiguous non-true mass concentrated >= 80% on neighboring targets.",
        "",
        "## Reproduction Summary",
        "",
        f"- Maximum absolute failure-summary delta versus cache: {max_failure_delta:.6g}.",
        f"- Maximum absolute choice-state delta versus cache: {max_choice_delta:.6g}.",
        "",
        "## Tests",
        "",
        "Panel a uses paired Wilcoxon signed-rank tests, one-sided `less` for Shared-control < BCI-only.",
        "Panel b uses paired Wilcoxon: `less` for execution errors and `two-sided` for decoder-direction and other failures.",
        "Panel c uses paired Wilcoxon, one-sided `greater` for Shared-control > BCI-only.",
        "No multiple-testing correction is applied by the current plotting implementation.",
        "",
        "```",
        stats_df.to_csv(index=False).strip(),
        "```",
        "",
        "## Output Files",
        "",
    ]
    for label, path in paths.items():
        lines.append(f"- `{label}`: `{path.relative_to(PROJECT_ROOT)}`")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(base_dir: str, monkeys: list[str], experiments: list[str], write_candidate_figure: bool = False) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials, sessions = collect_raw_trials(base_dir, monkeys, experiments)
    failure = summarize_failure_props(trials)
    choice = summarize_choice_state(trials)
    accounting = summarize_outcome_accounting(trials)
    fig3_compare = compare_to_fig3_sessions(sessions)
    cached_failure, cached_choice = load_cached_tables()
    failure_cols = [
        "n_trials",
        "success",
        "stuck_obstacle",
        "overshoot",
        "not_close_to_true_z",
        "not_long_enough",
        "wrong_target_choice",
        "neighbor_choice",
        "ambiguous_choice",
        "ai_bci_failure_target",
        "wrong_target_other",
        "ambiguous_choice_unknown",
        "other",
    ]
    choice_cols = [
        "p_success_off",
        "p_success_on",
        "p_success_off_behavioral",
        "p_success_on_behavioral",
        "n_off",
        "n_on",
        "n_off_behavioral",
        "n_on_behavioral",
    ]
    failure_cmp = compare_tables(
        failure,
        cached_failure,
        ["monkey", "experiment", "condition"],
        failure_cols,
    )
    choice_cmp = compare_tables(
        choice,
        cached_choice,
        ["monkey", "experiment", "state"],
        choice_cols,
    )
    stats_df = stats_for_panels(failure, choice)

    paths = {
        "trial_level_provenance": OUT_DIR / "figure5_trial_level_provenance.csv",
        "session_inventory": OUT_DIR / "figure5_session_inventory.csv",
        "figure3_session_comparison": OUT_DIR / "figure5_vs_figure3_session_comparison.csv",
        "neighbor_definitions": OUT_DIR / "figure5_neighbor_definitions.csv",
        "raw_failure_props_per_unit": OUT_DIR / "figure5_raw_failure_props_per_unit.csv",
        "raw_success_state_per_unit": OUT_DIR / "figure5_raw_success_state_per_unit.csv",
        "two_nearest_success_state_per_unit": OUT_DIR / "figure5_two_nearest_success_state_per_unit.csv",
        "outcome_accounting": OUT_DIR / "figure5_outcome_accounting.csv",
        "disagreement_examples": OUT_DIR / "figure5_disagreement_examples.csv",
        "old_vs_new_failure_props": OUT_DIR / "figure5_old_vs_new_failure_props.csv",
        "old_vs_new_success_state": OUT_DIR / "figure5_old_vs_new_success_state.csv",
        "summary_stats_and_tests": OUT_DIR / "figure5_summary_stats_and_tests.csv",
        "metadata": OUT_DIR / "figure5_verification_metadata.json",
        "report": OUT_DIR / "figure5_failure_verification_report.md",
    }
    if write_candidate_figure:
        paths.update(
            {
                "candidate_svg": FIG_DIR / "figure5_failure_verification_candidate.svg",
                "candidate_pdf": FIG_DIR / "figure5_failure_verification_candidate.pdf",
                "candidate_png": FIG_DIR / "figure5_failure_verification_candidate.png",
            }
        )

    trials.to_csv(paths["trial_level_provenance"], index=False)
    export_disagreement_examples(trials, paths["disagreement_examples"])
    sessions.to_csv(paths["session_inventory"], index=False)
    fig3_compare.to_csv(paths["figure3_session_comparison"], index=False)
    neighbor_definition_table().to_csv(paths["neighbor_definitions"], index=False)
    failure.to_csv(paths["raw_failure_props_per_unit"], index=False)
    choice.to_csv(paths["raw_success_state_per_unit"], index=False)
    summarize_choice_state_variant(trials, "choice_state_two_nearest").to_csv(
        paths["two_nearest_success_state_per_unit"], index=False
    )
    accounting.to_csv(paths["outcome_accounting"], index=False)
    failure_cmp.to_csv(paths["old_vs_new_failure_props"], index=False)
    choice_cmp.to_csv(paths["old_vs_new_success_state"], index=False)
    stats_df.to_csv(paths["summary_stats_and_tests"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "base_dir": base_dir,
                "monkeys": monkeys,
                "experiments": experiments,
                "eligible_answers": sorted(ELIGIBLE_ANSWERS),
                "excluded_answers": sorted(EXCLUDED_ANSWERS),
                "source_cache": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if write_candidate_figure:
        figure_paths = make_candidate_figure(failure, choice)
        paths["candidate_svg"] = figure_paths[0]
        paths["candidate_pdf"] = figure_paths[1]
        paths["candidate_png"] = figure_paths[2]
    write_report(paths, stats_df, failure_cmp, choice_cmp)
    return paths


def recompute_grand_summary(base_dir: str, monkeys: list[str], experiments: list[str]) -> dict:
    trials, _sessions = collect_raw_trials(base_dir, monkeys, experiments)
    return {
        "failure_props_per_unit": summarize_failure_props(trials).to_dict("records"),
        "success_state_per_unit": summarize_choice_state(trials).to_dict("records"),
        "figure5_neighbor_definition": "arc",
        "figure5_neighbor_definition_note": (
            "Panel C adjacent choice uses the recovered arc-adjacent goal map: "
            "edge targets have one adjacent neighbor, interior targets have two."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument(
        "--write-candidate-figure",
        action="store_true",
        help="Also write the candidate Figure 5 verification panel files. Default leaves existing figure files untouched.",
    )
    args = parser.parse_args()
    base_dir = load_base_dir(args.base_dir)
    paths = run(args.base_dir and base_dir or base_dir, args.monkeys, args.experiments, args.write_candidate_figure)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
