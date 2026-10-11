"""
Session-level obstacle-task trajectory metrics for AI ON vs AI OFF.

This script addresses reviewer requests for quantitative comparisons beyond
success rate, including explicit autonomous override time. It writes trial-level
metrics, session-level condition summaries, paired session comparisons, and a
short manuscript-facing statement.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
EXPERIMENT_LABELS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle follow-up",
}
TARGET_WINDOW_RADIUS = 2.3
AVATAR_RADIUS = 0.25
OBSTACLE_RADIUS = 0.45
DWELL_SAMPLES = 10
FALLBACK_DT_S = 0.05
MOVEMENT_SPEED_EPS = 1e-9
OVERRIDE_PAD_MS = 0.0


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_base_dir(cli_base_dir: str | None) -> str:
    if cli_base_dir:
        return cli_base_dir
    import yaml

    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trial_id(trial) -> int | float:
    for key in ("trial", "trial_id", "trialId"):
        value = _get(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return np.nan
    return np.nan


def _vec2_from_pos(value) -> np.ndarray:
    if value is None:
        return np.full(2, np.nan)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(2, np.nan)
    if arr.size >= 3:
        return np.asarray([arr[0], arr[2]], dtype=float)
    if arr.size >= 2:
        return np.asarray([arr[0], arr[1]], dtype=float)
    return np.full(2, np.nan)


def _as_scalar(value) -> float:
    if value is None:
        return np.nan
    try:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return np.nan


def _timestamp_scale_to_seconds(timestamps: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size and np.nanmedian(diffs) > 5.0:
        return 0.001
    return 1.0


def _trajectory_arrays(trial) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj = _get(trial, "avatarTrajectory", None)
    if traj is None:
        return np.array([]), np.array([]), np.array([])
    try:
        x = np.asarray(traj["x"], dtype=float)
        z = np.asarray(traj["z"], dtype=float)
    except Exception:
        return np.array([]), np.array([]), np.array([])
    t = np.asarray(traj.get("time", np.arange(len(x)) * FALLBACK_DT_S), dtype=float)
    n = min(len(x), len(z), len(t))
    return x[:n], z[:n], t[:n]


def _sample_dt_s(t: np.ndarray, n: int) -> float:
    finite = t[np.isfinite(t)]
    if finite.size >= 3:
        diffs = np.diff(finite)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.nanmedian(diffs) * _timestamp_scale_to_seconds(finite))
    return FALLBACK_DT_S


def _duration_s(t: np.ndarray, n: int) -> float:
    finite = t[np.isfinite(t)]
    if finite.size >= 2:
        duration = float((finite[-1] - finite[0]) * _timestamp_scale_to_seconds(finite))
        if np.isfinite(duration) and duration > 0:
            return duration
    return float(max(n - 1, 0) * FALLBACK_DT_S)


def _first_true_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _first_dwell_index(mask: np.ndarray, dwell_samples: int = DWELL_SAMPLES) -> int | None:
    if mask.size < dwell_samples:
        return None
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if ok else 0
        if run >= dwell_samples:
            return int(i - dwell_samples + 1)
    return None


def _movement_onset_raw_time(trial, x: np.ndarray, z: np.ndarray, t: np.ndarray) -> float:
    for velocity_field in ("modelVelocity", "avatarVelocity"):
        velocity = _get(trial, velocity_field, None)
        if not isinstance(velocity, dict):
            continue
        try:
            vx = np.asarray(velocity.get("vx", []), dtype=float)
            vz = np.asarray(velocity.get("vz", []), dtype=float)
            vt = np.asarray(velocity.get("time", []), dtype=float)
        except Exception:
            continue
        n = min(len(vx), len(vz), len(vt))
        if n == 0:
            continue
        speed = np.sqrt(vx[:n] ** 2 + vz[:n] ** 2)
        idx = np.flatnonzero(np.isfinite(speed) & (speed > MOVEMENT_SPEED_EPS))
        if idx.size:
            return float(vt[idx[0]])

    if len(x) >= 2 and len(t) >= 2:
        dx = np.diff(x)
        dz = np.diff(z)
        idx = np.flatnonzero(np.isfinite(dx) & np.isfinite(dz) & ((dx ** 2 + dz ** 2) > MOVEMENT_SPEED_EPS))
        if idx.size:
            return float(t[idx[0]])
    return float(t[0]) if len(t) else np.nan


def _completion_raw_time(trial, t: np.ndarray) -> float:
    for field in ("targetReached", "stop"):
        value = _as_scalar(_get(trial, field, None))
        if np.isfinite(value):
            return value
    return float(t[-1]) if len(t) else np.nan


def _elapsed_from_onset(raw_time: float, onset_raw: float, scale: float) -> float:
    if not (np.isfinite(raw_time) and np.isfinite(onset_raw)):
        return np.nan
    elapsed = (raw_time - onset_raw) * scale
    return float(elapsed) if elapsed >= 0 else np.nan


def _mean_squared_jerk(x: np.ndarray, z: np.ndarray, dt_s: float) -> float:
    if len(x) < 4 or not np.isfinite(dt_s) or dt_s <= 0:
        return np.nan
    jx = np.diff(x, n=3) / (dt_s ** 3)
    jz = np.diff(z, n=3) / (dt_s ** 3)
    jerk2 = jx ** 2 + jz ** 2
    return float(np.nanmean(jerk2)) if jerk2.size else np.nan


def _velocity_log_timestamps(trial) -> np.ndarray:
    recs = _get(trial, "aiVelocities", []) or []
    if len(recs) < 2:
        return np.array([])
    return np.asarray([_as_scalar(rec.get("OutputTimestamp")) for rec in recs], dtype=float)


def _timestamp_scale_to_ms(timestamps: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size and np.nanmedian(diffs) < 1.0:
        return 1000.0
    return 1.0


def _finite_events(events) -> list:
    if events is None:
        return []
    if not isinstance(events, (list, tuple, np.ndarray)):
        events = [events]
    out = []
    for ev in events:
        if isinstance(ev, dict):
            out.append(ev)
            continue
        value = _as_scalar(ev)
        if np.isfinite(value):
            out.append(ev)
    return out


def _events_to_indices(events, output_ts: np.ndarray, n: int) -> list[int]:
    idxs: list[int] = []
    if not events:
        return idxs
    has_ts = np.isfinite(output_ts).any()
    for ev in events:
        if isinstance(ev, (int, np.integer)) and 0 <= int(ev) < n:
            idxs.append(int(ev))
            continue
        if isinstance(ev, dict):
            used = False
            for key in ("idx", "index", "i", "frame"):
                if key in ev:
                    try:
                        ii = int(ev[key])
                        if 0 <= ii < n:
                            idxs.append(ii)
                            used = True
                    except Exception:
                        pass
                    break
            if used:
                continue
            for key in ("time", "timestamp", "ts", "t", "output_ts", "unitytime", "unityrealtime"):
                if key in ev:
                    value = _as_scalar(ev[key])
                    if np.isfinite(value) and has_ts:
                        idxs.append(int(np.nanargmin(np.abs(output_ts - value))))
                    break
            continue
        value = _as_scalar(ev)
        if np.isfinite(value) and has_ts:
            idxs.append(int(np.nanargmin(np.abs(output_ts - value))))
    return sorted(set(i for i in idxs if 0 <= i < n))


def _explicit_override_mask(trial, output_ts: np.ndarray, pad_ms: float = OVERRIDE_PAD_MS) -> tuple[np.ndarray, int]:
    n = len(output_ts)
    mask = np.zeros(n, dtype=bool)
    on_events = _finite_events(_get(trial, "aiControlOn", []))
    off_events = _finite_events(_get(trial, "aiControlOff", []))
    if not on_events and not off_events:
        return mask, 0

    on_idx = _events_to_indices(on_events, output_ts, n)
    off_idx = _events_to_indices(off_events, output_ts, n)
    if not on_idx and not off_idx:
        return mask, 0

    while off_idx and (not on_idx or off_idx[0] < on_idx[0]):
        off_idx.pop(0)

    event_mask = np.zeros(n, dtype=bool)
    j = 0
    segments: list[tuple[int, int]] = []
    for start in sorted(on_idx):
        while j < len(off_idx) and off_idx[j] <= start:
            j += 1
        end = off_idx[j] if j < len(off_idx) else n - 1
        segments.append((start, end))
        event_mask[start : end + 1] = True
        if j < len(off_idx):
            j += 1

    if not segments:
        return event_mask, int(event_mask.sum())

    if np.isfinite(output_ts).any():
        scale = _timestamp_scale_to_ms(output_ts)
        pad_units = pad_ms / scale
        for start, end in segments:
            left = output_ts[start] - pad_units
            right = output_ts[end] + pad_units
            if np.isfinite(left) and np.isfinite(right):
                mask |= (output_ts >= left) & (output_ts <= right)
    else:
        mask = event_mask.copy()

    return mask, int(event_mask.sum())


def _override_metrics(trial, condition: str, trial_duration_s: float, n_traj_samples: int) -> dict:
    if condition != "AI":
        return {
            "n_velocity_samples": 0,
            "n_override_samples": 0,
            "n_override_events": 0,
            "n_150ms_override_events": 0.0,
            "override_duration_s": 0.0,
            "override_fraction": 0.0,
            "has_override": False,
        }

    n_override_events = len(_finite_events(_get(trial, "aiControlOn", [])))
    output_ts = _velocity_log_timestamps(trial)
    if output_ts.size >= 2:
        mask, n_explicit = _explicit_override_mask(trial, output_ts, pad_ms=0.0)
        scale = _timestamp_scale_to_seconds(output_ts)
        finite = output_ts[np.isfinite(output_ts)]
        if finite.size >= 2:
            dt_s = float(np.nanmedian(np.diff(np.sort(finite))) * scale)
        else:
            dt_s = trial_duration_s / max(len(output_ts), 1)
        n_velocity = int(len(output_ts))
        n_override = int(n_explicit)
        override_duration = float(n_override * dt_s) if np.isfinite(dt_s) else np.nan
    else:
        n_velocity = int(n_traj_samples)
        n_override = 0
        override_duration = 0.0

    denom = n_velocity if n_velocity > 0 else np.nan
    return {
        "n_velocity_samples": n_velocity,
        "n_override_samples": n_override,
        "n_override_events": int(n_override_events),
        "n_150ms_override_events": float(n_override / 3.0),
        "override_duration_s": override_duration,
        "override_fraction": float(n_override / denom) if np.isfinite(denom) else np.nan,
        "has_override": bool(n_override > 0),
    }


def _include_trial(experiment: str, trial) -> bool:
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return _get(trial, "obstacleAppearedTime", None) is not None
    return True


def _trial_metrics(trial, monkey: str, experiment: str, session_index: int, session_file: str) -> dict | None:
    if not _include_trial(experiment, trial):
        return None

    try:
        factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        factor = np.nan
    if factor not in (0.0, 1.0):
        return None
    condition = "AI" if factor == 1.0 else "No AI"

    x, z, t = _trajectory_arrays(trial)
    if len(x) < 3:
        return None
    dt_s = _sample_dt_s(t, len(x))
    scale = _timestamp_scale_to_seconds(t)
    duration_s = _duration_s(t, len(x))
    movement_onset_raw = _movement_onset_raw_time(trial, x, z, t)
    completion_raw = _completion_raw_time(trial, t)
    target = _vec2_from_pos(_get(trial, "targetPosition", None))
    obstacle = _vec2_from_pos(_get(trial, "obstaclePosition", None))
    if not np.isfinite(target).all():
        return None

    path_segments = np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)
    path_length = float(np.nansum(path_segments))
    start_to_target_center_distance = float(np.linalg.norm(np.asarray([x[0], z[0]]) - target))
    reference_path_distance = max(start_to_target_center_distance - TARGET_WINDOW_RADIUS, 0.0)
    path_efficiency = start_to_target_center_distance / path_length if path_length > 0 else np.nan
    path_length_ratio = path_length / start_to_target_center_distance if start_to_target_center_distance > 0 else np.nan
    target_window_path_efficiency = reference_path_distance / path_length if path_length > 0 else np.nan
    target_window_path_ratio = path_length / reference_path_distance if reference_path_distance > 0 else np.nan

    speed = path_segments / dt_s if dt_s > 0 else np.full_like(path_segments, np.nan)
    distance_to_target_center = np.sqrt((x - target[0]) ** 2 + (z - target[1]) ** 2)
    inside_target = distance_to_target_center <= TARGET_WINDOW_RADIUS
    first_entry_idx = _first_true_index(inside_target)
    first_dwell_idx = _first_dwell_index(inside_target)
    first_entry_time = t[first_entry_idx] if first_entry_idx is not None else np.nan
    inferred_dwell_start_time = t[first_dwell_idx] if first_dwell_idx is not None else np.nan
    success = int(_get(trial, "answer", None) == 1)

    if np.isfinite(obstacle).all():
        d_obstacle = np.sqrt((x - obstacle[0]) ** 2 + (z - obstacle[1]) ** 2)
        min_clearance = float(np.nanmin(d_obstacle) - AVATAR_RADIUS - OBSTACLE_RADIUS)
    else:
        min_clearance = np.nan

    override = _override_metrics(trial, condition, duration_s, len(x))
    return {
        "monkey": monkey,
        "experiment": experiment,
        "task_label": EXPERIMENT_LABELS.get(experiment, experiment),
        "session_index": session_index,
        "session_file": session_file,
        "session_uid": f"{monkey}|{experiment}|{session_index}|{session_file}",
        "trial_id": _trial_id(trial),
        "condition": condition,
        "success": success,
        "n_trajectory_samples": int(len(x)),
        "trial_duration_s": duration_s,
        "movement_onset_time_raw": movement_onset_raw,
        "completion_time_raw": completion_raw,
        "path_length_m": path_length,
        "straight_distance_m": start_to_target_center_distance,
        "target_window_reference_distance_m": reference_path_distance,
        "path_length_efficiency": path_efficiency,
        "path_length_ratio_to_straight": path_length_ratio,
        "successful_path_length_efficiency": path_efficiency if success else np.nan,
        "successful_path_length_ratio_to_straight": path_length_ratio if success else np.nan,
        "successful_target_window_path_efficiency": target_window_path_efficiency if success else np.nan,
        "successful_target_window_path_ratio": target_window_path_ratio if success else np.nan,
        "time_to_first_target_entry_s": _elapsed_from_onset(first_entry_time, movement_onset_raw, scale),
        "inferred_time_to_dwell_start_s": _elapsed_from_onset(inferred_dwell_start_time, movement_onset_raw, scale),
        "time_to_successful_dwell_s": _elapsed_from_onset(completion_raw, movement_onset_raw, scale) if success else np.nan,
        "mean_executed_speed_mps": float(np.nanmean(speed)) if speed.size else np.nan,
        "successful_mean_executed_speed_mps": float(np.nanmean(speed)) if success and speed.size else np.nan,
        "obstacle_clearance_m": min_clearance,
        "smoothness_mean_squared_jerk": _mean_squared_jerk(x, z, dt_s),
        "successful_smoothness_mean_squared_jerk": _mean_squared_jerk(x, z, dt_s) if success else np.nan,
        **override,
    }


def collect_trial_metrics(monkeys: list[str], experiments: list[str], base_dir: str) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    rows = []
    for experiment in experiments:
        if experiment not in EXPERIMENT_LABELS:
            continue
        print(f"\n[INFO] Processing {experiment}")
        for monkey in monkeys:
            all_trials, *_rest, pkl_files, _ai_trials = load_files(experiment, monkey, base_dir)
            for session_index, trials in enumerate(all_trials):
                session_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                for trial in trials:
                    row = _trial_metrics(trial, monkey, experiment, session_index, session_file)
                    if row is not None:
                        rows.append(row)
                print(f"  {monkey} session {session_index}: collected {len(rows)} total rows")
    return pd.DataFrame(rows)


def summarize_sessions(trials: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "path_length_efficiency",
        "path_length_ratio_to_straight",
        "successful_path_length_efficiency",
        "successful_path_length_ratio_to_straight",
        "successful_target_window_path_efficiency",
        "successful_target_window_path_ratio",
        "time_to_first_target_entry_s",
        "time_to_successful_dwell_s",
        "mean_executed_speed_mps",
        "successful_mean_executed_speed_mps",
        "obstacle_clearance_m",
        "smoothness_mean_squared_jerk",
        "successful_smoothness_mean_squared_jerk",
        "n_override_events",
        "n_150ms_override_events",
        "override_duration_s",
        "override_fraction",
    ]
    rows = []
    group_cols = ["monkey", "experiment", "task_label", "session_index", "session_file", "session_uid", "condition"]
    for keys, group in trials.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        row = {
            **base,
            "n_trials": int(len(group)),
            "success_rate": float(group["success"].mean()),
            "fraction_trials_with_override": float(group["has_override"].mean()),
            "total_trial_time_s": float(group["trial_duration_s"].sum()),
            "total_override_time_s": float(group["override_duration_s"].sum()),
            "fraction_total_time_in_override": (
                float(group["override_duration_s"].sum() / group["trial_duration_s"].sum())
                if group["trial_duration_s"].sum() > 0 else np.nan
            ),
        }
        for col in metric_cols:
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"median_{col}"] = float(group[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(session_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "path_length_efficiency",
        "successful_path_length_ratio_to_straight",
        "successful_path_length_efficiency",
        "successful_target_window_path_ratio",
        "successful_target_window_path_efficiency",
        "time_to_first_target_entry_s",
        "time_to_successful_dwell_s",
        "successful_mean_executed_speed_mps",
        "mean_executed_speed_mps",
        "obstacle_clearance_m",
        "successful_smoothness_mean_squared_jerk",
        "smoothness_mean_squared_jerk",
        "override_duration_s",
        "n_override_events",
        "n_150ms_override_events",
        "override_fraction",
        "fraction_trials_with_override",
        "fraction_total_time_in_override",
    ]
    rows = []
    for task_label, task_df in session_summary.groupby("task_label", dropna=False):
        for summary_prefix in ("mean", "median"):
            for metric in metrics:
                if metric.startswith("fraction_") and summary_prefix != "mean":
                    continue
                col = metric if metric.startswith("fraction_") else f"{summary_prefix}_{metric}"
                if col not in task_df.columns:
                    continue
                piv = task_df.pivot_table(index="session_uid", columns="condition", values=col, aggfunc="first")
                if not {"AI", "No AI"}.issubset(piv.columns):
                    continue
                paired = piv[["AI", "No AI"]].replace([np.inf, -np.inf], np.nan).dropna()
                if paired.empty:
                    continue
                delta = paired["AI"] - paired["No AI"]
                if len(delta) >= 2 and np.any(np.abs(delta) > 0):
                    try:
                        stat, p_value = stats.wilcoxon(paired["AI"], paired["No AI"], alternative="two-sided")
                    except Exception:
                        stat, p_value = np.nan, np.nan
                else:
                    stat, p_value = np.nan, np.nan
                rows.append({
                    "task_label": task_label,
                    "summary": summary_prefix if not metric.startswith("fraction_") else "session",
                    "metric": metric,
                    "n_paired_sessions": int(len(paired)),
                    "ai_mean": float(paired["AI"].mean()),
                    "no_ai_mean": float(paired["No AI"].mean()),
                    "mean_delta_ai_minus_no_ai": float(delta.mean()),
                    "median_delta_ai_minus_no_ai": float(delta.median()),
                    "wilcoxon_statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "wilcoxon_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                })
    return pd.DataFrame(rows)


def plot_metric_deltas(comparisons: pd.DataFrame) -> None:
    if comparisons.empty:
        return
    out_dir = PROJECT_ROOT / "figures" / "Extended Data Figure 3" / "Obstacle session metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    display = comparisons[
        (comparisons["summary"] == "mean")
        & comparisons["metric"].isin([
            "path_length_efficiency",
            "successful_target_window_path_efficiency",
            "time_to_first_target_entry_s",
            "time_to_successful_dwell_s",
            "successful_mean_executed_speed_mps",
            "obstacle_clearance_m",
            "successful_smoothness_mean_squared_jerk",
            "override_fraction",
        ])
    ].copy()
    if display.empty:
        return
    labels = {
        "path_length_efficiency": "Path efficiency to center",
        "successful_target_window_path_efficiency": "Path efficiency",
        "time_to_first_target_entry_s": "First entry",
        "time_to_successful_dwell_s": "Dwell",
        "successful_mean_executed_speed_mps": "Speed",
        "obstacle_clearance_m": "Clearance",
        "successful_smoothness_mean_squared_jerk": "Jerk",
        "override_fraction": "Override",
    }
    tasks = list(display["task_label"].drop_duplicates())
    metrics = list(labels)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.2, 2.4 * len(tasks)), squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        sub = display[display["task_label"] == task].set_index("metric")
        y = [sub.loc[m, "mean_delta_ai_minus_no_ai"] if m in sub.index else np.nan for m in metrics]
        ax.bar(np.arange(len(metrics)), y, color="#3a7f78", width=0.7)
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([labels[m] for m in metrics], rotation=35, ha="right")
        ax.set_ylabel("AI - No AI")
        ax.set_title(task)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(out_dir / f"obstacle_session_metric_deltas.{ext}", dpi=300)
    plt.close(fig)


def write_statement(trials: pd.DataFrame, session_summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    out_path = PROJECT_ROOT / "tables" / "obstacle_session_metrics_statement.md"
    lines = [
        "# Obstacle Session Metrics",
        "",
        "Trial trajectories were summarized for obstacle tasks by AI condition and then paired at the session level.",
        f"Path efficiency was computed on successful trials as the straight-line distance from movement start to the edge of the {TARGET_WINDOW_RADIUS:g} m target window divided by actual trajectory path length; the inverse path-length ratio is retained as an audit field.",
        f"Target-window entry used the configured {TARGET_WINDOW_RADIUS:g} m target-window radius around the cued target.",
        "Time to first target entry is movement onset to first entry into the target window.",
        "Total acquisition time is movement onset to successful completion, using `targetReached` when present and otherwise `trial.stop`; it is reported only for successful trials.",
        "Primary movement speed is mean executed speed on successful trials only. Mean speed over all included trials is retained as `mean_executed_speed_mps` for the optional overall task-level speed analysis.",
        "Primary trajectory smoothness is mean squared jerk on successful trials only. All-trial MSJ is retained as `smoothness_mean_squared_jerk` for the optional overall task-level smoothness analysis.",
        f"`inferred_time_to_dwell_start_s` is also retained as an audit field based on {DWELL_SAMPLES} consecutive target-window samples, but it is not the primary acquisition-time metric.",
        "Obstacle clearance is the minimum avatar-obstacle center distance minus the avatar and obstacle radii.",
        "Autonomous override was quantified from explicit aiControlOn/aiControlOff intervals as override samples divided by velocity-log samples.",
        "The reviewer-facing override event count treats one 150-ms safety override as three 50-ms velocity-log bins, i.e. `n_override_samples / 3`; explicit `aiControlOn` episode counts are retained separately as a raw-log audit field.",
        "",
        f"The trial-level table contains {len(trials):,} trials across {session_summary['session_uid'].nunique() if not session_summary.empty else 0} sessions.",
        "",
    ]

    ai_trials = trials[trials["condition"] == "AI"]
    if not ai_trials.empty:
        total_override = float(ai_trials["override_duration_s"].sum())
        total_time = float(ai_trials["trial_duration_s"].sum())
        total_150ms_events = float(ai_trials["n_150ms_override_events"].sum())
        total_log_episodes = int(ai_trials["n_override_events"].sum())
        event_q25, event_q75 = np.nanpercentile(ai_trials["n_150ms_override_events"], [25, 75])
        trials_with_events = ai_trials[ai_trials["n_150ms_override_events"] > 0]
        if not trials_with_events.empty:
            event_positive_q25, event_positive_q75 = np.nanpercentile(trials_with_events["n_150ms_override_events"], [25, 75])
            event_positive_line = (
                f"- Among shared-control trials with any explicit override, the median was "
                f"{trials_with_events['n_150ms_override_events'].median():.1f} 150-ms event equivalents "
                f"(IQR {event_positive_q25:.1f}-{event_positive_q75:.1f})."
            )
        else:
            event_positive_line = "- Among shared-control trials with any explicit override, event count was unavailable."
        lines.extend([
            "## Override Summary",
            "",
            f"- Total 150-ms override-event equivalents: {total_150ms_events:,.1f}.",
            f"- Total explicit `aiControlOn` episodes retained as raw-log audit field: {total_log_episodes:,}.",
            f"- Shared-control trials with any explicit override: {100.0 * ai_trials['has_override'].mean():.1f}%.",
            f"- Median 150-ms override-event equivalents per shared-control trial: {ai_trials['n_150ms_override_events'].median():.1f} (IQR {event_q25:.1f}-{event_q75:.1f}).",
            event_positive_line,
            f"- Mean 150-ms override-event equivalents per shared-control trial: {ai_trials['n_150ms_override_events'].mean():.2f}.",
            f"- Median override time per shared-control trial: {ai_trials['override_duration_s'].median():.3f} s.",
            f"- Mean override time per shared-control trial: {ai_trials['override_duration_s'].mean():.3f} s.",
            f"- Fraction of total shared-control time in explicit override: {100.0 * total_override / total_time:.2f}%." if total_time > 0 else "- Fraction of total shared-control time in explicit override: unavailable.",
            "",
        ])

    key_metrics = [
        "successful_target_window_path_efficiency",
        "successful_target_window_path_ratio",
        "time_to_first_target_entry_s",
        "time_to_successful_dwell_s",
        "successful_mean_executed_speed_mps",
        "mean_executed_speed_mps",
        "obstacle_clearance_m",
        "successful_smoothness_mean_squared_jerk",
        "smoothness_mean_squared_jerk",
        "n_150ms_override_events",
        "override_fraction",
    ]
    mean_rows = comparisons[(comparisons["summary"] == "mean") & comparisons["metric"].isin(key_metrics)]
    if not mean_rows.empty:
        lines.extend(["## Session-Level Comparisons", ""])
        for _, row in mean_rows.iterrows():
            p_txt = "NA" if not np.isfinite(row["wilcoxon_p_value"]) else f"{row['wilcoxon_p_value']:.3g}"
            lines.append(
                f"- {row['task_label']}, {row['metric']}: mean AI={row['ai_mean']:.4g}, "
                f"No AI={row['no_ai_mean']:.4g}, delta={row['mean_delta_ai_minus_no_ai']:.4g}, "
                f"n={int(row['n_paired_sessions'])}, Wilcoxon p={p_txt}."
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_obstacle_session_metrics(monkeys: list[str], experiments: list[str], base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    trials = collect_trial_metrics(monkeys, experiments, base_dir)
    if trials.empty:
        raise RuntimeError("No obstacle trial metrics were collected.")
    session_summary = summarize_sessions(trials)
    comparisons = paired_comparisons(session_summary)

    trials.to_csv(tables_dir / "obstacle_session_metric_trials.csv", index=False)
    session_summary.to_csv(tables_dir / "obstacle_session_metric_summary.csv", index=False)
    comparisons.to_csv(tables_dir / "obstacle_session_metric_paired_comparisons.csv", index=False)
    plot_metric_deltas(comparisons)
    write_statement(trials, session_summary, comparisons)
    return trials, session_summary, comparisons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute obstacle-task session metrics for AI ON vs AI OFF.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args(argv)
    base_dir = load_base_dir(args.base_dir)
    run_obstacle_session_metrics(args.monkeys, args.experiments, base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
