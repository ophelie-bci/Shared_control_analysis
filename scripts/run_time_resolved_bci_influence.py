"""
Time-resolved BCI influence on executed shared-control velocity.

For actual shared-control obstacle trials, this script asks whether the decoded
BCI command remains reflected in the executed command throughout the trial. It
excludes explicit aiControlOn/aiControlOff safety-override periods, plus a
150-ms buffer, because those bins are intentionally prior-only.

The main progress-bin model removes the trivial common effect of target-directed
movement by projecting velocity onto the component orthogonal to the
instantaneous avatar-to-target direction:

    executed_velocity_perp ~ current_BCI_velocity_perp
                             + previous_executed_velocity_perp
                             + prior_confidence

with cluster-robust standard errors by session.
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
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
EXPERIMENT_LABELS = {
    "AI Obstacle": ("Fixed Obstacle", "Fixed Obstacle"),
    "AI Appearing Obstacle": ("Appearing Obstacle", "Appearing Obstacle"),
    "AI Appearing Obstacle 2": ("Appearing Obstacle", "Appearing Obstacle follow-up"),
}
OVERRIDE_PAD_MS = 150.0
N_PROGRESS_BINS = 10
MIN_SPEED = 1e-6
ALPHA_FLOOR = 0.2


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


def _as_scalar(value) -> float:
    if value is None:
        return np.nan
    try:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return np.nan


def _vec3(value) -> np.ndarray:
    if value is None:
        return np.full(3, np.nan, dtype=float)
    if isinstance(value, dict):
        for keys in (("vx", "vy", "vz"), ("x", "y", "z"), ("Item1", "Item2", "Item3")):
            if all(k in value for k in keys):
                try:
                    return np.asarray([value[keys[0]], value[keys[1]], value[keys[2]]], dtype=float)
                except Exception:
                    pass
        if "vx" in value and "vz" in value:
            return np.asarray([value["vx"], 0.0, value["vz"]], dtype=float)
        if "x" in value and "z" in value:
            return np.asarray([value["x"], 0.0, value["z"]], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(3, np.nan, dtype=float)
    out = np.full(3, np.nan, dtype=float)
    if arr.size >= 3:
        out[:] = arr[:3]
    elif arr.size == 2:
        out[:] = [arr[0], 0.0, arr[1]]
    elif arr.size == 1:
        out[:] = [arr[0], 0.0, 0.0]
    return out


def _interp_positions(trial, sample_times: np.ndarray) -> np.ndarray:
    traj = _get(trial, "avatarTrajectory", {})
    if not isinstance(traj, dict):
        return np.full((len(sample_times), 3), np.nan, dtype=float)
    try:
        tt = np.asarray(traj["time"], dtype=float)
        x = np.asarray(traj["x"], dtype=float)
        y = np.asarray(traj.get("y", np.zeros_like(x)), dtype=float)
        z = np.asarray(traj["z"], dtype=float)
    except Exception:
        return np.full((len(sample_times), 3), np.nan, dtype=float)
    n = min(len(tt), len(x), len(y), len(z))
    if n < 2:
        return np.full((len(sample_times), 3), np.nan, dtype=float)
    tt, x, y, z = tt[:n], x[:n], y[:n], z[:n]
    order = np.argsort(tt)
    tt, x, y, z = tt[order], x[order], y[order], z[order]
    return np.column_stack(
        [
            np.interp(sample_times, tt, x, left=np.nan, right=np.nan),
            np.interp(sample_times, tt, y, left=np.nan, right=np.nan),
            np.interp(sample_times, tt, z, left=np.nan, right=np.nan),
        ]
    )


def _unit2(vectors: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norm, eps, None)


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


def _timestamp_scale_to_ms(timestamps: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size and np.nanmedian(diffs) < 1.0:
        return 1000.0
    return 1.0


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


def _robust_alpha(entropy: np.ndarray, floor: float = ALPHA_FLOOR) -> np.ndarray:
    entropy = np.asarray(entropy, dtype=float)
    if entropy.size == 0 or np.all(np.isnan(entropy)):
        return np.full_like(entropy, np.nan, dtype=float)
    hmax_obs = np.nanmax(entropy)
    if not np.isfinite(hmax_obs) or hmax_obs <= 0:
        hmax_obs = np.nanpercentile(entropy, 95)
    k_est = max(2, int(np.round(np.exp(hmax_obs))))
    hmax = float(np.log(k_est))
    alpha = 1.0 - (np.clip(entropy, 0.0, hmax) / hmax)
    return np.clip(np.nan_to_num(alpha, nan=float(floor), posinf=float(floor), neginf=1.0), floor, 1.0)


def _extract_trial_samples(
    trial,
    monkey: str,
    experiment: str,
    task_family: str,
    task_label: str,
    session_index: int,
    session_file: str,
) -> pd.DataFrame:
    try:
        factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        factor = np.nan
    if factor != 1.0:
        return pd.DataFrame()

    recs = _get(trial, "aiVelocities", []) or []
    if len(recs) < 3:
        return pd.DataFrame()

    inputs = np.vstack([_vec3(rec.get("Input")) for rec in recs])
    outputs = np.vstack([_vec3(rec.get("Output")) for rec in recs])
    output_ts = np.asarray([_as_scalar(rec.get("OutputTimestamp")) for rec in recs], dtype=float)
    entropy_lb = np.asarray([_as_scalar(rec.get("EntropyLb")) for rec in recs], dtype=float)
    alpha = _robust_alpha(entropy_lb)
    positions = _interp_positions(trial, output_ts)
    target_xz = _vec3(_get(trial, "targetPosition", None))[[0, 2]]

    finite_ts = output_ts[np.isfinite(output_ts)]
    if finite_ts.size >= 2 and np.nanmax(finite_ts) > np.nanmin(finite_ts):
        progress = (output_ts - np.nanmin(finite_ts)) / (np.nanmax(finite_ts) - np.nanmin(finite_ts))
    else:
        progress = np.linspace(0.0, 1.0, len(recs))
    progress = np.clip(progress, 0.0, 1.0)

    override_mask, n_explicit_override_bins = _explicit_override_mask(trial, output_ts)
    bci_xz = inputs[:, [0, 2]]
    exec_xz = outputs[:, [0, 2]]
    target_vec = target_xz.reshape(1, 2) - positions[:, [0, 2]]
    target_dir = _unit2(target_vec)
    perp_dir = np.column_stack([-target_dir[:, 1], target_dir[:, 0]])
    bci_perp = np.sum(bci_xz * perp_dir, axis=1)
    exec_perp = np.sum(exec_xz * perp_dir, axis=1)
    speed_bci = np.linalg.norm(bci_xz, axis=1)
    speed_exec = np.linalg.norm(exec_xz, axis=1)
    denom = np.clip(speed_bci * speed_exec, MIN_SPEED, None)
    directional_alignment = np.sum(bci_xz * exec_xz, axis=1) / denom
    directional_alignment = np.clip(directional_alignment, -1.0, 1.0)

    keep = (
        ~override_mask
        & np.isfinite(progress)
        & np.isfinite(alpha)
        & np.isfinite(bci_xz).all(axis=1)
        & np.isfinite(exec_xz).all(axis=1)
        & np.isfinite(target_dir).all(axis=1)
        & np.isfinite(bci_perp)
        & np.isfinite(exec_perp)
        & (speed_bci > MIN_SPEED)
        & (speed_exec > MIN_SPEED)
    )
    if not np.any(keep):
        return pd.DataFrame()

    bins = np.minimum((progress * N_PROGRESS_BINS).astype(int), N_PROGRESS_BINS - 1)
    prev_exec = np.vstack([np.full(2, np.nan), exec_xz[:-1]])
    prev_exec_perp = np.sum(prev_exec * perp_dir, axis=1)
    scale = _timestamp_scale_to_ms(output_ts)
    trial_duration_ms = (
        (np.nanmax(finite_ts) - np.nanmin(finite_ts)) * scale
        if finite_ts.size >= 2
        else np.nan
    )

    idxs = np.where(keep)[0]
    return pd.DataFrame(
        {
            "monkey": monkey,
            "experiment": experiment,
            "task_family": task_family,
            "task_label": task_label,
            "session_index": session_index,
            "session_file": session_file,
            "session_uid": f"{monkey}|{experiment}|{session_index}|{session_file}",
            "trial_id": _trial_id(trial),
            "success": int(_get(trial, "answer", None) == 1),
            "sample_index": idxs,
            "progress": progress[idxs],
            "progress_bin": bins[idxs],
            "progress_bin_start": bins[idxs] / N_PROGRESS_BINS,
            "progress_bin_end": (bins[idxs] + 1) / N_PROGRESS_BINS,
            "trial_duration_ms": trial_duration_ms,
            "vx_bci": bci_xz[idxs, 0],
            "vz_bci": bci_xz[idxs, 1],
            "vx_exec": exec_xz[idxs, 0],
            "vz_exec": exec_xz[idxs, 1],
            "vx_exec_prev": prev_exec[idxs, 0],
            "vz_exec_prev": prev_exec[idxs, 1],
            "bci_perp_to_target": bci_perp[idxs],
            "exec_perp_to_target": exec_perp[idxs],
            "exec_prev_perp_to_target": prev_exec_perp[idxs],
            "target_dir_x": target_dir[idxs, 0],
            "target_dir_z": target_dir[idxs, 1],
            "speed_bci": speed_bci[idxs],
            "speed_exec": speed_exec[idxs],
            "directional_alignment": directional_alignment[idxs],
            "alpha": alpha[idxs],
            "explicit_override_bins": n_explicit_override_bins,
            "excluded_override_plus_pad_bins": int(override_mask.sum()),
            "n_velocity_bins_trial": len(recs),
        }
    )


def collect_samples(monkeys: list[str], experiments: list[str], base_dir: str) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    frames = []
    for experiment in experiments:
        if experiment not in EXPERIMENT_LABELS:
            continue
        task_family, task_label = EXPERIMENT_LABELS[experiment]
        print(f"\n[INFO] Processing {experiment}")
        for monkey in monkeys:
            all_trials, *_rest, pkl_files, _ai_trials = load_files(experiment, monkey, base_dir)
            if not all_trials:
                continue
            for session_index, trials in enumerate(all_trials):
                session_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                session_frames = [
                    _extract_trial_samples(
                        trial,
                        monkey,
                        experiment,
                        task_family,
                        task_label,
                        session_index,
                        session_file,
                    )
                    for trial in trials
                ]
                session_frames = [df for df in session_frames if not df.empty]
                if session_frames:
                    session_df = pd.concat(session_frames, ignore_index=True)
                    frames.append(session_df)
                    print(
                        f"  {monkey} session {session_index}: "
                        f"{session_df['trial_id'].nunique()} trials, {len(session_df)} non-override bins"
                    )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return values * np.nan
    return (values - float(values.mean())) / std


def _fit_unique_bci_model(df: pd.DataFrame) -> dict:
    model_base = df[
        ["session_uid", "trial_id", "exec_perp_to_target", "bci_perp_to_target", "exec_prev_perp_to_target", "alpha"]
    ].rename(
        columns={
            "exec_perp_to_target": "exec",
            "bci_perp_to_target": "bci",
            "exec_prev_perp_to_target": "prev_exec",
        }
    )
    for col in ("exec", "bci", "prev_exec", "alpha"):
        model_base[col] = pd.to_numeric(model_base[col], errors="coerce")
    model_base = model_base.replace([np.inf, -np.inf], np.nan).dropna()
    n_sessions = int(model_base["session_uid"].nunique())
    if len(model_base) < 40 or n_sessions < 2:
        return {
            "beta_current_bci": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "partial_r2_current_bci": np.nan,
            "n_model_samples": len(model_base),
            "n_session_clusters": n_sessions,
        }

    model_df = pd.DataFrame(
        {
            "session_uid": model_base["session_uid"].astype(str),
            "exec_z": _zscore(model_base["exec"]),
            "bci_z": _zscore(model_base["bci"]),
            "prev_exec_z": _zscore(model_base["prev_exec"]),
            "alpha_z": _zscore(model_base["alpha"]),
        }
    ).dropna()
    if len(model_df) < 40 or model_df["session_uid"].nunique() < 2:
        return {
            "beta_current_bci": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "partial_r2_current_bci": np.nan,
            "n_model_samples": len(model_df),
            "n_session_clusters": int(model_df["session_uid"].nunique()),
        }

    y = model_df["exec_z"].to_numpy(dtype=float)
    x_full = sm.add_constant(model_df[["bci_z", "prev_exec_z", "alpha_z"]], has_constant="add")
    x_reduced = sm.add_constant(model_df[["prev_exec_z", "alpha_z"]], has_constant="add")
    full = sm.OLS(y, x_full).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["session_uid"].to_numpy()},
    )
    reduced = sm.OLS(y, x_reduced).fit()
    sse_full = float(np.sum(np.square(full.resid)))
    sse_reduced = float(np.sum(np.square(reduced.resid)))
    partial_r2 = (sse_reduced - sse_full) / sse_reduced if sse_reduced > 0 else np.nan
    ci = full.conf_int().loc["bci_z"]
    return {
        "beta_current_bci": float(full.params["bci_z"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(full.pvalues["bci_z"]),
        "partial_r2_current_bci": float(partial_r2),
        "n_model_samples": int(len(model_df)),
        "n_session_clusters": int(model_df["session_uid"].nunique()),
    }


def summarize_by_progress(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["task_family", "progress_bin", "progress_bin_start", "progress_bin_end"]
    for keys, group in samples.groupby(group_cols, dropna=False):
        task_family, progress_bin, bin_start, bin_end = keys
        model = _fit_unique_bci_model(group)
        rows.append(
            {
                "task_family": task_family,
                "progress_bin": int(progress_bin),
                "progress_bin_start": float(bin_start),
                "progress_bin_end": float(bin_end),
                "progress_bin_center": (float(bin_start) + float(bin_end)) / 2.0,
                "n_samples": int(len(group)),
                "n_trials": int(group[["session_uid", "trial_id"]].drop_duplicates().shape[0]),
                "n_sessions": int(group["session_uid"].nunique()),
                "mean_directional_alignment": float(group["directional_alignment"].mean()),
                "sem_directional_alignment": float(group["directional_alignment"].sem()),
                "median_directional_alignment": float(group["directional_alignment"].median()),
                "mean_alpha": float(group["alpha"].mean()),
                **model,
            }
        )
    return pd.DataFrame(rows).sort_values(["task_family", "progress_bin"]).reset_index(drop=True)


def summarize_epoch(bin_summary: pd.DataFrame) -> pd.DataFrame:
    epochs = {
        "early_0_30": (0.0, 0.3),
        "middle_40_70": (0.4, 0.7),
        "late_70_100": (0.7, 1.0),
    }
    rows = []
    for task_family, task_df in bin_summary.groupby("task_family"):
        for epoch, (lo, hi) in epochs.items():
            sub = task_df[
                (task_df["progress_bin_center"] >= lo)
                & (task_df["progress_bin_center"] < hi)
                & np.isfinite(task_df["beta_current_bci"])
            ]
            if sub.empty:
                continue
            rows.append(
                {
                    "task_family": task_family,
                    "epoch": epoch,
                    "n_bins": int(len(sub)),
                    "mean_beta_current_bci": float(sub["beta_current_bci"].mean()),
                    "min_beta_current_bci": float(sub["beta_current_bci"].min()),
                    "max_p_value": float(sub["p_value"].max()),
                    "n_positive_beta_bins": int((sub["beta_current_bci"] > 0).sum()),
                    "mean_directional_alignment": float(sub["mean_directional_alignment"].mean()),
                    "min_directional_alignment": float(sub["mean_directional_alignment"].min()),
                }
            )
    return pd.DataFrame(rows)


def plot_influence(bin_summary: pd.DataFrame) -> None:
    out_dir = PROJECT_ROOT / "figures" / "Extended Data Figure 3" / "BCI influence"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Fixed Obstacle": "#2f7f78", "Appearing Obstacle": "#9a5b23"}
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True)

    for task_family, sub in bin_summary.groupby("task_family"):
        sub = sub.sort_values("progress_bin_center")
        x = sub["progress_bin_center"].to_numpy(dtype=float) * 100.0
        color = colors.get(task_family, "black")
        axes[0].plot(x, sub["mean_directional_alignment"], marker="o", color=color, label=task_family)
        axes[0].fill_between(
            x,
            sub["mean_directional_alignment"] - sub["sem_directional_alignment"],
            sub["mean_directional_alignment"] + sub["sem_directional_alignment"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        axes[1].plot(x, sub["beta_current_bci"], marker="o", color=color, label=task_family)
        axes[1].fill_between(x, sub["ci_low"], sub["ci_high"], color=color, alpha=0.18, linewidth=0)

    axes[0].axhline(0, color="0.2", linewidth=0.8)
    axes[0].set_ylabel("BCI-executed\nalignment", fontsize=9)
    axes[0].set_title("Directional alignment outside safety override", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].axhline(0, color="0.2", linewidth=0.8)
    axes[1].set_ylabel("Unique current BCI\nstandardized beta", fontsize=9)
    axes[1].set_xlabel("Trial progress (%)", fontsize=9)
    axes[1].set_title("Perpendicular BCI influence after previous command + alpha", fontsize=10)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 100)
        ax.grid(axis="y", color="0.9", linewidth=0.8)

    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(out_dir / f"time_resolved_bci_influence.{ext}", dpi=300)
    plt.close(fig)


def write_statement(bin_summary: pd.DataFrame, epoch_summary: pd.DataFrame, samples: pd.DataFrame) -> None:
    out_path = PROJECT_ROOT / "tables" / "time_resolved_bci_influence_statement.md"
    lines = [
        "# Time-Resolved BCI Influence on Executed Command",
        "",
        "Shared-control obstacle trials were analyzed after excluding explicit safety-override bins and a 150-ms buffer around each override epoch.",
        "",
    ]
    excluded = (
        samples[["session_uid", "trial_id", "n_velocity_bins_trial", "excluded_override_plus_pad_bins"]]
        .drop_duplicates()
    )
    lines.append(
        f"The retained sample table contains {len(samples):,} non-override velocity bins from "
        f"{samples[['session_uid', 'trial_id']].drop_duplicates().shape[0]:,} shared-control trials "
        f"across {samples['session_uid'].nunique()} sessions."
    )
    lines.append(
        f"Across trials with velocity logs, {int(excluded['excluded_override_plus_pad_bins'].sum()):,} bins were removed as explicit override plus pad windows."
    )
    lines.append("")

    for task_family in ("Fixed Obstacle", "Appearing Obstacle"):
        sub = epoch_summary[epoch_summary["task_family"] == task_family]
        if sub.empty:
            continue
        lines.append(f"## {task_family}")
        for _, row in sub.iterrows():
            label = row["epoch"].replace("_", "-")
            lines.append(
                f"- {label}: mean unique current-BCI beta = {row['mean_beta_current_bci']:.3f}; "
                f"minimum bin beta = {row['min_beta_current_bci']:.3f}; "
                f"{int(row['n_positive_beta_bins'])}/{int(row['n_bins'])} bins positive; "
                f"maximum bin p = {row['max_p_value']:.3g}; "
                f"mean directional alignment = {row['mean_directional_alignment']:.3f}."
            )
        lines.append("")

    positive_mid_late = []
    for task_family in ("Fixed Obstacle", "Appearing Obstacle"):
        sub = epoch_summary[
            (epoch_summary["task_family"] == task_family)
            & (epoch_summary["epoch"].isin(["middle_40_70", "late_70_100"]))
        ]
        if not sub.empty and (sub["min_beta_current_bci"] > 0).all():
            positive_mid_late.append(task_family)
    if set(positive_mid_late) == {"Fixed Obstacle", "Appearing Obstacle"}:
        lines.extend(
            [
                "Reviewer-facing summary:",
                "",
                "Outside the explicit 150-ms safety-override windows, the component of current decoded BCI velocity orthogonal to the instantaneous target direction retained a positive unique contribution to the corresponding executed-velocity component through the middle and late portions of both obstacle tasks, after accounting for previous executed command and prior confidence.",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_time_resolved_bci_influence(
    monkeys: list[str],
    experiments: list[str],
    base_dir: str,
    samples_input: list[str] | None = None,
    samples_output: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if samples_input:
        samples = pd.concat([pd.read_csv(path) for path in samples_input], ignore_index=True)
    else:
        samples = collect_samples(monkeys, experiments, base_dir)
    if samples.empty:
        raise RuntimeError("No shared-control obstacle velocity samples were found.")
    sample_path = Path(samples_output).resolve() if samples_output else tables_dir / "time_resolved_bci_influence_samples.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(sample_path, index=False)
    if sample_path != tables_dir / "time_resolved_bci_influence_samples.csv":
        samples.to_csv(tables_dir / "time_resolved_bci_influence_samples.csv", index=False)

    bin_summary = summarize_by_progress(samples)
    bin_summary.to_csv(tables_dir / "time_resolved_bci_influence_by_progress.csv", index=False)

    epoch_summary = summarize_epoch(bin_summary)
    epoch_summary.to_csv(tables_dir / "time_resolved_bci_influence_epoch_summary.csv", index=False)

    plot_influence(bin_summary)
    write_statement(bin_summary, epoch_summary, samples)

    print("\n[RESULT] Time-resolved BCI influence by progress bin")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(
            bin_summary[
                [
                    "task_family",
                    "progress_bin",
                    "n_samples",
                    "n_trials",
                    "n_sessions",
                    "mean_directional_alignment",
                    "beta_current_bci",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "partial_r2_current_bci",
                ]
            ].to_string(index=False)
        )
    return samples, bin_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--samples-input", nargs="+", default=None, help="Existing sample CSVs to summarize.")
    parser.add_argument("--samples-output", default=None, help="Optional sample CSV path to write.")
    return parser.parse_args()


def main() -> int:
    os.chdir(PROJECT_ROOT)
    _ensure_import_path()
    args = parse_args()
    base_dir = load_base_dir(args.base_dir)
    run_time_resolved_bci_influence(
        args.monkeys,
        args.experiments,
        base_dir,
        samples_input=args.samples_input,
        samples_output=args.samples_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
