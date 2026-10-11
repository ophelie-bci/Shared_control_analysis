"""
Respawn old-vs-new target reorientation analysis.

For shared-control Respawn trials, align samples to targetJumpTime and compute
signed evidence that each velocity source favors the new target over the old:

    E(t) = vhat(t) dot uhat_new(t) - vhat(t) dot uhat_old(t)

E < 0 favors the pre-respawn target, E = 0 is neutral, and E > 0 favors the
post-respawn target. The script extracts switching latencies for decoded BCI
input, executed/shared-control output, and any explicit prior vector found in
the log. The controller lag is:

    t_exec - t_BCI

Final manuscript Respawn analyses explicitly use answer code 1 for successful
trials and codes {3,5,6} for unsuccessful experimental outcomes. Codes {4,9,10}
from the broader loader pool are excluded.
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
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
EXPERIMENT = "AI Respawn"
WINDOW_MS = (-1000.0, 2000.0)
BIN_MS = 50.0
SUSTAIN_MS = 200.0
MIN_SPEED = 0.05
ALPHA_FLOOR = 0.2
PRIOR_VECTOR_KEYS = (
    "Prior",
    "prior",
    "PriorVelocity",
    "priorVelocity",
    "TemporalPrior",
    "temporalPrior",
    "MeanPrior",
    "meanPrior",
    "Prediction",
    "prediction",
)


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
        for keys in (("x", "y", "z"), ("Item1", "Item2", "Item3")):
            if all(k in value for k in keys):
                return np.asarray([value[keys[0]], value[keys[1]], value[keys[2]]], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(3, np.nan, dtype=float)
    out = np.full(3, np.nan, dtype=float)
    out[: min(3, len(arr))] = arr[:3]
    return out


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(norm, eps, None)


def _xz(vec3: np.ndarray) -> np.ndarray:
    return np.asarray(vec3, dtype=float)[[0, 2]]


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


def _extract_prior_vector(record) -> np.ndarray:
    for key in PRIOR_VECTOR_KEYS:
        if key in record:
            vec = _vec3(record.get(key))
            if np.any(np.isfinite(vec)):
                return vec
    return np.full(3, np.nan, dtype=float)


def _interpolate_positions_from_trajectory(trial, sample_times: np.ndarray) -> np.ndarray:
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


def _to_relative_ms(sample_times: np.ndarray, event_time: float) -> np.ndarray:
    rel = np.asarray(sample_times, dtype=float) - float(event_time)
    finite = rel[np.isfinite(rel)]
    if finite.size >= 3 and np.nanmedian(np.abs(np.diff(np.sort(sample_times)))) < 1.0:
        rel = rel * 1000.0
    return rel


def _old_new_evidence(vel: np.ndarray, pos: np.ndarray, old_target, new_target) -> np.ndarray:
    vel = np.asarray(vel, dtype=float)
    pos = np.asarray(pos, dtype=float)
    old_xz = _xz(_vec3(old_target))
    new_xz = _xz(_vec3(new_target))
    pos_xz = pos[:, [0, 2]]
    vel_xz = vel[:, [0, 2]]
    valid = (
        np.all(np.isfinite(pos_xz), axis=1)
        & np.all(np.isfinite(vel_xz), axis=1)
        & np.isfinite(old_xz).all()
        & np.isfinite(new_xz).all()
    )
    out = np.full(len(vel_xz), np.nan, dtype=float)
    if not np.any(valid):
        return out
    vhat = _unit(vel_xz[valid])
    u_old = _unit(old_xz[None, :] - pos_xz[valid])
    u_new = _unit(new_xz[None, :] - pos_xz[valid])
    out[valid] = np.sum(vhat * u_new, axis=1) - np.sum(vhat * u_old, axis=1)
    speeds = np.linalg.norm(vel_xz, axis=1)
    out[speeds < MIN_SPEED] = np.nan
    return out


def _sustained_crossing_time(t_rel_ms: np.ndarray, evidence: np.ndarray, sustain_ms: float = SUSTAIN_MS) -> float:
    valid = np.isfinite(t_rel_ms) & np.isfinite(evidence) & (t_rel_ms >= 0)
    if not np.any(valid):
        return np.nan
    t = np.asarray(t_rel_ms[valid], dtype=float)
    e = np.asarray(evidence[valid], dtype=float)
    order = np.argsort(t)
    t = t[order]
    e = e[order]
    ok = e > 0
    for start in np.flatnonzero(ok):
        end = start
        while end + 1 < len(ok) and ok[end + 1]:
            end += 1
        if t[end] - t[start] >= sustain_ms:
            return float(t[start])
    return np.nan


def _window_mean(t_rel_ms: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    mask = (t_rel_ms >= lo) & (t_rel_ms < hi) & np.isfinite(values)
    return float(np.nanmean(values[mask])) if np.any(mask) else np.nan


def extract_trial_reorientation(trial, monkey: str, session_index: int, session_file: str) -> tuple[list[dict], dict] | None:
    if float(_get(trial, "aiVelocityFactor", np.nan)) != 1.0:
        return None
    jump_time = _as_scalar(_get(trial, "targetJumpTime"))
    old_target = _get(trial, "targetPosition")
    new_target = _get(trial, "targetJumpPosition")
    records = _get(trial, "aiVelocities", []) or []
    if not np.isfinite(jump_time) or not records:
        return None

    rows = []
    sample_times, v_bci, v_exec, v_prior, pos, entropy = [], [], [], [], [], []
    for rec in records:
        timestamp = _as_scalar(rec.get("OutputTimestamp", rec.get("InputTimestamp")))
        if not np.isfinite(timestamp):
            continue
        sample_times.append(timestamp)
        v_bci.append(_vec3(rec.get("Input")))
        v_exec.append(_vec3(rec.get("Output")))
        v_prior.append(_extract_prior_vector(rec))
        pos.append(_vec3(rec.get("AvatarPosition")))
        entropy.append(_as_scalar(rec.get("EntropyLb")))

    if len(sample_times) < 3:
        return None

    sample_times = np.asarray(sample_times, dtype=float)
    v_bci = np.vstack(v_bci)
    v_exec = np.vstack(v_exec)
    v_prior = np.vstack(v_prior)
    pos = np.vstack(pos)
    entropy = np.asarray(entropy, dtype=float)

    missing_pos = ~np.all(np.isfinite(pos), axis=1)
    if np.any(missing_pos):
        interp_pos = _interpolate_positions_from_trajectory(trial, sample_times)
        pos[missing_pos] = interp_pos[missing_pos]

    t_rel_ms = _to_relative_ms(sample_times, jump_time)
    keep = (t_rel_ms >= WINDOW_MS[0]) & (t_rel_ms <= WINDOW_MS[1])
    if not np.any(keep):
        return None

    t_rel_ms = t_rel_ms[keep]
    v_bci = v_bci[keep]
    v_exec = v_exec[keep]
    v_prior = v_prior[keep]
    pos = pos[keep]
    entropy = entropy[keep]
    alpha = _robust_alpha(entropy)

    e_bci = _old_new_evidence(v_bci, pos, old_target, new_target)
    e_exec = _old_new_evidence(v_exec, pos, old_target, new_target)
    e_prior = _old_new_evidence(v_prior, pos, old_target, new_target)

    answer = _get(trial, "answer", np.nan)
    trial_id = _get(trial, "trial", _get(trial, "trialIndex", np.nan))
    outcome = "successful" if answer == 1 else "unsuccessful"

    t_bci = _sustained_crossing_time(t_rel_ms, e_bci)
    t_exec = _sustained_crossing_time(t_rel_ms, e_exec)
    t_prior = _sustained_crossing_time(t_rel_ms, e_prior)
    lag = t_exec - t_bci if np.isfinite(t_bci) and np.isfinite(t_exec) else np.nan

    trial_row = {
        "monkey": monkey,
        "experiment": EXPERIMENT,
        "session_index": session_index,
        "session_file": session_file,
        "trial": trial_id,
        "answer": answer,
        "outcome": outcome,
        "n_samples": int(len(t_rel_ms)),
        "t_bci_ms": t_bci,
        "t_exec_ms": t_exec,
        "t_prior_ms": t_prior,
        "bci_crossed_within_window": bool(np.isfinite(t_bci)),
        "exec_crossed_within_window": bool(np.isfinite(t_exec)),
        "prior_crossed_within_window": bool(np.isfinite(t_prior)),
        "controller_lag_ms": lag,
        "alpha_pre_500_0": _window_mean(t_rel_ms, alpha, -500.0, 0.0),
        "alpha_early_0_500": _window_mean(t_rel_ms, alpha, 0.0, 500.0),
        "e_bci_early_0_500": _window_mean(t_rel_ms, e_bci, 0.0, 500.0),
        "e_bci_early_200_800": _window_mean(t_rel_ms, e_bci, 200.0, 800.0),
        "e_exec_early_0_500": _window_mean(t_rel_ms, e_exec, 0.0, 500.0),
        "e_exec_early_200_800": _window_mean(t_rel_ms, e_exec, 200.0, 800.0),
        "e_prior_early_0_500": _window_mean(t_rel_ms, e_prior, 0.0, 500.0),
        "e_prior_early_200_800": _window_mean(t_rel_ms, e_prior, 200.0, 800.0),
    }

    for t, eb, ee, ep, a in zip(t_rel_ms, e_bci, e_exec, e_prior, alpha):
        rows.append({
            "monkey": monkey,
            "experiment": EXPERIMENT,
            "session_index": session_index,
            "session_file": session_file,
            "trial": trial_id,
            "answer": answer,
            "outcome": outcome,
            "time_from_jump_ms": float(t),
            "e_bci": float(eb) if np.isfinite(eb) else np.nan,
            "e_exec": float(ee) if np.isfinite(ee) else np.nan,
            "e_prior": float(ep) if np.isfinite(ep) else np.nan,
            "alpha": float(a) if np.isfinite(a) else np.nan,
        })
    return rows, trial_row


def collect_respawn_reorientation(monkeys: list[str], base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_import_path()
    from src.load import load_files
    from src.respawn_inclusion import (
        RESPAWN_ELIGIBLE_ANSWERS,
        iter_final_respawn_trials,
    )

    sample_rows = []
    trial_rows = []
    for monkey in monkeys:
        all_trials, _all_correct, _all_incorrect, _training, _channels, _nb_channels, pkl_files, _ai = load_files(
            EXPERIMENT,
            monkey,
            base_dir,
        )
        if all_trials == []:
            continue
        for trial, session_index, session_file in iter_final_respawn_trials(all_trials, pkl_files, monkey, EXPERIMENT):
            extracted = extract_trial_reorientation(trial, monkey, session_index, session_file)
            if extracted is None:
                continue
            rows, trial_row = extracted
            trial_row["eligible_answer_code"] = True
            trial_row["included_answer_codes"] = "{" + ",".join(str(x) for x in sorted(RESPAWN_ELIGIBLE_ANSWERS)) + "}"
            for row in rows:
                row["eligible_answer_code"] = True
                row["included_answer_codes"] = trial_row["included_answer_codes"]
            sample_rows.extend(rows)
            trial_rows.append(trial_row)
    return pd.DataFrame(sample_rows), pd.DataFrame(trial_rows)


def make_binned_profiles(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()
    bins = np.arange(WINDOW_MS[0], WINDOW_MS[1] + BIN_MS, BIN_MS)
    df = sample_df.copy()
    df["time_bin_ms"] = pd.cut(df["time_from_jump_ms"], bins=bins, labels=bins[:-1], include_lowest=True)
    records = []
    for (outcome, bin_start), group in df.groupby(["outcome", "time_bin_ms"], observed=True):
        row = {
            "outcome": outcome,
            "time_bin_ms": float(bin_start),
            "n_samples": int(len(group)),
            "n_trials": int(group["trial"].nunique()),
        }
        for col in ("e_bci", "e_exec", "e_prior", "alpha"):
            vals = group[col].astype(float)
            row[f"{col}_mean"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
            row[f"{col}_sem"] = (
                float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals))))
                if np.sum(np.isfinite(vals)) > 1 else np.nan
            )
        records.append(row)
    return pd.DataFrame(records)


def summarize_trials(trial_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "t_bci_ms",
        "t_exec_ms",
        "t_prior_ms",
        "controller_lag_ms",
        "alpha_pre_500_0",
        "alpha_early_0_500",
        "e_bci_early_0_500",
        "e_bci_early_200_800",
        "e_exec_early_0_500",
        "e_exec_early_200_800",
        "e_prior_early_0_500",
        "e_prior_early_200_800",
    ]
    rows = []
    for metric in metrics:
        correct = trial_df.loc[trial_df["outcome"] == "successful", metric].astype(float).dropna()
        incorrect = trial_df.loc[trial_df["outcome"] == "unsuccessful", metric].astype(float).dropna()
        u_stat = p_value = np.nan
        if len(correct) > 0 and len(incorrect) > 0:
            test = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
            u_stat = float(test.statistic)
            p_value = float(test.pvalue)
        rows.append({
            "metric": metric,
            "n_successful": int(len(correct)),
            "successful_median": float(np.nanmedian(correct)) if len(correct) else np.nan,
            "successful_iqr": (
                float(np.nanpercentile(correct, 75) - np.nanpercentile(correct, 25))
                if len(correct) else np.nan
            ),
            "n_unsuccessful": int(len(incorrect)),
            "unsuccessful_median": float(np.nanmedian(incorrect)) if len(incorrect) else np.nan,
            "unsuccessful_iqr": (
                float(np.nanpercentile(incorrect, 75) - np.nanpercentile(incorrect, 25))
                if len(incorrect) else np.nan
            ),
            "unsuccessful_minus_successful_median": (
                float(np.nanmedian(incorrect) - np.nanmedian(correct))
                if len(correct) and len(incorrect) else np.nan
            ),
            "mannwhitney_u": u_stat,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def summarize_crossing_rates(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame()
    rows = []
    for outcome, group in trial_df.groupby("outcome", dropna=False):
        row = {"outcome": outcome, "n_trials": int(len(group))}
        for col, label in (
            ("bci_crossed_within_window", "bci"),
            ("exec_crossed_within_window", "exec"),
            ("prior_crossed_within_window", "prior"),
        ):
            if col not in group:
                row[f"{label}_crossing_fraction"] = np.nan
                row[f"{label}_noncrossing_fraction"] = np.nan
                row[f"{label}_n_crossed"] = 0
                row[f"{label}_n_noncrossing"] = 0
                continue
            crossed = group[col].astype(bool)
            row[f"{label}_crossing_fraction"] = float(crossed.mean()) if len(crossed) else np.nan
            row[f"{label}_noncrossing_fraction"] = float((~crossed).mean()) if len(crossed) else np.nan
            row[f"{label}_n_crossed"] = int(crossed.sum())
            row[f"{label}_n_noncrossing"] = int((~crossed).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def match_on_prejump_alpha(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["alpha_pre_500_0", "controller_lag_ms"]
    if trial_df.empty or any(col not in trial_df for col in required):
        return pd.DataFrame(), pd.DataFrame()

    cols = [
        "monkey",
        "session_index",
        "trial",
        "outcome",
        "alpha_pre_500_0",
        "t_bci_ms",
        "t_exec_ms",
        "controller_lag_ms",
    ]
    successful = (
        trial_df[(trial_df["outcome"] == "successful")]
        .dropna(subset=required)
        .loc[:, cols]
        .copy()
    )
    unsuccessful = (
        trial_df[(trial_df["outcome"] == "unsuccessful")]
        .dropna(subset=required)
        .loc[:, cols]
        .copy()
    )
    if successful.empty or unsuccessful.empty:
        return pd.DataFrame(), pd.DataFrame()

    pairs = []
    for monkey, fail_group in unsuccessful.groupby("monkey", dropna=False):
        available = successful[successful["monkey"] == monkey].copy()
        if available.empty:
            continue
        for _, fail in fail_group.sort_values("alpha_pre_500_0").iterrows():
            if available.empty:
                break
            distances = (available["alpha_pre_500_0"] - fail["alpha_pre_500_0"]).abs()
            match_idx = distances.idxmin()
            match = available.loc[match_idx]
            pairs.append({
                "monkey": monkey,
                "unsuccessful_session": fail["session_index"],
                "unsuccessful_trial": fail["trial"],
                "matched_successful_session": match["session_index"],
                "matched_successful_trial": match["trial"],
                "alpha_unsuccessful": fail["alpha_pre_500_0"],
                "alpha_successful": match["alpha_pre_500_0"],
                "alpha_abs_difference": abs(fail["alpha_pre_500_0"] - match["alpha_pre_500_0"]),
                "lag_unsuccessful_ms": fail["controller_lag_ms"],
                "lag_successful_ms": match["controller_lag_ms"],
                "lag_difference_unsuccessful_minus_successful_ms": (
                    fail["controller_lag_ms"] - match["controller_lag_ms"]
                ),
                "t_bci_unsuccessful_ms": fail["t_bci_ms"],
                "t_bci_successful_ms": match["t_bci_ms"],
                "t_exec_unsuccessful_ms": fail["t_exec_ms"],
                "t_exec_successful_ms": match["t_exec_ms"],
            })
            available = available.drop(index=match_idx)

    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return pair_df, pd.DataFrame()

    diff = pair_df["lag_difference_unsuccessful_minus_successful_ms"].astype(float).dropna()
    alpha_diff = pair_df["alpha_abs_difference"].astype(float).dropna()
    wilcoxon_stat = p_value = np.nan
    if len(diff) >= 1 and np.any(np.asarray(diff) != 0):
        test = stats.wilcoxon(diff, alternative="two-sided")
        wilcoxon_stat = float(test.statistic)
        p_value = float(test.pvalue)

    summary_df = pd.DataFrame([{
        "n_pairs": int(len(pair_df)),
        "median_abs_alpha_difference": float(np.nanmedian(alpha_diff)) if len(alpha_diff) else np.nan,
        "median_lag_difference_unsuccessful_minus_successful_ms": (
            float(np.nanmedian(diff)) if len(diff) else np.nan
        ),
        "wilcoxon_statistic": wilcoxon_stat,
        "p_value": p_value,
    }])
    return pair_df, summary_df


def _zscore_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        values = out[col].astype(float)
        sd = values.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[col] = 0.0
        else:
            out[col] = (values - values.mean()) / sd
    return out


def match_on_bci_evidence(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_cols = ["t_bci_ms", "e_bci_early_200_800", "alpha_pre_500_0", "controller_lag_ms"]
    if trial_df.empty or any(col not in trial_df for col in match_cols):
        return pd.DataFrame(), pd.DataFrame()

    cols = [
        "monkey",
        "session_index",
        "trial",
        "outcome",
        "alpha_pre_500_0",
        "t_bci_ms",
        "e_bci_early_200_800",
        "t_exec_ms",
        "controller_lag_ms",
    ]
    usable = trial_df.dropna(subset=match_cols).loc[:, cols].copy()
    successful = usable[usable["outcome"] == "successful"].copy()
    unsuccessful = usable[usable["outcome"] == "unsuccessful"].copy()
    if successful.empty or unsuccessful.empty:
        return pd.DataFrame(), pd.DataFrame()

    pairs = []
    distance_cols = ["t_bci_ms", "e_bci_early_200_800", "alpha_pre_500_0"]
    for monkey, fail_group in unsuccessful.groupby("monkey", dropna=False):
        available = successful[successful["monkey"] == monkey].copy()
        if available.empty:
            continue
        combined = pd.concat([available, fail_group], ignore_index=False)
        scaled = _zscore_frame(combined, distance_cols)
        available_scaled = scaled.loc[available.index]
        fail_scaled = scaled.loc[fail_group.index]

        for fail_idx, fail in fail_group.iterrows():
            if available.empty:
                break
            diffs = available_scaled.loc[available.index, distance_cols] - fail_scaled.loc[fail_idx, distance_cols]
            distances = np.sqrt((diffs.astype(float) ** 2).sum(axis=1))
            match_idx = distances.idxmin()
            match = available.loc[match_idx]
            pairs.append({
                "monkey": monkey,
                "unsuccessful_session": fail["session_index"],
                "unsuccessful_trial": fail["trial"],
                "matched_successful_session": match["session_index"],
                "matched_successful_trial": match["trial"],
                "matching_distance_z": float(distances.loc[match_idx]),
                "t_bci_unsuccessful_ms": fail["t_bci_ms"],
                "t_bci_successful_ms": match["t_bci_ms"],
                "t_bci_difference_ms": fail["t_bci_ms"] - match["t_bci_ms"],
                "e_bci_200_800_unsuccessful": fail["e_bci_early_200_800"],
                "e_bci_200_800_successful": match["e_bci_early_200_800"],
                "e_bci_200_800_difference": fail["e_bci_early_200_800"] - match["e_bci_early_200_800"],
                "alpha_unsuccessful": fail["alpha_pre_500_0"],
                "alpha_successful": match["alpha_pre_500_0"],
                "alpha_difference": fail["alpha_pre_500_0"] - match["alpha_pre_500_0"],
                "lag_unsuccessful_ms": fail["controller_lag_ms"],
                "lag_successful_ms": match["controller_lag_ms"],
                "lag_difference_unsuccessful_minus_successful_ms": (
                    fail["controller_lag_ms"] - match["controller_lag_ms"]
                ),
                "t_exec_unsuccessful_ms": fail["t_exec_ms"],
                "t_exec_successful_ms": match["t_exec_ms"],
            })
            available = available.drop(index=match_idx)

    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return pair_df, pd.DataFrame()

    diff = pair_df["lag_difference_unsuccessful_minus_successful_ms"].astype(float).dropna()
    wilcoxon_stat = p_value = np.nan
    if len(diff) >= 1 and np.any(np.asarray(diff) != 0):
        test = stats.wilcoxon(diff, alternative="two-sided")
        wilcoxon_stat = float(test.statistic)
        p_value = float(test.pvalue)

    summary_df = pd.DataFrame([{
        "n_pairs": int(len(pair_df)),
        "median_matching_distance_z": float(np.nanmedian(pair_df["matching_distance_z"])),
        "median_abs_t_bci_difference_ms": float(np.nanmedian(np.abs(pair_df["t_bci_difference_ms"]))),
        "median_abs_e_bci_200_800_difference": float(np.nanmedian(np.abs(pair_df["e_bci_200_800_difference"]))),
        "median_abs_alpha_difference": float(np.nanmedian(np.abs(pair_df["alpha_difference"]))),
        "median_lag_difference_unsuccessful_minus_successful_ms": (
            float(np.nanmedian(diff)) if len(diff) else np.nan
        ),
        "wilcoxon_statistic": wilcoxon_stat,
        "p_value": p_value,
    }])
    return pair_df, summary_df


def run_success_regression(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Supporting trial-level logistic regression.

    P-values and confidence intervals use cluster-robust standard errors by
    session, because trials within a session are not independent. Trials with
    undefined t_BCI or controller lag are excluded from this model; their
    prevalence is reported separately by summarize_crossing_rates().
    """
    predictors = ["t_bci_ms", "controller_lag_ms", "alpha_pre_500_0", "e_bci_early_200_800"]
    if trial_df.empty or any(col not in trial_df for col in predictors):
        return pd.DataFrame()
    df = trial_df.dropna(subset=predictors).copy()
    if df.empty or df["outcome"].nunique() < 2:
        return pd.DataFrame()

    df["success"] = (df["outcome"] == "successful").astype(int)
    x = _zscore_frame(df[predictors], predictors)
    x = sm.add_constant(x, has_constant="add")
    y = df["success"]

    try:
        cluster_groups = df["session_index"].astype(str)
        fit = sm.Logit(y, x).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": cluster_groups},
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    params = pd.Series(fit.params, index=x.columns)
    pvalues = pd.Series(fit.pvalues, index=x.columns)
    conf = pd.DataFrame(fit.conf_int(), index=x.columns)
    for name in x.columns:
        rows.append({
            "model": "logit_success_zscored_predictors_clustered_by_session",
            "term": name,
            "n_trials_used": int(len(df)),
            "n_trials_available": int(len(trial_df)),
            "n_trials_excluded_missing_predictors": int(len(trial_df) - len(df)),
            "n_session_clusters": int(df["session_index"].nunique()),
            "coef_log_odds": float(params[name]),
            "odds_ratio": float(np.exp(params[name])),
            "cluster_robust_ci_low_odds_ratio": float(np.exp(conf.loc[name, 0])),
            "cluster_robust_ci_high_odds_ratio": float(np.exp(conf.loc[name, 1])),
            "cluster_robust_p_value": float(pvalues[name]),
            "aic": float(fit.aic),
            "pseudo_r2_mcfadden": float(fit.prsquared),
        })
    return pd.DataFrame(rows)


def plot_profiles(profile_df: pd.DataFrame, output_dir: Path) -> None:
    if profile_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        ("successful", "e_bci"): "#1B6CA8",
        ("successful", "e_exec"): "#2A9D55",
        ("successful", "e_prior"): "#7B4CC2",
        ("unsuccessful", "e_bci"): "#8DB9D9",
        ("unsuccessful", "e_exec"): "#E06C4F",
        ("unsuccessful", "e_prior"): "#B795D9",
    }
    labels = {
        "e_bci": "Decoded BCI",
        "e_exec": "Executed",
        "e_prior": "Prior",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for outcome in ("successful", "unsuccessful"):
        subset = profile_df[profile_df["outcome"] == outcome].sort_values("time_bin_ms")
        if subset.empty:
            continue
        for metric in ("e_bci", "e_exec", "e_prior"):
            mean_col = f"{metric}_mean"
            sem_col = f"{metric}_sem"
            if mean_col not in subset or subset[mean_col].isna().all():
                continue
            x = subset["time_bin_ms"].to_numpy(dtype=float) + BIN_MS / 2.0
            y = subset[mean_col].to_numpy(dtype=float)
            sem = subset[sem_col].to_numpy(dtype=float)
            label = f"{labels[metric]} ({outcome})"
            color = colors[(outcome, metric)]
            linestyle = "-" if outcome == "successful" else "--"
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.8, label=label)
            finite = np.isfinite(y) & np.isfinite(sem)
            if np.any(finite):
                ax.fill_between(x[finite], y[finite] - sem[finite], y[finite] + sem[finite], color=color, alpha=0.16)

    ax.axvline(0, color="0.2", linestyle=":", linewidth=1.2)
    ax.axhline(0, color="0.35", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Time from target jump (ms)")
    ax.set_ylabel("Evidence for new vs old target")
    ax.set_title("Respawn reorientation evidence")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(output_dir / f"respawn_reorientation_evidence.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for outcome, color in (("successful", "#2A9D55"), ("unsuccessful", "#E06C4F")):
        subset = profile_df[profile_df["outcome"] == outcome].sort_values("time_bin_ms")
        if subset.empty or subset["alpha_mean"].isna().all():
            continue
        x = subset["time_bin_ms"].to_numpy(dtype=float) + BIN_MS / 2.0
        y = subset["alpha_mean"].to_numpy(dtype=float)
        sem = subset["alpha_sem"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=1.8, label=outcome)
        finite = np.isfinite(y) & np.isfinite(sem)
        if np.any(finite):
            ax.fill_between(x[finite], y[finite] - sem[finite], y[finite] + sem[finite], color=color, alpha=0.18)
    ax.axvline(0, color="0.2", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Time from target jump (ms)")
    ax.set_ylabel("Prior Confidence Index")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(output_dir / f"respawn_prior_confidence_success_split.{suffix}", dpi=300)
    plt.close(fig)


def _fmt(value: float, unit: str = "") -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.1f}{unit}"


def write_statement(
    summary_df: pd.DataFrame,
    matched_summary_df: pd.DataFrame,
    bci_matched_summary_df: pd.DataFrame,
    crossing_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lookup = {row["metric"]: row for _, row in summary_df.iterrows()}
    lag = lookup.get("controller_lag_ms", {})
    tbci = lookup.get("t_bci_ms", {})
    texec = lookup.get("t_exec_ms", {})
    alpha_pre = lookup.get("alpha_pre_500_0", {})

    lines = [
        "# Respawn Reorientation Lag",
        "",
        "Shared-control Respawn trials were aligned to the target jump.",
        (
            "Evidence is defined as the unit velocity projection toward the new "
            "target minus the projection toward the old target."
        ),
        "",
        (
            f"Decoded BCI switch latency: successful {_fmt(tbci.get('successful_median'), ' ms')}, "
            f"unsuccessful {_fmt(tbci.get('unsuccessful_median'), ' ms')}."
        ),
        (
            f"Executed switch latency: successful {_fmt(texec.get('successful_median'), ' ms')}, "
            f"unsuccessful {_fmt(texec.get('unsuccessful_median'), ' ms')}."
        ),
        (
            f"Controller lag (executed minus BCI): successful {_fmt(lag.get('successful_median'), ' ms')}, "
            f"unsuccessful {_fmt(lag.get('unsuccessful_median'), ' ms')}; "
            f"difference {_fmt(lag.get('unsuccessful_minus_successful_median'), ' ms')}."
        ),
        (
            f"Pre-jump alpha: successful {_fmt(alpha_pre.get('successful_median'))}, "
            f"unsuccessful {_fmt(alpha_pre.get('unsuccessful_median'))}."
        ),
    ]
    if not matched_summary_df.empty:
        matched = matched_summary_df.iloc[0]
        lines.extend([
            "",
            (
                "Nearest-neighbor control matched unsuccessful trials to successful "
                "trials within monkey by pre-jump alpha."
            ),
            (
                f"Matched controller-lag difference: {_fmt(matched.get('median_lag_difference_unsuccessful_minus_successful_ms'), ' ms')} "
                f"over {int(matched.get('n_pairs', 0))} pairs; median absolute alpha mismatch "
                f"{_fmt(matched.get('median_abs_alpha_difference'))}."
            ),
        ])
    if not bci_matched_summary_df.empty:
        matched = bci_matched_summary_df.iloc[0]
        lines.extend([
            "",
            (
                "BCI-matched control matched unsuccessful trials to successful "
                "trials within monkey using t_BCI, E_BCI from 0.2-0.8 s, and pre-jump alpha."
            ),
            (
                f"BCI-matched controller-lag difference: "
                f"{_fmt(matched.get('median_lag_difference_unsuccessful_minus_successful_ms'), ' ms')} "
                f"over {int(matched.get('n_pairs', 0))} pairs; p = "
                f"{_fmt(matched.get('p_value'))}."
            ),
        ])
    if not crossing_df.empty:
        lines.append("")
        for _, row in crossing_df.iterrows():
            lines.append(
                f"{row['outcome']} non-crossing fractions: BCI "
                f"{_fmt(100.0 * row.get('bci_noncrossing_fraction'), '%')}, executed "
                f"{_fmt(100.0 * row.get('exec_noncrossing_fraction'), '%')}."
            )
    if not regression_df.empty:
        lag_rows = regression_df[regression_df["term"] == "controller_lag_ms"]
        if not lag_rows.empty:
            lag = lag_rows.iloc[0]
            lines.extend([
                "",
                (
                    "Trial-level logistic regression controlled for t_BCI, "
                    "controller lag, pre-jump alpha, and early BCI evidence, "
                    "with cluster-robust standard errors by session."
                ),
                (
                    f"Controller-lag odds ratio per z-scored increase: "
                    f"{_fmt(lag.get('odds_ratio'))}; cluster-robust p = "
                    f"{_fmt(lag.get('cluster_robust_p_value'))}. "
                    f"Rows with undefined latency predictors were excluded from "
                    f"this supporting model and are counted in the non-crossing table."
                ),
            ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_respawn_reorientation_lag(monkeys: list[str], base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_df, trial_df = collect_respawn_reorientation(monkeys, base_dir)
    if trial_df.empty:
        raise RuntimeError("No shared-control Respawn trials with AI velocity logs were found.")

    profile_df = make_binned_profiles(sample_df)
    summary_df = summarize_trials(trial_df)
    crossing_df = summarize_crossing_rates(trial_df)
    matched_pairs_df, matched_summary_df = match_on_prejump_alpha(trial_df)
    bci_matched_pairs_df, bci_matched_summary_df = match_on_bci_evidence(trial_df)
    regression_df = run_success_regression(trial_df)

    tables_dir = PROJECT_ROOT / "tables"
    figure_dir = PROJECT_ROOT / "figures" / "Figure 5" / "Respawn reorientation lag"
    tables_dir.mkdir(exist_ok=True)

    sample_df.to_csv(tables_dir / "respawn_reorientation_evidence_samples.csv", index=False)
    trial_df.to_csv(tables_dir / "respawn_reorientation_lag_by_trial.csv", index=False)
    profile_df.to_csv(tables_dir / "respawn_reorientation_profiles.csv", index=False)
    summary_df.to_csv(tables_dir / "respawn_reorientation_lag_summary.csv", index=False)
    crossing_df.to_csv(tables_dir / "respawn_reorientation_crossing_rates.csv", index=False)
    matched_pairs_df.to_csv(tables_dir / "respawn_reorientation_alpha_matched_pairs.csv", index=False)
    matched_summary_df.to_csv(tables_dir / "respawn_reorientation_alpha_matched_summary.csv", index=False)
    bci_matched_pairs_df.to_csv(tables_dir / "respawn_reorientation_bci_matched_pairs.csv", index=False)
    bci_matched_summary_df.to_csv(tables_dir / "respawn_reorientation_bci_matched_summary.csv", index=False)
    regression_df.to_csv(tables_dir / "respawn_reorientation_success_regression.csv", index=False)
    write_statement(
        summary_df,
        matched_summary_df,
        bci_matched_summary_df,
        crossing_df,
        regression_df,
        tables_dir / "respawn_reorientation_lag_statement.md",
    )
    plot_profiles(profile_df, figure_dir)

    print(f"Wrote {tables_dir / 'respawn_reorientation_evidence_samples.csv'}")
    print(f"Wrote {tables_dir / 'respawn_reorientation_lag_by_trial.csv'}")
    print(f"Wrote {tables_dir / 'respawn_reorientation_lag_summary.csv'}")
    print(f"Wrote {figure_dir / 'respawn_reorientation_evidence.svg'}")
    return sample_df, trial_df, summary_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Respawn reorientation-lag analysis.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    base_dir = load_base_dir(args.base_dir)
    run_respawn_reorientation_lag(args.monkeys, base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
