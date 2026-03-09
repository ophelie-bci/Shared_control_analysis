import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict
from scipy.stats import pearsonr, mannwhitneyu
import statsmodels.api as sm

import os, glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d

import numpy as np
from collections import defaultdict

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial import procrustes

from collections import defaultdict, Counter
import numpy as np

def categorize_choice_state(trial,
                            known_goals,
                            goal_idx,
                            neighbors,
                            choice_thresh=0.6,
                            speed_thresh=0.1):
    """
    Returns a high-level choice category for this trial based on modelVelocity.

    category ∈ {
        'correct_choice',
        'wrong_choice',
        'neighbor_choice',   # ambiguous but locally focused on neighbors
        'ambiguous_choice',  # ambiguous and more globally spread
        'no_model'
    }
    """

    mv = getattr(trial, "modelVelocity", None)
    if mv is None:
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    try:
        vx = np.asarray(mv['vx'])
        vz = np.asarray(mv['vz'])
    except Exception:
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    if vx.size == 0 or vz.size == 0:
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    xs = np.asarray(trial.avatarTrajectory['x'])
    zs = np.asarray(trial.avatarTrajectory['z'])

    T = min(len(vx), len(xs), len(zs))
    if T < 5:
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    vx = vx[:T]
    vz = vz[:T]
    xs_ = xs[:T]
    zs_ = zs[:T]

    v = np.stack([vx, vz], axis=1)
    speed = np.linalg.norm(v, axis=1)
    valid = speed > speed_thresh
    if not np.any(valid):
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    v = v[valid]
    xs_ = xs_[valid]
    zs_ = zs_[valid]

    # Per-goal counts
    counts = {g: 0 for g in known_goals}
    total = 0
    for i in range(len(v)):
        vi = v[i]
        vi_norm = np.linalg.norm(vi)
        if vi_norm < 1e-9:
            continue

        best_goal_frame = None
        best_cos = -np.inf
        for (ox, oz) in known_goals:
            gvec = np.array([ox - xs_[i], oz - zs_[i]])
            gnorm = np.linalg.norm(gvec)
            if gnorm < 1e-9:
                continue
            cos = float(np.dot(vi, gvec) / (vi_norm * gnorm + 1e-12))
            if cos > best_cos:
                best_cos = cos
                best_goal_frame = (ox, oz)

        if best_goal_frame is not None and best_cos > 0.0:
            counts[best_goal_frame] += 1
            total += 1

    if total == 0:
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    fracs = {g: c / total for g, c in counts.items() if c > 0}
    if not fracs:
        return "no_model", "no_model", 0.0, 0.0, 0.0, 0.0

    # Best + true
    gx, gz = float(trial.targetPosition[0]), float(trial.targetPosition[2])
    true_goal = (gx, gz)
    best_goal = max(fracs, key=fracs.get)
    best_frac = fracs[best_goal]
    true_frac = fracs.get(true_goal, 0.0)

    # Base choice label (60% rule)
    if best_frac < choice_thresh:
        choice_label = "ambiguous_choice"
    elif best_goal == true_goal:
        choice_label = "correct_choice"
    else:
        choice_label = "wrong_choice"

    # Neighbor vs global ambiguity
    neighbor_frac = 0.0
    other_frac = 0.0
    neighbor_rel = 0.0

    if true_goal in goal_idx:
        idx_true = goal_idx[true_goal]
        neigh_idx = neighbors[idx_true]
        neigh_goals = [known_goals[i] for i in neigh_idx]

        neighbor_frac = sum(fracs.get(g, 0.0) for g in neigh_goals)
        other_frac = max(0.0, 1.0 - true_frac - neighbor_frac)
        non_true_mass = max(1e-9, 1.0 - true_frac)
        neighbor_rel = neighbor_frac / non_true_mass

    # Map to final category
    if choice_label == "ambiguous_choice":
        if true_goal in goal_idx and neighbor_rel >= 0.8:
            category = "neighbor_choice"
        else:
            category = "ambiguous_choice"
    else:
        category = choice_label

    return category, choice_label, float(true_frac), float(neighbor_frac), float(other_frac), float(neighbor_rel)

def build_behavior_df_from_correct(
    correct_on,   # dict: target_label -> list[trial]  (AI ON)
    correct_off,  # dict: target_label -> list[trial]  (AI OFF)
    session_id=None
):
    """
    Build a per-trial DataFrame with workspace trajectories (x,z).

    Returns df_beh with columns:
      - 'target'     (string label, e.g. 'left', 'right', ...)
      - 'ai_status'  (1 = AI ON, 0 = AI OFF)
      - 'traj'       (np.array of shape (T,2) for [x,z])
      - 'session'    (optional session index)
    """
    rows = []

    # AI ON
    for target_label, trials in correct_on.items():
        for tr in trials:
            xs = np.asarray(tr.avatarTrajectory['x'])
            zs = np.asarray(tr.avatarTrajectory['z'])
            if xs.size == 0 or zs.size == 0:
                continue
            traj_xz = np.stack([xs, zs], axis=1)  # (T,2)
            rows.append({
                'target':   target_label,
                'ai_status': 1,
                'traj':     traj_xz,
                'session':  session_id,
            })

    # AI OFF
    for target_label, trials in correct_off.items():
        for tr in trials:
            xs = np.asarray(tr.avatarTrajectory['x'])
            zs = np.asarray(tr.avatarTrajectory['z'])
            if xs.size == 0 or zs.size == 0:
                continue
            traj_xz = np.stack([xs, zs], axis=1)  # (T,2)
            rows.append({
                'target':   target_label,
                'ai_status': 0,
                'traj':     traj_xz,
                'session':  session_id,
            })

    if not rows:
        return pd.DataFrame(columns=['target','ai_status','traj','session'])
    return pd.DataFrame(rows)

def _resample_by_arclength(traj, K=100):
    """traj: (T,D). Return K points uniformly along arc length."""
    if traj.shape[0] < 2:
        return np.tile(traj[0], (K,1))
    diffs = np.diff(traj, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    if s[-1] <= 0:
        return np.repeat(traj[:1], K, axis=0)
    s /= s[-1]
    s_new = np.linspace(0, 1, K)
    out = np.column_stack([np.interp(s_new, s, traj[:,d]) for d in range(traj.shape[1])])
    return out  # (K,D)

def _mean_traj(trials, K=100):
    """trials: (N,T,D). Returns mean (K,D) and std (K,D) after arc-length resampling."""
    rs = np.stack([_resample_by_arclength(t, K) for t in trials], axis=0)  # (N,K,D)
    return rs.mean(axis=0), rs.std(axis=0)

def _pairwise_distance_profile(M1, M2):
    """M1,M2: (K,D) mean trajectories. Returns d(s) and summary stats."""
    d = np.linalg.norm(M1 - M2, axis=1)       # (K,)
    return d, float(d.mean()), float(d.sum()/len(d))  # (profile, mean, AUC)

def compute_pairwise_behavior_metrics_df(
    df_beh,
    cat1,
    cat2,
    ai_status,
    traj_col='traj',
    K_resample=100
):
    """
    Analog of compute_pairwise_metrics_df, but in BEHAVIOR space (x,z).
    Inputs:
      df_beh: DataFrame with columns ['target','ai_status', traj_col]
      cat1, cat2: target labels (e.g., 'left', 'right', ...)
      ai_status: 0, 1 or 'compare_ai'
        - 0 or 1: compare target cat1 vs cat2 within that condition
        - 'compare_ai': same target, cat1==cat2, AI ON vs AI OFF
      traj_col: column name with (T,2) arrays
      K_resample: #points for time-normalized mean trajectories

    Returns dict with:
       'cat1','cat2',
       'mean_distance',
       'distance_auc',
       'procrustes_disparity',
       'mean_traj_cat1','mean_traj_cat2',
       'std_traj_cat1','std_traj_cat2'
    or {"error": ...} if insufficient trials.
    """
    # --- select trials ---
    if ai_status == 'compare_ai':
        # same target, ON vs OFF
        t1 = df_beh[(df_beh['target'] == cat1) & (df_beh['ai_status'] == 1)][traj_col].tolist()
        t2 = df_beh[(df_beh['target'] == cat2) & (df_beh['ai_status'] == 0)][traj_col].tolist()
    else:
        t1 = df_beh[(df_beh['target'] == cat1) & (df_beh['ai_status'] == ai_status)][traj_col].tolist()
        t2 = df_beh[(df_beh['target'] == cat2) & (df_beh['ai_status'] == ai_status)][traj_col].tolist()

    if len(t1) < 3 or len(t2) < 3:
        return {"cat1": cat1, "cat2": cat2, "error": "Insufficient trials"}

    # Convert to array-of-objects; _mean_traj should already handle this
    trials1 = np.array(t1, dtype=object)  # list of (T,2)
    trials2 = np.array(t2, dtype=object)

    # time-normalized mean trajectories + std, using your existing helper
    mean1, std1 = _mean_traj(trials1, K=K_resample)  # (K,2)
    mean2, std2 = _mean_traj(trials2, K=K_resample)  # (K,2)

    # pairwise distance profile (same helper as latents)
    d_prof, d_mean, d_auc = _pairwise_distance_profile(mean1, mean2)

    # Procrustes disparity on the mean curves
    _, _, disparity = procrustes(mean1, mean2)

    return {
        "cat1": cat1,
        "cat2": cat2,
        "mean_distance": d_mean,
        "distance_auc": d_auc,
        "procrustes_disparity": disparity,
        "mean_traj_cat1": mean1,
        "mean_traj_cat2": mean2,
        "std_traj_cat1": std1,
        "std_traj_cat2": std2,
    }

def _pca_reduce(X, k=3):
    Xc = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T

from scipy.spatial import ConvexHull
def hull_volume_of_target_centroids(
    df,
    status,
    latent_col="latents",
    K=80,
    reduce_to=3,
):
    """
    Compute convex-hull volume (or area in 2D) of per-target centroids
    in whatever space is stored in `latent_col`.

    - df: DataFrame with columns ['ai_status', 'target', latent_col]
          where df[latent_col] entries are arrays of shape (T_i, D)
          (T_i may differ between trials).
    - status: ai_status value to select (e.g. 0 = OFF, 1 = ON).
    - K: number of time points to resample each trajectory to.
    - reduce_to: dimensionality for hull (2 for behavior x/z, 3 for latents, etc.).

    Returns:
        float hull volume (area in 2D) or np.nan if not enough data.
    """

    # select ON or OFF
    sub = df[df["ai_status"] == status]
    if sub.empty:
        return np.nan

    centroids = []

    for tgt in sub["target"].unique():
        # all trials for this target
        trials = sub[sub["target"] == tgt][latent_col].tolist()
        if len(trials) < 3:
            # need at least a few trials to get a stable centroid
            continue

        # time-normalize and average across trials
        # _mean_traj handles variable-length trajectories
        mean_traj, _ = _mean_traj(trials, K=K)  # (K, D)
        if not np.any(np.isfinite(mean_traj)):
            continue

        # centroid = average across time of the mean trajectory
        centroid = np.nanmean(mean_traj, axis=0)  # (D,)
        centroids.append(centroid)

    # need at least 3 targets to form a non-degenerate hull
    if len(centroids) < 3:
        return np.nan

    C = np.vstack(centroids)  # (n_targets, D)

    # center the cloud
    C0 = C - C.mean(axis=0, keepdims=True)

    # optional dimensionality reduction
    if reduce_to is not None and C0.shape[1] > reduce_to:
        U, S, Vt = np.linalg.svd(C0, full_matrices=False)
        C0 = C0 @ Vt.T[:, :reduce_to]  # (n_targets, reduce_to)

    # after reduction, we still need more points than dims
    if C0.shape[0] <= C0.shape[1]:
        return np.nan

    try:
        hull = ConvexHull(C0)
        # in 2D this is area; in 3D+ it's volume
        return float(hull.volume)
    except Exception:
        return np.nan

def summarize_behavior_geometry(df_beh, monkey, experiment, K_resample=100):
    """
    Compute pairwise target-geometry metrics in BEHAVIOR space (x,z)
    for AI ON vs AI OFF:
      - mean pairwise mean-trajectory distance (ON and OFF)
      - mean Procrustes disparity (ON and OFF)
      - hull area (convex hull of per-target centroids; ON and OFF)

    Prints a compact summary and returns a dict.
    """
    def _safe_mean(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if len(xs) else np.nan

    targets = sorted(df_beh['target'].unique())
    if len(targets) < 2:
        print(f"[behavior-geom] {monkey} – {experiment}: <2 targets, skipping.")
        return {}

    # --- pairwise metrics for ON and OFF ---
    pair_on = []
    pair_off = []

    for cat1, cat2 in combinations(targets, 2):
        # AI ON
        res_on = compute_pairwise_behavior_metrics_df(
            df_beh, cat1, cat2, ai_status=1,
            traj_col='traj', K_resample=K_resample
        )
        if "error" not in res_on:
            pair_on.append(res_on)

        # AI OFF
        res_off = compute_pairwise_behavior_metrics_df(
            df_beh, cat1, cat2, ai_status=0,
            traj_col='traj', K_resample=K_resample
        )
        if "error" not in res_off:
            pair_off.append(res_off)

    if not pair_on or not pair_off:
        print(f"[behavior-geom] {monkey} – {experiment}: insufficient pairs.")
        return {}

    # mean distance across pairs
    on_md  = [r["mean_distance"] for r in pair_on]
    off_md = [r["mean_distance"] for r in pair_off]
    mean_md_on  = _safe_mean(on_md)
    mean_md_off = _safe_mean(off_md)
    compression_ratio = mean_md_on / mean_md_off if mean_md_off > 0 else np.nan

    # Procrustes disparity across pairs
    on_proc  = [r["procrustes_disparity"] for r in pair_on]
    off_proc = [r["procrustes_disparity"] for r in pair_off]
    mean_proc_on  = _safe_mean(on_proc)
    mean_proc_off = _safe_mean(off_proc)
    proc_ratio = mean_proc_on / mean_proc_off if mean_proc_off > 0 else np.nan

    # Hull area (using your existing hull_volume_of_target_centroids on x,z)
    # Treat 'traj' as latent_col; reduce_to=2 for (x,z).
    vol_on  = hull_volume_of_target_centroids(
        df_beh, status=1, latent_col='traj', K=K_resample, reduce_to=2
    )
    vol_off = hull_volume_of_target_centroids(
        df_beh, status=0, latent_col='traj', K=K_resample, reduce_to=2
    )
    hull_ratio = (vol_on / vol_off) if (
        np.isfinite(vol_on) and np.isfinite(vol_off) and vol_off > 0
    ) else np.nan

    print(f"[behavior-geom] {monkey} – {experiment}:")
    print(f"  Mean pairwise distance OFF = {mean_md_off:.3f} m")
    print(f"  Mean pairwise distance ON  = {mean_md_on:.3f} m "
          f"(ON/OFF = {compression_ratio:.2f}x)")
    print(f"  Procrustes disparity OFF   = {mean_proc_off:.3f}")
    print(f"  Procrustes disparity ON    = {mean_proc_on:.3f} "
          f"(ON/OFF = {proc_ratio:.2f}x)")
    print(f"  Hull area OFF              = {vol_off:.3f}")
    print(f"  Hull area ON               = {vol_on:.3f} "
          f"(ON/OFF = {hull_ratio:.2f}x)")

    return {
        "mean_distance_off": mean_md_off,
        "mean_distance_on":  mean_md_on,
        "compression_ratio": compression_ratio,
        "procrustes_off":    mean_proc_off,
        "procrustes_on":     mean_proc_on,
        "procrustes_ratio":  proc_ratio,
        "hull_off":          vol_off,
        "hull_on":           vol_on,
        "hull_ratio":        hull_ratio,
        "pairwise_on":       pair_on,
        "pairwise_off":      pair_off,
    }

def _resample_path_xy(xs, zs, K=100):
    """Resample (xs,zs) to K points along normalized time."""
    xs = np.asarray(xs)
    zs = np.asarray(zs)
    T  = len(xs)
    if T < 2:
        return np.stack([xs, zs], axis=1)

    t = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, K)
    x_new = np.interp(t_new, t, xs)
    z_new = np.interp(t_new, t, zs)
    return np.stack([x_new, z_new], axis=1)   # (K,2)


def compute_traj_similarity_by_side(traj_groups, monkey, experiment, K=100):
    """
    traj_groups[(target_label, side)] = {'AI': [ (T,2) arrays ], 'No AI': [ (T,2) arrays ]}
    Prints per-target, per-side RMS stats and returns a list of dicts.
    """
    results = []

    for (tlabel, side), cond in sorted(traj_groups.items()):
        paths_on  = cond['AI']
        paths_off = cond['No AI']
        n_on, n_off = len(paths_on), len(paths_off)

        # require at least a few trials in each
        if n_on < 3 or n_off < 3:
            continue

        # resample to K points
        on_rs  = np.stack([_resample_path_xy(p[:,0], p[:,1], K=K) for p in paths_on], axis=0)   # (N_on,K,2)
        off_rs = np.stack([_resample_path_xy(p[:,0], p[:,1], K=K) for p in paths_off], axis=0) # (N_off,K,2)

        mean_on  = on_rs.mean(axis=0)   # (K,2)
        mean_off = off_rs.mean(axis=0)  # (K,2)

        diff   = mean_on - mean_off
        dist   = np.linalg.norm(diff, axis=1)
        rms_on_off = float(np.sqrt(np.mean(dist**2)))
        max_on_off = float(np.max(dist))

        # within-condition RMS to own mean
        dist_on  = np.linalg.norm(on_rs  - mean_on[None, :, :], axis=2)  # (N_on,K)
        dist_off = np.linalg.norm(off_rs - mean_off[None, :, :], axis=2) # (N_off,K)
        rms_within_on  = float(np.sqrt(np.mean(dist_on**2)))
        rms_within_off = float(np.sqrt(np.mean(dist_off**2)))

        print(
            f"[traj-by-side] {monkey} – {experiment}: "
            f"target {tlabel}, side={side}, "
            f"n_on={n_on}, n_off={n_off}, "
            f"RMS={rms_on_off:.3f} m, max={max_on_off:.3f} m "
            f"rms within ON={rms_within_on:.3f} m rms within OFF={rms_within_off:.3f} m"
        )

        results.append({
            'monkey': monkey,
            'experiment': experiment,
            'target': tlabel,
            'side': side,
            'n_on': n_on,
            'n_off': n_off,
            'rms_on_off': rms_on_off,
            'max_on_off': max_on_off,
            'rms_within_on': rms_within_on,
            'rms_within_off': rms_within_off,
        })

    return results

# -------------------------- core helpers --------------------------

def _xz_from(pos):
    if pos is None:
        return None
    if isinstance(pos, dict):
        return (float(pos.get("x", np.nan)), float(pos.get("z", np.nan)))
    try:
        return (float(pos[0]), float(pos[2]))
    except Exception:
        return None

def _thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
    """Return indices RELATIVE to the kept order."""
    idx = list(range(len(keep_idx)))
    # 1) stride
    if stride and stride > 1:
        idx = idx[::stride]
    # 2) time-based
    if min_dt is not None:
        ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
        ts_vals = []
        for k in keep_idx:
            t = ai_recs[k].get(ts_key)
            if hasattr(t, "timestamp"):
                t = t.timestamp()
            ts_vals.append(np.nan if t is None else float(t))
        kept_rel, last_t = [], None
        for j in idx:
            tj = ts_vals[j]
            if np.isnan(tj) or last_t is None or (tj - last_t) >= float(min_dt):
                kept_rel.append(j); last_t = tj
        idx = kept_rel
    # 3) distance-based
    if min_dist is not None and len(idx) > 1:
        kept_rel = [idx[0]]
        last_p = pos[idx[0]]
        for j in idx[1:]:
            if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                kept_rel.append(j); last_p = pos[j]
        idx = kept_rel
    return idx

def _robust_alpha(ent_vals, floor=0.2):
    """α = 1 − normalized entropy (5–95% stretch), clipped to [floor,1]."""
    ent_vals = np.asarray(ent_vals, float)
    if ent_vals.size == 0 or np.all(np.isnan(ent_vals)):
        return np.ones_like(ent_vals)
    lo, hi = np.nanpercentile(ent_vals, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(ent_vals), np.nanmax(ent_vals)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.ones_like(ent_vals)
    a = 1.0 - (ent_vals - lo) / (hi - lo)
    return np.clip(np.nan_to_num(a, nan=1.0), float(floor), 1.0)

def _infer_ai_mask(ai_recs):
    keys = ("UseAIVelocity", "AIOn", "AiOn", "AI_ON", "ai_on")
    mask = []
    for rec in ai_recs:
        v = None
        for k in keys:
            if k in rec:
                v = rec[k]; break
        if v is None:
            mask.append(False)
        else:
            try:
                mask.append(bool(v) if isinstance(v, (bool, int)) else float(v) > 0)
            except Exception:
                mask.append(False)
    return np.array(mask, dtype=bool)

def _cos_and_angle(v1, v2, eps=1e-9):
    """Return cosine similarity and angular error in degrees."""
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1, axis=1) + eps
    n2 = np.linalg.norm(v2, axis=1) + eps
    cos = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return cos, ang, n1, n2

def ai_on_indices_from_trial(trial, eps=1e-6):
    """
    Build AI-ON sample indices from AI toggle events (aiControlOn / aiControlOff),
    and extract aligned velocities:
        - inputs  = vBCI
        - outputs = vAI

    Returns
    -------
    ai_idxs : list[int]
        Indices into trial['aiVelocities'] where AI is ON.
    inputs : ndarray, shape (N, D)  (D=3 if available; otherwise [vx, 0, vz])
        Decoded BCI command per sample.
    outputs : ndarray, shape (N, D)
        Executed / AI-adjusted velocity per sample.
    time_info : dict
        {'input_ts': array(N,), 'output_ts': array(N,)} in seconds (float, or nan).

    Notes
    -----
    - Accepts aiControlOn/aiControlOff as:
        * list of absolute indices (ints), or
        * list of timestamps (float or datetime), or
        * list of dicts with keys like {'idx'| 'index' | 'frame'} or {'time' | 'timestamp'}.
    - If no usable events are found, falls back to heuristic: outputs != inputs.
    """
    recs = trial.get("aiVelocities", None)
    if not recs:
        return [], np.zeros((0, 3)), np.zeros((0, 3)), {"input_ts": np.array([]), "output_ts": np.array([])}
    N = len(recs)

    # ---------- small helpers ----------
    def _first_key(d, keys):
        for k in keys:
            if k in d:
                return k
        return None

    def _as_float_ts(t):
        # Handle datetime, string (with comma decimal), or float
        if t is None:
            return np.nan
        try:
            if hasattr(t, "timestamp"):
                return float(t.timestamp())
            if isinstance(t, str):
                t = t.replace(",", ".")
                return float(t)
            return float(t)
        except Exception:
            return np.nan

    def _to_vec3(v):
        """
        Accept dict {vx,vy,vz}/{x,y,z} or array-like of len 2 or 3.
        If only 2D is present, return [vx, 0, vz]. Fill NaNs if missing.
        """
        if v is None:
            return np.array([np.nan, np.nan, np.nan], float)
        if isinstance(v, dict):
            # Prefer velocity keys then position keys
            for xyz in (("vx", "vy", "vz"), ("x", "y", "z")):
                if all(k in v for k in xyz):
                    try:
                        return np.array([float(v[xyz[0]]), float(v[xyz[1]]), float(v[xyz[2]])], float)
                    except Exception:
                        break
            # Common partials: vx,vz or x,z
            if all(k in v for k in ("vx", "vz")):
                return np.array([float(v.get("vx", np.nan)), 0.0, float(v.get("vz", np.nan))], float)
            if all(k in v for k in ("x", "z")):
                return np.array([float(v.get("x", np.nan)), 0.0, float(v.get("z", np.nan))], float)
            return np.array([np.nan, np.nan, np.nan], float)

        try:
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size >= 3:
                return arr[:3]
            if arr.size == 2:
                return np.array([arr[0], 0.0, arr[1]], float)
            if arr.size == 1:
                return np.array([arr[0], 0.0, 0.0], float)
        except Exception:
            pass
        return np.array([np.nan, np.nan, np.nan], float)

    # ---------- pick field names for velocities & timestamps ----------
    in_candidates  = ("BCIVelocity","UserVelocity","InputVelocity","VelocityInput",
                      "decodedVelocity","BCIAction","UserAction","IntendedVelocity","input","Input")
    out_candidates = ("AIVelocity","OutputVelocity","VelocityOutput","AIAdjustedVelocity",
                      "BlendedVelocity","ExecutedVelocity","velocity","Velocity","Action","ActionOut","output","Output")
    ts_in_candidates  = ("InputTimestamp","input_ts","BCITimestamp","Time","time")
    ts_out_candidates = ("OutputTimestamp","output_ts","AItimestamp","UnityTime","unitytime","unityrealtime")

    in_key = out_key = None
    for r in recs:
        in_key  = in_key  or _first_key(r, in_candidates)
        out_key = out_key or _first_key(r, out_candidates)
        if in_key and out_key:
            break

    # ---------- build arrays ----------
    inputs = np.vstack([_to_vec3(recs[i].get(in_key))  if (in_key  and in_key  in recs[i]) else np.array([np.nan, np.nan, np.nan]) for i in range(N)])
    outputs= np.vstack([_to_vec3(recs[i].get(out_key)) if (out_key and out_key in recs[i]) else np.array([np.nan, np.nan, np.nan]) for i in range(N)])

    input_ts  = np.array([_as_float_ts(recs[i].get(_first_key(recs[i], ts_in_candidates)))  for i in range(N)], float)
    output_ts = np.array([_as_float_ts(recs[i].get(_first_key(recs[i], ts_out_candidates))) for i in range(N)], float)

    # ---------- read aiControlOn / aiControlOff from TRIAL ----------
    on_events  = trial.get("aiControlOn",  [])
    off_events = trial.get("aiControlOff", [])

    def _events_to_indices(events):
        """
        Convert a list of heterogeneous items (int, float/datetime, dict) to
        absolute indices into recs [0..N-1], using OutputTimestamp for mapping if needed.
        """
        idxs = []
        if events is None:
            return idxs
        for ev in (events if isinstance(events, (list, tuple)) else [events]):
            # 1) direct int index
            if isinstance(ev, (int, np.integer)):
                if 0 <= int(ev) < N:
                    idxs.append(int(ev))
                continue
            # 2) dict with index-ish
            if isinstance(ev, dict):
                for k in ("idx","index","i","frame"):
                    if k in ev:
                        try:
                            ii = int(ev[k])
                            if 0 <= ii < N:
                                idxs.append(ii)
                            continue
                        except Exception:
                            pass
                # dict with time-ish
                for k in ("time","timestamp","ts","t","output_ts","unitytime","unityrealtime"):
                    if k in ev:
                        ts = _as_float_ts(ev[k])
                        if np.isfinite(ts) and np.isfinite(output_ts).any():
                            # nearest timestamp index
                            ii = int(np.nanargmin(np.abs(output_ts - ts)))
                            idxs.append(ii)
                        break
                continue
            # 3) timestamp (float or datetime)
            ts = _as_float_ts(ev)
            if np.isfinite(ts) and np.isfinite(output_ts).any():
                ii = int(np.nanargmin(np.abs(output_ts - ts)))
                idxs.append(ii)
        # sanitize & sort unique
        idxs = sorted(set([i for i in idxs if 0 <= i < N]))
        return idxs

    on_idx  = _events_to_indices(on_events)
    off_idx = _events_to_indices(off_events)

    # ---------- build AI-ON mask from on/off pairs ----------
    ai_on_mask = np.zeros(N, dtype=bool)
    if on_idx or off_idx:
        # Ensure alternating segments; if last 'on' has no 'off', it runs to the end.
        on_idx  = sorted(on_idx)
        off_idx = sorted(off_idx)

        # If first event is OFF before any ON, drop early offs
        while off_idx and (not on_idx or off_idx[0] < on_idx[0]):
            off_idx.pop(0)

        # Pair them
        j = 0
        for i, start in enumerate(on_idx):
            # find the next off strictly after start
            while j < len(off_idx) and off_idx[j] <= start:
                j += 1
            end = off_idx[j] if j < len(off_idx) else (N - 1)
            ai_on_mask[start:end+1] = True
            if j < len(off_idx):
                j += 1
    else:
        # ---------- last-resort fallback: outputs != inputs beyond tiny epsilon ----------
        diffs = np.linalg.norm(outputs - inputs, axis=1)
        ai_on_mask = np.isfinite(diffs) & (diffs > eps)

    ai_idxs = np.where(ai_on_mask)[0].tolist()
    time_info = {"input_ts": input_ts, "output_ts": output_ts}
    return ai_idxs, inputs, outputs, time_info

import numpy as np
import pandas as pd

def add_fine_alignment_metrics(df_points):
    df = df_points.copy()

    # unit helper
    def _unit(x, y, eps=1e-9):
        n = np.sqrt(x*x + y*y) + eps
        return x/n, y/n

    # heading unit vectors
    bx, bz = _unit(df["vx_bci"].values, df["vz_bci"].values)
    ax, az = _unit(df["vx_ai"].values,  df["vz_ai"].values)

    # 1) micro-rotation (degrees) and small-angle energy (1-cos)
    cos_ba = bx*ax + bz*az
    cos_ba = np.clip(cos_ba, -1.0, 1.0)
    df["dtheta_deg"] = np.degrees(np.arccos(cos_ba))
    df["one_minus_cos"] = 1.0 - cos_ba  # ~ (theta_rad^2)/2 for small angles

    # 2) goal-alignment gain
    gx, gz = df["tgt_x"].values - df["pos_x"].values, df["tgt_z"].values - df["pos_z"].values
    gx, gz = _unit(gx, gz)
    cos_b_goal = bx*gx + bz*gz
    cos_a_goal = ax*gx + az*gz
    df["delta_cos_goal"] = cos_a_goal - cos_b_goal   # >0 => AI turns toward goal

    # 3) avoidance gain (optional; if obstacle present)
    ox = df["obs_x"].values - df["pos_x"].values
    oz = df["obs_z"].values - df["pos_z"].values
    has_obs = np.isfinite(ox) & np.isfinite(oz)
    ox[~has_obs], oz[~has_obs] = 0.0, 0.0  # dummy for unit()
    ox, oz = _unit(ox, oz)
    # away from obstacle = -r_obs direction
    mx, mz = -ox, -oz
    cos_b_away = bx*mx + bz*mz
    cos_a_away = ax*mx + az*mz
    df["delta_cos_away"] = np.where(has_obs, cos_a_away - cos_b_away, np.nan)  # >0 => AI turns away

    # 4) lateral-error reduction relative to goal line
    # t = goal unit; v_perp = v - (v·t)t
    dot_b_t = bx*gx + bz*gz
    dot_a_t = ax*gx + az*gz
    b_perp_x = bx - dot_b_t*gx; b_perp_z = bz - dot_b_t*gz
    a_perp_x = ax - dot_a_t*gx; a_perp_z = az - dot_a_t*gz
    df["delta_cross_track"] = np.hypot(a_perp_x, a_perp_z) - np.hypot(b_perp_x, b_perp_z)  # <0 => AI reduces lateral drift

    # 5) speed ratio
    df["speed_ratio"] = df["speed_ai"] / (df["speed_bci"] + 1e-9)

    return df
# -------------------------- main function --------------------------

def compute_alignment(
    monkey, experiment, base_dir,
    file_path=None,          # optional: force a single aiLog .pkl
    stride=5, min_dt=None, min_dist=None,
    min_alpha=0.2,
    use_output_ts=True,
    only_answer=None        # e.g., 1 for correct-only; 0/!=1 for incorrect-only; None = all
):
    """
    Alignment analysis between BCI input velocity and AI-adjusted output velocity.

    Assumptions:
      - You have ai_on_indices_from_trial(trial) -> (ai_idxs, inputs, outputs, time_info)
      - 'inputs' = vBCI, 'outputs' = vAI (as requested)
      - Positions/entropy available in trial['aiVelocities'] records.

    Returns:
      df_points : per-sample DataFrame with cosine similarity, angle, α, distances, speeds, AI_ON, etc.
      df_trials : per-trial summary stats.
    """
    # resolve files
    if file_path is None:
        data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
        pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    else:
        pkl_files = [file_path]

    rows = []
    trial_rows = []

    for fp in pkl_files:
        try:
            with open(fp, "rb") as f:
                trials = pickle.load(f)[1]
        except Exception as e:
            print(f"[WARN] Could not load {fp}: {e}")
            continue

        for trial in trials:
            if only_answer is not None:
                ans = trial.get("answer")
                if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
                    continue

            ai_recs = trial.get("aiVelocities")
            if not ai_recs:
                continue

            # --- positions kept indices (for alignment with entropy & distances)
            xs, zs, keep_idx = [], [], []
            for i, rec in enumerate(ai_recs):
                p = _xz_from(rec.get("AvatarPosition"))
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
            if len(keep_idx) < 3:
                continue
            pos = np.column_stack([np.asarray(xs), np.asarray(zs)])

            # --- entropy for ALL records -> select kept -> thin -> α
            ent_raw = []
            for rec in ai_recs:
                e = rec.get("EntropyLb")
                if isinstance(e, (list, tuple, np.ndarray)):
                    e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
                ent_raw.append(np.nan if e is None else float(e))
            ent_raw = np.asarray(ent_raw, float)
            ent_kept = ent_raw[keep_idx]

            idx_rel = _thin_indices(pos, keep_idx, ai_recs,
                                    stride=stride, min_dt=min_dt, min_dist=min_dist,
                                    use_output_ts=use_output_ts)
            if len(idx_rel) < 3:
                continue
            pos = pos[idx_rel]
            ent_kept = ent_kept[idx_rel]
            alpha_conf = _robust_alpha(ent_kept, floor=min_alpha)

            # --- distances
            tgt = _xz_from(trial.get("targetPosition"))
            obs = _xz_from(trial.get("obstaclePosition"))
            d_tgt = np.linalg.norm(pos - np.array(tgt)[None, :], axis=1) if tgt is not None else np.full(len(pos), np.nan)
            d_obs = np.linalg.norm(pos - np.array(obs)[None, :], axis=1) if obs is not None else np.full(len(pos), np.nan)

            # --- AI ON flags aligned to samples
            try:
                ai_idxs, inputs, outputs, _ = ai_on_indices_from_trial(trial)
                ai_on_set = set(ai_idxs)
                ai_on_mask_full = np.array([(k in ai_on_set) for k in keep_idx], dtype=bool)
            except Exception:
                inputs = outputs = None
                ai_on_mask_full = _infer_ai_mask(ai_recs)
                ai_on_mask_full = ai_on_mask_full[keep_idx] if len(ai_on_mask_full) > max(keep_idx) else np.zeros(len(keep_idx), bool)

            ai_on_mask = ai_on_mask_full[idx_rel]

            # --- velocities (vBCI = inputs, vAI = outputs) aligned to samples
            if inputs is None or outputs is None:
                # try to fall back to per-record fields if your logs store them
                print(f"[WARN] Missing inputs/outputs for trial {trial.get('trial')} in {os.path.basename(fp)}; skipping.")
                continue

            # Expect full arrays; keep only XZ components
            try:
                v_bci_all = np.asarray(inputs)[:, [0, 2]]
                v_ai_all  = np.asarray(outputs)[:, [0, 2]]
            except Exception:
                v_bci_all = np.asarray(inputs, float)
                v_ai_all  = np.asarray(outputs, float)
                if v_bci_all.ndim != 2 or v_bci_all.shape[1] != 2:
                    raise ValueError("Inputs/outputs must provide at least (vx, vz).")

            # map from kept/thinned samples to velocity rows
            kept_abs = [keep_idx[j] for j in idx_rel]
            if max(kept_abs) >= len(v_bci_all) or max(kept_abs) >= len(v_ai_all):
                # not enough rows; skip trial
                print(f"[WARN] Velocity arrays shorter than samples in trial {trial.get('trial')} ({os.path.basename(fp)}); skipping this trial.")
                continue

            v_bci = v_bci_all[kept_abs]
            v_ai  = v_ai_all[kept_abs]

            # --- alignment metrics
            cos_sim, ang_deg, sp_bci, sp_ai = _cos_and_angle(v_bci, v_ai, eps=1e-9)
            
            # --- store per-sample
            for j in range(len(pos)):
                # --- store per-sample (ADD pos and target/obstacle coords) ---
                px, pz = pos[j, 0], pos[j, 1]
                tx, tz = (tgt if tgt is not None else (np.nan, np.nan))
                ox, oz = (obs if obs is not None else (np.nan, np.nan))
                rows.append({
                    "file": os.path.basename(fp),
                    "trial": int(trial.get("trial", -1)),
                    "idx": int(j),

                    # policy / distances
                    "alpha": float(alpha_conf[j]),
                    "dist_target": float(d_tgt[j]),
                    "dist_obstacle": float(d_obs[j]),
                    "ai_on": bool(ai_on_mask[j]),

                    # positions (NEW)
                    "pos_x": float(px), "pos_z": float(pz),
                    "tgt_x": float(tx), "tgt_z": float(tz),
                    "obs_x": float(ox), "obs_z": float(oz),

                    # alignment basic
                    "cos_bci_ai": float(cos_sim[j]),
                    "angle_bci_ai_deg": float(ang_deg[j]),
                    "speed_bci": float(sp_bci[j]),
                    "speed_ai": float(sp_ai[j]),

                    # velocities
                    "vx_bci": float(v_bci[j,0]), "vz_bci": float(v_bci[j,1]),
                    "vx_ai":  float(v_ai[j,0]),  "vz_ai":  float(v_ai[j,1]),
                })

            # --- per-trial summary
            good = np.isfinite(cos_sim) & np.isfinite(alpha_conf)
            r_alpha = np.corrcoef(alpha_conf[good], cos_sim[good])[0,1] if good.sum() >= 3 else np.nan
            good_t = np.isfinite(d_tgt) & np.isfinite(cos_sim)
            r_tgt  = np.corrcoef(d_tgt[good_t], cos_sim[good_t])[0,1] if good_t.sum() >= 3 else np.nan
            good_o = np.isfinite(d_obs) & np.isfinite(cos_sim)
            r_obs  = np.corrcoef(d_obs[good_o], cos_sim[good_o])[0,1] if good_o.sum() >= 3 else np.nan

            trial_rows.append({
                "file": os.path.basename(fp),
                "trial": int(trial.get("trial", -1)),
                "n_samples": int(len(pos)),
                "cos_mean": float(np.nanmean(cos_sim)),
                "cos_median": float(np.nanmedian(cos_sim)),
                "angle_median_deg": float(np.nanmedian(ang_deg)),
                "frac_aligned_cos>0.8": float(np.nanmean(cos_sim > 0.8)),
                "r_alpha_cos": float(r_alpha),
                "r_dist_target_cos": float(r_tgt),
                "r_dist_obstacle_cos": float(r_obs),
            })

    df_points = pd.DataFrame(rows)
    df_trials = pd.DataFrame(trial_rows)
    return df_points, df_trials

# -------------------------- optional plotting --------------------------

def plot_alignment_binned(df_points, x="alpha", nbins=12, title="", invert_x=False):
    """
    Binned trend of cosine alignment vs a regressor (alpha or distance).
    """
    xvals = np.asarray(df_points[x], float)
    yvals = np.asarray(df_points["cos_bci_ai"], float)
    good = np.isfinite(xvals) & np.isfinite(yvals)
    if good.sum() < 5:
        print("[WARN] Not enough valid samples to plot."); return

    lo, hi = np.percentile(xvals[good], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(xvals[good]), np.nanmax(xvals[good])
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means, ns = [], []
    for i in range(nbins):
        m = (xvals >= edges[i]) & (xvals < edges[i+1]) & good
        means.append(np.nan if m.sum() == 0 else np.nanmean(yvals[m]))
        ns.append(int(m.sum()))
    means = np.asarray(means, float)

    r = np.corrcoef(xvals[good], yvals[good])[0,1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(xvals[good], yvals[good], s=6, alpha=0.25, label="samples")
    ax.plot(centers, means, "-o", lw=2, label="binned mean")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel("cosine(vBCI, vAI)")
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(frameon=True)
    ax.text(0.02, 0.95, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="top")
    if invert_x: ax.invert_xaxis()
    plt.tight_layout(); plt.show()


# ---------- core: BCI→target alignment (takeover excluded) ----------
def input_to_target_alignment_series(trial, vmin=0.05, exclude_takeover=True):
    """
    Return (ts, cos_in_target) where cos_in_target is the cosine between
    BCI velocity (Input) and the vector from current position to the TRUE target.
    Excludes small speeds and (optionally) takeover window.
    """
    # Steps (timestamps + BCI velocities)
    aiV = trial.get("aiVelocities") if isinstance(trial, dict) else getattr(trial, "aiVelocities", None)
    if not aiV:
        return np.array([]), np.array([])

    ts  = np.array([_as_float_timestamp(s.get("InputTimestamp")) for s in aiV], float)
    In  = np.array([_to_xz(s.get("Input")) for s in aiV], float)   # (N,2) xz

    # Speed gate to avoid angle blow-ups
    in_norm = np.linalg.norm(In, axis=1)
    valid = np.isfinite(ts) & np.isfinite(in_norm) & (in_norm >= vmin)

    # Exclude takeover
    if exclude_takeover:
        on  = _as_float_timestamp(trial.get("aiControlOn")  if isinstance(trial, dict) else getattr(trial, "aiControlOn",  None))
        off = _as_float_timestamp(trial.get("aiControlOff") if isinstance(trial, dict) else getattr(trial, "aiControlOff", None))
        if np.isfinite(on) and np.isfinite(off) and off >= on:
            valid &= ~((ts >= on) & (ts <= off))

    if not np.any(valid):
        return np.array([]), np.array([])

    ts  = ts[valid]
    In  = In[valid]
    in_norm = in_norm[valid]

    # Approximate positions along the trajectory aligned to velocity samples
    x, z = _get_traj(trial)
    if x is None or len(x) == 0:
        return np.array([]), np.array([])

    idx = np.linspace(0, len(x)-1, len(ts)).astype(int)
    pos = np.c_[x[idx], z[idx]]

    # True target center
    tgt = trial.get("targetPosition") if isinstance(trial, dict) else getattr(trial, "targetPosition", None)
    gx, gz = float(tgt[0]), float(tgt[2])
    goal_vec = np.c_[gx - pos[:,0], gz - pos[:,1]]
    goal_norm = np.linalg.norm(goal_vec, axis=1)

    # Valid where goal direction is defined
    g_valid = goal_norm > 1e-9
    if not np.any(g_valid):
        return np.array([]), np.array([])

    In  = In[g_valid]; in_norm = in_norm[g_valid]
    ts  = ts[g_valid]; goal_vec = goal_vec[g_valid]; goal_norm = goal_norm[g_valid]

    # Cosine between In and goal ray: (In · ĝ) / ||In||
    gh = goal_vec / goal_norm[:, None]
    cos_in_target = np.einsum("ij,ij->i", In, gh) / (in_norm + 1e-12)
    # clip to [-1,1] for safety
    cos_in_target = np.clip(cos_in_target, -1.0, 1.0)

    return ts, cos_in_target

# ---------- event-aligned profiles (apex / target entry) ----------
def event_aligned_input_to_target(trials, label, event="obstacle", window=(-0.4, 0.4), n_bins=100, vmin=0.05):
    bins = [[] for _ in range(n_bins)]
    edges = np.linspace(window[0], window[1], n_bins+1)
    centers = 0.5*(edges[:-1]+edges[1:])

    for tr in trials:
        ts, cosIN = input_to_target_alignment_series(tr, vmin=vmin, exclude_takeover=True)
        if ts.size == 0: 
            continue

        # pick event index from trajectory
        if event == "obstacle":
            idx = event_idx_obstacle_apex(tr)
        else:
            idx = event_idx_target_entry(tr)
        if idx is None: 
            continue

        x, _ = _get_traj(tr)
        ev_norm = idx / max(1, (len(x)-1)) if x is not None else 0.5

        ts_norm = _norm_times(ts)
        t_rel = ts_norm - ev_norm
        keep = (t_rel >= window[0]) & (t_rel <= window[1])
        if not np.any(keep):
            continue

        b = np.digitize(t_rel[keep], edges) - 1
        for c, bi in zip(cosIN[keep], b):
            if 0 <= bi < n_bins:
                bins[bi].append(c)

    def _mean_sem(bins):
        m, s = [], []
        for lst in bins:
            if len(lst) == 0:
                m.append(np.nan); s.append(np.nan)
            else:
                arr = np.asarray(lst, float)
                m.append(np.nanmean(arr))
                s.append(np.nanstd(arr, ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
        return np.array(m), np.array(s)

    m, s = _mean_sem(bins)
    return {"time": centers, "mean": m, "sem": s, "label": label, "event": event, "window": window}

# ---------- windowed group stats (overshoot vs correct) ----------
def window_input_to_target_mean(trial, center="target", window=(-0.05,+0.05), vmin=0.05):
    ts, c = input_to_target_alignment_series(trial, vmin=vmin, exclude_takeover=True)
    if ts.size == 0: return np.nan
    idx = event_idx_obstacle_apex(trial) if center=="obstacle" else event_idx_target_entry(trial)
    if idx is None: return np.nan
    x,_ = _get_traj(trial)
    ev_norm = idx / max(1, (len(x)-1)) if x is not None else 0.5
    t_rel = _norm_times(ts) - ev_norm
    keep = (t_rel >= window[0]) & (t_rel <= window[1])
    return float(np.nanmean(c[keep])) if np.any(keep) else np.nan

def compute_additive_vs_multiplicative_effect(success_rate_ai_on_all, success_rate_ai_off_all):
    # Convert to numpy arrays
    sr_on  = np.array(success_rate_ai_on_all)
    sr_off = np.array(success_rate_ai_off_all)

    # Drop NaNs if any
    mask = ~np.isnan(sr_on) & ~np.isnan(sr_off)
    sr_on, sr_off = sr_on[mask], sr_off[mask]

    if len(sr_on) > 2:
        # --- Additive model: SR_on = intercept + 1*SR_off
        X_add = sm.add_constant(sr_off)   # adds intercept
        model_add = sm.OLS(sr_on, X_add).fit()

        # --- Multiplicative model: SR_on = slope*SR_off (no intercept)
        model_mult = sm.OLS(sr_on, sr_off).fit()

        print("\n[Additive model]")
        print(model_add.summary())
        print("\n[Multiplicative model]")
        print(model_mult.summary())

        # --- Compare AIC
        print(f"\nModel comparison: AIC additive={model_add.aic:.2f}, multiplicative={model_mult.aic:.2f}")

def summarize_input_alignment_windows(overshoot_trials, correct_trials, vmin=0.05):
    W = {
        "Apex ±50ms":    ("obstacle", (-0.05, +0.05)),
        "Pre-entry":     ("target",   (-0.10, -0.02)),
        "Entry":         ("target",   (-0.01, +0.03)),
        "Post-entry":    ("target",   ( 0.00, +0.05)),
    }
    out = {}
    for name, (center, win) in W.items():
        ov = np.array([window_input_to_target_mean(t, center, win, vmin=vmin) for t in overshoot_trials], float)
        co = np.array([window_input_to_target_mean(t, center, win, vmin=vmin) for t in correct_trials],   float)
        ov = ov[np.isfinite(ov)]; co = co[np.isfinite(co)]
        if ov.size and co.size:
            U = mannwhitneyu(ov, co, alternative="two-sided")
            print(f"{name}: over={np.nanmean(ov):.3f} | corr={np.nanmean(co):.3f} | Δ={np.nanmean(co)-np.nanmean(ov):.3f} | U={U.statistic:.0f}, p={U.pvalue:.3e}")
            out[name] = {"over": ov, "corr": co, "U": U}
        else:
            print(f"{name}: insufficient data")
            out[name] = None
    return out

def prior_entropy_bounds_series(trial, clip01=True, exclude_takeover=True):
    """
    Return (ts, H) using per-step EntropyUb/EntropyLb from aiVelocities.
    H = 0.5 * (Lb + Ub) when both present; else whichever is present.
    If clip01=True, clip to [0,1].
    """
    aiV = trial.get("aiVelocities") if isinstance(trial, dict) else getattr(trial, "aiVelocities", None)
    if not aiV:
        return np.array([]), np.array([])

    ts = np.array([_as_float_timestamp(s.get("InputTimestamp")) for s in aiV], float)

    H = []
    for s in aiV:
        ub = s.get("EntropyUb"); lb = s.get("EntropyLb")
        ub = float(ub) if ub is not None and np.isfinite(ub) else np.nan
        lb = float(lb) if lb is not None and np.isfinite(lb) else np.nan
        if np.isfinite(ub) and np.isfinite(lb):
            h = 0.5 * (lb + ub)
        elif np.isfinite(ub):
            h = ub
        elif np.isfinite(lb):
            h = lb
        else:
            h = np.nan
        H.append(h)
    H = np.asarray(H, float)
    if clip01:
        H = np.clip(H, 0.0, 1.0)

    # exclude takeover window
    if exclude_takeover:
        on  = _as_float_timestamp(trial.get("aiControlOn")  if isinstance(trial, dict) else getattr(trial, "aiControlOn",  None))
        off = _as_float_timestamp(trial.get("aiControlOff") if isinstance(trial, dict) else getattr(trial, "aiControlOff", None))
        if np.isfinite(on) and np.isfinite(off) and off >= on:
            keep = ~((ts >= on) & (ts <= off))
            ts, H = ts[keep], H[keep]

    m = np.isfinite(ts) & np.isfinite(H)
    return ts[m], H[m]
def event_aligned_entropy_bounds(trials, label, event="obstacle", window=(-0.4, 0.4), n_bins=100):
    bins = [[] for _ in range(n_bins)]
    edges = np.linspace(window[0], window[1], n_bins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for tr in trials:
        ts, H = prior_entropy_bounds_series(trial=tr, clip01=True, exclude_takeover=True)
        if ts.size == 0:
            continue

        # pick event index from trajectory
        idx = event_idx_obstacle_apex(tr) if event == "obstacle" else event_idx_target_entry(tr)
        if idx is None:
            continue

        x, _ = _get_traj(tr)
        ev_norm = idx / max(1, (len(x) - 1)) if x is not None else 0.5

        ts_norm = _norm_times(ts)
        t_rel = ts_norm - ev_norm
        keep = (t_rel >= window[0]) & (t_rel <= window[1])
        if not np.any(keep):
            continue

        b = np.digitize(t_rel[keep], edges) - 1
        for h, bi in zip(H[keep], b):
            if 0 <= bi < n_bins:
                bins[bi].append(h)

    means, sems = [], []
    for lst in bins:
        if not lst:
            means.append(np.nan); sems.append(np.nan)
        else:
            arr = np.asarray(lst, float)
            means.append(np.nanmean(arr))
            sems.append(np.nanstd(arr, ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else 0.0)

    return np.array(centers), np.array(means), np.array(sems)
def window_entropy_bound_mean(trial, center="obstacle", window=(-0.05, 0.05)):
    ts, H = prior_entropy_bounds_series(trial=trial, clip01=True, exclude_takeover=True)
    if ts.size == 0: return np.nan
    idx = event_idx_obstacle_apex(trial) if center=="obstacle" else event_idx_target_entry(trial)
    if idx is None: return np.nan
    x,_ = _get_traj(trial)
    ev_norm = idx / max(1, (len(x)-1)) if x is not None else 0.5
    t_rel = _norm_times(ts) - ev_norm
    keep = (t_rel >= window[0]) & (t_rel <= window[1])
    return float(np.nanmean(H[keep])) if np.any(keep) else np.nan

def summarize_entropy_bounds_windows(overshoot_trials, correct_trials):
    W = {
        "Apex ±50ms":    ("obstacle", (-0.05, +0.05)),
        "Pre-entry":     ("target",   (-0.10, -0.02)),
        "Entry":         ("target",   (-0.01, +0.03)),
        "Post-entry":    ("target",   ( 0.00, +0.05)),
    }
    for name, (center, win) in W.items():
        ov = np.array([window_entropy_bound_mean(t, center, win) for t in overshoot_trials], float)
        co = np.array([window_entropy_bound_mean(t, center, win) for t in correct_trials],   float)
        ov = ov[np.isfinite(ov)]; co = co[np.isfinite(co)]
        if ov.size and co.size:
            U = mannwhitneyu(ov, co, alternative="two-sided")
            print(f"{name}: over={np.nanmean(ov):.3f} | corr={np.nanmean(co):.3f} | Δ={np.nanmean(co)-np.nanmean(ov):.3f} | U={U.statistic:.0f}, p={U.pvalue:.3e}")
        else:
            print(f"{name}: insufficient data")
def entropy_bounds_action_correlation(trials, center="target", window=(-0.05, +0.05), vmin=0.05):
    H_all, gpar_all, perp_all = [], [], []
    for tr in trials:
        # controller metrics on velocity timestamps
        ts, _, gpar, perp = correction_metrics_for_trial(tr, vmin=vmin)
        if ts.size == 0: continue

        # entropy series
        tH, H = prior_entropy_bounds_series(tr, clip01=True, exclude_takeover=True)
        if tH.size == 0: continue

        # event center
        idx = event_idx_target_entry(tr) if center=="target" else event_idx_obstacle_apex(tr)
        if idx is None: continue
        x,_ = _get_traj(tr); ev_norm = idx / max(1, (len(x)-1)) if x is not None else 0.5

        tn  = _norm_times(ts); tHn = _norm_times(tH)
        t_rel  = tn  - ev_norm
        tH_rel = tHn - ev_norm
        keep_v = (t_rel >= window[0]) & (t_rel <= window[1])

        # interpolate H onto ts within the window
        if not np.any(keep_v): continue
        order = np.argsort(tH)
        H_interp = np.interp(ts[keep_v], tH[order], H[order])

        H_all.append(H_interp)
        gpar_all.append(gpar[keep_v])
        perp_all.append(perp[keep_v])

    if not H_all: return np.nan, np.nan
    H_all    = np.concatenate(H_all)
    gpar_all = np.concatenate(gpar_all)
    perp_all = np.concatenate(perp_all)
    r_g = pearsonr(H_all, gpar_all)[0]
    r_p = pearsonr(H_all, perp_all)[0]
    return r_g, r_p


# ===================== helpers (single definitions) =====================
def _to_xz(vec):
    """Return (x,z) as floats; works with dicts and sequences; returns (nan,nan) if unknown."""
    if vec is None:
        return np.nan, np.nan
    if isinstance(vec, dict):
        if "x" in vec and "z" in vec:   return float(vec["x"]), float(vec["z"])
        if "vx" in vec and "vz" in vec: return float(vec["vx"]), float(vec["vz"])
        for k in ("v", "vel", "velocity"):
            if k in vec:
                return _to_xz(vec[k])
        return np.nan, np.nan
    a = np.asarray(vec, float).ravel()
    if a.size >= 3: return float(a[0]), float(a[2])   # [x,y,z] → (x,z)
    if a.size == 2: return float(a[0]), float(a[1])   # [x,z]
    return np.nan, np.nan

def _as_float_timestamp(x):
    """Scalarize timestamps that might be lists/arrays/datetimes."""
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).ravel()
        x = arr[0] if arr.size else np.nan
    if hasattr(x, "timestamp"):
        return float(x.timestamp())
    try:
        return float(x)
    except Exception:
        return np.nan

def _norm_times(ts, t0=None, t1=None):
    """Normalize time array ts to [0,1] with optional explicit t0/t1."""
    ts = np.asarray(ts, float)
    if ts.size == 0:
        return ts
    if t0 is None or not np.isfinite(t0): t0 = np.nanmin(ts) if np.isfinite(ts).any() else 0.0
    if t1 is None or not np.isfinite(t1): t1 = np.nanmax(ts) if np.isfinite(ts).any() else 1.0
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.linspace(0, 1, ts.size)
    return (ts - t0) / (t1 - t0)

def _angle_deg_batch(u, v, eps=1e-12):
    """
    Batched angle(s) between vectors u and v in degrees.
    - Accepts 1D (len 2/3) or 2D arrays (N×2/3). If 3D, uses (x,z).
    - Returns shape (N,) (or (1,) if single).
    """
    U = np.asarray(u, float)
    V = np.asarray(v, float)
    if U.ndim == 1: U = U.reshape(1, -1)
    if V.ndim == 1: V = V.reshape(1, -1)
    if U.shape[1] >= 3: U = U[:, [0, 2]]
    if V.shape[1] >= 3: V = V[:, [0, 2]]
    if U.shape[1] != 2 or V.shape[1] != 2:
        return np.full((U.shape[0],), np.nan)
    nu = np.linalg.norm(U, axis=1)
    nv = np.linalg.norm(V, axis=1)
    dots = np.einsum("ij,ij->i", U, V)
    cos = np.full_like(dots, np.nan, dtype=float)
    valid = np.isfinite(nu) & np.isfinite(nv) & (nu > eps) & (nv > eps) & np.isfinite(dots)
    cos[valid] = dots[valid] / (nu[valid] * nv[valid])
    np.clip(cos, -1.0, 1.0, out=cos, where=np.isfinite(cos))
    return np.degrees(np.arccos(cos))

# ===================== metrics (single definitions) =====================
def correction_metrics_for_trial(trial, vmin=0.05):
    """
    Return (ts, angle_deg, g_parallel, perp_ratio) per-step for a trial (speed-gated).
    - angle_deg: angle between BCI (Input) and AI-adjusted (Output)
    - g_parallel: (Out·In) / ||In||^2     (braking/amplification along user's intent)
    - perp_ratio: ||Out - proj_in(Out)|| / ||In|| (steering orthogonal to intent)
    """
    aiV = trial.get("aiVelocities") if isinstance(trial, dict) else getattr(trial, "aiVelocities", None)
    if not aiV:
        return np.array([]), np.array([]), np.array([]), np.array([])

    ts  = np.array([_as_float_timestamp(s.get("InputTimestamp")) for s in aiV], float)
    In  = np.array([_to_xz(s.get("Input"))  for s in aiV], float)
    Out = np.array([_to_xz(s.get("Output")) for s in aiV], float)

    in_norm  = np.linalg.norm(In,  axis=1)
    out_norm = np.linalg.norm(Out, axis=1)
    mask = np.isfinite(ts) & np.isfinite(in_norm) & np.isfinite(out_norm) & (in_norm >= vmin) & (out_norm >= vmin)
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([]), np.array([])

    ts  = ts[mask]
    In  = In[mask]
    Out = Out[mask]
    in_norm = np.linalg.norm(In, axis=1)  # recompute after mask

    ang = _angle_deg_batch(In, Out)

    dots = np.einsum("ij,ij->i", In, Out)
    g_parallel = dots / (in_norm**2 + 1e-12)

    proj = (g_parallel[:, None] * In)
    perp = Out - proj
    perp_ratio = np.linalg.norm(perp, axis=1) / (in_norm + 1e-12)

    return ts, ang, g_parallel, perp_ratio

def correction_metrics_for_trial_range(trial, vmin=0.05, frac_range=None):
    """
    Wrapper that applies a normalized-time slice frac_range=(f0,f1) on ts.
    Returns (angle_deg, g_parallel, perp_ratio) arrays after slicing.
    """
    ts, ang, gpar, perp = correction_metrics_for_trial(trial, vmin=vmin)
    if ts.size and frac_range is not None:
        f0, f1 = frac_range
        tnorm = _norm_times(ts)  # normalize on available velocity timestamps
        keep = (tnorm >= f0) & (tnorm <= f1)
        ang, gpar, perp = ang[keep], gpar[keep], perp[keep]
    return ang, gpar, perp

def metrics_summary(trials, vmin=0.05, frac_range=None):
    """Per-trial means and pooled means for angle_deg, g_parallel, perp_ratio."""
    mean_angle, mean_gpar, mean_perp = [], [], []
    pool_angle, pool_gpar, pool_perp = [], [], []
    for tr in trials:
        a, g, p = correction_metrics_for_trial_range(tr, vmin=vmin, frac_range=frac_range)
        if a.size:
            mean_angle.append(float(np.nanmean(a))); pool_angle.append(a)
            mean_gpar.append(float(np.nanmean(g)));  pool_gpar.append(g)
            mean_perp.append(float(np.nanmean(p)));  pool_perp.append(p)
    pool_angle = np.concatenate(pool_angle) if pool_angle else np.array([])
    pool_gpar  = np.concatenate(pool_gpar)  if pool_gpar  else np.array([])
    pool_perp  = np.concatenate(pool_perp)  if pool_perp  else np.array([])
    def _m(arr): return float(np.nanmean(arr)) if arr.size else np.nan
    return {
        "per_trial": {
            "angle_deg": np.array(mean_angle, float),
            "g_parallel": np.array(mean_gpar, float),
            "perp_ratio": np.array(mean_perp, float),
        },
        "pooled_mean": {
            "angle_deg": _m(pool_angle),
            "g_parallel": _m(pool_gpar),
            "perp_ratio": _m(pool_perp),
        },
        "counts": {"n_trials": len(mean_angle), "n_steps": int(pool_angle.size)}
    }

# ===================== trajectory + events (single definitions) =====================
def _get_traj(trial):
    traj = trial.get("avatarTrajectory") if isinstance(trial, dict) else getattr(trial, "avatarTrajectory", None)
    if isinstance(traj, dict) and "x" in traj and "z" in traj:
        x = np.asarray(traj["x"], float); z = np.asarray(traj["z"], float)
        if x.size and z.size: return x, z
    return None, None

def _obstacle_center(trial):
    o = trial.get("obstaclePosition") if isinstance(trial, dict) else getattr(trial, "obstaclePosition", None)
    if o is None: return None, None
    a = np.asarray(o, float).ravel()
    return (a[0], a[2]) if a.size >= 3 else (None, None)

def _target_center(trial):
    t = trial.get("targetPosition") if isinstance(trial, dict) else getattr(trial, "targetPosition", None)
    if t is None: return None, None
    a = np.asarray(t, float).ravel()
    return (a[0], a[2]) if a.size >= 3 else (None, None)

def event_idx_obstacle_apex(trial):
    x, z = _get_traj(trial); ox, oz = _obstacle_center(trial)
    if x is None or ox is None: return None
    d2 = (x - ox)**2 + (z - oz)**2
    return int(np.nanargmin(d2))

def event_idx_target_entry(trial, window_size=4.2):
    x, z = _get_traj(trial); tx, tz = _target_center(trial)
    if x is None or tx is None: return None
    h = window_size/2.0
    inside = (x >= tx - h) & (x <= tx + h) & (z >= tz - h) & (z <= tz + h)
    return int(np.argmax(inside)) if inside.any() else (len(x) - 1)

# ===================== event-aligned profiles (single definition) =====================
def event_aligned_profiles(trials, label, event="obstacle", window=(-0.4, 0.4), n_bins=100,
                           vmin=0.05, target_window_size=4.2):
    """
    Mean±SEM profiles around an event:
      - event="obstacle": closest approach to obstacle center
      - event="target": first entry into target window
    Time axis uses velocity timestamps normalized to [0,1], centered by the
    normalized trajectory-index of the event.
    """
    bins_ang  = [[] for _ in range(n_bins)]
    bins_gpar = [[] for _ in range(n_bins)]
    bins_perp = [[] for _ in range(n_bins)]
    edges = np.linspace(window[0], window[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for tr in trials:
        ts, ang, gpar, perp = correction_metrics_for_trial(tr, vmin=vmin)
        if ts.size == 0:
            continue

        if event == "obstacle":
            idx = event_idx_obstacle_apex(tr)
        else:
            idx = event_idx_target_entry(tr, window_size=target_window_size)
        if idx is None:
            continue

        x, z = _get_traj(tr)
        ev_norm = idx / max(1, (len(x) - 1)) if x is not None else 0.5

        ts_norm = _norm_times(ts)
        t_rel = ts_norm - ev_norm
        keep = (t_rel >= window[0]) & (t_rel <= window[1])
        if not np.any(keep):
            continue

        idx_bins = np.digitize(t_rel[keep], edges) - 1
        for a, g, p, b in zip(ang[keep], gpar[keep], perp[keep], idx_bins):
            if 0 <= b < n_bins:
                bins_ang[b].append(a)
                bins_gpar[b].append(g)
                bins_perp[b].append(p)

    def _mean_sem(bins):
        m, s = [], []
        for lst in bins:
            if not lst:
                m.append(np.nan); s.append(np.nan)
            else:
                arr = np.asarray(lst, float)
                m.append(np.nanmean(arr))
                s.append(np.nanstd(arr, ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
        return np.array(m), np.array(s)

    mA, sA = _mean_sem(bins_ang)
    mG, sG = _mean_sem(bins_gpar)
    mP, sP = _mean_sem(bins_perp)

    plt.figure(figsize=(11, 4))
    plt.plot(centers, mA, label=f"Angle ({label})")
    plt.fill_between(centers, mA - sA, mA + sA, alpha=0.2)
    plt.axvline(0, linestyle="--"); plt.grid(True)
    plt.xlabel("Time (normalized, centered on event)"); plt.ylabel("Angle (deg)")
    ttl = "Obstacle apex" if event == "obstacle" else "Target entry"
    plt.title(f"BCI–AI Angle around {ttl} — {label} (speed-gated)")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11, 4))
    plt.plot(centers, mG, label=f"Parallel gain ({label})")
    plt.fill_between(centers, mG - sG, mG + sG, alpha=0.2)
    plt.axvline(0, linestyle="--"); plt.grid(True)
    plt.xlabel("Time (normalized, centered on event)"); plt.ylabel("g_parallel  (Out·In / ||In||²)")
    plt.title(f"Parallel (along-intent) correction around {ttl} — {label}")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11, 4))
    plt.plot(centers, mP, label=f"Perp ratio ({label})")
    plt.fill_between(centers, mP - sP, mP + sP, alpha=0.2)
    plt.axvline(0, linestyle="--"); plt.grid(True)
    plt.xlabel("Time (normalized, centered on event)"); plt.ylabel("Perp steering / ||In||")
    plt.title(f"Perpendicular steering around {ttl} — {label}")
    plt.tight_layout(); plt.show()

    return {"time": centers, "angle": (mA, sA), "g_parallel": (mG, sG), "perp_ratio": (mP, sP),
            "params": {"event": event, "window": window, "n_bins": n_bins, "vmin": vmin}}

def analyze_shared_control_strength(ai_trials):
    """Analyze relationship between entropy gap and AI-BCI angle deviation excluding 100% takeover."""
    gap_values = []
    deviation_values = []

    for trial in ai_trials:
        if not trial["aiVelocities"]:
            continue

        # Extract entropy values
        entropy_lb = np.array([v["EntropyLb"] for v in trial["aiVelocities"]])
        entropy_ub = np.array([v["EntropyUb"] for v in trial["aiVelocities"]])
        gap = entropy_ub - entropy_lb

        # Compute deviations
        deviations = compute_velocity_angle_deviation(trial["aiVelocities"])

        # Identify takeover frames
        on = trial.get("aiControlOn")
        off = trial.get("aiControlOff")
        timestamps = np.array([v["InputTimestamp"][0] if isinstance(v["InputTimestamp"][0], float)
                                else v["InputTimestamp"][0].timestamp()
                                for v in trial["aiVelocities"]])
        takeover_mask = (~np.isnan(on) & ~np.isnan(off) &
                         (timestamps >= on) & (timestamps <= off))

        # Keep only non-takeover frames
        for g, d, tmask in zip(gap, deviations, takeover_mask):
            if not tmask and not np.isnan(g) and not np.isnan(d):
                gap_values.append(g)
                deviation_values.append(d)

    gap_values = np.array(gap_values)
    deviation_values = np.array(deviation_values)

    # === Bin gaps ===
    bins = np.linspace(np.min(gap_values), np.max(gap_values), 6)  # 5 bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_dev_per_bin = []
    std_dev_per_bin = []

    for i in range(len(bins) - 1):
        mask = (gap_values >= bins[i]) & (gap_values < bins[i + 1])
        mean_dev_per_bin.append(np.mean(deviation_values[mask]) if np.any(mask) else np.nan)
        std_dev_per_bin.append(np.std(deviation_values[mask]) if np.any(mask) else np.nan)

    # === Plot ===
    plt.figure(figsize=(8, 5))
    plt.errorbar(bin_centers, mean_dev_per_bin, yerr=std_dev_per_bin,
                 fmt='-o', capsize=4)
    plt.xlabel("Entropy Gap (Ub - Lb)")
    plt.ylabel("Mean Angle Deviation (degrees)")
    plt.title("Shared Control Strength vs. Entropy Gap (No Takeover Frames)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n=== Shared Control Scaling with Confidence (No Takeover) ===")
    for bc, m, s in zip(bin_centers, mean_dev_per_bin, std_dev_per_bin):
        print(f"Gap ~ {bc:.2f}: {m:.2f}° ± {s:.2f}°")

    return gap_values, deviation_values

def disagreement_timecourse(monkey, experiment, base_dir, n_bins=100, show=True, strong_thresh_deg=15):
    """
    Plot and summarize BCI–AI velocity angle deviation across normalized time,
    split by AI 100% takeover vs no-takeover.

    Args
    ----
    monkey, experiment, base_dir : str
        Where to find .../<monkey>/<experiment>/aiLog/*.pkl
    n_bins : int
        Number of bins for normalized-time profiles (0..1).
    show : bool
        Show the aggregated plot.
    strong_thresh_deg : float
        Threshold to report % of 'strong' corrections (e.g., >15°).
    """
    data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))

    # per-bin buckets
    bins_takeover = [[] for _ in range(n_bins)]
    bins_nontakeover = [[] for _ in range(n_bins)]

    # flat lists for overall stats
    all_angles_takeover = []
    all_angles_nontakeover = []

    def _norm_time(ts):
        # normalize timestamps to [0,1]
        ts = np.asarray(ts, dtype=float)
        if ts.size < 2 or not np.isfinite(ts[0]) or not np.isfinite(ts[-1]) or ts[-1] == ts[0]:
            # fallback if timestamps are weird
            return np.linspace(0, 1, ts.size)
        return (ts - ts[0]) / (ts[-1] - ts[0])

    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            trials = pickle.load(f)[1]
        # keep trials with aiVelocities and valid outcome labels (1,3,5,6 like your other code)
        ai_trials = [t for t in trials if t.get("aiVelocities")]
        ai_trials = [t for t in ai_trials if t.get("answer") in {1,3,5,6}]

        for tr in ai_trials:
            aiV = tr["aiVelocities"]
            # timestamps
            ts = []
            for v in aiV:
                t = v["InputTimestamp"][0]
                ts.append(t if isinstance(t, float) else t.timestamp())
            ts = np.asarray(ts, dtype=float)
            if ts.size == 0:
                continue
            tnorm = _norm_time(ts)

            # takeover window
            on = tr.get("aiControlOn")
            off = tr.get("aiControlOff")
            has_takeover = np.isfinite(on) and np.isfinite(off)
            if not has_takeover:
                # mask will be all False
                takeover_mask = np.zeros_like(ts, dtype=bool)
            else:
                takeover_mask = (ts >= on) & (ts <= off)

            # angles per time-point (vx,vz)
            angles = compute_velocity_angle_deviation(aiV)
            angles = np.asarray(angles, dtype=float)

            # drop NaNs
            valid = np.isfinite(angles) & np.isfinite(tnorm)
            if not np.any(valid):
                continue
            angles = angles[valid]
            tnorm = tnorm[valid]
            takeover_mask = takeover_mask[valid]

            # bin them
            idx = np.clip((tnorm * (n_bins - 1)).astype(int), 0, n_bins - 1)
            for a, i, is_to in zip(angles, idx, takeover_mask):
                if is_to:
                    bins_takeover[i].append(a)
                    all_angles_takeover.append(a)
                else:
                    bins_nontakeover[i].append(a)
                    all_angles_nontakeover.append(a)

    # convert bins to mean±SEM (use std/sqrt(n), guard empty bins)
    def _mean_sem(bin_lists):
        means, sems = [], []
        for lst in bin_lists:
            if len(lst) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                arr = np.asarray(lst, float)
                means.append(np.nanmean(arr))
                # ddof=1 if n>1 else 0 avoids warnings
                if len(arr) > 1:
                    sems.append(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)))
                else:
                    sems.append(0.0)
        return np.array(means), np.array(sems)

    m_to, s_to = _mean_sem(bins_takeover)
    m_no, s_no = _mean_sem(bins_nontakeover)
    x = np.linspace(0, 1, n_bins)

    # Plot
    if show:
        plt.figure(figsize=(12, 6))
        # takeover line + band
        plt.plot(x, m_to, label="Angle dev. (AI takeover)")
        plt.fill_between(x, m_to - s_to, m_to + s_to, alpha=0.2)
        # no-takeover line + band
        plt.plot(x, m_no, label="Angle dev. (No takeover)")
        plt.fill_between(x, m_no - s_no, m_no + s_no, alpha=0.2)
        plt.xlabel("Normalized Time")
        plt.ylabel("Angle Deviation (degrees)")
        plt.title("BCI–AI Disagreement Over Time (takeover vs no-takeover)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Summary stats
    def _safe_mean(arr):
        return float(np.nan) if len(arr) == 0 else float(np.mean(arr))
    def _percent_over(arr, thr):
        if len(arr) == 0:
            return np.nan
        arr = np.asarray(arr, float)
        return 100.0 * np.sum(arr > thr) / len(arr)

    mean_to = _safe_mean(all_angles_takeover)
    mean_no = _safe_mean(all_angles_nontakeover)
    pct_strong_to = _percent_over(all_angles_takeover, strong_thresh_deg)
    pct_strong_no = _percent_over(all_angles_nontakeover, strong_thresh_deg)

    print("\n=== BCI–AI Disagreement over Time (aggregated) ===")
    print(f"→ Mean angle deviation WITH AI takeover:    {mean_to:.2f}°")
    print(f"→ Mean angle deviation WITHOUT AI takeover: {mean_no:.2f}°")
    print(f"→ % of steps > {strong_thresh_deg:.0f}° WITH takeover:    {pct_strong_to:.1f}%")
    print(f"→ % of steps > {strong_thresh_deg:.0f}° WITHOUT takeover: {pct_strong_no:.1f}%")

    return {
        "time": x,
        "mean_takeover": m_to, "sem_takeover": s_to,
        "mean_nontakeover": m_no, "sem_nontakeover": s_no,
        "overall_mean_takeover": mean_to,
        "overall_mean_nontakeover": mean_no,
        "pct_strong_takeover": pct_strong_to,
        "pct_strong_nontakeover": pct_strong_no,
    }

def analyze_entropy_angle_relationship(ai_trials):
    """
    Analyze how entropy gap (Ub-Lb) relates to the angular deviation between
    BCI-decoded and AI-modified velocities. Runs per-timestep correlation,
    and compares angles for AI takeover vs non-takeover trials.
    """
    all_entropy_gaps = []
    all_angles = []
    takeover_flags = []
    success_flags = []

    for trial in ai_trials:
        # Extract entropy bounds
        entropy_lb = []
        entropy_ub = []
        for v in trial["aiVelocities"]:
            lb = v["EntropyLb"]
            ub = v["EntropyUb"]
            entropy_lb.append(lb if not np.isnan(lb) else np.nan)
            entropy_ub.append(ub if not np.isnan(ub) else np.nan)
        entropy_lb = np.array(entropy_lb)
        entropy_ub = np.array(entropy_ub)
        entropy_gap = entropy_ub - entropy_lb

        # Extract velocity deviation angles (vx, vz only)
        angles = compute_velocity_angle_deviation(trial["aiVelocities"])
        angles = np.array(angles)

        # Store per time point (ignoring NaNs in either)
        valid_mask = ~np.isnan(entropy_gap) & ~np.isnan(angles)
        all_entropy_gaps.extend(entropy_gap[valid_mask])
        all_angles.extend(angles[valid_mask])

        # Store trial-level flags
        takeover_flags.append(not np.isnan(trial.get("aiControlOn")) and not np.isnan(trial.get("aiControlOff")))
        success_flags.append(trial.get("answer") == 1)

    all_entropy_gaps = np.array(all_entropy_gaps)
    all_angles = np.array(all_angles)
    takeover_flags = np.array(takeover_flags)
    success_flags = np.array(success_flags)

    # === Correlation analysis ===
    if len(all_entropy_gaps) > 2:
        r, p = pearsonr(all_entropy_gaps, all_angles)
        print(f"→ Correlation between entropy gap and angle deviation: r = {r:.3f}, p = {p:.4g}")
    else:
        print("Not enough data for correlation.")

    # === Compare angles with vs without AI takeover ===
    takeover_angles = []
    non_takeover_angles = []
    for trial, takeover in zip(ai_trials, takeover_flags):
        trial_angles = compute_velocity_angle_deviation(trial["aiVelocities"])
        trial_angles = [a for a in trial_angles if not np.isnan(a)]
        if not trial_angles:
            continue
        if takeover:
            takeover_angles.extend(trial_angles)
        else:
            non_takeover_angles.extend(trial_angles)

    if takeover_angles and non_takeover_angles:
        print(f"→ Mean angle deviation WITH AI takeover: {np.mean(takeover_angles):.2f}°")
        print(f"→ Mean angle deviation WITHOUT AI takeover: {np.mean(non_takeover_angles):.2f}°")

    # === Plot scatter of gap vs angle ===
    plt.figure(figsize=(6, 6))
    plt.scatter(all_entropy_gaps, all_angles, alpha=0.3, s=10)
    plt.xlabel("Entropy Gap (Ub - Lb)")
    plt.ylabel("Angle Deviation (degrees)")
    plt.title("BCI–AI Disagreement vs. Shared-Control Clarity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def entropy_analysis(monkey, experiment, base_dir):
    data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))

    all_ub_curves = []
    all_lb_curves = []
    all_gaps = []
    success_flags = []
    ai_takeover_flags = []

    for file_path in pkl_files:
        print(f"\nLoading {os.path.basename(file_path)}")

        with open(file_path, "rb") as f:
            trials = pickle.load(f)[1]
            ai_trials = [trial for trial in trials if trial["aiVelocities"] != []]
            ai_trials = [trial for trial in ai_trials if trial["answer"] in {1, 3, 5, 6}]
            
            analyze_shared_control_strength(ai_trials)
            analyze_entropy_angle_relationship(ai_trials)

            for trial in ai_trials:
                timestamps = []
                entropy_lb = []
                entropy_ub = []

                for v in trial["aiVelocities"]:
                    t = v["InputTimestamp"][0]
                    if not isinstance(t, float):
                        t = t.timestamp()
                    timestamps.append(t)
                    lb = v["EntropyLb"]
                    ub = v["EntropyUb"]
                    entropy_lb.append(lb if not np.isnan(lb) else np.nan)
                    entropy_ub.append(ub if not np.isnan(ub) else np.nan)

                if len(entropy_lb) < 2:
                    continue

                entropy_lb = np.array(entropy_lb)
                entropy_ub = np.array(entropy_ub)
                gap = entropy_ub - entropy_lb

                # Interpolate to fixed length
                x_old = np.linspace(0, 1, len(entropy_lb))
                x_new = np.linspace(0, 1, 100)
                entropy_lb_interp = np.interp(x_new, x_old, entropy_lb)
                entropy_ub_interp = np.interp(x_new, x_old, entropy_ub)
                gap_interp = np.interp(x_new, x_old, gap)

                all_lb_curves.append(entropy_lb_interp)
                all_ub_curves.append(entropy_ub_interp)
                all_gaps.append(gap_interp)

                success_flags.append(trial.get("answer") == 1)
                ai_takeover_flags.append(
                    not np.isnan(trial.get("aiControlOn")) and not np.isnan(trial.get("aiControlOff"))
                )

                # Per-trial plot with AI shading
                iterations = np.arange(len(entropy_lb))
                timestamps = np.array(timestamps)
                plt.figure(figsize=(10, 5))
                plt.plot(iterations, entropy_lb, label="Entropy Lower Bound (Lb)")
                plt.plot(iterations, entropy_ub, label="Entropy Upper Bound (Ub)")

                on = trial.get("aiControlOn")
                off = trial.get("aiControlOff")
                if not np.isnan(on) and not np.isnan(off):
                    ai_mask = (timestamps >= on) & (timestamps <= off)
                    if np.any(ai_mask):
                        start_idx = np.argmax(ai_mask)
                        end_idx = len(ai_mask) - 1 - np.argmax(ai_mask[::-1])
                        plt.axvspan(start_idx, end_idx, color='red', alpha=0.2, label="AI takeover")

                plt.xlabel("Iteration (time step)")
                plt.ylabel("Entropy")
                plt.title(f"Entropy bounds for trial {trial['trial']}, answer: {trial['answer']}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                # plt.show()
                plt.close() 

    # === Group-level entropy dynamics ===
    all_lb_curves = np.vstack(all_lb_curves)
    all_ub_curves = np.vstack(all_ub_curves)
    all_gaps = np.vstack(all_gaps)
    success_flags = np.array(success_flags)
    ai_takeover_flags = np.array(ai_takeover_flags)
    time_axis = np.linspace(0, 1, all_ub_curves.shape[1])

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, np.nanmean(all_ub_curves, axis=0), label="Entropy Ub")
    plt.plot(time_axis, np.nanmean(all_lb_curves, axis=0), label="Entropy Lb")
    plt.plot(time_axis, np.nanmean(all_gaps, axis=0), linestyle="--", label="Ub - Lb (gap)")
    plt.fill_between(
        time_axis,
        np.nanmean(all_gaps, axis=0) - np.nanstd(all_gaps, axis=0),
        np.nanmean(all_gaps, axis=0) + np.nanstd(all_gaps, axis=0),
        alpha=0.3, color="gray", label="gap ±1 std"
    )
    plt.xlabel("Normalized Time")
    plt.ylabel("Entropy")
    plt.title("Average Shared Control Entropy Dynamics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close() 

    # === Summary metrics ===
    gap_per_trial = np.nanmean(all_gaps, axis=1)
    print("\n=== Summary Shared-Control Entropy Metrics ===")
    print(f"→ Mean gap (Ub - Lb) for successful trials: {np.nanmean(gap_per_trial[success_flags]):.3f}")
    print(f"→ Mean gap (Ub - Lb) for failed trials:     {np.nanmean(gap_per_trial[~success_flags]):.3f}")
    print(f"→ Mean gap (Ub - Lb) with AI takeover:       {np.nanmean(gap_per_trial[ai_takeover_flags]):.3f}")
    print(f"→ Mean gap (Ub - Lb) without AI takeover:    {np.nanmean(gap_per_trial[~ai_takeover_flags]):.3f}")
    print("Done")

def compute_velocity_angle_deviation(ai_velocities):
    """Returns list of angle deviations (in degrees) between BCI and AI velocity (vx, vz) per time point."""
    angles = []
    for v in ai_velocities:
        v_bci = np.array(v["Input"])[[0, 2]]   # keep only vx and vz
        v_ai = np.array(v["Output"])[[0, 2]]   # keep only vx and vz

        # Sanity check: avoid division by zero
        norm_bci = np.linalg.norm(v_bci)
        norm_ai = np.linalg.norm(v_ai)
        if norm_bci == 0 or norm_ai == 0:
            angles.append(np.nan)
            continue

        dot_product = np.dot(v_bci, v_ai)
        cosine_angle = np.clip(dot_product / (norm_bci * norm_ai), -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)
    return angles

def ai_intervention (monkey, experiment, base_dir):
    # === Load your .pkl file ===
    data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
    # === Get all .pkl files ===
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))

    ai_freq_all = []
    percent_time_all = []
    ai_frequency_per_target_all = []
    per_target_freq_accumulator = defaultdict(list)
    all_durations_steps = []
    all_durations_steps_std = []
    all_durations_ms = []
    all_durations_ms_std = []
    all_success_with_ai = []
    all_success_without_ai = []

    # NEW: pooled counters for risk-stratified analysis (across files)
    pooled_succ_risky = pooled_fail_risky = 0
    pooled_succ_low   = pooled_fail_low   = 0

    for file_path in pkl_files:
        print(f"\nLoading {os.path.basename(file_path)}")

        with open(file_path, "rb") as f:
            trials = pickle.load(f)[1]
            ai_trials = [trial for trial in trials if trial["aiVelocities"]!=[]]
            ai_trials = [trial for trial in ai_trials if trial["answer"] in {1, 3, 5, 6}]
            # === Compute metrics ===
            ai_trial_frequency = compute_ai_trial_frequency(ai_trials)
            percent_time_under_ai = compute_percent_time_under_ai(ai_trials)
            ai_frequency_per_target = compute_ai_frequency_per_target(ai_trials)
            durations_steps, mean_steps, std_steps, mean_ms, std_ms = compute_ai_intervention_durations(ai_trials)
            success_with_ai, success_without_ai = compute_success_rate_effect_of_ai(ai_trials)

            # NEW: risk-stratified success (risky = any full takeover; low-risk = no takeover)
            rs = compute_risk_stratified_success(ai_trials)
            pooled_succ_risky += rs["counts"]["succ_risky"]
            pooled_fail_risky += rs["counts"]["fail_risky"]
            pooled_succ_low   += rs["counts"]["succ_low"]
            pooled_fail_low   += rs["counts"]["fail_low"]

            ai_freq_all.append(ai_trial_frequency)
            percent_time_all.append(percent_time_under_ai)
            ai_frequency_per_target_all.append(ai_frequency_per_target)
            all_success_with_ai.append(success_with_ai)
            all_success_without_ai.append(success_without_ai)
            all_durations_steps.append(mean_steps)
            all_durations_steps_std.append(std_steps)
            all_durations_ms.append(mean_ms)
            all_durations_ms_std.append(std_ms)
            for target, freq in ai_frequency_per_target.items():
                per_target_freq_accumulator[tuple(target)].append(freq)  # target must be hashable (tuple)

            ### === Compute angle between BCI and AI velocity
            for trial in ai_trials:
                aiVelocities = trial["aiVelocities"]
                angles = compute_velocity_angle_deviation(aiVelocities)
    # === Compute averages ===
    mean_ai_freq = np.mean(ai_freq_all) if ai_freq_all else np.nan
    mean_time_under_ai = np.mean(percent_time_all) if percent_time_all else np.nan
    mean_freq_per_target = {
        target: np.mean(freqs)
        for target, freqs in per_target_freq_accumulator.items()
    }
    mean_success_with_ai = np.mean(all_success_with_ai) if all_success_with_ai else np.nan
    mean_success_without_ai = np.mean(all_success_without_ai) if all_success_without_ai else np.nan
    mean_duration_steps = np.mean(all_durations_steps) if all_durations_steps else np.nan
    std_duration_steps = np.mean(all_durations_steps_std) if all_durations_steps_std else np.nan
    mean_duration_ms = np.mean(all_durations_ms) if all_durations_ms else np.nan
    std_duration_ms = np.mean(all_durations_ms_std) if all_durations_ms_std else np.nan

    # === NEW: pooled risk-stratified results ===
    risky_total = pooled_succ_risky + pooled_fail_risky
    low_total   = pooled_succ_low   + pooled_fail_low
    pct_risky   = 100.0 * pooled_succ_risky / risky_total if risky_total > 0 else np.nan
    pct_low     = 100.0 * pooled_succ_low   / low_total   if low_total   > 0 else np.nan

    # Odds ratio (with correction)
    a = pooled_succ_risky + 0.5
    b = pooled_fail_risky + 0.5
    c = pooled_succ_low   + 0.5
    d = pooled_fail_low   + 0.5
    or_value = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    or_lo = np.exp(np.log(or_value) - 1.96 * se)
    or_hi = np.exp(np.log(or_value) + 1.96 * se)

    print("\n=== Averaged AI Intervention Metrics ===")
    print(f"→ Avg. % of trials with AI intervention: {mean_ai_freq:.2f}%")
    print(f"→ Avg. % of time steps under AI control: {mean_time_under_ai:.2f}%")
    print("\n=== Success Rates in Risky vs Low-Risk Trials ===")
    print(f"→ Success in risky trials (with takeover):    {pct_risky:.2f}%  "
          f"[{pooled_succ_risky}/{risky_total}]")
    print(f"→ Success in low-risk trials (no takeover):   {pct_low:.2f}%  "
          f"[{pooled_succ_low}/{low_total}]")
    print(f"→ Odds ratio (risky vs low-risk): {or_value:.2f}  [95% CI: {or_lo:.2f}, {or_hi:.2f}]")
    print("\n=== Success Rates in Risky Trials ===")
    print(f"→ With AI takeover:    {mean_success_with_ai:.2f}%")
    print(f"→ Without AI takeover: {mean_success_without_ai:.2f}%")
    print(f"→ Avg. AI intervention duration: {mean_duration_ms:.1f} ms "
      f"(± {std_duration_ms:.1f} ms), or {mean_duration_steps:.2f} steps (± {std_duration_steps:.2f})")
    print("\n→ Avg. AI intervention frequency per target:")
    for target, mean_freq in mean_freq_per_target.items():
        print(f"   Target {target}: {mean_freq:.2f}%")

# --- NEW: helper for risk-stratified success (risky = takeover; low-risk = no takeover)
def compute_risk_stratified_success(ai_trials):
    succ_risky = fail_risky = succ_low = fail_low = 0

    for tr in ai_trials:
        success = (tr.get("answer") == 1)
        on = tr.get("aiControlOn")
        off = tr.get("aiControlOff")
        has_takeover = (on is not None and off is not None and not np.isnan(on) and not np.isnan(off))

        if has_takeover:
            if success: succ_risky += 1
            else:       fail_risky += 1
        else:
            if success: succ_low += 1
            else:       fail_low += 1

    # percentages
    risky_total = succ_risky + fail_risky
    low_total   = succ_low   + fail_low
    pct_risky   = 100.0 * succ_risky / risky_total if risky_total > 0 else np.nan
    pct_low     = 100.0 * succ_low   / low_total   if low_total   > 0 else np.nan

    # Odds ratio with Haldane–Anscombe correction to avoid division by zero
    a = succ_risky + 0.5
    b = fail_risky + 0.5
    c = succ_low   + 0.5
    d = fail_low   + 0.5
    or_value = (a * d) / (b * c)

    # 95% CI for log(OR)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = np.exp(np.log(or_value) - 1.96 * se)
    hi = np.exp(np.log(or_value) + 1.96 * se)

    return {
        "pct_risky": pct_risky,
        "pct_low": pct_low,
        "counts": {
            "succ_risky": succ_risky, "fail_risky": fail_risky,
            "succ_low":   succ_low,   "fail_low":   fail_low
        },
        "or": or_value,
        "or_ci": (lo, hi)
    }

def compute_ai_trial_frequency(ai_trials):
    total_trials = len(ai_trials)
    ai_trials_with_intervention = sum(
        not np.isnan(trial["aiControlOn"]) and not np.isnan(trial["aiControlOff"])
        for trial in ai_trials
    )
    percent_with_ai = 100 * ai_trials_with_intervention / total_trials
    return percent_with_ai

def compute_ai_frequency_per_target(trials):
    per_target_counts = defaultdict(lambda: {"with_ai": 0, "total": 0})

    for trial in trials:
        target_pos = trial.get("targetPosition")
        if target_pos is None:
            continue

        target = tuple(target_pos)  # make hashable
        per_target_counts[target]["total"] += 1

        ai_on = trial.get("aiControlOn")
        ai_off = trial.get("aiControlOff")
        if ai_on is not None and not np.isnan(ai_on) and ai_off is not None and not np.isnan(ai_off):
            per_target_counts[target]["with_ai"] += 1

    freq_per_target = {
        target: 100 * counts["with_ai"] / counts["total"]
        for target, counts in per_target_counts.items()
        if counts["total"] > 0
    }

    return freq_per_target

def compute_percent_time_under_ai(ai_trials):
    total_steps = 0
    ai_steps = 0

    for trial in ai_trials:
        control_on = trial["aiControlOn"]
        control_off = trial["aiControlOff"]

        for v in trial["aiVelocities"]:
            t = v["InputTimestamp"][0]
            if not isinstance(t, float):
                t = t.timestamp()

            total_steps += 1
            if not np.isnan(control_on) and not np.isnan(control_off):
                if control_on <= t <= control_off:
                    ai_steps += 1

    percent_ai = 100 * ai_steps / total_steps
    return percent_ai

def compute_ai_intervention_durations(ai_trials):
    durations_steps = []
    durations_ms = []

    for trial in ai_trials:
        on = trial.get("aiControlOn")
        off = trial.get("aiControlOff")

        if np.isnan(on) or np.isnan(off):
            continue  # Skip trials without full AI control

        # Extract timestamps from aiVelocities
        timestamps = [v["InputTimestamp"][0] for v in trial["aiVelocities"]]
        timestamps = [t.timestamp() if not isinstance(t, float) else t for t in timestamps]

        # Count how many timestamps fall in the control window
        step_count = sum(on <= t <= off for t in timestamps)
        durations_steps.append(step_count)

        # Use exact timestamps to compute duration in ms
        duration_ms = (off - on)
        durations_ms.append(duration_ms)

    if not durations_steps:
        return [], np.nan, np.nan, np.nan, np.nan

    return (
        np.array(durations_steps),
        np.mean(durations_steps),
        np.std(durations_steps),
        np.mean(durations_ms),
        np.std(durations_ms),
    )

def compute_success_rate_effect_of_ai(trials):
    with_ai = [trial for trial in trials if not np.isnan(trial["aiControlOn"])]
    without_ai = [trial for trial in trials if np.isnan(trial["aiControlOn"])]

    correct_with_ai = sum(trial["answer"] == 1 for trial in with_ai)
    correct_without_ai = sum(trial["answer"] == 1 for trial in without_ai)

    success_with_ai = 100 * correct_with_ai / len(with_ai) if with_ai else np.nan
    success_without_ai = 100 * correct_without_ai / len(without_ai) if without_ai else np.nan

    return success_with_ai, success_without_ai