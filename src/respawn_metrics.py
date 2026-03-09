# import numpy as np
# import numpy as np
# import pandas as pd
# from typing import Dict, Any, List, Optional

# DEG = np.pi / 180.0

# # ---------- Utilities ----------
# def unit(v, eps=1e-9):
#     v = np.asarray(v, dtype=float)
#     n = np.linalg.norm(v, axis=-1, keepdims=True)
#     return v / np.clip(n, eps, None)

# def angle_between(u, v):
#     u = unit(u); v = unit(v)
#     cos = np.clip(np.sum(u*v, axis=-1), -1.0, 1.0)
#     return np.arccos(cos)  # radians

# def sustain_latency(times_ms, angles_rad, thresh_deg=30.0, sustain_ms=150.0) -> float:
#     """Return latency (ms) from t0 to the first sustained period where angle<thresh_deg for >= sustain_ms."""
#     if len(times_ms) == 0:
#         return np.nan
#     ok = (angles_rad < (thresh_deg * DEG)).astype(int)
#     dt = np.diff(times_ms, prepend=times_ms[0])
#     run = 0.0
#     for i in range(len(ok)):
#         run = (run + dt[i]) if ok[i] == 1 else 0.0
#         if run >= sustain_ms:
#             # backtrack to the start of the sustained window
#             run2 = 0.0
#             j = i
#             while j >= 0 and ok[j] == 1 and run2 < sustain_ms:
#                 run2 += dt[j]
#                 j -= 1
#             return float(times_ms[j+1] - times_ms[0])
#     return np.nan

# def _interp_fill_nans(x: np.ndarray) -> np.ndarray:
#     """Linear interpolate 1D array in-place where possible; constant fill if all-nan."""
#     out = x.copy()
#     idx = np.where(~np.isnan(out))[0]
#     if idx.size == 0:
#         out[:] = 0.0
#         return out
#     nans = np.isnan(out)
#     out[nans] = np.interp(np.where(nans)[0], idx, out[idx])
#     return out

# def _stack_time_series(rows: List[Dict[str, Any]]):
#     """Build arrays T, v_bci, v_ai, pos from aiVelocities rows."""
#     T, v_bci, v_ai, pos = [], [], [], []
#     for r in rows:
#         # Prefer OutputTimestamp (often aligns to controller), fall back to InputTimestamp
#         tfield = 'OutputTimestamp' if r.get('OutputTimestamp') is not None else 'InputTimestamp'
#         t_arr = r.get(tfield)
#         if t_arr is None:
#             continue
#         t = np.asarray(t_arr).ravel()
#         if t.size == 0:
#             continue
#         T.append(t[0])
#         v_bci.append(np.asarray(r['Input'], dtype=float))
#         v_ai.append(np.asarray(r['Output'], dtype=float))
#         ap = r.get('AvatarPosition')
#         if isinstance(ap, (list, tuple, np.ndarray)):
#             pos.append(np.asarray(ap, dtype=float))
#         else:
#             pos.append(np.array([np.nan, np.nan, np.nan], dtype=float))
#     if len(T) == 0:
#         return np.array([]), np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,3))
#     T = np.asarray(T, dtype=float)
#     v_bci = np.vstack(v_bci)
#     v_ai  = np.vstack(v_ai)
#     pos   = np.vstack(pos)
#     # Interpolate missing positions per dimension if needed
#     for k in range(3):
#         pos[:, k] = _interp_fill_nans(pos[:, k])
#     return T, v_bci, v_ai, pos

# # ---------- Core per-trial computation ----------
# def compute_respawn_metrics_for_trial(
#     trial: Dict[str, Any],
#     session_idx: int,
#     respawn_time_key: str = 'targetJumpTime',
#     old_target_key: str = 'targetPosition',
#     new_target_key: str = 'targetJumpPosition',
# ) -> Dict[str, Any]:
#     """Compute metrics for one trial. Returns a row dict (safe with missing fields)."""
#     row_out = {
#         'session': session_idx,
#         'trial': trial.get('trial', np.nan),
#         'answer': trial.get('answer', np.nan),
#         'start': trial.get('start', np.nan),
#         'stop': trial.get('stop', np.nan),
#         # Metrics (default NaN)
#         'T_bci_ms': np.nan,
#         'T_ai_ms': np.nan,
#         'lag_ms': np.nan,
#         'vpar_bci_AUC_0_500': np.nan,
#         'vpar_ai_AUC_0_500': np.nan,
#         'peak_MI_0_500': np.nan,
#         't_decay_MI_ms': np.nan,
#         'path_dev_bci_0_1500': np.nan,
#         'path_dev_ai_0_1500': np.nan,
#     }

#     rows = trial.get('aiVelocities', None)
#     if not rows:
#         return row_out  # nothing to compute

#     T_all, v_bci_all, v_ai_all, pos_all = _stack_time_series(rows)
#     if T_all.size == 0:
#         return row_out

#     # Extract respawn info (if present)
#     t0 = trial.get(respawn_time_key, None)
#     old_tgt = trial.get(old_target_key, None)
#     new_tgt = trial.get(new_target_key, None)

#     # If respawn info missing, we can still compute some metrics relative to the (single) targetPosition if present
#     # but the respawn-specific ones (latency vs new target, MI vs old/new) will remain NaN.
#     if t0 is None or new_tgt is None or old_tgt is None:
#         # Try to fall back to 'targetPosition' as the "current" target for directional components
#         tgt = trial.get('targetPosition', None)
#         if tgt is None:
#             return row_out  # cannot do meaningful direction metrics
#         # Build a minimal set: v_parallel AUC in first 500ms after start of trial logging
#         # Align to the first timestamp as t0 proxy (not "respawn", just to compute AUC)
#         T = T_all - T_all[0]
#         g = unit(np.asarray(tgt, dtype=float) - pos_all)
#         vpar_bci = np.sum(v_bci_all * g, axis=1)
#         vpar_ai  = np.sum(v_ai_all  * g, axis=1)
#         early = T <= 500
#         if np.any(early):
#             row_out['vpar_bci_AUC_0_500'] = float(np.trapz(vpar_bci[early], T[early]))
#             row_out['vpar_ai_AUC_0_500']  = float(np.trapz(vpar_ai[early],  T[early]))
#         # Path deviation proxy (0–1.5 s)
#         win = T <= 1500
#         lateral_bci = np.linalg.norm(v_bci_all - (vpar_bci[:,None]*g), axis=1)
#         lateral_ai  = np.linalg.norm(v_ai_all  - (vpar_ai[:,None]*g),  axis=1)
#         if np.any(win):
#             row_out['path_dev_bci_0_1500'] = float(np.trapz(lateral_bci[win], T[win]))
#             row_out['path_dev_ai_0_1500']  = float(np.trapz(lateral_ai[win],  T[win]))
#         return row_out

#     # Full respawn analysis
#     t0 = float(t0)
#     old_tgt = np.asarray(old_tgt, dtype=float)
#     new_tgt = np.asarray(new_tgt, dtype=float)

#     # Keep samples after respawn; realign time so t=0 at respawn
#     mask = T_all >= t0
#     if not np.any(mask):
#         return row_out
#     T = T_all[mask] - t0
#     v_bci = v_bci_all[mask]
#     v_ai  = v_ai_all[mask]
#     pos   = pos_all[mask]

#     # Direction fields to old/new targets from current position
#     g_old = unit(old_tgt - pos)
#     g_new = unit(new_tgt - pos)

#     # Angles to NEW target
#     ang_bci = angle_between(v_bci, g_new)   # radians
#     ang_ai  = angle_between(v_ai,  g_new)

#     # Reorientation latencies
#     row_out['T_bci_ms'] = sustain_latency(T, ang_bci, 30.0, 150.0)
#     row_out['T_ai_ms']  = sustain_latency(T, ang_ai,  30.0, 150.0)
#     if not np.isnan(row_out['T_bci_ms']) and not np.isnan(row_out['T_ai_ms']):
#         row_out['lag_ms'] = float(row_out['T_ai_ms'] - row_out['T_bci_ms'])

#     # Parallel velocity to NEW target (AUC in first 500 ms)
#     vpar_bci = np.sum(v_bci * g_new, axis=1)
#     vpar_ai  = np.sum(v_ai  * g_new, axis=1)
#     early = T <= 500.0
#     if np.any(early):
#         row_out['vpar_bci_AUC_0_500'] = float(np.trapz(vpar_bci[early], T[early]))
#         row_out['vpar_ai_AUC_0_500']  = float(np.trapz(vpar_ai[early],  T[early]))

#     # Mismatch index (old vs new) from AI velocity
#     cos_old = np.sum(unit(v_ai) * g_old, axis=1)
#     cos_new = np.sum(unit(v_ai) * g_new, axis=1)
#     MI = cos_old - cos_new
#     early_idx = np.where(early)[0]
#     if early_idx.size > 0:
#         row_out['peak_MI_0_500'] = float(np.nanmax(MI[early_idx]))
#     # Time to MI <= 0
#     t_decay = np.nan
#     for tt, mi in zip(T, MI):
#         if tt >= 0 and mi <= 0:
#             t_decay = float(tt); break
#     row_out['t_decay_MI_ms'] = t_decay

#     # Path deviation proxy (integrated lateral speed) over 0–1.5 s
#     win = T <= 1500.0
#     if np.any(win):
#         lateral_bci = np.linalg.norm(v_bci - (vpar_bci[:,None]*g_new), axis=1)
#         lateral_ai  = np.linalg.norm(v_ai  - (vpar_ai[:,None]*g_new),  axis=1)
#         row_out['path_dev_bci_0_1500'] = float(np.trapz(lateral_bci[win], T[win]))
#         row_out['path_dev_ai_0_1500']  = float(np.trapz(lateral_ai[win],  T[win]))

#     return row_out

# # =========================
# # Configurable constants
# # =========================
# THRESH_DEG_DEFAULT = 15.0   # half-cone angle for alignment
# SUSTAIN_MS_DEFAULT = 200.0  # duration alignment must be sustained
# BINS_MS = [(0, 500), (500, 1000), (1000, 1500)]

# DEG = np.pi / 180.0

# # =========================
# # Helpers
# # =========================
# def unit(v, eps: float = 1e-9):
#     v = np.asarray(v, dtype=float)
#     n = np.linalg.norm(v, axis=-1, keepdims=True)
#     return v / np.clip(n, eps, None)

# def angle_between(u, v):
#     u = unit(u); v = unit(v)
#     cos = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
#     return np.arccos(cos)  # radians

# def sustain_latency(times_ms: np.ndarray,
#                     angles_rad: np.ndarray,
#                     thresh_deg: float = THRESH_DEG_DEFAULT,
#                     sustain_ms: float = SUSTAIN_MS_DEFAULT) -> float:
#     """
#     Return latency (ms) from times_ms[0] to first time alignment is sustained.
#     Alignment = angle < thresh_deg for >= sustain_ms consecutively.
#     """
#     if times_ms.size == 0 or angles_rad.size == 0:
#         return np.nan
#     ok = (angles_rad < (thresh_deg * DEG)).astype(int)
#     dt = np.diff(times_ms, prepend=times_ms[0])
#     run = 0.0
#     for i in range(len(ok)):
#         run = (run + dt[i]) if ok[i] == 1 else 0.0
#         if run >= sustain_ms:
#             # backtrack to the earliest time within this sustained window
#             run2 = 0.0
#             j = i
#             while j >= 0 and ok[j] == 1 and run2 < sustain_ms:
#                 run2 += dt[j]
#                 j -= 1
#             return float(times_ms[j + 1] - times_ms[0])
#     return np.nan

# def _empty_metrics_dict() -> Dict[str, float]:
#     return {
#         "T_bci_ms": np.nan, "T_ai_ms": np.nan, "lag_ms": np.nan,
#         "vpar_bci_AUC_0_500": np.nan, "vpar_ai_AUC_0_500": np.nan,
#         "peak_MI_0_500": np.nan, "t_decay_MI_ms": np.nan,
#         "path_dev_bci_0_1500": np.nan, "path_dev_ai_0_1500": np.nan,
#         "path_dev_bci_0_500": np.nan, "path_dev_bci_500_1000": np.nan, "path_dev_bci_1000_1500": np.nan,
#         "path_dev_ai_0_500": np.nan,  "path_dev_ai_500_1000": np.nan,  "path_dev_ai_1000_1500": np.nan,
#         "cos_sim_mean_0_500": np.nan, "cos_sim_mean_500_1000": np.nan, "cos_sim_mean_1000_1500": np.nan,
#     }

# def _trap_in_window(y: np.ndarray, t: np.ndarray, lo: float, hi: float) -> float:
#     m = (t >= lo) & (t < hi)
#     return float(np.trapz(y[m], t[m])) if np.any(m) else np.nan

# def _coerce_vec3(x) -> np.ndarray:
#     """
#     Accepts list/tuple/np.ndarray or dict with keys ('x','y','z') or ('Item1','Item2','Item3').
#     Returns np.ndarray shape (3,) with dtype float; NaNs if not coercible.
#     """
#     if x is None:
#         return np.array([np.nan, np.nan, np.nan], dtype=float)
#     if isinstance(x, (list, tuple, np.ndarray)):
#         arr = np.asarray(x, dtype=float).ravel()
#         if arr.size >= 3:
#             return arr[:3]
#         out = np.full(3, np.nan, dtype=float)
#         out[:arr.size] = arr
#         return out
#     if isinstance(x, dict):
#         for keys in (('x', 'y', 'z'), ('Item1', 'Item2', 'Item3')):
#             if all(k in x for k in keys):
#                 return np.array([x[keys[0]], x[keys[1]], x[keys[2]]], dtype=float)
#     return np.array([np.nan, np.nan, np.nan], dtype=float)

# # =========================
# # Per-trial metric computation
# # =========================
# def compute_respawn_metrics(trial: Dict[str, Any],
#                             respawn_time_ms: float,
#                             old_target, new_target,
#                             thresh_deg: float = THRESH_DEG_DEFAULT,
#                             sustain_ms: float = SUSTAIN_MS_DEFAULT) -> Dict[str, float]:
#     """
#     trial: dict with 'aiVelocities' list; each element has:
#       'Input' (3,), 'Output' (3,), 'InputTimestamp'/'OutputTimestamp' (TNS ms), 'AvatarPosition' (3,) or nan
#     respawn_time_ms: float (TNS ms)
#     old_target, new_target: 3D coords at (or immediately around) respawn (x,y,z or dict)
#     """
#     rows = getattr(trial,'aiVelocities', [])
#     T, v_bci, v_ai, pos = [], [], [], []

#     for r in rows:
#         t = r.get('OutputTimestamp', None)
#         if t is None:
#             t = r.get('InputTimestamp', None)
#         if t is None:
#             continue
#         T.append(float(np.asarray(t).ravel()[0]))
#         v_bci.append(_coerce_vec3(r.get('Input')))
#         v_ai.append(_coerce_vec3(r.get('Output')))
#         pos.append(_coerce_vec3(r.get('AvatarPosition')))

#     if len(T) == 0 or not np.isfinite(respawn_time_ms):
#         return _empty_metrics_dict()

#     T = np.asarray(T, dtype=float)
#     v_bci = np.vstack(v_bci).astype(float)
#     v_ai  = np.vstack(v_ai).astype(float)
#     pos   = np.vstack(pos).astype(float)

#     # Keep samples after respawn and re-zero time
#     mask = T >= respawn_time_ms
#     if not np.any(mask):
#         return _empty_metrics_dict()
#     T = T[mask] - respawn_time_ms
#     v_bci = v_bci[mask]; v_ai = v_ai[mask]; pos = pos[mask]

#     # Fill missing positions (per-axis interp; if none available, zeros)
#     for k in range(3):
#         col = pos[:, k]
#         if np.isnan(col).any():
#             idx = np.where(~np.isnan(col))[0]
#             if idx.size > 0:
#                 nan_idx = np.where(np.isnan(col))[0]
#                 col[nan_idx] = np.interp(nan_idx, idx, col[idx])
#             else:
#                 col[:] = 0.0
#         pos[:, k] = col

#     old_target = _coerce_vec3(old_target)
#     new_target = _coerce_vec3(new_target)

#     g_old = unit(old_target - pos)
#     g_new = unit(new_target - pos)

#     # Angles to new goal & parallel speed
#     ang_bci = angle_between(v_bci, g_new)
#     ang_ai  = angle_between(v_ai,  g_new)
#     vpar_bci = np.sum(v_bci * g_new, axis=1)
#     vpar_ai  = np.sum(v_ai  * g_new, axis=1)

#     # Reorientation latencies
#     T_bci = sustain_latency(T, ang_bci, thresh_deg=thresh_deg, sustain_ms=sustain_ms)
#     T_ai  = sustain_latency(T, ang_ai,  thresh_deg=thresh_deg, sustain_ms=sustain_ms)
#     lag_ms = (T_ai - T_bci) if (np.isfinite(T_bci) and np.isfinite(T_ai)) else np.nan

#     # Mismatch index (AI relative to old vs new goals)
#     cos_old = np.sum(unit(v_ai) * g_old, axis=1)
#     cos_new = np.sum(unit(v_ai) * g_new, axis=1)
#     MI = cos_old - cos_new
#     early = T <= 500
#     peak_MI = np.nanmax(MI[early]) if np.any(early) else np.nan
#     t_decay = np.nan
#     for tt, mi in zip(T, MI):
#         if tt >= 0 and mi <= 0:
#             t_decay = float(tt); break

#     # Lateral speed & path deviation (overall + binned)
#     lat_bci = np.linalg.norm(v_bci - (vpar_bci[:, None] * g_new), axis=1)
#     lat_ai  = np.linalg.norm(v_ai  - (vpar_ai[:,  None] * g_new), axis=1)

#     pathdev_bci_bins = [_trap_in_window(lat_bci, T, a, b) for (a, b) in BINS_MS]
#     pathdev_ai_bins  = [_trap_in_window(lat_ai,  T, a, b) for (a, b) in BINS_MS]

#     win1500 = T <= 1500
#     pathdev_bci_1500 = float(np.trapz(lat_bci[win1500], T[win1500])) if np.any(win1500) else np.nan
#     pathdev_ai_1500  = float(np.trapz(lat_ai[win1500],  T[win1500])) if np.any(win1500) else np.nan

#     # BCI–AI cosine similarity (listening), binned
#     cos_sim = np.sum(unit(v_bci) * unit(v_ai), axis=1)
#     cos_mean_bins = [
#         np.nanmean(cos_sim[(T >= a) & (T < b)]) if np.any((T >= a) & (T < b)) else np.nan
#         for (a, b) in BINS_MS
#     ]

#     # Early parallel drive (AUC 0–500 ms) for continuity with earlier table
#     vpar_bci_auc_0_500 = float(np.trapz(vpar_bci[early], T[early])) if np.any(early) else np.nan
#     vpar_ai_auc_0_500  = float(np.trapz(vpar_ai[early],  T[early])) if np.any(early) else np.nan

#     return {
#         "T_bci_ms": T_bci, "T_ai_ms": T_ai, "lag_ms": lag_ms,
#         "vpar_bci_AUC_0_500": vpar_bci_auc_0_500,
#         "vpar_ai_AUC_0_500":  vpar_ai_auc_0_500,
#         "peak_MI_0_500": peak_MI, "t_decay_MI_ms": t_decay,
#         "path_dev_bci_0_1500": pathdev_bci_1500, "path_dev_ai_0_1500": pathdev_ai_1500,
#         "path_dev_bci_0_500": pathdev_bci_bins[0],
#         "path_dev_bci_500_1000": pathdev_bci_bins[1],
#         "path_dev_bci_1000_1500": pathdev_bci_bins[2],
#         "path_dev_ai_0_500": pathdev_ai_bins[0],
#         "path_dev_ai_500_1000": pathdev_ai_bins[1],
#         "path_dev_ai_1000_1500": pathdev_ai_bins[2],
#         "cos_sim_mean_0_500":     cos_mean_bins[0],
#         "cos_sim_mean_500_1000":  cos_mean_bins[1],
#         "cos_sim_mean_1000_1500": cos_mean_bins[2],
#     }

# # =========================
# # Wrapper for a single trial dict
# # =========================
# def compute_respawn_metrics_for_trial(tr: Dict[str, Any],
#                                       session_idx: int,
#                                       thresh_deg: float = THRESH_DEG_DEFAULT,
#                                       sustain_ms: float = SUSTAIN_MS_DEFAULT) -> Dict[str, Any]:
#     """
#     Extracts inputs from `tr` and calls compute_respawn_metrics. Adds session/trial ids.
#     Expects keys: 'targetJumpTime' (TNS ms), 'targetPosition', 'targetJumpPosition', 'aiVelocities'
#     """
#     respawn_time_ms = getattr(tr, 'targetJumpTime', np.nan)  # should be TNS ms (float)

#     old_target = getattr(tr, 'targetPosition', [np.nan, np.nan, np.nan])

#     # Correct trial attribute name is targetJumpPosition (not the old dict typo)
#     new_target = getattr(tr, 'targetJumpPosition', [np.nan, np.nan, np.nan])

#     out = compute_respawn_metrics(tr, respawn_time_ms, old_target, new_target,
#                                   thresh_deg=thresh_deg, sustain_ms=sustain_ms)
#     out['session'] = session_idx
#     # optional: keep a trial index if present
#     out['trial_index'] = getattr(tr,'trialIndex', np.nan)
#     return out

# # =========================
# # Aggregation across sessions
# # =========================
# def analyze_respawn_over_sessions(
#     ai_trials: List[List[Dict[str, Any]]],
#     only_incorrect: bool = False,
#     only_correct: bool = False,
# ) -> Dict[str, pd.DataFrame]:
#     """
#     ai_trials: list of sessions; each session is a list of trial dicts.
#     Returns dict with:
#       - trial_df: per-trial metrics
#       - session_summary: median/IQR per session
#       - overall_summary: pooled median/IQR across sessions
#     """
#     per_trial_rows = []
#     for s_idx, session in enumerate(ai_trials, start=1):
#         for tr in session:
#             ans = tr.answer
#             if only_correct and ans != 1:
#                 continue
#             if only_incorrect and ans == 1:
#                 continue
#             row = compute_respawn_metrics_for_trial(tr, session_idx=s_idx)
#             per_trial_rows.append(row)

#     trial_df = pd.DataFrame(per_trial_rows)

#     # Ensure all expected columns exist
#     REQUIRED = [
#         'T_bci_ms','T_ai_ms','lag_ms',
#         'vpar_bci_AUC_0_500','vpar_ai_AUC_0_500',
#         'peak_MI_0_500','t_decay_MI_ms',
#         'path_dev_bci_0_1500','path_dev_ai_0_1500',
#         'path_dev_bci_0_500','path_dev_bci_500_1000','path_dev_bci_1000_1500',
#         'path_dev_ai_0_500','path_dev_ai_500_1000','path_dev_ai_1000_1500',
#         'cos_sim_mean_0_500','cos_sim_mean_500_1000','cos_sim_mean_1000_1500',
#     ]
#     for c in REQUIRED:
#         if c not in trial_df.columns:
#             trial_df[c] = np.nan

#     metric_cols = REQUIRED

#     def _agg_iqr(x: pd.Series) -> pd.Series:
#         valid = np.sum(~np.isnan(x)) > 0
#         q1 = np.nanpercentile(x, 25) if valid else np.nan
#         q3 = np.nanpercentile(x, 75) if valid else np.nan
#         return pd.Series({'median': np.nanmedian(x), 'q1': q1, 'q3': q3})

#     if 'session' not in trial_df.columns:
#         trial_df['session'] = 1

#     session_summary = (
#         trial_df.groupby('session', dropna=False)[metric_cols]
#                 .apply(lambda df: df.apply(_agg_iqr).unstack())
#                 .reset_index()
#     )

#     # Flatten MultiIndex columns like 'T_bci_ms_median'
#     session_summary.columns = ['session'] + [f"{m}_{stat}" for m, stat in session_summary.columns.tolist()[1:]]

#     overall = trial_df[metric_cols].apply(_agg_iqr).T
#     overall = overall.rename_axis('metric').reset_index()

#     return {'trial_df': trial_df, 'session_summary': session_summary, 'overall_summary': overall}

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats

# =========================
# Configurable constants
# =========================
THRESH_DEG_DEFAULT = 15.0    # half-cone angle for alignment
SUSTAIN_MS_DEFAULT = 200.0   # duration alignment must be sustained
BINS_MS = [(0, 500), (500, 1000), (1000, 1500)]
DEG = np.pi / 180.0

# =========================
# Core helpers
# =========================
def unit(v, eps: float = 1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def angle_between(u, v):
    u = unit(u); v = unit(v)
    cos = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    return np.arccos(cos)  # radians

def sustain_latency(times_ms: np.ndarray,
                    angles_rad: np.ndarray,
                    thresh_deg: float = THRESH_DEG_DEFAULT,
                    sustain_ms: float = SUSTAIN_MS_DEFAULT) -> float:
    """
    Latency from t0 to first time angle<thresh_deg sustained for >= sustain_ms.
    """
    if times_ms.size == 0 or angles_rad.size == 0:
        return np.nan
    ok = (angles_rad < (thresh_deg * DEG)).astype(int)
    dt = np.diff(times_ms, prepend=times_ms[0])
    run = 0.0
    for i in range(len(ok)):
        run = (run + dt[i]) if ok[i] == 1 else 0.0
        if run >= sustain_ms:
            # backtrack to the earliest time within this sustained window
            run2 = 0.0
            j = i
            while j >= 0 and ok[j] == 1 and run2 < sustain_ms:
                run2 += dt[j]
                j -= 1
            return float(times_ms[j + 1] - times_ms[0])
    return np.nan

def _trap_in_window(y: np.ndarray, t: np.ndarray, lo: float, hi: float) -> float:
    m = (t >= lo) & (t < hi)
    return float(np.trapz(y[m], t[m])) if np.any(m) else np.nan

def _coerce_vec3(x) -> np.ndarray:
    """
    Accepts list/tuple/np.ndarray or dict with keys ('x','y','z') or ('Item1','Item2','Item3').
    Returns (3,) float array; NaNs if not coercible.
    """
    if x is None:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size >= 3:
            return arr[:3]
        out = np.full(3, np.nan, dtype=float)
        out[:arr.size] = arr
        return out
    if isinstance(x, dict):
        for keys in (('x', 'y', 'z'), ('Item1', 'Item2', 'Item3')):
            if all(k in x for k in keys):
                return np.array([x[keys[0]], x[keys[1]], x[keys[2]]], dtype=float)
    return np.array([np.nan, np.nan, np.nan], dtype=float)

def _empty_metrics_dict() -> Dict[str, float]:
    return {
        "T_bci_ms": np.nan, "T_ai_ms": np.nan, "lag_ms": np.nan,
        "vpar_bci_AUC_0_500": np.nan, "vpar_ai_AUC_0_500": np.nan,
        "peak_MI_0_500": np.nan, "t_decay_MI_ms": np.nan,
        "path_dev_bci_0_1500": np.nan, "path_dev_ai_0_1500": np.nan,
        "path_dev_bci_0_500": np.nan, "path_dev_bci_500_1000": np.nan, "path_dev_bci_1000_1500": np.nan,
        "path_dev_ai_0_500": np.nan,  "path_dev_ai_500_1000": np.nan,  "path_dev_ai_1000_1500": np.nan,
        "cos_sim_mean_0_500": np.nan, "cos_sim_mean_500_1000": np.nan, "cos_sim_mean_1000_1500": np.nan,
    }

# =========================
# Per-trial metric computation
# =========================
def compute_respawn_metrics(trial: Any,
                            respawn_time_ms: float,
                            old_target, new_target,
                            thresh_deg: float = THRESH_DEG_DEFAULT,
                            sustain_ms: float = SUSTAIN_MS_DEFAULT) -> Dict[str, float]:
    """
    trial: object with attribute 'aiVelocities' (list of dicts). Each dict may have:
      'Input' (3,), 'Output' (3,), 'InputTimestamp'/'OutputTimestamp' (TNS ms), 'AvatarPosition' (3,) or nan.
    respawn_time_ms: float (TNS ms)
    old_target, new_target: 3D coords (x,y,z or dict)
    """
    rows = getattr(trial, 'aiVelocities', []) or []
    T, v_bci, v_ai, pos = [], [], [], []

    for r in rows:
        t = r.get('OutputTimestamp', None)
        if t is None:
            t = r.get('InputTimestamp', None)
        if t is None:
            continue
        T.append(float(np.asarray(t).ravel()[0]))
        v_bci.append(_coerce_vec3(r.get('Input')))
        v_ai.append(_coerce_vec3(r.get('Output')))
        pos.append(_coerce_vec3(r.get('AvatarPosition')))

    if len(T) == 0 or not np.isfinite(respawn_time_ms):
        return _empty_metrics_dict()

    T = np.asarray(T, dtype=float)
    v_bci = np.vstack(v_bci).astype(float)
    v_ai  = np.vstack(v_ai).astype(float)
    pos   = np.vstack(pos).astype(float)

    # Keep samples after respawn and re-zero time
    mask = T >= respawn_time_ms
    if not np.any(mask):
        return _empty_metrics_dict()
    T = T[mask] - respawn_time_ms
    v_bci = v_bci[mask]; v_ai = v_ai[mask]; pos = pos[mask]

    # Fill missing positions (per-axis interp; if none available, zeros)
    for k in range(3):
        col = pos[:, k]
        if np.isnan(col).any():
            idx = np.where(~np.isnan(col))[0]
            if idx.size > 0:
                nan_idx = np.where(np.isnan(col))[0]
                col[nan_idx] = np.interp(nan_idx, idx, col[idx])
            else:
                col[:] = 0.0
        pos[:, k] = col

    old_target = _coerce_vec3(old_target)
    new_target = _coerce_vec3(new_target)

    g_old = unit(old_target - pos)
    g_new = unit(new_target - pos)

    # Angles to new goal & parallel speed
    ang_bci = angle_between(v_bci, g_new)
    ang_ai  = angle_between(v_ai,  g_new)
    vpar_bci = np.sum(v_bci * g_new, axis=1)
    vpar_ai  = np.sum(v_ai  * g_new, axis=1)

    # Reorientation latencies
    T_bci = sustain_latency(T, ang_bci, thresh_deg=thresh_deg, sustain_ms=sustain_ms)
    T_ai  = sustain_latency(T, ang_ai,  thresh_deg=thresh_deg, sustain_ms=sustain_ms)
    lag_ms = (T_ai - T_bci) if (np.isfinite(T_bci) and np.isfinite(T_ai)) else np.nan

    # Mismatch index (AI relative to old vs new goals)
    cos_old = np.sum(unit(v_ai) * g_old, axis=1)
    cos_new = np.sum(unit(v_ai) * g_new, axis=1)
    MI = cos_old - cos_new
    early = T <= 500
    peak_MI = np.nanmax(MI[early]) if np.any(early) else np.nan
    t_decay = np.nan
    for tt, mi in zip(T, MI):
        if tt >= 0 and mi <= 0:
            t_decay = float(tt); break

    # Lateral speed & path deviation (overall + binned)
    lat_bci = np.linalg.norm(v_bci - (vpar_bci[:, None] * g_new), axis=1)
    lat_ai  = np.linalg.norm(v_ai  - (vpar_ai[:,  None] * g_new), axis=1)

    pathdev_bci_bins = [_trap_in_window(lat_bci, T, a, b) for (a, b) in BINS_MS]
    pathdev_ai_bins  = [_trap_in_window(lat_ai,  T, a, b) for (a, b) in BINS_MS]

    win1500 = T <= 1500
    pathdev_bci_1500 = float(np.trapz(lat_bci[win1500], T[win1500])) if np.any(win1500) else np.nan
    pathdev_ai_1500  = float(np.trapz(lat_ai[win1500],  T[win1500])) if np.any(win1500) else np.nan

    # BCI–AI cosine similarity (listening), binned
    cos_sim = np.sum(unit(v_bci) * unit(v_ai), axis=1)
    cos_mean_bins = [
        np.nanmean(cos_sim[(T >= a) & (T < b)]) if np.any((T >= a) & (T < b)) else np.nan
        for (a, b) in BINS_MS
    ]

    # Early parallel drive (AUC 0–500 ms)
    vpar_bci_auc_0_500 = float(np.trapz(vpar_bci[early], T[early])) if np.any(early) else np.nan
    vpar_ai_auc_0_500  = float(np.trapz(vpar_ai[early],  T[early])) if np.any(early) else np.nan

    return {
        "T_bci_ms": T_bci, "T_ai_ms": T_ai, "lag_ms": lag_ms,
        "vpar_bci_AUC_0_500": vpar_bci_auc_0_500,
        "vpar_ai_AUC_0_500":  vpar_ai_auc_0_500,
        "peak_MI_0_500": peak_MI, "t_decay_MI_ms": t_decay,
        "path_dev_bci_0_1500": pathdev_bci_1500, "path_dev_ai_0_1500": pathdev_ai_1500,
        "path_dev_bci_0_500": pathdev_bci_bins[0],
        "path_dev_bci_500_1000": pathdev_bci_bins[1],
        "path_dev_bci_1000_1500": pathdev_bci_bins[2],
        "path_dev_ai_0_500": pathdev_ai_bins[0],
        "path_dev_ai_500_1000": pathdev_ai_bins[1],
        "path_dev_ai_1000_1500": pathdev_ai_bins[2],
        "cos_sim_mean_0_500":     cos_mean_bins[0],
        "cos_sim_mean_500_1000":  cos_mean_bins[1],
        "cos_sim_mean_1000_1500": cos_mean_bins[2],
    }

def compute_respawn_metrics_for_trial(tr: Any,
                                      session_idx: int,
                                      thresh_deg: float = THRESH_DEG_DEFAULT,
                                      sustain_ms: float = SUSTAIN_MS_DEFAULT) -> Dict[str, Any]:
    """
    Trial-like object attributes expected:
      - targetJumpTime (TNS ms), targetPosition, targetJumpPosition, aiVelocities, answer
    """
    respawn_time_ms = getattr(tr, 'targetJumpTime', np.nan)
    old_target = getattr(tr, 'targetPosition', [np.nan, np.nan, np.nan])
    new_target = getattr(tr, 'targetJumpPosition', [np.nan, np.nan, np.nan])

    out = compute_respawn_metrics(tr, respawn_time_ms, old_target, new_target,
                                  thresh_deg=thresh_deg, sustain_ms=sustain_ms)
    out['session'] = session_idx
    out['answer']  = getattr(tr, 'answer', np.nan)
    out['trial_index'] = getattr(tr, 'trialIndex', np.nan)
    return out

# =========================
# Aggregation
# =========================
REQUIRED_METRICS = [
    'T_bci_ms','T_ai_ms','lag_ms',
    'vpar_bci_AUC_0_500','vpar_ai_AUC_0_500',
    'peak_MI_0_500','t_decay_MI_ms',
    'path_dev_bci_0_1500','path_dev_ai_0_1500',
    'path_dev_bci_0_500','path_dev_bci_500_1000','path_dev_bci_1000_1500',
    'path_dev_ai_0_500','path_dev_ai_500_1000','path_dev_ai_1000_1500',
    'cos_sim_mean_0_500','cos_sim_mean_500_1000','cos_sim_mean_1000_1500',
]

def _agg_iqr(x: pd.Series) -> pd.Series:
    valid = np.sum(~np.isnan(x)) > 0
    q1 = np.nanpercentile(x, 25) if valid else np.nan
    q3 = np.nanpercentile(x, 75) if valid else np.nan
    return pd.Series({'median': np.nanmedian(x), 'q1': q1, 'q3': q3})

def analyze_respawn_over_sessions(
    ai_trials: List[List[Any]],
    only_incorrect: bool = False,
    only_correct: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Returns per-trial metrics + session/overall summaries, with optional filtering.
    """
    per_trial_rows = []
    for s_idx, session in enumerate(ai_trials, start=1):
        for tr in session:
            ans = getattr(tr, 'answer', np.nan)
            if only_correct  and ans != 1:  continue
            if only_incorrect and ans == 1:  continue
            row = compute_respawn_metrics_for_trial(tr, session_idx=s_idx)
            per_trial_rows.append(row)

    trial_df = pd.DataFrame(per_trial_rows)
    if trial_df.empty:
        return {'trial_df': trial_df, 'session_summary': pd.DataFrame(), 'overall_summary': pd.DataFrame()}

    for c in REQUIRED_METRICS:
        if c not in trial_df.columns:
            trial_df[c] = np.nan
    if 'session' not in trial_df.columns:
        trial_df['session'] = 1

    session_summary = (
        trial_df.groupby('session', dropna=False)[REQUIRED_METRICS]
                .apply(lambda df: df.apply(_agg_iqr).unstack())
                .reset_index()
    )
    session_summary.columns = ['session'] + [f"{m}_{stat}" for m, stat in session_summary.columns.tolist()[1:]]

    overall = trial_df[REQUIRED_METRICS].apply(_agg_iqr).T
    overall = overall.rename_axis('metric').reset_index()

    return {'trial_df': trial_df, 'session_summary': session_summary, 'overall_summary': overall}

# =========================
# Statistical tests (Correct vs Incorrect)
# =========================
def _benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH-FDR correction; returns q-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q[order] = q_sorted
    return q.tolist()

def _per_session_medians(trial_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns df with columns: session, correct, incorrect (medians for sessions that have both).
    """
    dfc = (trial_df[trial_df['answer']==1]
           .groupby('session')[metric].median().rename('correct'))
    dfi = (trial_df[trial_df['answer']!=1]
           .groupby('session')[metric].median().rename('incorrect'))
    out = pd.concat([dfc, dfi], axis=1).dropna().reset_index()
    return out

def _cliffs_delta_from_U(u_stat: float, n1: int, n2: int) -> float:
    """Approximate Cliff’s delta from Mann–Whitney U (no-ties assumption)."""
    if n1 == 0 or n2 == 0:
        return np.nan
    return (2.0*u_stat)/(n1*n2) - 1.0

def compute_stats_correct_vs_incorrect(trial_df_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      - session_level_stats: paired Wilcoxon on per-session medians (sessions with both groups),
                             with median/IQR of (incorrect - correct) and BH-FDR q-values.
      - pooled_trial_stats: Mann–Whitney U on pooled trials, Cliff’s delta, BH-FDR q-values.
    """
    if trial_df_all.empty:
        return {'session_level_stats': pd.DataFrame(), 'pooled_trial_stats': pd.DataFrame()}

    metrics = REQUIRED_METRICS

    # Session-level (paired) on per-session medians
    sess_rows = []
    for m in metrics:
        med_df = _per_session_medians(trial_df_all, m)
        if med_df.empty:
            continue
        x = med_df['incorrect'].values
        y = med_df['correct'].values
        diffs = x - y
        try:
            w = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', method='approx')
            p = float(w.pvalue); stat = float(w.statistic)
        except Exception:
            p = np.nan; stat = np.nan
        sess_rows.append({
            'metric': m,
            'N_sessions': int(med_df.shape[0]),
            'median_diff_incorrect_minus_correct': float(np.nanmedian(diffs)),
            'diff_q1': float(np.nanpercentile(diffs, 25)),
            'diff_q3': float(np.nanpercentile(diffs, 75)),
            'wilcoxon_stat': stat,
            'p_value': p,
        })
    session_level_stats = pd.DataFrame(sess_rows)
    if not session_level_stats.empty:
        session_level_stats['q_value'] = _benjamini_hochberg(session_level_stats['p_value'].fillna(1.0).tolist())

    # Pooled trials (independent) + Cliff’s delta
    pool_rows = []
    df_c = trial_df_all[trial_df_all['answer'] == 1]
    df_i = trial_df_all[trial_df_all['answer'] != 1]
    for m in metrics:
        x = df_i[m].dropna().values; n1 = len(x)
        y = df_c[m].dropna().values; n2 = len(y)
        if n1 == 0 or n2 == 0:
            continue
        try:
            u = stats.mannwhitneyu(x, y, alternative='two-sided', method='auto')
            p = float(u.pvalue); ustat = float(u.statistic)
        except Exception:
            p = np.nan; ustat = np.nan
        delta = _cliffs_delta_from_U(ustat, n1, n2) if np.isfinite(ustat) else np.nan
        pool_rows.append({
            'metric': m, 'n_incorrect': n1, 'n_correct': n2,
            'mw_U': ustat, 'p_value': p, 'cliffs_delta': delta,
            'median_incorrect': float(np.nanmedian(x)) if n1>0 else np.nan,
            'median_correct':   float(np.nanmedian(y)) if n2>0 else np.nan,
        })
    pooled_trial_stats = pd.DataFrame(pool_rows)
    if not pooled_trial_stats.empty:
        pooled_trial_stats['q_value'] = _benjamini_hochberg(pooled_trial_stats['p_value'].fillna(1.0).tolist())

    return {'session_level_stats': session_level_stats,
            'pooled_trial_stats': pooled_trial_stats}

# =========================
# Convenience: run everything in one go
# =========================
def analyze_with_stats(ai_trials: List[List[Any]]) -> Dict[str, pd.DataFrame]:
    """
    Produces:
      - all_trials_df (per-trial metrics, no filtering)
      - correct / incorrect summaries (session + overall)
      - session_level_stats (paired Wilcoxon on per-session medians)
      - pooled_trial_stats (Mann–Whitney + Cliff’s delta)
    """
    # All trials
    all_rows = []
    for s_idx, session in enumerate(ai_trials, start=1):
        for tr in session:
            all_rows.append(compute_respawn_metrics_for_trial(tr, session_idx=s_idx))
    all_trials_df = pd.DataFrame(all_rows)

    correct = analyze_respawn_over_sessions(ai_trials, only_correct=True)
    incorrect = analyze_respawn_over_sessions(ai_trials, only_incorrect=True)
    stats_out = compute_stats_correct_vs_incorrect(all_trials_df)

    return {
        'all_trials_df': all_trials_df,
        'correct_trial_df': correct['trial_df'],
        'incorrect_trial_df': incorrect['trial_df'],
        'correct_session_summary': correct['session_summary'],
        'incorrect_session_summary': incorrect['session_summary'],
        'correct_overall_summary': correct['overall_summary'],
        'incorrect_overall_summary': incorrect['overall_summary'],
        'session_level_stats': stats_out['session_level_stats'],
        'pooled_trial_stats': stats_out['pooled_trial_stats'],
    }

