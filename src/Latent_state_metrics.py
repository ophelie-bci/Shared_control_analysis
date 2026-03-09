import pandas as pd
import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.interpolate import interp1d
from collections import defaultdict
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

### MAIN ###
def extract_latents_per_trial(latents, correct_trials, target_map,
                              n_timepoints=80, use_dims=(0,1,2,3,4,5), tol=1e-2):
    latents = _unwrap_latents_obj(latents)
    rows = []

    def to_target_tuple(tp):
        tp = np.asarray(tp, float)
        for k in target_map.keys():
            if np.allclose(tp, k, atol=tol, rtol=0):
                return k
        return tuple(np.round(tp, 2))

    def _is_ai_on(v, tol=1e-6):
        return v is not None and float(v) >= 1.0 - tol

    for idx, tr in enumerate(correct_trials):
        trial_id = tr.trial
        ai_status = 1 if _is_ai_on(tr.aiVelocityFactor) else 0

        # get latents for this trial_id (your existing logic) ...
        if isinstance(latents, dict): 
            lat_seq = (latents.get(trial_id) or latents.get(str(trial_id))) 
            if lat_seq is None: # fall back to positional index if present 
                lat_seq = latents.get(idx) or latents.get(str(idx)) 
        else: 
            lat_seq = latents[trial_id] if trial_id < len(latents) else latents[idx] 
        
        lat_seq = np.asarray(lat_seq) 
        if lat_seq.ndim != 2 or lat_seq.shape[0] < 2: 
            continue # skip malformed/too-short trials

        start = compute_movement_onset(tr)
        end   = compute_target_reached_indices([tr])[0]
        if end is None or end <= start or end > lat_seq.shape[0]:
            end = lat_seq.shape[0]

        lat_seq = lat_seq[start:end]
        if lat_seq.shape[0] < 2:
            continue

        lat_seq = lat_seq[:, list(use_dims)]

        # single interpolation pass here
        t_orig = np.linspace(0, 1, len(lat_seq))
        t_new  = np.linspace(0, 1, n_timepoints)
        interp = interp1d(t_orig, lat_seq, axis=0, kind='linear', fill_value="extrapolate")
        latents_interp = interp(t_new)

        rows.append({
            "trial_id": int(trial_id),
            "ai_status": ai_status,
            "target": to_target_tuple(tr.targetPosition),  # tolerant
            "latents": latents_interp,
        })

    return pd.DataFrame(rows)


def compute_pairwise_metrics_df(df, cat1, cat2, ai_status, latent_col='latents', k=3,
                                vaf_k=2, seed=123, decoder_D=None, zscore_per_trial=False):
    rng = np.random.default_rng(seed)

    if ai_status == 'compare_ai':
        t1 = df[(df['target'] == cat1) & (df['ai_status'] == 1)][latent_col].tolist()
        t2 = df[(df['target'] == cat2) & (df['ai_status'] == 0)][latent_col].tolist()
    else:
        t1 = df[(df['target'] == cat1) & (df['ai_status'] == ai_status)][latent_col].tolist()
        t2 = df[(df['target'] == cat2) & (df['ai_status'] == ai_status)][latent_col].tolist()

    if len(t1) < 3 or len(t2) < 3:
        return {"cat1": cat1, "cat2": cat2, "error": "Insufficient trials"}

    trials1 = np.array(t1)  # (N1, T, D)
    trials2 = np.array(t2)  # (N2, T, D)

    # mean trajectories
    mean1 = trials1.mean(axis=0)   # (T, D)
    mean2 = trials2.mean(axis=0)   # (T, D)

    # Subspace angles
    subspace = subspace_metrics(trials1, trials2, k=k, zscore_per_trial=zscore_per_trial)

    # # VAF with top-k
    # D = mean1.shape[1]
    # k = min(vaf_k, D)
    # U,S,Vt = np.linalg.svd(mean1, full_matrices=False)
    # B = Vt.T[:, :k]
    # projected = (mean2 @ B) @ B.T
    # total_var = np.sum(mean2**2)
    # recon_var = np.sum(projected**2)
    # vaf_percent = 100 * recon_var / total_var if total_var > 0 else np.nan

    # Procrustes
    _, _, disparity = procrustes(mean1, mean2)

    # CCA on equalized row counts (deterministic slice)
    X1 = trials1.reshape(-1, trials1.shape[-1])
    X2 = trials2.reshape(-1, trials2.shape[-1])
    m = min(len(X1), len(X2))
    X1, X2 = X1[:m], X2[:m]

    scaler1, scaler2 = StandardScaler(), StandardScaler()
    X1_std = scaler1.fit_transform(X1)
    X2_std = scaler2.fit_transform(X2)
    cca = CCA(n_components=min(3, X1_std.shape[1]))
    Xc, Yc = cca.fit_transform(X1_std, X2_std)
    cca_corr = np.mean([np.corrcoef(Xc[:,i], Yc[:,i])[0,1] for i in range(Xc.shape[1])])

    # Classification (seeded CV + seeded permutation)
    # X_class = np.concatenate([trials1, trials2], axis=0).reshape(len(trials1)+len(trials2), -1)
    # y_class = np.array([1]*len(trials1) + [0]*len(trials2))
    # clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # acc, acc_p = permutation_test_accuracy(X_class, y_class, clf, cv=cv, n_permutations=1000, random_state=seed)
    X_class = np.concatenate([trials1, trials2], axis=0).reshape(len(trials1)+len(trials2), -1)
    y_class = np.array([1]*len(trials1) + [0]*len(trials2))

    n_comp = min(3, X_class.shape[1])
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc, acc_p = permutation_test_accuracy(X_class, y_class, pipe, cv=cv,
                                       n_permutations=1000, random_state=seed)
    # # ---- NEW: decoder in-/out-of-plane energy (optional) ----
    # out1 = out2 = in1 = in2 = np.nan
    # out1_std = out2_std = in1_std = in2_std = np.nan
    # decoder_rank = np.nan
    # if decoder_D is not None:
    #     P_D, P_null, decoder_rank = _decoder_projectors(decoder_D)
    #     out1, out1_std, in1, in1_std = _out_of_plane_stats(trials1, P_D, P_null)
    #     out2, out2_std, in2, in2_std = _out_of_plane_stats(trials2, P_D, P_null)

    # === Mean trajectory geometry (for figures + compression downstream) ===
    K_resample = 100  # or 200 if you want smoother ribbons
    mean1, std1 = _mean_traj(trials1, K=K_resample)  # (K,D)
    mean2, std2 = _mean_traj(trials2, K=K_resample)  # (K,D)
    d_prof, d_mean, d_auc = _pairwise_distance_profile(mean1, mean2)
    d_rms = float(np.sqrt(np.mean(d_prof**2)))

    out = {
        "cat1": cat1, "cat2": cat2,
        "mean_distance": d_mean,
        "angle_1_deg": subspace["angle_1_deg"], "mean_angle_deg": subspace["mean_angle_deg"],
        "align_X_in_Y": subspace["align_X_in_Y"],
        "align_Y_in_X": subspace["align_Y_in_X"],
        "align_sym":    subspace["align_sym"],
        "outside_X_wrt_Y": subspace["outside_X_wrt_Y"],
        "outside_Y_wrt_X": subspace["outside_Y_wrt_X"],
        "procrustes_disparity": disparity,
        "cca_correlation": cca_corr,
        "classification_accuracy":acc,
        "classification_accuracy_p": acc_p,
        # "VAF": vaf_percent ,

        # NEW: pairwise geometry to support H2 and your figures
        "mean_distance": d_mean,           # scalar; use this for compression aggregation
        "distance_auc": d_auc,             # scalar; optional
        "rms_distance": d_rms,
        "mean_traj_cat1": mean1,           # (K,D) keep if you want to plot directly
        "mean_traj_cat2": mean2,           # (K,D)
        "std_traj_cat1": std1,             # (K,D) for ribbons
        "std_traj_cat2": std2,             # (K,D)
        # optional (large): "distance_profile": d_prof,
    }
    return out

#### HELP FUNCTIONS #### 
def _decoder_projectors(decoder_D, tol=1e-12):
    """
    From a 2×D (or r×D) readout matrix D, build orthogonal projectors onto
    span(Dᵀ) and its orthogonal complement (null(D)).
    """
    decoder_D = np.asarray(decoder_D)
    U, S, Vt = np.linalg.svd(decoder_D, full_matrices=False)
    r = np.sum(S > tol)
    B = Vt[:r].T                    # D×r, orthonormal basis for span(Dᵀ)
    P_D = B @ B.T                   # projector onto decoder subspace
    P_null = np.eye(B.shape[0]) - P_D
    return P_D, P_null, int(r)

def _out_of_plane_stats(trials, P_D, P_null):
    """
    trials: (N, T, D) array of latents per trial.
    Returns mean/std of per-trial out-of-plane fractions (and in-plane).
    """
    out_fracs, in_fracs = [], []
    for Z in trials:                # Z: (T, D)
        tot = float(np.sum(Z**2))
        if tot <= 0:
            out_fracs.append(np.nan); in_fracs.append(np.nan); continue
        Z_out = Z @ P_null
        Z_in  = Z @ P_D
        e_out = float(np.sum(Z_out**2))
        e_in  = float(np.sum(Z_in**2))
        out_fracs.append(e_out / tot)
        in_fracs.append(e_in / tot)
    out_fracs = np.array(out_fracs, dtype=float)
    in_fracs  = np.array(in_fracs,  dtype=float)
    return (np.nanmean(out_fracs), np.nanstd(out_fracs),
            np.nanmean(in_fracs),  np.nanstd(in_fracs))

def target_label(pos, target_map):
    label = target_map.get(tuple(np.asarray(pos, float).round(2)), "unknown")
    return label


def _unwrap_latents_obj(latents):
    # Accept dicts with common keys or plain list/array
    if isinstance(latents, dict):
        for k in ("Z", "latents", "trial_latents"):
            if k in latents:
                return latents[k]
    return latents

def compute_movement_onset(trial, delay=200):
    """Compute the movement onset time, defined as the first time the sphere moves after the delay, 
    and return it as a bin index."""
    
    time = np.array(trial.modelVelocity['time'])
    vx = np.array(trial.modelVelocity['vx'])
    vz = np.array(trial.modelVelocity['vz'])
    
    # Compute the index where movement starts after the delay (200ms)
    start_index = np.argmin(np.abs(time - delay / 1000))  # Find the start index (delay in seconds)
    
    # Movement onset: first time after the delay where velocity is non-zero
    movement_onset = np.argmax(np.abs(vx[start_index:]) > 0.000001) + start_index
    return movement_onset if movement_onset > 0 else 0

def compute_target_reached_indices(trials):
    indices = []
    for trial in trials:
        try:
            if hasattr(trial, 'targetReached') and hasattr(trial, 'avatarVelocity'):
                target_time = trial.targetReached
                time_array = trial.avatarVelocity['time']
                closest_idx = np.argmin(np.abs(time_array - target_time))
                indices.append(closest_idx)
            else:
                indices.append(None)
        except Exception as e:
            print(f"Error with trial {getattr(trial, 'trial', '?')}: {e}")
            indices.append(None)
    return indices

def permutation_test_accuracy(
    X, y, clf, cv=None, n_permutations=1000, random_state=None, max_splits=5
):
    """
    Cross-validated accuracy with a permutation test (one-sided p: P(perm_acc >= acc)).

    - Automatically caps n_splits to the smallest class count (>=2).
    - If there aren't >=2 samples per class, returns (nan, 1.0) instead of crashing.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    clf : sklearn estimator
    cv : sklearn splitter or None
        If None, a StratifiedKFold with safe n_splits is created.
        If provided but has too many splits, it is replaced by a safe StratifiedKFold.
    n_permutations : int
    random_state : int or None
    max_splits : int
        Upper bound on number of folds (default: 5).

    Returns
    -------
    acc : float
        Mean CV accuracy on the real labels.
    p_value : float
        One-sided permutation p-value: fraction of permuted accuracies >= acc.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Determine safe number of splits
    _, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if counts.size else 0
    if min_class < 2:
        return np.nan, 1.0

    n_splits = min(max_splits, min_class)
    # Build/adjust CV if needed
    if not isinstance(cv, StratifiedKFold) or getattr(cv, "n_splits", None) is None or cv.n_splits > min_class:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rng = np.random.default_rng(random_state)

    # Actual accuracy
    actual_scores = []
    for train_idx, test_idx in cv.split(X, y):
        c = clone(clf)
        c.fit(X[train_idx], y[train_idx])
        actual_scores.append(c.score(X[test_idx], y[test_idx]))
    acc = float(np.mean(actual_scores)) if len(actual_scores) else np.nan

    # Permutation distribution
    permuted_scores = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_scores = []
        for train_idx, test_idx in cv.split(X, y_perm):
            c = clone(clf)
            c.fit(X[train_idx], y_perm[train_idx])
            perm_scores.append(c.score(X[test_idx], y_perm[test_idx]))
        if perm_scores:
            permuted_scores.append(np.mean(perm_scores))

    p_value = float(np.mean([s >= acc for s in permuted_scores])) if permuted_scores else 1.0
    return acc, p_value

def subspace_metrics(trials1, trials2, k=3, zscore_per_trial=False):
    # stack all trials & normalize within-trial to avoid mean biases
    X1 = _stack_trials(trials1, center_per_trial=True, zscore_per_trial=zscore_per_trial)
    X2 = _stack_trials(trials2, center_per_trial=True, zscore_per_trial=zscore_per_trial)

    # top-k bases
    U1 = _pca_basis(X1, k)
    U2 = _pca_basis(X2, k)

    # covariances
    C1 = _cov(X1)
    C2 = _cov(X2)

    # ---- variance explained by top-k PCs (inside the full latent space) ----
    # total variance = trace of covariance
    total_var_X = float(np.trace(C1))
    total_var_Y = float(np.trace(C2))
    # variance captured by the k-dim subspace: trace(U^T C U)
    var_k_X = float(np.trace(U1.T @ C1 @ U1))
    var_k_Y = float(np.trace(U2.T @ C2 @ U2))

    var_explained_X = var_k_X / total_var_X if total_var_X > 0 else np.nan
    var_explained_Y = var_k_Y / total_var_Y if total_var_Y > 0 else np.nan

    # principal angles
    ang = _principal_angles(U1, U2)
    angle_1_deg   = float(np.degrees(ang[0]))
    mean_angle_deg = float(np.degrees(ang).mean())

    # alignment indices
    C1 = _cov(X1)
    C2 = _cov(X2)
    align_X_in_Y = _alignment_index(C1, U1, U2)  # AI(trials1 -> basis of trials2)
    align_Y_in_X = _alignment_index(C2, U2, U1)  # AI(trials2 -> basis of trials1)
    align_sym = float(np.sqrt(align_X_in_Y * align_Y_in_X))  # symmetric summary

    # outside-manifold energy (both directions)
    out_X_wrt_Y = _outside_energy(X1, U2)
    out_Y_wrt_X = _outside_energy(X2, U1)

    return {
        "angle_1_deg": angle_1_deg,
        "mean_angle_deg": mean_angle_deg,
        "align_X_in_Y": align_X_in_Y,
        "align_Y_in_X": align_Y_in_X,
        "align_sym": align_sym,
        "outside_X_wrt_Y": out_X_wrt_Y,
        "outside_Y_wrt_X": out_Y_wrt_X,
    }

def _stack_trials(trials, center_per_trial=True, zscore_per_trial=False):
    """
    trials: list/array of shape [(T,D), ...]  -> returns X: (sum_T, D)
    """
    Xs = []
    for tr in trials:
        X = np.asarray(tr)
        if center_per_trial:
            X = X - X.mean(axis=0, keepdims=True)
        if zscore_per_trial:
            sd = X.std(axis=0, keepdims=True) + 1e-9
            X = X / sd
        Xs.append(X)
    return np.vstack(Xs)

def _pca_basis(X, k):
    """Return top-k orthonormal basis (D x k)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # columns of V (rows of Vt) span feature space
    return Vt[:k].T  # (D x k)

def _principal_angles(U1, U2):
    """U1,U2: (D x k) with orthonormal columns -> principal angles (rad)."""
    M = U1.T @ U2
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)  # (k,)

def _proj_matrix(U):
    return U @ U.T  # (D x D)

def _cov(X):
    # X: (N x D)
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / max(len(Xc)-1, 1)

def _alignment_index(C_X, U_X, U_Y, eps=1e-12):
    """
    Fraction of X-variance captured by Y's k-dim subspace,
    normalized by X's own top-k variance.
      AI(X->Y) = tr(U_Y^T C_X U_Y) / tr(U_X^T C_X U_X)
    """
    num  = np.trace(U_Y.T @ C_X @ U_Y)
    denom = np.trace(U_X.T @ C_X @ U_X) + eps
    return float(num / denom)

def _outside_energy(X, U_Y, eps=1e-12):
    """
    Energy of X outside Y's subspace, as fraction of total energy.
      = ||(I - P_Y) X||_F^2 / ||X||_F^2
    """
    PY = _proj_matrix(U_Y)         # (D x D)
    Xproj = X @ PY                 # (N x D)
    R = X - Xproj
    num = np.sum(R*R)
    denom = np.sum(X*X) + eps
    return float(num / denom)


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

from itertools import combinations
from scipy.spatial import ConvexHull

def compression_ratio_from_pairwise(results_on, results_off):
    """results_*: list of dicts returned by compute_pairwise_metrics_df for a fixed status."""
    mu_on  = np.mean([r["mean_distance"] for r in results_on])
    mu_off = np.mean([r["mean_distance"] for r in results_off])
    return float(mu_on / mu_off)

def _pca_reduce(X, k=3):
    Xc = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T

def hull_volume_of_target_centroids(df, status, latent_col="latents", K=100, reduce_to=3):
    centroids = []
    for t in sorted(df["target"].unique()):
        trials = df[(df["target"]==t) & (df["ai_status"]==status)][latent_col].tolist()
        if len(trials) == 0: 
            continue
        T = np.stack(trials)                # (N,T,D)
        M,_ = _mean_traj(T, K=K)            # (K,D)
        centroids.append(M.mean(axis=0))    # (D,)
    if len(centroids) < 4:
        return np.nan
    C = np.stack(centroids)                 # (n_targets, D)
    C3 = _pca_reduce(C, k=reduce_to)        # (n_targets, 3)
    return float(ConvexHull(C3).volume)

def status_level_geometry(df, latent_col="latents"):
    vol_on  = hull_volume_of_target_centroids(df, status=1, latent_col=latent_col)
    vol_off = hull_volume_of_target_centroids(df, status=0, latent_col=latent_col)
    return {
        "hull_volume_on":  vol_on,
        "hull_volume_off": vol_off,
        "hull_ratio":      (vol_on/vol_off) if (vol_on>0 and vol_off>0) else np.nan,
    }
### AGGREGATE RESULTS ###
# ---------------- helpers ----------------

def _canon_pair(a, b):
    """Canonicalize so (A,B) and (B,A) are the same key."""
    return tuple(sorted([tuple(a), tuple(b)]))

def _np_mean_sd(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0: return np.nan, np.nan
    if x.size == 1: return float(x[0]), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))

def _combine_p_stouffer(pvals, weights=None):
    p = np.asarray(pvals, dtype=np.float64)
    p = p[~np.isnan(p)]
    if p.size == 0: return np.nan
    z = norm.isf(p)  # one-sided
    if weights is None:
        zc = np.sum(z) / np.sqrt(len(z))
    else:
        w = np.asarray(weights, dtype=np.float64)[:len(z)]
        zc = np.sum(w*z) / np.sqrt(np.sum(w**2))
    return float(norm.sf(zc))

def _fisher_z_mean(r):
    r = np.asarray(r, dtype=np.float64)
    r = r[~np.isnan(r)]
    if r.size == 0: return np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(np.mean(z)))

def _as_label(key, label_map):
    # key can be a tuple of floats; map to human label if provided
    return label_map.get(tuple(key), key) if label_map else key

# -------------- core aggregator ----------------

# def aggregate_all_sessions(
#     all_results,
#     target_map=None,
#     alpha=0.05,
#     fisher_for_cca=True,
#     keep_raw_angles=True,
# ):
#     """
#     all_results: list of per-session dicts with keys:
#        'target_vs_target_ai_on', 'target_vs_target_ai_off', 'ai_on_vs_off_same_target'
#        where each entry is a dict with metrics or {'error': ...}

#     Returns dict with DataFrames:
#       - ai_on
#       - ai_off
#       - ai_on_vs_off_same_target
#       - delta_on_minus_off  (paired per session)
#     """
#     n_sessions = len(all_results)
#     label_map = {k:v for k,v in (target_map or {}).items()}  # tuple->label

#     # Buckets for aggregation
#     bucket_on   = defaultdict(list)   # keyed by pair
#     bucket_off  = defaultdict(list)
#     bucket_same = defaultdict(list)   # keyed by single target

#     # Also keep per-session lookups for Δ (ON−OFF) pairing
#     per_sess_on  = []
#     per_sess_off = []

#     for sess in all_results:
#         sess_on  = {}
#         sess_off = {}
#         # ON pairs
#         for rec in sess.get('target_vs_target_ai_on', []):
#             if 'error' in rec: continue
#             key = _canon_pair(rec['cat1'], rec['cat2'])
#             bucket_on[key].append(rec)
#             sess_on[key] = rec
#         # OFF pairs
#         for rec in sess.get('target_vs_target_ai_off', []):
#             if 'error' in rec: continue
#             key = _canon_pair(rec['cat1'], rec['cat2'])
#             bucket_off[key].append(rec)
#             sess_off[key] = rec
#         # SAME target (ON vs OFF)
#         for rec in sess.get('ai_on_vs_off_same_target', []):
#             if 'error' in rec: continue
#             # in your code cat1==cat2; use one key
#             key = tuple(rec['cat1'])
#             bucket_same[key].append(rec)

#         per_sess_on.append(sess_on)
#         per_sess_off.append(sess_off)

#     def _aggregate_bucket(bucket, is_pair=True):
#         rows = []
#         for key, recs in bucket.items():
#             # pull arrays per metric
#             def vec(m):
#                 return [r.get(m, np.nan) for r in recs]

#             # angles: keep raw values exactly (no cosine-mean)
#             th1_mean, th1_sd = _np_mean_sd(vec('principal_angle_1'))
#             thm_mean, thm_sd = _np_mean_sd(vec('mean_principal_angle'))

#             # procrustes / accuracy / p / vaf: simple mean/sd
#             proc_mean, proc_sd = _np_mean_sd(vec('procrustes_disparity'))
#             acc_mean,  acc_sd  = _np_mean_sd(vec('classification_accuracy'))
#             p_mean,    p_sd    = _np_mean_sd(vec('classification_accuracy_p'))
#             vaf_mean,  vaf_sd  = _np_mean_sd(vec('VAF'))

#             # cca: optionally Fisher-z mean; also report raw mean
#             cca_raw = np.asarray(vec('cca_correlation'), dtype=np.float64)
#             cca_mean_raw, cca_sd = _np_mean_sd(cca_raw)
#             cca_mean = _fisher_z_mean(cca_raw) if fisher_for_cca else cca_mean_raw

#             # p-combine + fraction significant
#             pvals = np.asarray(vec('classification_accuracy_p'), dtype=np.float64)
#             p_comb = _combine_p_stouffer(pvals)
#             acc_sig_frac = float(np.mean(pvals < alpha)) if pvals.size else np.nan

#             base = {
#                 'n_sessions': n_sessions,
#                 'n_valid': len(recs),
#                 'n_error': n_sessions - len(recs),
#                 'coverage': (len(recs) / n_sessions) if n_sessions else np.nan,

#                 'principal_angle_1_mean': th1_mean,
#                 'principal_angle_1_std': th1_sd,
#                 'mean_principal_angle_mean': thm_mean,
#                 'mean_principal_angle_std': thm_sd,

#                 'procrustes_disparity_mean': proc_mean,
#                 'procrustes_disparity_std': proc_sd,

#                 'cca_correlation_mean': cca_mean,
#                 'cca_correlation_std': cca_sd,
#                 'cca_correlation_mean_raw': cca_mean_raw,  # optional visibility

#                 'classification_accuracy_mean': acc_mean,
#                 'classification_accuracy_std': acc_sd,
#                 'classification_accuracy_p_mean': p_mean,
#                 'classification_accuracy_p_std': p_sd,
#                 'classification_accuracy_p_combined': p_comb,
#                 'acc_sig_frac': acc_sig_frac,

#                 'VAF_mean': vaf_mean,
#                 'VAF_std': vaf_sd,
#             }

#             if is_pair:
#                 cat1, cat2 = key
#                 base['cat1'] = _as_label(cat1, label_map)
#                 base['cat2'] = _as_label(cat2, label_map)
#             else:
#                 base['target'] = _as_label(key, label_map)

#             rows.append(base)

#         df = pd.DataFrame(rows)
#         if not len(df):
#             return df

#         # nice order
#         if 'cat1' in df.columns:
#             first = ['cat1','cat2','n_sessions','n_valid','n_error','coverage']
#         else:
#             first = ['target','n_sessions','n_valid','n_error','coverage']
#         cols = first + [c for c in df.columns if c not in first]
#         return df[cols].sort_values(by='coverage', ascending=False).reset_index(drop=True)

#     df_on   = _aggregate_bucket(bucket_on,  is_pair=True)
#     df_off  = _aggregate_bucket(bucket_off, is_pair=True)
#     df_same = _aggregate_bucket(bucket_same, is_pair=False)

#     # ---------- Paired Δ(ON−OFF) per pair (by session) ----------
#     delta_rows = []
#     # build union of observed pairs
#     all_pairs = set(bucket_on.keys()) | set(bucket_off.keys())
#     for pair in sorted(all_pairs):
#         per_session_diffs = []
#         for s_on, s_off in zip(per_sess_on, per_sess_off):
#             if pair in s_on and pair in s_off:
#                 ron, roff = s_on[pair], s_off[pair]
#                 per_session_diffs.append({
#                     'procrustes_disparity': ron.get('procrustes_disparity') - roff.get('procrustes_disparity'),
#                     'classification_accuracy': ron.get('classification_accuracy') - roff.get('classification_accuracy'),
#                     'cca_correlation': ron.get('cca_correlation') - roff.get('cca_correlation'),
#                     'VAF': ron.get('VAF') - roff.get('VAF'),
#                     'principal_angle_1': ron.get('principal_angle_1') - roff.get('principal_angle_1'),
#                     'mean_principal_angle': ron.get('mean_principal_angle') - roff.get('mean_principal_angle'),
#                 })
#         if not per_session_diffs:
#             continue

#         def agg(name):
#             vals = [d[name] for d in per_session_diffs]
#             return _np_mean_sd(vals)

#         cat1, cat2 = pair
#         row = {
#             'cat1': _as_label(cat1, label_map),
#             'cat2': _as_label(cat2, label_map),
#         }
#         for m in ['procrustes_disparity','classification_accuracy','cca_correlation',
#                   'VAF','principal_angle_1','mean_principal_angle']:
#             mu, sd = agg(m)
#             row[f'delta_{m}_mean'] = mu
#             row[f'delta_{m}_std']  = sd

#         row['n_sessions'] = n_sessions
#         row['n_valid'] = len(per_session_diffs)
#         row['n_error'] = n_sessions - len(per_session_diffs)
#         row['coverage'] = len(per_session_diffs) / n_sessions if n_sessions else np.nan
#         delta_rows.append(row)

#     df_delta = pd.DataFrame(delta_rows)
#     if df_delta is None or df_delta.empty:
#         df_delta = pd.DataFrame(columns=[
#             'cat1','cat2','n_sessions','n_valid','n_error','coverage',
#             'delta_procrustes_disparity_mean','delta_procrustes_disparity_std',
#             'delta_classification_accuracy_mean','delta_classification_accuracy_std',
#             'delta_cca_correlation_mean','delta_cca_correlation_std'
#         ])
#     else:
#         for m in ['procrustes_disparity','classification_accuracy','cca_correlation']:
#             mean_col = f'delta_{m}_mean'
#             bare_col = f'delta_{m}'
#             if mean_col in df_delta.columns and bare_col not in df_delta.columns:
#                 df_delta[bare_col] = df_delta[mean_col]

#     return {
#         'ai_on': df_on,
#         'ai_off': df_off,
#         'ai_on_vs_off_same_target': df_same,
#         'delta_on_minus_off': df_delta
#     }

# def aggregate_all_sessions(
#     all_results,
#     target_map=None,
#     alpha=0.05,
#     fisher_for_cca=True,
#     keep_raw_angles=True,
# ):
#     """
#     all_results: list of per-session dicts with keys:
#        'target_vs_target_ai_on', 'target_vs_target_ai_off', 'ai_on_vs_off_same_target'
#        where each entry is a dict with metrics or {'error': ...}

#     Returns dict with DataFrames:
#       - ai_on
#       - ai_off
#       - ai_on_vs_off_same_target
#       - delta_on_minus_off  (paired per session)
#     """
#     import numpy as np
#     import pandas as pd
#     from collections import defaultdict

#     n_sessions = len(all_results)
#     label_map = {k: v for k, v in (target_map or {}).items()}

#     # minimal helper: vectorize with aliases
#     def _vec(recs, *keys):
#         vals = []
#         for r in recs:
#             v = np.nan
#             for k in keys:
#                 if k in r and r[k] is not None:
#                     v = r[k]
#                     break
#             vals.append(v)
#         return vals

#     # Buckets for aggregation
#     bucket_on   = defaultdict(list)   # keyed by pair
#     bucket_off  = defaultdict(list)
#     bucket_same = defaultdict(list)   # keyed by single target

#     # Also keep per-session lookups for Δ (ON−OFF) pairing
#     per_sess_on  = []
#     per_sess_off = []

#     for sess in all_results:
#         sess_on  = {}
#         sess_off = {}

#         # ON pairs
#         for rec in sess.get('target_vs_target_ai_on', []):
#             if 'error' in rec: 
#                 continue
#             key = _canon_pair(rec['cat1'], rec['cat2'])
#             bucket_on[key].append(rec)
#             sess_on[key] = rec

#         # OFF pairs
#         for rec in sess.get('target_vs_target_ai_off', []):
#             if 'error' in rec: 
#                 continue
#             key = _canon_pair(rec['cat1'], rec['cat2'])
#             bucket_off[key].append(rec)
#             sess_off[key] = rec

#         # SAME target (ON vs OFF) — tolerate target or cat1/cat2
#         for rec in sess.get('ai_on_vs_off_same_target', []):
#             if 'error' in rec: 
#                 continue
#             tgt = rec.get('cat1', rec.get('target', rec.get('cat2')))
#             key = tuple(tgt) if isinstance(tgt, (list, tuple, np.ndarray)) else tgt
#             bucket_same[key].append(rec)

#         per_sess_on.append(sess_on)
#         per_sess_off.append(sess_off)

#     def _aggregate_bucket(bucket, is_pair=True):
#         rows = []
#         for key, recs in bucket.items():
#             # ---- existing metrics (now with aliases for angles) ----
#             th1_mean, th1_sd = _np_mean_sd(_vec(recs, 'principal_angle_1', 'angle_1_deg'))
#             thm_mean, thm_sd = _np_mean_sd(_vec(recs, 'mean_principal_angle', 'mean_angle_deg'))

#             proc_mean, proc_sd = _np_mean_sd(_vec(recs, 'procrustes_disparity'))
#             vaf_mean,  vaf_sd  = _np_mean_sd(_vec(recs, 'VAF'))

#             cca_raw = np.asarray(_vec(recs, 'cca_correlation'), dtype=float)
#             cca_mean_raw, cca_sd = _np_mean_sd(cca_raw)
#             cca_mean = _fisher_z_mean(cca_raw) if fisher_for_cca else cca_mean_raw

#             acc_mean, acc_sd = _np_mean_sd(_vec(recs, 'classification_accuracy'))
#             p_mean,   p_sd   = _np_mean_sd(_vec(recs, 'classification_accuracy_p'))
#             p_comb = _combine_p_stouffer(_vec(recs, 'classification_accuracy_p'))
#             p_arr  = np.asarray(_vec(recs, 'classification_accuracy_p'), dtype=float)
#             acc_sig_frac = float(np.mean(p_arr < alpha)) if p_arr.size else np.nan

#             # ---- NEW: add your extra subspace/overlap metrics ----
#             grass_mean, grass_sd = _np_mean_sd(_vec(recs, 'grassmann'))
#             axy_mean, axy_sd     = _np_mean_sd(_vec(recs, 'align_X_in_Y'))
#             ayx_mean, ayx_sd     = _np_mean_sd(_vec(recs, 'align_Y_in_X'))
#             asym_mean, asym_sd   = _np_mean_sd(_vec(recs, 'align_sym'))
#             outx_mean, outx_sd   = _np_mean_sd(_vec(recs, 'outside_X_wrt_Y'))
#             outy_mean, outy_sd   = _np_mean_sd(_vec(recs, 'outside_Y_wrt_X'))

#             # ---- NEW: decoder-plane metrics (may be NaN throughout) ----
#             rank_mean, rank_sd = _np_mean_sd(_vec(recs, 'decoder_rank'))
#             o1m, o1s = _np_mean_sd(_vec(recs, 'out_of_plane_frac_cat1_mean'))
#             i1m, i1s = _np_mean_sd(_vec(recs, 'in_plane_frac_cat1_mean'))
#             o2m, o2s = _np_mean_sd(_vec(recs, 'out_of_plane_frac_cat2_mean'))
#             i2m, i2s = _np_mean_sd(_vec(recs, 'in_plane_frac_cat2_mean'))
#             doop_m, doop_s = _np_mean_sd(_vec(recs, 'delta_out_of_plane_frac'))

#             base = {
#                 'n_sessions': n_sessions,
#                 'n_valid': len(recs),
#                 'n_error': n_sessions - len(recs),
#                 'coverage': (len(recs) / n_sessions) if n_sessions else np.nan,

#                 # angles (kept as raw degrees)
#                 'principal_angle_1_mean': th1_mean,
#                 'principal_angle_1_std': th1_sd,
#                 'mean_principal_angle_mean': thm_mean,
#                 'mean_principal_angle_std': thm_sd,

#                 # geometry
#                 'procrustes_disparity_mean': proc_mean,
#                 'procrustes_disparity_std': proc_sd,

#                 # shared dynamics
#                 'cca_correlation_mean': cca_mean,
#                 'cca_correlation_std': cca_sd,
#                 'cca_correlation_mean_raw': cca_mean_raw,

#                 # separability
#                 'classification_accuracy_mean': acc_mean,
#                 'classification_accuracy_std': acc_sd,
#                 'classification_accuracy_p_mean': p_mean,
#                 'classification_accuracy_p_std': p_sd,
#                 'classification_accuracy_p_combined': p_comb,
#                 'acc_sig_frac': acc_sig_frac,

#                 # variance explained
#                 'VAF_mean': vaf_mean,
#                 'VAF_std':  vaf_sd,

#                 # NEW: alignment / grassmann / outside
#                 'grassmann_mean': grass_mean,
#                 'grassmann_std':  grass_sd,
#                 'align_X_in_Y_mean': axy_mean,
#                 'align_X_in_Y_std':  axy_sd,
#                 'align_Y_in_X_mean': ayx_mean,
#                 'align_Y_in_X_std':  ayx_sd,
#                 'align_sym_mean': asym_mean,
#                 'align_sym_std':  asym_sd,
#                 'outside_X_wrt_Y_mean': outx_mean,
#                 'outside_X_wrt_Y_std':  outx_sd,
#                 'outside_Y_wrt_X_mean': outy_mean,
#                 'outside_Y_wrt_X_std':  outy_sd,

#                 # NEW: decoder-plane
#                 'decoder_rank_mean': rank_mean,
#                 'decoder_rank_std':  rank_sd,
#                 'out_of_plane_frac_cat1_mean_mean': o1m,
#                 'out_of_plane_frac_cat1_mean_std':  o1s,
#                 'in_plane_frac_cat1_mean_mean':     i1m,
#                 'in_plane_frac_cat1_mean_std':      i1s,
#                 'out_of_plane_frac_cat2_mean_mean': o2m,
#                 'out_of_plane_frac_cat2_mean_std':  o2s,
#                 'in_plane_frac_cat2_mean_mean':     i2m,
#                 'in_plane_frac_cat2_mean_std':      i2s,
#                 'delta_out_of_plane_frac_mean':     doop_m,
#                 'delta_out_of_plane_frac_std':      doop_s,
#             }

#             if is_pair:
#                 cat1, cat2 = key
#                 base['cat1'] = _as_label(cat1, label_map)
#                 base['cat2'] = _as_label(cat2, label_map)
#             else:
#                 base['target'] = _as_label(key, label_map)

#             rows.append(base)

#         df = pd.DataFrame(rows)
#         if not len(df):
#             return df

#         # nice order
#         if 'cat1' in df.columns:
#             first = ['cat1','cat2','n_sessions','n_valid','n_error','coverage']
#         else:
#             first = ['target','n_sessions','n_valid','n_error','coverage']
#         cols = first + [c for c in df.columns if c not in first]
#         return df[cols].sort_values(by='coverage', ascending=False).reset_index(drop=True)

#     df_on   = _aggregate_bucket(bucket_on,  is_pair=True)
#     df_off  = _aggregate_bucket(bucket_off, is_pair=True)
#     df_same = _aggregate_bucket(bucket_same, is_pair=False)

#     # ---------- Paired Δ(ON−OFF) per pair (by session) ----------
#     delta_rows = []
#     all_pairs = set(bucket_on.keys()) | set(bucket_off.keys())
#     for pair in sorted(all_pairs):
#         per_session_diffs = []
#         for s_on, s_off in zip(per_sess_on, per_sess_off):
#             if pair in s_on and pair in s_off:
#                 ron, roff = s_on[pair], s_off[pair]
#                 per_session_diffs.append({
#                     # keep original metric names; no new deltas needed here
#                     'procrustes_disparity': ron.get('procrustes_disparity') - roff.get('procrustes_disparity'),
#                     'classification_accuracy': ron.get('classification_accuracy') - roff.get('classification_accuracy'),
#                     'cca_correlation': ron.get('cca_correlation') - roff.get('cca_correlation'),
#                     'VAF': ron.get('VAF') - roff.get('VAF'),
#                     'principal_angle_1': (ron.get('principal_angle_1', ron.get('angle_1_deg')) 
#                                           - roff.get('principal_angle_1', roff.get('angle_1_deg'))),
#                     'mean_principal_angle': (ron.get('mean_principal_angle', ron.get('mean_angle_deg')) 
#                                              - roff.get('mean_principal_angle', roff.get('mean_angle_deg'))),
#                 })
#         if not per_session_diffs:
#             continue

#         def agg(name):
#             vals = [d[name] for d in per_session_diffs]
#             return _np_mean_sd(vals)

#         cat1, cat2 = pair
#         row = {
#             'cat1': _as_label(cat1, label_map),
#             'cat2': _as_label(cat2, label_map),
#         }
#         for m in ['procrustes_disparity','classification_accuracy','cca_correlation',
#                   'VAF','principal_angle_1','mean_principal_angle']:
#             mu, sd = agg(m)
#             row[f'delta_{m}_mean'] = mu
#             row[f'delta_{m}_std']  = sd

#         row['n_sessions'] = n_sessions
#         row['n_valid'] = len(per_session_diffs)
#         row['n_error'] = n_sessions - len(per_session_diffs)
#         row['coverage'] = len(per_session_diffs) / n_sessions if n_sessions else np.nan
#         delta_rows.append(row)

#     import pandas as pd
#     df_delta = pd.DataFrame(delta_rows)
#     if df_delta is None or df_delta.empty:
#         df_delta = pd.DataFrame(columns=[
#             'cat1','cat2','n_sessions','n_valid','n_error','coverage',
#             'delta_procrustes_disparity_mean','delta_procrustes_disparity_std',
#             'delta_classification_accuracy_mean','delta_classification_accuracy_std',
#             'delta_cca_correlation_mean','delta_cca_correlation_std',
#             'delta_VAF_mean','delta_VAF_std',
#             'delta_principal_angle_1_mean','delta_principal_angle_1_std',
#             'delta_mean_principal_angle_mean','delta_mean_principal_angle_std',
#         ])
#     else:
#         for m in ['procrustes_disparity','classification_accuracy','cca_correlation']:
#             mean_col = f'delta_{m}_mean'
#             bare_col = f'delta_{m}'
#             if mean_col in df_delta.columns and bare_col not in df_delta.columns:
#                 df_delta[bare_col] = df_delta[mean_col]

#     return {
#         'ai_on': df_on,
#         'ai_off': df_off,
#         'ai_on_vs_off_same_target': df_same,
#         'delta_on_minus_off': df_delta
#     }

def aggregate_all_sessions(
    all_results,
    target_map=None,
    alpha=0.05,
    fisher_for_cca=True,
    keep_raw_angles=True,
):
    """
    all_results: list of per-session dicts with keys:
       'target_vs_target_ai_on', 'target_vs_target_ai_off', 'ai_on_vs_off_same_target'
       where each entry is a dict with metrics or {'error': ...}

    Returns dict with DataFrames:
      - ai_on
      - ai_off
      - ai_on_vs_off_same_target
      - delta_on_minus_off  (paired per session, for key metrics)
    """
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    n_sessions = len(all_results)
    label_map = {k: v for k, v in (target_map or {}).items()}

    # minimal helper: vectorize with aliases
    def _vec(recs, *keys):
        vals = []
        for r in recs:
            v = np.nan
            for k in keys:
                if k in r and r[k] is not None:
                    v = r[k]
                    break
            vals.append(v)
        return vals

    # Buckets for aggregation
    bucket_on   = defaultdict(list)   # keyed by pair
    bucket_off  = defaultdict(list)
    bucket_same = defaultdict(list)   # keyed by single target

    # Also keep per-session lookups for Δ (ON−OFF) pairing
    per_sess_on  = []
    per_sess_off = []

    for sess in all_results:
        sess_on  = {}
        sess_off = {}

        # ON pairs
        for rec in sess.get('target_vs_target_ai_on', []):
            if 'error' in rec:
                continue
            key = _canon_pair(rec['cat1'], rec['cat2'])
            bucket_on[key].append(rec)
            sess_on[key] = rec

        # OFF pairs
        for rec in sess.get('target_vs_target_ai_off', []):
            if 'error' in rec:
                continue
            key = _canon_pair(rec['cat1'], rec['cat2'])
            bucket_off[key].append(rec)
            sess_off[key] = rec

        # SAME target (ON vs OFF)
        for rec in sess.get('ai_on_vs_off_same_target', []):
            if 'error' in rec:
                continue
            tgt = rec.get('cat1', rec.get('target', rec.get('cat2')))
            key = tuple(tgt) if isinstance(tgt, (list, tuple, np.ndarray)) else tgt
            bucket_same[key].append(rec)

        per_sess_on.append(sess_on)
        per_sess_off.append(sess_off)

    def _aggregate_bucket(bucket, is_pair=True):
        rows = []
        for key, recs in bucket.items():
            # ---- angles (with aliases) ----
            th1_mean, th1_sd = _np_mean_sd(_vec(recs, 'principal_angle_1', 'angle_1_deg'))
            thm_mean, thm_sd = _np_mean_sd(_vec(recs, 'mean_principal_angle', 'mean_angle_deg'))

            # ---- geometry ----
            proc_mean, proc_sd = _np_mean_sd(_vec(recs, 'procrustes_disparity'))
            md_mean,  md_sd    = _np_mean_sd(_vec(recs, 'mean_distance'))
            auc_mean, auc_sd   = _np_mean_sd(_vec(recs, 'distance_auc'))

            # ---- shared dynamics (CCA) ----
            cca_raw = np.asarray(_vec(recs, 'cca_correlation'), dtype=float)
            cca_mean_raw, cca_sd = _np_mean_sd(cca_raw)
            cca_mean = _fisher_z_mean(cca_raw) if fisher_for_cca else cca_mean_raw

            # ---- separability (classifier) ----
            acc_mean, acc_sd = _np_mean_sd(_vec(recs, 'classification_accuracy'))
            p_mean,   p_sd   = _np_mean_sd(_vec(recs, 'classification_accuracy_p'))
            p_comb = _combine_p_stouffer(_vec(recs, 'classification_accuracy_p'))
            p_arr  = np.asarray(_vec(recs, 'classification_accuracy_p'), dtype=float)
            acc_sig_frac = float(np.mean(p_arr < alpha)) if p_arr.size else np.nan

            # ---- subspace overlap / outside-energy ----
            axy_mean, axy_sd   = _np_mean_sd(_vec(recs, 'align_X_in_Y'))
            ayx_mean, ayx_sd   = _np_mean_sd(_vec(recs, 'align_Y_in_X'))
            asym_mean, asym_sd = _np_mean_sd(_vec(recs, 'align_sym'))
            outx_mean, outx_sd = _np_mean_sd(_vec(recs, 'outside_X_wrt_Y'))
            outy_mean, outy_sd = _np_mean_sd(_vec(recs, 'outside_Y_wrt_X'))

            base = {
                'n_sessions': n_sessions,
                'n_valid': len(recs),
                'n_error': n_sessions - len(recs),
                'coverage': (len(recs) / n_sessions) if n_sessions else np.nan,

                # angles (degrees)
                'principal_angle_1_mean': th1_mean,
                'principal_angle_1_std': th1_sd,
                'mean_principal_angle_mean': thm_mean,
                'mean_principal_angle_std': thm_sd,

                # geometry
                'procrustes_disparity_mean': proc_mean,
                'procrustes_disparity_std': proc_sd,
                'mean_distance_mean': md_mean,
                'mean_distance_std': md_sd,
                'distance_auc_mean': auc_mean,
                'distance_auc_std': auc_sd,

                # shared dynamics
                'cca_correlation_mean': cca_mean,
                'cca_correlation_std': cca_sd,
                'cca_correlation_mean_raw': cca_mean_raw,

                # separability
                'classification_accuracy_mean': acc_mean,
                'classification_accuracy_std': acc_sd,
                'classification_accuracy_p_mean': p_mean,
                'classification_accuracy_p_std': p_sd,
                'classification_accuracy_p_combined': p_comb,
                'acc_sig_frac': acc_sig_frac,

                # alignment / outside-mass
                'align_X_in_Y_mean': axy_mean,
                'align_X_in_Y_std':  axy_sd,
                'align_Y_in_X_mean': ayx_mean,
                'align_Y_in_X_std':  ayx_sd,
                'align_sym_mean': asym_mean,
                'align_sym_std':  asym_sd,
                'outside_X_wrt_Y_mean': outx_mean,
                'outside_X_wrt_Y_std':  outx_sd,
                'outside_Y_wrt_X_mean': outy_mean,
                'outside_Y_wrt_X_std':  outy_sd,
            }

            if is_pair:
                cat1, cat2 = key
                base['cat1'] = _as_label(cat1, label_map)
                base['cat2'] = _as_label(cat2, label_map)
            else:
                base['target'] = _as_label(key, label_map)

            rows.append(base)

        df = pd.DataFrame(rows)
        if not len(df):
            return df

        # nice order
        if 'cat1' in df.columns:
            first = ['cat1','cat2','n_sessions','n_valid','n_error','coverage']
        else:
            first = ['target','n_sessions','n_valid','n_error','coverage']
        cols = first + [c for c in df.columns if c not in first]
        return df[cols].sort_values(by='coverage', ascending=False).reset_index(drop=True)

    df_on   = _aggregate_bucket(bucket_on,  is_pair=True)
    df_off  = _aggregate_bucket(bucket_off, is_pair=True)
    df_same = _aggregate_bucket(bucket_same, is_pair=False)

    # ---------- Paired Δ(ON−OFF) per pair (by session) ----------
    delta_rows = []
    all_pairs = set(bucket_on.keys()) | set(bucket_off.keys())
    for pair in sorted(all_pairs):
        per_session_diffs = []
        for s_on, s_off in zip(per_sess_on, per_sess_off):
            if pair in s_on and pair in s_off:
                ron, roff = s_on[pair], s_off[pair]
                per_session_diffs.append({
                    'procrustes_disparity': ron.get('procrustes_disparity') - roff.get('procrustes_disparity'),
                    'classification_accuracy': ron.get('classification_accuracy') - roff.get('classification_accuracy'),
                    'cca_correlation': ron.get('cca_correlation') - roff.get('cca_correlation'),
                    'mean_distance': ron.get('mean_distance') - roff.get('mean_distance'),
                    'principal_angle_1': (
                        ron.get('principal_angle_1', ron.get('angle_1_deg'))
                        - roff.get('principal_angle_1', roff.get('angle_1_deg'))
                    ),
                    'mean_principal_angle': (
                        ron.get('mean_principal_angle', ron.get('mean_angle_deg'))
                        - roff.get('mean_principal_angle', roff.get('mean_angle_deg'))
                    ),
                })
        if not per_session_diffs:
            continue

        def agg(name):
            vals = [d[name] for d in per_session_diffs]
            return _np_mean_sd(vals)

        cat1, cat2 = pair
        row = {
            'cat1': _as_label(cat1, label_map),
            'cat2': _as_label(cat2, label_map),
        }
        for m in [
            'procrustes_disparity',
            'classification_accuracy',
            'cca_correlation',
            'mean_distance',
            'principal_angle_1',
            'mean_principal_angle',
        ]:
            mu, sd = agg(m)
            row[f'delta_{m}_mean'] = mu
            row[f'delta_{m}_std']  = sd

        row['n_sessions'] = n_sessions
        row['n_valid'] = len(per_session_diffs)
        row['n_error'] = n_sessions - len(per_session_diffs)
        row['coverage'] = len(per_session_diffs) / n_sessions if n_sessions else np.nan
        delta_rows.append(row)

    import pandas as pd
    df_delta = pd.DataFrame(delta_rows)
    if df_delta is None or df_delta.empty:
        df_delta = pd.DataFrame(columns=[
            'cat1','cat2','n_sessions','n_valid','n_error','coverage',
            'delta_procrustes_disparity_mean','delta_procrustes_disparity_std',
            'delta_classification_accuracy_mean','delta_classification_accuracy_std',
            'delta_cca_correlation_mean','delta_cca_correlation_std',
            'delta_mean_distance_mean','delta_mean_distance_std',
            'delta_principal_angle_1_mean','delta_principal_angle_1_std',
            'delta_mean_principal_angle_mean','delta_mean_principal_angle_std',
        ])
    else:
        # convenience "bare" columns
        for m in ['procrustes_disparity','classification_accuracy','cca_correlation','mean_distance']:
            mean_col = f'delta_{m}_mean'
            bare_col = f'delta_{m}'
            if mean_col in df_delta.columns and bare_col not in df_delta.columns:
                df_delta[bare_col] = df_delta[mean_col]

    return {
        'ai_on': df_on,
        'ai_off': df_off,
        'ai_on_vs_off_same_target': df_same,
        'delta_on_minus_off': df_delta,
    }
