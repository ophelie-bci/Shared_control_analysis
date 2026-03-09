import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.interpolate import interp1d
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


# --------------------------------------------------------------------
# Target mapping helpers
# --------------------------------------------------------------------

def _get_target_map(task):
    """Return mapping from 3D target position to categorical label."""
    fixed_targets = {
        (0.,   0.75, 9.2): "straight",
        (7.,   0.75, 6.):  "right",
        (-7.,  0.75, 6.):  "left",
        (-3.5, 0.75, 8.5): "slight_left",
        (3.5,  0.75, 8.5): "slight_right",
    }

    moving_targets = {
        (0.,  1., 9.2): "straight",
        (6.,  1., 7.):  "right",
        (-6., 1., 7.):  "left",
        (-3., 1., 8.7): "slight_left",
        (3.,  1., 8.7): "slight_right",
    }

    if task == "fixedCamera":
        return fixed_targets
    else:
        return moving_targets


# --------------------------------------------------------------------
# Basic interpolation of latent trajectories
# --------------------------------------------------------------------

def interpolate_trials_normalized(trials, target_len=80):
    """
    Interpolate a list of latent trajectories to a fixed length in normalized time.
    
    Parameters
    ----------
    trials : list of (T_i, D) arrays
        One array per trial (time x latent_dim).
    target_len : int
        Number of timepoints after resampling.

    Returns
    -------
    arr : (n_trials, target_len, D) float array
    """
    interpolated = []
    for trial in trials:
        trial = np.asarray(trial, float)
        if trial.ndim != 2:
            raise ValueError(f"Expected trial shape (T, D), got {trial.shape}")
        T, D = trial.shape
        if T < 2:
            # cannot interpolate a single point; skip
            continue

        original_time = np.linspace(0, 1, T)
        target_time = np.linspace(0, 1, target_len)

        interp_trial = np.zeros((target_len, D), dtype=float)
        for d in range(D):
            f = interp1d(original_time, trial[:, d], kind="linear")
            interp_trial[:, d] = f(target_time)
        interpolated.append(interp_trial)

    if len(interpolated) == 0:
        return np.empty((0, target_len, 0))

    return np.stack(interpolated, axis=0)


# --------------------------------------------------------------------
# Simple summary plots for interpolated latents
# --------------------------------------------------------------------

def plot_interpolated_latents_summary(interpolated_latents, title_prefix=""):
    """
    Make simple summary plots:
      1) Per latent dim, overlay all targets (AI ON)
      2) Per latent dim, overlay all targets (AI OFF)
      3) Grid: per target, AI ON vs AI OFF (averages)
    
    interpolated_latents: dict
        {category: {"ai_on": (n_on, T, D), "ai_off": (n_off, T, D)}}
    """
    categories = list(interpolated_latents.keys())
    if not categories:
        print("No categories to plot.")
        return

    # infer latent dimensionality
    D = 6
    for cat in categories:
        for status in ["ai_on", "ai_off"]:
            arr = np.asarray(interpolated_latents[cat][status])
            if arr.size > 0:
                D = arr.shape[-1]
                break

    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

    # ---- 1. Average trajectories for AI ON ----
    fig1, axes1 = plt.subplots(D, 1, figsize=(10, 2.0 * D), sharex=True)
    if D == 1:
        axes1 = [axes1]

    for dim in range(D):
        ax = axes1[dim]
        for idx, category in enumerate(categories):
            trials = np.asarray(interpolated_latents[category]["ai_on"])
            if trials.size == 0:
                continue
            avg = trials.mean(axis=0)  # (T, D)
            ax.plot(avg[:, dim], label=category, color=colors[idx])
        ax.set_ylabel(f"L{dim+1}")
        if dim == 0:
            ax.legend()
    axes1[-1].set_xlabel("Normalized time (bins)")
    fig1.suptitle(f"{title_prefix}Average latent trajectories per target (AI ON)", y=0.99)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # ---- 2. Average trajectories for AI OFF ----
    fig2, axes2 = plt.subplots(D, 1, figsize=(10, 2.0 * D), sharex=True)
    if D == 1:
        axes2 = [axes2]

    for dim in range(D):
        ax = axes2[dim]
        for idx, category in enumerate(categories):
            trials = np.asarray(interpolated_latents[category]["ai_off"])
            if trials.size == 0:
                continue
            avg = trials.mean(axis=0)
            ax.plot(avg[:, dim], label=category, color=colors[idx])
        ax.set_ylabel(f"L{dim+1}")
        if dim == 0:
            ax.legend()
    axes2[-1].set_xlabel("Normalized time (bins)")
    fig2.suptitle(f"{title_prefix}Average latent trajectories per target (AI OFF)", y=0.99)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # ---- 3. AI ON vs OFF per target ----
    n_cat = len(categories)
    fig3, axes3 = plt.subplots(D, n_cat, figsize=(3.5 * n_cat, 2.0 * D), sharex=True)
    if D == 1:
        axes3 = np.expand_dims(axes3, axis=0)
    if n_cat == 1:
        axes3 = np.expand_dims(axes3, axis=1)

    for col, category in enumerate(categories):
        on_trials = np.asarray(interpolated_latents[category]["ai_on"])
        off_trials = np.asarray(interpolated_latents[category]["ai_off"])
        if on_trials.size == 0 or off_trials.size == 0:
            continue
        avg_on = on_trials.mean(axis=0)
        avg_off = off_trials.mean(axis=0)
        for dim in range(D):
            ax = axes3[dim, col]
            ax.plot(avg_on[:, dim], label="AI ON", color="tab:blue")
            ax.plot(avg_off[:, dim], label="AI OFF", color="tab:red", linestyle="--")
            if dim == 0:
                ax.set_title(category)
            if col == 0:
                ax.set_ylabel(f"L{dim+1}")
            if dim == D - 1:
                ax.set_xlabel("Normalized time (bins)")

    handles, labels = axes3[0, 0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="upper center", ncol=2)
    fig3.suptitle(f"{title_prefix}AI ON vs OFF per target (averaged latents)", y=0.98)
    fig3.tight_layout(rect=[0, 0, 1, 0.94])


# --------------------------------------------------------------------
# Pairwise metrics between trajectories / manifolds
# --------------------------------------------------------------------

def _compute_pairwise_metrics(cat1, cat2, interpolated_latents, ai_status):
    """
    Compare two target manifolds (cat1 vs cat2) for a given AI status.

    Returns dict with:
      - principal_angle_1
      - mean_principal_angle
      - procrustes_disparity
      - cca_correlation
      - VAF (percent)
    """
    trials1 = np.asarray(interpolated_latents[cat1][ai_status])
    trials2 = np.asarray(interpolated_latents[cat2][ai_status])

    if trials1.shape[0] < 2 or trials2.shape[0] < 2:
        return {"cat1": cat1, "cat2": cat2, "error": "Insufficient trials"}

    try:
        # Average trajectories (T, D)
        mean1 = trials1.mean(axis=0)
        mean2 = trials2.mean(axis=0)

        # Principal angles between subspaces (time x dim → dim x time)
        angles = subspace_angles(mean1.T, mean2.T)
        angle_1 = float(np.degrees(angles[0]))
        mean_angle = float(np.degrees(angles).mean())

        # VAF: project mean2 onto basis of mean1
        U, S, Vt = np.linalg.svd(mean1, full_matrices=False)
        basis = Vt.T  # D x D
        proj = mean2 @ basis @ basis.T
        total_var = float(np.sum(mean2 ** 2))
        recon_var = float(np.sum(proj ** 2))
        vaf_percent = 100.0 * recon_var / (total_var + 1e-12)

        # Procrustes disparity
        _, _, disparity = procrustes(mean1, mean2)

        # CCA between all sampled points from the two manifolds
        X1 = trials1.reshape(-1, trials1.shape[-1])
        X2 = trials2.reshape(-1, trials2.shape[-1])
        min_len = min(X1.shape[0], X2.shape[0])
        X1 = X1[:min_len]
        X2 = X2[:min_len]

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        X1_std = scaler1.fit_transform(X1)
        X2_std = scaler2.fit_transform(X2)

        n_comp = min(3, X1_std.shape[1], X2_std.shape[1])
        cca = CCA(n_components=n_comp)
        Xc, Yc = cca.fit_transform(X1_std, X2_std)
        cca_corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(n_comp)]
        cca_corr = float(np.mean(cca_corrs))

        return {
            "cat1": cat1,
            "cat2": cat2,
            "principal_angle_1": angle_1,
            "mean_principal_angle": mean_angle,
            "procrustes_disparity": float(disparity),
            "cca_correlation": cca_corr,
            "VAF": float(vaf_percent),
        }
    except Exception as e:
        return {"cat1": cat1, "cat2": cat2, "error": str(e)}


# --------------------------------------------------------------------
# MAIN FUNCTION 1:
# Per target, AI ON vs AI OFF, plus metrics  (this returns `results`)
# --------------------------------------------------------------------

def plot_latent_trajectories_average_target_ai_on(
    latents, targets, ai, task, target_len=80
):
    """
    Group trials by target & AI condition, interpolate, plot, and compute metrics.

    Parameters
    ----------
    latents : list of (T_i, 6) arrays
    targets : list of (3,) arrays
    ai      : list/array of ints (1=AI ON, 0/else=AI OFF)
    task    : "fixedCamera" or "movingCamera"
    target_len : int
        number of samples after time-normalization

    Returns
    -------
    results : dict
        {
          "task": task,
          "per_category": {
             category: {principal_angle_1, mean_principal_angle,
                        procrustes_disparity, cca_correlation, VAF}
          },
          "target_vs_target_ai_on": [pairwise comparisons],
          "target_vs_target_ai_off": [pairwise comparisons],
        }
    """
    target_map = _get_target_map(task)

    # 1) collect trials by category and AI status
    trial_categories = {cat: {"ai_on": [], "ai_off": []}
                        for cat in target_map.values()}

    for trial_lat, target_pos, trial_ai in zip(latents, targets, ai):
        target_tuple = tuple(np.asarray(target_pos, float))
        if target_tuple not in target_map:
            continue
        category = target_map[target_tuple]
        if trial_ai == 1:
            trial_categories[category]["ai_on"].append(np.asarray(trial_lat, float))
        else:
            trial_categories[category]["ai_off"].append(np.asarray(trial_lat, float))

    # 2) interpolate each category/condition
    interpolated_latents = {
        cat: {"ai_on": [], "ai_off": []} for cat in trial_categories.keys()
    }
    for category, status_dict in trial_categories.items():
        for status in ["ai_on", "ai_off"]:
            trials = status_dict[status]
            if len(trials) == 0:
                interpolated_latents[category][status] = np.empty((0, target_len, 0))
                continue
            interpolated_latents[category][status] = interpolate_trials_normalized(
                trials, target_len=target_len
            )

    # 3) plots
    plot_interpolated_latents_summary(interpolated_latents, title_prefix="")

    # 4) metrics: AI ON vs AI OFF within each target
    results = {
        "task": task,
        "per_category": {},
        "target_vs_target_ai_on": [],
        "target_vs_target_ai_off": [],
    }

    for category in trial_categories.keys():
        on_arr = interpolated_latents[category]["ai_on"]
        off_arr = interpolated_latents[category]["ai_off"]
        if on_arr.shape[0] < 2 or off_arr.shape[0] < 2:
            continue

        mean_on = on_arr.mean(axis=0)   # (T, D)
        mean_off = off_arr.mean(axis=0)

        # principal angles
        angles = subspace_angles(mean_on.T, mean_off.T)
        angle_1 = float(np.degrees(angles[0]))
        mean_angle = float(np.degrees(angles).mean())

        # VAF: project OFF onto ON basis
        U, S, Vt = np.linalg.svd(mean_on, full_matrices=False)
        basis = Vt.T
        proj = mean_off @ basis @ basis.T
        total_var = float(np.sum(mean_off ** 2))
        recon_var = float(np.sum(proj ** 2))
        vaf_percent = 100.0 * recon_var / (total_var + 1e-12)

        # Procrustes disparity
        _, _, disparity = procrustes(mean_on, mean_off)

        # CCA
        X_on = on_arr.reshape(-1, on_arr.shape[-1])
        X_off = off_arr.reshape(-1, off_arr.shape[-1])
        min_len = min(X_on.shape[0], X_off.shape[0])
        X_on = X_on[:min_len]
        X_off = X_off[:min_len]
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        X1_std = scaler1.fit_transform(X_on)
        X2_std = scaler2.fit_transform(X_off)

        n_comp = min(3, X1_std.shape[1], X2_std.shape[1])
        cca = CCA(n_components=n_comp)
        Xc, Yc = cca.fit_transform(X1_std, X2_std)
        cca_corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(n_comp)]
        cca_corr = float(np.mean(cca_corrs))

        results["per_category"][category] = {
            "principal_angle_1": angle_1,
            "mean_principal_angle": mean_angle,
            "procrustes_disparity": float(disparity),
            "cca_correlation": cca_corr,
            "VAF": float(vaf_percent),
        }

    # 5) pairwise comparisons between targets (within same AI status)
    cats = list(trial_categories.keys())
    for cat1, cat2 in combinations(cats, 2):
        res_on = _compute_pairwise_metrics(cat1, cat2, interpolated_latents, "ai_on")
        res_off = _compute_pairwise_metrics(cat1, cat2, interpolated_latents, "ai_off")
        results["target_vs_target_ai_on"].append(res_on)
        results["target_vs_target_ai_off"].append(res_off)

    return results


# --------------------------------------------------------------------
# MAIN FUNCTION 2:
# Average latent trajectories per target (ignoring AI)
# --------------------------------------------------------------------

def plot_latent_trajectories_average(latents, targets, task, target_len=80):
    """
    Ignore AI and just average latent trajectories per target category.
    Makes:
      - Heatmap per target (latent dim x time)
      - 3D trajectory (dims 1–3) per target
    """
    target_map = _get_target_map(task)

    # group trials by target
    trial_categories = {cat: [] for cat in target_map.values()}

    for trial_lat, target_pos in zip(latents, targets):
        target_tuple = tuple(np.asarray(target_pos, float))
        if target_tuple not in target_map:
            continue
        category = target_map[target_tuple]
        trial_categories[category].append(np.asarray(trial_lat, float))

    # interpolate to a single global length
    all_lengths = [
        len(trial) for trials in trial_categories.values() for trial in trials
        if len(trial) > 1
    ]
    if not all_lengths:
        print("No valid trials to plot.")
        return

    global_len = max(all_lengths)
    interpolated = {}
    for cat, trials in trial_categories.items():
        if len(trials) == 0:
            continue
        interpolated[cat] = interpolate_trials_normalized(trials, target_len=global_len)

    # heatmaps per target
    for cat, arr in interpolated.items():
        avg = arr.mean(axis=0)  # (T, D)
        avg = avg  # (T, D)

        plt.figure(figsize=(10, 4))
        sns.heatmap(
            avg.T,
            cmap="viridis",
            xticklabels=False,
            yticklabels=[f"L{i+1}" for i in range(avg.shape[1])],
        )
        plt.title(f"Average latent trajectory over time – {cat}")
        plt.xlabel("Normalized time")
        plt.ylabel("Latent dimension")
        plt.tight_layout()
        plt.show()

    # 3D trajectory (first 3 dims)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    color_map = {
        "straight": "g",
        "right": "r",
        "left": "b",
        "slight_right": "y",
        "slight_left": "m",
    }

    for cat, arr in interpolated.items():
        avg = arr.mean(axis=0)  # (T, D)
        color = color_map.get(cat, "k")
        ax.plot(avg[:, 0], avg[:, 1], avg[:, 2], color=color, label=cat)
        ax.scatter(avg[0, 0], avg[0, 1], avg[0, 2], color=color, marker="X", s=60)

    ax.set_xlabel("L1")
    ax.set_ylabel("L2")
    ax.set_zlabel("L3")
    ax.set_title("Average latent trajectories by target")
    ax.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# MAIN FUNCTION 3:
# Average AI ON vs AI OFF across all targets
# --------------------------------------------------------------------

def plot_average_latent_trajectories_aion_vs_aioff(
    latents, targets, ai, task, target_len=80
):
    """
    Collapse across targets; show single average trajectory for AI ON vs AI OFF.
    """
    target_map = _get_target_map(task)

    all_on = []
    all_off = []

    for trial_lat, target_pos, trial_ai in zip(latents, targets, ai):
        target_tuple = tuple(np.asarray(target_pos, float))
        if target_tuple not in target_map:
            continue
        if trial_ai == 1:
            all_on.append(np.asarray(trial_lat, float))
        else:
            all_off.append(np.asarray(trial_lat, float))

    if not all_on and not all_off:
        print("No trials found for selected task/targets.")
        return

    # choose common length
    lengths = [len(t) for t in (all_on + all_off) if len(t) > 1]
    if not lengths:
        print("Not enough samples to interpolate.")
        return
    common_len = max(lengths)

    on_arr = interpolate_trials_normalized(all_on, target_len=common_len) if all_on else None
    off_arr = interpolate_trials_normalized(all_off, target_len=common_len) if all_off else None

    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    if on_arr is not None and on_arr.size > 0:
        avg_on = on_arr.mean(axis=0)
        ax.plot(avg_on[:, 0], avg_on[:, 1], avg_on[:, 2],
                color="tab:green", label="AI ON")
    if off_arr is not None and off_arr.size > 0:
        avg_off = off_arr.mean(axis=0)
        ax.plot(avg_off[:, 0], avg_off[:, 1], avg_off[:, 2],
                color="tab:red", label="AI OFF")

    ax.set_xlabel("L1")
    ax.set_ylabel("L2")
    ax.set_zlabel("L3")
    ax.set_title("Average latent trajectories – AI ON vs AI OFF (all targets)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # per-latent time courses
    if on_arr is not None and on_arr.size > 0:
        D = on_arr.shape[-1]
    elif off_arr is not None and off_arr.size > 0:
        D = off_arr.shape[-1]
    else:
        return

    fig, axes = plt.subplots(D, 1, figsize=(10, 2.0 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        if on_arr is not None and on_arr.size > 0:
            avg_on = on_arr.mean(axis=0)
            ax.plot(avg_on[:, d], color="tab:green", label="AI ON")
        if off_arr is not None and off_arr.size > 0:
            avg_off = off_arr.mean(axis=0)
            ax.plot(avg_off[:, d], color="tab:red", linestyle="--", label="AI OFF")
        ax.set_ylabel(f"L{d+1}")
        if d == 0:
            ax.legend()

    axes[-1].set_xlabel("Normalized time (bins)")
    fig.suptitle("Average latent trajectories – AI ON vs AI OFF (per latent dim)",
                 y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])


# --------------------------------------------------------------------
# MAIN FUNCTION 4:
# Per-target 3D trajectories with AI ON vs AI OFF overlay
# --------------------------------------------------------------------

def plot_latent_trajectories_ai(latents, targets, ai, task, target_len=80):
    """
    For each target category, plot 3D latent trajectories (dims 1–3)
    overlaying AI ON vs AI OFF averages.
    """
    target_map = _get_target_map(task)

    # group by target & AI status
    trial_categories = {cat: {"ai_on": [], "ai_off": []}
                        for cat in target_map.values()}

    for trial_lat, target_pos, trial_ai in zip(latents, targets, ai):
        target_tuple = tuple(np.asarray(target_pos, float))
        if target_tuple not in target_map:
            continue
        category = target_map[target_tuple]
        if trial_ai == 1:
            trial_categories[category]["ai_on"].append(np.asarray(trial_lat, float))
        else:
            trial_categories[category]["ai_off"].append(np.asarray(trial_lat, float))

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    for category, status_dict in trial_categories.items():
        on_trials = status_dict["ai_on"]
        off_trials = status_dict["ai_off"]
        if len(on_trials) == 0 and len(off_trials) == 0:
            continue

        lengths = [len(t) for t in (on_trials + off_trials) if len(t) > 1]
        if not lengths:
            continue
        this_len = max(lengths)

        on_arr = interpolate_trials_normalized(on_trials, target_len=this_len) if on_trials else None
        off_arr = interpolate_trials_normalized(off_trials, target_len=this_len) if off_trials else None

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        if on_arr is not None and on_arr.size > 0:
            avg_on = on_arr.mean(axis=0)
            ax.plot(avg_on[:, 0], avg_on[:, 1], avg_on[:, 2],
                    color="tab:green", label="AI ON")
        if off_arr is not None and off_arr.size > 0:
            avg_off = off_arr.mean(axis=0)
            ax.plot(avg_off[:, 0], avg_off[:, 1], avg_off[:, 2],
                    color="tab:red", linestyle="--", label="AI OFF")

        ax.set_xlabel("L1")
        ax.set_ylabel("L2")
        ax.set_zlabel("L3")
        ax.set_title(f"Latent trajectories – {category} target")
        ax.legend()
        plt.tight_layout()
        plt.show()
