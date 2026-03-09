import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy import stats
from scipy.stats import linregress, levene, ttest_rel, spearmanr
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro

from src.constants import target_mapping

def gain_vs_baseline_across_sessions(success_rate_ai_on_all,
                                    success_rate_ai_off_all,
                                    method="pearson",
                                    n_boot=5000,
                                    seed=0,
                                    plot=False,
                                    title=None):
    """
    Session analogue of "gain vs baseline":

        Δ_i = ON_i - OFF_i
        correlate Δ_i with OFF_i

    Interprets whether AI helps more on "bad days" (low OFF baseline):
      - negative correlation: bigger gains when baseline is low (help more on bad days)
      - positive correlation: bigger gains when baseline is high
      - near zero: gains roughly baseline-independent

    Returns dict with r, p, n, CI, and arrays.
    """
    import numpy as np
    from scipy import stats

    on  = np.asarray(success_rate_ai_on_all, dtype=float)
    off = np.asarray(success_rate_ai_off_all, dtype=float)

    if on.shape != off.shape:
        raise ValueError(f"Length mismatch: AI ON has {len(on)} sessions, AI OFF has {len(off)} sessions.")

    mask = np.isfinite(on) & np.isfinite(off)
    on  = on[mask]
    off = off[mask]
    n = len(off)

    if n < 3:
        return {"r": np.nan, "p": np.nan, "n": n, "ci_lo": np.nan, "ci_hi": np.nan,
                "baseline_off": off, "gain": (on - off)}

    gain = on - off  # Δ_i

    method = method.lower()
    if method == "pearson":
        r, p = stats.pearsonr(off, gain)
        corr_fn = lambda a, b: stats.pearsonr(a, b)[0]
    elif method == "spearman":
        r, p = stats.spearmanr(off, gain)
        corr_fn = lambda a, b: stats.spearmanr(a, b)[0]
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    ci_lo = ci_hi = np.nan
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        rs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)  # resample sessions
            rs[i] = corr_fn(off[idx], gain[idx])
        ci_lo, ci_hi = np.nanpercentile(rs, [2.5, 97.5])

    out = {
        "r": float(r), "p": float(p), "n": int(n),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "baseline_off": off, "gain": gain
    }

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(off, gain)
        plt.axhline(0)  # zero-gain line
        plt.xlabel("Baseline (AI OFF accuracy per session)")
        plt.ylabel("AI Gain Δ = (AI ON − AI OFF) per session")
        t = title if title is not None else f"{method} r={out['r']:.3f}, p={out['p']:.3g}, n={out['n']}"
        if n_boot and n_boot > 0:
            t += f", 95% CI [{out['ci_lo']:.3f}, {out['ci_hi']:.3f}]"
        plt.title(t)
        plt.grid(True, alpha=0.3)

    return out

def on_off_relationship_per_session(success_rate_ai_on_all,
                                   success_rate_ai_off_all,
                                   n_boot=5000,
                                   seed=0,
                                   do_leave_one_out=True):
    """
    Analyze session-wise relationship between AI ON and AI OFF success rates.

    Returns:
      - Pearson + Spearman correlation
      - OLS regression: ON = intercept + slope*OFF (better than 'gain vs baseline' coupling)
      - Bootstrap 95% CIs for r, slope, intercept
      - Optional leave-one-out Pearson r to detect outlier-driven effects
    """
    import numpy as np
    from scipy import stats

    on = np.asarray(success_rate_ai_on_all, float)
    off = np.asarray(success_rate_ai_off_all, float)
    if on.shape != off.shape:
        raise ValueError(f"Length mismatch: ON={len(on)}, OFF={len(off)}")

    m = np.isfinite(on) & np.isfinite(off)
    on, off = on[m], off[m]
    n = len(on)
    if n < 3:
        return {"n": n}

    # Correlations
    r_p, p_p = stats.pearsonr(off, on)
    r_s, p_s = stats.spearmanr(off, on)

    # Regression (preferred)
    lr = stats.linregress(off, on)  # slope, intercept, rvalue, pvalue(slope), stderr
    slope, intercept = lr.slope, lr.intercept
    p_slope = lr.pvalue
    r_reg = lr.rvalue
    r2 = r_reg**2

    # Bootstrap CIs (resample sessions)
    rng = np.random.default_rng(seed)
    boot_rp = np.empty(n_boot, float)
    boot_slope = np.empty(n_boot, float)
    boot_intercept = np.empty(n_boot, float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        o, a = off[idx], on[idx]
        boot_rp[i] = stats.pearsonr(o, a)[0]
        lr_i = stats.linregress(o, a)
        boot_slope[i] = lr_i.slope
        boot_intercept[i] = lr_i.intercept

    ci_rp = tuple(np.percentile(boot_rp, [2.5, 97.5]))
    ci_slope = tuple(np.percentile(boot_slope, [2.5, 97.5]))
    ci_intercept = tuple(np.percentile(boot_intercept, [2.5, 97.5]))

    out = {
        "n": int(n),
        "pearson_r": float(r_p), "pearson_p": float(p_p), "pearson_ci": (float(ci_rp[0]), float(ci_rp[1])),
        "spearman_r": float(r_s), "spearman_p": float(p_s),
        "slope": float(slope), "intercept": float(intercept),
        "slope_p": float(p_slope),
        "slope_ci": (float(ci_slope[0]), float(ci_slope[1])),
        "intercept_ci": (float(ci_intercept[0]), float(ci_intercept[1])),
        "r2": float(r2),
    }

    if do_leave_one_out:
        loo = []
        for k in range(n):
            mask = np.ones(n, dtype=bool)
            mask[k] = False
            rr = stats.pearsonr(off[mask], on[mask])[0]
            loo.append(rr)
        out["loo_pearson_r"] = [float(x) for x in loo]
        out["loo_min_r"] = float(np.min(loo))
        out["loo_max_r"] = float(np.max(loo))

    return out

def corr_ai_on_vs_off_per_session(success_rate_ai_on_all,
                                 success_rate_ai_off_all,
                                 method="pearson",
                                 n_boot=5000,
                                 seed=0,
                                 plot=False,
                                 title=None):
    """
    Correlate AI ON vs AI OFF accuracy ACROSS SESSIONS (each session = one point).

    Parameters
    ----------
    success_rate_ai_on_all : list/array of float
        Overall AI ON accuracy per session.
    success_rate_ai_off_all : list/array of float
        Overall AI OFF accuracy per session.
    method : {"pearson","spearman"}
    n_boot : int
        Bootstrap samples for 95% CI on r. Set 0 to skip CI.
    seed : int
    plot : bool
        If True, scatter plot ON vs OFF with identity line.
    title : str or None

    Returns
    -------
    out : dict
        {"r","p","n","ci_lo","ci_hi","x","y"}
    """
    import numpy as np
    from scipy import stats

    x = np.asarray(success_rate_ai_on_all, dtype=float)
    y = np.asarray(success_rate_ai_off_all, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"Length mismatch: AI ON has {len(x)} sessions, AI OFF has {len(y)} sessions.")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 3:
        return {"r": np.nan, "p": np.nan, "n": n, "ci_lo": np.nan, "ci_hi": np.nan, "x": x, "y": y}

    method = method.lower()
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
        corr_fn = lambda a, b: stats.pearsonr(a, b)[0]
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
        corr_fn = lambda a, b: stats.spearmanr(a, b)[0]
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    ci_lo = ci_hi = np.nan
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        rs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)  # resample sessions with replacement
            rs[i] = corr_fn(x[idx], y[idx])
        ci_lo, ci_hi = np.nanpercentile(rs, [2.5, 97.5])

    out = {"r": float(r), "p": float(p), "n": int(n),
           "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
           "x": x, "y": y}

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(y, x)  # OFF on x-axis, ON on y-axis (common for "AI gain" thinking)
        mn = np.nanmin([x.min(), y.min()])
        mx = np.nanmax([x.max(), y.max()])
        plt.plot([mn, mx], [mn, mx])  # identity line
        plt.xlabel("AI OFF accuracy (per session)")
        plt.ylabel("AI ON accuracy (per session)")
        t = title if title is not None else f"{method} r={out['r']:.3f}, p={out['p']:.3g}, n={out['n']}"
        if n_boot and n_boot > 0:
            t += f", 95% CI [{out['ci_lo']:.3f}, {out['ci_hi']:.3f}]"
        plt.title(t)
        plt.grid(True, alpha=0.3)

    return out

def compute_success_rate_stats(success_rate_ai_on_all, success_rate_ai_off_all):
    """
    Compute summary statistics, variability, slope, and paired significance test
    between AI-on and AI-off session success rates.
    """

    # Convert to numpy arrays
    success_rate_ai_on_all = np.array(success_rate_ai_on_all)
    success_rate_ai_off_all = np.array(success_rate_ai_off_all)
    sessions = np.arange(len(success_rate_ai_on_all))

    # Check for normality in data
    differences = success_rate_ai_on_all - success_rate_ai_off_all
    # Shapiro-Wilk test
    stat, p = shapiro(differences)
    print(f"Shapiro-Wilk test: W = {stat:.4f}, p = {p:.4f}")
    if p > 0.05:
        print("✅ The differences appear to be normally distributed (p > 0.05).")
    else:
        print("❌ The differences are not normally distributed (p ≤ 0.05).")

    # Variability
    variability_on = success_rate_ai_on_all.std()
    variability_off = success_rate_ai_off_all.std()
    # Variability difference significance
    _, p_var = levene(success_rate_ai_on_all, success_rate_ai_off_all)

    # # Learning slope
    # reg_on = LinearRegression().fit(sessions.reshape(-1, 1), success_rate_ai_on_all)
    # reg_off = LinearRegression().fit(sessions.reshape(-1, 1), success_rate_ai_off_all)
    # slope_on = reg_on.coef_[0]
    # slope_off = reg_off.coef_[0]

    # # Learning slope significance (scipy linregress for p-value)
    # _, _, _, p_slope_on, _ = linregress(sessions, success_rate_ai_on_all)
    # _, _, _, p_slope_off, _ = linregress(sessions, success_rate_ai_off_all)

    slope_on, p_slope_on= spearmanr(sessions, success_rate_ai_on_all)
    slope_off,p_slope_off= spearmanr(sessions, success_rate_ai_off_all)


    # Paired test
    statistic, p_value = ttest_rel(success_rate_ai_on_all, success_rate_ai_off_all)
    print(f"t-test: statistic {statistic}, p_value {p_value}")
    return {
        "success_rate_ai_on": success_rate_ai_on_all,
        "success_rate_ai_off": success_rate_ai_off_all,
        "variability_on": variability_on,
        "variability_off": variability_off,
        "slope_on": slope_on,
        "slope_off": slope_off,
        "t_statistic": statistic,
        "p_value": p_value,
        "p_slope_on": p_slope_on,
        "p_slope_off": p_slope_off,
        "p_variability": p_var
    }


def calculate_accuracy_per_target(correct_trials_per_target: Dict, incorrect_trials_per_target: Dict) -> Tuple[Dict[str, float], int, float]:
    """
    Calculates:
    - accuracy per target,
    - total number of trials,
    - overall session accuracy.
    """
    accuracy_per_target = {}
    total_correct = 0
    total_incorrect = 0

    for target in correct_trials_per_target:
        correct = len(correct_trials_per_target[target])
        incorrect = len(incorrect_trials_per_target.get(target, []))
        total = correct + incorrect

        total_correct += correct
        total_incorrect += incorrect

        accuracy_per_target[target] = correct / total if total > 0 else 0

    total_trials = total_correct + total_incorrect
    total_accuracy = total_correct / total_trials if total_trials > 0 else 0

    return accuracy_per_target, total_trials, total_accuracy