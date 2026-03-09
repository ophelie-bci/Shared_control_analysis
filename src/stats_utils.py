from typing import Dict, Tuple
from scipy.stats import ttest_ind, ttest_rel, t
import numpy as np
from scipy import stats

from src.constants import CENTRAL_TARGETS, PERIPHERAL_TARGETS

def ci95_t(values):
        """95% CI for the mean (t-interval) over session-wise values."""
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n == 0:
            return np.nan, (np.nan, np.nan)
        m = x.mean()
        if n < 3:
            return m, (m, m)
        s = x.std(ddof=1)
        if not np.isfinite(s) or s == 0:
            return m, (m, m)
        se = s / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n-1)
        return m, (m - tcrit * se, m + tcrit * se)

def compute_ai_gain_by_target(ai_on_acc: Dict[str, float], ai_off_acc: Dict[str, float]) -> Dict[str, float]:
    """
    Computes AI gain (AI_ON - AI_OFF) per target.
    """
    return {
        target: ai_on_acc.get(target, 0.0) - ai_off_acc.get(target, 0.0)
        for target in set(ai_on_acc) | set(ai_off_acc)
    }
def test_ai_gain_central_vs_peripheral(per_target_gains):
    """
    per_target_gains: dict[target] -> list of session-wise gains (AI ON - AI OFF),
    each list length <= n_sessions (missing values allowed).
    Returns:
        mean_central, mean_peripheral, p_value, cohen_dz, (ci_low, ci_high), n_sessions_used
    """
    CENTRAL_TARGETS    = ("slight_left", "straight", "slight_right")
    PERIPHERAL_TARGETS = ("left", "right") 
    # Pad each target list to the same length with NaN
    lists = [np.asarray(per_target_gains.get(t, []), float) for t in CENTRAL_TARGETS + PERIPHERAL_TARGETS]
    if not lists:
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), 0
    nmax = max(len(a) for a in lists)

    def pad(a, n):  # pad to n with NaN at the end
        if len(a) < n:
            return np.pad(a, (0, n-len(a)), constant_values=np.nan)
        return a

    C = np.vstack([pad(np.asarray(per_target_gains.get(t, []), float), nmax) for t in CENTRAL_TARGETS])     # 3×S
    P = np.vstack([pad(np.asarray(per_target_gains.get(t, []), float), nmax) for t in PERIPHERAL_TARGETS])  # 2×S

    # Per-session means for each group
    meanC_s = np.nanmean(C, axis=0)  # shape (S,)
    meanP_s = np.nanmean(P, axis=0)  # shape (S,)

    # Keep sessions where both groups are available
    mask = np.isfinite(meanC_s) & np.isfinite(meanP_s)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), int(mask.sum())

    mean_central    = float(np.mean(meanC_s[mask]))
    mean_peripheral = float(np.mean(meanP_s[mask]))

    # Paired test and effect size (peripheral - central)
    t_stat, p_val = ttest_rel(meanP_s[mask], meanC_s[mask])
    dz = float(t_stat / np.sqrt(mask.sum()))               # Cohen's d_z

    # 95% CI for the paired mean difference
    diffs = meanP_s[mask] - meanC_s[mask]
    se = np.std(diffs, ddof=1) / np.sqrt(mask.sum())
    tcrit = t.ppf(0.975, df=mask.sum()-1)
    ci = (float(diffs.mean() - tcrit*se), float(diffs.mean() + tcrit*se))

    return mean_central, mean_peripheral, float(p_val), dz, ci, int(mask.sum())
# def test_ai_gain_central_vs_peripheral(ai_gain: Dict[str, float]) -> Tuple[float, float, float, float]:
#     """
#     Computes mean AI gain for central vs peripheral targets and runs a t-test.
#     Returns:
#         - mean_gain_central
#         - mean_gain_peripheral
#         - p-value (t-test)
#         - effect size (Cohen's d)
#     """
#     central_gains = [ai_gain[t] for t in CENTRAL_TARGETS if t in ai_gain]
#     peripheral_gains = [ai_gain[t] for t in PERIPHERAL_TARGETS if t in ai_gain]

#     mean_central = np.mean(central_gains)
#     mean_peripheral = np.mean(peripheral_gains)

#     t_stat, p_value = ttest_ind(peripheral_gains, central_gains, equal_var=False)

#     # Cohen's d
#     pooled_std = np.sqrt((np.std(peripheral_gains, ddof=1) ** 2 + np.std(central_gains, ddof=1) ** 2) / 2)
#     cohen_d = (mean_peripheral - mean_central) / pooled_std if pooled_std > 0 else 0.0

#     return mean_central, mean_peripheral, p_value, cohen_d