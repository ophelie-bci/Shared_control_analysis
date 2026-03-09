import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import pandas as pd
import glob
import pickle
from itertools import zip_longest
import math
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib import colors as mcolors
import matplotlib as mpl
from scipy.interpolate import interp1d

# Set font embedding for SVG export
mpl.rcParams['svg.fonttype'] = 'none'

from src.constants import target_mapping, target_to_obstacle_mapping
from src.stats_utils import ci95_t
from src.load import load_files

# from .constants import target_mapping, target_to_obstacle_mapping
# from .stats_utils import ci95_t
# from .load import load_files

def save_plot(fig, plot_name, subfolder):
    """Save a matplotlib figure in the figures/<subfolder>/ directory as SVG."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
    subfolder_path = os.path.join(base_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    fig_path = os.path.join(subfolder_path, f"{plot_name}.svg")
    fig.savefig(fig_path, format='svg', bbox_inches='tight')
    print(f"Saved: {fig_path}")

# --- helpers reused by plots too ---
def _resample_by_arclength(traj, K=100):
    if traj.shape[0] < 2:
        return np.tile(traj[0], (K,1))
    d = np.diff(traj, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
    s /= (s[-1] if s[-1] > 0 else 1.0)
    s_new = np.linspace(0, 1, K)
    return np.column_stack([np.interp(s_new, s, traj[:,j]) for j in range(traj.shape[1])])

def _mean_traj(trials, K=100):
    rs = np.stack([_resample_by_arclength(t, K) for t in trials], axis=0)   # (N,K,D)
    return rs.mean(axis=0), rs.std(axis=0)                                  # (K,D),(K,D)

def _pairwise_distance_profile(M1, M2):
    d = np.linalg.norm(M1 - M2, axis=1)
    return float(d.mean())

def _pca_reduce(X, k=3):
    Xc = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T

def plot_outcome_and_failure_modes_per_trial(grand_summary,
                                             experiments=None,
                                             monkeys=None):
    """
    Make bar plots like:
      'Outcome and failure modes per trial: No AI vs AI
       Monkey 3 – AI Obstacle'

    Parameters
    ----------
    grand_summary : dict
        Output of analyze_trajectories, expected to contain
        grand_summary["failure_props_per_unit"] = list of dicts,
        each with keys:
          - 'monkey', 'experiment', 'condition' ('No AI' or 'AI')
          - 'n_trials', 'success'
          - 'stuck_obstacle', 'ambiguous_choice',
            'wrong_choice', 'neighbor_choice',
            'overshoot', 'other'
    experiments : list or None
        If given, only plot these experiments.
    monkeys : list or None
        If given, only plot these monkeys.
    """

    if "failure_props_per_unit" not in grand_summary or not grand_summary["failure_props_per_unit"]:
        print("[plot_outcome_and_failure_modes_per_trial] "
              "No failure_props_per_unit in grand_summary.")
        return

    rows = grand_summary["failure_props_per_unit"]

    # Optional filtering
    if experiments is not None:
        rows = [r for r in rows if r.get("experiment") in experiments]
    if monkeys is not None:
        rows = [r for r in rows if r.get("monkey") in monkeys]

    if not rows:
        print("[plot_outcome_and_failure_modes_per_trial] "
              "No rows left after filtering.")
        return

    # Group by (monkey, experiment)
    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("monkey"), r.get("experiment"))
        grouped[key].append(r)

    categories = [
        "Success",
        "Stuck obstacle",
        "Ambiguous choice",
        "Wrong choice",
        "Neighbor choice",
        "Overshoot",
        "Other",
    ]

    # Loop over each (monkey, experiment) pair
    for (monkey, experiment), subrows in grouped.items():
        # Get the two conditions
        cond_map = {}
        for r in subrows:
            cond = r.get("condition", "")
            cond_map[cond] = r

        # We expect something like "No AI" and "AI"
        no_ai = cond_map.get("No AI") or cond_map.get("NoAI")
        ai_on = cond_map.get("AI")    or cond_map.get("AI On") or cond_map.get("AI_ON")

        if no_ai is None and ai_on is None:
            continue  # nothing to plot for this pair

        fig, ax = plt.subplots(figsize=(8, 4.5))

        x = np.arange(len(categories))
        width = 0.38

        def get_counts(row):
            if row is None:
                return np.zeros(len(categories), dtype=float), 1
            n_trials = max(1, int(row.get("n_trials", 1)))  # avoid div-by-zero
            counts = np.array([
                row.get("success", 0),
                row.get("stuck_obstacle", 0),
                row.get("ambiguous_choice", 0),
                row.get("wrong_choice", 0),
                row.get("neighbor_choice", 0),
                row.get("overshoot", 0),
                row.get("other", 0),
            ], dtype=float)
            return counts, n_trials

        counts_off, n_off = get_counts(no_ai)
        counts_on,  n_on  = get_counts(ai_on)

        pct_off = 100.0 * counts_off / n_off
        pct_on  = 100.0 * counts_on  / n_on

        bars_off = ax.bar(x - width/2, pct_off, width,
                          label=f"No AI (n={n_off})")
        bars_on  = ax.bar(x + width/2, pct_on, width,
                          label=f"AI (n={n_on})")

        # Annotate bars with percentages
        def annotate_bars(bars, values):
            for bar, val in zip(bars, values):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 1.0,
                        f"{val:.1f}%",
                        ha='center', va='bottom', fontsize=8)

        annotate_bars(bars_off, pct_off)
        annotate_bars(bars_on,  pct_on)

        ax.set_ylabel("Percentage of all trials (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_ylim(0, max(pct_off.max(), pct_on.max()) * 1.15)

        ax.legend(frameon=False)
        ax.set_title(f"Outcome and failure modes per trial: No AI vs AI\n"
                     f"{monkey} – {experiment}")

        ax.grid(axis='y', linestyle=':', alpha=0.5)
        fig.tight_layout()

    # function returns nothing, just makes the figures

from collections import defaultdict

from scipy.stats import wilcoxon

from scipy.stats import wilcoxon

# def plot_failure_modes_global(
#     grand_summary,
#     save_prefix="global_failure_modes"
# ):
#     """
#     Global barplot of failure-mode proportions,
#     averaged over monkey×experiment units, with
#     paired Wilcoxon significance (AI vs No AI).

#     Uses grand_summary["failure_props_per_unit"], where each row is:
#       {
#         "monkey": ...,
#         "experiment": ...,
#         "condition": "AI" or "No AI",
#         "success": ...,
#         "stuck_obstacle": ...,
#         "overshoot": ...,
#         "wrong_target_choice": ...,
#         "neighbor_choice": ...,
#         "ambiguous_choice": ...,
#         "ai_bci_failure_target": ...,
#         "wrong_target_other": ...,
#         "ambiguous_choice_unknown": ...,
#         "not_long_enough": ...,
#         "not_close_to_true_z": ...,
#         "other": ...
#       }

#     We plot four categories (fractions of ALL trials):
#       - stuck_obstacle
#       - wrong_target   (pooled wrong-target family)
#       - overshoot
#       - other          (timing / distance / residual)
#     """

#     rows = grand_summary.get("failure_props_per_unit", [])
#     if not rows:
#         print("[plot_failure_modes_global] No failure_props_per_unit in grand_summary.")
#         return

#     # Categories shown in the figure
#     categories = ["stuck_obstacle", "overshoot", "wrong_target", "other"]

#     # --- 1) Build per-unit dict and select units with both AI and No-AI ---
#     units = {}  # (monkey, experiment, condition) -> row
#     for r in rows:
#         key = (r.get("monkey", "?"),
#                r.get("experiment", "?"),
#                r.get("condition"))
#         units[key] = r

#     pair_keys = sorted({
#         (m, e)
#         for (m, e, c) in units.keys()
#         if (m, e, "AI") in units and (m, e, "No AI") in units
#     })

#     if not pair_keys:
#         print("[plot_failure_modes_global] No paired AI/No-AI units found.")
#         return

#     # Helper to compute the four categories for a single row
#     def compute_cats(r):
#         stuck = float(r.get("stuck_obstacle", 0.0))

#         wrong = (
#             float(r.get("wrong_target_choice", 0.0)) +
#             float(r.get("neighbor_choice", 0.0)) +
#             float(r.get("ambiguous_choice", 0.0)) +
#             float(r.get("ai_bci_failure_target", 0.0)) +
#             float(r.get("wrong_target_other", 0.0)) +
#             float(r.get("ambiguous_choice_unknown", 0.0))
#         )

#         overshoot = float(r.get("overshoot", 0.0))

#         other = (
#             float(r.get("not_long_enough", 0.0)) +
#             float(r.get("not_close_to_true_z", 0.0)) +
#             float(r.get("other", 0.0))
#         )

#         return {
#             "stuck_obstacle": stuck,
#             "overshoot": overshoot,
#             "wrong_target": wrong,
#             "other": other,
#         }

#     # --- 2) Collect per-unit values for AI and No-AI, aligned per unit ---
#     vals_off = {c: [] for c in categories}
#     vals_on  = {c: [] for c in categories}

#     for (m, e) in pair_keys:
#         r_off = units[(m, e, "No AI")]
#         r_on  = units[(m, e, "AI")]
#         cats_off = compute_cats(r_off)
#         cats_on  = compute_cats(r_on)
#         for c in categories:
#             vals_off[c].append(cats_off[c])
#             vals_on[c].append(cats_on[c])

#     # --- 3) Means, SEMs (in %) and paired Wilcoxon p-values ---
#     def mean_sem(arr):
#         if len(arr) == 0:
#             return np.nan, np.nan
#         a = np.asarray(arr, dtype=float)
#         m = np.mean(a)
#         if len(a) > 1:
#             s = np.std(a, ddof=1) / np.sqrt(len(a))
#         else:
#             s = np.nan
#         return m, s

#     def p_to_stars(p):
#         if not np.isfinite(p):
#             return ""
#         if p < 0.001:
#             return "***"
#         elif p < 0.01:
#             return "**"
#         elif p < 0.05:
#             return "*"
#         else:
#             return ""

#     means_off, sems_off, means_on, sems_on, pvals = [], [], [], [], []

#     for c in categories:
#         arr_off = np.asarray(vals_off[c], dtype=float)
#         arr_on  = np.asarray(vals_on[c], dtype=float)

#         # Means/SEMs in percent
#         m_off, s_off = mean_sem(arr_off * 100.0)
#         m_on,  s_on  = mean_sem(arr_on * 100.0)
#         means_off.append(m_off)
#         sems_off.append(s_off)
#         means_on.append(m_on)
#         sems_on.append(s_on)

#         # Paired Wilcoxon on fractions (0–1)
#         if len(arr_off) > 1 and np.any(arr_on != arr_off):
#             try:
#                 stat, p = wilcoxon(arr_on, arr_off, alternative="two-sided")
#             except ValueError:
#                 p = np.nan
#         else:
#             p = np.nan
#         pvals.append(p)

#     # --- 4) Plot ---
#     x = np.arange(len(categories))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(7, 4))

#     # No AI = red, AI = green
#     bars_off = ax.bar(
#         x - width / 2,
#         means_off,
#         width,
#         yerr=sems_off,
#         label="AI Off",
#         color="#e74c3c",
#         edgecolor="black",
#         linewidth=0.5,
#         capsize=4,
#         alpha=0.9,
#     )
#     bars_on = ax.bar(
#         x + width / 2,
#         means_on,
#         width,
#         yerr=sems_on,
#         label="AI On",
#         color="#2ecc71",
#         edgecolor="black",
#         linewidth=0.5,
#         capsize=4,
#         alpha=0.9,
#     )

#     ax.set_xticks(x)
#     ax.set_xticklabels(
#         ["Stuck\nat\nobstacle", "Overshoot", "Wrong\ntarget", "Other"],
#         rotation=0
#     )
#     ax.set_ylabel("Percentage of all trials (%)")

#     ymax = max(max(means_off), max(means_on)) * 1.35
#     ax.set_ylim(0, ymax)

#     ax.legend(frameon=False)
#     ax.set_title("Failure modes (averaged over monkey×task)")

#     # --- 5) Significance bars (asterisks) ---
#     h = ymax * 0.03  # vertical spacing for bars/text

#     for i, (bar_off, bar_on, p) in enumerate(zip(bars_off, bars_on, pvals)):
#         stars = p_to_stars(p)
#         if not stars:
#             continue

#         x1 = bar_off.get_x() + bar_off.get_width() / 2.0
#         x2 = bar_on.get_x() + bar_on.get_width() / 2.0
#         y1 = bar_off.get_height()
#         y2 = bar_on.get_height()
#         y  = max(y1, y2) + h

#         # draw connection line
#         ax.plot([x1, x1, x2, x2],
#                 [y,  y + h, y + h, y],
#                 color="black", linewidth=1.0)
#         # draw stars
#         ax.text((x1 + x2) / 2.0, y + h * 1.1, stars,
#                 ha="center", va="bottom", fontsize=10)

#     plt.tight_layout()
#     save_plot(fig, f"{save_prefix}_failure_modes", "Failure modes (global)")
#     plt.show(fig)

def plot_failure_modes_global(
    grand_summary,
    save_prefix="execution_failure_modes"
):
    """
    Execution failure modes with paired AI vs No-AI comparisons.

    Uses grand_summary["failure_props_per_unit"], where each row is:
      {
        "monkey": ...,
        "experiment": ...,
        "condition": "AI" or "No AI",
        "stuck_obstacle": ...,
        "overshoot": ...,
        "not_close_to_true_z": ...,
        "not_long_enough": ...,
        ...
      }

    We plot four execution categories (fractions of ALL trials):
      - stuck_obstacle
      - overshoot
      - not_close_to_true_z
      - not_long_enough

    For stats we run a paired Wilcoxon **one-sided** test with
    alternative="less" on (AI On vs AI Off), i.e. testing whether
    AI On has *lower* failure rate than No AI.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = grand_summary.get("failure_props_per_unit", [])
    if not rows:
        print("[plot_failure_modes_global] No failure_props_per_unit in grand_summary.")
        return

    # Execution-only categories
    categories = [
        "stuck_obstacle",
        "overshoot",
        "not_close_to_true_z",
        "not_long_enough",
    ]

    # -------------------- build unit lookup --------------------
    units = {}  # (monkey, experiment, condition) -> row
    for r in rows:
        m = r.get("monkey", "?")
        e = r.get("experiment", "?")
        c = r.get("condition")
        if c not in ("AI", "No AI"):
            continue
        units[(m, e, c)] = r

    if not units:
        print("[plot_failure_modes_global] No AI/No-AI rows found.")
        return

    # Helper: extract execution cats for one row
    def compute_cats(r):
        return {
            "stuck_obstacle": float(r.get("stuck_obstacle", 0.0)),
            "overshoot": float(r.get("overshoot", 0.0)),
            "not_close_to_true_z": float(r.get("not_close_to_true_z", 0.0)),
            "not_long_enough": float(r.get("not_long_enough", 0.0)),
        }

    def mean_sem(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        m = float(np.mean(a))
        if a.size > 1:
            s = float(np.std(a, ddof=1) / np.sqrt(a.size))
        else:
            s = np.nan
        return m, s

    def p_to_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def collect_pairs(pair_keys):
        """Return dicts category -> list of vals for No AI / AI."""
        vals_off = {c: [] for c in categories}
        vals_on  = {c: [] for c in categories}
        for (m, e) in pair_keys:
            r_off = units.get((m, e, "No AI"), None)
            r_on  = units.get((m, e, "AI"), None)
            if r_off is None or r_on is None:
                continue
            cats_off = compute_cats(r_off)
            cats_on  = compute_cats(r_on)
            for c in categories:
                vals_off[c].append(cats_off[c])
                vals_on[c].append(cats_on[c])
        return vals_off, vals_on

    def make_plot(pair_keys, title, outname):
        if not pair_keys:
            print(f"[plot_failure_modes_global] {title}: no paired AI/No-AI units.")
            return

        vals_off, vals_on = collect_pairs(pair_keys)
        if all(len(vals_off[c]) == 0 for c in categories):
            print(f"[plot_failure_modes_global] {title}: paired keys but no usable rows.")
            return

        means_off, sems_off, means_on, sems_on, pvals = [], [], [], [], []

        for c in categories:
            arr_off = np.asarray(vals_off[c], dtype=float)
            arr_on  = np.asarray(vals_on[c], dtype=float)

            m_off, s_off = mean_sem(arr_off * 100.0)
            m_on,  s_on  = mean_sem(arr_on  * 100.0)
            means_off.append(m_off); sems_off.append(s_off)
            means_on.append(m_on);   sems_on.append(s_on)

            # one-sided paired Wilcoxon: AI On < No AI
            p = np.nan
            if wilcoxon is not None and arr_off.size > 1:
                ok = np.isfinite(arr_off) & np.isfinite(arr_on)
                a0 = arr_off[ok]  # No AI
                a1 = arr_on[ok]   # AI
                if a0.size > 1 and np.any(a1 != a0):
                    try:
                        # tests median(a1 - a0) < 0  -> AI On < No AI
                        _, p = wilcoxon(a1, a0, alternative="less")
                    except Exception:
                        p = np.nan
            pvals.append(p)

        x = np.arange(len(categories), dtype=float)
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))

        def draw_group(offset, means, sems, label, facecolor):
            xs = x + offset
            bars = ax.bar(
                xs,
                means,
                width,
                label=label,
                color=facecolor,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
            means_a = np.asarray(means, float)
            sems_a  = np.asarray(sems, float)
            ok = np.isfinite(means_a) & np.isfinite(sems_a)
            if np.any(ok):
                ax.errorbar(
                    xs[ok],
                    means_a[ok],
                    yerr=sems_a[ok],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=4,
                    capthick=1.0,
                    zorder=5,
                )
            return bars

        bars_off = draw_group(-width / 2, means_off, sems_off, "AI Off", "#e74c3c")
        bars_on  = draw_group(+width / 2, means_on,  sems_on,  "AI On",  "#2ecc71")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                "Stuck\nat\nobstacle",
                "Overshoot",
                "Not close\nenough",
                "Not long\nenough",
            ],
            rotation=0,
        )
        ax.set_ylabel("Percentage of all trials (%)")
        ax.set_title(title)
        ax.legend(frameon=False)

        finite_vals = [v for v in list(means_off) + list(means_on) if np.isfinite(v)]
        ymax = max(finite_vals) * 1.35 if finite_vals else 1.0
        ymax = max(1e-6, ymax)
        ax.set_ylim(0, ymax)

        # significance bars
        h = ymax * 0.03
        for bar_off, bar_on, p in zip(bars_off, bars_on, pvals):
            stars = p_to_stars(p)
            if not stars:
                continue
            x1 = bar_off.get_x() + bar_off.get_width() / 2.0
            x2 = bar_on.get_x()  + bar_on.get_width()  / 2.0
            y  = max(bar_off.get_height(), bar_on.get_height()) + h
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    color="black", linewidth=1.0)
            ax.text((x1 + x2) / 2.0, y + h * 1.1, stars,
                    ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        save_plot(fig, outname, title)
        plt.show()

    # =========================
    # 1) GLOBAL (all monkeys × tasks)
    # =========================
    global_pair_keys = sorted({
        (m, e)
        for (m, e, c) in units.keys()
        if (m, e, "AI") in units and (m, e, "No AI") in units
    })
    make_plot(
        global_pair_keys,
        title="Execution failure modes (all monkeys × all tasks)",
        outname=f"{save_prefix}__global"
    )

    # # =========================
    # # 2) PER MONKEY (across tasks)
    # # =========================
    # monkey_list = sorted({m for (m, e, c) in units.keys()})
    # for m in monkey_list:
    #     pair_keys_m = sorted({
    #         (m2, e)
    #         for (m2, e, c) in units.keys()
    #         if m2 == m and (m2, e, "AI") in units and (m2, e, "No AI") in units
    #     })
    #     make_plot(
    #         pair_keys_m,
    #         title=f"Execution failure modes (monkey {m}; across tasks)",
    #         outname=f"{save_prefix}__monkey_{m}"
    #     )

    # # =========================
    # # 3) PER TASK / EXPERIMENT (across monkeys)
    # # =========================
    # experiment_list = sorted({e for (m, e, c) in units.keys()})
    # for e in experiment_list:
    #     pair_keys_e = sorted({
    #         (m, e2)
    #         for (m, e2, c) in units.keys()
    #         if e2 == e and (m, e2, "AI") in units and (m, e2, "No AI") in units
    #     })
    #     make_plot(
    #         pair_keys_e,
    #         title=f"Execution failure modes (task {e}; across monkeys)",
    #         outname=f"{save_prefix}__task_{e}"
    #     )
# def plot_behavioral_failure_by_choice_state(
#     grand_summary,
#     save_prefix="behavioral_failure_by_choice_state",
#     title_prefix="Behavioural failures by choice state",
# ):
#     """
#     For each choice state (correct / neighbor / ambiguous / wrong),
#     compute, per monkey×task unit:

#         p_fail_off_behavioral = 1 - p_success_off_behavioral
#         p_fail_on_behavioral  = 1 - p_success_on_behavioral

#     using grand_summary["success_state_per_unit"], and plot AI Off vs AI On
#     with paired one-sided Wilcoxon tests (H1: AI On has lower failure rate).

#     This is strictly:
#         P(fail | choice state, no execution error)
#     where "fail" = any non-execution failure (mostly wrong-target family).
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     try:
#         from scipy.stats import wilcoxon
#     except Exception:
#         wilcoxon = None

#     rows = grand_summary.get("success_state_per_unit", [])
#     if not rows:
#         print("[plot_behavioral_failure_by_choice_state] No success_state_per_unit in grand_summary.")
#         return

#     states_order = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]
#     state_labels = {
#         "correct_choice":   "Correct\nchoice",
#         "neighbor_choice":  "Neighbor\nchoice",
#         "ambiguous_choice": "Ambiguous",
#         "wrong_choice":     "Wrong\nchoice",
#     }

#     # ---- collect per-unit failure rates per state ----
#     vals_off = {s: [] for s in states_order}
#     vals_on  = {s: [] for s in states_order}

#     for r in rows:
#         s = r.get("state", None)
#         if s not in states_order:
#             continue

#         p_succ_off = r.get("p_success_off_behavioral", np.nan)
#         p_succ_on  = r.get("p_success_on_behavioral",  np.nan)

#         # turn success → failure; keep NaNs as NaNs
#         if np.isfinite(p_succ_off):
#             vals_off[s].append(1.0 - float(p_succ_off))
#         else:
#             vals_off[s].append(np.nan)

#         if np.isfinite(p_succ_on):
#             vals_on[s].append(1.0 - float(p_succ_on))
#         else:
#             vals_on[s].append(np.nan)

#     def mean_sem(a):
#         a = np.asarray(a, float)
#         a = a[np.isfinite(a)]
#         if a.size == 0:
#             return np.nan, np.nan
#         m = float(np.mean(a))
#         if a.size > 1:
#             s = float(np.std(a, ddof=1) / np.sqrt(a.size))
#         else:
#             s = np.nan
#         return m, s

#     def p_to_stars(p):
#         if not np.isfinite(p):
#             return ""
#         if p < 0.001:
#             return "***"
#         if p < 0.01:
#             return "**"
#         if p < 0.05:
#             return "*"
#         return ""

#     means_off, sems_off, means_on, sems_on, pvals = [], [], [], [], []

#     for s in states_order:
#         arr_off = np.asarray(vals_off[s], float)
#         arr_on  = np.asarray(vals_on[s],  float)

#         m_off, se_off = mean_sem(arr_off * 100.0)  # convert to %
#         m_on,  se_on  = mean_sem(arr_on  * 100.0)

#         means_off.append(m_off); sems_off.append(se_off)
#         means_on.append(m_on);   sems_on.append(se_on)

#         # paired Wilcoxon, one-sided: AI On failure < AI Off failure
#         p = np.nan
#         if wilcoxon is not None:
#             ok = np.isfinite(arr_off) & np.isfinite(arr_on)
#             a0 = arr_off[ok]
#             a1 = arr_on[ok]
#             if a0.size > 1 and np.any(a1 != a0):
#                 try:
#                     _, p = wilcoxon(a1, a0, alternative="less")
#                 except Exception:
#                     p = np.nan
#         pvals.append(p)

#     # ---- plot ----
#     x = np.arange(len(states_order), dtype=float)
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(7, 4))

#     def draw_group(offset, means, sems, label, color):
#         xs = x + offset
#         bars = ax.bar(
#             xs,
#             means,
#             width,
#             label=label,
#             color=color,
#             edgecolor="black",
#             linewidth=0.5,
#             alpha=0.9,
#         )
#         means_a = np.asarray(means, float)
#         sems_a  = np.asarray(sems, float)
#         ok = np.isfinite(means_a) & np.isfinite(sems_a)
#         if np.any(ok):
#             ax.errorbar(
#                 xs[ok],
#                 means_a[ok],
#                 yerr=sems_a[ok],
#                 fmt="none",
#                 ecolor="black",
#                 elinewidth=1.0,
#                 capsize=4,
#                 capthick=1.0,
#                 zorder=5,
#             )
#         return bars

#     bars_off = draw_group(-width/2, means_off, sems_off, "AI Off", "#e74c3c")
#     bars_on  = draw_group(+width/2, means_on,  sems_on,  "AI On",  "#2ecc71")

#     ax.set_xticks(x)
#     ax.set_xticklabels([state_labels[s] for s in states_order])
#     ax.set_ylabel("P(fail | choice state, no execution) (%)")
#     ax.set_title(f"{title_prefix} (behavioural errors only)")
#     ax.legend(frameon=False)

#     # y-limits
#     finite_vals = [v for v in list(means_off) + list(means_on) if np.isfinite(v)]
#     ymax = max(finite_vals) if finite_vals else 1.0
#     ymax = max(1e-6, ymax) * 1.35
#     ax.set_ylim(0, ymax)

#     # significance stars
#     h = ymax * 0.03
#     for bar_off, bar_on, p in zip(bars_off, bars_on, pvals):
#         stars = p_to_stars(p)
#         if not stars:
#             continue
#         x1 = bar_off.get_x() + bar_off.get_width()/2.0
#         x2 = bar_on.get_x()  + bar_on.get_width()/2.0
#         y  = max(bar_off.get_height(), bar_on.get_height()) + h
#         ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", linewidth=1.0)
#         ax.text((x1+x2)/2.0, y + h*1.1, stars, ha="center", va="bottom", fontsize=10)

#     plt.tight_layout()
#     # optional: if you have save_plot(...) helper:
#     try:
#         save_plot(fig, f"{save_prefix}", ax.get_title())
#     except Exception:
#         pass
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def plot_ambiguity_metric_deltas(
    grand_summary,
    metric="entropy_soft_norm",          # pass "entropy_soft_norm" or "median_entropy_soft_norm"
    subset="global",                     # "global", "neighbor", "all", or None
    outcome=None,                        # None | "correct" | "incorrect" | "all"  (requires rows to have r["outcome"])
    save_prefix="ambiguity_metric",
    show=True,
):
    """
    Makes 3 figures for a metric computed on (ambiguous) trials:
      1) ALL monkey×task units (paired deltas, AI - No AI)
      2) per monkey (across all tasks; average within monkey, then paired across monkeys)
      3) per task/experiment (across all monkeys; average within task, then paired across tasks)

    Expects:
      grand_summary["ambiguity_metrics_per_unit"] = list of dict rows like:
        {
          "monkey": "...",
          "experiment": "...",
          "condition": "AI" or "No AI",
          "subset": "all" | "global" | "neighbor",
          "outcome": "correct" | "incorrect" | "all",   # OPTIONAL but needed if you want outcome filtering
          "median_entropy_soft_norm": float,
          "median_switch_rate_hz": float,
          ...
        }
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    import re

    rows = grand_summary.get("ambiguity_metrics_per_unit", [])
    if not rows:
        print("[plot_ambiguity_metric_deltas] No ambiguity_metrics_per_unit in grand_summary.")
        return

    # ---- resolve metric key ----
    metric_key = metric if metric.startswith("median_") else f"median_{metric}"

    # ---- filter by subset ----
    if subset is not None:
        rows = [r for r in rows if str(r.get("subset", "")).lower() == str(subset).lower()]
        if not rows:
            print(f"[plot_ambiguity_metric_deltas] No rows left after subset='{subset}' filter.")
            return

    # ---- OPTIONAL filter by outcome (only works if your rows include it) ----
    if outcome is not None and str(outcome).lower() != "all":
        has_outcome_key = any(("outcome" in r) for r in rows)
        if not has_outcome_key:
            print("[plot_ambiguity_metric_deltas] outcome filter requested, but rows have no 'outcome' key. "
                  "Upstream summary must add r['outcome'] = 'correct'/'incorrect'/'all'. Proceeding without outcome filter.")
        else:
            rows = [r for r in rows if str(r.get("outcome", "")).lower() == str(outcome).lower()]
            if not rows:
                print(f"[plot_ambiguity_metric_deltas] No rows left after outcome='{outcome}' filter.")
                return

    # ---- normalize condition strings to exactly "AI" / "No AI" ----
    def _norm_condition(c):
        c = str(c).strip().lower()
        if c in ("ai", "ai on", "on", "1", "1.0", "true"):
            return "AI"
        if c in ("no ai", "ai off", "off", "0", "0.0", "false"):
            return "No AI"
        return str(c)

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    # ---- index by (monkey, experiment, condition) ----
    units = {}
    for r in rows:
        m = r.get("monkey", None)
        e = r.get("experiment", None)
        c = _norm_condition(r.get("condition", None))
        if m is None or e is None or c not in ("AI", "No AI"):
            continue
        units[(m, e, c)] = r

    pair_keys = sorted({
        (m, e)
        for (m, e, c) in units.keys()
        if (m, e, "AI") in units and (m, e, "No AI") in units
    })
    if not pair_keys:
        print("[plot_ambiguity_metric_deltas] No paired AI/No-AI units found for this subset/outcome.")
        return

    # ---- extract paired arrays for a list of (m,e) pairs ----
    def _paired_arrays(pairs):
        on = []
        off = []
        labels = []
        for (m, e) in pairs:
            r_on = units.get((m, e, "AI"), None)
            r_off = units.get((m, e, "No AI"), None)
            if r_on is None or r_off is None:
                continue
            v_on = _safe_float(r_on.get(metric_key, np.nan))
            v_off = _safe_float(r_off.get(metric_key, np.nan))
            if np.isfinite(v_on) and np.isfinite(v_off):
                on.append(v_on)
                off.append(v_off)
                labels.append((m, e))
        return np.asarray(on, float), np.asarray(off, float), labels

    def _mean_sem(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan, np.nan
        m = float(np.mean(x))
        sem = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan
        return m, sem

    def _wilcoxon_p(on, off):
        ok = np.isfinite(on) & np.isfinite(off)
        on = on[ok]; off = off[ok]
        if on.size < 2:
            return np.nan, int(on.size)
        if np.allclose(on, off, equal_nan=False):
            return 1.0, int(on.size)
        try:
            p = float(wilcoxon(on, off, alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        return p, int(on.size)

    # ---- manual mean±SEM glyph (no errorbar -> avoids StopIteration) ----
    def _draw_mean_sem(ax, x0, mean, sem):
        ax.scatter([x0], [mean], s=80, facecolor="black", edgecolor="black", zorder=4)
        if np.isfinite(sem) and sem > 0:
            ax.vlines(x0, mean - sem, mean + sem, color="black", linewidth=1.6, zorder=3)
            cap = 0.08
            ax.hlines([mean - sem, mean + sem], x0 - cap, x0 + cap, color="black", linewidth=1.6, zorder=3)

    def _sanitize(s: str, maxlen: int = 120) -> str:
        s = str(s).replace("×", "x").replace("—", "-").replace("–", "-")
        s = re.sub(r'[<>:"/\\|?*]+', "_", s)   # Windows-illegal chars
        s = re.sub(r"\s+", " ", s).strip().rstrip(". ")
        return s[:maxlen] if len(s) > maxlen else s

    def _save(fig, out_name, title):
        out_name = _sanitize(out_name)
        title = _sanitize(title)
        if "save_plot" in globals():
            save_plot(fig, out_name, title)
        else:
            fig.savefig(f"{out_name}.png", dpi=200)

    def _plot_deltas(deltas, xticklabels, title, out_name):
        deltas = np.asarray(deltas, float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            print(f"[plot_ambiguity_metric_deltas] No finite deltas for {title}.")
            return None

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axhline(0, color="black", linewidth=1.0, alpha=0.6, zorder=1)

        x = np.arange(len(xticklabels), dtype=float)

        rng = np.random.RandomState(0)
        for i, d in enumerate(deltas):
            xi = x[i] if len(x) == len(deltas) else 0.0
            jit = float(rng.randn() * 0.03)
            ax.scatter([xi + jit], [d], s=45, facecolor="white", edgecolor="black", linewidth=0.9, zorder=3)

        m, sem = _mean_sem(deltas)
        x_mean = float(np.mean(x)) if len(x) > 0 else 0.0
        _draw_mean_sem(ax, x_mean, m, sem)

        ax.set_xticks(x if len(x) == len(xticklabels) else [0.0])
        ax.set_xticklabels(xticklabels if len(x) == len(xticklabels) else [xticklabels[0]])
        ax.set_ylabel(f"Δ {metric_key}  (AI − No AI)")
        ax.set_title(title)

        lo, hi = np.nanpercentile(deltas, [5, 95])
        if np.isclose(lo, hi):
            lo -= 1e-6; hi += 1e-6
        pad = 0.2 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        _save(fig, out_name, title)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # label suffix for filenames/titles
    outcome_str = "all" if (outcome is None) else str(outcome).lower()

    # =========================
    # 1) ALL monkey×task units
    # =========================
    on_all, off_all, _ = _paired_arrays(pair_keys)
    deltas_all = on_all - off_all
    p_all, n_all = _wilcoxon_p(on_all, off_all)

    title1 = f"{metric_key} (subset={subset}, outcome={outcome_str}) — all monkey×task units | Wilcoxon p={p_all:.3g}, n={n_all}"
    fig1 = _plot_deltas(
        deltas_all,
        xticklabels=[f"All units (n={len(deltas_all)})"],
        title=title1,
        out_name=f"{save_prefix}_{metric_key}_{subset}_{outcome_str}_ALL_units",
    )

    # ==================================
    # 2) Per monkey (across all tasks)
    # ==================================
    monkeys = sorted({m for (m, _e) in pair_keys})
    on_m, off_m, deltas_m, labels_m = [], [], [], []
    for m in monkeys:
        pairs_m = [(mm, ee) for (mm, ee) in pair_keys if mm == m]
        on, off, _ = _paired_arrays(pairs_m)
        if on.size == 0:
            continue
        on_mean = float(np.mean(on))
        off_mean = float(np.mean(off))
        on_m.append(on_mean); off_m.append(off_mean)
        deltas_m.append(on_mean - off_mean)
        labels_m.append(m)

    on_m = np.asarray(on_m, float); off_m = np.asarray(off_m, float)
    p_m, n_m = _wilcoxon_p(on_m, off_m)

    title2 = f"{metric_key} (subset={subset}, outcome={outcome_str}) — per monkey (avg across tasks) | Wilcoxon p={p_m:.3g}, n={n_m}"
    fig2 = None
    if len(deltas_m) > 0:
        fig2 = _plot_deltas(
            np.asarray(deltas_m, float),
            xticklabels=labels_m,
            title=title2,
            out_name=f"{save_prefix}_{metric_key}_{subset}_{outcome_str}_PER_monkey",
        )

    # ======================================
    # 3) Per task/experiment (across monkeys)
    # ======================================
    tasks = sorted({e for (_m, e) in pair_keys})
    on_t, off_t, deltas_t, labels_t = [], [], [], []
    for e in tasks:
        pairs_e = [(mm, ee) for (mm, ee) in pair_keys if ee == e]
        on, off, _ = _paired_arrays(pairs_e)
        if on.size == 0:
            continue
        on_mean = float(np.mean(on))
        off_mean = float(np.mean(off))
        on_t.append(on_mean); off_t.append(off_mean)
        deltas_t.append(on_mean - off_mean)
        labels_t.append(e)

    on_t = np.asarray(on_t, float); off_t = np.asarray(off_t, float)
    p_t, n_t = _wilcoxon_p(on_t, off_t)

    title3 = f"{metric_key} (subset={subset}, outcome={outcome_str}) — per task (avg across monkeys) | Wilcoxon p={p_t:.3g}, n={n_t}"
    fig3 = None
    if len(deltas_t) > 0:
        fig3 = _plot_deltas(
            np.asarray(deltas_t, float),
            xticklabels=labels_t,
            title=title3,
            out_name=f"{save_prefix}_{metric_key}_{subset}_{outcome_str}_PER_task",
        )

    return fig1, fig2, fig3




def plot_wrong_target_subtypes(grand_summary, save_prefix="wrong_target_subtypes"):
    """
    Show how AI uses intention, based on wrong_target subtypes.

    We use four categories (fractions of ALL trials), per unit
    (monkey×experiment), and compare AI Off vs AI On:

      1) Correct choice → wrong target
         ('wrong_target_correct_choice')

      2) Ambiguous choice → wrong target
         ('wrong_target_ambiguous_choice')

      3) Monkey wrong and followed
         ('wrong_target_monkey_followed')

      4) Monkey wrong and overruled
         ('wrong_target_monkey_overruled')

    Each bar = mean over units; errorbar = SEM.
    Wilcoxon signed-rank (one-sided: AI On < AI Off) on fractions.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = grand_summary.get("wrong_target_subtypes_per_unit", [])
    if not rows:
        print("[plot_wrong_target_subtypes] No wrong_target_subtypes_per_unit in grand_summary.")
        return

    # internal keys + pretty labels
    cats = [
        "wt_correct_choice",
        "wt_ambiguous_choice",
        "wt_monkey_followed",
        "wt_monkey_overruled",
    ]
    cat_labels = [
        "Correct choice\n→ wrong target",
        "Ambiguous choice\n→ wrong target",
        "Monkey wrong\nand followed",
        "Monkey wrong\nand overruled",
    ]

    # map (monkey, experiment, condition) -> row
    units = {}
    for r in rows:
        m = r.get("monkey", "?")
        e = r.get("experiment", "?")
        c = r.get("condition")
        if c not in ("AI", "No AI"):
            continue
        units[(m, e, c)] = r

    # paired keys: (monkey, experiment) with both AI and No-AI
    pair_keys = sorted({
        (m, e)
        for (m, e, c) in units.keys()
        if (m, e, "AI") in units and (m, e, "No AI") in units
    })
    if not pair_keys:
        print("[plot_wrong_target_subtypes] No paired AI/No-AI units found.")
        return

    # convert a row to per-category fractions of ALL trials
    def compute_cats(r):
        denom = float(r.get("n_trials_total", 0) or 0)
        if denom <= 0:
            return {k: np.nan for k in cats}

        wt_corr   = float(r.get("n_wrong_target_correct_choice", 0))
        wt_amb    = float(r.get("n_wrong_target_ambiguous_choice", 0))
        wtm_follow = float(r.get("n_wrong_target_monkey_followed", 0))
        wtm_over   = float(r.get("n_wrong_target_monkey_overruled", 0))

        return {
            "wt_correct_choice":   wt_corr   / denom,
            "wt_ambiguous_choice": wt_amb    / denom,
            "wt_monkey_followed":  wtm_follow / denom,
            "wt_monkey_overruled": wtm_over   / denom,
        }

    # collect per-unit values for AI Off / AI On
    vals_off = {k: [] for k in cats}
    vals_on  = {k: [] for k in cats}

    for (m, e) in pair_keys:
        r_off = units[(m, e, "No AI")]
        r_on  = units[(m, e, "AI")]
        co = compute_cats(r_off)
        cn = compute_cats(r_on)
        for k in cats:
            vals_off[k].append(co[k])
            vals_on[k].append(cn[k])

    def mean_sem(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        m = float(np.mean(a))
        if a.size > 1:
            s = float(np.std(a, ddof=1) / np.sqrt(a.size))
        else:
            s = np.nan
        return m, s

    def p_to_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    means_off, sems_off, means_on, sems_on, pvals = [], [], [], [], []

    for k in cats:
        arr_off = np.asarray(vals_off[k], dtype=float)
        arr_on  = np.asarray(vals_on[k], dtype=float)

        m_off, s_off = mean_sem(arr_off * 100.0)
        m_on,  s_on  = mean_sem(arr_on  * 100.0)

        means_off.append(m_off)
        sems_off.append(s_off)
        means_on.append(m_on)
        sems_on.append(s_on)

        # one-sided Wilcoxon: AI On < No AI (fewer bad outcomes)
        p = np.nan
        if wilcoxon is not None:
            ok = np.isfinite(arr_off) & np.isfinite(arr_on)
            a0 = arr_off[ok]; a1 = arr_on[ok]
            if a0.size > 1 and np.any(a1 != a0):
                try:
                    _, p = wilcoxon(a1, a0, alternative="less")
                except Exception:
                    p = np.nan
        pvals.append(p)

    # --- plotting ---
    x = np.arange(len(cats), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    def draw_group(offset, means, sems, label, color):
        xs = x + offset
        bars = ax.bar(
            xs,
            means,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        means_a = np.asarray(means, float)
        sems_a  = np.asarray(sems, float)
        ok = np.isfinite(means_a) & np.isfinite(sems_a)
        if np.any(ok):
            ax.errorbar(
                xs[ok],
                means_a[ok],
                yerr=sems_a[ok],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
                capthick=1.0,
                zorder=5,
            )
        return bars

    bars_off = draw_group(-width/2, means_off, sems_off, "AI Off", "#e74c3c")
    bars_on  = draw_group(+width/2, means_on,  sems_on,  "AI On",  "#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=0)
    ax.set_ylabel("Percentage of all trials (%)")
    ax.set_title("Wrong-target subtypes (averaged over monkey×task)")
    ax.legend(frameon=False)

    finite_vals = [v for v in list(means_off) + list(means_on) if np.isfinite(v)]
    ymax = max(finite_vals) * 1.35 if finite_vals else 1.0
    ax.set_ylim(0, ymax)

    # significance bars
    h = ymax * 0.03
    for bar_off, bar_on, p in zip(bars_off, bars_on, pvals):
        stars = p_to_stars(p)
        if not stars:
            continue
        x1 = bar_off.get_x() + bar_off.get_width()/2.0
        x2 = bar_on.get_x()  + bar_on.get_width()/2.0
        y  = max(bar_off.get_height(), bar_on.get_height()) + h
        ax.plot([x1, x1, x2, x2],
                [y,  y + h, y + h, y],
                color="black", linewidth=1.0)
        ax.text((x1 + x2)/2.0, y + h*1.1, stars,
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_plot(fig, f"{save_prefix}_wrong_target_subtypes",
              "Wrong-target subtypes (global)")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon, ttest_rel   # make sure this import is present

def plot_success_by_choice_state_global(
    grand_summary,
    save_prefix="global_success_by_state"
):
    """
    Global barplot of P(success | choice state) for AI vs No-AI,
    averaged over monkey×experiment units, with significance stars.
    Uses grand_summary["success_state_per_unit"].
    """

    rows = grand_summary.get("success_state_per_unit", [])
    if not rows:
        print("[plot_success_by_choice_state_global] No success_state_per_unit in grand_summary.")
        return

    # The states we care about, in a fixed order
    states = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]

    # -------- helper: collect per-unit values and paired p-values --------
    def collect_vals_and_pvals():
        vals_off = {s: [] for s in states}
        vals_on  = {s: [] for s in states}
        pvals    = {s: np.nan for s in states}

        for s in states:
            # group by (monkey, experiment) for this state
            per_unit = defaultdict(dict)
            off_vals = []
            on_vals = []
            for r in rows:
                if r.get("state") != s:
                    continue

                p_off = float(r.get("p_success_off", np.nan))
                p_on  = float(r.get("p_success_on", np.nan))

                if np.isfinite(p_off) and np.isfinite(p_on):
                    off_vals.append(p_off)
                    on_vals.append(p_on)

            vals_off[s] = off_vals
            vals_on[s]  = on_vals

            if len(off_vals) >= 2:
                diffs = np.array(on_vals) - np.array(off_vals)
                try:
                    stat, p = wilcoxon(diffs)
                except ValueError:
                    stat, p = ttest_rel(on_vals, off_vals)
                pvals[s] = p

        return vals_off, vals_on, pvals

    vals_off, vals_on, pvals = collect_vals_and_pvals()
    print(pvals)
    def mean_sem(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan, np.nan
        m = np.mean(arr)
        s = np.std(arr, ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan
        return m, s

    means_off, sems_off, means_on, sems_on = [], [], [], []
    for s in states:
        m_off, s_off = mean_sem(vals_off[s])
        m_on,  s_on  = mean_sem(vals_on[s])
        means_off.append(m_off)
        sems_off.append(s_off)
        means_on.append(m_on)
        sems_on.append(s_on)

    x = np.arange(len(states))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_off = ax.bar(
        x - width/2,
        means_off,
        width,
        yerr=sems_off,
        label="AI Off",
        capsize=4,
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    bars_on = ax.bar(
        x + width/2,
        means_on,
        width,
        yerr=sems_on,
        label="AI On",
        capsize=4,
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Correct\nchoice", "Neighbor\nchoice", "Ambiguous", "Wrong\nchoice"],
        rotation=0
    )
    ax.set_ylabel("P(success | choice state)")

    # ------------- significance stars -------------
    def star_for_p(p):
        if np.isnan(p):
            return ""
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 5e-2:
            return "*"
        else:
            return ""

    y_max_for_ylim = 0.0
    for i, s in enumerate(states):
        p = pvals[s]
        star = star_for_p(p)
        if star == "":
            continue

        
        # top of the taller bar (mean + sem)
        m_off = means_off[i]
        m_on  = means_on[i]
        s_off = 0 if np.isnan(sems_off[i]) else sems_off[i]
        s_on  = 0 if np.isnan(sems_on[i]) else sems_on[i]
        y_bar = max(m_off + s_off, m_on + s_on)

        # vertical extent for the sig bar
        gap = 0.04          # gap above the error bars
        h   = 0.02          # height of the little vertical ticks

        y = y_bar + gap

        # x positions at the centers of the two bars
        x1 = x[i] - width / 2.0
        x2 = x[i] + width / 2.0

        # draw: up from left bar, across, down to right bar
        ax.plot(
            [x1, x1, x2, x2],
            [y,  y + h, y + h, y],
            color="black",
            linewidth=1.0,
        )

        # asterisks in the middle above the bar
        ax.text(
            (x1 + x2) / 2.0,
            y + h * 1.1,
            star,                 # e.g. "*", "**", or "***"
            ha="center",
            va="bottom",
            fontsize=10,
        )
        y_max_for_ylim = max(y_max_for_ylim, y + 0.05)

    # leave a bit of headroom for stars
    ax.set_ylim(0, max(1.0, y_max_for_ylim))

    ax.legend(frameon=False)
    ax.set_title("Success probability by choice state\n(averaged over monkey×task)")

    plt.tight_layout()
    save_plot(fig, f"{save_prefix}_success_by_state", "Success by choice state (global)")
    plt.close(fig)

def plot_success_by_choice_state_global_final(
    grand_summary,
    behavioral_only=False,
    save_prefix="global_success_by_state"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon, ttest_rel

    rows = grand_summary.get("success_state_per_unit", [])
    if not rows:
        print("[plot_success_by_choice_state_global_final] No success_state_per_unit in grand_summary.")
        return

    states = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]
    state_labels = {
        "correct_choice": "Correct\nchoice",
        "neighbor_choice": "Neighbor\nchoice",
        "ambiguous_choice": "Ambiguous\nchoice",
        "wrong_choice": "Incorrect\nchoice",
    }

    vals_off = {s: [] for s in states}
    vals_on  = {s: [] for s in states}
    pvals    = {s: np.nan for s in states}
    test_used = {s: None for s in states}
    n_pairs = {s: 0 for s in states}

    # -------- collect paired values and run paired test per state --------
    for s in states:
        off_vals = []
        on_vals = []

        for r in rows:
            if r.get("state") != s:
                continue

            if behavioral_only:
                p_off = r.get("p_success_off_behavioral", np.nan)
                p_on  = r.get("p_success_on_behavioral", np.nan)
            else:
                p_off = r.get("p_success_off", np.nan)
                p_on  = r.get("p_success_on", np.nan)

            try:
                p_off = float(p_off)
                p_on  = float(p_on)
            except Exception:
                continue

            if np.isfinite(p_off) and np.isfinite(p_on):
                off_vals.append(p_off)
                on_vals.append(p_on)

        off_vals = np.asarray(off_vals, dtype=float)
        on_vals = np.asarray(on_vals, dtype=float)

        vals_off[s] = off_vals
        vals_on[s] = on_vals
        n_pairs[s] = len(off_vals)

        if len(off_vals) < 2:
            continue

        diffs = on_vals - off_vals

        # If all paired differences are exactly zero, no test is meaningful
        if np.allclose(diffs, 0, equal_nan=False):
            pvals[s] = 1.0
            test_used[s] = "all_equal"
            continue

        # Prefer Wilcoxon for paired comparisons; fall back to paired t-test
        try:
            _, p = wilcoxon(
                on_vals,
                off_vals,
                zero_method="wilcox",
                alternative="greater",
                method="exact"
            )
            pvals[s] = p
            test_used[s] = "wilcoxon_1sided_greater"
        except Exception:
            try:
                _, p = wilcoxon(
                    on_vals,
                    off_vals,
                    zero_method="wilcox",
                    alternative="greater"
                )
                pvals[s] = p
                test_used[s] = "wilcoxon_1sided_greater"
            except Exception:
                try:
                    _, p = ttest_rel(on_vals, off_vals, nan_policy="omit")
                    # convert two-sided paired t-test p to one-sided in the expected direction
                    mean_diff = np.nanmean(on_vals - off_vals)
                    if np.isfinite(p):
                        p = p / 2 if mean_diff > 0 else 1 - (p / 2)
                    pvals[s] = p
                    test_used[s] = "ttest_rel_1sided_greater"
                except Exception:
                    pvals[s] = np.nan
                    test_used[s] = None

    def mean_sem(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan
        m = np.mean(arr)
        sem = np.std(arr, ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan
        return m, sem

    means_off, sems_off, means_on, sems_on = [], [], [], []
    for s in states:
        m_off, se_off = mean_sem(vals_off[s])
        m_on,  se_on  = mean_sem(vals_on[s])
        means_off.append(m_off)
        sems_off.append(se_off)
        means_on.append(m_on)
        sems_on.append(se_on)

    x = np.arange(len(states))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width/2,
        means_off,
        width,
        yerr=sems_off,
        label="BCI-only",
        capsize=4,
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.bar(
        x + width/2,
        means_on,
        width,
        yerr=sems_on,
        label="Shared-control",
        capsize=4,
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([state_labels[s] for s in states], rotation=0)
    ax.set_ylabel("P(success | choice state)")
    ax.legend(frameon=False)

    def star_for_p(p):
        if not np.isfinite(p):
            return ""
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 5e-2:
            return "*"
        else:
            return ""

    # -------- add significance brackets --------
    y_max_for_ylim = 1.0
    for i, s in enumerate(states):
        p = pvals[s]
        star = star_for_p(p)
        if star == "":
            continue

        m_off = means_off[i]
        m_on  = means_on[i]
        se_off = 0.0 if np.isnan(sems_off[i]) else sems_off[i]
        se_on  = 0.0 if np.isnan(sems_on[i]) else sems_on[i]
        y_bar = max(m_off + se_off, m_on + se_on)

        gap = 0.03
        h = 0.02
        y = y_bar + gap

        x1 = x[i] - width / 2.0
        x2 = x[i] + width / 2.0

        ax.plot(
            [x1, x1, x2, x2],
            [y,  y + h, y + h, y],
            color="black",
            linewidth=1.0,
        )

        ax.text(
            (x1 + x2) / 2.0,
            y + h + 0.005,
            star,
            ha="center",
            va="bottom",
            fontsize=11,
        )

        y_max_for_ylim = max(y_max_for_ylim, y + h + 0.06)

    ax.set_ylim(0, max(1.05, y_max_for_ylim))

    ax.set_title(
        "Success by choice state"
        + ("\n(behavioural errors only)" if behavioral_only else "")
    )

    # optional: print stats in console
    print("\n[plot_success_by_choice_state_global_final] Paired tests per state:")
    for s in states:
        print(
            f"  {s}: n={n_pairs[s]}, test={test_used[s]}, "
            f"p={pvals[s]:.4g}" if np.isfinite(pvals[s]) else
            f"  {s}: n={n_pairs[s]}, test={test_used[s]}, p=nan"
        )

    plt.tight_layout()

    try:
        save_plot(fig, f"{save_prefix}_success_by_state", "Success by choice state (global)")
    except Exception:
        pass

    plt.show()
    plt.close(fig)

def plot_success_by_choice_state_split(
    grand_summary,
    behavioral_only=False,
    min_trials_per_cell=10,
    experiments_of_interest=None,  # e.g. ["AI Obstacle", "AI Appearing Obstacle"]
    monkeys_of_interest=None       # e.g. ["Monkey 1", "Monkey 3"]
):
    """
    Make success-by-choice-state barplots split:

      (1) Per experiment (task), pooled across monkeys.
      (2) Per monkey, pooled across experiments (tasks).

    Parameters
    ----------
    grand_summary : dict
        Output of analyze_trajectories(...). Must contain key
        'success_state_per_unit', a list of dicts with fields:
          - 'monkey', 'experiment', 'state'
          - p_success_off, p_success_on
          - p_success_off_behavioral, p_success_on_behavioral
          - n_off, n_on
          - n_off_behavioral, n_on_behavioral

    behavioral_only : bool
        If True, use the *_behavioral fields (i.e. only trials where failures
        are not stuck_obstacle / overshoot / etc.). If False, use all trials.

    min_trials_per_cell : int
        Require at least this many trials for BOTH AI-off and AI-on in a
        given (monkey × experiment × state) entry to include that unit.

    experiments_of_interest : list or None
        If provided, only these experiment names are plotted. Otherwise all
        unique experiments in the summary are used.

    monkeys_of_interest : list or None
        If provided, only these monkeys are plotted. Otherwise all unique
        monkeys in the summary are used.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from math import sqrt
    from scipy.stats import wilcoxon

    rows = grand_summary.get("success_state_per_unit", [])
    if not rows:
        print("[WARN] grand_summary['success_state_per_unit'] is empty.")
        return

    # ----- fixed state order and labels -----
    states = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]
    pretty = {
        "correct_choice":   "Correct\nchoice",
        "neighbor_choice":  "Neighbor\nchoice",
        "ambiguous_choice": "Ambiguous",
        "wrong_choice":     "Wrong\nchoice",
    }

    def _aggregate_rows(subrows):
        """
        Aggregate a subset of rows (e.g. all for one experiment, or all for
        one monkey) into per-state means, SEMs, and paired Wilcoxon p-values.
        """
        from collections import defaultdict

        per_state = {s: {"off": [], "on": [], "diffs": []} for s in states}

        for row in subrows:
            st = row.get("state")
            if st not in states:
                continue

            if behavioral_only:
                n_off = row.get("n_off_behavioral", 0) or 0
                n_on  = row.get("n_on_behavioral", 0) or 0
                p_off = row.get("p_success_off_behavioral", np.nan)
                p_on  = row.get("p_success_on_behavioral", np.nan)
            else:
                n_off = row.get("n_off", 0) or 0
                n_on  = row.get("n_on", 0) or 0
                p_off = row.get("p_success_off", np.nan)
                p_on  = row.get("p_success_on", np.nan)

            # skip if not enough trials or NaNs
            if (n_off < min_trials_per_cell) or (n_on < min_trials_per_cell):
                continue
            if not np.isfinite(p_off) or not np.isfinite(p_on):
                continue

            per_state[st]["off"].append(float(p_off))
            per_state[st]["on"].append(float(p_on))
            per_state[st]["diffs"].append(float(p_on - p_off))

        means_off, sem_off = [], []
        means_on, sem_on = [], []
        p_vals = []

        for st in states:
            off_arr = np.asarray(per_state[st]["off"], float)
            on_arr  = np.asarray(per_state[st]["on"], float)
            diff_arr = np.asarray(per_state[st]["diffs"], float)

            if off_arr.size == 0:
                means_off.append(np.nan)
                sem_off.append(np.nan)
                means_on.append(np.nan)
                sem_on.append(np.nan)
                p_vals.append(np.nan)
                continue

            means_off.append(float(off_arr.mean()))
            means_on.append(float(on_arr.mean()))

            n_units = off_arr.size
            if n_units > 1:
                sem_off.append(float(off_arr.std(ddof=1) / sqrt(n_units)))
                sem_on.append(float(on_arr.std(ddof=1) / sqrt(n_units)))
            else:
                sem_off.append(0.0)
                sem_on.append(0.0)

            if diff_arr.size >= 2 and np.any(diff_arr != 0):
                try:
                    p_state = wilcoxon(diff_arr, alternative="two-sided").pvalue
                except Exception:
                    p_state = np.nan
            else:
                p_state = np.nan
            p_vals.append(float(p_state) if p_state is not None else np.nan)

        return (
            np.array(means_off, float),
            np.array(sem_off, float),
            np.array(means_on, float),
            np.array(sem_on, float),
            np.array(p_vals, float),
        )

    def _plot_bars(title, fname, means_off, sem_off, means_on, sem_on, p_vals):
        import numpy as np
        import matplotlib.pyplot as plt

        # If everything is NaN, skip plotting
        if (not np.isfinite(means_off).any()) and (not np.isfinite(means_on).any()):
            print(f"[INFO] No valid data for {title}, skipping.")
            return

        x = np.arange(len(states))
        width = 0.35
        color_off = "#d6604d"  # red-ish
        color_on  = "#5aae61"  # green-ish

        fig, ax = plt.subplots(figsize=(6, 4))

        bars_off = ax.bar(
            x - width / 2,
            means_off,
            width,
            yerr=sem_off,
            color=color_off,
            label="AI Off",
            capsize=4,
            alpha=0.9,
        )
        bars_on = ax.bar(
            x + width / 2,
            means_on,
            width,
            yerr=sem_on,
            color=color_on,
            label="AI On",
            capsize=4,
            alpha=0.9,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([pretty[s] for s in states])
        ax.set_ylabel("P(success | choice state)")
        ax.set_ylim(0.0, 1.05)

        if behavioral_only:
            full_title = f"{title}\n(behavioural errors only)"
        else:
            full_title = f"{title}\n(all errors)"
        ax.set_title(full_title)

        ax.legend(frameon=True)
        ax.grid(axis="y", alpha=0.3)

        # significance markers (Wilcoxon ON vs OFF across units)
        for i, p in enumerate(p_vals):
            if not np.isfinite(p) or p >= 0.05:
                continue
            y_max = np.nanmax([
                means_off[i] + (sem_off[i] if np.isfinite(sem_off[i]) else 0.0),
                means_on[i]  + (sem_on[i]  if np.isfinite(sem_on[i])  else 0.0),
            ])
            y_star = y_max + 0.04
            ax.plot([x[i] - width / 2, x[i] + width / 2],
                    [y_star, y_star],
                    color="k",
                    linewidth=1.2)
            ax.text(x[i], y_star + 0.01, "*", ha="center", va="bottom", fontsize=12)

        plt.tight_layout()
        try:
            save_plot(fig, fname, subfolder="Summary")
        except Exception:
            pass
        plt.show()
        plt.close(fig)

    # ----- choose experiments / monkeys -----
    all_exps = sorted({r.get("experiment") for r in rows})
    all_monkeys = sorted({r.get("monkey") for r in rows})

    if experiments_of_interest is None:
        experiments = all_exps
    else:
        experiments = [e for e in experiments_of_interest if e in all_exps]

    if monkeys_of_interest is None:
        monkeys = all_monkeys
    else:
        monkeys = [m for m in monkeys_of_interest if m in all_monkeys]

    # ----- 1) per experiment (over monkeys) -----
    for exp in experiments:
        subrows = [r for r in rows if r.get("experiment") == exp]
        if not subrows:
            continue
        means_off, sem_off, means_on, sem_on, p_vals = _aggregate_rows(subrows)
        print(f"\n[Per-experiment] {exp}")
        for st, mo, mn, p in zip(states, means_off, means_on, p_vals):
            if np.isfinite(mo) and np.isfinite(mn):
                print(f"  {st:15s}: off={mo:.3f}, on={mn:.3f}, p={p if np.isfinite(p) else np.nan}")
        _plot_bars(
            title=f"{exp} – success by choice state (over monkeys)",
            fname=f"success_by_choice_state_{exp.replace(' ', '_')}",
            means_off=means_off,
            sem_off=sem_off,
            means_on=means_on,
            sem_on=sem_on,
            p_vals=p_vals,
        )

    # ----- 2) per monkey (over experiments) -----
    for m in monkeys:
        subrows = [r for r in rows if r.get("monkey") == m]
        if not subrows:
            continue
        means_off, sem_off, means_on, sem_on, p_vals = _aggregate_rows(subrows)
        print(f"\n[Per-monkey] {m}")
        for st, mo, mn, p in zip(states, means_off, means_on, p_vals):
            if np.isfinite(mo) and np.isfinite(mn):
                print(f"  {st:15s}: off={mo:.3f}, on={mn:.3f}, p={p if np.isfinite(p) else np.nan}")
        _plot_bars(
            title=f"{m} – success by choice state (over tasks)",
            fname=f"success_by_choice_state_{m.replace(' ', '_')}",
            means_off=means_off,
            sem_off=sem_off,
            means_on=means_on,
            sem_on=sem_on,
            p_vals=p_vals,
        )

from scipy.spatial import ConvexHull

def plot_hull_overlay(df, latent_col="latents", K=150, dims=(0,1,2), title="Target-centroid hulls"):
    def centroids(status):
        C = []
        for t in sorted(df["target"].unique()):
            trials = df[(df["target"]==t) & (df["ai_status"]==status)][latent_col].tolist()
            if len(trials) == 0: continue
            M,_ = _mean_traj(np.stack(trials), K=K)
            C.append(M.mean(axis=0))
        return np.stack(C) if len(C)>0 else None

    C_on  = centroids(1)
    C_off = centroids(0)
    # reduce to 3D for plotting
    C_all = np.vstack([C for C in [C_on, C_off] if C is not None])
    B = _pca_reduce(C_all, k=3)
    split = len(C_on) if C_on is not None else 0
    Bon, Boff = B[:split], B[split:]

    fig = plt.figure(figsize=(6.5,5.5))
    ax = fig.add_subplot(111, projection="3d")
    if Bon is not None and Bon.shape[0] >= 4:
        hull_on = ConvexHull(Bon)
        for s in hull_on.simplices:
            ax.plot_trisurf(Bon[s,0], Bon[s,1], Bon[s,2], alpha=0.25)
        ax.scatter(Bon[:,0], Bon[:,1], Bon[:,2], s=40, label="ON")
    if Boff is not None and Boff.shape[0] >= 4:
        hull_off = ConvexHull(Boff)
        for s in hull_off.simplices:
            ax.plot_trisurf(Boff[s,0], Boff[s,1], Boff[s,2], alpha=0.25)
        ax.scatter(Boff[:,0], Boff[:,1], Boff[:,2], s=40, label="OFF")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.show()
    return fig

def plot_pairwise_mean_distance_bars(results_on, results_off,
                                     title="Pairwise separation (mean distance)"):
    def _key(r):
        return (r.get("cat1"), r.get("cat2"))

    def _md_or_nan(r):
        if r is None or ("error" in r):
            return np.nan
        v = r.get("mean_distance", np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    # Preserve order: first OFF entries, then any ON-only pairs
    keys, seen = [], set()
    for r in (results_off or []):
        k = _key(r); 
        if k not in seen: keys.append(k); seen.add(k)
    for r in (results_on or []):
        k = _key(r); 
        if k not in seen: keys.append(k); seen.add(k)

    names    = [f"{k[0]} vs {k[1]}" for k in keys]
    off_vals = np.array([_md_or_nan(next((r for r in results_off if _key(r)==k), None))
                         for k in keys], dtype=float)
    on_vals  = np.array([_md_or_nan(next((r for r in results_on  if _key(r)==k), None))
                         for k in keys], dtype=float)

    # Figure
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 0.4*len(names)+4), 4.5))
    ax.bar(x-w/2, off_vals, width=w, label="OFF")
    ax.bar(x+w/2, on_vals,  width=w, label="ON")
    ax.set_ylabel("Mean trajectory distance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")

    # Safe ratio (ignore NaNs); handle zero/NaN denominator
    on_m  = np.nanmean(on_vals)  if np.isfinite(on_vals).any()  else np.nan
    off_m = np.nanmean(off_vals) if np.isfinite(off_vals).any() else np.nan
    if not np.isfinite(on_m) or not np.isfinite(off_m):
        ratio_txt = "n/a"
    elif off_m == 0:
        ratio_txt = "∞ (OFF mean=0)"
    else:
        ratio_txt = f"{on_m/off_m:.2f}"

    n_on  = int(np.isfinite(on_vals).sum())
    n_off = int(np.isfinite(off_vals).sum())
    ax.set_title(f"{title}\nCompression ratio ON/OFF = {ratio_txt} "
                 f"(n_on={n_on}, n_off={n_off})")

    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

def _as_float_timestamp(x):
    # handles scalars, 0-length, 1-length arrays, datetime objects
    if isinstance(x, (list, tuple, np.ndarray)):
        a = np.asarray(x).ravel()
        x = a[0] if a.size else np.nan
    if hasattr(x, "timestamp"):
        return float(x.timestamp())
    try:
        return float(x)
    except Exception:
        return np.nan

def _xz(vec):
    a = np.asarray(vec, float).ravel()
    if a.size >= 3:  # (x, y, z) or (vx, vy, vz)
        return float(a[0]), float(a[2])
    if a.size == 2:  # already (x, z) or (vx, vz)
        return float(a[0]), float(a[1])
    return np.nan, np.nan

# def plot_bci_vs_ai_near_target(trial, title_prefix="", window_size=4.2, pad=4.0,
#                                arrow_every=2, vmin=1e-3, arrow_scale=0.8):
#     """
#     Zoomed plot near the target showing:
#       - recorded trajectory (from avatarTrajectory)
#       - BCI (Input) arrows
#       - AI (Output) arrows
#       - ideal arrows toward the target
#     """
#     # ---- trajectory from avatarTrajectory ----
#     traj = trial.get('avatarTrajectory') if isinstance(trial, dict) else getattr(trial, 'avatarTrajectory', None)
#     if not traj:
#         print("[warn] trial has no 'avatarTrajectory'")
#         return

#     t_pose = np.asarray(traj['time'], float)
#     X = np.asarray(traj['x'], float)
#     Z = np.asarray(traj['z'], float)
#     if X.size < 2 or Z.size < 2:
#         print("[warn] trajectory too short")
#         return

#     # # ---- target center + plot window ----
#     # tx, tz = _xz(trial.targetPosition)
#     # half = window_size/2.0
#     # xlim = (tx - half - pad, tx + half + pad)
#     # zlim = (tz - half - pad, tz + half + pad)

#     # ---- set view window ----
#     tx, tz = _xz(trial.targetPosition)
#     half = window_size/2.0
#     extent = "tail"
#     tail_frac = 0.5
#     if extent == "full":
#         xlim = (np.nanmin(X) - pad, np.nanmax(X) + pad)
#         zlim = (np.nanmin(Z) - pad, np.nanmax(Z) + pad)
#     elif extent == "tail" and tail_frac:
#         k0 = max(0, int(len(X) * (1 - float(tail_frac))))
#         xlim = (np.nanmin(X[k0:]) - pad, np.nanmax(X[k0:]) + pad)
#         zlim = (np.nanmin(Z[k0:]) - pad, np.nanmax(Z[k0:]) + pad)
#     else:
#         tx, tz = _xz(trial.targetPosition)
#         half = window_size / 2.0
#         xlim = (tx - half - pad, tx + half + pad)
#         zlim = (tz - half - pad, tz + half + pad)

#     # ---- BCI/AI samples from aiVelocities, align to nearest pose sample ----
#     recs = trial.aiVelocities
#     if not recs:
#         print("[warn] trial has no 'aiVelocities'")
#         return

#     ts = np.array([_as_float_timestamp(r.get('OutputTimestamp')) for r in recs], float)
#     In = np.array([_xz(r.get('Input'))  for r in recs], float)   # BCI velocity (vx, vz)
#     Out= np.array([_xz(r.get('Output')) for r in recs], float)   # AI-adjusted velocity (vx, vz)

#     # nearest neighbor in time from velocity samples to pose samples
#     idx = np.searchsorted(t_pose, ts, side='left')
#     idx = np.clip(idx, 0, len(t_pose)-1)

#     # keep only arrows whose anchor point falls inside the zoom window
#     anchors_x = X[idx]; anchors_z = Z[idx]
#     inside = (anchors_x >= xlim[0]) & (anchors_x <= xlim[1]) & \
#              (anchors_z >= zlim[0]) & (anchors_z <= zlim[1])

#     # speed gate to avoid drawing zero-length arrows
#     speed_in  = np.linalg.norm(In,  axis=1)
#     speed_out = np.linalg.norm(Out, axis=1)
#     ok = inside & np.isfinite(speed_in) & np.isfinite(speed_out) & (speed_in >= vmin) & (speed_out >= vmin)

#     if not np.any(ok):
#         print("[info] no valid BCI/AI samples in the target zoom window for this trial")
#         return

#     anchors = np.column_stack([anchors_x[ok], anchors_z[ok]])
#     vin = In[ok]; vout = Out[ok]

#     # ideal direction to target from each anchor
#     to_targ = np.column_stack([tx - anchors[:,0], tz - anchors[:,1]])
#     nt = np.linalg.norm(to_targ, axis=1, keepdims=True) + 1e-12
#     to_targ_unit = to_targ / nt

#     # normalize arrows for visualization
#     def _unit(v):
#         n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
#         return v / n

#     u_in  = _unit(vin)  * arrow_scale
#     u_out = _unit(vout) * arrow_scale
#     u_tar = to_targ_unit * arrow_scale

#     # ---- plot ----
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_aspect('equal', adjustable='datalim')
#     ax.set_title(f"{title_prefix} Trial {trial.trial} — near target")

#     # trajectory (full), then zoom
#     ax.plot(X, Z, color='#999999', lw=1.5, alpha=0.7)

#     # target window (outer 4.2 and inner 1.5 square)
#     ax.add_patch(plt.Rectangle((tx - half, tz - half), window_size, window_size,
#                                fill=False, edgecolor='g', lw=2.0))
#     ax.add_patch(plt.Rectangle((tx - 0.75, tz - 0.75), 1.5, 1.5,
#                                fill=True, color=(0,1,0,0.15), lw=0))

#     # BCI (orange), AI (teal), toward-target (gray) arrows
#     step = max(1, int(arrow_every))
#     for k in range(0, anchors.shape[0], step):
#         x0, z0 = anchors[k]
#         ax.arrow(x0, z0, u_in[k,0],  u_in[k,1],  head_width=0.12, length_includes_head=True, color='#F46E49', alpha=0.95)
#         ax.arrow(x0, z0, u_out[k,0], u_out[k,1], head_width=0.12, length_includes_head=True, color='#2AA198', alpha=0.95)
#         ax.arrow(x0, z0, u_tar[k,0], u_tar[k,1], head_width=0.10, length_includes_head=True, color='#666666', alpha=0.6)

#     ax.set_xlim(*xlim); ax.set_ylim(*zlim)
#     ax.set_xlabel('X'); ax.set_ylabel('Z')
#     from matplotlib.lines import Line2D
#     ax.legend(handles=[
#         Line2D([0],[0], color='#F46E49', lw=2, label='BCI (Input)'),
#         Line2D([0],[0], color='#2AA198', lw=2, label='AI (Output)'),
#         Line2D([0],[0], color='#666666', lw=2, label='Toward target'),
#         Line2D([0],[0], color='#999999', lw=2, label='Trajectory'),
#     ], loc='best', frameon=True)
#     ax.grid(True, alpha=0.2)
#     plt.tight_layout()
#     plt.show()

#     # quick numeric readout: is BCI pointing to target?
#     cos_in  = np.einsum('ij,ij->i', _unit(vin), to_targ_unit)
#     cos_out = np.einsum('ij,ij->i', _unit(vout), to_targ_unit)
#     ang_in  = np.degrees(np.arccos(np.clip(cos_in,  -1, 1)))
#     ang_out = np.degrees(np.arccos(np.clip(cos_out, -1, 1)))
#     print(f"[trial {trial.trial}] mean angle-to-target:  BCI={np.nanmean(ang_in):.1f}°,  AI={np.nanmean(ang_out):.1f}°  (lower is better)")

def plot_bci_vs_ai_near_target(
    trial, title_prefix="", window_size=4.2, pad=2.0,
    arrow_every=2, vmin=1e-3, arrow_scale=0.8,
    extent="target",          # 'target' | 'full' | 'tail'
    tail_pts=250,             # if extent='tail', how many last trajectory points
    draw_where="axes"         # 'window' | 'axes' | 'all'
):
    # ---- trajectory ----
    traj = trial.get('avatarTrajectory') if isinstance(trial, dict) else getattr(trial, 'avatarTrajectory', None)
    if not traj: 
        print("[warn] trial has no 'avatarTrajectory'"); return
    t_pose = np.asarray(traj['time'], float)
    X = np.asarray(traj['x'], float); Z = np.asarray(traj['z'], float)
    if X.size < 2 or Z.size < 2:
        print("[warn] trajectory too short"); return

    # ---- target (box stays fixed size) ----
    tx, tz = _xz(trial.targetPosition)
    half = window_size/2.0

    # ---- axes limits (what you see on screen) ----
    if extent == "target":
        xlim = (tx - half - pad, tx + half + pad)
        zlim = (tz - half - pad, tz + half + pad)
        traj_slice = slice(None)
    elif extent == "full":
        xlim = (np.nanmin(X)-pad, np.nanmax(X)+pad)
        zlim = (np.nanmin(Z)-pad, np.nanmax(Z)+pad)
        traj_slice = slice(None)
    elif extent == "tail":
        k0 = max(0, len(X)-int(tail_pts))
        x_tail = X[k0:]; z_tail = Z[k0:]
        xlim = (np.nanmin(x_tail)-pad, np.nanmax(x_tail)+pad)
        zlim = (np.nanmin(z_tail)-pad, np.nanmax(z_tail)+pad)
        traj_slice = slice(k0, None)
    else:
        raise ValueError("extent must be 'target'|'full'|'tail'")

    # ---- BCI/AI samples ----
    recs = trial.aiVelocities
    if not recs: 
        print("[warn] trial has no 'aiVelocities'"); return
    ts  = np.array([_as_float_timestamp(r.get('OutputTimestamp')) for r in recs], float)
    In  = np.array([_xz(r.get('Input'))  for r in recs], float)
    Out = np.array([_xz(r.get('Output')) for r in recs], float)

    # match velocity samples to nearest pose sample (for anchor positions)
    idx = np.searchsorted(t_pose, ts, side='left')
    idx = np.clip(idx, 0, len(t_pose)-1)
    anchors_x = X[idx]; anchors_z = Z[idx]

    # spatial masks for where arrows are allowed
    in_target_window = (anchors_x >= tx - half) & (anchors_x <= tx + half) & \
                       (anchors_z >= tz - half) & (anchors_z <= tz + half)
    in_axes = (anchors_x >= xlim[0]) & (anchors_x <= xlim[1]) & \
              (anchors_z >= zlim[0]) & (anchors_z <= zlim[1])

    if draw_where == "window":
        place_mask = in_target_window
    elif draw_where == "axes":
        place_mask = in_axes
    elif draw_where == "all":
        place_mask = np.ones_like(in_axes, dtype=bool)
    else:
        raise ValueError("draw_where must be 'window'|'axes'|'all'")

    # speed gate (relax if you want the last slow arrows)
    speed_in  = np.linalg.norm(In,  axis=1)
    speed_out = np.linalg.norm(Out, axis=1)
    speed_mask = (np.isfinite(speed_in) & np.isfinite(speed_out) &
                  ((speed_in >= vmin) | (speed_out >= vmin)))  # OR keeps slow tails

    ok = place_mask & speed_mask
    if not np.any(ok):
        print("[info] no BCI/AI samples to draw under current filters"); return

    anchors = np.column_stack([anchors_x[ok], anchors_z[ok]])
    vin = In[ok]; vout = Out[ok]

    # toward-target unit vectors
    to_targ = np.column_stack([tx - anchors[:,0], tz - anchors[:,1]])
    nt = np.linalg.norm(to_targ, axis=1, keepdims=True) + 1e-12
    to_targ_unit = to_targ / nt

    def _unit(v):
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / n
    u_in  = _unit(vin)  * arrow_scale
    u_out = _unit(vout) * arrow_scale
    u_tar = to_targ_unit * arrow_scale

    print("N pose:", len(trial.avatarTrajectory['time']))
    print("N aiVel:", len(trial.aiVelocities))

    # last timestamps
    def _ts(step):  
        ti = _as_float_timestamp(step.get('InputTimestamp'))
        to = _as_float_timestamp(step.get('OutputTimestamp'))
        return ti if np.isfinite(ti) else to

    ts = np.array([_ts(s) for s in trial.aiVelocities], float)
    print("pose last t:", float(trial.avatarTrajectory['time'][-1]))
    print("aiVel last t:", np.nanmax(ts))

    # how many final velocity samples were dropped by the speed gate?
    In  = np.array([_xz(s.get('Input'))  for s in trial.aiVelocities], float)
    Out = np.array([_xz(s.get('Output')) for s in trial.aiVelocities], float)
    spd_in, spd_out = np.linalg.norm(In,1), np.linalg.norm(Out,1)
    print("dropped by vmin:", np.sum((spd_in < 1e-3) & (spd_out < 1e-3)))
    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(f"{title_prefix} Trial {trial.trial} — near target")

    ax.plot(X[traj_slice], Z[traj_slice], color='#999999', lw=1.5, alpha=0.8)

    # draw fixed-size target box (doesn't change with extent)
    ax.add_patch(plt.Rectangle((tx - half, tz - half), window_size, window_size,
                               fill=False, edgecolor='g', lw=2.0))
    ax.add_patch(plt.Rectangle((tx - 0.75, tz - 0.75), 1.5, 1.5,
                               fill=True, color=(0,1,0,0.15), lw=0))

    step = max(1, int(arrow_every))
    for k in range(0, anchors.shape[0], step):
        x0, z0 = anchors[k]
        ax.arrow(x0, z0, u_in[k,0],  u_in[k,1],  head_width=0.12, length_includes_head=True, color='#F46E49', alpha=0.95)
        ax.arrow(x0, z0, u_out[k,0], u_out[k,1], head_width=0.12, length_includes_head=True, color='#2AA198', alpha=0.95)
        ax.arrow(x0, z0, u_tar[k,0], u_tar[k,1], head_width=0.10, length_includes_head=True, color='#666666', alpha=0.6)

    ax.set_xlim(*xlim); ax.set_ylim(*zlim)
    ax.set_xlabel('X'); ax.set_ylabel('Z')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='#F46E49', lw=2, label='BCI (Input)'),
        Line2D([0],[0], color='#2AA198', lw=2, label='AI (Output)'),
        Line2D([0],[0], color='#666666', lw=2, label='Toward target'),
        Line2D([0],[0], color='#999999', lw=2, label='Trajectory'),
    ], loc='best', frameon=True)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()

# ----------------- helpers -----------------
def _to_xz(vec):
    if vec is None: return None
    if isinstance(vec, dict):
        if "x" in vec and "z" in vec:   return float(vec["x"]), float(vec["z"])
        if "vx" in vec and "vz" in vec: return float(vec["vx"]), float(vec["vz"])
    a = np.asarray(vec, float).ravel()
    if a.size >= 3: return float(a[0]), float(a[2])  # [x,y,z] -> (x,z)
    if a.size == 2: return float(a[0]), float(a[1])  # [x,z]
    return None

def _angle_deg(u, v, eps=1e-12):
    if u is None or v is None: return np.nan
    ux, uz = u; vx, vz = v
    nu = np.hypot(ux, uz); nv = np.hypot(vx, vz)
    if not np.isfinite(nu) or not np.isfinite(nv) or nu < eps or nv < eps: return np.nan
    c = (ux*vx + uz*vz) / (nu*nv)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def _as_float_timestamp(x):
    if x is None: return np.nan
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).ravel()
        x = arr[0] if arr.size else np.nan
    if hasattr(x, "timestamp"): return float(x.timestamp())
    try: return float(x)
    except Exception: return np.nan

def _norm_times(ts, t0=None, t1=None):
    ts = np.asarray(ts, float)
    if ts.size == 0: return ts
    if t0 is None or not np.isfinite(t0): t0 = np.nanmin(ts)
    if t1 is None or not np.isfinite(t1): t1 = np.nanmax(ts)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.linspace(0, 1, ts.size)
    return (ts - t0) / (t1 - t0)

# ----------------- core extraction -----------------
def trial_angles_excl_takeover(trial):
    """Return (tnorm, angles) for one trial, EXCLUDING [aiControlOn, aiControlOff]."""
    aiV = trial.get("aiVelocities") if isinstance(trial, dict) else getattr(trial, "aiVelocities", None)
    if not aiV:
        return np.array([], float), np.array([], float)

    # timestamps & angles
    ts  = np.array([_as_float_timestamp(s.get("InputTimestamp")) for s in aiV], float)
    ang = np.array([_angle_deg(_to_xz(s.get("Input")), _to_xz(s.get("Output"))) for s in aiV], float)

    # mask: valid angles
    mask = np.isfinite(ang)

    # exclude takeover window if present
    on  = trial.get("aiControlOn")  if isinstance(trial, dict) else getattr(trial, "aiControlOn",  None)
    off = trial.get("aiControlOff") if isinstance(trial, dict) else getattr(trial, "aiControlOff", None)
    on, off = _as_float_timestamp(on), _as_float_timestamp(off)
    if np.isfinite(on) and np.isfinite(off) and off >= on:
        mask &= ~((ts >= on) & (ts <= off))

    ts = ts[mask]
    ang = ang[mask]

    # normalize time over whole trial span if available
    t0 = trial.get("start") if isinstance(trial, dict) else getattr(trial, "start", None)
    t1 = trial.get("stop")  if isinstance(trial, dict) else getattr(trial, "stop",  None)
    t0 = _as_float_timestamp(t0); t1 = _as_float_timestamp(t1)
    tnorm = _norm_times(ts, t0=t0, t1=t1)
    return tnorm, ang

# ----------------- plotting -----------------
def plot_overshoot_timecourse(overshoot_trials, n_bins=100, last_half=False):
    """Mean ± SEM angle vs normalized time for overshoot trials, excluding takeover."""
    bins = [[] for _ in range(n_bins)]
    for tr in overshoot_trials:
        tnorm, ang = trial_angles_excl_takeover(tr)
        if tnorm.size == 0: 
            continue
        if last_half:
            keep = (tnorm >= 0.5)
            tnorm, ang = tnorm[keep], ang[keep]
        if tnorm.size == 0:
            continue
        idx = np.clip((tnorm * (n_bins - 1)).astype(int), 0, n_bins - 1)
        for a, i in zip(ang, idx):
            bins[i].append(a)

    means, sems = [], []
    for b in bins:
        if len(b) == 0:
            means.append(np.nan); sems.append(np.nan)
        else:
            arr = np.asarray(b, float)
            means.append(np.nanmean(arr))
            sems.append(np.nanstd(arr, ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else 0.0)

    x = np.linspace(0, 1, n_bins)
    plt.figure(figsize=(10, 5))
    plt.plot(x, means, label="Angle (overshoot, excl. takeover)")
    lo = np.array(means) - np.array(sems)
    hi = np.array(means) + np.array(sems)
    plt.fill_between(x, lo, hi, alpha=0.2)
    plt.xlabel("Normalized Time (full trial)")
    plt.ylabel("Angle (degrees)")
    plt.title("BCI–AI Angle Over Time (Overshoot Trials, Excluding Takeover)" + (" — Last Half" if last_half else ""))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_overshoot_hist_last_half(overshoot_trials, bins=40):
    """Histogram of per-trial mean angles in the last half (excluding takeover)."""
    vals = []
    for tr in overshoot_trials:
        tnorm, ang = trial_angles_excl_takeover(tr)
        keep = (tnorm >= 0.5)
        if np.any(keep):
            vals.append(float(np.nanmean(ang[keep])))
    vals = np.array(vals, float)
    plt.figure(figsize=(8, 5))
    plt.hist(vals[np.isfinite(vals)], bins=bins)
    plt.xlabel("Per-trial mean angle (deg)")
    plt.ylabel("Count")
    plt.title("Overshoot: Per-trial Mean Angle (Last Half, Excluding Takeover)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"n trials plotted: {np.sum(np.isfinite(vals))}")
    if np.isfinite(vals).any():
        print(f"Mean={np.nanmean(vals):.2f}°, Median={np.nanmedian(vals):.2f}°, SD={np.nanstd(vals, ddof=1):.2f}°")

def plot_pie_chart_failure_modes(failure_reason_counts, ai_status, monkey, experiment):
    # Group rare categories into "Other"
    grouped_counts = {
        'ambiguous_choice': failure_reason_counts['ambiguous_choice'],
        'wrong_target_choice': failure_reason_counts['wrong_target_choice'],
        'ai_bci_failure': failure_reason_counts['ai_bci_failure_any'],
        'neighbor_choice': failure_reason_counts['neighbor_choice'],
        'stuck_obstacle': failure_reason_counts['stuck_obstacle'],
        'overshoot': failure_reason_counts['overshoot'],
        'other': (
            failure_reason_counts['other']
            + failure_reason_counts['not_long_enough']
            + failure_reason_counts['not_close_to_true_z']
        )
    }

    total_incorrect = sum(grouped_counts.values())
    if total_incorrect > 0:
        labels = ['Ambiguous choice', 'Wrong choice', 'Neighbor choice', 'Stuck at obstacle', 'Overshoot', 'Other']
        colors = {
            'Ambiguous choice': '#ff7f0e',     # orange
            'Wrong choice': "#f9fd01",     # yellow
            'Neighbor choice': "#fd017f",
            'Stuck at obstacle': '#2ca02c',# green
            'Overshoot': '#9467bd',        # purple
            'Other': '#7f7f7f',            # gray
        }
        sizes = [
            grouped_counts['ambiguous_choice'],
            grouped_counts['wrong_target_choice'],
            grouped_counts['neighbor_choice'],
            grouped_counts['stuck_obstacle'],
            grouped_counts['overshoot'],
            grouped_counts['other']
        ]

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            autopct='%1.1f%%',
            startangle=90,
            colors=[colors[l] for l in labels],
            textprops={'color': 'white'},
            pctdistance=1.15
        )

        # Remove labels from pie and use legend instead
        for t in texts:
            t.set_text("")

        ax.legend(
            wedges,
            [f"{lab} ({np.floor((np.array(n)/total_incorrect) * 100 + 0.5).astype(int)}%)" for lab, n in zip(labels, sizes)],
            title="Failure Mode",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )

        ax.set_title(f'Failure Modes in {ai_status} Incorrect Trials\n{monkey} – {experiment}')
        ax.axis('equal')
        plt.tight_layout()
        plt.show()
        save_plot(fig, f"piechart_{monkey}_{experiment}_{ai_status}", "Failure mode")
        plt.close()

        print(f"\n[Results: Failure modes in {ai_status} incorrect trials]")
        for lab, n in zip(labels, sizes):
            print(f"{lab}: {n}/{total_incorrect} ({100*n/total_incorrect:.1f}%)")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def plot_success_by_choice_state(
    success_by_choice_AI_OFF,
    total_by_choice_AI_OFF,
    success_by_choice_AI_ON,
    total_by_choice_AI_ON,
    monkey,
    experiment,
):
    """
    Bar plot of success probability per choice state
    for AI OFF vs AI ON.
    """

    categories = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]
    nice_labels = {
        "correct_choice":   "Correct choice",
        "neighbor_choice":  "Neighbor choice",
        "ambiguous_choice": "Ambiguous choice",
        "wrong_choice":     "Wrong choice",
    }

    x = np.arange(len(categories))
    width = 0.35

    rates_off = []
    ci_off    = []
    rates_on  = []
    ci_on     = []

    for cat in categories:
        # --- No-AI ---
        tot_off = total_by_choice_AI_OFF.get(cat, 0)
        suc_off = success_by_choice_AI_OFF.get(cat, 0)
        if tot_off > 0:
            p_off = suc_off / tot_off
            se_off = np.sqrt(p_off * (1.0 - p_off) / tot_off)
            rates_off.append(p_off)
            ci_off.append(1.96 * se_off)   # ~95% CI (normal approx)
        else:
            rates_off.append(np.nan)
            ci_off.append(0.0)

        # --- AI ---
        tot_on = total_by_choice_AI_ON.get(cat, 0)
        suc_on = success_by_choice_AI_ON.get(cat, 0)
        if tot_on > 0:
            p_on = suc_on / tot_on
            se_on = np.sqrt(p_on * (1.0 - p_on) / tot_on)
            rates_on.append(p_on)
            ci_on.append(1.96 * se_on)
        else:
            rates_on.append(np.nan)
            ci_on.append(0.0)

    rates_off = np.array(rates_off, float)
    rates_on  = np.array(rates_on, float)
    ci_off    = np.array(ci_off, float)
    ci_on     = np.array(ci_on, float)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Bars: No-AI (light gray) and AI (blue)
    off_bars = ax.bar(
        x - width/2,
        rates_off,
        width,
        yerr=ci_off,
        capsize=4,
        label="No AI",
        alpha=0.6,
        color="lightgray",
        edgecolor="black"
    )

    on_bars = ax.bar(
        x + width/2,
        rates_on,
        width,
        yerr=ci_on,
        capsize=4,
        label="AI",
        alpha=0.9,
        color="#1f77b4",
        edgecolor="black"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([nice_labels[c] for c in categories], rotation=20, ha="right")
    ax.set_ylabel("Success probability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{experiment} – {monkey}\nSuccess by decoder-inferred choice state")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    # If you already have a save_plot helper:
    try:
        save_plot(fig, f"success_by_choice_{monkey}_{experiment}", "Success by choice state")
    except NameError:
        pass  # or plt.savefig("...")

    plt.show()
    plt.close(fig)

def plot_failure_mode_counts(
    failure_reason_counts_AI_OFF,
    failure_reason_counts_AI_ON,
    monkey,
    experiment,
    total_trials_no_ai,
    total_trials_ai,
):
    """
    Barplot of failure modes + CORRECT as PERCENTAGE OF ALL TRIALS per condition.
    total_trials_no_ai / total_trials_ai should be:
        (# correct + # incorrect) for each condition.
    """

    # Helper to group counts into bins (INCORRECT trials only)
    def grouped(d):
        return {
            'stuck_obstacle': int(d.get('stuck_obstacle', 0)),
            'ambiguous_choice':   int(d.get('ambiguous_choice', 0)),
            'wrong_choice': int(d.get('wrong_target_choice', 0)),
            'neighbor_choice': int(d.get('neighbor_choice', 0)),
            'overshoot':      int(d.get('overshoot', 0)),
            'other': (
                int(d.get('other', 0))
                + int(d.get('not_long_enough', 0))
                + int(d.get('not_close_to_true_z', 0))
            )
        }

    off = grouped(failure_reason_counts_AI_OFF)
    on  = grouped(failure_reason_counts_AI_ON)

    # keys for the failure modes (we'll prepend "correct" later)
    fail_order = [
        ('stuck_obstacle',     'Stuck obstacle'),
        ('ambiguous_choice',       'Ambiguous choice'),
        ('wrong_choice',    'Wrong choice'),
        ('neighbor_choice',     'Neighbor choice'),
        ('overshoot',          'Overshoot'),
        ('other',              'Other'),
    ]

    # Raw failure counts (per mode)
    no_ai_fail_counts = np.array([off[k] for k, _ in fail_order], dtype=float)
    ai_fail_counts    = np.array([on[k]  for k, _ in fail_order], dtype=float)

    # Totals of incorrect trials (sanity check)
    no_ai_fail_total = int(no_ai_fail_counts.sum())
    ai_fail_total    = int(ai_fail_counts.sum())

    # Compute correct trials from totals
    correct_no_ai = max(0, int(total_trials_no_ai - no_ai_fail_total))
    correct_ai    = max(0, int(total_trials_ai    - ai_fail_total))

    print(f"[INFO] Nb of NO AI trials = {total_trials_no_ai} "
          f"(correct={correct_no_ai}, incorrect={no_ai_fail_total})")
    print(f"[INFO] Nb of AI trials    = {total_trials_ai} "
          f"(correct={correct_ai}, incorrect={ai_fail_total})")

    if total_trials_no_ai <= 0 or total_trials_ai <= 0:
        raise ValueError("total_trials_no_ai and total_trials_ai must be > 0")

    # Append "Correct" as first category, then the failure modes
    labels = ['Success'] + [lab for _, lab in fail_order]

    no_ai_counts_all = np.concatenate(([correct_no_ai], no_ai_fail_counts))
    ai_counts_all    = np.concatenate(([correct_ai],    ai_fail_counts))

    # Convert to percentage of ALL TRIALS
    no_ai_pct = 100.0 * no_ai_counts_all / float(total_trials_no_ai)
    ai_pct    = 100.0 * ai_counts_all    / float(total_trials_ai)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(
        x - width/2,
        no_ai_pct,
        width,
        label=f'No AI (n={total_trials_no_ai})',
        color='#f28e8c'
    )
    b2 = ax.bar(
        x + width/2,
        ai_pct,
        width,
        label=f'AI (n={total_trials_ai})',
        color='#53b26b'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Percentage of all trials (%)')
    ax.set_title(f'Outcome and failure modes per trial: No AI vs AI\n{monkey} – {experiment}')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # annotate bars with percentages (one decimal)
    ymax = max(np.max(no_ai_pct), np.max(ai_pct)) if len(no_ai_pct) else 0
    offset = max(0.5, 0.03 * ymax)

    def annotate(bars, values):
        for r, v in zip(bars, values):
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width()/2.0,
                h + offset,
                f'{v:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )

    annotate(b1, no_ai_pct)
    annotate(b2, ai_pct)

    plt.tight_layout()
    plt.show()
    save_plot(fig, f"barplot_failure_modes_pct_{monkey}_{experiment}_with_correct", "Failure mode")
    plt.close(fig)


def plot_individual_trials(
    monkey, experiment, base_dir,
    # --- visuals / sampling controls ---
    stride=5,             # keep every k-th sample (set None to disable)
    min_dt=None,          # keep sample if >= this many seconds since last kept (None to disable)
    min_dist=None,        # keep sample if moved >= this many scene units since last kept (None to disable)
    arrow_scale=0.4,      # velocity arrow length scale
    min_alpha=0.2         # minimum opacity when entropy is high
):
    """
    Plot per-trial trajectories using RECORDED positions from aiVelocities[i]["AvatarPosition"].

    - No integration: we draw exactly what the log reports.
    - AI-on segments in green; AI-off in orange.
    - Transparency ~ 1 - normalized(EntropyLb).
    - Optional thinning by stride, min_dt (seconds), and/or min_dist (scene units).
    """
    import os, glob, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.cm import ScalarMappable

    ALL_TARGET_POSITIONS = np.array([
        [-7.0, 0.75, 6.0],
        [-3.5, 0.75, 8.5],
        [ 0.0, 0.75, 9.2],
        [ 3.5, 0.75, 8.5],
        [ 7.0, 0.75, 6.0],
    ], dtype=float)
    # --------------- helpers -----------------
    def thin_trajectory(pos, keep_idx, alpha_vals, ai_on_mask, ai_recs,
                        stride=None, min_dt=None, min_dist=None, use_output_ts=True):
        """
        Returns thinned (pos, keep_idx, alpha_vals, ai_on_mask) according to:
          - stride: keep every k-th sample
          - min_dt: keep if time since last kept >= min_dt (seconds)
          - min_dist: keep if distance since last kept >= min_dist (scene units)
        Order: stride -> time -> distance
        """
        idx = list(range(len(keep_idx)))

        # 1) stride (simple decimation)
        if stride and stride > 1:
            idx = idx[::stride]

        # 2) time-based thinning
        if min_dt is not None:
            ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
            ts_vals = []
            for k in keep_idx:
                t = ai_recs[k].get(ts_key)
                # convert datetime to epoch seconds if needed
                if hasattr(t, "timestamp"):
                    t = t.timestamp()
                ts_vals.append(float(t) if t is not None else np.nan)

            kept = []
            last_t = None
            for j in idx:
                tj = ts_vals[j]
                if np.isnan(tj):
                    kept.append(j)
                    last_t = tj
                    continue
                if last_t is None or (tj - last_t) >= float(min_dt):
                    kept.append(j)
                    last_t = tj
            idx = kept

        # 3) distance-based thinning
        if min_dist is not None and len(idx) > 1:
            kept = [idx[0]]
            last_p = pos[idx[0]]
            for j in idx[1:]:
                if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                    kept.append(j)
                    last_p = pos[j]
            idx = kept

        # Apply thinning indices
        pos_th   = pos[idx]
        keep_idx_th = [keep_idx[j] for j in idx]
        alpha_th = alpha_vals[idx]
        ai_on_th = ai_on_mask[idx]
        return pos_th, keep_idx_th, alpha_th, ai_on_th

    def add_square(ax, center, width=1.5, color='k', fill=False, lw=2, label=None,):
        if center is None:
            return
        x, _, z = center   # Unity: (x, y, z)
        import matplotlib.pyplot as plt
        rect = plt.Rectangle(
            (x - width/2, z - width/2), width, width,
            edgecolor=color,
            facecolor=(color if fill else 'none'),
            linewidth=lw,
            linestyle=("--" if not fill else "-"),
            label=label
        )
        ax.add_patch(rect)
        return rect

    # --------------- load data -----------------
    data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))

    # If you want to force a single file, set it here; otherwise iterate all
    file_path = r'X:\Method paper data\Monkey 1\AI Obstacle\aiLog\navdecodingsphereaiobstacle_Loki_20241223_1106_E_aitrials.pkl'
    # for file_path in pkl_files:
    with open(file_path, "rb") as f:
        trials = pickle.load(f)[1]

    # Only trials that have aiVelocities and (optionally) are successful (answer==1)
    ai_trials = [t for t in trials if t.get("aiVelocities")]
    ai_trials = [t for t in ai_trials if t.get("answer") in {1}]  # extend set if you want other outcomes

    for trial in ai_trials:
        # fig, ax = plt.subplots()
        fig, (ax, axr) = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios':[3, 2]})
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f"Trial {trial['trial']}")

        # Indices where AI is on, and I/O velocities (still used for arrows)
        ai_idxs, inputs, outputs, _time_info = ai_on_indices_from_trial(trial)
        ai_on_set = set(ai_idxs)

        # --------- Build trajectory from recorded AvatarPosition ----------
        ai_recs = trial.get("aiVelocities", [])

        xs, zs, keep_idx = [], [], []
        for i, rec in enumerate(ai_recs):
            ap = rec.get("AvatarPosition", None)
            if ap is None or (isinstance(ap, float) and np.isnan(ap)):
                continue
            if isinstance(ap, dict):
                x = ap.get("x", np.nan); z = ap.get("z", np.nan)
            else:
                # assume (x, y, z[, rot]) or similar
                x = ap[0] if len(ap) >= 1 else np.nan
                z = ap[2] if len(ap) >= 3 else np.nan
            if x is None or z is None or np.isnan(x) or np.isnan(z):
                continue

            xs.append(float(x))
            zs.append(float(z))
            keep_idx.append(i)

        if len(keep_idx) < 2:
            plt.close(fig)
            continue

        pos = np.column_stack([np.asarray(xs), np.asarray(zs)])  # (K, 2) in (X, Z)

        # --------- Entropy -> alpha (align to kept indices) ----------
        ent_raw = []
        for rec in ai_recs:
            e = rec.get("EntropyLb")
            if isinstance(e, (list, tuple, np.ndarray)):
                arr = np.asarray(e, float).ravel()
                e = float(arr[0]) if arr.size else np.nan
            ent_raw.append(np.nan if e is None else float(e))

        entropy = np.asarray(ent_raw, dtype=float)

        # Compute alpha from the full entropy vector (so the scale includes the
        # true near-uniform prior at the beginning), then align to keep_idx.
        alpha_all = robust_alpha(entropy, floor=min_alpha)
        alpha_vals = alpha_all[keep_idx]
        # # --------- Entropy -> alpha (align to kept indices) ----------
        # ent_raw = []
        # for rec in ai_recs:
        #     e = rec.get("EntropyLb")
        #     if isinstance(e, (list, tuple, np.ndarray)):
        #         arr = np.asarray(e).ravel()
        #         e = float(arr[0]) if arr.size else np.nan
        #     ent_raw.append(np.nan if e is None else float(e))
        # entropy = np.asarray(ent_raw, dtype=float)
        # entropy_kept = entropy[keep_idx]
        # if np.all(np.isnan(entropy_kept)) or np.isclose(np.nanmax(entropy_kept) - np.nanmin(entropy_kept), 0):
        #     alpha_vals = np.ones(len(keep_idx), dtype=float)
        # else:
        #     e_min, e_max = np.nanmin(entropy_kept), np.nanmax(entropy_kept)
        #     alpha_vals = 1.0 - (entropy_kept - e_min) / (e_max - e_min)  # higher entropy -> more transparent
        #     alpha_vals = np.clip(np.nan_to_num(alpha_vals, nan=1.0), float(min_alpha), 1.0)

        # AI-on flags remapped to kept indices
        ai_on_mask = np.array([(k in ai_on_set) for k in keep_idx], dtype=bool)

        # --------- Optional thinning to match original density ----------
        pos, keep_idx, alpha_vals, ai_on_mask = thin_trajectory(
            pos, keep_idx, alpha_vals, ai_on_mask, ai_recs,
            stride=stride, min_dt=min_dt, min_dist=min_dist, use_output_ts=True
        )

        # --------- Optional: remove last N points (offline artifact near target) ----------
        N_REMOVE_END = 3
        if pos.shape[0] > N_REMOVE_END:
            pos        = pos[:-N_REMOVE_END]
            alpha_vals = alpha_vals[:-N_REMOVE_END]
            ai_on_mask = ai_on_mask[:-N_REMOVE_END]
            keep_idx   = keep_idx[:-N_REMOVE_END]
            if times_th is not None:
                times_th = times_th[:-N_REMOVE_END]

        # --------- Draw target & obstacle ----------
        add_square(ax, trial.get("targetPosition"),  width=1.5, color='g', fill=True,  lw=1.5)
        # add_square(ax, trial.get("targetPosition"),  width=4.2, color='g', fill=False, lw=2)
        add_square(ax, trial.get("obstaclePosition"), width=0.9, color='r', fill=True,  lw=1.5)
        def _as_tuple(center):
            # accepts dict {'x','y','z'} or iterable (x,y,z)
            if center is None:
                return None
            if isinstance(center, dict):
                return (center.get('x'), center.get('y', 0.0), center.get('z'))
            return tuple(center)

        curr_t = _as_tuple(trial.get("targetPosition"))

        other_targets_handle = None
        other_targets_labeled = False

        for c in ALL_TARGET_POSITIONS:
            ct = _as_tuple(c)
            if ct is None: 
                continue
            if curr_t is not None and np.allclose([ct[0], ct[2]], [curr_t[0], curr_t[2]], atol=1e-6):
                continue  # skip active

            lbl = "Other candidate targets" if not other_targets_labeled else "_nolegend_"
            h = add_square(ax, ct, width=1.5, color='#7f7f7f', fill=False, lw=1.5)
            if not other_targets_labeled and h is not None:
                other_targets_handle = h
                other_targets_labeled = True

        color_on, color_off = "#D81B60" , "#6C6E6F"

        # --- Blues colormap (opaque) + per-trial quantile stretch ---
        base = plt.cm.Blues(np.linspace(0.30, 0.95, 256))
        base[:, 3] = 1.0                     # no transparency
        cmap = ListedColormap(base)
        vmin, vmax = np.nanpercentile(alpha_vals, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
        colors = cmap(norm(alpha_vals))        # color for each point
        # --------- Plot segments (alpha by entropy, color by AI state) ----------
        for i in range(len(pos) - 1):
            base_color = color_off if ai_on_mask[i] else color_off
            ax.plot(pos[i:i+2, 0], pos[i:i+2, 1],
                    color=base_color, alpha=alpha_vals[i], linewidth=2)

        # breadcrumbs every "stride" (or every 3 if stride is None)
        on_idx  = np.flatnonzero(ai_on_mask)
        off_idx = np.flatnonzero(~ai_on_mask)
        # AI-on: 
        ax.scatter(
            pos[on_idx, 0], pos[on_idx, 1],
            s=70, c=color_on,
            edgecolors='black', marker = 'D', linewidths=1, alpha=0.7, zorder=6, label='AI override'
        )
        # AI-off: Blues colormap (use the colors you already computed)
        ax.scatter(
            pos[off_idx, 0], pos[off_idx, 1],
            s=70, c=colors[off_idx],
            edgecolors='white', linewidths=0.7, alpha=1.0, zorder=3
        )
        # start & end markers
        ax.scatter(pos[0, 0], pos[0, 1], s=60, facecolor='white', edgecolor='k', zorder=4, label = "Start Position")
        last_i = len(pos) - 1
        base_color_last = color_on if ai_on_mask[last_i] else color_off

        # --------- Velocity arrows (optional; drawn only where we kept samples) ----------
        if inputs is not None and len(inputs) > 0:
            if len(inputs) <= max(keep_idx, default=-1):
                pass  # not enough outputs to align; skip arrows
            else:
                v_all = inputs[:, [0, 2]]  # (vx, vz)
                v = v_all[keep_idx]
                n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
                v = (v / n) * float(arrow_scale)
                # for j in range(0, len(pos), max(4, step)):
                for j in range(0, len(pos), 1):
                    ax.arrow(pos[j, 0], pos[j, 1], v[j, 0], v[j, 1],
                                head_width=0.11, length_includes_head=True,
                                color='black', alpha=1, zorder=10)
        # proxy arrow for legend
        from matplotlib.patches import FancyArrowPatch, FancyArrow
        from matplotlib.legend_handler import HandlerPatch
        def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            # draw a right-pointing arrow centered vertically in the legend box
            return FancyArrow(
                xdescent, ydescent + height/2.0,  # start (x, y)
                width, 0.0,                       # dx, dy
                length_includes_head=True,
                head_width=0.6 * height,
                head_length=0.35 * width,
                color='black'
            )

        arrow_handle = FancyArrowPatch((0, 0), (1.0, 0.0),
                                    arrowstyle='->', mutation_scale=16,
                                    color='black', linewidth=2, label='BCI velocity')

        # --------- Final touches ----------
        # --- colorbar for AI share (dots) ---
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Posterior Confidence Index")
        mid = (vmin + vmax) / 2.0
        cbar.set_ticks([0,0.5,1])
        cbar.set_ticklabels([f"0", f"0.5", f"1"])

        # handles, labels = ax.get_legend_handles_labels()
        from matplotlib.patches import Rectangle
        def _legend_square(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            # draw a centered square inside the legend handle box
            size = 1.2 * min(width, height)
            x = xdescent + (width  - size) / 2.0
            y = ydescent + (height - size) / 2.0
            sq = Rectangle((x, y), size, size)
            # carry over original styles
            sq.set_facecolor(orig_handle.get_facecolor())
            sq.set_edgecolor(orig_handle.get_edgecolor())
            sq.set_linestyle(orig_handle.get_linestyle())
            sq.set_linewidth(orig_handle.get_linewidth() or 1.5)
            return sq
        # proxy squares
        square_active = Rectangle((0,0), 1, 1, facecolor='g', edgecolor='g', label='Active target')
        square_obst   = Rectangle((0,0), 1, 1, facecolor='r', edgecolor='r', label='Obstacle')
        square_other  = Rectangle((0,0), 1, 1, facecolor='none', edgecolor='k', linestyle='--', label='Other candidate targets')

        # gather existing handles/labels and add proxies
        handles, _ = ax.get_legend_handles_labels()
        handles += [square_active, square_obst, square_other, arrow_handle]  


        by_label = {h.get_label(): h for h in handles}

        ax.legend(by_label.values(), by_label.keys(),
                loc='best', frameon=True,
                handler_map={
                    FancyArrowPatch: HandlerPatch(patch_func=_legend_arrow),  # you already have this
                    Rectangle:       HandlerPatch(patch_func=_legend_square), # <-- makes them squares
                })
        ax.margins(0.05)
        ax.set_xlim(-10, 10); ax.set_ylim(-1, 11) 

        # === RIGHT PANEL: distance → AI-policy confidence (cleaner) ===
        tgt = trial.get("targetPosition")
        if tgt is not None:
            tx, tz = (tgt["x"], tgt["z"]) if isinstance(tgt, dict) else (tgt[0], tgt[2])
            dists = np.linalg.norm(pos - np.array([tx, tz]), axis=1)

            # scatter (light gray)
            axr.scatter(dists, alpha_vals, s=14, color='0.5', alpha=0.35, edgecolors='none', label='Samples')

            # equal-count (quantile) bins for smoother line
            qbins = 8
            edges = np.quantile(dists, np.linspace(0, 1, qbins + 1))
            idx = np.digitize(dists, edges[1:-1], right=True)

            bin_x, bin_y, bin_sem = [], [], []
            for b in range(qbins):
                mask = (idx == b)
                if mask.sum() == 0:
                    continue
                x = dists[mask].mean()
                y = alpha_vals[mask].mean()
                sem = alpha_vals[mask].std(ddof=1) / np.sqrt(mask.sum())
                bin_x.append(x); bin_y.append(y); bin_sem.append(sem)
            bin_x = np.asarray(bin_x); bin_y = np.asarray(bin_y); bin_sem = np.asarray(bin_sem)

            # # SEM band + line
            # axr.fill_between(bin_x, bin_y - bin_sem, bin_y + bin_sem, alpha=0.15, lw=0)
            axr.plot(bin_x, bin_y, '-o', color='k', lw=2, ms=4, label='Binned mean')

            # Pearson r (+ p) with safe fallback
            try:
                from scipy.stats import pearsonr
                m = np.isfinite(dists) & np.isfinite(alpha_vals)
                if m.sum() >= 3:
                    r, p = pearsonr(dists[m], alpha_vals[m])
                    axr.text(0.02, 0.98, f"r = {r:.2f}, p = {p:.3g}",
                            transform=axr.transAxes, ha='left', va='top',
                            fontsize=10, bbox=dict(boxstyle='round,pad=0.25',
                                                    facecolor='white', alpha=0.8, lw=0))
            except Exception:
                pass
            
            annotate_entropy_axes(axr, trial, eps=1e-3, target_radius=0.5, x_coord="target_dist")
            axr.legend(loc='best', frameon=True)
            # axes / grid
            axr.set_ylim(0, 1)
            axr.set_xlim(dists.max() + 0.2, dists.min() - 0.2)  # invert so "closer → right"
            axr.set_xlabel("Distance to target center")
            axr.set_ylabel("Posterior Confidence Index")
            axr.grid(True, alpha=0.25)

        plt.tight_layout()
        out_name = f"{file_path}_trial{trial['trial']:03d}"
        save_plot(fig, out_name, os.path.join(base_dir, "trial_plots", monkey, experiment))
        plt.show()
def plot_individual_trials(
    monkey, experiment, base_dir,
    # --- visuals / sampling controls ---
    stride=5,             # keep every k-th sample (set None to disable)
    min_dt=None,          # keep sample if >= this many seconds since last kept (None to disable)
    min_dist=None,        # keep sample if moved >= this many scene units since last kept (None to disable)
    arrow_scale=0.4,      # velocity arrow length scale
    min_alpha=0.2         # minimum opacity when entropy is high
):
    """
    Plot per-trial trajectories using RECORDED positions from aiVelocities[i]["AvatarPosition"].

    Left panel:
      - Recorded avatar path in X–Z.
      - AI-on samples annotated with diamonds, AI-off with dots.
      - Transparency ~ posterior confidence index α.

    Right panel:
      - Posterior confidence index α vs time from trial start (single trial).
    """
    import os, glob, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import FancyArrowPatch, FancyArrow, Rectangle
    from matplotlib.legend_handler import HandlerPatch
    from scipy.stats import pearsonr

    ALL_TARGET_POSITIONS = np.array([
        [-7.0, 0.75, 6.0],
        [-3.5, 0.75, 8.5],
        [ 0.0, 0.75, 9.2],
        [ 3.5, 0.75, 8.5],
        [ 7.0, 0.75, 6.0],
    ], dtype=float)

    # --------------- helpers -----------------
    def thin_trajectory(pos, keep_idx, alpha_vals, ai_on_mask, ai_recs,
                        stride=None, min_dt=None, min_dist=None, use_output_ts=True,
                        times=None):
        """
        Returns thinned (pos, keep_idx, alpha_vals, ai_on_mask, times_th) according to:
          - stride: keep every k-th sample
          - min_dt: keep if time since last kept >= min_dt (seconds)
          - min_dist: keep if distance since last kept >= min_dist (scene units)

        `times` should be an array aligned to `keep_idx` (one time per kept sample),
        in seconds. If None, timestamps are recomputed from ai_recs.
        """
        import numpy as np

        idx = list(range(len(keep_idx)))

        # 1) stride (simple decimation)
        if stride and stride > 1:
            idx = idx[::stride]

        # 2) time-based thinning
        if min_dt is not None:
            if times is not None:
                ts_vals = np.asarray(times, float)
            else:
                ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
                ts_vals = []
                for k in keep_idx:
                    t = ai_recs[k].get(ts_key)
                    if hasattr(t, "timestamp"):
                        t = t.timestamp()
                    ts_vals.append(np.nan if t is None else float(t))
                ts_vals = np.asarray(ts_vals, float)

            kept = []
            last_t = None
            for j in idx:
                tj = ts_vals[j]
                if np.isnan(tj):
                    kept.append(j)
                    last_t = tj
                    continue
                if last_t is None or (tj - last_t) >= float(min_dt):
                    kept.append(j)
                    last_t = tj
            idx = kept

        # 3) distance-based thinning
        if min_dist is not None and len(idx) > 1:
            kept = [idx[0]]
            last_p = pos[idx[0]]
            for j in idx[1:]:
                if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                    kept.append(j)
                    last_p = pos[j]
            idx = kept

        # Apply thinning indices
        pos_th      = pos[idx]
        keep_idx_th = [keep_idx[j] for j in idx]
        alpha_th    = alpha_vals[idx]
        ai_on_th    = ai_on_mask[idx]

        if times is not None:
            times_th = np.asarray(times, float)[idx]
        else:
            times_th = None

        return pos_th, keep_idx_th, alpha_th, ai_on_th, times_th

    def add_square(ax, center, width=1.5, color='k', fill=False, lw=2, label=None):
        if center is None:
            return
        x, _, z = center   # Unity: (x, y, z)
        rect = plt.Rectangle(
            (x - width/2, z - width/2), width, width,
            edgecolor=color,
            facecolor=(color if fill else 'none'),
            linewidth=lw,
            linestyle=("--" if not fill else "-"),
            label=label
        )
        ax.add_patch(rect)
        return rect

    def _as_tuple(center):
        # accepts dict {'x','y','z'} or iterable (x,y,z)
        if center is None:
            return None
        if isinstance(center, dict):
            return (center.get('x'), center.get('y', 0.0), center.get('z'))
        return tuple(center)

    def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        return FancyArrow(
            xdescent, ydescent + height/2.0,
            width, 0.0,
            length_includes_head=True,
            head_width=0.6 * height,
            head_length=0.35 * width,
            color='black'
        )

    def _legend_square(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        # draw a centered square inside the legend handle box
        size = 1.2 * min(width, height)
        x = xdescent + (width  - size) / 2.0
        y = ydescent + (height - size) / 2.0
        sq = Rectangle((x, y), size, size)
        sq.set_facecolor(orig_handle.get_facecolor())
        sq.set_edgecolor(orig_handle.get_edgecolor())
        sq.set_linestyle(orig_handle.get_linestyle())
        sq.set_linewidth(orig_handle.get_linewidth() or 1.5)
        return sq

    # --------------- load data -----------------
    data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))

    # For now: force a specific file; change if needed
    file_path = r"X:\Method paper data\Monkey 3\AI Respawn\Reset_prior_analyze\navdecodingsphereairespawn_Maui_20250617_1151_A_prior_reset.pkl"

    with open(file_path, "rb") as f:
        new_trials = pickle.load(f)
    new_trials = new_trials['trials']
    trials = []
    for tr in new_trials:
        old = new_trial_to_old_trialdict(tr)
        if old is not None:
            trials.append(old)

    # Only trials with aiVelocities and correct answer
    ai_trials = [t for t in trials if t.get("aiVelocities")]
    ai_trials = [t for t in ai_trials if t.get("answer") in {1}]
    ai_trials = [t for t in trials if t.aiVelocities]
    ai_trials = [t for t in ai_trials if t.answer in {5}]
 
    
    for trial in ai_trials:
        fig, (ax, axr) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={'width_ratios': [3, 2]}
        )
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f"Trial {trial['trial']}")
        # ax.set_title(f"Trial {trial.trial}")

        # Indices where AI is on + BCI inputs/outputs
        ai_idxs, inputs, outputs, _time_info = ai_on_indices_from_trial(trial)
        ai_on_set = set(ai_idxs)

        ai_recs = trial.get("aiVelocities", [])
        # ai_recs = trial.aiVelocities

        # --------- Build trajectory from recorded AvatarPosition ----------
        xs, zs, keep_idx = [], [], []
        for i, rec in enumerate(ai_recs):
            ap = rec.get("AvatarPosition", None)
            # ap = trial.avatarTrajectory
            if ap is None or (isinstance(ap, float) and np.isnan(ap)):
                continue
            if isinstance(ap, dict):
                x = ap.get("x", np.nan)
                z = ap.get("z", np.nan)
            else:
                x = ap[0] if len(ap) >= 1 else np.nan
                z = ap[2] if len(ap) >= 3 else np.nan

            if x is None or z is None or np.isnan(x) or np.isnan(z):
                continue

            xs.append(float(x))
            zs.append(float(z))
            keep_idx.append(i)

        if len(keep_idx) < 2:
            plt.close(fig)
            continue

        pos = np.column_stack([np.asarray(xs), np.asarray(zs)])  # (K, 2) in (X, Z)

        # --------- Entropy -> alpha (align to kept indices) ----------
        ent_raw = []
        for rec in ai_recs:
            e = rec.get("EntropyLb")
            if isinstance(e, (list, tuple, np.ndarray)):
                arr = np.asarray(e, float).ravel()
                e = float(arr[0]) if arr.size else np.nan
            ent_raw.append(np.nan if e is None else float(e))
        entropy = np.asarray(ent_raw, dtype=float)

        alpha_all = robust_alpha(entropy, floor=min_alpha)
        alpha_vals = alpha_all[keep_idx]

        # AI-on flags mapped to kept indices
        ai_on_mask = np.array([(k in ai_on_set) for k in keep_idx], dtype=bool)

        # --------- Time vector from timestamps (aligned to kept indices) ----------
        ts_full = []
        for i, rec in enumerate(ai_recs):
            t = rec.get("OutputTimestamp", None) or rec.get("InputTimestamp", None)
            if hasattr(t, "timestamp"):
                t = t.timestamp()
            ts_full.append(np.nan if t is None else float(t))
        ts_full = np.asarray(ts_full, float)
        if ts_full.size >= (max(keep_idx) + 1):
            times_keep = ts_full[keep_idx]
        else:
            times_keep = np.full(len(keep_idx), np.nan)

        # --------- Optional thinning ----------
        pos, keep_idx, alpha_vals, ai_on_mask, times_th = thin_trajectory(
            pos, keep_idx, alpha_vals, ai_on_mask, ai_recs,
            stride=stride, min_dt=min_dt, min_dist=min_dist,
            use_output_ts=True, times=times_keep
        )

        # --------- Draw target & obstacle ----------
        add_square(ax, trial.get("targetPosition"),  width=1.5, color='grey', fill=True,  lw=1.5)
        add_square(ax, trial.get("targetJumpPosition"),  width=1.5, color='g', fill=True,  lw=1.5)
        # add_square(ax, trial.targetPosition,  width=1.5, color='grey', fill=True,  lw=1.5)
        # add_square(ax, trial.targetJumpPosition,  width=1.5, color='g', fill=True,  lw=1.5)
        # add_square(ax, trial.get("obstaclePosition"), width=0.9, color='r', fill=True,  lw=1.5)

        curr_t = _as_tuple(trial.get("targetPosition"))
        jump_t = _as_tuple(trial.get("targetJumpPosition"))
        # curr_t = _as_tuple(trial.targetPosition)
        # jump_t = _as_tuple(trial.targetJumpPosition)

        other_targets_handle = None
        other_targets_labeled = False

        for c in ALL_TARGET_POSITIONS:
            ct = _as_tuple(c)
            if ct is None:
                continue

            # Skip active target
            if curr_t is not None and np.allclose([ct[0], ct[2]], [curr_t[0], curr_t[2]], atol=1e-6):
                continue

            # Skip jump target too
            if jump_t is not None and np.allclose([ct[0], ct[2]], [jump_t[0], jump_t[2]], atol=1e-6):
                continue

            lbl = "Other candidate targets" if not other_targets_labeled else "_nolegend_"
            h = add_square(ax, ct, width=1.5, color="#7f7f7f", fill=False, lw=1.5)
            if not other_targets_labeled and h is not None:
                other_targets_handle = h
                other_targets_labeled = True


        color_on, color_off = "#D81B60", "#6C6E6F"

        # --- static colormap, α as opacity only ---
        base = plt.cm.Blues(np.linspace(0.30, 0.95, 256))
        base[:, 3] = 1.0
        cmap = ListedColormap(base)
        norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
        colors = cmap(norm(alpha_vals))

        # --------- Path segments ----------
        for i in range(len(pos) - 1):
            base_color = color_off  # keep path in gray, use diamonds for AI takeover
            ax.plot(pos[i:i+2, 0], pos[i:i+2, 1],
                    color=base_color, alpha=alpha_vals[i], linewidth=2)

        # AI-on diamonds
        on_idx  = np.flatnonzero(ai_on_mask)
        off_idx = np.flatnonzero(~ai_on_mask)
        ax.scatter(
            pos[on_idx, 0], pos[on_idx, 1],
            s=70, c=color_on,
            edgecolors='black', marker='D', linewidths=1,
            alpha=0.7, zorder=6, label='AI override'
        )
        # AI-off dots colored by α
        ax.scatter(
            pos[off_idx, 0], pos[off_idx, 1],
            s=70, c=colors[off_idx],
            edgecolors='white', linewidths=0.7,
            alpha=1.0, zorder=3
        )

        # start marker
        ax.scatter(pos[0, 0], pos[0, 1], s=60,
                   facecolor='white', edgecolor='k',
                   zorder=4, label="Start Position")

        # --------- Velocity arrows (BCI velocity) ----------
        if inputs is not None and len(inputs) > 0:
            if len(inputs) > max(keep_idx, default=-1):
                v_all = inputs[:, [0, 2]]  # (vx, vz)
                v = v_all[keep_idx]
                n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
                v = (v / n) * float(arrow_scale)
                for j in range(0, len(pos), 1):
                    ax.arrow(pos[j, 0], pos[j, 1], v[j, 0], v[j, 1],
                             head_width=0.11, length_includes_head=True,
                             color='black', alpha=1, zorder=10)

        arrow_handle = FancyArrowPatch((0, 0), (1.0, 0.0),
                                       arrowstyle='->', mutation_scale=16,
                                       color='black', linewidth=2, label='BCI velocity')

        # --------- Colorbar for α ----------
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Posterior Confidence Index")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["0", "0.5", "1"])

        # Respawn time
        ax.axhline(y=2, color="red", lw=1.5, linestyle="-", zorder=0)
        respawn_line = ax.axhline(y=2, color="red", lw=1.5, linestyle="-", zorder=0)

        # Legend proxies
        square_active = Rectangle((0, 0), 1, 1, facecolor='g', edgecolor='g', label='Active target')
        square_obst   = Rectangle((0, 0), 1, 1, facecolor='r', edgecolor='r', label='Obstacle')
        square_other  = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='k',
                                  linestyle='--', label='Other candidate targets')
        from matplotlib.lines import Line2D
        respawn_handle = Line2D([0], [0], color="red", lw=1.5, label="Respawn boundary (z=2)")

        handles, _ = ax.get_legend_handles_labels()
        handles += [square_active, square_obst, square_other, arrow_handle, respawn_handle]
        by_label = {h.get_label(): h for h in handles}

        ax.legend(
            by_label.values(), by_label.keys(),
            loc='best', frameon=True,
            handler_map={
                FancyArrowPatch: HandlerPatch(patch_func=_legend_arrow),
                Rectangle:       HandlerPatch(patch_func=_legend_square),
            }
        )

        ax.margins(0.05)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1, 11)

        # === RIGHT PANEL: α vs time from trial start ===
        if times_th is not None and np.isfinite(times_th).sum() >= 3:
            # convert to seconds relative to first finite timestamp
            mask_ts = np.isfinite(times_th)
            t0 = times_th[mask_ts][0]
            t_rel = times_th - t0

            # --- NEW: add green bands where AI override is ON ---
            # use ai_on_mask (already thinned, same length as times_th)
            add_ai_override_bands(axr, t_rel, ai_on_mask, color="#8BC34A", alpha=0.20)

            # scatter
            m = np.isfinite(t_rel) & np.isfinite(alpha_vals)
            t_rel_valid = t_rel[m]
            alpha_valid = alpha_vals[m]
            axr.scatter(
                t_rel_valid, alpha_valid,
                s=14, color='0.5', alpha=0.35,
                edgecolors='none', label='Samples'
            )

            # quantile bins over time
            if t_rel_valid.size >= 4:
                qbins = 8
                edges = np.quantile(t_rel_valid, np.linspace(0, 1, qbins + 1))
                idx = np.digitize(t_rel_valid, edges[1:-1], right=True)

                bin_x, bin_y, bin_sem = [], [], []
                for b in range(qbins):
                    mask_b = (idx == b)
                    if mask_b.sum() == 0:
                        continue
                    x = t_rel_valid[mask_b].mean()
                    y = alpha_valid[mask_b].mean()
                    sem = alpha_valid[mask_b].std(ddof=1) / np.sqrt(mask_b.sum())
                    bin_x.append(x); bin_y.append(y); bin_sem.append(sem)
                bin_x = np.asarray(bin_x); bin_y = np.asarray(bin_y)

                axr.plot(bin_x, bin_y, '-o', color='k', lw=2, ms=4, label='Binned mean')

            # correlation r(time, α)
            if t_rel_valid.size >= 3:
                r, p = pearsonr(t_rel_valid, alpha_valid)
                axr.text(
                    0.02, 0.98, f"r = {r:.2f}, p = {p:.3g}",
                    transform=axr.transAxes, ha='left', va='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.8, lw=0)
                )

            # OPTIONAL: annotate events along time axis if your helper supports it
            # try:
            #     annotate_entropy_axes(axr, trial, eps=1e-3,
            #                           target_radius=0.5, x_coord="time")
            # except Exception:
            #     pass

            
            # --- Legend including AI override band ---
            # proxy patch for AI override
            ai_band_patch = Rectangle((0, 0), 1, 1,
                                      facecolor="#8BC34A", edgecolor="none",
                                      alpha=0.4, label="AI override")
            handles_r, labels_r = axr.get_legend_handles_labels()
            handles_r.append(ai_band_patch)
            axr.legend(handles_r, [h.get_label() for h in handles_r],
                       loc='best', frameon=True)

            axr.set_ylim(0, 1)
            axr.set_xlim(t_rel_valid.min() - 0.1, t_rel_valid.max() + 0.1)
            axr.set_xlabel("Time from trial start (s)")
            axr.set_ylabel("Prior Confidence Index")
            axr.grid(True, alpha=0.25)

        plt.tight_layout()
        out_name = f"{file_path}_trial{trial['trial']:03d}"
        # out_name = f"{file_path}_trial{trial.trial:03d}"
        save_plot(fig, out_name, os.path.join(base_dir, "trial_plots", monkey, experiment))
        plt.show()
        # plt.close(fig)

import numpy as np
from dateutil.parser import isoparse

import numpy as np
from dateutil.parser import isoparse

def new_trial_to_old_trialdict(tr):
    """
    Convert NEW trial dict -> OLD trial dict expected by your plotting script.

    OLD expects trial["aiVelocities"] list of dicts with:
      - AvatarPosition: {'x','y','z'}
      - Input:  (3,)  (model/BCI velocity)
      - Output: (3,)  (AI velocity)
      - InputTimestamp / OutputTimestamp (numeric ok)
      - EntropyLb
    """

    samples = tr.get("samples", {})

    pos = np.asarray(samples.get("position", []), float)         # (Np,3)
    ent = np.asarray(samples.get("entropy", []), float).reshape(-1)  # (Ne,)
    bci = np.asarray(samples.get("bci_velocity", []), float)     # (Nb,2) vx,vz
    aiv = np.asarray(samples.get("ai_velocity", []), float)      # (Na,2) vx,vz

    if pos.ndim != 2 or pos.shape[0] < 2:
        return None

    N = pos.shape[0]

    # --- helper: pad/trim to N rows ---
    def _pad_1d(x, N, fill=np.nan):
        x = np.asarray(x, float).reshape(-1)
        if x.size < N:
            return np.concatenate([x, np.full(N - x.size, fill)])
        return x[:N]

    def _pad_2d(x, N, fill=np.nan):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 2:
            x = np.empty((0, 2), float)
        if x.shape[0] < N:
            pad = np.full((N - x.shape[0], 2), fill, float)
            return np.vstack([x, pad])
        return x[:N, :]

    ent = _pad_1d(ent, N, fill=np.nan)
    bci = _pad_2d(bci, N, fill=np.nan)
    aiv = _pad_2d(aiv, N, fill=np.nan)

    # --- timestamps (seconds, epoch) ---
    t0 = isoparse(tr["start_time"]).timestamp()
    t1 = isoparse(tr["end_time"]).timestamp()
    if not np.isfinite(t1) or t1 <= t0:
        ts = t0 + np.arange(N) * 0.05
    else:
        ts = np.linspace(t0, t1, N, endpoint=False)

    aiVelocities = []
    for i in range(N):
        # expand 2D velocities -> 3D (vx, vy=0, vz)
        inp = np.array([bci[i, 0], 0.0, bci[i, 1]], dtype=float)
        out = np.array([aiv[i, 0], 0.0, aiv[i, 1]], dtype=float)

        aiVelocities.append({
            "AvatarPosition": {"x": float(pos[i, 0]), "y": float(pos[i, 1]), "z": float(pos[i, 2])},
            "Input": inp,
            "Output": tuple(out),  # your _extract_vel... does list(v['Output']) so tuple is safe
            "EntropyLb": float(ent[i]) if np.isfinite(ent[i]) else np.nan,
            "EntropyUb": np.nan,
            "Latency": np.nan,
            "OutputTimestamp": float(ts[i]),
            "InputTimestamp": float(ts[i]),
        })

    g =  tr.get("true_goal", None)
    targetPosition = None if g is None else np.array(
        [float(g["x"]), float(g.get("y", 0.75)), float(g["z"])],
        dtype=float
    )
    h =  tr.get("target_jump_position", None)
    targetJumpPosition = None if h is None else np.array(
        [float(h["x"]), float(h.get("y", 0.75)), float(h["z"])],
        dtype=float
    )
    out_trial = {
        "trial": int(tr.get("trial_id", -1)),
        "answer": int(tr.get("answer", -999)),
        "aiVelocities": aiVelocities,
        "multipleAiControlBlocks": False,

        # keep these in old format (dict {'x','y','z'} is fine)
        "targetPosition": targetPosition,
        "targetJumpPosition": targetJumpPosition,
        "obstaclePosition": tr.get("obstaclePosition", None),

        # optional extras (won’t hurt)
        "aiControlOn": np.array([]),
        "aiControlOff": np.array([]),
    }
    return out_trial



def add_ai_override_bands(ax, t_rel, ai_on_mask, color="#8BC34A", alpha=0.20):
    """
    Shade time intervals where AI override is active.

    ax        : matplotlib axis
    t_rel     : 1D array of times (relative, e.g. from trial start)
    ai_on_mask: boolean mask (same length as t_rel), True where AI is ON
    """
    import numpy as np

    t_rel = np.asarray(t_rel, float)
    ai_on_mask = np.asarray(ai_on_mask, bool)

    good = np.isfinite(t_rel)
    is_on = ai_on_mask & good
    if not np.any(is_on):
        return

    idx = np.where(is_on)[0]
    if idx.size == 0:
        return

    # group into contiguous segments
    start = idx[0]
    prev  = idx[0]
    segments = []
    for j in idx[1:]:
        if j == prev + 1:
            prev = j
        else:
            segments.append((start, prev))
            start = j
            prev = j
    segments.append((start, prev))

    for s, e in segments:
        t0 = t_rel[s]
        t1 = t_rel[e]
        ax.axvspan(t0, t1, color=color, alpha=alpha,
                    zorder=-1, label="_ai_override_band_")

def annotate_entropy_axes(ax, trial, eps=1e-3, target_radius=0.5,
                          x_coord="target_dist"):
    """
    Mark movement onset, closest-to-obstacle, and target entry.

    x_coord:
        "index"       -> x = sample index
        "unity_x"     -> x = Unity X position
        "target_dist" -> x = distance to target center
    """
    import numpy as np

    def xz_or_zero(rec, key="AvatarPosition"):
        p = rec.get(key)
        if isinstance(p, dict):
            x, z = p.get("x", np.nan), p.get("z", np.nan)
        else:
            a = np.asarray(p, float).ravel() if p is not None else np.array([np.nan, np.nan, np.nan])
            x = a[0] if a.size > 0 else np.nan
            z = a[2] if a.size > 2 else np.nan
        x = 0.0 if not np.isfinite(x) else float(x)
        z = 0.0 if not np.isfinite(z) else float(z)
        return x, z

    av = trial["aiVelocities"]
    pos = np.array([xz_or_zero(rec) for rec in av], dtype=float)  # (N, 2)
    vel = np.asarray([rec["Output"] for rec in av], float)
    Hlb = np.asarray([rec["EntropyLb"] for rec in av], float)

    # movement onset
    speed = np.linalg.norm(vel, axis=1)
    i_move = int(np.argmax(speed > eps)) if np.any(speed > eps) else None

    # obstacle & target as (x,z)
    def center_xz(center):
        if isinstance(center, dict):
            return np.array([center["x"], center["z"]], float)
        a = np.asarray(center, float).ravel()
        return a[[0, 2]] if a.size >= 3 else np.array([np.nan, np.nan])

    obst = center_xz(trial["obstaclePosition"])
    tgt  = center_xz(trial["targetPosition"])

    d_obs = np.linalg.norm(pos - obst[None, :], axis=1)
    i_close = int(np.argmin(d_obs)) if np.isfinite(d_obs).any() else None

    d_tgt = np.linalg.norm(pos - tgt[None, :], axis=1)
    inside = np.where(d_tgt <= target_radius)[0]
    i_hit = int(inside[0]) if inside.size else None

    # choose x coordinate for each index
    def x_for(i):
        if i is None or not (0 <= i < len(Hlb)):
            return None
        if x_coord == "index":
            return float(i)
        elif x_coord == "unity_x":
            return float(pos[i, 0])
        elif x_coord == "target_dist":
            return float(d_tgt[i])
        else:
            return float(i)

    def vline(i, c, label):
        x = x_for(i)
        if x is not None and np.isfinite(x):
            ax.axvline(x, color=c, ls='--', lw=1.5, label=label)

    # vline(i_move,  'tab:gray',  'Move onset')
    vline(i_close, 'crimson',   'Closest to obstacle')
    # vline(i_hit,   'tab:green', 'Enter target')

# def annotate_entropy_axes(ax, trial, eps=1e-3, target_radius=0.5):
#     # arrays over samples
#     def xz_or_zero(rec, key="AvatarPosition"):
#         p = rec.get(key)
#         if isinstance(p, dict):                       # {'x':..., 'y':..., 'z':...}
#             x, z = p.get("x", np.nan), p.get("z", np.nan)
#         else:                                         # [x, y, z] or tuple/np.array/None
#             a = np.asarray(p, float).ravel() if p is not None else np.array([np.nan, np.nan, np.nan])
#             x = a[0] if a.size > 0 else np.nan
#             z = a[2] if a.size > 2 else np.nan
#         # replace NaN/inf with 0 as requested
#         x = 0.0 if not np.isfinite(x) else float(x)
#         z = 0.0 if not np.isfinite(z) else float(z)
#         return x, z

#     pos = np.array([xz_or_zero(rec) for rec in trial["aiVelocities"]], dtype=float)  # shape (N, 2)
#     # pos = np.asarray([np.c_[rec["AvatarPosition"][0], rec["AvatarPosition"][2]] for rec in trial["aiVelocities"]], float)
#     vel = np.asarray([rec["Output"] for rec in trial["aiVelocities"]], float)
#     Hlb = np.asarray([rec["EntropyLb"] for rec in trial["aiVelocities"]], float)

#     # movement onset
#     speed = np.linalg.norm(vel, axis=1)
#     i_move = int(np.argmax(speed > eps)) if np.any(speed > eps) else None

#     # closest obstacle
#     obst = np.array([trial["obstaclePosition"][0], trial["obstaclePosition"][2]], float)
#     d_obs = np.linalg.norm(pos - obst[None, :], axis=1)
#     i_close = int(np.argmin(d_obs)) if np.isfinite(d_obs).any() else None

#     # target entry
#     tgt = np.array([trial["targetPosition"][0], trial["targetPosition"][2]], float)
#     d_tgt = np.linalg.norm(pos - tgt[None, :], axis=1)
#     inside = np.where(d_tgt <= target_radius)[0]
#     i_hit = int(inside[0]) if inside.size else None

#     def vline(i, c, label):
#         if i is not None and 0 <= i < len(Hlb):
#             ax.axvline(i, color=c, ls='--', lw=1.5, label=label)

#     vline(i_move,  'tab:gray', 'move onset')
#     vline(i_close, 'crimson',  'closest to obstacle')
#     vline(i_hit,   'tab:green','enter target')

def plot_alpha_vs_distance_per_trial(
    monkey, experiment, base_dir,
    file_path=None,
    stride=5,
    min_dt=None,
    min_dist=None,
    nbins=10,
    min_alpha=0.2,
    invert_x=True,
    only_answer=None
):
    import os, glob, pickle
    import numpy as np
    import matplotlib.pyplot as plt

    def xz_from(pos):
        if pos is None:
            return None
        if isinstance(pos, dict):
            return (float(pos.get("x", np.nan)), float(pos.get("z", np.nan)))
        try:
            return (float(pos[0]), float(pos[2]))
        except Exception:
            return None

    def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
        # returns indices RELATIVE to the kept sample order
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

        return idx  # relative to kept order

    def robust_alpha(ent_vals, floor=min_alpha):
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

    def scatter_panel(ax, dists, alpha_vals, title, xlabel):
        mask = np.isfinite(dists) & np.isfinite(alpha_vals)
        ax.scatter(dists[mask], alpha_vals[mask], s=20, color='#6C6C6C', alpha=0.5)

        if mask.sum() >= 3:
            lo, hi = dists[mask].min(), dists[mask].max()
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                edges = np.linspace(lo, hi, nbins + 1)
                mids = 0.5 * (edges[:-1] + edges[1:])
                means = []
                for i in range(nbins):
                    m = (dists >= edges[i]) & (dists < edges[i+1]) & mask
                    means.append(np.nan if m.sum() == 0 else np.nanmean(alpha_vals[m]))
                means = np.asarray(means, float)
                ax.plot(mids, means, '-o', color='k', lw=2, ms=5)

            r = np.corrcoef(dists[mask], alpha_vals[mask])[0, 1]
            ax.text(0.02, 0.95, f"r = {r:.2f} (n={mask.sum()})",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('α = 1 − normalized EntropyLb')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        if invert_x:
            ax.invert_xaxis()

    # ---------------- load files ----------------
    if file_path is None:
        data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
        pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    else:
        pkl_files = [file_path]

    results = []
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

            # 1) build kept samples (pos + index into ai_recs)
            xs, zs, keep_idx = [], [], []
            for i, rec in enumerate(ai_recs):
                p = xz_from(rec.get("AvatarPosition"))
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
            if len(keep_idx) < 3:
                continue
            pos = np.column_stack([np.asarray(xs), np.asarray(zs)])

            # 2) entropy for ALL records, then select with keep_idx (FIX)
            ent_raw = []
            for rec in ai_recs:
                e = rec.get("EntropyLb")
                if isinstance(e, (list, tuple, np.ndarray)):
                    e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
                ent_raw.append(np.nan if e is None else float(e))
            ent_raw = np.asarray(ent_raw, float)
            ent_kept = ent_raw[keep_idx]

            # 3) thinning using REAL keep_idx (FIX)
            idx_rel = thin_indices(pos, keep_idx, ai_recs,
                                   stride=stride, min_dt=min_dt, min_dist=min_dist, use_output_ts=True)
            if len(idx_rel) < 3:
                continue
            pos = pos[idx_rel]
            ent_kept = ent_kept[idx_rel]

            # 4) alpha from entropy
            alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

            # 5) distances
            tgt = xz_from(trial.get("targetPosition"))
            obs = xz_from(trial.get("obstaclePosition"))
            d_tgt = np.linalg.norm(pos - np.array(tgt)[None, :], axis=1) if tgt is not None else np.full(len(pos), np.nan)
            d_obs = np.linalg.norm(pos - np.array(obs)[None, :], axis=1) if obs is not None else np.full(len(pos), np.nan)

            # 6) plot two panels
            fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
            # obstacle
            if np.isfinite(d_obs).any():
                scatter_panel(axl, d_obs, alpha_vals,
                              title='α vs. distance to obstacle (per trial)',
                              xlabel='Distance to obstacle center')
            else:
                axl.set_visible(False)
            # target
            scatter_panel(axr, d_tgt, alpha_vals,
                          title='α vs. distance to target (per trial)',
                          xlabel='Distance to target center')

            trial_id = int(trial.get("trial", -1))
            fig.suptitle(f"{monkey} – {experiment} – Trial {trial_id}  ({os.path.basename(fp)})", y=1.02, fontsize=11)
            fig.tight_layout()
            try:
                out_name = f"alpha_vs_distance_{monkey}_{experiment}_trial{trial_id:03d}"
                save_plot(fig, out_name, subfolder="Entropy")
            except Exception:
                pass
            plt.show(); plt.close(fig)

            # simple stats
            mask_t = np.isfinite(d_tgt) & np.isfinite(alpha_vals)
            r_t = np.corrcoef(d_tgt[mask_t], alpha_vals[mask_t])[0, 1] if mask_t.sum() >= 3 else np.nan
            if np.isfinite(d_obs).any():
                mask_o = np.isfinite(d_obs) & np.isfinite(alpha_vals)
                r_o = np.corrcoef(d_obs[mask_o], alpha_vals[mask_o])[0, 1] if mask_o.sum() >= 3 else np.nan
            else:
                r_o = np.nan

            results.append({
                "file": os.path.basename(fp),
                "trial": trial_id,
                "n_samples": int(len(alpha_vals)),
                "r_target": float(r_t),
                "r_obstacle": float(r_o),
            })

    return results

def robust_alpha(ent_vals, floor=0.2):
        """
        Convert PRIOR entropy values to an AI-confidence index in [floor, 1].
        Uses alpha = 1 - H / Hmax with Hmax ≈ ln(K), estimated from the
        maximum observed entropy (i.e., near-uniform prior).
        """
        import numpy as np

        ent_vals = np.asarray(ent_vals, float)

        # Empty or all-NaN → just return the floor (don't force 1.0)
        if ent_vals.size == 0 or np.all(np.isnan(ent_vals)):
            return np.full_like(ent_vals, float(floor), dtype=float)

        # Estimate Hmax = ln(K) from the data (K ≈ exp(max entropy)).
        # Fallback to 95th percentile if max is non-finite or <= 0.
        hmax_obs = np.nanmax(ent_vals)
        if not np.isfinite(hmax_obs) or hmax_obs <= 0:
            hmax_obs = np.nanpercentile(ent_vals, 95)

        K_est = int(np.round(np.exp(hmax_obs)))
        if K_est < 2:
            K_est = 2  # minimum sensible number of candidates
        Hmax = float(np.log(K_est))

        # Robustness: clip entropy into [0, Hmax] (winsorize upper tail only).
        H = np.clip(ent_vals, 0.0, Hmax)

        # Proper normalization (no percentile re-scaling)
        alpha = 1.0 - (H / Hmax)

        # Clean up numerics and clamp to [floor, 1]
        alpha = np.nan_to_num(alpha, nan=float(floor), posinf=float(floor), neginf=1.0)
        return np.clip(alpha, float(floor), 1.0)

def plot_alpha_vs_obstacle_distance_post_appearance(
    monkey, experiment, base_dir,
    file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
    stride=4,                # thinning: keep every k-th sample
    min_dt=None,             # thinning: keep if >= this seconds since last kept
    min_dist=None,           # thinning: keep if moved >= this distance since last kept
    nbins=12,                # common bin count across trials
    min_alpha=0.2,           # floor after per-trial robust normalization
    invert_x=True,           # invert x-axis (near on right)
    only_answer=None,        # 1 to keep only correct, 0/!=1 for incorrect, None for all
    task=None,
    show_per_trial=False     # draw faint per-trial lines (can be busy)
):
    """
    Like plot_alpha_vs_distance_average, but for the OBSTACLE plot we use ONLY
    samples at/after obstacle appearance in 'appearing obstacle' tasks.

    Returns a dict with the OBSTACLE summary:
      - 'centers','mean','lo','hi','n_trials_per_bin','r_pooled','r_trials'
      - 'obstacle_appeared_do_mean','_std','_values'
      - 'ai_control_window': mean ± SD of ON→OFF distances (obstacle)
    """
    import os, glob, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    # ---------------- helpers ----------------
    def safe_pearsonr(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan, np.nan
        try:
            r, p = pearsonr(x[m], y[m])
        except Exception:
            r, p = np.nan, np.nan
        return r, p

    def xz_from(pos, i=None):
        if pos is None:
            return None
        if i is not None:
            if isinstance(pos, dict):
                return (pos["x"][i], pos["z"][i])
        else:
            if isinstance(pos, dict):
                return (float(pos[0]), float(pos[2]))
        try:
            return (float(pos[0]), float(pos[2]))
        except Exception:
            return None

    def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None,
                     use_output_ts=True):
        idx = list(range(len(keep_idx)))
        if stride and stride > 1:
            idx = idx[::stride]

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

        if min_dist is not None and len(idx) > 1:
            kept_rel = [idx[0]]
            last_p = pos[idx[0]]
            for j in idx[1:]:
                if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                    kept_rel.append(j); last_p = pos[j]
            idx = kept_rel
        return idx

    def build_bins_from_pooled(d_all, nb):
        d_all = np.asarray(d_all, float)
        d_all = d_all[np.isfinite(d_all)]
        if d_all.size < 5:
            return None, None
        lo, hi = np.percentile(d_all, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(d_all), np.nanmax(d_all)
        edges = np.linspace(lo, hi, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    def per_trial_binned_means(d, a, edges):
        d = np.asarray(d, float); a = np.asarray(a, float)
        good = np.isfinite(d) & np.isfinite(a)
        if good.sum() == 0:
            return np.full(len(edges) - 1, np.nan)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d >= edges[i]) & (d < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a[m])
        return out

    def aggregate_across_trials(per_trial_mat):
        per_trial_mat = np.asarray(per_trial_mat, float)
        mean = np.nanmean(per_trial_mat, axis=0)
        std = np.nanstd(per_trial_mat, axis=0, ddof=1)
        n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
        sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
        return mean, lo, hi, n

    def pooled_binned_means(d_all, a_all, edges):
        d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
        good = np.isfinite(d_all) & np.isfinite(a_all)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a_all[m])
        return out

    # ---------------- collect per-trial OBSTACLE data ----------------
    if file_path is None:
        data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
        pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    else:
        pkl_files = [file_path]

    trials_obst = []
    trials_alpha_o = []
    r_trials_obst = []
    pooled_d_o, pooled_a_o = [], []
    appear_do_to_obstacle = []      # distance to OBSTACLE at appearance
    ai_control_bands_do = []        # (dist_to_obstacle_on, dist_to_obstacle_off) pairs

    # you already have this elsewhere; re-use your version
    all_trials, all_correct, all_incorrect, all_training, all_channels, \
        nb_channels, pkl_files, ai_trials = load_files(
            experiment, monkey, base_dir=base_dir
        )

    for session in all_trials:
        for trial in session:
            # ---------------- trial-level filters ----------------
            if only_answer is not None:
                ans = trial.answer
                if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
                    continue

            if task == "AI Respawn":
                if getattr(trial, "targetJumpTime", None) is None:
                    continue

            ai_recs = trial.aiVelocities
            if not ai_recs:
                continue

            # Skip if aiVelocity is zero at the start
            start_idx = next((i for i, rec in enumerate(ai_recs)
                              if np.linalg.norm(rec['Output']) > 0), None)
            if start_idx is None:
                continue
            ai_recs = ai_recs[start_idx:]

            # Keep samples with valid position
            xs, zs, keep_idx = [], [], []
            for i, rec in enumerate(ai_recs):
                p = xz_from(trial.avatarTrajectory, i)
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
            if len(keep_idx) < 3:
                continue
            pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

            # Entropy for ALL records, then keep
            ent_raw = []
            for rec in ai_recs:
                e = rec.get("EntropyLb")
                if isinstance(e, (list, tuple, np.ndarray)):
                    e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
                ent_raw.append(np.nan if e is None else float(e))
            ent_raw = np.asarray(ent_raw, float)
            ent_kept = ent_raw[keep_idx]

            # Thinning (relative indices)
            idx_rel = thin_indices(pos_full, keep_idx, ai_recs,
                                   stride=stride, min_dt=min_dt,
                                   min_dist=min_dist, use_output_ts=True)
            if len(idx_rel) < 3:
                continue

            pos = pos_full[idx_rel]
            ent_kept = ent_kept[idx_rel]

            # --- time vector aligned with pos / ent_kept ---
            traj_t_all = None
            if isinstance(trial.avatarTrajectory, dict):
                traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []),
                                        dtype=float)
            if traj_t_all is not None and traj_t_all.size > 0:
                traj_t_kept = traj_t_all[keep_idx][idx_rel]
                # NEW: fixed obstacle start time = first kept sample (first non-zero Output, valid pos, after thinning)
            if event_attr is None:
                if traj_t_kept.size == 0:
                    continue
                t_event = float(traj_t_kept[0])
            else:
                traj_t_kept = None

            # Alpha (same α used for target & obstacle; we only keep obstacle here)
            alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

            # Distances to obstacle (full, before any masking)
            obs = xz_from(trial.obstaclePosition)
            d_obs_full = (np.linalg.norm(pos - np.array(obs)[None, :], axis=1)
                          if obs is not None else
                          np.full(len(pos), np.nan))

            # ---------- AI CONTROL ON→OFF DISTANCES (OBSTACLE) ----------
            try:
                traj_times = None
                if isinstance(trial.avatarTrajectory, dict):
                    traj_times = np.asarray(trial.avatarTrajectory.get("time", []),
                                            dtype=float)
                if traj_times is None or traj_times.size == 0:
                    raise ValueError

                def _to_array(val):
                    if val is None:
                        return np.array([], dtype=float)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.asarray(val, dtype=float)
                    else:
                        arr = np.array([float(val)], dtype=float)
                    return arr[np.isfinite(arr)]

                v_on  = trial.aiControlOn
                v_off = trial.aiControlOff
                on_arr  = _to_array(v_on)
                off_arr = _to_array(v_off)

                n_pairs = min(on_arr.size, off_arr.size)
                if n_pairs > 0:
                    obs_arr = np.array(obs, float) if (obs is not None
                                                       and np.all(np.isfinite(obs))) else None

                    for k in range(n_pairs):
                        t_on  = on_arr[k]
                        t_off = off_arr[k]

                        idx_on  = int(np.nanargmin(np.abs(traj_times - t_on)))
                        idx_off = int(np.nanargmin(np.abs(traj_times - t_off)))

                        pos_on  = xz_from(trial.avatarTrajectory, idx_on)
                        pos_off = xz_from(trial.avatarTrajectory, idx_off)
                        if pos_on is None or pos_off is None:
                            continue

                        pos_on  = np.array(pos_on, float)
                        pos_off = np.array(pos_off, float)
                        if not (np.all(np.isfinite(pos_on)) and
                                np.all(np.isfinite(pos_off))):
                            continue

                        if obs_arr is not None:
                            do_on  = float(np.linalg.norm(pos_on  - obs_arr))
                            do_off = float(np.linalg.norm(pos_off - obs_arr))
                        else:
                            do_on = do_off = np.nan

                        if np.isfinite(do_on) and np.isfinite(do_off):
                            ai_control_bands_do.append((do_on, do_off))

            except Exception:
                pass  # skip AI-control bands if anything goes wrong

            # ---------- OBSTACLE APPEARED DISTANCE (in obstacle units) ----------
            if task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
                try:
                    t_app = getattr(trial, "obstacleAppearedTime", None) \
                            if not isinstance(trial, dict) else trial.get("obstacleAppearedTime", None)
                    if (t_app is not None) and isinstance(trial.avatarTrajectory, dict):
                        traj_t = np.asarray(trial.avatarTrajectory.get("time", []),
                                            dtype=float)
                        if traj_t.size >= 1:
                            idx_app = int(np.nanargmin(np.abs(traj_t - float(t_app))))
                            pos_app = xz_from(trial.avatarTrajectory, idx_app)
                            if (pos_app is not None) and np.all(np.isfinite(pos_app)):
                                pos_app = np.array(pos_app, float)
                                if obs is not None and np.all(np.isfinite(obs)):
                                    appear_do_to_obstacle.append(
                                        float(np.linalg.norm(
                                            pos_app - np.array(obs, float)))
                                    )
                except Exception:
                    pass

            # ---------- NEW: drop pre-appearance samples for OBSTACLE curve ----------
            d_obs_for_plot = d_obs_full.copy()
            alpha_obs = alpha_vals.copy()

            if task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
                t_app = getattr(trial, "obstacleAppearedTime", None) \
                        if not isinstance(trial, dict) else trial.get("obstacleAppearedTime", None)
                if (t_app is not None) and (traj_t_kept is not None):
                    mask_post = traj_t_kept >= float(t_app)
                    if mask_post.sum() >= 3:
                        d_obs_for_plot = d_obs_full[mask_post]
                        alpha_obs      = alpha_vals[mask_post]
                    else:
                        d_obs_for_plot = np.array([], dtype=float)
                        alpha_obs      = np.array([], dtype=float)

            # ---------- store OBSTACLE-only data ----------
            if np.isfinite(d_obs_for_plot).any():
                pooled_d_o.append(d_obs_for_plot)
                pooled_a_o.append(alpha_obs)
                trials_obst.append(d_obs_for_plot)
                trials_alpha_o.append(alpha_obs)

                m = np.isfinite(d_obs_for_plot) & np.isfinite(alpha_obs)
                r_trials_obst.append(
                    np.corrcoef(d_obs_for_plot[m], alpha_obs[m])[0, 1]
                    if m.sum() >= 3 else np.nan
                )

    # ---------------- aggregate & plot OBSTACLE results ----------------
    result = {}

    if not pooled_d_o:
        print("[WARN] No valid OBSTACLE samples across trials.")
        return result

    pooled_d_o = np.concatenate(pooled_d_o)
    pooled_a_o = np.concatenate(pooled_a_o)

    if pooled_d_o.size == 0:
        print("[WARN] No valid OBSTACLE samples after appearance masking.")
        return result

    edges_o, centers_o = build_bins_from_pooled(pooled_d_o, nbins)
    if edges_o is None:
        print("[WARN] Not enough OBSTACLE data to build bins.")
        return result

    per_trial_mat_o = [
        per_trial_binned_means(d, a, edges_o)
        for d, a in zip(trials_obst, trials_alpha_o)
    ]
    per_trial_mat_o = np.asarray(per_trial_mat_o, float)

    mean_o, lo_o, hi_o, n_o = aggregate_across_trials(per_trial_mat_o)
    pooled_curve_o = pooled_binned_means(pooled_d_o, pooled_a_o, edges_o)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    if show_per_trial:
        for row in per_trial_mat_o:
            ax.plot(centers_o, row, lw=0.8, alpha=0.2)

    ax.fill_between(centers_o, lo_o, hi_o, alpha=0.25,
                    label="Across-trials 95% CI")
    ax.plot(centers_o, mean_o, lw=2.5, label="Across-trials mean")
    ax.plot(centers_o, pooled_curve_o, '--', lw=2,
            label="Pooled (all samples)")

    # ----- mark OBSTACLE APPEARED distance (in obstacle units) -----
    if task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2") and len(appear_do_to_obstacle) > 0:
        mu_app_do = float(np.mean(appear_do_to_obstacle))
        sd_app_do = float(np.std(appear_do_to_obstacle, ddof=1)) \
            if len(appear_do_to_obstacle) > 1 else 0.0

        ax.axvline(mu_app_do, color="tab:purple", ls='-', lw=2,
                   label="Obstacle appeared (mean)")
        if np.isfinite(sd_app_do) and sd_app_do > 0:
            ax.axvspan(mu_app_do - sd_app_do,
                       mu_app_do + sd_app_do,
                       color="tab:purple", alpha=0.12,
                       label="Appeared ±1 SD")

        y_top = ax.get_ylim()[1]
        ax.annotate(fr"appear μ={mu_app_do:.2f} ± {sd_app_do:.2f}",
                    xy=(mu_app_do, y_top), xytext=(5, -6),
                    textcoords='offset points', ha='left', va='top',
                    color="tab:purple")

    # ----- GREEN BANDS: AI CONTROL ON→OFF (OBSTACLE DISTANCE, mean ± SD) -----
    if len(ai_control_bands_do) > 0:
        starts = []
        ends   = []
        for do_on, do_off in ai_control_bands_do:
            if not (np.isfinite(do_on) and np.isfinite(do_off)):
                continue
            left, right = sorted((do_on, do_off))
            starts.append(left)
            ends.append(right)

        if starts:
            starts = np.asarray(starts, float)
            ends   = np.asarray(ends,   float)

            mean_start = np.nanmean(starts)
            mean_end   = np.nanmean(ends)
            std_start  = np.nanstd(starts)
            std_end    = np.nanstd(ends)

            low  = mean_start - std_start
            high = mean_end   + std_end

            ax.axvspan(
                low, high,
                color='green', alpha=0.15,
                label="AI override window (mean ± 1 SD)",
                zorder=3
            )
            ax.axvline(
                mean_start, color='green',
                linestyle='--', linewidth=1.2,
                label="AI override window (mean)",
                zorder=4
            )
            ax.axvline(
                mean_end, color='green',
                linestyle='--', linewidth=1.2,
                zorder=4
            )

            result['ai_control_window'] = {
                "mean_start": float(mean_start),
                "mean_end": float(mean_end),
                "std_start": float(std_start),
                "std_end": float(std_end),
                "starts": starts.tolist(),
                "ends": ends.tolist(),
            }

    # pooled Pearson r (and p)  — uses OBSTACLE pools
    r_pool, p_pool = safe_pearsonr(pooled_d_o, pooled_a_o)
    med_r = np.nanmedian(
        [r for r in np.asarray(r_trials_obst, float) if np.isfinite(r)]
    ) if len(r_trials_obst) else np.nan

    def fmt_p(p):
        if not np.isfinite(p):
            return "p=NA"
        return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

    ax.text(
        0.02, 0.95,
        f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
        transform=ax.transAxes, ha='left', va='top', fontsize=10
    )

    ax.set_xlabel("Distance to obstacle center")
    ax.set_ylabel("Posterior Confidence Index")
    ax.set_title(f"{monkey} – {experiment}\nAcross-trials α vs. distance to obstacle\n(post-appearance only in appearing-obstacle tasks)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    if invert_x:
        ax.invert_xaxis()
    plt.tight_layout()

    try:
        save_plot(fig,
                  f"alpha_vs_distance_OBSTACLE_POST_APP_across_{monkey}_{experiment}",
                  subfolder="Entropy")
    except Exception:
        pass
    plt.show(); plt.close(fig)

    result['obstacle'] = {
        'centers': centers_o,
        'mean': mean_o,
        'lo': lo_o,
        'hi': hi_o,
        'n_trials_per_bin': n_o,
        'r_pooled': r_pool,
        'r_trials': r_trials_obst,
        'obstacle_appeared_do_mean': float(np.mean(appear_do_to_obstacle))
            if len(appear_do_to_obstacle) else np.nan,
        'obstacle_appeared_do_std': float(np.std(appear_do_to_obstacle, ddof=1))
            if len(appear_do_to_obstacle) > 1 else
            (0.0 if len(appear_do_to_obstacle) == 1 else np.nan),
        'obstacle_appeared_do_values': appear_do_to_obstacle,
    }

    return result

def plot_alpha_vs_time_and_target_distance(
    monkey, experiment, base_dir,
    file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
    stride=4,                # thinning: keep every k-th sample
    min_dt=None,             # thinning: keep if >= this seconds since last kept
    min_dist=None,           # thinning: keep if moved >= this distance since last kept
    nbins_time=24,           # bins for time
    nbins_dist=12,           # bins for distance-to-target
    min_alpha=0.2,           # floor after per-trial robust normalization
    only_answer=None,        # 1 to keep only correct, 0/!=1 for incorrect, None for all
    task=None,
    show_per_trial=False     # draw faint per-trial lines
):
    """
    Compute:
      (1) α(t) vs time from trial start
      (2) α vs distance-to-target over the whole trial

    Uses the same thinning + robust_alpha + load_files as your main function.
    """
    import os, glob
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    # --------- helpers copied/simplified from your function ---------
    def safe_pearsonr(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan, np.nan
        try:
            r, p = pearsonr(x[m], y[m])
        except Exception:
            r, p = np.nan, np.nan
        return r, p

    def xz_from(pos, i=None):
        if pos is None:
            return None
        if i is not None:
            if isinstance(pos, dict):
                return (pos["x"][i], pos["z"][i])
        else:
            if isinstance(pos, dict):
                return (float(pos[0]), float(pos[2]))
        try:
            return (float(pos[0]), float(pos[2]))
        except Exception:
            return None

    def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
        idx = list(range(len(keep_idx)))
        if stride and stride > 1:
            idx = idx[::stride]
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
        if min_dist is not None and len(idx) > 1:
            kept_rel = [idx[0]]
            last_p = pos[idx[0]]
            for j in idx[1:]:
                if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                    kept_rel.append(j); last_p = pos[j]
            idx = kept_rel
        return idx

    def build_bins_from_pooled(d_all, nb):
        d_all = np.asarray(d_all, float)
        d_all = d_all[np.isfinite(d_all)]
        if d_all.size < 5:
            return None, None
        lo, hi = np.percentile(d_all, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(d_all), np.nanmax(d_all)
        edges = np.linspace(lo, hi, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    def per_trial_binned_means(d, a, edges):
        d = np.asarray(d, float); a = np.asarray(a, float)
        good = np.isfinite(d) & np.isfinite(a)
        if good.sum() == 0:
            return np.full(len(edges) - 1, np.nan)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d >= edges[i]) & (d < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a[m])
        return out

    def aggregate_across_trials(per_trial_mat):
        per_trial_mat = np.asarray(per_trial_mat, float)
        mean = np.nanmean(per_trial_mat, axis=0)
        std = np.nanstd(per_trial_mat, axis=0, ddof=1)
        n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
        sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
        return mean, lo, hi, n

    def pooled_binned_means(d_all, a_all, edges):
        d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
        good = np.isfinite(d_all) & np.isfinite(a_all)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a_all[m])
        return out

    # --------- collect trials (same load_files as your main code) ---------
    all_trials, all_correct, all_incorrect, all_training, all_channels, \
        nb_channels, pkl_files, ai_trials = load_files(
            experiment, monkey, base_dir=base_dir
        )

    # containers
    trials_time = []
    trials_alpha_time = []
    pooled_t, pooled_a_t = [], []
    r_trials_time = []

    trials_dist = []
    trials_alpha_dist = []
    pooled_d, pooled_a_d = [], []
    r_trials_dist = []

    for session in all_trials:
        for trial in session:
            # filter on correctness if requested
            if only_answer is not None:
                ans = trial.answer
                if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
                    continue

            ai_recs = trial.aiVelocities
            if not ai_recs:
                continue

            # skip if aiVelocity is zero at start
            start_idx = next((i for i, rec in enumerate(ai_recs)
                              if np.linalg.norm(rec['Output']) > 0), None)
            if start_idx is None:
                continue
            ai_recs = ai_recs[start_idx:]

            # positions from avatarTrajectory
            xs, zs, keep_idx = [], [], []
            for i, rec in enumerate(ai_recs):
                p = xz_from(trial.avatarTrajectory, i)
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
            if len(keep_idx) < 3:
                continue
            pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

            # entropy
            ent_raw = []
            for rec in ai_recs:
                e = rec.get("EntropyLb")
                if isinstance(e, (list, tuple, np.ndarray)):
                    e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
                ent_raw.append(np.nan if e is None else float(e))
            ent_raw = np.asarray(ent_raw, float)
            ent_kept = ent_raw[keep_idx]

            # thinning
            idx_rel = thin_indices(pos_full, keep_idx, ai_recs,
                                   stride=stride, min_dt=min_dt,
                                   min_dist=min_dist, use_output_ts=True)
            if len(idx_rel) < 3:
                continue

            pos = pos_full[idx_rel]
            ent_kept = ent_kept[idx_rel]

            # time vector aligned with pos / ent_kept
            traj_t_all = None
            if isinstance(trial.avatarTrajectory, dict):
                traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []),
                                        dtype=float)
            if traj_t_all is None or traj_t_all.size == 0:
                continue
            try:
                traj_t_kept = traj_t_all[keep_idx][idx_rel]
            except Exception:
                continue

            # time from trial start (relative)
            t_rel = traj_t_kept - traj_t_kept[0]

            # alpha
            alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

            # distance to TARGET (for whole trial)
            tgt = xz_from(trial.targetPosition)
            d_tgt = (np.linalg.norm(pos - np.array(tgt)[None, :], axis=1)
                     if tgt is not None else
                     np.full(len(pos), np.nan))

            # require some finite values
            good_time = np.isfinite(t_rel) & np.isfinite(alpha_vals)
            if good_time.sum() < 3:
                continue

            # store time series
            t_rel = t_rel[good_time]
            a_t   = alpha_vals[good_time]
            trials_time.append(t_rel)
            trials_alpha_time.append(a_t)
            pooled_t.append(t_rel)
            pooled_a_t.append(a_t)
            r_trials_time.append(
                np.corrcoef(t_rel, a_t)[0, 1] if t_rel.size >= 3 else np.nan
            )

            # store distance-to-target series (full trial)
            good_dist = np.isfinite(d_tgt) & np.isfinite(alpha_vals)
            if good_dist.sum() >= 3:
                d_good = d_tgt[good_dist]
                a_d    = alpha_vals[good_dist]
                trials_dist.append(d_good)
                trials_alpha_dist.append(a_d)
                pooled_d.append(d_good)
                pooled_a_d.append(a_d)
                r_trials_dist.append(
                    np.corrcoef(d_good, a_d)[0, 1] if d_good.size >= 3 else np.nan
                )

    result = {}

    # ---------- α vs TIME ----------
    if pooled_t:
        pooled_t_all = np.concatenate(pooled_t)
        pooled_a_t_all = np.concatenate(pooled_a_t)

        edges_time, centers_time = build_bins_from_pooled(pooled_t_all, nbins_time)
        if edges_time is not None:
            per_trial_mat_time = [
                per_trial_binned_means(t, a, edges_time)
                for t, a in zip(trials_time, trials_alpha_time)
            ]
            per_trial_mat_time = np.asarray(per_trial_mat_time, float)
            mean_time, lo_time, hi_time, n_time = aggregate_across_trials(per_trial_mat_time)
            pooled_curve_time = pooled_binned_means(pooled_t_all, pooled_a_t_all, edges_time)

            fig, ax = plt.subplots(figsize=(8.8, 5.0))
            if show_per_trial:
                for row in per_trial_mat_time:
                    ax.plot(centers_time, row, lw=0.8, alpha=0.2)

            ax.fill_between(centers_time, lo_time, hi_time, alpha=0.25,
                            label="Across-trials 95% CI")
            ax.plot(centers_time, mean_time, lw=2.5, label="Across-trials mean")
            ax.plot(centers_time, pooled_curve_time, '--', lw=2,
                    label="Pooled (all samples)")

            r_pool_time, p_pool_time = safe_pearsonr(pooled_t_all, pooled_a_t_all)
            med_r_time = np.nanmedian(
                [r for r in np.asarray(r_trials_time, float) if np.isfinite(r)]
            ) if len(r_trials_time) else np.nan

            def fmt_p(p):
                if not np.isfinite(p):
                    return "p=NA"
                return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

            ax.text(0.02, 0.95,
                    f"r_pooled={r_pool_time:.2f} ({fmt_p(p_pool_time)})\n"
                    f"median r_trial={med_r_time:.2f}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=10)

            ax.set_xlabel("Time from trial start (s)")
            ax.set_ylabel("Posterior Confidence Index")
            ax.set_title(f"{monkey} – {experiment}\nα vs time from trial start")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True)
            plt.tight_layout()
            try:
                save_plot(fig,
                          f"alpha_vs_TIME_from_start_{monkey}_{experiment}",
                          subfolder="Entropy")
            except Exception:
                pass
            plt.show(); plt.close(fig)

            result['time'] = {
                'centers': centers_time,
                'mean': mean_time,
                'lo': lo_time,
                'hi': hi_time,
                'n_trials_per_bin': n_time,
                'r_pooled': r_pool_time,
                'r_trials': r_trials_time,
            }

    # ---------- α vs DISTANCE TO TARGET (whole trial) ----------
    if pooled_d:
        pooled_d_all = np.concatenate(pooled_d)
        pooled_a_d_all = np.concatenate(pooled_a_d)

        edges_d, centers_d = build_bins_from_pooled(pooled_d_all, nbins_dist)
        if edges_d is not None:
            per_trial_mat_dist = [
                per_trial_binned_means(d, a, edges_d)
                for d, a in zip(trials_dist, trials_alpha_dist)
            ]
            per_trial_mat_dist = np.asarray(per_trial_mat_dist, float)
            mean_d, lo_d, hi_d, n_d = aggregate_across_trials(per_trial_mat_dist)
            pooled_curve_d = pooled_binned_means(pooled_d_all, pooled_a_d_all, edges_d)

            fig, ax = plt.subplots(figsize=(8.8, 5.0))
            if show_per_trial:
                for row in per_trial_mat_dist:
                    ax.plot(centers_d, row, lw=0.8, alpha=0.2)

            ax.fill_between(centers_d, lo_d, hi_d, alpha=0.25,
                            label="Across-trials 95% CI")
            ax.plot(centers_d, mean_d, lw=2.5, label="Across-trials mean")
            ax.plot(centers_d, pooled_curve_d, '--', lw=2,
                    label="Pooled (all samples)")

            r_pool_d, p_pool_d = safe_pearsonr(pooled_d_all, pooled_a_d_all)
            med_r_d = np.nanmedian(
                [r for r in np.asarray(r_trials_dist, float) if np.isfinite(r)]
            ) if len(r_trials_dist) else np.nan

            def fmt_p(p):
                if not np.isfinite(p):
                    return "p=NA"
                return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

            ax.text(0.02, 0.95,
                    f"r_pooled={r_pool_d:.2f} ({fmt_p(p_pool_d)})\n"
                    f"median r_trial={med_r_d:.2f}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=10)

            ax.set_xlabel("Distance to target center")
            ax.set_ylabel("Posterior Confidence Index")
            ax.set_title(f"{monkey} – {experiment}\nα vs distance to target (whole trial)")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True)
            # if you want the “near target on the right” convention:
            # ax.invert_xaxis()
            plt.tight_layout()
            try:
                save_plot(fig,
                          f"alpha_vs_TARGET_DISTANCE_whole_{monkey}_{experiment}",
                          subfolder="Entropy")
            except Exception:
                pass
            plt.show(); plt.close(fig)

            result['target'] = {
                'centers': centers_d,
                'mean': mean_d,
                'lo': lo_d,
                'hi': hi_d,
                'n_trials_per_bin': n_d,
                'r_pooled': r_pool_d,
                'r_trials': r_trials_dist,
            }

    return result
# def plot_alpha_vs_time_from_obstacle_appearance(
#     monkey, experiment, base_dir,
#     file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
#     stride=4,                # thinning: keep every k-th sample
#     min_dt=None,             # thinning: keep if >= this seconds since last kept
#     min_dist=None,           # thinning: keep if moved >= this distance since last kept
#     t_window=(-2.0, 4.0),    # time window [s] relative to obstacle appearance
#     nbins=24,                # common bin count across trials
#     min_alpha=0.2,           # floor after per-trial robust normalization
#     only_answer=None,        # 1 to keep only correct, 0/!=1 for incorrect, None for all
#     task=None,
#     show_per_trial=False     # draw faint per-trial lines
# ):
#     """
#     α (= 1 − normalized EntropyLb) vs *time from obstacle appearance*.

#     For each trial:
#       - build α(t) from EntropyLb with robust per-trial normalization
#       - build time vector aligned with α
#       - compute t_rel = time - obstacleAppearedTime
#       - keep t_rel within [t_window[0], t_window[1]]
#       - bin α vs t_rel using a common set of time bins
#       - average across trials, show mean ± 95% CI and pooled curve

#     Also computes:
#       - pre/early/late α around the real event,
#       - dip depth and recovery time,
#       - pre/early α around a random pseudo-event (control).
#     """

#     import os, glob
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import pearsonr, wilcoxon

#     # ---------------- helpers ----------------
#     def safe_pearsonr(x, y):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         m = np.isfinite(x) & np.isfinite(y)
#         if m.sum() < 3:
#             return np.nan, np.nan
#         try:
#             r, p = pearsonr(x[m], y[m])
#         except Exception:
#             r, p = np.nan, np.nan
#         return r, p

#     def xz_from(pos, i=None):
#         if pos is None:
#             return None
#         if i is not None:
#             if isinstance(pos, dict):
#                 return (pos["x"][i], pos["z"][i])
#         else:
#             if isinstance(pos, dict):
#                 return (float(pos[0]), float(pos[2]))
#         try:
#             return (float(pos[0]), float(pos[2]))
#         except Exception:
#             return None

#     def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
#         idx = list(range(len(keep_idx)))
#         if stride and stride > 1:
#             idx = idx[::stride]
#         if min_dt is not None:
#             ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
#             ts_vals = []
#             for k in keep_idx:
#                 t = ai_recs[k].get(ts_key)
#                 if hasattr(t, "timestamp"):
#                     t = t.timestamp()
#                 ts_vals.append(np.nan if t is None else float(t))
#             kept_rel, last_t = [], None
#             for j in idx:
#                 tj = ts_vals[j]
#                 if np.isnan(tj) or last_t is None or (tj - last_t) >= float(min_dt):
#                     kept_rel.append(j); last_t = tj
#             idx = kept_rel
#         if min_dist is not None and len(idx) > 1:
#             kept_rel = [idx[0]]
#             last_p = pos[idx[0]]
#             for j in idx[1:]:
#                 if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
#                     kept_rel.append(j); last_p = pos[j]
#             idx = kept_rel
#         return idx

#     def build_bins_from_pooled(d_all, nb):
#         d_all = np.asarray(d_all, float)
#         d_all = d_all[np.isfinite(d_all)]
#         if d_all.size < 5:
#             return None, None
#         lo, hi = np.percentile(d_all, [1, 99])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = np.nanmin(d_all), np.nanmax(d_all)
#         edges = np.linspace(lo, hi, nb + 1)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#         return edges, centers

#     def per_trial_binned_means(d, a, edges):
#         d = np.asarray(d, float); a = np.asarray(a, float)
#         good = np.isfinite(d) & np.isfinite(a)
#         if good.sum() == 0:
#             return np.full(len(edges) - 1, np.nan)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d >= edges[i]) & (d < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a[m])
#         return out

#     def aggregate_across_trials(per_trial_mat):
#         per_trial_mat = np.asarray(per_trial_mat, float)
#         mean = np.nanmean(per_trial_mat, axis=0)
#         std = np.nanstd(per_trial_mat, axis=0, ddof=1)
#         n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
#         sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
#         lo = mean - 1.96 * sem
#         hi = mean + 1.96 * sem
#         return mean, lo, hi, n

#     def pooled_binned_means(d_all, a_all, edges):
#         d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
#         good = np.isfinite(d_all) & np.isfinite(a_all)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a_all[m])
#         return out

#     # ==== NEW: simple window mean helper ====
#     def window_mean(t, a, w):
#         """Mean α in time window w = (t_min, t_max) on a single trial."""
#         t = np.asarray(t, float); a = np.asarray(a, float)
#         m = (t >= w[0]) & (t <= w[1]) & np.isfinite(t) & np.isfinite(a)
#         return np.nan if m.sum() == 0 else float(np.nanmean(a[m]))

#     # ==== NEW: define windows (all in seconds, because t_rel is in s) ====
#     pre_window      = (-600, -200)
#     early_window    = (0.0,  400)
#     late_window     = (1600,  2400)
#     dip_window      = (0.0,  800)   # where we search for the minimum after event
#     plateau_window  = (1600,  2600)   # “late” plateau for recovery
#     pseudo_range    = (-1200, -400)  # center of random pseudo-events (no real event)

#     # ---------------- determine "appearing obstacle" condition ----------------
#     effective_task = task if task is not None else experiment
#     is_appearing = effective_task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2")
#     if not is_appearing:
#         print("[WARN] plot_alpha_vs_time_from_obstacle_appearance is only meaningful for appearing-obstacle tasks.")

#     # ---------------- collect data ----------------
#     if file_path is None:
#         data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
#         pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#     else:
#         pkl_files = [file_path]

#     all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, \
#         pkl_files, ai_trials = load_files(experiment, monkey, base_dir=base_dir)

#     trials_t = []
#     trials_alpha = []
#     pooled_t, pooled_a = [], []
#     r_trials_time = []

#     # ==== NEW: per-trial summary arrays ====
#     pre_vals, early_vals, late_vals = [], [], []
#     pre_pseudo_vals, early_pseudo_vals = [], []
#     dip_depths, recovery_times = [], []

#     tmin, tmax = float(t_window[0]), float(t_window[1])

#     for session in all_trials:
#         for trial in session:
#             if only_answer is not None:
#                 ans = trial.answer
#                 if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
#                     continue

#             ai_recs = trial.aiVelocities
#             if not ai_recs:
#                 continue

#             t_app = getattr(trial, "obstacleAppearedTime", None) \
#                     if not isinstance(trial, dict) else trial.get("obstacleAppearedTime", None)
#             if t_app is None:
#                 continue
#             t_app = float(t_app)

#             # Skip if aiVelocity is zero at the start
#             start_idx = next((i for i, rec in enumerate(ai_recs)
#                               if np.linalg.norm(rec['Output']) > 0), None)
#             if start_idx is None:
#                 continue
#             ai_recs = ai_recs[start_idx:]

#             # positions from avatarTrajectory
#             xs, zs, keep_idx = [], [], []
#             for i, rec in enumerate(ai_recs):
#                 p = xz_from(trial.avatarTrajectory, i)
#                 if p is None or np.isnan(p[0]) or np.isnan(p[1]):
#                     continue
#                 xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
#             if len(keep_idx) < 3:
#                 continue
#             pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

#             # entropy (full ai_recs, then restrict to keep_idx)
#             ent_raw = []
#             for rec in ai_recs:
#                 e = rec.get("EntropyLb")
#                 if isinstance(e, (list, tuple, np.ndarray)):
#                     e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
#                 ent_raw.append(np.nan if e is None else float(e))
#             ent_raw = np.asarray(ent_raw, float)
#             ent_kept = ent_raw[keep_idx]

#             # thinning indices
#             idx_rel = thin_indices(pos_full, keep_idx, ai_recs,
#                                    stride=stride, min_dt=min_dt,
#                                    min_dist=min_dist, use_output_ts=True)
#             if len(idx_rel) < 3:
#                 continue

#             pos = pos_full[idx_rel]
#             ent_kept = ent_kept[idx_rel]

#             # time vector aligned with pos / ent_kept
#             traj_t_all = None
#             if isinstance(trial.avatarTrajectory, dict):
#                 traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []),
#                                         dtype=float)
#             if traj_t_all is None or traj_t_all.size == 0:
#                 continue
#             try:
#                 traj_t_kept = traj_t_all[keep_idx][idx_rel]
#             except Exception:
#                 continue

#             # Alpha
#             alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

#             # relative time to obstacle appearance
#             t_rel = traj_t_kept - t_app

#             # mask window
#             mask = (t_rel >= tmin) & (t_rel <= tmax) & np.isfinite(alpha_vals) & np.isfinite(t_rel)
#             if mask.sum() < 3:
#                 continue

#             t_rel_trial = t_rel[mask]
#             alpha_trial = alpha_vals[mask]

#             trials_t.append(t_rel_trial)
#             trials_alpha.append(alpha_trial)
#             pooled_t.append(t_rel_trial)
#             pooled_a.append(alpha_trial)

#             r_trials_time.append(
#                 np.corrcoef(t_rel_trial, alpha_trial)[0, 1] if t_rel_trial.size >= 3 else np.nan
#             )

#             # ==== NEW: real-event pre/early/late, dip depth & recovery ====
#             pre  = window_mean(t_rel_trial, alpha_trial, pre_window)
#             early = window_mean(t_rel_trial, alpha_trial, early_window)
#             late  = window_mean(t_rel_trial, alpha_trial, late_window)

#             pre_vals.append(pre)
#             early_vals.append(early)
#             late_vals.append(late)

#             # dip search (after event) and recovery
#             dip_depth = np.nan
#             recovery_time = np.nan

#             if np.isfinite(pre):
#                 # dip
#                 m_dip = (t_rel_trial >= dip_window[0]) & (t_rel_trial <= dip_window[1])
#                 if np.any(m_dip):
#                     a_seg = alpha_trial[m_dip]
#                     t_seg = t_rel_trial[m_dip]
#                     try:
#                         idx_min = int(np.nanargmin(a_seg))
#                         alpha_min = float(a_seg[idx_min])
#                         t_min = float(t_seg[idx_min])
#                         dip_depth = alpha_min - pre
#                     except (ValueError, TypeError):
#                         alpha_min = np.nan
#                         t_min = np.nan

#                 # recovery: time to half of plateau
#                 alpha_plateau = window_mean(t_rel_trial, alpha_trial, plateau_window)
#                 if np.isfinite(alpha_plateau) and alpha_plateau > pre:
#                     thresh = pre + 0.5 * (alpha_plateau - pre)
#                     m_post = (t_rel_trial >= 0.0) & np.isfinite(alpha_trial)
#                     if np.any(m_post):
#                         t_post = t_rel_trial[m_post]
#                         a_post = alpha_trial[m_post]
#                         idx = np.where(a_post >= thresh)[0]
#                         if idx.size > 0:
#                             recovery_time = float(t_post[idx[0]])  # in seconds

#             dip_depths.append(dip_depth)
#             recovery_times.append(recovery_time)

#             # ==== NEW: pseudo-event control ====
#             # choose a random pseudo-event center in pseudo_range (no real event)
#             pseudo_t0 = np.random.uniform(pseudo_range[0], pseudo_range[1])
#             t_centered = t_rel_trial - pseudo_t0  # pseudo-event at 0
#             pre_p  = window_mean(t_centered, alpha_trial, pre_window)
#             early_p = window_mean(t_centered, alpha_trial, early_window)
#             pre_pseudo_vals.append(pre_p)
#             early_pseudo_vals.append(early_p)

#     result = {}

#     if not pooled_t:
#         print("[WARN] No valid samples around obstacle appearance.")
#         return result

#     pooled_t = np.concatenate(pooled_t)
#     pooled_a = np.concatenate(pooled_a)

#     edges_t, centers_t = build_bins_from_pooled(pooled_t, nbins)
#     if edges_t is None:
#         print("[WARN] Not enough data to build time bins.")
#         return result

#     per_trial_mat_t = [
#         per_trial_binned_means(t, a, edges_t)
#         for t, a in zip(trials_t, trials_alpha)
#     ]
#     per_trial_mat_t = np.asarray(per_trial_mat_t, float)

#     mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)
#     pooled_curve_t = pooled_binned_means(pooled_t, pooled_a, edges_t)

#     # ---------------- plot ----------------
#     fig, ax = plt.subplots(figsize=(8.8, 5.0))

#     x_centers_ms = centers_t * 1000.0  # convert to ms for plotting

#     if show_per_trial:
#         for row in per_trial_mat_t:
#             ax.plot(x_centers_ms, row, lw=0.8, alpha=0.2)

#     ax.fill_between(x_centers_ms, lo_t, hi_t, alpha=0.25, label="Across-trials 95% CI")
#     ax.plot(x_centers_ms, mean_t, lw=2.5, label="Across-trials mean")
#     ax.plot(x_centers_ms, pooled_curve_t, '--', lw=2, label="Pooled (all samples)")

#     # vertical line at appearance (t=0)
#     ax.axvline(0.0, color='tab:purple', ls='-', lw=2, label="Obstacle appearance (t=0)")

#     r_pool, p_pool = safe_pearsonr(pooled_t, pooled_a)
#     med_r = np.nanmedian(
#         [r for r in np.asarray(r_trials_time, float) if np.isfinite(r)]
#     ) if len(r_trials_time) else np.nan

#     def fmt_p(p):
#         if not np.isfinite(p):
#             return "p=NA"
#         return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

#     ax.text(
#         0.02, 0.95,
#         f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
#         transform=ax.transAxes, ha='left', va='top', fontsize=10
#     )

#     ax.set_xlabel("Time from obstacle appearance (ms)")
#     ax.set_ylabel("Posterior Confidence Index")
#     ax.set_title(f"{monkey} – {experiment}\nα vs time from obstacle appearance")
#     ax.grid(True, alpha=0.3)
#     ax.legend(frameon=True)
#     plt.tight_layout()

#     try:
#         save_plot(fig, f"alpha_vs_time_FROM_OBSTACLE_APP_{monkey}_{experiment}", subfolder="Entropy")
#     except Exception:
#         pass
#     plt.show(); plt.close(fig)

#     # ---------------- summaries for stats / bar plots ----------------
#     pre_vals   = np.asarray(pre_vals, float)
#     early_vals = np.asarray(early_vals, float)
#     late_vals  = np.asarray(late_vals, float)
#     delta_early = early_vals - pre_vals
#     delta_late  = late_vals  - pre_vals

#     pre_pseudo_vals   = np.asarray(pre_pseudo_vals, float)
#     early_pseudo_vals = np.asarray(early_pseudo_vals, float)
#     delta_pseudo = early_pseudo_vals - pre_pseudo_vals

#     dip_depths     = np.asarray(dip_depths, float)      # typically negative if real dip
#     recovery_times = np.asarray(recovery_times, float)  # in seconds

#     # real-event stats
#     def wilcoxon_or_nan(x):
#         x = np.asarray(x, float)
#         x = x[np.isfinite(x)]
#         if x.size < 3:
#             return np.nan
#         return float(wilcoxon(x, alternative='two-sided').pvalue)

#     p_delta_early = wilcoxon_or_nan(delta_early)
#     p_delta_late  = wilcoxon_or_nan(delta_late)
#     p_delta_pseudo = wilcoxon_or_nan(delta_pseudo)
#     p_dip_depth    = wilcoxon_or_nan(dip_depths)

#     result['time'] = {
#         'centers_s': centers_t,
#         'mean': mean_t,
#         'lo': lo_t,
#         'hi': hi_t,
#         'n_trials_per_bin': n_t,
#         'r_pooled': r_pool,
#         'r_trials': r_trials_time,
#         't_window': t_window,
#     }

#     # ==== NEW: event-locked window summary ====
#     result['event_windows'] = {
#         'pre_mean':   float(np.nanmean(pre_vals)),
#         'early_mean': float(np.nanmean(early_vals)),
#         'late_mean':  float(np.nanmean(late_vals)),
#         'delta_early_mean': float(np.nanmean(delta_early)),
#         'delta_early_median': float(np.nanmedian(delta_early)),
#         'delta_early_p': p_delta_early,
#         'delta_late_mean': float(np.nanmean(delta_late)),
#         'delta_late_median': float(np.nanmedian(delta_late)),
#         'delta_late_p': p_delta_late,
#         'n_trials': int(np.sum(np.isfinite(pre_vals) & np.isfinite(early_vals))),
#     }

#     # ==== NEW: dip depth and recovery time (for design-rule story) ====
#     result['dip_recovery'] = {
#         'dip_depth_mean': float(np.nanmean(dip_depths)),
#         'dip_depth_median': float(np.nanmedian(dip_depths)),
#         'dip_depth_p': p_dip_depth,
#         'recovery_time_median_s': float(np.nanmedian(recovery_times)),
#         'recovery_time_mean_s': float(np.nanmean(recovery_times)),
#         'recovery_time_values_s': recovery_times,
#     }

#     # ==== NEW: pseudo-event control ====
#     result['pseudo_control'] = {
#         'delta_pseudo_mean': float(np.nanmean(delta_pseudo)),
#         'delta_pseudo_median': float(np.nanmedian(delta_pseudo)),
#         'delta_pseudo_p': p_delta_pseudo,
#     }

#     return result
# def plot_alpha_vs_time_from_obstacle_appearance(
#     monkey, experiment, base_dir,
#     file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
#     stride=4,                # thinning: keep every k-th sample
#     min_dt=None,             # thinning: keep if >= this time since last kept (same units as timestamps)
#     min_dist=None,           # thinning: keep if moved >= this distance since last kept
#     t_window=(-2000.0, 3000.0),  # time window [ms] relative to event
#     nbins=24,                # common bin count across trials
#     min_alpha=0.2,           # floor after per-trial robust normalization
#     only_answer=None,        # 1 to keep only correct, 0/!=1 for incorrect, None for all
#     task=None,
#     show_per_trial=False,    # draw faint per-trial lines
#     fixed_ylim=(0.2, 0.75),  # y-limits for comparability across panels; set to None to auto
# ):
#     """
#     α (= 1 − normalized EntropyLb) vs time from event (obstacle appearance or target respawn).

#     For each trial:
#       - build α(t) from EntropyLb with robust per-trial normalization
#       - align time to event:
#           * Respawn tasks  -> targetJumpTime
#           * Obstacle tasks -> obstacleAppearedTime
#       - keep samples within t_window (ms)
#       - bin α vs t_rel using a common set of time bins; plot mean ± 95% CI

#     Metrics per trial (real events):
#       - reference level: mean α in [-200, 0] ms (around the event)
#       - dip depth: minimum α in [0, 800] ms minus reference
#       - recovery time: first time AFTER the dip minimum where α(t) ≥ reference

#     Pseudo-event control:
#       - several pseudo centers per trial drawn from:
#           * pre-event region:  [-1200, -400] ms
#           * post-event region: [1000, 2500] ms
#       - for each pseudo center, compute the same dip depth (min in [0, 800] − ref in [-200, 0])
#       - per-trial pseudo dip = mean across pseudo centers

#     Additional paired metric:
#       - real_vs_pseudo: per trial (real dip − pseudo dip), with mean/median and Wilcoxon vs 0.

#     Returns a dict with:
#       - 'time': binned mean/CI, r_pooled, etc.
#       - 'event_windows': pre/early/late means (for completeness)
#       - 'dip_recovery': dip depth stats + recovery times (ms)
#       - 'pseudo_control': pseudo dip depth stats
#       - 'real_vs_pseudo': paired real–pseudo contrast stats
#     """

#     import os, glob
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import pearsonr, wilcoxon

#     # ---------------- helpers ----------------
#     def safe_pearsonr(x, y):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         m = np.isfinite(x) & np.isfinite(y)
#         if m.sum() < 3:
#             return np.nan, np.nan
#         try:
#             r, p = pearsonr(x[m], y[m])
#         except Exception:
#             r, p = np.nan, np.nan
#         return r, p

#     def xz_from(pos, i=None):
#         if pos is None:
#             return None
#         if i is not None:
#             if isinstance(pos, dict):
#                 return (pos["x"][i], pos["z"][i])
#         else:
#             if isinstance(pos, dict):
#                 return (float(pos[0]), float(pos[2]))
#         try:
#             return (float(pos[0]), float(pos[2]))
#         except Exception:
#             return None

#     def thin_indices(pos, keep_idx, ai_recs,
#                      stride=None, min_dt=None, min_dist=None,
#                      use_output_ts=True):
#         """
#         Thinning indices for pos/ai_recs:
#           - stride: keep every k-th sample
#           - min_dt: min time difference between kept samples (same units as timestamps)
#           - min_dist: min spatial distance between kept samples
#         """
#         idx = list(range(len(keep_idx)))
#         if stride and stride > 1:
#             idx = idx[::stride]

#         if min_dt is not None:
#             ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
#             ts_vals = []
#             for k in keep_idx:
#                 t = ai_recs[k].get(ts_key)
#                 if hasattr(t, "timestamp"):
#                     t = t.timestamp()
#                 ts_vals.append(np.nan if t is None else float(t))
#             kept_rel, last_t = [], None
#             for j in idx:
#                 tj = ts_vals[j]
#                 if np.isnan(tj) or last_t is None or (tj - last_t) >= float(min_dt):
#                     kept_rel.append(j); last_t = tj
#             idx = kept_rel

#         if min_dist is not None and len(idx) > 1:
#             kept_rel = [idx[0]]
#             last_p = pos[idx[0]]
#             for j in idx[1:]:
#                 if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
#                     kept_rel.append(j); last_p = pos[j]
#             idx = kept_rel

#         return idx

#     def build_bins_from_pooled(d_all, nb):
#         d_all = np.asarray(d_all, float)
#         d_all = d_all[np.isfinite(d_all)]
#         if d_all.size < 5:
#             return None, None
#         lo, hi = np.percentile(d_all, [1, 99])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = np.nanmin(d_all), np.nanmax(d_all)
#         edges = np.linspace(lo, hi, nb + 1)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#         return edges, centers

#     def per_trial_binned_means(d, a, edges):
#         d = np.asarray(d, float); a = np.asarray(a, float)
#         good = np.isfinite(d) & np.isfinite(a)
#         if good.sum() == 0:
#             return np.full(len(edges) - 1, np.nan)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d >= edges[i]) & (d < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a[m])
#         return out

#     def aggregate_across_trials(per_trial_mat):
#         per_trial_mat = np.asarray(per_trial_mat, float)
#         mean = np.nanmean(per_trial_mat, axis=0)
#         std = np.nanstd(per_trial_mat, axis=0, ddof=1)
#         n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
#         sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
#         lo = mean - 1.96 * sem
#         hi = mean + 1.96 * sem
#         return mean, lo, hi, n

#     def window_mean(t, a, w):
#         """Mean α in time window w = (t_min, t_max) on a single trial (times in ms)."""
#         t = np.asarray(t, float); a = np.asarray(a, float)
#         m = (t >= w[0]) & (t <= w[1]) & np.isfinite(t) & np.isfinite(a)
#         return np.nan if m.sum() == 0 else float(np.nanmean(a[m]))

#     def window_min(t, a, w):
#         """Min α in time window w = (t_min, t_max) on a single trial (times in ms)."""
#         t = np.asarray(t, float); a = np.asarray(a, float)
#         m = (t >= w[0]) & (t <= w[1]) & np.isfinite(t) & np.isfinite(a)
#         return np.nan if m.sum() == 0 else float(np.nanmin(a[m]))

#     # ---- windows in ms ----
#     pre_window    = (-600.0, -200.0)   # descriptive pre
#     ref_window    = (-400.0,    0.0)   # reference around the event
#     early_window  = (0.0,      400.0)
#     late_window   = (1600.0,  2400.0)
#     dip_window    = (0.0,      800.0)  # where we search for the dip

#     # pseudo-event regions (no real event), in ms relative to real event
#     pre_pseudo_range  = (-1200.0, -400.0)
#     post_pseudo_range = (1000.0, 2500.0)
#     n_pseudo_per_trial = 5

#     # ---------------- determine which event to align to ----------------
#     effective_task = (task if task is not None else experiment) or ""
#     eff_lower = effective_task.lower()

#     if "respawn" in eff_lower or "jump" in eff_lower:
#         event_attr = "targetJumpTime"
#         event_name = "target respawn"
#     elif "appearing obstacle" in eff_lower or "obstacle" in eff_lower:
#         event_attr = "obstacleAppearedTime"
#         event_name = "obstacle appearance"
#     else:
#         event_attr = "obstacleAppearedTime"
#         event_name = "event"
#         print(f"[WARN] Unknown task '{effective_task}', defaulting to obstacleAppearedTime")

#     # ---------------- collect data ----------------
#     if file_path is None:
#         data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
#         pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#     else:
#         pkl_files = [file_path]

#     all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, \
#         pkl_files, ai_trials = load_files(experiment, monkey, base_dir=base_dir)

#     trials_t = []
#     trials_alpha = []
#     pooled_t, pooled_a = [], []
#     r_trials_time = []

#     pre_vals, early_vals, late_vals = [], [], []
#     dip_depths = []
#     recovery_times = []         # per trial, ms
#     delta_pseudo_dip = []       # per trial, mean pseudo dip across pseudo centers

#     tmin, tmax = float(t_window[0]), float(t_window[1])

#     for session in all_trials:
#         for trial in session:

#             # answer filter
#             if only_answer is not None:
#                 ans = trial.answer
#                 if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
#                     continue

#             ai_recs = trial.aiVelocities
#             if not ai_recs:
#                 continue

#             # --- event time (obstacle or respawn) ---
#             t_ev = getattr(trial, event_attr, None) \
#                    if not isinstance(trial, dict) else trial.get(event_attr, None)
#             if t_ev is None:
#                 continue
#             t_ev = float(t_ev)  # assumed ms, same units as avatarTrajectory['time']

#             # skip if aiVelocity is zero throughout
#             start_idx = next((i for i, rec in enumerate(ai_recs)
#                               if np.linalg.norm(rec['Output']) > 0), None)
#             if start_idx is None:
#                 continue
#             ai_recs = ai_recs[start_idx:]

#             # positions from avatarTrajectory
#             xs, zs, keep_idx = [], [], []
#             for i, rec in enumerate(ai_recs):
#                 p = xz_from(trial.avatarTrajectory, i)
#                 if p is None or np.isnan(p[0]) or np.isnan(p[1]):
#                     continue
#                 xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
#             if len(keep_idx) < 3:
#                 continue
#             pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

#             # entropy (full ai_recs, then restrict to keep_idx)
#             ent_raw = []
#             for rec in ai_recs:
#                 e = rec.get("EntropyLb")
#                 if isinstance(e, (list, tuple, np.ndarray)):
#                     e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
#                 ent_raw.append(np.nan if e is None else float(e))
#             ent_raw = np.asarray(ent_raw, float)
#             ent_kept = ent_raw[keep_idx]

#             # thinning indices
#             idx_rel = thin_indices(pos_full, keep_idx, ai_recs,
#                                    stride=stride, min_dt=min_dt,
#                                    min_dist=min_dist, use_output_ts=True)
#             if len(idx_rel) < 3:
#                 continue

#             pos = pos_full[idx_rel]
#             ent_kept = ent_kept[idx_rel]

#             # time vector aligned with pos / ent_kept
#             traj_t_all = None
#             if isinstance(trial.avatarTrajectory, dict):
#                 traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []),
#                                         dtype=float)  # expected ms
#             if traj_t_all is None or traj_t_all.size == 0:
#                 continue
#             try:
#                 traj_t_kept = traj_t_all[keep_idx][idx_rel]
#             except Exception:
#                 continue

#             # Alpha
#             alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

#             # relative time to event (ms)
#             t_rel = traj_t_kept - t_ev

#             # mask window
#             mask = (t_rel >= tmin) & (t_rel <= tmax) & \
#                    np.isfinite(alpha_vals) & np.isfinite(t_rel)
#             if mask.sum() < 3:
#                 continue

#             t_rel_trial = t_rel[mask]
#             alpha_trial = alpha_vals[mask]

#             trials_t.append(t_rel_trial)
#             trials_alpha.append(alpha_trial)
#             pooled_t.append(t_rel_trial)
#             pooled_a.append(alpha_trial)

#             # per-trial correlation with time (descriptive)
#             r_trials_time.append(
#                 np.corrcoef(t_rel_trial, alpha_trial)[0, 1]
#                 if t_rel_trial.size >= 3 else np.nan
#             )

#             # ---- pre / early / late means (pre uses [-600,-200]) ----
#             pre  = window_mean(t_rel_trial, alpha_trial, pre_window)
#             early = window_mean(t_rel_trial, alpha_trial, early_window)
#             late  = window_mean(t_rel_trial, alpha_trial, late_window)

#             pre_vals.append(pre)
#             early_vals.append(early)
#             late_vals.append(late)

#             # ---- dip depth and recovery time (relative to event reference) ----
#             dip_depth = np.nan
#             t_rec = np.nan

#             # reference = mean α just before event
#             ref = window_mean(t_rel_trial, alpha_trial, ref_window)

#             if np.isfinite(ref):
#                 # 1) find the dip minimum in the dip window
#                 m_dip = (t_rel_trial >= dip_window[0]) & (t_rel_trial <= dip_window[1]) & np.isfinite(alpha_trial)
#                 if np.any(m_dip):
#                     a_dip = alpha_trial[m_dip]
#                     t_dip = t_rel_trial[m_dip]

#                     try:
#                         idx_min = int(np.nanargmin(a_dip))
#                         dip_min = float(a_dip[idx_min])
#                         t_dip_min = float(t_dip[idx_min])

#                         # dip depth = minimum after event minus reference at event
#                         dip_depth = dip_min - ref

#                         # 2) recovery: first time AFTER the dip minimum where α ≥ reference
#                         m_rec = (t_rel_trial >= t_dip_min) & np.isfinite(alpha_trial)
#                         if np.any(m_rec):
#                             t_after = t_rel_trial[m_rec]
#                             a_after = alpha_trial[m_rec]
#                             idx_rec = np.where(a_after >= ref)[0]
#                             if idx_rec.size > 0:
#                                 t_rec = float(t_after[idx_rec[0]])  # ms from event
#                     except ValueError:
#                         pass

#             dip_depths.append(dip_depth)
#             recovery_times.append(t_rec)

#             # ---- pseudo-event control: multiple pseudo centers per trial ----
#             pseudo_deltas_this = []

#             for _ in range(n_pseudo_per_trial):
#                 # choose pre- or post-event pseudo region
#                 if np.random.rand() < 0.5:
#                     r_lo, r_hi = pre_pseudo_range
#                 else:
#                     r_lo, r_hi = post_pseudo_range

#                 pseudo_t0 = np.random.uniform(r_lo, r_hi)
#                 t_centered = t_rel_trial - pseudo_t0

#                 ref_p = window_mean(t_centered, alpha_trial, ref_window)
#                 dip_min_p = window_min(t_centered, alpha_trial, dip_window)

#                 if np.isfinite(ref_p) and np.isfinite(dip_min_p):
#                     pseudo_deltas_this.append(dip_min_p - ref_p)

#             if len(pseudo_deltas_this) > 0:
#                 delta_pseudo_dip.append(float(np.nanmean(pseudo_deltas_this)))
#             else:
#                 delta_pseudo_dip.append(np.nan)

#     result = {}

#     if not pooled_t:
#         print("[WARN] No valid samples around event.")
#         return result

#     pooled_t = np.concatenate(pooled_t)
#     pooled_a = np.concatenate(pooled_a)

#     # convert per-trial summaries to arrays
#     pre_vals   = np.asarray(pre_vals, float)
#     early_vals = np.asarray(early_vals, float)
#     late_vals  = np.asarray(late_vals, float)
#     dip_depths = np.asarray(dip_depths, float)
#     recovery_times = np.asarray(recovery_times, float)        # ms
#     delta_pseudo_dip = np.asarray(delta_pseudo_dip, float)

#     # median recovery time (ms) for shading
#     if np.isfinite(recovery_times).sum() >= 3:
#         recovery_median = float(np.nanmedian(recovery_times))
#     else:
#         recovery_median = None

#     edges_t, centers_t = build_bins_from_pooled(pooled_t, nbins)
#     if edges_t is None:
#         print("[WARN] Not enough data to build time bins.")
#         return result

#     per_trial_mat_t = [
#         per_trial_binned_means(t, a, edges_t)
#         for t, a in zip(trials_t, trials_alpha)
#     ]
#     per_trial_mat_t = np.asarray(per_trial_mat_t, float)

#     mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)

#     # ---------------- plot ----------------
#     fig, ax = plt.subplots(figsize=(6.0, 4.0))

#     x_centers = centers_t  # ms

#     # grey band: 0 -> median recovery-to-reference time (visual only)
#     if recovery_median is not None and np.isfinite(recovery_median) and recovery_median > 0:
#         ax.axvspan(0.0, recovery_median,
#                    color='0.95', alpha=0.6, zorder=0)  # behind everything

#     if show_per_trial:
#         for row in per_trial_mat_t:
#             ax.plot(x_centers, row, lw=0.6, alpha=0.15)

#     ax.fill_between(x_centers, lo_t, hi_t, alpha=0.25,
#                     label="Across-trials 95% CI")
#     ax.plot(x_centers, mean_t, lw=2.0, label="Across-trials mean")

#     # vertical line at event (t=0)
#     ax.axvline(0.0, color='tab:purple', ls='-', lw=2,
#                label=f"{event_name} (t=0)")

#     # correlation text
#     r_pool, p_pool = safe_pearsonr(pooled_t, pooled_a)
#     med_r = np.nanmedian(
#         [r for r in np.asarray(r_trials_time, float) if np.isfinite(r)]
#     ) if len(r_trials_time) else np.nan

#     def fmt_p(p):
#         if not np.isfinite(p):
#             return "p=NA"
#         return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

#     ax.text(
#         0.02, 0.97,
#         f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
#         transform=ax.transAxes, ha='left', va='top', fontsize=9
#     )

#     ax.set_xlabel(f"Time from {event_name} (ms)")
#     ax.set_ylabel("Posterior Confidence Index")
#     ax.set_title(f"{monkey} – {experiment}\nα vs time from {event_name}")
#     ax.set_xlim(tmin, tmax)
#     if fixed_ylim is not None:
#         ax.set_ylim(*fixed_ylim)
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='lower right', frameon=True, fontsize=8)
#     plt.tight_layout()

#     try:
#         save_plot(fig, f"alpha_vs_time_FROM_EVENT_{monkey}_{experiment}",
#                   subfolder="Entropy")
#     except Exception:
#         pass
#     plt.show(); plt.close(fig)

#     # ---------------- summaries for stats / bar plots ----------------
#     delta_early = early_vals - pre_vals
#     delta_late  = late_vals  - pre_vals

#     def wilcoxon_or_nan(x):
#         x = np.asarray(x, float)
#         x = x[np.isfinite(x)]
#         if x.size < 3:
#             return np.nan
#         return float(wilcoxon(x, alternative='two-sided').pvalue)

#     p_delta_early = wilcoxon_or_nan(delta_early)
#     p_delta_late  = wilcoxon_or_nan(delta_late)
#     p_delta_pseudo = wilcoxon_or_nan(delta_pseudo_dip)
#     p_dip_depth    = wilcoxon_or_nan(dip_depths)

#     # NEW: paired real–pseudo dip difference
#     real_minus_pseudo = dip_depths - delta_pseudo_dip
#     p_real_minus_pseudo = wilcoxon_or_nan(real_minus_pseudo)

#     result['time'] = {
#         'centers_s': centers_t,       # ms, kept name for backwards compatibility
#         'mean': mean_t,
#         'lo': lo_t,
#         'hi': hi_t,
#         'n_trials_per_bin': n_t,
#         'r_pooled': r_pool,
#         'r_trials': r_trials_time,
#         't_window': t_window,
#     }

#     result['event_windows'] = {
#         'pre_mean':   float(np.nanmean(pre_vals)),
#         'early_mean': float(np.nanmean(early_vals)),
#         'late_mean':  float(np.nanmean(late_vals)),
#         'delta_early_mean': float(np.nanmean(delta_early)),
#         'delta_early_median': float(np.nanmedian(delta_early)),
#         'delta_early_p': p_delta_early,
#         'delta_late_mean': float(np.nanmean(delta_late)),
#         'delta_late_median': float(np.nanmedian(delta_late)),
#         'delta_late_p': p_delta_late,
#         'n_trials': int(np.sum(np.isfinite(pre_vals) & np.isfinite(early_vals))),
#     }

#     result['dip_recovery'] = {
#         'dip_depth_mean': float(np.nanmean(dip_depths)),
#         'dip_depth_median': float(np.nanmedian(dip_depths)),
#         'dip_depth_p': p_dip_depth,
#         # still labelled *_s for compatibility, but values are in ms
#         'recovery_time_median_s': float(np.nanmedian(recovery_times)),
#         'recovery_time_mean_s': float(np.nanmean(recovery_times)),
#         'recovery_time_values_s': recovery_times,
#     }

#     result['pseudo_control'] = {
#         'delta_pseudo_mean': float(np.nanmean(delta_pseudo_dip)),
#         'delta_pseudo_median': float(np.nanmedian(delta_pseudo_dip)),
#         'delta_pseudo_p': p_delta_pseudo,
#     }

#     result['real_vs_pseudo'] = {
#         'diff_mean': float(np.nanmean(real_minus_pseudo)),
#         'diff_median': float(np.nanmedian(real_minus_pseudo)),
#         'diff_p': p_real_minus_pseudo,
#     }

#     return result
# def plot_alpha_vs_time_from_obstacle_appearance(
#     monkey,
#     experiment,
#     base_dir,
#     file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
#     stride=4,                # thinning: keep every k-th sample
#     min_dt=None,             # thinning: keep if >= this seconds since last kept
#     min_dist=None,           # thinning: keep if moved >= this distance since last kept
#     t_window=(-2000.0, 3000.0),  # time window [ms] relative to event
#     nbins=24,                # common bin count across trials
#     min_alpha=0.2,           # floor after per-trial robust normalization
#     only_answer=None,        # 1 = only correct, 0 = only incorrect, None = all
#     task=None,
#     show_per_trial=False,    # draw faint per-trial lines
#     min_align_frac=None,     # optional: min fraction of aligned velocity (Respawn only)
#     max_align_frac=None      # optional: max fraction of aligned velocity (Respawn only)
# ):
#     """
#     α (= 1 − normalized EntropyLb) vs time from event (obstacle appearance or target jump).

#     For each trial:
#       - build α(t) from EntropyLb with robust per-trial normalization
#       - align time to event (obstacleAppearedTime or targetJumpTime)
#       - compute:
#           * event-locked α(t) (mean ± 95% CI)
#           * dip depth relative to pre-event α (−400..0 ms)
#           * recovery time to pre-event α, searching from dip to 3 s
#           * pseudo-event dips (random times before/after) as control
#           * OPTIONAL: alignment fraction p_align (Respawn) between velocity and new target
#             in 0..1000 ms, with optional filtering by [min_align_frac, max_align_frac].
#     """

#     import os, glob
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import pearsonr, wilcoxon

#     # ---------------- helpers ----------------
#     def safe_pearsonr(x, y):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         m = np.isfinite(x) & np.isfinite(y)
#         if m.sum() < 3:
#             return np.nan, np.nan
#         try:
#             r, p = pearsonr(x[m], y[m])
#         except Exception:
#             r, p = np.nan, np.nan
#         return r, p

#     def xz_from(pos, i=None):
#         if pos is None:
#             return None
#         if isinstance(pos, dict):
#             if i is not None:
#                 return (pos["x"][i], pos["z"][i])
#             # center (for target)
#             return (float(pos["x"][0]), float(pos["z"][0]))
#         try:
#             return (float(pos[0]), float(pos[2]))
#         except Exception:
#             return None

#     def thin_indices(pos, keep_idx, ai_recs,
#                      stride=None, min_dt=None, min_dist=None,
#                      use_output_ts=True):
#         idx = list(range(len(keep_idx)))
#         if stride and stride > 1:
#             idx = idx[::stride]

#         if min_dt is not None:
#             ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
#             ts_vals = []
#             for k in keep_idx:
#                 t = ai_recs[k].get(ts_key)
#                 if hasattr(t, "timestamp"):
#                     t = t.timestamp()
#                 ts_vals.append(np.nan if t is None else float(t))
#             kept_rel, last_t = [], None
#             for j in idx:
#                 tj = ts_vals[j]
#                 if np.isnan(tj) or last_t is None or (tj - last_t) >= float(min_dt):
#                     kept_rel.append(j); last_t = tj
#             idx = kept_rel

#         if min_dist is not None and len(idx) > 1:
#             kept_rel = [idx[0]]
#             last_p = pos[idx[0]]
#             for j in idx[1:]:
#                 if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
#                     kept_rel.append(j); last_p = pos[j]
#             idx = kept_rel
#         return idx

#     def build_bins_from_pooled(d_all, nb):
#         d_all = np.asarray(d_all, float)
#         d_all = d_all[np.isfinite(d_all)]
#         if d_all.size < 5:
#             return None, None
#         lo, hi = np.percentile(d_all, [1, 99])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = np.nanmin(d_all), np.nanmax(d_all)
#         edges = np.linspace(lo, hi, nb + 1)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#         return edges, centers

#     def per_trial_binned_means(d, a, edges):
#         d = np.asarray(d, float); a = np.asarray(a, float)
#         good = np.isfinite(d) & np.isfinite(a)
#         if good.sum() == 0:
#             return np.full(len(edges) - 1, np.nan)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d >= edges[i]) & (d < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a[m])
#         return out

#     def aggregate_across_trials(per_trial_mat):
#         per_trial_mat = np.asarray(per_trial_mat, float)
#         mean = np.nanmean(per_trial_mat, axis=0)
#         std = np.nanstd(per_trial_mat, axis=0, ddof=1)
#         n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
#         sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
#         lo = mean - 1.96 * sem
#         hi = mean + 1.96 * sem
#         return mean, lo, hi, n

#     def pooled_binned_means(d_all, a_all, edges):
#         d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
#         good = np.isfinite(d_all) & np.isfinite(a_all)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a[m])
#         return out

#     def wilcoxon_or_nan(x):
#         x = np.asarray(x, float)
#         x = x[np.isfinite(x)]
#         if x.size < 3:
#             return np.nan
#         return float(wilcoxon(x, alternative='two-sided').pvalue)

#     def target_center_from(trial):
#         tgt = getattr(trial, "targetPosition", None) if not isinstance(trial, dict) else trial.get("targetPosition", None)
#         if tgt is None:
#             return None
#         if isinstance(tgt, dict):
#             xs = tgt.get("x", None)
#             zs = tgt.get("z", None)
#             try:
#                 tx = float(xs[0] if hasattr(xs, "__len__") else xs)
#                 tz = float(zs[0] if hasattr(zs, "__len__") else zs)
#                 return np.array([tx, tz], float)
#             except Exception:
#                 return None
#         arr = np.asarray(tgt, float).ravel()
#         if arr.size >= 3:
#             return arr[[0, 2]]
#         if arr.size >= 2:
#             return arr[:2]
#         return None

#     def get_velocity_xz(rec):
#         # Try to get "pure" model/BCI velocity if present, else fall back to Output
#         v = rec.get("ModelVelocity", None)
#         if v is None:
#             v = rec.get("Output", None)
#         if v is None:
#             v = rec.get("Velocity", None)
#         if v is None:
#             return None
#         v = np.asarray(v, float).ravel()
#         if v.size >= 3:
#             return v[[0, 2]]
#         if v.size >= 2:
#             return v[:2]
#         return None

#     # ---- window definitions (all in ms, because t_rel is in ms) ----
#     ref_window = (-400.0, 0.0)          # pre-event baseline for dip
#     dip_window = (0.0, 800.0)           # where we search for the minimum after event
#     recovery_search_max = 3000.0        # upper bound for recovery search
#     recovery_band = (400.0, 1200.0)     # visual band for “typical” recovery (for the plot)

#     # Velocity alignment window (Respawn): 0–1000 ms after target jump
#     align_window = (0.0, 2000.0)
#     align_cos_deg = 20.0
#     align_cos_thresh = np.cos(np.deg2rad(align_cos_deg))

#     # Pseudo-event ranges (for controls)
#     pseudo_pre_range = (-1200.0, -600.0)   # random centers before event
#     pseudo_post_range = (1000.0, 3000.0)   # random centers after typical recovery
#     N_PSEUDO_PRE = 3
#     N_PSEUDO_POST = 3

#     # ---------------- determine event type ----------------
#     effective_task = task if task is not None else experiment
#     if "Respawn" in effective_task:
#         event_attr = "targetJumpTime"
#     else:
#         event_attr = "obstacleAppearedTime"

#     # ---------------- load trials ----------------
#     if file_path is None:
#         data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
#         pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#     else:
#         pkl_files = [file_path]

#     all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, \
#         pkl_files, ai_trials = load_files(experiment, monkey, base_dir=base_dir)

#     trials_t = []
#     trials_alpha = []
#     pooled_t, pooled_a = [], []
#     r_trials_time = []

#     dip_depths = []
#     recovery_times = []
#     pseudo_means = []
#     align_fracs = []

#     tmin, tmax = float(t_window[0]), float(t_window[1])

#     # ---------------- per-trial loop ----------------
#     for session in all_trials:
#         for trial in session:
#             # filter by behavioral correctness if requested
#             if only_answer is not None:
#                 ans = getattr(trial, "answer", None)
#                 if ans is None:
#                     continue
#                 if only_answer == 1 and ans != 1:
#                     continue
#                 if only_answer == 0 and ans == 1:
#                     continue

#             ai_recs = trial.aiVelocities
#             if not ai_recs:
#                 continue

#             # choose event time
#             t_event = getattr(trial, event_attr, None) \
#                 if not isinstance(trial, dict) else trial.get(event_attr, None)
#             if t_event is None:
#                 continue
#             t_event = float(t_event)

#             # Skip if velocity is zero everywhere (no movement)
#             start_idx = next(
#                 (i for i, rec in enumerate(ai_recs)
#                  if np.linalg.norm(rec['Output']) > 0),
#                 None
#             )
#             if start_idx is None:
#                 continue
#             ai_recs = ai_recs[start_idx:]

#             # positions from avatarTrajectory
#             xs, zs, keep_idx = [], [], []
#             for i, rec in enumerate(ai_recs):
#                 p = xz_from(trial.avatarTrajectory, i)
#                 if p is None or np.isnan(p[0]) or np.isnan(p[1]):
#                     continue
#                 xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
#             if len(keep_idx) < 3:
#                 continue
#             pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

#             # entropy (full ai_recs, then restrict to keep_idx)
#             ent_raw = []
#             for rec in ai_recs:
#                 e = rec.get("EntropyLb")
#                 if isinstance(e, (list, tuple, np.ndarray)):
#                     e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
#                 ent_raw.append(np.nan if e is None else float(e))
#             ent_raw = np.asarray(ent_raw, float)
#             ent_kept = ent_raw[keep_idx]

#             # thinning
#             idx_rel = thin_indices(pos_full, keep_idx, ai_recs,
#                                    stride=stride, min_dt=min_dt,
#                                    min_dist=min_dist, use_output_ts=True)
#             if len(idx_rel) < 3:
#                 continue

#             pos = pos_full[idx_rel]
#             ent_kept = ent_kept[idx_rel]

#             # time vector aligned with pos / ent_kept
#             traj_t_all = None
#             if isinstance(trial.avatarTrajectory, dict):
#                 traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []),
#                                         dtype=float)
#             if traj_t_all is None or traj_t_all.size == 0:
#                 continue
#             try:
#                 traj_t_kept = traj_t_all[keep_idx][idx_rel]
#             except Exception:
#                 continue

#             # Alpha
#             alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

#             # relative time to event
#             t_rel = traj_t_kept - t_event

#             # mask within global window
#             mask = (t_rel >= tmin) & (t_rel <= tmax) \
#                    & np.isfinite(alpha_vals) & np.isfinite(t_rel)
#             if mask.sum() < 3:
#                 continue

#             t_rel_trial = t_rel[mask]
#             alpha_trial = alpha_vals[mask]
#             pos_trial = pos[mask]

#             # mapping back to ai_recs indices (for velocities)
#             keep_idx_arr = np.asarray(keep_idx)[idx_rel]
#             idx_ai_masked = keep_idx_arr[mask]

#             # ---- compute alignment fraction (Respawn only) ----
#             p_align = np.nan
#             if event_attr == "targetJumpTime":
#                 # tgt_center = target_center_from(trial)
#                 tgt_center = trial.targetJumpPosition
#                 if tgt_center is not None:
#                     mask_align = (t_rel_trial >= align_window[0]) & (t_rel_trial <= align_window[1])
#                     if np.any(mask_align):
#                         pos_align = pos_trial[mask_align]
#                         idx_ai_align = idx_ai_masked[mask_align]
#                         vel_list = []
#                         for j in idx_ai_align:
#                             if j < 0 or j >= len(ai_recs):
#                                 vel_list.append([np.nan, np.nan])
#                                 continue
#                             v = get_velocity_xz(ai_recs[j])
#                             if v is None or np.any(~np.isfinite(v)):
#                                 vel_list.append([np.nan, np.nan])
#                             else:
#                                 vel_list.append(v)
#                         vel_arr = np.asarray(vel_list, float)
#                         d_vec = tgt_center[None, :] - pos_align
#                         v_norm = np.linalg.norm(vel_arr, axis=1)
#                         d_norm = np.linalg.norm(d_vec, axis=1)
#                         valid = (v_norm > 1e-6) & (d_norm > 1e-6) \
#                                 & np.all(np.isfinite(vel_arr), axis=1) \
#                                 & np.all(np.isfinite(d_vec), axis=1)
#                         if np.any(valid):
#                             cosang = np.sum(vel_arr[valid] * d_vec[valid], axis=1) / (v_norm[valid] * d_norm[valid])
#                             cosang = np.clip(cosang, -1.0, 1.0)
#                             p_align = float(np.mean(cosang >= align_cos_thresh))

#             align_fracs.append(p_align)

#             # optional filter on alignment fraction (Respawn only)
#             if (min_align_frac is not None or max_align_frac is not None) and event_attr == "targetJumpTime":
#                 if not np.isfinite(p_align):
#                     continue
#                 if (min_align_frac is not None and p_align < min_align_frac) or \
#                    (max_align_frac is not None and p_align > max_align_frac):
#                     # reject this trial based on alignment
#                     continue

#             # store time courses
#             trials_t.append(t_rel_trial)
#             trials_alpha.append(alpha_trial)
#             pooled_t.append(t_rel_trial)
#             pooled_a.append(alpha_trial)

#             r_trials_time.append(
#                 np.corrcoef(t_rel_trial, alpha_trial)[0, 1]
#                 if t_rel_trial.size >= 3 else np.nan
#             )

#             # ---- dip depth & recovery for this trial ----
#             t_rel_trial = np.asarray(t_rel_trial, float)
#             alpha_trial = np.asarray(alpha_trial, float)

#             # pre-event baseline
#             m_ref = (t_rel_trial >= ref_window[0]) & (t_rel_trial <= ref_window[1])
#             pre_val = np.nan
#             if np.any(m_ref):
#                 pre_val = float(np.nanmean(alpha_trial[m_ref]))

#             # dip region
#             dip_val = np.nan
#             dip_time = np.nan
#             m_dip = (t_rel_trial >= dip_window[0]) & (t_rel_trial <= dip_window[1])
#             if np.any(m_dip):
#                 a_seg = alpha_trial[m_dip]
#                 t_seg = t_rel_trial[m_dip]
#                 try:
#                     idx_min = int(np.nanargmin(a_seg))
#                     dip_val = float(a_seg[idx_min])
#                     dip_time = float(t_seg[idx_min])
#                 except (ValueError, TypeError):
#                     dip_val = np.nan
#                     dip_time = np.nan

#             if np.isfinite(pre_val) and np.isfinite(dip_val):
#                 dip_depth = dip_val - pre_val
#             else:
#                 dip_depth = np.nan

#             # recovery time: first time after dip where α >= pre_val
#             recovery_time = np.nan
#             if np.isfinite(pre_val) and np.isfinite(dip_time):
#                 m_post = (t_rel_trial > dip_time) & (t_rel_trial <= recovery_search_max)
#                 if np.any(m_post):
#                     t_post = t_rel_trial[m_post]
#                     a_post = alpha_trial[m_post]
#                     idx_recover = np.where(a_post >= pre_val)[0]
#                     if idx_recover.size > 0:
#                         recovery_time = float(t_post[idx_recover[0]])

#             dip_depths.append(dip_depth)
#             recovery_times.append(recovery_time)

#             # ---- pseudo-event dips (control) ----
#             pseudo_dips_this_trial = []

#             def compute_pseudo_dip(center_ms):
#                 t_centered = t_rel_trial - float(center_ms)
#                 m_ref_p = (t_centered >= ref_window[0]) & (t_centered <= ref_window[1])
#                 if not np.any(m_ref_p):
#                     return np.nan
#                 pre_p = float(np.nanmean(alpha_trial[m_ref_p]))
#                 m_dip_p = (t_centered >= dip_window[0]) & (t_centered <= dip_window[1])
#                 if not np.any(m_dip_p):
#                     return np.nan
#                 a_seg_p = alpha_trial[m_dip_p]
#                 try:
#                     dip_p = float(np.nanmin(a_seg_p))
#                 except (ValueError, TypeError):
#                     return np.nan
#                 return dip_p - pre_p

#             # pseudo events before
#             for _ in range(N_PSEUDO_PRE):
#                 center = np.random.uniform(pseudo_pre_range[0], pseudo_pre_range[1])
#                 pseudo_d = compute_pseudo_dip(center)
#                 if np.isfinite(pseudo_d):
#                     pseudo_dips_this_trial.append(pseudo_d)

#             # pseudo events after
#             for _ in range(N_PSEUDO_POST):
#                 center = np.random.uniform(pseudo_post_range[0], pseudo_post_range[1])
#                 pseudo_d = compute_pseudo_dip(center)
#                 if np.isfinite(pseudo_d):
#                     pseudo_dips_this_trial.append(pseudo_d)

#             if pseudo_dips_this_trial:
#                 pseudo_means.append(float(np.nanmean(pseudo_dips_this_trial)))
#             else:
#                 pseudo_means.append(np.nan)

#     # ---------------- aggregate across trials ----------------
#     result = {}

#     if not pooled_t:
#         print("[WARN] No valid samples around event.")
#         return result

#     pooled_t = np.concatenate(pooled_t)
#     pooled_a = np.concatenate(pooled_a)

#     edges_t, centers_t = build_bins_from_pooled(pooled_t, nbins)
#     if edges_t is None:
#         print("[WARN] Not enough data to build time bins.")
#         return result

#     per_trial_mat_t = [
#         per_trial_binned_means(t, a, edges_t)
#         for t, a in zip(trials_t, trials_alpha)
#     ]
#     per_trial_mat_t = np.asarray(per_trial_mat_t, float)

#     mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)

#     # ---------------- plot ----------------
#     fig, ax = plt.subplots(figsize=(8.8, 5.0))

#     centers_ms = centers_t  # these are already in ms if t_rel was ms

#     if show_per_trial:
#         for row in per_trial_mat_t:
#             ax.plot(centers_ms, row, lw=0.8, alpha=0.2)

#     # gray band for “recovery zone”
#     ax.axvspan(recovery_band[0], recovery_band[1],
#                color='0.9', alpha=0.4, label="Recovery window")

#     ax.fill_between(centers_ms, lo_t, hi_t, alpha=0.25,
#                     label="Across-trials 95% CI")
#     ax.plot(centers_ms, mean_t, lw=2.5, label="Across-trials mean")

#     # vertical line at event (t=0)
#     ax.axvline(0.0, color='tab:purple', ls='-', lw=2,
#                label="Event (t=0)")

#     r_pool, p_pool = safe_pearsonr(pooled_t, pooled_a)
#     med_r = np.nanmedian(
#         [r for r in np.asarray(r_trials_time, float) if np.isfinite(r)]
#     ) if len(r_trials_time) else np.nan

#     def fmt_p(p):
#         if not np.isfinite(p):
#             return "p=NA"
#         return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

#     ax.text(
#         0.02, 0.95,
#         f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
#         transform=ax.transAxes, ha='left', va='top', fontsize=10
#     )

#     ax.set_xlabel("Time from event (ms)")
#     ax.set_ylabel("Posterior Confidence Index α")
#     ax.set_title(f"{monkey} – {experiment}\nα vs time from event")
#     ax.grid(True, alpha=0.3)
#     ax.legend(frameon=True)
#     plt.tight_layout()

#     try:
#         save_plot(fig,
#                   f"alpha_vs_time_FROM_EVENT_{monkey}_{experiment}",
#                   subfolder="Entropy")
#     except Exception:
#         pass
#     plt.show()
#     plt.close(fig)

#     # ---------------- stats summaries ----------------
#     dip_depths = np.asarray(dip_depths, float)
#     recovery_times = np.asarray(recovery_times, float)
#     pseudo_means = np.asarray(pseudo_means, float)

#     p_dip = wilcoxon_or_nan(dip_depths)
#     p_pseudo = wilcoxon_or_nan(pseudo_means)

#     m_both = np.isfinite(dip_depths) & np.isfinite(pseudo_means)
#     diffs = dip_depths[m_both] - pseudo_means[m_both]
#     p_diff = wilcoxon_or_nan(diffs)

#     result['time'] = {
#         'centers_s': centers_ms,   # kept name for backward compatibility (actually ms)
#         'mean': mean_t,
#         'lo': lo_t,
#         'hi': hi_t,
#         'n_trials_per_bin': n_t,
#         'r_pooled': r_pool,
#         'r_trials': r_trials_time,
#         't_window': t_window,
#     }

#     result['dip_recovery'] = {
#         'dip_depth_mean': float(np.nanmean(dip_depths)),
#         'dip_depth_median': float(np.nanmedian(dip_depths)),
#         'dip_depth_p': p_dip,
#         'recovery_time_median_s': float(np.nanmedian(recovery_times)),
#         'recovery_time_mean_s': float(np.nanmean(recovery_times)),
#         'recovery_time_values_s': recovery_times,
#     }

#     result['pseudo_control'] = {
#         'delta_pseudo_mean': float(np.nanmean(pseudo_means)),
#         'delta_pseudo_median': float(np.nanmedian(pseudo_means)),
#         'delta_pseudo_p': p_pseudo,
#     }

#     result['real_vs_pseudo'] = {
#         'diff_mean': float(np.nanmean(diffs)) if diffs.size > 0 else np.nan,
#         'diff_median': float(np.nanmedian(diffs)) if diffs.size > 0 else np.nan,
#         'diff_p': p_diff,
#     }

#     result['alignment'] = {
#         'align_fracs': np.asarray(align_fracs, float),  # one value per *considered* trial
#         'align_window_ms': align_window,
#         'align_cos_threshold': align_cos_thresh,
#         'min_align_frac': min_align_frac,
#         'max_align_frac': max_align_frac,
#     }

#     return result
# def plot_alpha_vs_time_from_obstacle_appearance(
#     monkey, experiment, base_dir,
#     file_path=None,
#     stride=4,
#     min_dt=None,
#     min_dist=None,
#     t_window=(-2000.0, 3000.0),
#     nbins=24,
#     min_alpha=0.2,
#     only_answer=None,
#     task=None,
#     show_per_trial=False,
#     min_align_frac=None,   # NEW: filter by intention (true goal fraction)
#     max_align_frac=None,   # NEW: optional upper bound
# ):
#     """
#     α (= 1 − normalized EntropyLb) vs time from event (obstacle appearance or target jump).

#     Event = obstacleAppearedTime for Appearing Obstacle tasks,
#             targetJumpTime for Respawn tasks.

#     Uses modelVelocity to compute, in a post-event window, the fraction of frames
#     where the decoded intention points to the TRUE target, with the same 60% rule
#     as classify_no_ai_failure_reason:

#       choice_label:
#         'correct_choice' if best_goal == true_goal and best_frac ≥ 0.6
#         'wrong_choice'   if best_goal != true_goal and best_frac ≥ 0.6
#         'ambiguous_choice' otherwise

#     Returned result dict still has:
#       - 'time'
#       - 'event_windows'
#       - 'dip_recovery'
#       - 'pseudo_control'
#       - 'real_vs_pseudo'
#     plus:
#       - 'intention' (align_fracs, choice_labels, window_ms)
#     """
#     import os, glob
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import pearsonr, wilcoxon

#     # ---- canonical goal locations in x–z (same as classify_no_ai_failure_reason) ----
#     KNOWN_GOALS = np.array([
#         [ 7.0,  6.0],
#         [ 3.5,  8.5],
#         [ 0.0,  9.2],
#         [-3.5,  8.5],
#         [-7.0,  6.0],
#     ], dtype=float)

#     # time window (same units as avatarTrajectory['time']) after event
#     # used to compute modelVelocity-based intention fractions
#     align_window = (0.0, 1000.0)   # 0–1000 ms after event

#     # ---------- helpers ----------
#     def safe_pearsonr(x, y):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         m = np.isfinite(x) & np.isfinite(y)
#         if m.sum() < 3:
#             return np.nan, np.nan
#         try:
#             r, p = pearsonr(x[m], y[m])
#         except Exception:
#             r, p = np.nan, np.nan
#         return r, p

#     def xz_from(pos, i=None):
#         if pos is None:
#             return None
#         if i is not None:
#             if isinstance(pos, dict):
#                 return (pos["x"][i], pos["z"][i])
#         else:
#             if isinstance(pos, dict):
#                 return (float(pos[0]), float(pos[2]))
#         try:
#             return (float(pos[0]), float(pos[2]))
#         except Exception:
#             return None

#     def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
#         idx = list(range(len(keep_idx)))
#         if stride and stride > 1:
#             idx = idx[::stride]
#         if min_dt is not None:
#             ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
#             ts_vals = []
#             for k in keep_idx:
#                 t = ai_recs[k].get(ts_key)
#                 if hasattr(t, "timestamp"):
#                     t = t.timestamp()
#                 ts_vals.append(np.nan if t is None else float(t))
#             kept_rel, last_t = [], None
#             for j in idx:
#                 tj = ts_vals[j]
#                 if np.isnan(tj) or last_t is None or (tj - last_t) >= float(min_dt):
#                     kept_rel.append(j); last_t = tj
#             idx = kept_rel
#         if min_dist is not None and len(idx) > 1:
#             kept_rel = [idx[0]]
#             last_p = pos[idx[0]]
#             for j in idx[1:]:
#                 if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
#                     kept_rel.append(j); last_p = pos[j]
#             idx = kept_rel
#         return idx

#     def build_bins_from_pooled(d_all, nb):
#         d_all = np.asarray(d_all, float)
#         d_all = d_all[np.isfinite(d_all)]
#         if d_all.size < 5:
#             return None, None
#         lo, hi = np.percentile(d_all, [1, 99])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = np.nanmin(d_all), np.nanmax(d_all)
#         edges = np.linspace(lo, hi, nb + 1)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#         return edges, centers

#     def per_trial_binned_means(d, a, edges):
#         d = np.asarray(d, float); a = np.asarray(a, float)
#         good = np.isfinite(d) & np.isfinite(a)
#         if good.sum() == 0:
#             return np.full(len(edges) - 1, np.nan)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d >= edges[i]) & (d < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a[m])
#         return out

#     def aggregate_across_trials(per_trial_mat):
#         per_trial_mat = np.asarray(per_trial_mat, float)
#         mean = np.nanmean(per_trial_mat, axis=0)
#         std = np.nanstd(per_trial_mat, axis=0, ddof=1)
#         n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
#         sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
#         lo = mean - 1.96 * sem
#         hi = mean + 1.96 * sem
#         return mean, lo, hi, n

#     def pooled_binned_means(d_all, a_all, edges):
#         d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
#         good = np.isfinite(d_all) & np.isfinite(a_all)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a_all[m])
#         return out

#     def window_mean(t, a, w):
#         """Mean α in time window w=(t_min, t_max) on a single trial."""
#         t = np.asarray(t, float); a = np.asarray(a, float)
#         m = (t >= w[0]) & (t <= w[1]) & np.isfinite(t) & np.isfinite(a)
#         return np.nan if m.sum() == 0 else float(np.nanmean(a[m]))

#     # windows in same units as avatarTrajectory['time'] (ms in your data)
#     pre_window      = (-400.0,   0.0)
#     early_window    = (   0.0,  400.0)
#     late_window     = (1600.0, 2400.0)
#     dip_window      = (   0.0,  800.0)
#     plateau_window  = (1600.0, 2600.0)

#     # pseudo-event centers (ms) before and after the real event
#     pseudo_range_pre  = (-1200.0,  -400.0)
#     pseudo_range_post = (  1000,  2500.0)

#     # recovery band for visualization (gray box on the plot)
#     recovery_band = (500.0, 1500.0)

#     # ---- helper for intention in a post-event window, using modelVelocity ----
#     def intention_from_model_velocity_window(
#         trial,
#         event_time,
#         known_goals,
#         align_window_ms,
#         choice_thresh=0.6,
#         speed_thresh=0.1,
#     ):
#         """
#         Same 60% rule as _compute_intention_from_model_velocity, but restricted
#         to [event_time + align_window_ms[0], event_time + align_window_ms[1]].

#         Returns
#         -------
#         choice_label : 'correct_choice' | 'wrong_choice' | 'ambiguous_choice' | 'no_model'
#         true_frac    : fraction of frames in the window assigned to the true goal
#         best_goal    : goal (x,z) with highest fraction
#         best_frac    : fraction for best_goal
#         """
#         mv = getattr(trial, 'modelVelocity', None)
#         if mv is None:
#             return 'no_model', np.nan, None, 0.0

#         try:
#             vx = np.asarray(mv['vx'], float)
#             vz = np.asarray(mv['vz'], float)
#         except Exception:
#             return 'no_model', np.nan, None, 0.0

#         xs_all = np.asarray(trial.avatarTrajectory['x'], float)
#         zs_all = np.asarray(trial.avatarTrajectory['z'], float)
#         t_all  = np.asarray(trial.avatarTrajectory.get('time', []), float)

#         T = min(len(vx), len(vz), len(xs_all), len(zs_all), len(t_all))
#         if T < 5:
#             return 'no_model', np.nan, None, 0.0

#         vx = vx[:T]; vz = vz[:T]
#         xs = xs_all[:T]; zs = zs_all[:T]; t = t_all[:T]

#         # relative time (ms in your data)
#         t_rel = t - float(event_time)

#         mask = (t_rel >= align_window_ms[0]) & (t_rel <= align_window_ms[1])
#         if not np.any(mask):
#             return 'no_model', np.nan, None, 0.0

#         vx_w = vx[mask]; vz_w = vz[mask]
#         xs_w = xs[mask]; zs_w = zs[mask]

#         v = np.stack([vx_w, vz_w], axis=1)
#         speed = np.linalg.norm(v, axis=1)
#         valid = speed > speed_thresh
#         if not np.any(valid):
#             return 'no_model', np.nan, None, 0.0

#         v = v[valid]; xs_w = xs_w[valid]; zs_w = zs_w[valid]

#         if event_attr == "targetJumpTime":
#             # Respawn
#             if isinstance(trial, dict):
#                 tgt = trial.get("targetJumpPosition", trial.get("targetPosition", None))
#             else:
#                 tgt = getattr(trial, "targetJumpPosition", None)
#                 if tgt is None:
#                     tgt = getattr(trial, "targetPosition", None)
#         else:
#             # Appearing Obstacle tasks
#             if isinstance(trial, dict):
#                 tgt = trial.get("targetPosition", None)
#             else:
#                 tgt = getattr(trial, "targetPosition", None)

#         true_goal_tuple = (float(tgt[0]), float(tgt[2]))

#         # per-goal counts
#         counts = {g: 0 for g in known_goals}
#         total = 0

#         for i in range(len(v)):
#             vi = v[i]
#             vi_norm = np.linalg.norm(vi)
#             if vi_norm < 1e-9:
#                 continue

#             best_goal_frame = None
#             best_cos = -np.inf

#             for (ox, oz) in known_goals:
#                 gvec = np.array([ox - xs_w[i], oz - zs_w[i]])
#                 gnorm = np.linalg.norm(gvec)
#                 if gnorm < 1e-9:
#                     continue
#                 cos = float(np.dot(vi, gvec) / (vi_norm * gnorm + 1e-12))
#                 if cos > best_cos:
#                     best_cos = cos
#                     best_goal_frame = (ox, oz)

#             # same rule: only count if pointing with positive cosine
#             if best_goal_frame is not None and best_cos > 0.0:
#                 counts[best_goal_frame] += 1
#                 total += 1

#         if total == 0:
#             return 'no_model', np.nan, None, 0.0

#         fracs = {g: c / total for g, c in counts.items() if c > 0}
#         if not fracs:
#             return 'no_model', np.nan, None, 0.0

#         best_goal = max(fracs, key=fracs.get)
#         best_frac = fracs[best_goal]
#         true_frac = fracs.get(true_goal_tuple, 0.0)

#         # 60% rule
#         if best_frac < choice_thresh:
#             choice_label = 'ambiguous_choice'
#         elif best_goal == true_goal_tuple:
#             choice_label = 'correct_choice'
#         else:
#             choice_label = 'wrong_choice'

#         return choice_label, float(true_frac), best_goal, float(best_frac)

#     # ---------------- determine event type ----------------
#     effective_task = task if task is not None else experiment
#     if effective_task in ("AI Respawn", "Respawn"):
#         event_attr = "targetJumpTime"
#     else:
#         # Appearing Obstacle 1 & 2
#         event_attr = "obstacleAppearedTime"

#     # ---------------- collect data ----------------
#     if file_path is None:
#         data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
#         pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#     else:
#         pkl_files = [file_path]

#     all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, \
#         pkl_files, ai_trials = load_files(experiment, monkey, base_dir=base_dir)

#     trials_t = []
#     trials_alpha = []
#     pooled_t, pooled_a = [], []
#     r_trials_time = []

#     pre_vals, early_vals, late_vals = [], [], []
#     dip_depths, recovery_times = [], []
#     pre_pseudo_vals, post_pseudo_vals = [], []

#     # intention quality per trial
#     align_fracs = []
#     align_choice_labels = []

#     tmin, tmax = float(t_window[0]), float(t_window[1])

#     for session in all_trials:
#         for trial in session:
#             # answer filter if requested
#             if only_answer is not None:
#                 ans = trial.answer
#                 if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
#                     continue

#             ai_recs = trial.aiVelocities
#             if not ai_recs:
#                 continue

#             # event time
#             if not isinstance(trial, dict):
#                 t_event = getattr(trial, event_attr, None)
#             else:
#                 t_event = trial.get(event_attr, None)
#             if t_event is None:
#                 continue
#             t_event = float(t_event)

#             # Skip if aiVelocity is zero at the start
#             start_idx = next((i for i, rec in enumerate(ai_recs)
#                               if np.linalg.norm(rec['Output']) > 0), None)
#             if start_idx is None:
#                 continue
#             ai_recs = ai_recs[start_idx:]

#             # positions from avatarTrajectory
#             xs, zs, keep_idx = [], [], []
#             for i, rec in enumerate(ai_recs):
#                 p = xz_from(trial.avatarTrajectory, i)
#                 if p is None or np.isnan(p[0]) or np.isnan(p[1]):
#                     continue
#                 xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
#             if len(keep_idx) < 3:
#                 continue
#             pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

#             # entropy from ai_recs
#             ent_raw = []
#             for rec in ai_recs:
#                 e = rec.get("EntropyLb")
#                 if isinstance(e, (list, tuple, np.ndarray)):
#                     e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
#                 ent_raw.append(np.nan if e is None else float(e))
#             ent_raw = np.asarray(ent_raw, float)
#             ent_kept = ent_raw[keep_idx]

#             # thinning
#             idx_rel = thin_indices(pos_full, keep_idx, ai_recs,
#                                    stride=stride, min_dt=min_dt,
#                                    min_dist=min_dist, use_output_ts=True)
#             if len(idx_rel) < 3:
#                 continue

#             pos = pos_full[idx_rel]
#             ent_kept = ent_kept[idx_rel]

#             # time vector from avatarTrajectory
#             traj_t_all = None
#             if isinstance(trial.avatarTrajectory, dict):
#                 traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []), dtype=float)
#             if traj_t_all is None or traj_t_all.size == 0:
#                 continue
#             try:
#                 traj_t_kept = traj_t_all[keep_idx][idx_rel]
#             except Exception:
#                 continue

#             alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

#             # relative time (ms in your data)
#             t_rel = traj_t_kept - t_event

#             # mask analysis window
#             mask = (t_rel >= tmin) & (t_rel <= tmax) & np.isfinite(alpha_vals) & np.isfinite(t_rel)
#             if mask.sum() < 3:
#                 continue

#             t_rel_trial = t_rel[mask]
#             alpha_trial = alpha_vals[mask]
#             pos_trial = pos[mask]

#             trials_t.append(t_rel_trial)
#             trials_alpha.append(alpha_trial)
#             pooled_t.append(t_rel_trial)
#             pooled_a.append(alpha_trial)

#             r_trials_time.append(
#                 np.corrcoef(t_rel_trial, alpha_trial)[0, 1] if t_rel_trial.size >= 3 else np.nan
#             )

#             # ---- intention quality (modelVelocity, same 60% rule) ----
#             choice_label, true_frac, best_goal, best_frac = intention_from_model_velocity_window(
#                 trial,
#                 event_time=t_event,
#                 known_goals=[(float(x), float(z)) for (x, z) in KNOWN_GOALS],
#                 align_window_ms=align_window,
#                 choice_thresh=0.6,
#                 speed_thresh=0.1,
#             )
#             align_fracs.append(true_frac)
#             align_choice_labels.append(choice_label)
#             # NOTE: if you ever want to *only* keep "correct_choice" trials,
#             # you can also gate here by choice_label before appending metrics.

#             # ---- real-event pre/early/late ----
#             pre_val   = window_mean(t_rel_trial, alpha_trial, pre_window)
#             early_val = window_mean(t_rel_trial, alpha_trial, early_window)
#             late_val  = window_mean(t_rel_trial, alpha_trial, late_window)

#             pre_vals.append(pre_val)
#             early_vals.append(early_val)
#             late_vals.append(late_val)

#             # ---- dip depth and recovery ----
#             dip_depth = np.nan
#             recovery_time = np.nan

#             if np.isfinite(pre_val):
#                 m_dip = (t_rel_trial >= dip_window[0]) & (t_rel_trial <= dip_window[1])
#                 if np.any(m_dip):
#                     a_seg = alpha_trial[m_dip]
#                     t_seg = t_rel_trial[m_dip]
#                     try:
#                         idx_min = int(np.nanargmin(a_seg))
#                         alpha_min = float(a_seg[idx_min])
#                         t_min = float(t_seg[idx_min])
#                         dip_depth = alpha_min - pre_val
#                     except Exception:
#                         alpha_min = np.nan
#                         t_min = np.nan

#                 # simple recovery: first time α ≥ pre_val again after the dip
#                 if np.isfinite(dip_depth):
#                     m_post = (t_rel_trial > t_min) & np.isfinite(alpha_trial)
#                     if np.any(m_post):
#                         t_post = t_rel_trial[m_post]
#                         a_post = alpha_trial[m_post]
#                         idx = np.where(a_post >= pre_val)[0]
#                         if idx.size > 0:
#                             recovery_time = float(t_post[idx[0]])

#             dip_depths.append(dip_depth)
#             recovery_times.append(recovery_time)

#             # ---- pseudo-events: one before, one after ----
#             # pre-pseudo
#             pseudo_t0_pre = np.random.uniform(pseudo_range_pre[0], pseudo_range_pre[1])
#             t_centered_pre = t_rel_trial - pseudo_t0_pre
#             pre_p  = window_mean(t_centered_pre, alpha_trial, pre_window)
#             post_p = window_mean(t_centered_pre, alpha_trial, early_window)
#             pre_pseudo_vals.append(pre_p)
#             post_pseudo_vals.append(post_p)

#             # post-pseudo (symmetry, not stored separately)
#             pseudo_t0_post = np.random.uniform(pseudo_range_post[0], pseudo_range_post[1])
#             _ = window_mean(t_rel_trial - pseudo_t0_post, alpha_trial, pre_window)
#             _ = window_mean(t_rel_trial - pseudo_t0_post, alpha_trial, early_window)

#     result = {}

#     if not pooled_t:
#         print("[WARN] No valid samples around event.")
#         return result

#     # ---------- optional filter by intention quality ----------
#     align_fracs = np.asarray(align_fracs, float)
#     keep_mask = np.isfinite(align_fracs)
#     if min_align_frac is not None:
#         keep_mask &= (align_fracs >= float(min_align_frac))
#     if max_align_frac is not None:
#         keep_mask &= (align_fracs <= float(max_align_frac))

#     if not np.any(keep_mask):
#         print("[WARN] All trials rejected by align_frac filter.")
#         return {}

#     # apply keep_mask to per-trial arrays
#     trials_t      = [t for k, t in enumerate(trials_t) if keep_mask[k]]
#     trials_alpha  = [a for k, a in enumerate(trials_alpha) if keep_mask[k]]
#     r_trials_time = [r for k, r in enumerate(r_trials_time) if keep_mask[k]]

#     pre_vals        = np.asarray(pre_vals, float)[keep_mask]
#     early_vals      = np.asarray(early_vals, float)[keep_mask]
#     late_vals       = np.asarray(late_vals, float)[keep_mask]
#     dip_depths      = np.asarray(dip_depths, float)[keep_mask]
#     recovery_times  = np.asarray(recovery_times, float)[keep_mask]
#     pre_pseudo_vals = np.asarray(pre_pseudo_vals, float)[keep_mask]
#     post_pseudo_vals = np.asarray(post_pseudo_vals, float)[keep_mask]
#     align_fracs_kept = align_fracs[keep_mask]
#     choice_labels_kept = [lab for k, lab in enumerate(align_choice_labels) if keep_mask[k]]

#     # rebuild pooled_t, pooled_a after masking
#     pooled_t = np.concatenate(trials_t)
#     pooled_a = np.concatenate(trials_alpha)

#     edges_t, centers_t = build_bins_from_pooled(pooled_t, nbins)
#     if edges_t is None:
#         print("[WARN] Not enough data to build time bins.")
#         return result

#     per_trial_mat_t = [
#         per_trial_binned_means(t, a, edges_t)
#         for t, a in zip(trials_t, trials_alpha)
#     ]
#     per_trial_mat_t = np.asarray(per_trial_mat_t, float)

#     mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)
#     pooled_curve_t = pooled_binned_means(pooled_t, pooled_a, edges_t)  # not plotted, but kept if you want it

#     # ---------------- plot ----------------
#     fig, ax = plt.subplots(figsize=(8.0, 5.0))

#     x_centers_ms = centers_t  # units are ms in your data

#     if show_per_trial:
#         for row in per_trial_mat_t:
#             ax.plot(x_centers_ms, row, lw=0.8, alpha=0.15)

#     ax.fill_between(x_centers_ms, lo_t, hi_t, alpha=0.25, label="Across-trials 95% CI")
#     ax.plot(x_centers_ms, mean_t, lw=2.5, label="Across-trials mean")

#     # event line
#     ax.axvline(0.0, color='tab:purple', ls='-', lw=2, label="Event (t=0)")

#     # recovery band (visual only)
#     ax.axvspan(recovery_band[0], recovery_band[1], color='0.9', alpha=0.5,
#                label="Recovery zone (visual)")

#     r_pool, p_pool = safe_pearsonr(pooled_t, pooled_a)
#     med_r = np.nanmedian(
#         [r for r in np.asarray(r_trials_time, float) if np.isfinite(r)]
#     ) if len(r_trials_time) else np.nan

#     def fmt_p(p):
#         if not np.isfinite(p):
#             return "p=NA"
#         return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

#     ax.text(
#         0.02, 0.95,
#         f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
#         transform=ax.transAxes, ha='left', va='top', fontsize=10
#     )

#     ax.set_xlabel("Time from event (ms)")
#     ax.set_ylabel("Posterior Confidence Index")
#     ax.set_title(f"{monkey} – {experiment}\nα vs time from event")
#     ax.grid(True, alpha=0.3)
#     ax.legend(frameon=True, fontsize=9)
#     plt.tight_layout()

#     try:
#         save_plot(fig, f"alpha_vs_time_FROM_EVENT_{monkey}_{experiment}", subfolder="Entropy")
#     except Exception:
#         pass
#     plt.show(); plt.close(fig)

#     # -------------- stats outputs --------------
#     def wilcoxon_or_nan(x):
#         x = np.asarray(x, float)
#         x = x[np.isfinite(x)]
#         if x.size < 3:
#             return np.nan
#         return float(wilcoxon(x, alternative='two-sided').pvalue)

#     delta_early = early_vals - pre_vals
#     delta_late  = late_vals  - pre_vals

#     # real-event stats
#     p_delta_early = wilcoxon_or_nan(delta_early)
#     p_delta_late  = wilcoxon_or_nan(delta_late)
#     p_dip_depth   = wilcoxon_or_nan(dip_depths)

#     # pseudo-event: pre vs post around pseudo-event
#     delta_pseudo = post_pseudo_vals - pre_pseudo_vals
#     p_delta_pseudo = wilcoxon_or_nan(delta_pseudo)

#     result['time'] = {
#         'centers_s': centers_t,
#         'mean': mean_t,
#         'lo': lo_t,
#         'hi': hi_t,
#         'n_trials_per_bin': n_t,
#         'r_pooled': r_pool,
#         'r_trials': r_trials_time,
#         't_window': t_window,
#     }

#     result['event_windows'] = {
#         'pre_mean':   float(np.nanmean(pre_vals)),
#         'early_mean': float(np.nanmean(early_vals)),
#         'late_mean':  float(np.nanmean(late_vals)),
#         'delta_early_mean':   float(np.nanmean(delta_early)),
#         'delta_early_median': float(np.nanmedian(delta_early)),
#         'delta_early_p':      p_delta_early,
#         'delta_late_mean':   float(np.nanmean(delta_late)),
#         'delta_late_median': float(np.nanmedian(delta_late)),
#         'delta_late_p':      p_delta_late,
#         'n_trials': int(np.sum(np.isfinite(pre_vals) & np.isfinite(early_vals))),
#     }

#     result['dip_recovery'] = {
#         'dip_depth_mean':   float(np.nanmean(dip_depths)),
#         'dip_depth_median': float(np.nanmedian(dip_depths)),
#         'dip_depth_p':      p_dip_depth,
#         'recovery_time_median_s': float(np.nanmedian(recovery_times)),
#         'recovery_time_mean_s':   float(np.nanmean(recovery_times)),
#         'recovery_time_values_s': recovery_times,
#     }

#     result['pseudo_control'] = {
#         'delta_pseudo_mean':   float(np.nanmean(delta_pseudo)),
#         'delta_pseudo_median': float(np.nanmedian(delta_pseudo)),
#         'delta_pseudo_p':      p_delta_pseudo,
#     }

#     # real vs pseudo dip comparison (per trial)
#     real_minus_pseudo = dip_depths - delta_pseudo
#     p_real_vs_pseudo = wilcoxon_or_nan(real_minus_pseudo)
#     result['real_vs_pseudo'] = {
#         'diff_mean':   float(np.nanmean(real_minus_pseudo)),
#         'diff_median': float(np.nanmedian(real_minus_pseudo)),
#         'diff_p':      p_real_vs_pseudo,
#     }

#     # intention stats
#     result['intention'] = {
#         'align_fracs': align_fracs_kept,        # true-goal fraction in window
#         'choice_labels': choice_labels_kept,    # 'correct_choice'/'wrong_choice'/...
#         'window_ms': align_window,
#     }

#     return result
#old version above
def plot_alpha_vs_time_from_obstacle_appearance(
    monkey, experiment, base_dir,
    file_path=None,
    stride=4,
    min_dt=None,
    min_dist=None,
    t_window=(-2000.0, 3000.0),
    nbins=24,
    min_alpha=0.2,
    only_answer=None,
    task=None,
    show_per_trial=False,
    min_align_frac=None,   # filter by intention (true goal fraction)
    max_align_frac=None,   # optional upper bound
):
    """
    α (= 1 − normalized EntropyLb) vs time from event (obstacle appearance or target jump).

    Event = obstacleAppearedTime for Appearing Obstacle tasks,
            targetJumpTime for Respawn tasks.

    NEW:
      For "AI Obstacle" (fixed obstacle), define the "event" as TRIAL START (t=0),
      and by default plot from start to end (full trial).

    Uses modelVelocity to compute, in a post-event window, the fraction of frames
    where the decoded intention points to the TRUE target, with the same 60% rule
    as classify_no_ai_failure_reason.

    Returned result dict still has:
      - 'time'
      - 'event_windows'
      - 'dip_recovery'
      - 'pseudo_control'
      - 'real_vs_pseudo'
    plus:
      - 'intention' (align_fracs, choice_labels, window_ms)
    """
    import os, glob
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, wilcoxon

    # ---- canonical goal locations in x–z (same as classify_no_ai_failure_reason) ----
    KNOWN_GOALS = np.array([
        [ 7.0,  6.0],
        [ 3.5,  8.5],
        [ 0.0,  9.2],
        [-3.5,  8.5],
        [-7.0,  6.0],
    ], dtype=float)

    # time window (same units as avatarTrajectory['time']) after event
    # used to compute modelVelocity-based intention fractions
    align_window = (0.0, 1000.0)   # 0–1000 ms after event/start

    # ---------------- determine event type ----------------
    effective_task = task if task is not None else experiment
    is_fixed_obstacle = effective_task in ("AI Obstacle", "Obstacle", "Fixed Obstacle")

    if effective_task in ("AI Respawn", "Respawn"):
        event_attr = "targetJumpTime"
    elif is_fixed_obstacle:
        event_attr = None  # NEW: use trial start as t=0
    else:
        # Appearing Obstacle 1 & 2
        event_attr = "obstacleAppearedTime"

    # ---------- helpers ----------
    def safe_pearsonr(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan, np.nan
        try:
            r, p = pearsonr(x[m], y[m])
        except Exception:
            r, p = np.nan, np.nan
        return r, p

    def xz_from(pos, i=None):
        if pos is None:
            return None
        if i is not None:
            if isinstance(pos, dict):
                return (pos["x"][i], pos["z"][i])
        else:
            if isinstance(pos, dict):
                return (float(pos[0]), float(pos[2]))
        try:
            return (float(pos[0]), float(pos[2]))
        except Exception:
            return None

    def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
        idx = list(range(len(keep_idx)))
        if stride and stride > 1:
            idx = idx[::stride]
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
        if min_dist is not None and len(idx) > 1:
            kept_rel = [idx[0]]
            last_p = pos[idx[0]]
            for j in idx[1:]:
                if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                    kept_rel.append(j); last_p = pos[j]
            idx = kept_rel
        return idx

    def build_bins_from_pooled(d_all, nb):
        d_all = np.asarray(d_all, float)
        d_all = d_all[np.isfinite(d_all)]
        if d_all.size < 5:
            return None, None
        lo, hi = np.percentile(d_all, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(d_all), np.nanmax(d_all)
        edges = np.linspace(lo, hi, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    def per_trial_binned_means(d, a, edges):
        d = np.asarray(d, float); a = np.asarray(a, float)
        good = np.isfinite(d) & np.isfinite(a)
        if good.sum() == 0:
            return np.full(len(edges) - 1, np.nan)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d >= edges[i]) & (d < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a[m])
        return out

    def aggregate_across_trials(per_trial_mat):
        per_trial_mat = np.asarray(per_trial_mat, float)
        mean = np.nanmean(per_trial_mat, axis=0)
        std = np.nanstd(per_trial_mat, axis=0, ddof=1)
        n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
        sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
        return mean, lo, hi, n

    def pooled_binned_means(d_all, a_all, edges):
        d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
        good = np.isfinite(d_all) & np.isfinite(a_all)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a_all[m])
        return out

    def window_mean(t, a, w):
        """Mean α in time window w=(t_min, t_max) on a single trial."""
        t = np.asarray(t, float); a = np.asarray(a, float)
        m = (t >= w[0]) & (t <= w[1]) & np.isfinite(t) & np.isfinite(a)
        return np.nan if m.sum() == 0 else float(np.nanmean(a[m]))

    # windows in same units as avatarTrajectory['time'] (ms in your data)
    pre_window      = (-400.0,   0.0)
    early_window    = (   0.0,  400.0)
    late_window     = (1600.0, 2400.0)
    dip_window      = (   0.0,  800.0)
    plateau_window  = (1600.0, 2600.0)

    # pseudo-event centers (ms) before and after the real event/start
    pseudo_range_pre  = (-1200.0,  -400.0)
    pseudo_range_post = (  1000.0,  2500.0)

    # recovery band for visualization (gray box on the plot)
    recovery_band = (500.0, 1500.0)

    # ---- helper for intention in a post-event window, using modelVelocity ----
    def intention_from_model_velocity_window(
        trial,
        event_time,
        known_goals,
        align_window_ms,
        choice_thresh=0.6,
        speed_thresh=0.1,
    ):
        """
        Same 60% rule as _compute_intention_from_model_velocity, but restricted
        to [event_time + align_window_ms[0], event_time + align_window_ms[1]].

        Returns
        -------
        choice_label : 'correct_choice' | 'wrong_choice' | 'ambiguous_choice' | 'no_model'
        true_frac    : fraction of frames in the window assigned to the true goal
        best_goal    : goal (x,z) with highest fraction
        best_frac    : fraction for best_goal
        """
        mv = getattr(trial, 'modelVelocity', None)
        if mv is None:
            return 'no_model', np.nan, None, 0.0

        try:
            vx = np.asarray(mv['vx'], float)
            vz = np.asarray(mv['vz'], float)
        except Exception:
            return 'no_model', np.nan, None, 0.0

        xs_all = np.asarray(trial.avatarTrajectory['x'], float)
        zs_all = np.asarray(trial.avatarTrajectory['z'], float)
        t_all  = np.asarray(trial.avatarTrajectory.get('time', []), float)

        T = min(len(vx), len(vz), len(xs_all), len(zs_all), len(t_all))
        if T < 5:
            return 'no_model', np.nan, None, 0.0

        vx = vx[:T]; vz = vz[:T]
        xs = xs_all[:T]; zs = zs_all[:T]; t = t_all[:T]

        # relative time (ms in your data)
        t_rel = t - float(event_time)

        mask = (t_rel >= align_window_ms[0]) & (t_rel <= align_window_ms[1])
        if not np.any(mask):
            return 'no_model', np.nan, None, 0.0

        vx_w = vx[mask]; vz_w = vz[mask]
        xs_w = xs[mask]; zs_w = zs[mask]

        v = np.stack([vx_w, vz_w], axis=1)
        speed = np.linalg.norm(v, axis=1)
        valid = speed > speed_thresh
        if not np.any(valid):
            return 'no_model', np.nan, None, 0.0

        v = v[valid]; xs_w = xs_w[valid]; zs_w = zs_w[valid]

        # Determine the "true target" reference
        if event_attr == "targetJumpTime":
            # Respawn
            if isinstance(trial, dict):
                tgt = trial.get("targetJumpPosition", trial.get("targetPosition", None))
            else:
                tgt = getattr(trial, "targetJumpPosition", None)
                if tgt is None:
                    tgt = getattr(trial, "targetPosition", None)
        else:
            # Appearing Obstacle or fixed obstacle (start as event)
            if isinstance(trial, dict):
                tgt = trial.get("targetPosition", None)
            else:
                tgt = getattr(trial, "targetPosition", None)

        if tgt is None:
            return 'no_model', np.nan, None, 0.0

        true_goal_tuple = (float(tgt[0]), float(tgt[2]))

        # per-goal counts
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
                gvec = np.array([ox - xs_w[i], oz - zs_w[i]])
                gnorm = np.linalg.norm(gvec)
                if gnorm < 1e-9:
                    continue
                cos = float(np.dot(vi, gvec) / (vi_norm * gnorm + 1e-12))
                if cos > best_cos:
                    best_cos = cos
                    best_goal_frame = (ox, oz)

            # same rule: only count if pointing with positive cosine
            if best_goal_frame is not None and best_cos > 0.0:
                counts[best_goal_frame] += 1
                total += 1

        if total == 0:
            return 'no_model', np.nan, None, 0.0

        fracs = {g: c / total for g, c in counts.items() if c > 0}
        if not fracs:
            return 'no_model', np.nan, None, 0.0

        best_goal = max(fracs, key=fracs.get)
        best_frac = fracs[best_goal]
        true_frac = fracs.get(true_goal_tuple, 0.0)

        # 60% rule
        if best_frac < choice_thresh:
            choice_label = 'ambiguous_choice'
        elif best_goal == true_goal_tuple:
            choice_label = 'correct_choice'
        else:
            choice_label = 'wrong_choice'

        return choice_label, float(true_frac), best_goal, float(best_frac)

    # ---------------- collect data ----------------
    rng = np.random.default_rng(0)
    if file_path is None:
        data_dir = os.path.join(base_dir, monkey, experiment, "AiFiles")
        pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    else:
        pkl_files = [file_path]

    all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, \
        pkl_files, ai_trials = load_files(experiment, monkey, base_dir=base_dir)

    trials_t = []
    trials_alpha = []
    pooled_t, pooled_a = [], []
    r_trials_time = []

    pre_vals, early_vals, late_vals = [], [], []
    closest_times = []
    dip_depths, recovery_times = [], []
    pre_pseudo_vals, post_pseudo_vals = [], []

    # optional event-centered storage (useful for debugging / future plotting)
    trials_t_event = []
    trials_alpha_event = []

    # intention quality per trial
    align_fracs = []
    align_choice_labels = []
    # session-level summaries for inferential stats
    session_stats = []

    # For fixed obstacle: default to full trial (start->end) unless user explicitly set a different t_window
    tmin, tmax = float(t_window[0]), float(t_window[1])
    if is_fixed_obstacle and t_window == (-2000.0, 3000.0):
        tmin, tmax = 0.0, np.inf

    for session in all_trials:
        # per-session containers
        sess_pre_vals = []
        sess_early_vals = []
        sess_late_vals = []
        sess_dip_depths = []
        sess_recovery_times = []
        sess_pre_pseudo_vals = []
        sess_post_pseudo_vals = []
        for trial in session:
            # default
            t_closest_rel_trial = np.nan

            # answer filter if requested
            if only_answer is not None:
                ans = trial.answer
                if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
                    continue

            ai_recs = trial.aiVelocities
            if not ai_recs:
                continue

            # ---- event/start time ----
            if event_attr is None:
                # fixed obstacle: event = trial start (t=0)
                t_event = getattr(trial, "start", None)
                if t_event is None:
                    # fallback to first trajectory time
                    if isinstance(trial.avatarTrajectory, dict):
                        tt = np.asarray(trial.avatarTrajectory.get("time", []), dtype=float)
                        t_event = float(tt[0]) if tt.size else None
                if t_event is None:
                    continue
                t_event = float(t_event)
            else:
                if not isinstance(trial, dict):
                    t_event = getattr(trial, event_attr, None)
                else:
                    t_event = trial.get(event_attr, None)
                if t_event is None:
                    continue
                t_event = float(t_event)

            # Skip if aiVelocity is zero at the start
            start_idx = next((i for i, rec in enumerate(ai_recs)
                              if np.linalg.norm(rec['Output']) > 0), None)
            if start_idx is None:
                continue
            ai_recs = ai_recs[start_idx:]

            # positions from avatarTrajectory
            xs, zs, keep_idx = [], [], []
            for i, rec in enumerate(ai_recs):
                p = xz_from(trial.avatarTrajectory, i)
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
            if len(keep_idx) < 3:
                continue
            pos_full = np.column_stack([np.asarray(xs), np.asarray(zs)])

            # entropy from ai_recs
            ent_raw = []
            for rec in ai_recs:
                e = rec.get("EntropyLb")
                if isinstance(e, (list, tuple, np.ndarray)):
                    e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
                ent_raw.append(np.nan if e is None else float(e))
            ent_raw = np.asarray(ent_raw, float)
            ent_kept = ent_raw[keep_idx]

            # thinning
            idx_rel = thin_indices(
                pos_full, keep_idx, ai_recs,
                stride=stride, min_dt=min_dt,
                min_dist=min_dist, use_output_ts=True
            )
            if len(idx_rel) < 3:
                continue

            pos = pos_full[idx_rel]
            ent_kept = ent_kept[idx_rel]

            # time vector from avatarTrajectory
            traj_t_all = None
            if isinstance(trial.avatarTrajectory, dict):
                traj_t_all = np.asarray(trial.avatarTrajectory.get("time", []), dtype=float)
            if traj_t_all is None or traj_t_all.size == 0:
                continue
            try:
                traj_t_kept = traj_t_all[keep_idx][idx_rel]
            except Exception:
                continue

            alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

            # relative time (ms in your data)
            t_rel = traj_t_kept - t_event

            # mask analysis window
            mask = (t_rel >= tmin) & (t_rel <= tmax) & np.isfinite(alpha_vals) & np.isfinite(t_rel)
            if mask.sum() < 3:
                continue

            t_rel_trial = t_rel[mask]
            alpha_trial = alpha_vals[mask]
            pos_trial = pos[mask]

            trials_t.append(t_rel_trial)
            trials_alpha.append(alpha_trial)
            pooled_t.append(t_rel_trial)
            pooled_a.append(alpha_trial)

            r_trials_time.append(
                np.corrcoef(t_rel_trial, alpha_trial)[0, 1] if t_rel_trial.size >= 3 else np.nan
            )

            # ---- intention quality (modelVelocity, same 60% rule) ----
            choice_label, true_frac, best_goal, best_frac = intention_from_model_velocity_window(
                trial,
                event_time=t_event,
                known_goals=[(float(x), float(z)) for (x, z) in KNOWN_GOALS],
                align_window_ms=align_window,
                choice_thresh=0.6,
                speed_thresh=0.1,
            )
            align_fracs.append(true_frac)
            align_choice_labels.append(choice_label)

            # ---- compute closest obstacle time FIRST for fixed obstacle ----
            if is_fixed_obstacle:
                try:
                    obst = getattr(trial, "obstaclePosition", None)
                    if obst is not None and len(obst) >= 3:
                        ox = float(obst[0])
                        oz = float(obst[2])

                        d = np.sqrt((pos_trial[:, 0] - ox)**2 + (pos_trial[:, 1] - oz)**2)
                        if np.any(np.isfinite(d)):
                            i_min = int(np.nanargmin(d))
                            t_closest_rel_trial = float(t_rel_trial[i_min])
                            closest_times.append(t_closest_rel_trial)
                except Exception:
                    t_closest_rel_trial = np.nan

            # ---- real-event/stat axis ----
            # For fixed obstacle, center on closest obstacle approach.
            # Otherwise, center on task event (already t=0).
            if is_fixed_obstacle and np.isfinite(t_closest_rel_trial):
                t_real = t_rel_trial - t_closest_rel_trial
                trials_t_event.append(t_real)
                trials_alpha_event.append(alpha_trial)
            else:
                t_real = t_rel_trial

            # ---- real-event pre/early/late ----
            pre_val   = window_mean(t_real, alpha_trial, pre_window)
            early_val = window_mean(t_real, alpha_trial, early_window)
            late_val  = window_mean(t_real, alpha_trial, late_window)

            pre_vals.append(pre_val)
            early_vals.append(early_val)
            late_vals.append(late_val)
            sess_pre_vals.append(pre_val)
            sess_early_vals.append(early_val)
            sess_late_vals.append(late_val)

            # ---- dip depth and recovery ----
            dip_depth = np.nan
            recovery_time = np.nan

            if np.isfinite(pre_val):
                m_dip = (t_real >= dip_window[0]) & (t_real <= dip_window[1])
                if np.any(m_dip):
                    a_seg = alpha_trial[m_dip]
                    t_seg = t_real[m_dip]
                    try:
                        idx_min = int(np.nanargmin(a_seg))
                        alpha_min = float(a_seg[idx_min])
                        t_minloc = float(t_seg[idx_min])
                        dip_depth = alpha_min - pre_val
                    except Exception:
                        alpha_min = np.nan
                        t_minloc = np.nan

                # simple recovery: first time α ≥ pre_val again after the dip
                if np.isfinite(dip_depth):
                    m_post = (t_real > t_minloc) & np.isfinite(alpha_trial)
                    if np.any(m_post):
                        t_post = t_real[m_post]
                        a_post = alpha_trial[m_post]
                        idx = np.where(a_post >= pre_val)[0]
                        if idx.size > 0:
                            recovery_time = float(t_post[idx[0]])

            dip_depths.append(dip_depth)
            recovery_times.append(recovery_time)
            sess_dip_depths.append(dip_depth)
            sess_recovery_times.append(recovery_time)

            # ---- pseudo-events ----
            # random anchor within the trial, excluding region near the real event for fixed obstacle
            trial_min = float(np.nanmin(t_rel_trial))
            trial_max = float(np.nanmax(t_rel_trial))

            # Need enough room for both pre and early windows
            pseudo_low  = trial_min - pre_window[0]   # pre_window[0] is typically negative
            pseudo_high = trial_max - early_window[1]

            pre_p, post_p = np.nan, np.nan

            if pseudo_high > pseudo_low:
                candidates = rng.uniform(pseudo_low, pseudo_high, size=50)

                if is_fixed_obstacle and np.isfinite(t_closest_rel_trial):
                    exclusion_margin = max(
                        abs(pre_window[0]), abs(pre_window[1]),
                        abs(early_window[0]), abs(early_window[1]),
                        abs(dip_window[0]), abs(dip_window[1]),
                    )
                    candidates = candidates[np.abs(candidates - t_closest_rel_trial) > exclusion_margin]

                for pseudo_t0 in candidates:
                    t_pseudo = t_rel_trial - pseudo_t0
                    pre_tmp  = window_mean(t_pseudo, alpha_trial, pre_window)
                    post_tmp = window_mean(t_pseudo, alpha_trial, early_window)

                    if np.isfinite(pre_tmp) and np.isfinite(post_tmp):
                        pre_p, post_p = pre_tmp, post_tmp
                        break

            pre_pseudo_vals.append(pre_p)
            post_pseudo_vals.append(post_p)
            sess_pre_pseudo_vals.append(pre_p)
            sess_post_pseudo_vals.append(post_p)

        # ---------- summarize this session ----------
        sess_pre_vals = np.asarray(sess_pre_vals, float)
        sess_early_vals = np.asarray(sess_early_vals, float)
        sess_late_vals = np.asarray(sess_late_vals, float)
        sess_dip_depths = np.asarray(sess_dip_depths, float)
        sess_recovery_times = np.asarray(sess_recovery_times, float)
        sess_pre_pseudo_vals = np.asarray(sess_pre_pseudo_vals, float)
        sess_post_pseudo_vals = np.asarray(sess_post_pseudo_vals, float)

        sess_delta_early = sess_early_vals - sess_pre_vals
        sess_delta_recovery = sess_late_vals - sess_early_vals
        sess_delta_pseudo = sess_post_pseudo_vals - sess_pre_pseudo_vals
        sess_real_minus_pseudo = sess_delta_early - sess_delta_pseudo

        sess_has_dip = (
            np.isfinite(sess_pre_vals) &
            np.isfinite(sess_early_vals) &
            np.isfinite(sess_late_vals) &
            (sess_early_vals < sess_pre_vals) &
            (sess_late_vals > sess_early_vals)
        )

        sess_dip_depth_valid = np.full_like(sess_dip_depths, np.nan, dtype=float)
        sess_dip_depth_valid[sess_has_dip] = sess_dip_depths[sess_has_dip]

        sess_recovery_time_valid = np.full_like(sess_recovery_times, np.nan, dtype=float)
        sess_recovery_time_valid[sess_has_dip] = sess_recovery_times[sess_has_dip]

        # one robust summary value per session
        session_stats.append({
            'delta_early_median': float(np.nanmedian(sess_delta_early)) if np.any(np.isfinite(sess_delta_early)) else np.nan,
            'delta_recovery_median': float(np.nanmedian(sess_delta_recovery)) if np.any(np.isfinite(sess_delta_recovery)) else np.nan,
            'delta_pseudo_median': float(np.nanmedian(sess_delta_pseudo)) if np.any(np.isfinite(sess_delta_pseudo)) else np.nan,
            'real_minus_pseudo_median': float(np.nanmedian(sess_real_minus_pseudo)) if np.any(np.isfinite(sess_real_minus_pseudo)) else np.nan,
            'dip_depth_valid_median': float(np.nanmedian(sess_dip_depth_valid)) if np.any(np.isfinite(sess_dip_depth_valid)) else np.nan,
            'recovery_time_median_ms': float(np.nanmedian(sess_recovery_time_valid)) if np.any(np.isfinite(sess_recovery_time_valid)) else np.nan,
            'n_trials': int(np.sum(np.isfinite(sess_pre_vals) & np.isfinite(sess_early_vals) & np.isfinite(sess_late_vals))),
            'n_dip_trials': int(np.sum(sess_has_dip)),
            'frac_dip_trials': float(np.mean(sess_has_dip)) if sess_has_dip.size else np.nan,
        })

    result = {}
    t_closest_med = np.nanmedian(closest_times)

    if not pooled_t:
        print("[WARN] No valid samples around event/start.")
        return result

    # ---------- optional filter by intention quality ----------
    align_fracs = np.asarray(align_fracs, float)
    # Only enforce finite align_fracs if user asked for an align_frac filter
    if (min_align_frac is None) and (max_align_frac is None):
        keep_mask = np.ones_like(align_fracs, dtype=bool)
    else:
        keep_mask = np.isfinite(align_fracs)
        if min_align_frac is not None:
            keep_mask &= (align_fracs >= float(min_align_frac))
        if max_align_frac is not None:
            keep_mask &= (align_fracs <= float(max_align_frac))

    if not np.any(keep_mask):
        print("[WARN] All trials rejected by align_frac filter.")
        return {}

    # apply keep_mask to per-trial arrays
    trials_t      = [t for k, t in enumerate(trials_t) if keep_mask[k]]
    trials_alpha  = [a for k, a in enumerate(trials_alpha) if keep_mask[k]]
    r_trials_time = [r for k, r in enumerate(r_trials_time) if keep_mask[k]]

    pre_vals         = np.asarray(pre_vals, float)[keep_mask]
    early_vals       = np.asarray(early_vals, float)[keep_mask]
    late_vals        = np.asarray(late_vals, float)[keep_mask]
    dip_depths       = np.asarray(dip_depths, float)[keep_mask]
    recovery_times   = np.asarray(recovery_times, float)[keep_mask]
    pre_pseudo_vals  = np.asarray(pre_pseudo_vals, float)[keep_mask]
    post_pseudo_vals = np.asarray(post_pseudo_vals, float)[keep_mask]
    align_fracs_kept = align_fracs[keep_mask]
    choice_labels_kept = [lab for k, lab in enumerate(align_choice_labels) if keep_mask[k]]

    # rebuild pooled after masking
    pooled_t = np.concatenate(trials_t)
    pooled_a = np.concatenate(trials_alpha)

    edges_t, centers_t = build_bins_from_pooled(pooled_t, nbins)
    if edges_t is None:
        print("[WARN] Not enough data to build time bins.")
        return result

    per_trial_mat_t = [
        per_trial_binned_means(t, a, edges_t)
        for t, a in zip(trials_t, trials_alpha)
    ]
    per_trial_mat_t = np.asarray(per_trial_mat_t, float)

    mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)
    _ = pooled_binned_means(pooled_t, pooled_a, edges_t)

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    x_centers_ms = centers_t  # units are ms in your data

    # AI Obstacle only -> shift plotted x-axis so first finite bin is at 0
    if is_fixed_obstacle:
        finite_bins = np.isfinite(mean_t) & np.isfinite(x_centers_ms)
        if np.any(finite_bins):
            x0 = float(x_centers_ms[np.where(finite_bins)[0][0]])
            x_centers_ms = x_centers_ms - x0

    if show_per_trial:
        for row in per_trial_mat_t:
            ax.plot(x_centers_ms, row, lw=0.8, alpha=0.15)

    ax.fill_between(x_centers_ms, lo_t, hi_t, alpha=0.25, label="Across-trials 95% CI")
    ax.plot(x_centers_ms, mean_t, lw=2.5, label="Across-trials mean")

    # event line + labels
    if is_fixed_obstacle:
        if np.isfinite(t_closest_med):
            x_line = (t_closest_med - x0) if ('x0' in locals() and np.isfinite(x0)) else t_closest_med
            ax.axvline(x_line, color='tab:purple', ls='-', lw=2, label="Closest to obstacle (median)")
    else:
        ax.axvline(0.0, color='tab:purple', ls='-', lw=2, label="Event (t=0)")

    # recovery band (visual only)
    r_pool, p_pool = np.nan, np.nan
    med_r = np.nan
    if not is_fixed_obstacle:
        ax.axvspan(recovery_band[0], recovery_band[1], color='0.9', alpha=0.5,
                   label="Recovery zone (visual)")

        r_pool, p_pool = safe_pearsonr(pooled_t, pooled_a)
        med_r = np.nanmedian(
            [r for r in np.asarray(r_trials_time, float) if np.isfinite(r)]
        ) if len(r_trials_time) else np.nan

    def fmt_p(p):
        if not np.isfinite(p):
            return "p=NA"
        return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

    ax.set_xlabel("Time from start (ms)" if is_fixed_obstacle else "Time from event (ms)")
    ax.set_ylabel("Prior Confidence Index")
    ax.set_title(
        f"{monkey} – {experiment}\n"
        f"α vs time from {'trial start' if is_fixed_obstacle else 'event'}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    try:
        save_plot(fig, f"alpha_vs_time_FROM_{'START' if is_fixed_obstacle else 'EVENT'}_{monkey}_{experiment}",
                  subfolder="Entropy")
    except Exception:
        pass
    plt.show()

    # -------------- stats outputs --------------
    def wilcoxon_or_nan(x, alternative='two-sided'):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        return float(wilcoxon(x, alternative=alternative).pvalue)
    def arr_from_session_stats(key):
        vals = np.asarray([d.get(key, np.nan) for d in session_stats], float)
        return vals
    # Primary event-locked metrics
    delta_early = early_vals - pre_vals          # dip metric: should be negative if there is a dip
    delta_recovery = late_vals - early_vals      # recovery metric: should be positive if recovery occurs

    # Pseudo-event control: same metric structure as real event
    delta_pseudo = post_pseudo_vals - pre_pseudo_vals

    # True dip criterion: early drop followed by later recovery
    has_dip = (
        np.isfinite(pre_vals) &
        np.isfinite(early_vals) &
        np.isfinite(late_vals) &
        (early_vals < pre_vals) &
        (late_vals > early_vals)
    )

    # Optional stricter dip-depth metric:
    # only keep min-based depth if the trial actually shows a dip shape
    dip_depth_valid = np.full_like(dip_depths, np.nan, dtype=float)
    dip_depth_valid[has_dip] = dip_depths[has_dip]

    recovery_time_valid = np.full_like(recovery_times, np.nan, dtype=float)
    recovery_time_valid[has_dip] = recovery_times[has_dip]

    # Real-event stats
    # For a true dip, delta_early should be < 0
    p_delta_early = wilcoxon_or_nan(delta_early)
    p_delta_recovery = wilcoxon_or_nan(delta_recovery)

    # Pseudo-event stats
    p_delta_pseudo = wilcoxon_or_nan(delta_pseudo)

    # Real vs pseudo comparison:
    # more negative values mean stronger real dip than pseudo fluctuation
    real_minus_pseudo = delta_early - delta_pseudo
    p_real_vs_pseudo = wilcoxon_or_nan(real_minus_pseudo)

    # Optional min-based dip-depth stats, only on true-dip trials
    p_dip_depth_valid = wilcoxon_or_nan(dip_depth_valid)

    # ---------- session-level inferential stats ----------
    sess_delta_early = arr_from_session_stats('delta_early_median')
    sess_delta_recovery = arr_from_session_stats('delta_recovery_median')
    sess_delta_pseudo = arr_from_session_stats('delta_pseudo_median')
    sess_real_minus_pseudo = arr_from_session_stats('real_minus_pseudo_median')
    sess_dip_depth_valid = arr_from_session_stats('dip_depth_valid_median')
    sess_recovery_time = arr_from_session_stats('recovery_time_median_ms')
    sess_frac_dip = arr_from_session_stats('frac_dip_trials')

    p_sess_delta_early = wilcoxon_or_nan(sess_delta_early)
    p_sess_delta_recovery = wilcoxon_or_nan(sess_delta_recovery)
    p_sess_delta_pseudo = wilcoxon_or_nan(sess_delta_pseudo)
    p_sess_real_vs_pseudo = wilcoxon_or_nan(sess_real_minus_pseudo)
    p_sess_dip_depth_valid = wilcoxon_or_nan(sess_dip_depth_valid)

    # result['time'] = {
    #     'centers_ms': centers_t,   # renamed from centers_s because these are ms
    #     'mean': mean_t,
    #     'lo': lo_t,
    #     'hi': hi_t,
    #     'n_trials_per_bin': n_t,
    #     'r_pooled': r_pool,
    #     'r_trials': r_trials_time,
    #     't_window': (tmin, tmax),
    # }

    # result['event_windows'] = {
    #     'pre_mean': float(np.nanmean(pre_vals)),
    #     'early_mean': float(np.nanmean(early_vals)),
    #     'late_mean': float(np.nanmean(late_vals)),

    #     # primary dip metric
    #     'delta_early_mean': float(np.nanmean(delta_early)),
    #     'delta_early_median': float(np.nanmedian(delta_early)),
    #     'delta_early_p': p_delta_early,

    #     # recovery metric
    #     'delta_recovery_mean': float(np.nanmean(delta_recovery)),
    #     'delta_recovery_median': float(np.nanmedian(delta_recovery)),
    #     'delta_recovery_p': p_delta_recovery,

    #     'n_trials': int(np.sum(np.isfinite(pre_vals) & np.isfinite(early_vals) & np.isfinite(late_vals))),
    #     'n_dip_trials': int(np.sum(has_dip)),
    #     'frac_dip_trials': float(np.mean(has_dip)) if has_dip.size else np.nan,
    # }

    # result['dip_recovery'] = {
    #     # only meaningful for trials with an actual dip shape
    #     'dip_depth_valid_mean': float(np.nanmean(dip_depth_valid)),
    #     'dip_depth_valid_median': float(np.nanmedian(dip_depth_valid)),
    #     'dip_depth_valid_p': p_dip_depth_valid,

    #     'recovery_time_median_ms': float(np.nanmedian(recovery_time_valid)),
    #     'recovery_time_mean_ms': float(np.nanmean(recovery_time_valid)),
    #     'recovery_time_values_ms': recovery_time_valid,
    # }

    # result['pseudo_control'] = {
    #     'delta_pseudo_mean': float(np.nanmean(delta_pseudo)),
    #     'delta_pseudo_median': float(np.nanmedian(delta_pseudo)),
    #     'delta_pseudo_p': p_delta_pseudo,
    # }

    # result['real_vs_pseudo'] = {
    #     'diff_mean': float(np.nanmean(real_minus_pseudo)),
    #     'diff_median': float(np.nanmedian(real_minus_pseudo)),
    #     'diff_p': p_real_vs_pseudo,
    # }

    # result['intention'] = {
    #     'align_fracs': align_fracs_kept,
    #     'choice_labels': choice_labels_kept,
    #     'window_ms': align_window,
    # }

    result['session_level'] = {
            'n_sessions': len(session_stats),

            'delta_early_median_across_sessions': float(np.nanmedian(sess_delta_early)),
            'delta_early_values': sess_delta_early,
            'delta_early_p': p_sess_delta_early,

            'delta_recovery_median_across_sessions': float(np.nanmedian(sess_delta_recovery)),
            'delta_recovery_values': sess_delta_recovery,
            'delta_recovery_p': p_sess_delta_recovery,

            'delta_pseudo_median_across_sessions': float(np.nanmedian(sess_delta_pseudo)),
            'delta_pseudo_values': sess_delta_pseudo,
            'delta_pseudo_p': p_sess_delta_pseudo,

            'real_minus_pseudo_median_across_sessions': float(np.nanmedian(sess_real_minus_pseudo)),
            'real_minus_pseudo_values': sess_real_minus_pseudo,
            'real_vs_pseudo_p': p_sess_real_vs_pseudo,

            'dip_depth_valid_median_across_sessions': float(np.nanmedian(sess_dip_depth_valid)),
            'dip_depth_valid_values': sess_dip_depth_valid,
            'dip_depth_valid_p': p_sess_dip_depth_valid,

            'recovery_time_median_across_sessions_ms': float(np.nanmedian(sess_recovery_time)),
            'recovery_time_values_ms': sess_recovery_time,

            'frac_dip_trials_median_across_sessions': float(np.nanmedian(sess_frac_dip)),
            'frac_dip_trials_values': sess_frac_dip,
        }

    return result


def plot_alpha_vs_distance_average(
    monkey, experiment, base_dir,
    file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
    stride=4,                # thinning: keep every k-th sample
    min_dt=None,             # thinning: keep if >= this seconds since last kept
    min_dist=None,           # thinning: keep if moved >= this distance since last kept
    nbins=12,                # common bin count across trials
    min_alpha=0.2,           # floor after per-trial robust normalization
    invert_x=True,           # invert x-axis (near on right)
    only_answer=None,        # 1 to keep only correct, 0/!=1 for incorrect, None for all
    task = None,
    show_per_trial=False     # draw faint per-trial lines (can be busy)
):
    """
    Across-trials average of α (= 1 − normalized EntropyLb) vs distance to target/obstacle.
    - Per-trial: build α from EntropyLb with 5–95% robust normalization.
    - Bin α by distance using a *common* set of bin edges (from pooled distances).
    - Average the per-trial binned curves; show mean ± 95% CI.
    - Also show a dashed 'pooled' curve (binned over all samples together).

    Returns a dict with:
      - 'target': {'centers','mean','lo','hi','n_trials_per_bin','r_pooled','r_trials'}
      - 'obstacle': same keys (if obstacles exist)
    """
    import os, glob, pickle
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------- helpers ----------------
    from scipy.stats import pearsonr

    def safe_pearsonr(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan, np.nan
        try:
            r, p = pearsonr(x[m], y[m])
        except Exception:
            r, p = np.nan, np.nan
        return r, p

    def xz_from(pos, i=None):
        if pos is None:
            return None
        if i != None:
            if isinstance(pos, dict):
                return (pos["x"][i], pos["z"][i])
        else:
            if isinstance(pos, dict):
                return (float(pos[0]), float(pos[2]))
        try:
            return (float(pos[0]), float(pos[2]))
        except Exception:
            return None

    def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
        idx = list(range(len(keep_idx)))
        if stride and stride > 1:
            idx = idx[::stride]
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
        if min_dist is not None and len(idx) > 1:
            kept_rel = [idx[0]]
            last_p = pos[idx[0]]
            for j in idx[1:]:
                if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                    kept_rel.append(j); last_p = pos[j]
            idx = kept_rel
        return idx
    def build_bins_from_pooled(d_all, nb):
        d_all = np.asarray(d_all, float)
        d_all = d_all[np.isfinite(d_all)]
        if d_all.size < 5:
            return None, None
        lo, hi = np.percentile(d_all, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(d_all), np.nanmax(d_all)
        edges = np.linspace(lo, hi, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers

    def per_trial_binned_means(d, a, edges):
        d = np.asarray(d, float); a = np.asarray(a, float)
        good = np.isfinite(d) & np.isfinite(a)
        if good.sum() == 0:
            return np.full(len(edges) - 1, np.nan)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d >= edges[i]) & (d < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a[m])
        return out

    def aggregate_across_trials(per_trial_mat):
        per_trial_mat = np.asarray(per_trial_mat, float)
        mean = np.nanmean(per_trial_mat, axis=0)
        std = np.nanstd(per_trial_mat, axis=0, ddof=1)
        n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
        sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
        return mean, lo, hi, n

    def pooled_binned_means(d_all, a_all, edges):
        d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
        good = np.isfinite(d_all) & np.isfinite(a_all)
        out = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
            if np.any(m):
                out[i] = np.nanmean(a_all[m])
        return out

    # ---------------- collect per-trial data ----------------
    if file_path is None:
        data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
        pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    else:
        pkl_files = [file_path]

    trials_target = []
    trials_alpha_t = []
    trials_obst = []
    trials_alpha_o = []

    r_trials_target = []
    r_trials_obst = []

    pooled_d_t, pooled_a_t = [], []
    pooled_d_o, pooled_a_o = [], []

    closest_dt_at_obs = []          # distance to TARGET at closest approach to obstacle
    appear_dt_to_target = []        # distance to TARGET when obstacle appears

    closest_do_at_obs = []          # distance to OBSTACLE at closest approach (should be near the min)
    appear_do_to_obstacle = []      # distance to OBSTACLE when obstacle appears

    # AI control bands: distances from when AI turns ON to when it turns OFF
    ai_control_bands_dt = []  # list of (dist_to_target_on, dist_to_target_off)
    ai_control_bands_do = []  # list of (dist_to_obstacle_on, dist_to_obstacle_off)

    all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, pkl_files, ai_trials = load_files(
        experiment, monkey, base_dir=base_dir)

    for session in all_trials:
        for trial in session:
            if only_answer is not None:
                ans = trial.answer
                if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
                    continue
            
            if task == "AI Respawn":
                ans = trial.targetJumpTime
                if ans == None:
                    continue

            ai_recs = trial.aiVelocities
            if not ai_recs:
                continue

            # Skip if aiVelocity is zero at the start
            start_idx = next((i for i, rec in enumerate(ai_recs) if np.linalg.norm(rec['Output']) > 0), None)
            if start_idx is None:
                continue  # Skip trial if no aiVelocity above 0 is found
            ai_recs = ai_recs[start_idx:]

            # Keep samples with valid position
            xs, zs, keep_idx = [], [], []
            for i, rec in enumerate(ai_recs):
                p = xz_from(trial.avatarTrajectory, i)
                if p is None or np.isnan(p[0]) or np.isnan(p[1]):
                    continue
                xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
            if len(keep_idx) < 3:
                continue
            pos = np.column_stack([np.asarray(xs), np.asarray(zs)])

            # Entropy for ALL records, then keep
            ent_raw = []
            for rec in ai_recs:
                e = rec.get("EntropyLb")
                if isinstance(e, (list, tuple, np.ndarray)):
                    e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
                ent_raw.append(np.nan if e is None else float(e))
            ent_raw = np.asarray(ent_raw, float)
            ent_kept = ent_raw[keep_idx]

            # Thinning (relative indices)
            idx_rel = thin_indices(pos, keep_idx, ai_recs, stride=stride, min_dt=min_dt, min_dist=min_dist, use_output_ts=True)
            if len(idx_rel) < 3:
                continue
            pos = pos[idx_rel]
            ent_kept = ent_kept[idx_rel]

            # Alpha
            alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

            # Distances
            tgt = xz_from(trial.targetPosition)
            obs = xz_from(trial.obstaclePosition)
            d_tgt = np.linalg.norm(pos - np.array(tgt)[None, :], axis=1) if tgt is not None else np.full(len(pos), np.nan)
            d_obs = np.linalg.norm(pos - np.array(obs)[None, :], axis=1) if obs is not None else np.full(len(pos), np.nan)

            # ---------- AI CONTROL ON→OFF DISTANCES ----------
            try:
                # time vector from avatar trajectory
                traj_times = None
                if isinstance(trial.avatarTrajectory, dict):
                    traj_times = np.asarray(trial.avatarTrajectory.get("time", []), dtype=float)
                if traj_times is None or traj_times.size == 0:
                    raise ValueError

                def _to_array(val):
                    if val is None:
                        return np.array([], dtype=float)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.asarray(val, dtype=float)
                    else:
                        arr = np.array([float(val)], dtype=float)
                    return arr[np.isfinite(arr)]

                # Get raw ON/OFF times (can be scalar or list)
                v_on  = trial.aiControlOn
                v_off = trial.aiControlOff

                on_arr  = _to_array(v_on)
                off_arr = _to_array(v_off)

                n_pairs = min(on_arr.size, off_arr.size)
                if n_pairs > 0:
                    tgt_arr = np.array(tgt, float) if (tgt is not None and np.all(np.isfinite(tgt))) else None
                    obs_arr = np.array(obs, float) if (obs is not None and np.all(np.isfinite(obs))) else None

                    for k in range(n_pairs):
                        t_on  = on_arr[k]
                        t_off = off_arr[k]

                        # Find closest indices in trajectory time
                        idx_on  = int(np.nanargmin(np.abs(traj_times - t_on)))
                        idx_off = int(np.nanargmin(np.abs(traj_times - t_off)))

                        pos_on  = xz_from(trial.avatarTrajectory, idx_on)
                        pos_off = xz_from(trial.avatarTrajectory, idx_off)
                        if pos_on is None or pos_off is None:
                            continue

                        pos_on  = np.array(pos_on, float)
                        pos_off = np.array(pos_off, float)
                        if not (np.all(np.isfinite(pos_on)) and np.all(np.isfinite(pos_off))):
                            continue

                        # Distances to TARGET
                        if tgt_arr is not None:
                            dt_on  = float(np.linalg.norm(pos_on  - tgt_arr))
                            dt_off = float(np.linalg.norm(pos_off - tgt_arr))
                        else:
                            dt_on = dt_off = np.nan

                        # Distances to OBSTACLE
                        if obs_arr is not None:
                            do_on  = float(np.linalg.norm(pos_on  - obs_arr))
                            do_off = float(np.linalg.norm(pos_off - obs_arr))
                        else:
                            do_on = do_off = np.nan

                        # Store ON→OFF bands if both ends are finite
                        if np.isfinite(dt_on) and np.isfinite(dt_off):
                            ai_control_bands_dt.append((dt_on, dt_off))
                        if np.isfinite(do_on) and np.isfinite(do_off):
                            ai_control_bands_do.append((do_on, do_off))

            except Exception:
                # be robust; just skip AI-control bands if anything goes wrong
                pass

            # Store pooled
            if np.isfinite(d_tgt).any():
                pooled_d_t.append(d_tgt)
                pooled_a_t.append(alpha_vals)
                trials_target.append(d_tgt)
                trials_alpha_t.append(alpha_vals)
                m = np.isfinite(d_tgt) & np.isfinite(alpha_vals)
                r_trials_target.append(np.corrcoef(d_tgt[m], alpha_vals[m])[0, 1] if m.sum() >= 3 else np.nan)

            if np.isfinite(d_obs).any():
                pooled_d_o.append(d_obs)
                pooled_a_o.append(alpha_vals)
                trials_obst.append(d_obs)
                trials_alpha_o.append(alpha_vals)
                m = np.isfinite(d_obs) & np.isfinite(alpha_vals)
                r_trials_obst.append(np.corrcoef(d_obs[m], alpha_vals[m])[0, 1] if m.sum() >= 3 else np.nan)

            if np.isfinite(d_obs).any() and np.isfinite(d_tgt).any():
                g = np.isfinite(d_obs) & np.isfinite(d_tgt)
                if np.any(g):
                    # index of min obstacle distance among valid samples
                    rel = np.nanargmin(d_obs[g])
                    j = np.where(g)[0][rel]
                    closest_dt_at_obs.append(float(d_tgt[j]))   # distance to TARGET
                    closest_do_at_obs.append(float(d_obs[j]))   # distance to OBSTACLE
            
            # Distance-to-target at obstacle appearance time (only for specified tasks)
            if task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
                try:
                    t_app = getattr(trial, "obstacleAppearedTime", None) \
                            if not isinstance(trial, dict) else trial.get("obstacleAppearedTime", None)
                    if (t_app is not None) and isinstance(trial.avatarTrajectory, dict):
                        traj_t = np.asarray(trial.avatarTrajectory.get("time", []), dtype=float)
                        if traj_t.size >= 1:
                            idx_app = int(np.nanargmin(np.abs(traj_t - float(t_app))))
                            pos_app = xz_from(trial.avatarTrajectory, idx_app)  # (x,z) at appearance
                            if (pos_app is not None) and np.all(np.isfinite(pos_app)):
                                pos_app = np.array(pos_app, float)

                                # distance to TARGET (if available)
                                if tgt is not None and np.all(np.isfinite(tgt)):
                                    appear_dt_to_target.append(
                                        float(np.linalg.norm(pos_app - np.array(tgt, float)))
                                    )

                                # distance to OBSTACLE (if available)
                                if obs is not None and np.all(np.isfinite(obs)):
                                    appear_do_to_obstacle.append(
                                        float(np.linalg.norm(pos_app - np.array(obs, float)))
                                    )
                except Exception:
                    pass
    # Nothing collected?
    if len(trials_target) == 0 and len(trials_obst) == 0:
        print("[WARN] No valid samples across trials.")
        return {}

    # Concatenate pooled
    pooled_d_t = np.concatenate(pooled_d_t) if pooled_d_t else np.array([])
    pooled_a_t = np.concatenate(pooled_a_t) if pooled_a_t else np.array([])
    pooled_d_o = np.concatenate(pooled_d_o) if pooled_d_o else np.array([])
    pooled_a_o = np.concatenate(pooled_a_o) if pooled_a_o else np.array([])

    # Build common bins from pooled distances
    result = {}

    # ---------- TARGET ---------- 
    if pooled_d_t.size > 0:
        edges_t, centers_t = build_bins_from_pooled(pooled_d_t, nbins)
        # Per-trial binned curves
        per_trial_mat_t = []
        for d, a in zip(trials_target, trials_alpha_t):
            per_trial_mat_t.append(per_trial_binned_means(d, a, edges_t))
        per_trial_mat_t = np.asarray(per_trial_mat_t, float)
        mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)
        pooled_curve_t = pooled_binned_means(pooled_d_t, pooled_a_t, edges_t)

        # Plot
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        if show_per_trial:
            for row in per_trial_mat_t:
                ax.plot(centers_t, row, lw=0.8, alpha=0.2)
        ax.fill_between(centers_t, lo_t, hi_t, alpha=0.25, label="Across-trials 95% CI")
        ax.plot(centers_t, mean_t, lw=2.5, label="Across-trials mean")
        ax.plot(centers_t, pooled_curve_t, '--', lw=2, label="Pooled (all samples)")

        # # Add vertical line at respawn time
        # for trial in all_trials:
        #     ax.axvline(x=trial.targetRespawn, color='r', linestyle='--', label="Respawn Moment")
        # Store distances at respawn time for averaging
        respawn_distances = []
        if experiment == "AI Respawn":
            for session in all_trials:
                for trial in session:
                    if trial.targetJumpTime != None:
                        # Find the position at the respawn timestamp
                        respawn_idx = int(np.argmin(np.abs(np.asarray(trial.avatarTrajectory["time"], dtype=float) - float(trial.targetJumpTime))))
                        if respawn_idx is not None:
                            # Calculate the distance to the target at this timestamp
                            respawn_pos = xz_from(trial.avatarTrajectory, respawn_idx)
                            if respawn_pos:
                                respawn_distance = np.linalg.norm(np.array(respawn_pos) - np.array(tgt))
                                respawn_distances.append(respawn_distance)
            
        # Calculate average and std for respawn distance across trials
        avg_respawn_distance = np.mean(respawn_distances) if respawn_distances else np.nan
        std_respawn_distance = np.std(respawn_distances) if respawn_distances else np.nan

        # Build common bins from pooled distances (this is the part where the bins are built for the plot)
        edges_t, centers_t = build_bins_from_pooled(pooled_d_t, nbins)
        # Per-trial binned curves
        per_trial_mat_t = []
        for d, a in zip(trials_target, trials_alpha_t):
            per_trial_mat_t.append(per_trial_binned_means(d, a, edges_t))
        per_trial_mat_t = np.asarray(per_trial_mat_t, float)
        mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)
        pooled_curve_t = pooled_binned_means(pooled_d_t, pooled_a_t, edges_t)

        # Plot
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        if show_per_trial:
            for row in per_trial_mat_t:
                ax.plot(centers_t, row, lw=0.8, alpha=0.2)
        ax.fill_between(centers_t, lo_t, hi_t, alpha=0.25, label="Across-trials 95% CI")
        ax.plot(centers_t, mean_t, lw=2.5, label="Across-trials mean")
        ax.plot(centers_t, pooled_curve_t, '--', lw=2, label="Pooled (all samples)")

        # Add vertical line at the average respawn distance
        ax.axvline(x=avg_respawn_distance, color='r', linestyle='--', label=f"Respawn Moment (avg)")

        # Add shaded region for std deviation (optional)
        ax.fill_betweenx(
            [0, 1], avg_respawn_distance - std_respawn_distance, avg_respawn_distance + std_respawn_distance,
            color='r', alpha=0.1, label=f"Respawn ± std")
        # NEW: annotate closest-to-obstacle x (distance-to-target at closest approach)
        if len(closest_dt_at_obs) > 0:
            mu_close = float(np.mean(closest_dt_at_obs))
            sd_close = float(np.std(closest_dt_at_obs, ddof=1)) if len(closest_dt_at_obs) > 1 else 0.0
        else:
            mu_close, sd_close = np.nan, np.nan

        if np.isfinite(mu_close):
            ax.axvline(mu_close, color= "crimson", ls='-', lw=2, label="Closest to obstacle (mean)")
            if np.isfinite(sd_close) and sd_close > 0:
                ax.axvspan(mu_close - sd_close, mu_close + sd_close,
                           color='crimson', alpha = 0.15, label="Closest ±1 SD", zorder=2)
            # small text annotation
            y_top = ax.get_ylim()[1]
            ax.annotate(fr"closest μ={mu_close:.2f} ± {sd_close:.2f}",
                        xy=(mu_close, y_top), xytext=(5, -6),
                        textcoords='offset points', ha='left', va='top', color='crimson')
        
        # Mean ± SD of distance-to-target when obstacle appeared (only for specified tasks)
        if task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2") and len(appear_dt_to_target) > 0:
            mu_app_dt = float(np.mean(appear_dt_to_target))
            sd_app_dt = float(np.std(appear_dt_to_target, ddof=1)) if len(appear_dt_to_target) > 1 else 0.0

            ax.axvline(mu_app_dt, color="tab:purple", ls='-', lw=2, label="Obstacle appeared (mean)")
            if sd_app_dt > 0:
                ax.axvspan(mu_app_dt - sd_app_dt, mu_app_dt + sd_app_dt,
                        color="tab:purple", alpha=0.12, label="Appeared ±1 SD")

            # optional tiny annotation at top
            y_top = ax.get_ylim()[1]
            ax.annotate(fr"appear μ={mu_app_dt:.2f} ± {sd_app_dt:.2f}",
                        xy=(mu_app_dt, y_top), xytext=(5, -6),
                        textcoords='offset points', ha='left', va='top', color="tab:purple")
        
        # ---------- GREEN BANDS: AI CONTROL ON→OFF (TARGET DISTANCE) ----------
        # if len(ai_control_bands_dt) > 0:
        #     band_label_used = False
        #     for dt_on, dt_off in ai_control_bands_dt:
        #         if not (np.isfinite(dt_on) and np.isfinite(dt_off)):
        #             continue
        #         left  = min(dt_on, dt_off)
        #         right = max(dt_on, dt_off)
        #         ax.axvspan(
        #             left, right,
        #             color='green', alpha=0.10,
        #             label="AI control ON→OFF" if not band_label_used else None
        #         )
        #         band_label_used = True
        # Annotations
        mask_pool = np.isfinite(pooled_d_t) & np.isfinite(pooled_a_t)
        r_pool = np.corrcoef(pooled_d_t[mask_pool], pooled_a_t[mask_pool])[0, 1] if mask_pool.sum() >= 3 else np.nan
        med_r = np.nanmedian(np.asarray(r_trials_target, float)) if len(r_trials_target) else np.nan
        ax.text(0.02, 0.7, f"r_pooled={r_pool:.2f}\nmedian r_trial={med_r:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=10)

        ax.set_xlabel("Distance to target center")
        ax.set_ylabel("Prior Confidence Index")
        ax.set_title(f"{monkey} – {experiment}\nAcross-trials α vs. distance to target")
        ax.grid(True, alpha=0.3); ax.legend(frameon=True)
        ax.set_ylim(0, 1) 
        if invert_x: ax.invert_xaxis()
        plt.tight_layout()
        try:
            save_plot(fig, f"alpha_vs_distance_TARGET_across_{monkey}_{experiment}", subfolder="Entropy")
        except Exception:
            pass
        plt.show(); plt.close(fig)

        result['target'] = {
            'centers': centers_t, 'mean': mean_t, 'lo': lo_t, 'hi': hi_t, 'n_trials_per_bin': n_t,
            'r_pooled': r_pool, 'r_trials': r_trials_target,
            'closest_to_obstacle': {
                'mean_dt': mu_close, 'std_dt': sd_close,
                'n': len(closest_dt_at_obs),
                'values': closest_dt_at_obs
            },
            'respawn_dt_mean': avg_respawn_distance,
            'respawn_dt_std': std_respawn_distance,
            'obstacle_appeared_dt_mean': float(np.mean(appear_dt_to_target)) if len(appear_dt_to_target) else np.nan,
            'obstacle_appeared_dt_std':  float(np.std(appear_dt_to_target, ddof=1)) if len(appear_dt_to_target) > 1 else (0.0 if len(appear_dt_to_target)==1 else np.nan),
            'obstacle_appeared_dt_values': appear_dt_to_target,
        }
        
        # ---------- SIGNIFICANCE TEST OF PCI DIP ----------
        # We test whether alpha drops between obstacle appearance and closest approach.
        from scipy.stats import wilcoxon

        dip_vals = []  # per-trial Δα = α at appearance − minimum α until closest-approach

        for d_tgt, a_vals in zip(trials_target, trials_alpha_t):
            # We can only compute dip if both timestamps exist for this trial
            if not (len(appear_dt_to_target) and len(closest_dt_at_obs)):
                continue

            # distance arrays for this trial
            d = np.asarray(d_tgt)
            a = np.asarray(a_vals)

            # locate indices
            # (approx nearest distance to the mean appearance & mean closest)
            if len(appear_dt_to_target):
                app_d = float(np.mean(appear_dt_to_target))
                i_app = int(np.argmin(np.abs(d - app_d)))
            else:
                continue

            if len(closest_dt_at_obs):
                clo_d = float(np.mean(closest_dt_at_obs))
                i_clo = int(np.argmin(np.abs(d - clo_d)))
            else:
                continue

            if i_clo <= i_app or i_clo >= len(a):
                continue

            alpha_app = a[i_app]
            alpha_min = float(np.nanmin(a[i_app:i_clo+1]))
            dip_vals.append(alpha_app - alpha_min)

        if len(dip_vals) >= 5:
            dip_vals = np.asarray(dip_vals, float)
            mean_dip = np.nanmean(dip_vals)
            med_dip  = np.nanmedian(dip_vals)
            stat, p_dip = wilcoxon(dip_vals, alternative="greater")
        else:
            mean_dip = med_dip = p_dip = np.nan

        # ---------- SIGNIFICANCE TEST AROUND RESPAWN (AI Respawn only) ----------
        respawn_dip_vals = []
        mean_respawn_dip = med_respawn_dip = p_respawn_dip = np.nan

        # Only meaningful in AI Respawn and if we have respawn distances
        if (task == "AI Respawn" or experiment == "AI Respawn") and len(respawn_distances) > 0:
            # Use mean respawn distance as anchor on the distance axis
            respawn_d = float(np.mean(respawn_distances))

            for d_tgt, a_vals in zip(trials_target, trials_alpha_t):
                d = np.asarray(d_tgt, float)
                a = np.asarray(a_vals, float)
                good = np.isfinite(d) & np.isfinite(a)
                if good.sum() < 3:
                    continue
                d = d[good]
                a = a[good]

                # index of sample closest to respawn distance
                i_resp = int(np.argmin(np.abs(d - respawn_d)))
                if i_resp >= len(a) - 1:
                    continue  # need at least one point after respawn

                alpha_resp = a[i_resp]
                alpha_min_post = float(np.nanmin(a[i_resp+1:]))

                respawn_dip_vals.append(alpha_resp - alpha_min_post)

            if len(respawn_dip_vals) >= 5:
                respawn_dip_vals = np.asarray(respawn_dip_vals, float)
                mean_respawn_dip = float(np.nanmean(respawn_dip_vals))
                med_respawn_dip  = float(np.nanmedian(respawn_dip_vals))
                _, p_respawn_dip  = wilcoxon(respawn_dip_vals, alternative="greater")
            else:
                respawn_dip_vals = np.array([], float)
        result['respawn_dip_test'] = {
                "mean_dip": float(mean_respawn_dip) if np.isfinite(mean_respawn_dip) else np.nan,
                "median_dip": float(med_respawn_dip) if np.isfinite(med_respawn_dip) else np.nan,
                "p_dip": float(p_respawn_dip) if np.isfinite(p_respawn_dip) else np.nan,
                "dip_values": respawn_dip_vals.tolist() if isinstance(respawn_dip_vals, np.ndarray) else []
            }

        # ---------- OBSTACLE ----------
        if pooled_d_o.size > 0:
            edges_o, centers_o = build_bins_from_pooled(pooled_d_o, nbins)

            # per-trial binned curves
            per_trial_mat_o = [
                per_trial_binned_means(d, a, edges_o)
                for d, a in zip(trials_obst, trials_alpha_o)
            ]
            per_trial_mat_o = np.asarray(per_trial_mat_o, float)

            mean_o, lo_o, hi_o, n_o = aggregate_across_trials(per_trial_mat_o)
            pooled_curve_o = pooled_binned_means(pooled_d_o, pooled_a_o, edges_o)

            fig, ax = plt.subplots(figsize=(8.8, 5.0))
            if show_per_trial:
                for row in per_trial_mat_o:
                    ax.plot(centers_o, row, lw=0.8, alpha=0.2)

            ax.fill_between(centers_o, lo_o, hi_o, alpha=0.25, label="Across-trials 95% CI")
            ax.plot(centers_o, mean_o, lw=2.5, label="Across-trials mean")
            ax.plot(centers_o, pooled_curve_o, '--', lw=2, label="Pooled (all samples)")

             # ----- Mark closest-to-obstacle and obstacle-appearance in OBSTACLE units -----
            # # Closest approach to obstacle: use closest_do_at_obs (distance to OBSTACLE)
            # if len(closest_do_at_obs) > 0:
            #     mu_close_do = float(np.mean(closest_do_at_obs))
            #     sd_close_do = float(np.std(closest_do_at_obs, ddof=1)) if len(closest_do_at_obs) > 1 else 0.0
            # else:
            #     mu_close_do = np.nan
            #     sd_close_do = np.nan

            # if np.isfinite(mu_close_do):
            #     ax.axvline(mu_close_do, color='crimson', ls='--', lw=2, label="Closest to obstacle (mean)")
            #     if np.isfinite(sd_close_do) and sd_close_do > 0:
            #         ax.axvspan(mu_close_do - sd_close_do,
            #                    mu_close_do + sd_close_do,
            #                    color='crimson', alpha=0.12, label="Closest ±1 SD")
            #     y_top = ax.get_ylim()[1]
            #     ax.annotate(fr"closest μ={mu_close_do:.2f} ± {sd_close_do:.2f}",
            #                 xy=(mu_close_do, y_top), xytext=(5, -6),
            #                 textcoords='offset points', ha='left', va='top', color='crimson')

            # Obstacle appeared: use appear_do_to_obstacle (distance to OBSTACLE at appearance)
            if task in ("AI Appearing Obstacle", "AI Appearing Obstacle 2") and len(appear_do_to_obstacle) > 0:
                mu_app_do = float(np.mean(appear_do_to_obstacle))
                sd_app_do = float(np.std(appear_do_to_obstacle, ddof=1)) if len(appear_do_to_obstacle) > 1 else 0.0

                ax.axvline(mu_app_do, color="tab:purple", ls='-', lw=2, label="Obstacle appeared (mean)")
                if np.isfinite(sd_app_do) and sd_app_do > 0:
                    ax.axvspan(mu_app_do - sd_app_do,
                               mu_app_do + sd_app_do,
                               color="tab:purple", alpha=0.12, label="Appeared ±1 SD")

                y_top = ax.get_ylim()[1]
                ax.annotate(fr"appear μ={mu_app_do:.2f} ± {sd_app_do:.2f}",
                            xy=(mu_app_do, y_top), xytext=(5, -6),
                            textcoords='offset points', ha='left', va='top', color="tab:purple")

            # ---------- GREEN BANDS: AI CONTROL ON→OFF (OBSTACLE DISTANCE) ----------
            if len(ai_control_bands_do) > 0:
                starts = []
                ends   = []

                # Collect per-trial windows and (optionally) show faint per-trial bands
                band_label_used2 = False
                for do_on, do_off in ai_control_bands_do:
                    if not (np.isfinite(do_on) and np.isfinite(do_off)):
                        continue
                    left, right = sorted((do_on, do_off))
                    starts.append(left)
                    ends.append(right)

                    # # optional: very faint per-trial bands
                    # ax.axvspan(
                    #     left, right,
                    #     color='green', alpha=0.03,
                    #     label="AI control active (per-trial)" if not band_label_used2 else None,
                    #     zorder=1
                    # )
                    # band_label_used2 = True

                if starts:
                    starts = np.asarray(starts, float)
                    ends   = np.asarray(ends,   float)

                    mean_start = np.nanmean(starts)
                    mean_end   = np.nanmean(ends)
                    std_start  = np.nanstd(starts)
                    std_end    = np.nanstd(ends)

                    low  = mean_start - std_start
                    high = mean_end   + std_end

                    # 1) ONE filled band: mean window ± SD
                    ax.axvspan(
                        low, high,
                        color='green', alpha=0.15,
                        label="AI override window (mean ± 1 SD)",
                        zorder=3
                    )

                    # 2) Dashed lines at mean ON / OFF
                    ax.axvline(
                        mean_start, color='green',
                        linestyle='--', linewidth=1.2,
                        label="AI override window (mean)",
                        zorder=4
                    )
                    ax.axvline(
                        mean_end, color='green',
                        linestyle='--', linewidth=1.2,
                        zorder=4
                    )
                        


            # pooled Pearson r (and p)  — uses OBSTACLE pools
            r_pool, p_pool = safe_pearsonr(pooled_d_o, pooled_a_o)
            # median of per-trial r’s for OBSTACLE
            med_r = np.nanmedian(
                [r for r in np.asarray(r_trials_obst, float) if np.isfinite(r)]
            ) if len(r_trials_obst) else np.nan

            def fmt_p(p):
                if not np.isfinite(p):
                    return "p=NA"
                return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

            ax.text(
                0.02, 0.95,
                f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=10
            )

            ax.set_xlabel("Distance to obstacle center")
            ax.set_ylabel("Posterior Confidence Index")
            ax.set_title(f"{monkey} – {experiment}\nAcross-trials α vs. distance to obstacle")
            ax.grid(True, alpha=0.3); ax.legend(frameon=True)
            if invert_x:
                ax.invert_xaxis()
            plt.tight_layout()
            try:
                save_plot(fig, f"alpha_vs_distance_OBSTACLE_across_{monkey}_{experiment}", subfolder="Entropy")
            except Exception:
                pass
            plt.show(); plt.close(fig)

            result['obstacle'] = {
                'centers': centers_o,
                'mean': mean_o,
                'lo': lo_o,
                'hi': hi_o,
                'n_trials_per_bin': n_o,
                'r_pooled': r_pool,
                'r_trials': r_trials_obst,
            }

            result['dip_test'] = {
                "mean_dip": float(mean_dip),
                "median_dip": float(med_dip),
                "p_dip": float(p_dip) if np.isfinite(p_dip) else np.nan,
                "dip_values": dip_vals.tolist() if isinstance(dip_vals, np.ndarray) else []
            }
            
    return result

# def plot_alpha_vs_distance_average(
#     monkey, experiment, base_dir,
#     file_path=None,          # optional: a single .pkl; otherwise glob all in aiLog/
#     stride=5,                # thinning: keep every k-th sample
#     min_dt=None,             # thinning: keep if >= this seconds since last kept
#     min_dist=None,           # thinning: keep if moved >= this distance since last kept
#     nbins=12,                # common bin count across trials
#     min_alpha=0.2,           # floor after per-trial robust normalization
#     invert_x=True,           # invert x-axis (near on right)
#     only_answer=None,        # 1 to keep only correct, 0/!=1 for incorrect, None for all
#     show_per_trial=False     # draw faint per-trial lines (can be busy)
# ):
#     """
#     Across-trials average of α (= 1 − normalized EntropyLb) vs distance to target/obstacle.
#     - Per-trial: build α from EntropyLb with 5–95% robust normalization.
#     - Bin α by distance using a *common* set of bin edges (from pooled distances).
#     - Average the per-trial binned curves; show mean ± 95% CI.
#     - Also show a dashed 'pooled' curve (binned over all samples together).

#     Returns a dict with:
#       - 'target': {'centers','mean','lo','hi','n_trials_per_bin','r_pooled','r_trials'}
#       - 'obstacle': same keys (if obstacles exist)
#     """
#     import os, glob, pickle
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # ---------------- helpers ----------------
#     from scipy.stats import pearsonr
#     import numpy as np

#     def safe_pearsonr(x, y):
#         x = np.asarray(x, float); y = np.asarray(y, float)
#         m = np.isfinite(x) & np.isfinite(y)
#         if m.sum() < 3:
#             return np.nan, np.nan
#         try:
#             r, p = pearsonr(x[m], y[m])
#         except Exception:
#             r, p = np.nan, np.nan
#         return r, p
#     def xz_from(pos, i=None):
#         if pos is None:
#             return None
#         if i != None:
#             if isinstance(pos, dict):
#                 return (pos["x"][i], pos["z"][i])
#         else:
#             if isinstance(pos, dict):
#                 return (float(pos[0]), float(pos[2]))
#         try:
#             return (float(pos[0]), float(pos[2]))
#         except Exception:
#             return None

#     def thin_indices(pos, keep_idx, ai_recs, stride=None, min_dt=None, min_dist=None, use_output_ts=True):
#         # returns indices RELATIVE to the kept order
#         idx = list(range(len(keep_idx)))
#         if stride and stride > 1:
#             idx = idx[::stride]
#         if min_dt is not None:
#             ts_key = "OutputTimestamp" if use_output_ts else "InputTimestamp"
#             ts_vals = []
#             for k in keep_idx:
#                 t = ai_recs[k].get(ts_key)
#                 if hasattr(t, "timestamp"):
#                     t = t.timestamp()
#                 ts_vals.append(np.nan if t is None else float(t))
#             kept_rel, last_t = [], None
#             for j in idx:
#                 tj = ts_vals[j]
#                 if np.isnan(tj) or last_t is None or (tj - last_t) >= float(min_dt):
#                     kept_rel.append(j); last_t = tj
#             idx = kept_rel
#         if min_dist is not None and len(idx) > 1:
#             kept_rel = [idx[0]]
#             last_p = pos[idx[0]]
#             for j in idx[1:]:
#                 if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
#                     kept_rel.append(j); last_p = pos[j]
#             idx = kept_rel
#         return idx

#     def robust_alpha(ent_vals, floor=min_alpha):
#         ent_vals = np.asarray(ent_vals, float)
#         if ent_vals.size == 0 or np.all(np.isnan(ent_vals)):
#             return np.ones_like(ent_vals)
#         lo, hi = np.nanpercentile(ent_vals, [5, 95])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = np.nanmin(ent_vals), np.nanmax(ent_vals)
#             if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#                 return np.ones_like(ent_vals)
#         a = 1.0 - (ent_vals - lo) / (hi - lo)
#         return np.clip(np.nan_to_num(a, nan=1.0), float(floor), 1.0)

#     def build_bins_from_pooled(d_all, nb):
#         d_all = np.asarray(d_all, float)
#         d_all = d_all[np.isfinite(d_all)]
#         if d_all.size < 5:
#             return None, None
#         lo, hi = np.percentile(d_all, [1, 99])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = np.nanmin(d_all), np.nanmax(d_all)
#         edges = np.linspace(lo, hi, nb + 1)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#         return edges, centers

#     def per_trial_binned_means(d, a, edges):
#         # returns vector length nbins with per-bin mean alpha (NaN if empty)
#         d = np.asarray(d, float); a = np.asarray(a, float)
#         good = np.isfinite(d) & np.isfinite(a)
#         if good.sum() == 0:
#             return np.full(len(edges) - 1, np.nan)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d >= edges[i]) & (d < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a[m])
#         return out

#     def aggregate_across_trials(per_trial_mat):
#         # mean ± 95% CI across rows (trials), ignoring NaNs per bin
#         per_trial_mat = np.asarray(per_trial_mat, float)  # shape: T x B
#         mean = np.nanmean(per_trial_mat, axis=0)
#         std = np.nanstd(per_trial_mat, axis=0, ddof=1)
#         n = np.sum(np.isfinite(per_trial_mat), axis=0).astype(float)
#         sem = np.where(n > 0, std / np.sqrt(np.maximum(n, 1.0)), np.nan)
#         lo = mean - 1.96 * sem
#         hi = mean + 1.96 * sem
#         return mean, lo, hi, n

#     def pooled_binned_means(d_all, a_all, edges):
#         d_all = np.asarray(d_all, float); a_all = np.asarray(a_all, float)
#         good = np.isfinite(d_all) & np.isfinite(a_all)
#         out = np.full(len(edges) - 1, np.nan)
#         for i in range(len(edges) - 1):
#             m = (d_all >= edges[i]) & (d_all < edges[i+1]) & good
#             if np.any(m):
#                 out[i] = np.nanmean(a_all[m])
#         return out

#     # ---------------- collect per-trial data ----------------
#     if file_path is None:
#         data_dir = os.path.join(base_dir, monkey, experiment, "aiLog")
#         pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#     else:
#         pkl_files = [file_path]

#     trials_target = []   # list of arrays (per trial) distances to target
#     trials_alpha_t = []  # list of arrays (per trial) alpha aligned to target dists
#     trials_obst = []     # same for obstacle
#     trials_alpha_o = []

#     r_trials_target = []  # per-trial r for target
#     r_trials_obst = []    # per-trial r for obstacle

#     pooled_d_t, pooled_a_t = [], []
#     pooled_d_o, pooled_a_o = [], []


#     all_trials, all_correct,all_incorrect, all_training, all_channels, nb_channels, pkl_files, ai_trials = load_files(
#                 experiment, monkey, base_dir=base_dir)

#     for session in all_trials:
#         for trial in session:
#             if only_answer is not None:
#                 ans = trial.answer
#                 if (only_answer == 1 and ans != 1) or (only_answer != 1 and ans == 1):
#                     continue

#             ai_recs = trial.aiVelocities
#             if not ai_recs:
#                 continue

#             # keep samples with valid position
#             xs, zs, keep_idx = [], [], []
#             for i, rec in enumerate(ai_recs):
#                 p = xz_from(trial.avatarTrajectory, i)
#                 if p is None or np.isnan(p[0]) or np.isnan(p[1]):
#                     continue
#                 xs.append(p[0]); zs.append(p[1]); keep_idx.append(i)
#             if len(keep_idx) < 3:
#                 continue
#             pos = np.column_stack([np.asarray(xs), np.asarray(zs)])

#             # entropy for ALL records, then keep
#             ent_raw = []
#             for rec in ai_recs:
#                 e = rec.get("EntropyLb")
#                 if isinstance(e, (list, tuple, np.ndarray)):
#                     e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
#                 ent_raw.append(np.nan if e is None else float(e))
#             ent_raw = np.asarray(ent_raw, float)
#             ent_kept = ent_raw[keep_idx]

#             # thinning (relative indices)
#             idx_rel = thin_indices(pos, keep_idx, ai_recs,
#                                    stride=stride, min_dt=min_dt, min_dist=min_dist, use_output_ts=True)
#             if len(idx_rel) < 3:
#                 continue
#             pos = pos[idx_rel]
#             ent_kept = ent_kept[idx_rel]

#             # alpha
            # alpha_vals = robust_alpha(ent_kept, floor=min_alpha)

#             # distances
#             tgt = xz_from(trial.targetPosition)
#             obs = xz_from(trial.obstaclePosition)
#             d_tgt = np.linalg.norm(pos - np.array(tgt)[None, :], axis=1) if tgt is not None else np.full(len(pos), np.nan)
#             d_obs = np.linalg.norm(pos - np.array(obs)[None, :], axis=1) if obs is not None else np.full(len(pos), np.nan)

#             # store pooled
#             if np.isfinite(d_tgt).any():
#                 pooled_d_t.append(d_tgt); pooled_a_t.append(alpha_vals)
#                 # per-trial vectors
#                 trials_target.append(d_tgt); trials_alpha_t.append(alpha_vals)
#                 # per-trial r
#                 m = np.isfinite(d_tgt) & np.isfinite(alpha_vals)
#                 r_trials_target.append(np.corrcoef(d_tgt[m], alpha_vals[m])[0,1] if m.sum() >= 3 else np.nan)

#             if np.isfinite(d_obs).any():
#                 pooled_d_o.append(d_obs); pooled_a_o.append(alpha_vals)
#                 trials_obst.append(d_obs); trials_alpha_o.append(alpha_vals)
#                 m = np.isfinite(d_obs) & np.isfinite(alpha_vals)
#                 r_trials_obst.append(np.corrcoef(d_obs[m], alpha_vals[m])[0,1] if m.sum() >= 3 else np.nan)

#     # nothing collected?
#     if len(trials_target) == 0 and len(trials_obst) == 0:
#         print("[WARN] No valid samples across trials.")
#         return {}

#     # concat pooled
#     pooled_d_t = np.concatenate(pooled_d_t) if pooled_d_t else np.array([])
#     pooled_a_t = np.concatenate(pooled_a_t) if pooled_a_t else np.array([])
#     pooled_d_o = np.concatenate(pooled_d_o) if pooled_d_o else np.array([])
#     pooled_a_o = np.concatenate(pooled_a_o) if pooled_a_o else np.array([])

#     # build common bins from pooled distances
#     result = {}

#     # ---------- TARGET ----------
#     if pooled_d_t.size > 0:
#         edges_t, centers_t = build_bins_from_pooled(pooled_d_t, nbins)
#         # per-trial binned curves
#         per_trial_mat_t = []
#         for d, a in zip(trials_target, trials_alpha_t):
#             per_trial_mat_t.append(per_trial_binned_means(d, a, edges_t))
#         per_trial_mat_t = np.asarray(per_trial_mat_t, float)   # T x B
#         mean_t, lo_t, hi_t, n_t = aggregate_across_trials(per_trial_mat_t)
#         pooled_curve_t = pooled_binned_means(pooled_d_t, pooled_a_t, edges_t)

#         # plot
#         fig, ax = plt.subplots(figsize=(8.8, 5.0))
#         if show_per_trial:
#             for row in per_trial_mat_t:
#                 ax.plot(centers_t, row, lw=0.8, alpha=0.2)
#         ax.fill_between(centers_t, lo_t, hi_t, alpha=0.25, label="Across-trials 95% CI")
#         ax.plot(centers_t, mean_t, lw=2.5, label="Across-trials mean")
#         ax.plot(centers_t, pooled_curve_t, '--', lw=2, label="Pooled (all samples)")

#         # annotations
#         mask_pool = np.isfinite(pooled_d_t) & np.isfinite(pooled_a_t)
#         r_pool = np.corrcoef(pooled_d_t[mask_pool], pooled_a_t[mask_pool])[0,1] if mask_pool.sum() >= 3 else np.nan
#         med_r = np.nanmedian(np.asarray(r_trials_target, float)) if len(r_trials_target) else np.nan
#         ax.text(0.02, 0.7, f"r_pooled={r_pool:.2f}\nmedian r_trial={med_r:.2f}",
#                 transform=ax.transAxes, ha='left', va='top', fontsize=10)

#         ax.set_xlabel("Distance to target center")
#         ax.set_ylabel("α = 1 − normalized EntropyLb")
#         ax.set_title(f"{monkey} – {experiment}\nAcross-trials α vs. distance to target")
#         ax.grid(True, alpha=0.3); ax.legend(frameon=True)
#         if invert_x: ax.invert_xaxis()
#         plt.tight_layout()
#         try:
#             save_plot(fig, f"alpha_vs_distance_TARGET_across_{monkey}_{experiment}", subfolder="Entropy")
#         except Exception:
#             pass
#         plt.show(); plt.close(fig)

#         result['target'] = {
#             'centers': centers_t, 'mean': mean_t, 'lo': lo_t, 'hi': hi_t, 'n_trials_per_bin': n_t,
#             'r_pooled': r_pool, 'r_trials': r_trials_target
#         }

#     # ---------- OBSTACLE ----------
#     if pooled_d_o.size > 0:
#         edges_o, centers_o = build_bins_from_pooled(pooled_d_o, nbins)
#         per_trial_mat_o = []
#         for d, a in zip(trials_obst, trials_alpha_o):
#             per_trial_mat_o.append(per_trial_binned_means(d, a, edges_o))
#         per_trial_mat_o = np.asarray(per_trial_mat_o, float)
#         mean_o, lo_o, hi_o, n_o = aggregate_across_trials(per_trial_mat_o)
#         pooled_curve_o = pooled_binned_means(pooled_d_o, pooled_a_o, edges_o)

#         fig, ax = plt.subplots(figsize=(8.8, 5.0))
#         if show_per_trial:
#             for row in per_trial_mat_o:
#                 ax.plot(centers_o, row, lw=0.8, alpha=0.2)
#         ax.fill_between(centers_o, lo_o, hi_o, alpha=0.25, label="Across-trials 95% CI")
#         ax.plot(centers_o, mean_o, lw=2.5, label="Across-trials mean")
#         ax.plot(centers_o, pooled_curve_o, '--', lw=2, label="Pooled (all samples)")

#         mask_pool = np.isfinite(pooled_d_o) & np.isfinite(pooled_a_o)
#         # r_pool = np.corrcoef(pooled_d_o[mask_pool], pooled_a_o[mask_pool])[0,1] if mask_pool.sum() >= 3 else np.nan
#         # med_r = np.nanmedian(np.asarray(r_trials_obst, float)) if len(r_trials_obst) else np.nan
#         # pooled Pearson r (and p)
#         r_pool, p_pool = safe_pearsonr(pooled_d_t, pooled_a_t)
#         # median of per-trial r’s (keep your existing r_trials_target list if you already compute it)
#         med_r = np.nanmedian([r for r in np.asarray(r_trials_target, float) if np.isfinite(r)]) if len(r_trials_target) else np.nan

#         # pretty p-value formatter
#         def fmt_p(p):
#             import numpy as np
#             if not np.isfinite(p):
#                 return "p=NA"
#             return f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}"

#         ax.text(
#             0.02, 0.95,
#             f"r_pooled={r_pool:.2f} ({fmt_p(p_pool)})\nmedian r_trial={med_r:.2f}",
#             transform=ax.transAxes, ha='left', va='top', fontsize=10
#         )

#         ax.set_xlabel("Distance to obstacle center")
#         ax.set_ylabel("α = 1 − normalized EntropyLb")
#         ax.set_title(f"{monkey} – {experiment}\nAcross-trials α vs. distance to obstacle")
#         ax.grid(True, alpha=0.3); ax.legend(frameon=True)
#         if invert_x: ax.invert_xaxis()
#         plt.tight_layout()
#         try:
#             save_plot(fig, f"alpha_vs_distance_OBSTACLE_across_{monkey}_{experiment}", subfolder="Entropy")
#         except Exception:
#             pass
#         plt.show(); plt.close(fig)

#         result['obstacle'] = {
#             'centers': centers_o, 'mean': mean_o, 'lo': lo_o, 'hi': hi_o, 'n_trials_per_bin': n_o,
#             'r_pooled': r_pool, 'r_trials': r_trials_obst
#         }

#     return result
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_summary_scatter(all_points, base_dir, title="Performance-dependent AI gain"):
    """
    all_points: list of dicts with keys:
      - 'monkey', 'experiment', 'target', 'baseline_off_pct', 'gain_pp'
    """
    allowed_experiments = {
        "AI Obstacle",
        "AI Appearing Obstacle",
        "AI Appearing Obstacle 2",
    }

    # Keep only selected experiments
    all_points = [p for p in all_points if p["experiment"] in allowed_experiments]

    if not all_points:
        print("[plot_summary_scatter] No data points to plot.")
        return

    # Map monkeys to colors, experiments to markers
    monkeys = sorted({p["monkey"] for p in all_points})
    exps = sorted({p["experiment"] for p in all_points})
    color_map = {m: c for m, c in zip(monkeys, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])}
    marker_map = {e: m for e, m in zip(exps, ["o", "s", "^", "D", "P", "X"])}

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    # Plot points grouped (for nice legends)
    for m in monkeys:
        for e in exps:
            pts = [p for p in all_points if p["monkey"] == m and p["experiment"] == e]
            if not pts:
                continue
            x = [p["baseline_off_pct"] for p in pts]
            y = [p["gain_pp"] for p in pts]
            ax.scatter(
                x, y,
                s=70, alpha=0.85, linewidths=0.7, edgecolors="k",
                c=color_map[m], marker=marker_map[e],
                label=f"{m} – {e}"
            )

    # Quadratic fit across all points
    X = np.array([p["baseline_off_pct"] for p in all_points], float)
    Y = np.array([p["gain_pp"] for p in all_points], float)

    # Fit y = a x^2 + b x + c
    coefs = np.polyfit(X, Y, deg=2)
    a, b, c = coefs
    xs = np.linspace(max(0, np.nanmin(X) - 5), min(100, np.nanmax(X) + 5), 200)
    ys = a * xs**2 + b * xs + c
    ax.plot(xs, ys, color="0.25", lw=2.0, label="Quadratic fit")

    # R^2 for the quadratic fit
    yhat = a * X**2 + b * X + c
    ss_res = np.sum((Y - yhat)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # OLS regression for significance testing
    X_design = np.column_stack([X**2, X])
    X_design = sm.add_constant(X_design)   # const, x^2, x
    model = sm.OLS(Y, X_design).fit()

    p_model = model.f_pvalue     # overall regression p-value
    p_x2 = model.pvalues[1]      # quadratic term
    p_x = model.pvalues[2]       # linear term

    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlabel("No-AI success per target (%)")
    ax.set_ylabel("AI gain per target (pp)")
    ax.set_title(
        f"{title}\n"
        f"Quadratic fit: y = {a:.3f}x² + {b:.3f}x + {c:.1f} "
        f"(R² = {r2:.2f}, model p = {p_model:.3g})"
    )

    # Build clean legends (monkeys = colors, experiments = markers)
    color_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[m], markeredgecolor="k",
                   markersize=8, label=m)
        for m in monkeys
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=marker_map[e], color='k',
                   linestyle='None', markersize=8, label=e)
        for e in exps
    ]
    fit_handle = [plt.Line2D([0], [0], color="0.25", lw=2, label="Quadratic fit")]

    leg1 = ax.legend(handles=color_handles, title="Monkey", frameon=False, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles + fit_handle, title="Task", frameon=False, loc="upper right")

    fig.tight_layout()
    save_plot(fig, "Performance-dependent AI gains", "AI gain with baseline")
    plt.show()
    plt.close(fig)

    print(model.summary())
    print(f"Overall model p-value: {p_model:.6g}")
    print(f"Quadratic term p-value: {p_x2:.6g}")
    print(f"Linear term p-value: {p_x:.6g}")

def plot_gain_with_labels(per_target_gains, acc_ai_off, monkey, experiment,
                          label_fmt="{:.0f}%", label_pos="auto"):
    """
    Bars: per-target AI gain (AI ON − AI OFF) in percentage points (pp).
    Text labels: No-AI (AI OFF) baseline success per target, overlaid on each bar.

    per_target_gains : dict[target] -> list of per-session gains (proportions in [0,1])
    acc_ai_off       : dict[target] -> list of per-session accuracies in [0,1] OR a single float
    label_fmt        : format for No-AI label (default '{:.0f}%')
    label_pos        : 'auto' (inside for tall bars / above for short), 'inside', or 'above'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import t

    def ci95_t(vals):
        vals = np.asarray([v for v in vals if np.isfinite(v)], float)
        if vals.size == 0: return np.nan, (np.nan, np.nan)
        m = vals.mean()
        if vals.size == 1: return m, (m, m)
        se = vals.std(ddof=1) / np.sqrt(vals.size)
        h  = t.ppf(0.975, df=vals.size-1) * se
        return m, (m-h, m+h)

    preferred = ["left","slight_left","straight","slight_right","right"]
    order = [t for t in preferred if t in per_target_gains] or sorted(per_target_gains.keys())
    x = np.arange(len(order))

    # --- AI gain (pp) + CI ---
    gain_mean_pp, gain_lo_pp, gain_hi_pp, ns = [], [], [], []
    for tgt in order:
        m,(lo,hi) = ci95_t(per_target_gains.get(tgt, []))
        gain_mean_pp.append(m*100.0)
        gain_lo_pp.append(max(0, (m-lo)*100.0) if np.isfinite(lo) else 0.0)
        gain_hi_pp.append(max(0, (hi-m)*100.0) if np.isfinite(hi) else 0.0)
        ns.append(np.isfinite(per_target_gains.get(tgt, [])).sum()
                  if hasattr(per_target_gains.get(tgt, []), "__len__") else 0)

    # --- No-AI baseline (mean %) for labels ---
    off_mean_pct = []
    for tgt in order:
        v = acc_ai_off.get(tgt, np.nan)
        if isinstance(v, (list, tuple, np.ndarray)):
            m,_ = ci95_t(v)
        else:
            m = float(v)
        off_mean_pct.append(100.0*m if np.isfinite(m) else np.nan)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(x, gain_mean_pp, yerr=[gain_lo_pp, gain_hi_pp], capsize=3, width=0.65,
                  color="#4C78A8", edgecolor="none", alpha=0.95, label="AI gain (pp)")
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=12, ha="right")
    ax.set_ylabel("AI gain (pp)")
    ax.set_title(f"{monkey} – {experiment}: per-target AI gain (bars) with No-AI baseline labels")

    # ---- SAFE y-limits that include the 95% CI whiskers ----
    import numpy as np

    means = np.asarray(gain_mean_pp, dtype=float)
    loerr = np.asarray(gain_lo_pp,   dtype=float)  # amount below mean
    hierr = np.asarray(gain_hi_pp,   dtype=float)  # amount above mean

    lower = means - loerr
    upper = means + hierr

    finite_lower = lower[np.isfinite(lower)]
    finite_upper = upper[np.isfinite(upper)]

    if finite_lower.size and finite_upper.size:
        y_min_ci = float(finite_lower.min())
        y_max_ci = float(finite_upper.max())
        y_range  = max(1.0, y_max_ci - y_min_ci)
        pad = max(4.0, 0.08 * y_range)            # small proportional padding
        ymin = min(0.0, y_min_ci - pad)           # include zero if below
        ymax = max(10.0, y_max_ci + pad)
        ax.set_ylim(ymin, ymax)

    # Overlay labels: 'AI off = XX%'
    for xi, bar, base in zip(x, bars, off_mean_pct):
        if not np.isfinite(base): continue
        txt = f"AI off = {label_fmt.format(base)}"
        height = bar.get_height()
        # position logic
        if label_pos == "inside" or (label_pos == "auto" and height >= 18):
            # inside near top
            y = height - 0.08*(ax.get_ylim()[1]-ax.get_ylim()[0])
            va, color = "top", "0.30"
        elif label_pos == "above" or (label_pos == "auto" and height < 18):
            # just above bar
            y = height + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0])
            va, color = "bottom", "0.30"
        ax.text(bar.get_x() + bar.get_width()/2, y, txt,
                ha="center", va=va, fontsize=8, color=color)

    # (optional) show n per target below ticks (keeps bars clean)
    ymin, ymax = ax.get_ylim()
    for xi, n in zip(x, ns):
        ax.text(xi, ymin + 0.035*(ymax-ymin), f"n={int(n)}",
                ha="center", va="bottom", fontsize=8, color="0.4")

    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_plot(fig, f"AI_gain_labels_{monkey}_{experiment}", "AI gain with baseline")  # if you have this helper
    plt.show(); plt.close(fig)

def plot_gain_with_ci(per_target_gains, acc_ai_off, monkey, experiment,
                      chance_level=None, show_n=True):
    """
    Bars: AI gain (AI ON − AI OFF) in percentage points (left axis).
    Markers: No-AI (AI OFF) success per target in % on a *separate, offset* right axis.
    acc_ai_off can be dict[target] -> list of per-session accuracies in [0,1] OR a float.
    """
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import t

    def ci95_t(a):
        a = np.asarray([v for v in a if np.isfinite(v)], float)
        if a.size == 0: return np.nan, (np.nan, np.nan)
        m = a.mean()
        if a.size == 1: return m, (m, m)
        se = a.std(ddof=1) / np.sqrt(a.size)
        h  = t.ppf(0.975, a.size-1) * se
        return m, (m-h, m+h)

    preferred = ["left","slight_left","straight","slight_right","right"]
    order = [t for t in preferred if t in per_target_gains] or sorted(per_target_gains)

    # --- left axis: AI gain (pp)
    means_pp, lo_pp, hi_pp, ns = [], [], [], []
    for tgt in order:
        m,(lo,hi) = ci95_t(per_target_gains.get(tgt, []))
        means_pp.append(m*100)
        lo_pp.append(max(0, (m-lo)*100) if np.isfinite(lo) else 0)
        hi_pp.append(max(0, (hi-m)*100) if np.isfinite(hi) else 0)
        ns.append(np.isfinite(per_target_gains.get(tgt, [])).sum() if hasattr(per_target_gains.get(tgt, []), "__len__") else 0)

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(x-0.15, means_pp, width=0.6, yerr=[lo_pp,hi_pp], capsize=3,
                  color="#4C78A8", alpha=0.9, edgecolor="none")
    ax.axhline(0, color="0.25", lw=1)
    ax.set_ylabel("AI gain (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=12, ha="right")

    # keep the range tight
    pad = max(6, 0.1*max(10, np.nanmax(np.abs(means_pp))))
    ax.set_ylim(min(0, np.nanmin(means_pp)-pad), np.nanmax(means_pp)+pad)

    if show_n:
        ymin, ymax = ax.get_ylim()
        for xi, n in zip(x, ns):
            ax.text(xi-0.15, ymin + 0.04*(ymax-ymin), f"n={int(n)}",
                    ha="center", va="bottom", fontsize=8, color="0.35")

    # --- right axis: No-AI baseline (%), offset markers so they don't overlap bars
    def mean_ci_pct(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            m,(lo,hi) = ci95_t(v)
        else:
            m,(lo,hi) = float(v), (float(v), float(v))
        return 100*m, 100*lo, 100*hi

    off_mean, off_lo, off_hi = [], [], []
    for tgt in order:
        m, lo, hi = mean_ci_pct(acc_ai_off.get(tgt, np.nan))
        off_mean.append(m); off_lo.append(lo); off_hi.append(hi)

    ax2 = ax.twinx()
    # auto-scale to band with padding
    valid = np.array([m for m in off_mean if np.isfinite(m)])
    if valid.size:
        rng = valid.max() - valid.min()
        pad_r = max(5, 0.15*(rng if rng > 0 else 20))
        ax2.set_ylim(max(0, valid.min()-pad_r), min(100, valid.max()+pad_r))
    else:
        ax2.set_ylim(0, 100)
    lo_err = [max(0, m-l) if np.isfinite(l) else 0 for m,l in zip(off_mean, off_lo)]
    hi_err = [max(0, h-m) if np.isfinite(h) else 0 for m,h in zip(off_mean, off_hi)]

    ax2.errorbar(x+0.25, off_mean, yerr=[lo_err,hi_err], fmt="o-",
                 color="#E45756", lw=1.7, ms=4, capsize=3, alpha=0.9)
    ax2.set_ylabel("No-AI success (%)")

    # optional chance line (right axis)
    if chance_level is not None:
        if isinstance(chance_level, dict):
            for xi,tgt in zip(x, order):
                if tgt in chance_level:
                    ax2.hlines(100*chance_level[tgt], xi-0.45, xi+0.45,
                               colors="0.7", linestyles="dashed", lw=1)
        else:
            ax2.axhline(100*float(chance_level), color="0.7", ls="dashed", lw=1)

    # legend
    ax.legend(handles=[
        Line2D([0],[0], lw=8, color="#4C78A8", label="AI gain (pp)"),
        Line2D([0],[0], marker="o", lw=1.7, color="#E45756", label="No-AI success (%)"),
    ], loc="upper left", frameon=False)

    ax.set_title(f"{monkey} – {experiment}: per-target AI gain (bars) and No-AI success (offset line)")
    fig.tight_layout()
    save_plot(fig, f"AI_gain_baseline_{monkey}_{experiment}", "AI gain with baseline")
    plt.show()
    plt.close(fig)

def save_plot(fig, plot_name, subfolder):
    """Save a matplotlib figure in the figures/<subfolder>/ directory as SVG."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
    subfolder_path = os.path.join(base_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    fig_path = os.path.join(subfolder_path, f"{plot_name}.svg")
    fig.savefig(fig_path, format='svg', bbox_inches='tight')
    print(f"Saved: {fig_path}")

def _nan_to_list(x):
    """Convert NaN/None/scalar/list/array to Python list."""
    if x is None: return []
    if isinstance(x, float) and math.isnan(x): return []
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    return [x]

def _extract_vel_and_ts(ai_vel_list):
    """Return arrays: inputs(N,3), outputs(N,3), timestamps(N,) sorted."""
    inputs  = np.array([v['Input'] for v in ai_vel_list], dtype=float)
    outputs = np.array([list(v['Output']) for v in ai_vel_list], dtype=float)
    tsi = np.array([float(np.asarray(v['InputTimestamp']).ravel()[0]) for v in ai_vel_list], dtype=float)
    tso = np.array([float(np.asarray(v['OutputTimestamp']).ravel()[0]) for v in ai_vel_list], dtype=float)
    ts = np.where(np.isfinite(tso), tso, tsi)
    order = np.argsort(ts)
    return inputs[order], outputs[order], ts[order]

def _seconds_and_dt(ts_raw):
    """Convert ms→s if needed, return (timestamps_in_s, dt_in_s, factor)."""
    dts = np.diff(ts_raw, prepend=ts_raw[0])
    med_gap = np.median(dts[dts > 0]) if np.any(dts > 0) else 0.0
    factor = 1/1000.0 if med_gap > 5 else 1.0
    ts_sec = ts_raw * factor
    dt_sec = np.diff(ts_sec, prepend=ts_sec[0])
    return ts_sec, dt_sec, factor

def _ai_blocks_mask(ts_sec, trial, ms_to_s_factor=1.0):
    """Boolean mask of AI-on periods based on aiControlOn/aiControlOff timestamps."""
    ons  = _nan_to_list(trial.get('aiControlOn'))
    # ons  = _nan_to_list(trial.aiControlOn)
    # offs = _nan_to_list(trial.aiControlOff)
    offs = _nan_to_list(trial.get('aiControlOff'))
    if not ons and not offs:
        return np.zeros_like(ts_sec, dtype=bool)

    mask = np.zeros_like(ts_sec, dtype=bool)
    ons  = [float(o) * ms_to_s_factor for o in ons]
    offs = [float(o) * ms_to_s_factor for o in offs]

    for on, off in zip_longest(ons, offs, fillvalue=None):
        if on is None: continue
        if off is None:
            mask |= (ts_sec >= on)
        else:
            mask |= (ts_sec >= on) & (ts_sec < off)
    return mask

def ai_on_indices_from_trial(trial):
    """Return (set of AI-on indices, inputs, outputs, (ts_sec, dt_sec))."""
    ai_log = trial.get('aiVelocities', [])
    # ai_log = trial.aiVelocities
    if not ai_log:
        return set(), None, None, None

    inputs, outputs, ts_raw = _extract_vel_and_ts(ai_log)
    ts_sec, dt_sec, factor = _seconds_and_dt(ts_raw)

    mask= _ai_blocks_mask(ts_sec, trial, ms_to_s_factor=factor)
    # changed   = np.linalg.norm(outputs - inputs, axis=1) > 1e-9

    # if mask_flag.any():
    #     mask = mask_flag | changed
    # else:
    #     mask = changed

    return set(np.where(mask)[0]), inputs, outputs, (ts_sec, dt_sec)

import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def first_entry_index_into_square(x, z, center, half_w):
    if center is None or np.isnan(center[0]) or np.isnan(center[1]):
        return None
    cx, cz = center
    inside = (np.abs(x - cx) <= half_w) & (np.abs(z - cz) <= half_w)
    idx = np.where(inside)[0]
    return int(idx[0]) if idx.size else None

def trim_at_first_entry(x, z, target_center, target_half, obst_center=None, obst_half=None):
    """
    Return x[:k+1], z[:k+1] where k is the first index entering either obstacle or target.
    If neither is entered, return full arrays.
    """
    k_tgt = first_entry_index_into_square(x, z, target_center, target_half)
    k_obs = first_entry_index_into_square(x, z, obst_center,  obst_half) if obst_center is not None else None

    ks = [k for k in (k_tgt, k_obs) if k is not None]
    if not ks:
        return x, z
    k = max(min(ks) - 1, 0)  # stop right before first entry
    return x[:k+1], z[:k+1]

def resample_by_arclength(x, z, n=50):
    if len(x) < 2:
        return None
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
    if s[-1] == 0:
        return None
    s_new = np.linspace(0, s[-1], n)
    x_new = np.interp(s_new, s, x)
    z_new = np.interp(s_new, s, z)
    return np.stack([x_new, z_new], axis=1)

# ---------- main plot ----------
def plot_avg_trajectories_per_target_and_avoidance(df_all, monkey, experiment,
                                                   target_half=0.6, obstacle_half=0.45,
                                                   n_samples=60, use_median=False, show_iqr=True):
    """
    Average trajectories per target, split by condition (AI/No AI) and avoidance side (left/right).
    Each trial is trimmed at first entry into either the obstacle or the target,
    then resampled by arc-length. Aggregation uses median (default) or mean.
    """
    if df_all.empty:
        print("DataFrame is empty.")
        return

    df_all = df_all.copy()
    unique_targets = sorted(df_all['target_label'].dropna().unique())
    n_targets = len(unique_targets)
    ncols = 3
    nrows = (n_targets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, target in enumerate(unique_targets):
        ax = axes[idx]
        df_target = df_all[df_all['target_label'] == target]

        # target center (x, z)
        sample_row = df_target.iloc[0]
        tpos = sample_row['target_positions']
        tgt_center = (float(tpos[0]), float(tpos[2]))

        # obstacle center (x, z) if available
        tgt_key = (round(tgt_center[0], 1), round(tgt_center[1], 1))
        obst_center = None
        if 'target_to_obstacle_mapping' in globals() and tgt_key in target_to_obstacle_mapping:
            ox, oz = target_to_obstacle_mapping[tgt_key]
            obst_center = (float(ox), float(oz))

        for side in ['left', 'right']:
            for condition, color in [('AI', 'g'), ('No AI', 'r')]:
                df_cond = df_target[(df_target['condition'] == condition) &
                                    (df_target['avoidance_side'] == side)]
                grouped = df_cond.groupby("trial_id")

                trajs = []
                for _, g in grouped:
                    x = g['x'].values
                    z = g['z'].values
                    # trim at first entry into obstacle or target
                    x_trim, z_trim = trim_at_first_entry(
                        x, z, tgt_center, target_half,
                        obst_center=obst_center, obst_half=obstacle_half
                    )
                    traj = resample_by_arclength(x_trim, z_trim, n=n_samples)
                    if traj is not None:
                        trajs.append(traj)

                if not trajs:
                    continue

                stack = np.stack(trajs)  # [n_traj, n_samples, 2]
                if use_median:
                    center = np.median(stack, axis=0)
                else:
                    center = np.mean(stack, axis=0)
                label = f"{condition} (mean trajectory)" if side == 'left' else None
                ax.plot(center[:, 0], center[:, 1], color=color, lw=2, label=label)

                if show_iqr and stack.shape[0] >= 3:
                    # IQR ribbon
                    q25 = np.percentile(stack, 25, axis=0)
                    q75 = np.percentile(stack, 75, axis=0)
                    ax.fill_betweenx(center[:, 1], q25[:, 0], q75[:, 0], color=color, alpha=0.08, linewidth=0, label = f"{condition} (IQR)")

        # draw obstacle/target
        if obst_center is not None:
            plot_square(ax, obst_center, width_length=2*obstacle_half, color='r')
        plot_square(ax, tgt_center, width_length=2*target_half, color='g')

        ax.set_title(f"Target: {target}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Z Position")
        ax.grid(True)
        ax.set_aspect('equal')

        # one legend only
        if idx == 0:
            ax.legend()
        else:
            leg = ax.get_legend()
            if leg: leg.remove()

    for ax in axes[n_targets:]:
        ax.axis('off')

    plt.tight_layout()
    save_plot(fig, f"average_trajectories_per_target_{monkey}_{experiment}",
              subfolder="Average trajectories")
    plt.show()

def plot_success_rate_per_target(stats_on, stats_off, title=None):
    """
    Plot success rate per target for AI ON vs AI OFF in fixed order.
    """

    # Define fixed target order
    target_order = ["left", "slight_left", "straight", "slight_right", "right"]

    # Convert stats to dicts for lookup
    on_dict = {d['target']: d for d in stats_on}
    off_dict = {d['target']: d for d in stats_off}

    # Extract in correct order
    targets = target_order
    success_on = [on_dict[t]['success_rate (%)'] for t in targets]
    success_off = [off_dict[t]['success_rate (%)'] for t in targets]
    n_on = [on_dict[t]['num_trials'] for t in targets]
    n_off = [off_dict[t]['num_trials'] for t in targets]

    # Bar positions
    x = np.arange(len(targets))
    width = 0.35

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_off = ax.bar(x - width/2, success_off, width, label='AI OFF', color='red', alpha=0.7)
    bars_on = ax.bar(x + width/2, success_on, width, label='AI ON', color='green')

    # Labels and style
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Target')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=20)
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title)
    ax.legend()

    # Annotate trial counts
    for i, bar in enumerate(bars_off):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"n={n_off[i]}", ha='center', va='bottom', fontsize=8, color='red')
    for i, bar in enumerate(bars_on):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"n={n_on[i]}", ha='center', va='bottom', fontsize=8, color='green')

    plt.tight_layout()
    plt.show()

def plot_square(ax, center, width_length, color, label=None ,style = None):
    """ Plot traget-window """
    half_width_length = width_length / 2
    vertices = np.array([
        [center[0] - half_width_length, center[1] - half_width_length],
        [center[0] - half_width_length, center[1] + half_width_length],
        [center[0] + half_width_length, center[1] + half_width_length],
        [center[0] + half_width_length, center[1] - half_width_length],
        [center[0] - half_width_length, center[1] - half_width_length]  # To close the square
    ])
    ax.fill(vertices[:, 0], vertices[:, 1], 'white', zorder=2) 
    if style == None:
        ax.plot(vertices[:, 0], vertices[:, 1], color= color, zorder=3) #label =str(label)
    else:
        ax.plot(vertices[:, 0], vertices[:, 1], style, color=color, zorder=3)

def plot_collision_heatmap_per_target(df_all, target_to_obstacle_mapping, r_avatar=0.25, r_obstacle=0.45):
    """
    Generate one heatmap per target label showing AI OFF collision points.
    """
    if df_all.empty:
        print("DataFrame is empty — no trajectories to process.")
        return

    df_no_ai = df_all[df_all['condition'] == 'No AI']
    if 'target_label' not in df_no_ai.columns:
        print("Missing 'target_label' column.")
        return

    unique_targets = sorted(df_no_ai['target_label'].dropna().unique())
    ncols = 3
    nrows = (len(unique_targets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, target in enumerate(unique_targets):
        ax = axes[i]
        collision_points = []

        df_target = df_no_ai[df_no_ai['target_label'] == target]
        for trial_id, group in df_target.groupby("trial_id"):
            x = group["x"].values
            z = group["z"].values
            traj = np.stack([x, z], axis=1)
            target_pos = group["target_positions"].iloc[0]
            if not isinstance(target_pos, (list, np.ndarray)) or len(target_pos) < 3:
                continue

            obstacle_key = (round(target_pos[0], 1), round(target_pos[2], 1))
            obstacle = np.array(target_to_obstacle_mapping.get(obstacle_key, [np.nan, np.nan]))
            if np.isnan(obstacle).any():
                continue

            dists = np.linalg.norm(traj - obstacle, axis=1)
            collision_radius = r_avatar + r_obstacle
            collision_mask = dists <= collision_radius
            collision_points.extend(traj[collision_mask])

        if collision_points:
            collision_array = np.array(collision_points)
            sns.kdeplot(
                x=collision_array[:, 0],
                y=collision_array[:, 1],
                fill=True,
                cmap="Reds",
                bw_adjust=0.3,
                levels=100,
                thresh=0.05,
                ax=ax
            )

        # Plot target and obstacle
        if isinstance(target_pos, (list, np.ndarray)) and len(target_pos) >= 3:
            target_key = (round(target_pos[0], 1), round(target_pos[2], 1))
            obs = target_to_obstacle_mapping.get(target_key)
            if obs is not None:
                ax.add_patch(plt.Rectangle((obs[0] - 0.45, obs[1] - 0.45), 0.9, 0.9, edgecolor='r', facecolor='none'))
            ax.add_patch(plt.Rectangle((target_key[0] - 0.6, target_key[1] - 0.6), 1.2, 1.2, edgecolor='g', facecolor='none'))

        ax.set_title(f"Target: {target}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect('equal')
        ax.grid(True)

    for ax in axes[len(unique_targets):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_success_rate_per_session(success_rate_ai_on, success_rate_ai_off, monkey, experiment, chance_level=None):
    x = np.arange(len(success_rate_ai_on))
    avg_ai_on = int(np.round(np.mean(success_rate_ai_on) * 100))
    avg_ai_off = int(np.round(np.mean(success_rate_ai_off) * 100))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.array(success_rate_ai_on) * 100, marker='o', color='green', label=f'AI On (avg = {avg_ai_on}%)')
    ax.plot(x, np.array(success_rate_ai_off) * 100, marker='o', color='red', label=f'AI Off (avg = {avg_ai_off}%)')

    ax.axhline(avg_ai_on, color='green', linestyle='--', alpha=0.5)
    ax.axhline(avg_ai_off, color='red', linestyle='--', alpha=0.5)

    if chance_level is not None:
        ax.axhline(chance_level * 100, color='gray', linestyle=':', linewidth=2, label=f'Chance level: {int(np.round(chance_level*100))}%')

    ax.set_ylim(0, 100)
    ax.set_xlabel('Session')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'{monkey} - Success Rate per Session ({experiment})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in x])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    save_plot(fig, f"success_rate_per_session_{monkey}_{experiment}", subfolder="success_rate")
    plt.show()
    # plt.close(fig)

def _time_normalize(traj, n_time=200):
    """Linear resample a (T,D) trajectory to n_time."""
    T, D = traj.shape
    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, n_time)
    out = np.empty((n_time, D), dtype=float)
    for d in range(D):
        out[:, d] = np.interp(dst, src, traj[:, d])
    return out


import numpy as np
import matplotlib.pyplot as plt

def plot_mean_latent_trajs_by_status(
        df_latents,
        task,
        dims=(0, 1, 2),
        suptitle="Mean latent trajectories per target (correct trials)",
        smooth_sigma=None  # None or 0 => no smoothing
):
    """
    3D mean latent trajectories for AI-OFF vs AI-ON.
    Assumes 'latents' are already interpolated to a common length per trial.
    """

    # --- Target mappings ---
    fixed_targets = {
        (0., 0.75, 9.2): "straight",
        (7., 0.75, 6.): "right",
        (-7., 0.75, 6.): "left",
        (-3.5, 0.75, 8.5): "slight_left",
        (3.5, 0.75, 8.5): "slight_right"
    }
    moving_targets = {
        (0., 1., 9.2): "straight",
        (6., 1., 7.): "right",
        (-6., 1., 7.): "left",
        (-3., 1., 8.7): "slight_left",
        (3., 1., 8.7): "slight_right"
    }
    house_targets = {
        (-4.3, 0, 13.8): "right",
        (4.3, 0, 13.8): "left"
    }

    if task == "fixedCamera":
        target_map = fixed_targets
    elif task == "movingCamera":
        target_map = moving_targets
    else:
        target_map = house_targets

    colors = {
        "straight": "g",
        "right": "r",
        "left": "b",
        "slight_right": "y",
        "slight_left": "m"
    }
    ai_titles = {0: "BCI-only (OFF)", 1: "AI-ON (shared-control)"}

    # optional smoothing
    do_smooth = smooth_sigma is not None and smooth_sigma > 0
    if do_smooth:
        from scipy.ndimage import gaussian_filter1d

    fig = plt.figure(figsize=(6.5, 9))
    axes = {}
    all_x, all_y, all_z = [], [], []  # to enforce same axis limits

    for k, ai in enumerate([0, 1], start=1):
        ax = fig.add_subplot(2, 1, k, projection='3d')
        axes[ai] = ax

        sub = df_latents[df_latents['ai_status'] == ai]
        trial_categories = {v: [] for v in target_map.values()}

        # --- collect trials per target category ---
        for _, row in sub.iterrows():
            t_tuple = tuple(row['target'])
            if t_tuple in target_map:
                trial_categories[target_map[t_tuple]].append(
                    np.asarray(row['latents'], float)
                )

        # --- mean trajectory per category ---
        for cat, trials in trial_categories.items():
            if not trials:
                continue

            arr = np.stack(trials)          # (n_trials, T, D)
            mean_traj = np.nanmean(arr, 0)  # (T, D)

            if do_smooth:
                mean_traj = gaussian_filter1d(mean_traj, sigma=smooth_sigma,
                                             axis=0, mode='nearest')

            x = mean_traj[:, dims[0]]
            y = mean_traj[:, dims[1]]
            z = mean_traj[:, dims[2]]

            # keep for global limits
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

            # plot
            ax.plot(x, y, z, color=colors.get(cat, 'k'), lw=2.2, label=cat)
            ax.scatter(x[0], y[0], z[0], c=colors.get(cat, 'k'),
                       marker='X', s=100)

        ax.set_title(ai_titles[ai])
        ax.set_xlabel(f"Latent {dims[0] + 1}")
        ax.set_ylabel(f"Latent {dims[1] + 1}")
        ax.set_zlabel(f"Latent {dims[2] + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    # --- enforce same axis limits on both panels ---
    if len(all_x) > 0:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)

        # small padding so nothing sits on the border
        def with_pad(lo, hi, frac=0.05):
            span = hi - lo
            if span == 0:
                span = 1e-6
            pad = span * frac
            return lo - pad, hi + pad

        x_min, x_max = with_pad(x_min, x_max)
        y_min, y_max = with_pad(y_min, y_max)
        z_min, z_max = with_pad(z_min, z_max)

        for ax in axes.values():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

    fig.suptitle(suptitle, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    save_plot(fig, suptitle, "Mean Trajectory")
    return fig

import numpy as np

def compute_session_mean_trajs(df_latents, target_map):
    """
    df_latents: output of extract_latents_per_trial for ONE session
                columns: ['trial_id', 'ai_status', 'target', 'latents']
    target_map: dict { (x,y,z) -> label_str }

    Returns:
        means[ai_status][label] = (T, D) mean trajectory for that session
    """
    means = {0: {}, 1: {}}

    for ai in [0, 1]:
        sub = df_latents[df_latents['ai_status'] == ai]

        # collect latents per label
        per_label = {}
        for _, row in sub.iterrows():
            t_tuple = tuple(row['target'])
            if t_tuple not in target_map:
                continue
            label = target_map[t_tuple]
            per_label.setdefault(label, []).append(np.asarray(row['latents'], float))

        # average trials -> one trajectory per label
        for label, trials in per_label.items():
            if not trials:
                continue
            arr = np.stack(trials, axis=0)   # (n_trials, T, D), T fixed by extract_latents_per_trial
            mean_traj = np.nanmean(arr, axis=0)  # (T, D)
            means[ai][label] = mean_traj

    return means

import matplotlib.pyplot as plt

def plot_mean_latent_trajs_across_sessions(session_means_list,
                                           dims=(0, 1, 2),
                                           suptitle="Mean latent trajectories (all sessions)"):
    """
    session_means_list: list of dicts like returned by compute_session_mean_trajs
                        each element: means[ai_status][label] = (T, D)

    dims: which latent dims to show in 3D
    """

    colors = {
        "straight": "g",
        "right": "r",
        "left": "b",
        "slight_right": "y",
        "slight_left": "m"
    }
    ai_titles = {0: "BCI-only (OFF)", 1: "AI-ON (shared-control)"}

    fig = plt.figure(figsize=(6.5, 9))
    axes = {}
    all_x, all_y, all_z = [], [], []

    for k, ai in enumerate([0, 1], start=1):
        ax = fig.add_subplot(2, 1, k, projection='3d')
        axes[ai] = ax

        # collect all labels present in at least one session for this ai
        labels = set()
        for sess_means in session_means_list:
            labels.update(sess_means[ai].keys())

        for label in sorted(labels):
            # gather this label's trajectory from all sessions that have it
            trajs = []
            for sess_means in session_means_list:
                if label in sess_means[ai]:
                    trajs.append(sess_means[ai][label])  # (T, D)

            if not trajs:
                continue

            arr = np.stack(trajs, axis=0)         # (n_sessions_with_label, T, D)
            mean_traj = np.nanmean(arr, axis=0)   # (T, D)

            x = mean_traj[:, dims[0]]
            y = mean_traj[:, dims[1]]
            z = mean_traj[:, dims[2]]

            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

            ax.plot(x, y, z, color=colors.get(label, 'k'), lw=2.2, label=label)
            ax.scatter(x[0], y[0], z[0], c=colors.get(label, 'k'),
                       marker='X', s=100)

        ax.set_title(ai_titles[ai])
        ax.set_xlabel(f"Latent {dims[0] + 1}")
        ax.set_ylabel(f"Latent {dims[1] + 1}")
        ax.set_zlabel(f"Latent {dims[2] + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    # same axis limits for ON/OFF
    if all_x:
        def with_pad(lo, hi, frac=0.05):
            span = hi - lo if hi > lo else 1e-6
            pad = span * frac
            return lo - pad, hi + pad

        x_min, x_max = with_pad(min(all_x), max(all_x))
        y_min, y_max = with_pad(min(all_y), max(all_y))
        z_min, z_max = with_pad(min(all_z), max(all_z))

        for ax in axes.values():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

    fig.suptitle(suptitle, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig

def plot_alpha_vs_time_single_trial_fixed_obstacle(
    monkey,
    experiment,
    base_dir,
    session_idx=0,
    trial_idx=0,
    stride=1,        # optional thinning: keep every k-th sample
    min_dt=None,     # optional: min time (s) between kept samples
    min_dist=None,   # optional: min spatial distance between kept samples
    min_alpha=0.2,   # floor after robust normalization
    show_title=True,
    save=True
):
    """
    Plot α (= 1 − normalized EntropyLb) vs *time from trial start*
    for a single trial in the Fixed Obstacle task.

    Arguments
    ---------
    monkey, experiment, base_dir : as in your other analysis functions
    session_idx : which session in all_trials to use
    trial_idx   : which trial within that session
    stride      : keep every k-th sample (after alignment)
    min_dt      : if not None, enforce at least this many seconds between
                  consecutive kept samples (after stride)
    min_dist    : if not None, enforce at least this many meters between
                  consecutive kept samples (after stride)
    min_alpha   : lower bound after per-trial robust normalization
    show_title  : whether to put a descriptive title on the figure
    save        : if True, calls save_plot(...)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- helper: simple thinning ----------
    def thin_indices(ts, pos, stride=1, min_dt=None, min_dist=None):
        """
        ts   : (T,) time array
        pos  : (T,2) positions
        Returns indices into ts/pos that satisfy thinning constraints.
        """
        N = len(ts)
        if N == 0:
            return []

        # base: stride
        idx = list(range(0, N, max(1, stride)))

        # time thinning
        if min_dt is not None:
            kept = []
            last_t = None
            for i in idx:
                t = ts[i]
                if not np.isfinite(t):
                    continue
                if last_t is None or (t - last_t) >= float(min_dt):
                    kept.append(i)
                    last_t = t
            idx = kept

        # distance thinning
        if min_dist is not None and len(idx) > 1:
            kept = [idx[0]]
            last_p = pos[idx[0]]
            for i in idx[1:]:
                if np.linalg.norm(pos[i] - last_p) >= float(min_dist):
                    kept.append(i)
                    last_p = pos[i]
            idx = kept

        return idx

    # ---------- load data and select trial ----------
    all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, \
        pkl_files, ai_trials = load_files(experiment, monkey, base_dir=base_dir)

    if session_idx < 0 or session_idx >= len(all_trials):
        print(f"[plot_alpha_single] Invalid session_idx {session_idx}")
        return

    session = all_trials[session_idx]
    if trial_idx < 0 or trial_idx >= len(session):
        print(f"[plot_alpha_single] Invalid trial_idx {trial_idx} for session {session_idx}")
        return

    trial = session[trial_idx]

    ai_recs = getattr(trial, "aiVelocities", None)
    if not ai_recs:
        print("[plot_alpha_single] No aiVelocities on this trial.")
        return

    traj = getattr(trial, "avatarTrajectory", None)
    if not isinstance(traj, dict) or "time" not in traj:
        print("[plot_alpha_single] Trial has no avatarTrajectory['time']; cannot align.")
        return

    # ---------- align aiVelocities with trajectory ----------
    # Find first non-zero AI output to avoid the flat initial segment
    start_idx = None
    for i, rec in enumerate(ai_recs):
        out = rec.get("Output", None)
        if out is None:
            continue
        v = np.asarray(out, float).ravel()
        if v.size >= 2 and np.linalg.norm(v[:2]) > 0:
            start_idx = i
            break

    if start_idx is None:
        print("[plot_alpha_single] AI output is zero for entire trial; skipping.")
        return

    ai_recs = ai_recs[start_idx:]

    # Time stamps from trajectory (assumed aligned 1:1 with AI records)
    times = np.asarray(traj.get("time", []), float)
    if times.size < len(ai_recs) + start_idx:
        # defensive: clip to available range
        T = min(times.size - start_idx, len(ai_recs))
        ai_recs = ai_recs[:T]
    else:
        T = len(ai_recs)

    times = times[start_idx:start_idx + T]

    # positions (x,z) for thinning
    xs = np.asarray(traj.get("x", []), float)[start_idx:start_idx + T]
    zs = np.asarray(traj.get("z", []), float)[start_idx:start_idx + T]
    if xs.size != T or zs.size != T:
        print("[plot_alpha_single] Trajectory length mismatch; skipping.")
        return

    pos_full = np.column_stack([xs, zs])

    # ---------- Entropy and α ----------
    ent_raw = []
    for rec in ai_recs:
        e = rec.get("EntropyLb", None)
        if isinstance(e, (list, tuple, np.ndarray)):
            e = float(np.asarray(e).ravel()[0]) if len(e) else np.nan
        ent_raw.append(np.nan if e is None else float(e))
    ent_raw = np.asarray(ent_raw, float)

    # basic sanity
    valid = np.isfinite(times) & np.isfinite(ent_raw)
    if valid.sum() < 3:
        print("[plot_alpha_single] Not enough valid (time, entropy) samples.")
        return

    times = times[valid]
    pos_full = pos_full[valid]
    ent_raw = ent_raw[valid]

    # ---------- thinning ----------
    idx_thin = thin_indices(times, pos_full, stride=stride, min_dt=min_dt, min_dist=min_dist)
    if len(idx_thin) < 3:
        print("[plot_alpha_single] Not enough samples after thinning.")
        return

    times = times[idx_thin]
    pos_full = pos_full[idx_thin]
    ent_raw = ent_raw[idx_thin]

    # ---------- per-trial α ----------
    # robust_alpha must already be defined in your codebase
    alpha_vals = robust_alpha(ent_raw, floor=min_alpha)

    # time relative to trial start
    t0 = times[0]
    t_rel = times - t0

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(t_rel, alpha_vals, lw=2.0)

    ax.set_xlabel("Time from trial start (s)")
    ax.set_ylabel("Posterior confidence index α")
    if show_title:
        ax.set_title(f"{monkey} – {experiment}\nSingle trial (session {session_idx}, trial {trial_idx})")

    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        fname = f"alpha_vs_time_single_fixedObs_{monkey}_{experiment}_s{session_idx}_tr{trial_idx}"
        try:
            save_plot(fig, fname, subfolder="Entropy_single_trials")
        except Exception:
            pass

    plt.show()
    plt.close(fig)
def plot_time_to_target_histograms(all_monkeys_time, training_time, monkey_names, colors, experiment):
    """
    Plot histograms of time-to-target values for each monkey, with vertical lines for training and decoding means.

    Args:
        all_monkeys_time (dict): { 'Monkey 1': [times], 'Monkey 2': [times], ... }
        training_time (float): Training time in ms to compare against.
        monkey_names (list): List of monkey identifiers to plot.
        colors (dict): Monkey-specific color mapping.
        experiment (str): Name of the experiment (used for output filename).
    """
    # Filter only monkeys with non-empty data
    available_monkeys = [m for m in monkey_names if len(all_monkeys_time.get(m, [])) > 0]
    n_monkeys = len(available_monkeys)
    
    fig, axes = plt.subplots(1, n_monkeys, figsize=(15, 5), sharey=False)
    plt.rcParams.update({'font.size': 8})
    
    for i, monkey in enumerate(available_monkeys):
        data = all_monkeys_time[monkey]
        ax = axes[i] if n_monkeys > 1 else axes  # handle case when n_monkeys == 1
        ax.hist(data, bins=25, alpha=0.6, edgecolor='black', color=colors[monkey])
        ax.axvline(training_time, linestyle='--', color='black', label=f'Training: {int(training_time)}ms')
        ax.axvline(np.mean(data), color='r', linestyle='--', label=f"Decoding: {int(np.mean(data))}ms")
        ax.set_title(monkey)
        ax.set_xlabel('Time to Target [ms]')
        max_x = 8000 if experiment == "Continuous Navigation" else 5500
        ax.set_xlim([min(data), max_x])
        if i == 0:
            ax.set_ylabel('Frequency')
        ax.legend()

    # Tight layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_plot(fig, f"{experiment}_time_to_target", "time_to_target")
    plt.close(fig)

def plot_failure_modes_nonexec_incorrect(
    grand_summary,
    save_prefix="failure_modes_nonexec_incorrect",
    alt_wrong="two-sided",
    alt_other="two-sided",
):
    """
    Plot failure composition *restricted to incorrect trials that are NOT execution errors*.

    We renormalize within:
        nonexec_incorrect = wrong_target + other

    Output bars are % of (non-exec incorrect) trials, NOT % of all trials.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = grand_summary.get("failure_props_per_unit", [])
    if not rows:
        print("[plot_failure_modes_nonexec] No failure_props_per_unit in grand_summary.")
        return

    categories = ["wrong_target", "other"]
    pretty = {
        "wrong_target": "Wrong\ntarget",
        "other": "Other",
    }
    alt_map = {
        "wrong_target": alt_wrong,
        "other": alt_other,
    }

    # -------------------- build unit lookup --------------------
    units = {}  # (monkey, experiment, condition) -> row
    for r in rows:
        m = r.get("monkey", "?")
        e = r.get("experiment", "?")
        c = r.get("condition")
        if c not in ("AI", "No AI"):
            continue
        units[(m, e, c)] = r

    if not units:
        print("[plot_failure_modes_nonexec] No AI/No-AI rows found.")
        return

    def _f(r, key):
        try:
            return float(r.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    def compute_wrong_other_renorm(r):
        # --- execution errors (excluded from this plot) ---
        execution = (
            _f(r, "stuck_obstacle") +
            _f(r, "overshoot") +
            _f(r, "not_long_enough") +
            _f(r, "not_close_to_true_z")
        )

        # --- your "wrong" family (as currently defined) ---
        wrong = (
            _f(r, "wrong_target_choice") +
            _f(r, "neighbor_choice") +
            _f(r, "ambiguous_choice") +
            _f(r, "ai_bci_failure_target") +
            _f(r, "wrong_target_other") +
            _f(r, "ambiguous_choice_unknown")
        )

        other = _f(r, "other")

        # non-exec incorrect pool
        denom = wrong + other

        # If denom==0, that unit has no non-exec incorrect trials (or missing fields)
        if not np.isfinite(denom) or denom <= 0:
            return {"wrong_target": np.nan, "other": np.nan, "denom": 0.0, "execution": execution}

        return {
            "wrong_target": wrong / denom,
            "other": other / denom,
            "denom": denom,
            "execution": execution,
        }

    def mean_sem(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        m = float(np.mean(a))
        sem = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else np.nan
        return m, sem

    def p_to_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def _wilcoxon_p(arr_on, arr_off, alternative):
        if wilcoxon is None:
            return np.nan
        arr_on = np.asarray(arr_on, float)
        arr_off = np.asarray(arr_off, float)
        ok = np.isfinite(arr_on) & np.isfinite(arr_off)
        a1 = arr_on[ok]
        a0 = arr_off[ok]
        if a0.size < 2:
            return np.nan
        if np.allclose(a1, a0, equal_nan=False):
            return 1.0
        try:
            return float(wilcoxon(a1, a0, alternative=alternative,
                                  zero_method="wilcox", mode="auto").pvalue)
        except Exception:
            return np.nan

    def collect_pairs(pair_keys):
        vals_off = {c: [] for c in categories}
        vals_on  = {c: [] for c in categories}
        denoms_off = []
        denoms_on  = []
        for (m, e) in pair_keys:
            r_off = units.get((m, e, "No AI"), None)
            r_on  = units.get((m, e, "AI"), None)
            if r_off is None or r_on is None:
                continue

            co = compute_wrong_other_renorm(r_off)
            cn = compute_wrong_other_renorm(r_on)

            # require denominator > 0 in BOTH conditions for a paired unit
            if (co["denom"] <= 0) or (cn["denom"] <= 0):
                continue

            for c in categories:
                vals_off[c].append(co[c])
                vals_on[c].append(cn[c])
            denoms_off.append(co["denom"])
            denoms_on.append(cn["denom"])
        return vals_off, vals_on, denoms_off, denoms_on

    def make_plot(pair_keys, title, outname):
        if not pair_keys:
            print(f"[plot_failure_modes_nonexec] {title}: no paired AI/No-AI units.")
            return

        vals_off, vals_on, den_off, den_on = collect_pairs(pair_keys)

        means_off, sems_off, means_on, sems_on, pvals, ns = [], [], [], [], [], []

        for c in categories:
            arr_off = np.asarray(vals_off[c], dtype=float)
            arr_on  = np.asarray(vals_on[c],  dtype=float)

            m_off, se_off = mean_sem(arr_off * 100.0)
            m_on,  se_on  = mean_sem(arr_on  * 100.0)

            means_off.append(m_off); sems_off.append(se_off)
            means_on.append(m_on);   sems_on.append(se_on)

            alt = alt_map.get(c, "two-sided")
            p = _wilcoxon_p(arr_on, arr_off, alternative=alt)
            pvals.append(p)

            ok = np.isfinite(arr_off) & np.isfinite(arr_on)
            ns.append(int(np.sum(ok)))

        x = np.arange(len(categories), dtype=float)
        width = 0.35
        fig, ax = plt.subplots(figsize=(6.4, 4.0))

        def draw_group(offset, means, sems, label, facecolor):
            xs = x + offset
            bars = ax.bar(xs, means, width, label=label,
                          color=facecolor, edgecolor="black",
                          linewidth=0.5, alpha=0.9)
            means_a = np.asarray(means, float)
            sems_a  = np.asarray(sems,  float)
            ok = np.isfinite(means_a) & np.isfinite(sems_a)
            if np.any(ok):
                ax.errorbar(xs[ok], means_a[ok], yerr=sems_a[ok],
                            fmt="none", ecolor="black",
                            elinewidth=1.0, capsize=4, capthick=1.0, zorder=5)
            return bars

        bars_off = draw_group(-width/2, means_off, sems_off, "AI Off", "#e74c3c")
        bars_on  = draw_group(+width/2, means_on,  sems_on,  "AI On",  "#2ecc71")

        ax.set_xticks(x)
        ax.set_xticklabels([pretty[c] for c in categories], rotation=0)
        ax.set_ylabel("Percentage of non-exec incorrect trials (%)")
        ax.set_title(title)
        ax.legend(frameon=False)

        ax.set_ylim(0, 100)

        # significance
        h = 2.5
        for bar_off, bar_on, p in zip(bars_off, bars_on, pvals):
            stars = p_to_stars(p)
            if not stars:
                continue
            x1 = bar_off.get_x() + bar_off.get_width()/2
            x2 = bar_on.get_x()  + bar_on.get_width()/2
            y  = max(bar_off.get_height(), bar_on.get_height()) + h
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", linewidth=1.0)
            ax.text((x1+x2)/2, y+h*1.05, stars, ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        try:
            save_plot(fig, outname, title)
        except Exception:
            pass
        plt.show()
        plt.close(fig)

        print(f"\n[{title}]")
        for c, p, n in zip(categories, pvals, ns):
            print(f"  {c:12s}: Wilcoxon ({alt_map.get(c,'two-sided')}), p={p if np.isfinite(p) else np.nan}, n={n}")

    # ========== GLOBAL ==========
    global_pair_keys = sorted({
        (m, e)
        for (m, e, c) in units.keys()
        if (m, e, "AI") in units and (m, e, "No AI") in units
    })
    make_plot(
        global_pair_keys,
        title="Non-execution incorrect outcomes — all monkeys × all tasks",
        outname=f"{save_prefix}__global"
    )

    # ========== PER MONKEY ==========
    monkey_list = sorted({m for (m, e, c) in units.keys()})
    for m in monkey_list:
        pair_keys_m = sorted({
            (mm, ee)
            for (mm, ee, c) in units.keys()
            if mm == m and (mm, ee, "AI") in units and (mm, ee, "No AI") in units
        })
        make_plot(
            pair_keys_m,
            title=f"Non-execution incorrect outcomes — monkey {m}",
            outname=f"{save_prefix}__monkey_{m}"
        )

    # ========== PER TASK ==========
    experiment_list = sorted({e for (m, e, c) in units.keys()})
    for e in experiment_list:
        pair_keys_e = sorted({
            (mm, ee)
            for (mm, ee, c) in units.keys()
            if ee == e and (mm, ee, "AI") in units and (mm, ee, "No AI") in units
        })
        make_plot(
            pair_keys_e,
            title=f"Non-execution incorrect outcomes — task {e}",
            outname=f"{save_prefix}__task_{e}"
        )

def plot_failure_modes_all(
    grand_summary,
    save_prefix="failure_modes_rebinned",
    alt_exec="less",          # test AI < NoAI for EXECUTION errors (one-sided)
    alt_wrong="two-sided",    # wrong-target can go either way
    alt_other="two-sided",    # other can go either way
):
    """
    Re-binned global failure modes with paired AI vs No-AI comparisons + significance.

    Categories (fractions of ALL trials):
      1) execution_error = stuck_obstacle + overshoot + not_long_enough + not_close_to_true_z
      2) wrong_target    = pooled wrong-target family
      3) other           = residual 'other' bucket

    Significance:
      - Paired Wilcoxon per category across units (monkey×experiment)
      - Stars drawn above each Off-vs-On pair
    """

    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    rows = grand_summary.get("failure_props_per_unit", [])
    if not rows:
        print("[plot_failure_modes_global] No failure_props_per_unit in grand_summary.")
        return

    # categories = ["execution_error", "wrong_target", "other"]
    mode = "all_trials_3cat"  # OR "nonexec_incorrect_2cat"

    if mode == "all_trials_3cat":
        categories = ["execution_error", "wrong_target", "other"]
        ylabel = "Percentage of all trials (%)"
        pretty = {"execution_error":"Execution\nerrors","wrong_target":"Wrong\ntarget","other":"Other"}
    else:
        categories = ["wrong_target", "other"]
        ylabel = "Percentage of non-exec incorrect trials (%)"
        pretty = {"wrong_target":"Wrong\ntarget","other":"Other"}

    # pretty = {
    #     "execution_error": "Execution\nerrors",
    #     "wrong_target": "Wrong\ntarget",
    #     "other": "Other",
    # }
    pretty = {
        "execution_error": "Execution\nerrors",
        "wrong_target": "Intent/\nchoice\nerrors",   # or "Decoder choice\nerrors"
        "other": "Other",
    }
    alt_map = {
        "execution_error": alt_exec,
        "wrong_target": alt_wrong,
        "other": alt_other,
    }

    # -------------------- build unit lookup --------------------
    units = {}  # (monkey, experiment, condition) -> row
    for r in rows:
        m = r.get("monkey", "?")
        e = r.get("experiment", "?")
        c = r.get("condition")
        if c not in ("AI", "No AI"):
            continue
        units[(m, e, c)] = r

    if not units:
        print("[plot_failure_modes_global] No AI/No-AI rows found.")
        return

    # ---- helpers ----
    def _f(r, key):
        try:
            return float(r.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    def compute_cats(r, mode="all_trials_3cat"):
        # --- execution failures (fractions of ALL trials) ---
        exec_all = (
            _f(r, "stuck_obstacle") +
            _f(r, "overshoot") +
            _f(r, "not_long_enough") +
            _f(r, "not_close_to_true_z")
        )

        # --- decoder-choice / intent-family bucket (fractions of ALL trials) ---
        wrong_all = (
            _f(r, "wrong_target_choice") +
            _f(r, "neighbor_choice") +
            _f(r, "ambiguous_choice") +
            _f(r, "ai_bci_failure_target") +
            _f(r, "wrong_target_other") +
            _f(r, "ambiguous_choice_unknown")
        )

        other_all = _f(r, "other")

        # total incorrect as fraction of ALL trials
        total_incorrect_all = 1.0 - _f(r, "success")

        if mode == "all_trials_3cat":
            # % of ALL trials (your original figure)
            return {
                "execution_error": exec_all,
                "wrong_target": wrong_all,
                "other": other_all,
            }

        if mode == "nonexec_incorrect_2cat":
            # Normalize within "incorrect trials that are NOT execution errors"
            denom = total_incorrect_all - exec_all
            if not np.isfinite(denom) or denom <= 0:
                return {"wrong_target": np.nan, "other": np.nan}

            return {
                "wrong_target": wrong_all / denom,
                "other": other_all / denom,
            }

        raise ValueError(f"Unknown mode: {mode}")

    def mean_sem(arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        m = float(np.mean(a))
        sem = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else np.nan
        return m, sem

    def p_to_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def _wilcoxon_p(arr_on, arr_off, alternative):
        if wilcoxon is None:
            return np.nan
        arr_on = np.asarray(arr_on, float)
        arr_off = np.asarray(arr_off, float)
        ok = np.isfinite(arr_on) & np.isfinite(arr_off)
        a1 = arr_on[ok]
        a0 = arr_off[ok]
        if a0.size < 2:
            return np.nan
        if np.allclose(a1, a0, equal_nan=False):
            return 1.0
        # if all diffs are zero, wilcoxon raises; catch it
        try:
            return float(wilcoxon(a1, a0, alternative=alternative, zero_method="wilcox", mode="auto").pvalue)
        except Exception:
            return np.nan

    def collect_pairs(pair_keys):
        vals_off = {c: [] for c in categories}
        vals_on  = {c: [] for c in categories}
        for (m, e) in pair_keys:
            r_off = units.get((m, e, "No AI"), None)
            r_on  = units.get((m, e, "AI"), None)
            if r_off is None or r_on is None:
                continue
            co = compute_cats(r_off, mode=mode)
            cn = compute_cats(r_on, mode=mode)
            for c in categories:
                vals_off[c].append(co[c])
                vals_on[c].append(cn[c])
        return vals_off, vals_on

    def make_plot(pair_keys, title, outname):
        if not pair_keys:
            print(f"[plot_failure_modes_global] {title}: no paired AI/No-AI units.")
            return

        vals_off, vals_on = collect_pairs(pair_keys)

        means_off, sems_off, means_on, sems_on, pvals = [], [], [], [], []
        ns = []

        for c in categories:
            arr_off = np.asarray(vals_off[c], dtype=float)
            arr_on  = np.asarray(vals_on[c],  dtype=float)

            m_off, se_off = mean_sem(arr_off * 100.0)
            m_on,  se_on  = mean_sem(arr_on  * 100.0)

            means_off.append(m_off); sems_off.append(se_off)
            means_on.append(m_on);   sems_on.append(se_on)

            alt = alt_map.get(c, "two-sided")
            p = _wilcoxon_p(arr_on, arr_off, alternative=alt)
            pvals.append(p)

            ok = np.isfinite(arr_off) & np.isfinite(arr_on)
            ns.append(int(np.sum(ok)))

        # ---- plot ----
        x = np.arange(len(categories), dtype=float)
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))

        def draw_group(offset, means, sems, label, facecolor):
            xs = x + offset
            bars = ax.bar(
                xs, means, width,
                label=label,
                color=facecolor,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
            means_a = np.asarray(means, float)
            sems_a  = np.asarray(sems,  float)
            ok = np.isfinite(means_a) & np.isfinite(sems_a)
            if np.any(ok):
                ax.errorbar(
                    xs[ok],
                    means_a[ok],
                    yerr=sems_a[ok],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=4,
                    capthick=1.0,
                    zorder=5,
                )
            return bars

        bars_off = draw_group(-width/2, means_off, sems_off, "AI Off", "#e74c3c")
        bars_on  = draw_group(+width/2, means_on,  sems_on,  "AI On",  "#2ecc71")

        ax.set_xticks(x)
        ax.set_xticklabels([pretty[c] for c in categories], rotation=0)
        ax.set_ylabel("Percentage of all trials (%)")
        ax.set_title(title)
        ax.legend(frameon=False)

        finite_vals = [v for v in list(means_off) + list(means_on) if np.isfinite(v)]
        ymax = (max(finite_vals) * 1.35) if finite_vals else 1.0
        ymax = max(1e-6, ymax)
        ax.set_ylim(0, ymax)

        # ---- significance brackets + stars ----
        h = ymax * 0.03  # vertical height of bracket
        for i, (bar_off, bar_on, p, n) in enumerate(zip(bars_off, bars_on, pvals, ns)):
            stars = p_to_stars(p)
            if not stars:
                continue

            x1 = bar_off.get_x() + bar_off.get_width() / 2.0
            x2 = bar_on.get_x()  + bar_on.get_width()  / 2.0
            y  = max(bar_off.get_height(), bar_on.get_height()) + h

            ax.plot([x1, x1, x2, x2],
                    [y,  y + h, y + h, y],
                    color="black", linewidth=1.0)
            ax.text((x1 + x2)/2.0, y + h*1.1, stars,
                    ha="center", va="bottom", fontsize=10)

            # optional: tiny n label under the stars (comment out if you dislike it)
            # ax.text((x1 + x2)/2.0, y + h*0.2, f"n={n}",
            #         ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        try:
            save_plot(fig, outname, title)
        except Exception:
            pass
        plt.show()
        plt.close(fig)

        # print stats to console (useful for paper)
        print(f"\n[{title}]")
        for c, p, n in zip(categories, pvals, ns):
            alt = alt_map.get(c, "two-sided")
            print(f"  {c:16s}: Wilcoxon ({alt}), p={p if np.isfinite(p) else np.nan}, n={n}")

    # =========================
    # 1) GLOBAL (all monkeys × tasks)
    # =========================
    global_pair_keys = sorted({
        (m, e)
        for (m, e, c) in units.keys()
        if (m, e, "AI") in units and (m, e, "No AI") in units
    })
    make_plot(
        global_pair_keys,
        title="Failure modes (execution grouped) — all monkeys × all tasks",
        outname=f"{save_prefix}__global"
    )

    # # =========================
    # # 2) PER MONKEY (across tasks)
    # # =========================
    # monkey_list = sorted({m for (m, e, c) in units.keys()})
    # for m in monkey_list:
    #     pair_keys_m = sorted({
    #         (mm, ee)
    #         for (mm, ee, c) in units.keys()
    #         if mm == m and (mm, ee, "AI") in units and (mm, ee, "No AI") in units
    #     })
    #     make_plot(
    #         pair_keys_m,
    #         title=f"Failure modes (execution grouped) — monkey {m} (across tasks)",
    #         outname=f"{save_prefix}__monkey_{m}"
    #     )

    # # =========================
    # # 3) PER TASK / EXPERIMENT (across monkeys)
    # # =========================
    # experiment_list = sorted({e for (m, e, c) in units.keys()})
    # for e in experiment_list:
    #     pair_keys_e = sorted({
    #         (mm, ee)
    #         for (mm, ee, c) in units.keys()
    #         if ee == e and (mm, ee, "AI") in units and (mm, ee, "No AI") in units
    #     })
    #     make_plot(
    #         pair_keys_e,
    #         title=f"Failure modes (execution grouped) — task {e} (across monkeys)",
    #         outname=f"{save_prefix}__task_{e}"
    #     )
