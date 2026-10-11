from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TASK_LABELS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle follow-up",
}

TARGET_LABELS = {
    "left": "Left",
    "slight_left": "Slight left",
    "straight": "Straight",
    "slight_right": "Slight right",
    "right": "Right",
}

MONKEY_COLORS = {
    "Monkey 1": "#E69F00",
    "Monkey 2": "#0072B2",
}

TASK_MARKERS = {
    "Appearing Obstacle": "o",
    "Appearing Obstacle follow-up": "s",
    "Fixed Obstacle": "^",
}


def save_plot(fig, plot_name, subfolder):
    figure_dir = os.environ.get("AI_PAPER_FIGURE_DIR", "Figure 4")
    out_dir = PROJECT_ROOT / "figures" / figure_dir / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{plot_name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    print(f"Saved: {path}")


def _ci95_t(values):
    from scipy.stats import t

    values = np.asarray([value for value in values if np.isfinite(value)], float)
    if values.size == 0:
        return np.nan, (np.nan, np.nan)
    mean = values.mean()
    if values.size == 1:
        return mean, (mean, mean)
    sem = values.std(ddof=1) / np.sqrt(values.size)
    half_width = t.ppf(0.975, df=values.size - 1) * sem
    return mean, (mean - half_width, mean + half_width)


def _load_target_session_rows(base_dir):
    table_path = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_target_session_rows.csv"
    if table_path.exists():
        return pd.read_csv(table_path)

    try:
        from scripts.run_gain_curve_pooling_audit import DEFAULT_DATASETS, collect_target_session_rows

        data = collect_target_session_rows(base_dir, DEFAULT_DATASETS)
        if not data.empty:
            table_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(table_path, index=False)
        return data
    except Exception as exc:
        print(f"[plot_summary_scatter] Could not load/build target-session rows: {exc}")
        return pd.DataFrame()


def _load_common_model_stats():
    stats_path = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_nested_model_comparisons.csv"
    if not stats_path.exists():
        return np.nan, np.nan
    stats = pd.read_csv(stats_path)
    common = stats[stats["model"].eq("common_quadratic_monkey_task_intercepts")]
    if common.empty:
        return np.nan, np.nan
    row = common.iloc[0]
    return float(row["quadratic_beta"]), float(row["quadratic_p_clustered"])


def plot_gain_with_labels(per_target_gains, acc_ai_off, monkey, experiment, label_fmt="{:.0f}%", label_pos="auto"):
    """Panels A/B: sessionwise shared-control gain with BCI-only baseline in target labels."""
    from matplotlib.lines import Line2D

    preferred = ["left", "slight_left", "straight", "slight_right", "right"]
    order = [target for target in preferred if target in per_target_gains] or sorted(per_target_gains.keys())
    x = np.arange(len(order))

    gain_mean_pp, gain_lo_pp, gain_hi_pp = [], [], []
    gain_points_pp = []
    for target in order:
        values = np.asarray(per_target_gains.get(target, []), dtype=float)
        values = values[np.isfinite(values)]
        mean, (lo, hi) = _ci95_t(values)
        gain_points_pp.append(values * 100.0)
        gain_mean_pp.append(mean * 100.0)
        gain_lo_pp.append(max(0, (mean - lo) * 100.0) if np.isfinite(lo) else 0.0)
        gain_hi_pp.append(max(0, (hi - mean) * 100.0) if np.isfinite(hi) else 0.0)

    off_mean_pct = []
    for target in order:
        value = acc_ai_off.get(target, np.nan)
        if isinstance(value, (list, tuple, np.ndarray)):
            mean, _ = _ci95_t(value)
        else:
            mean = float(value)
        off_mean_pct.append(100.0 * mean if np.isfinite(mean) else np.nan)

    xtick_labels = []
    for target, baseline in zip(order, off_mean_pct):
        target_label = TARGET_LABELS.get(target, target.replace("_", " ").capitalize())
        baseline_label = label_fmt.format(baseline) if np.isfinite(baseline) else "n/a"
        xtick_labels.append(f"{target_label}\nBCI-only: {baseline_label}")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    point_color = MONKEY_COLORS.get(monkey, "#4C78A8")
    mean_color = "#1F2933"

    for xi, values_pp in zip(x, gain_points_pp):
        if values_pp.size == 0:
            continue
        offsets = np.array([0.0]) if values_pp.size == 1 else np.linspace(-0.18, 0.18, values_pp.size)
        ax.scatter(
            xi + offsets,
            values_pp,
            s=26,
            marker="o",
            facecolors="white",
            edgecolors=point_color,
            linewidths=0.9,
            alpha=0.95,
            zorder=3,
        )

    ax.errorbar(
        x,
        gain_mean_pp,
        yerr=[gain_lo_pp, gain_hi_pp],
        fmt="D",
        markersize=5,
        color=mean_color,
        markerfacecolor=point_color,
        markeredgecolor=mean_color,
        ecolor=mean_color,
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        linestyle="none",
        zorder=4,
    )
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center")
    ax.set_ylabel("Shared-control gain (percentage points)")
    ax.set_title(monkey)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    means = np.asarray(gain_mean_pp, dtype=float)
    loerr = np.asarray(gain_lo_pp, dtype=float)
    hierr = np.asarray(gain_hi_pp, dtype=float)
    lower = means - loerr
    upper = means + hierr
    finite_points = np.concatenate([v[np.isfinite(v)] for v in gain_points_pp if v.size > 0]) if gain_points_pp else np.array([])
    finite_lower = lower[np.isfinite(lower)]
    finite_upper = upper[np.isfinite(upper)]
    if finite_lower.size and finite_upper.size:
        y_min_ci = float(finite_lower.min())
        y_max_ci = float(finite_upper.max())
        if finite_points.size:
            y_min_ci = min(y_min_ci, float(finite_points.min()))
            y_max_ci = max(y_max_ci, float(finite_points.max()))
        y_range = max(1.0, y_max_ci - y_min_ci)
        pad = max(4.0, 0.12 * y_range)
        ax.set_ylim(min(0.0, y_min_ci - pad), max(10.0, y_max_ci + pad))

    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=point_color, markersize=5, label="Session"),
        Line2D([0], [0], marker="D", color=mean_color, markerfacecolor=point_color,
               markeredgecolor=mean_color, markersize=5, linestyle="none",
               label="Mean +/- 95% CI"),
    ], loc="upper left", frameon=False)
    fig.text(
        0.13,
        0.02,
        "Values below each target label indicate mean BCI-only success across sessions for that target.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.25",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_plot(fig, f"AI_gain_labels_{monkey}_{experiment}", "AI gain with baseline")
    plt.close(fig)


def plot_summary_scatter(all_points, base_dir, title="Shared-control gain versus baseline BCI performance"):
    """Panel C: target x session observations plus the pooled quadratic model prediction."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    session_data = _load_target_session_rows(base_dir)
    if session_data.empty:
        print("[plot_summary_scatter] Falling back to target-summary points; target-session rows are unavailable.")
        plot_points = pd.DataFrame([point for point in all_points if point["experiment"] in TASK_LABELS])
        if plot_points.empty:
            print("[plot_summary_scatter] No data points to plot.")
            return
        plot_points["task"] = plot_points["experiment"].map(TASK_LABELS).fillna(plot_points["experiment"])
    else:
        session_data = session_data[session_data["experiment"].isin(TASK_LABELS)].copy()
        session_data["task"] = session_data["task"].fillna(session_data["experiment"].map(TASK_LABELS))
        session_data["task"] = session_data["task"].replace(TASK_LABELS)
        plot_points = session_data.copy()
        plot_points["baseline_off_pct"] = plot_points["baseline_pct"]

    monkeys = ["Monkey 1", "Monkey 2"]
    monkeys = [monkey for monkey in monkeys if monkey in set(plot_points["monkey"])] + [
        monkey for monkey in sorted(set(plot_points["monkey"])) if monkey not in {"Monkey 1", "Monkey 2"}
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for monkey in monkeys:
        points = plot_points[plot_points["monkey"].eq(monkey)]
        if points.empty:
            continue
        ax.scatter(
            points["baseline_off_pct"],
            points["gain_pp"],
            s=10,
            alpha=0.22,
            linewidths=0,
            c=MONKEY_COLORS.get(monkey, "0.35"),
            marker="o",
            zorder=1,
        )

    required_cols = {"bci_only_success", "shared_control_gain", "x_centered", "x_centered2", "session_uid"}
    if not session_data.empty and required_cols.issubset(session_data.columns):
        fit = smf.ols(
            "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task)",
            data=session_data,
        ).fit(cov_type="cluster", cov_kwds={"groups": session_data["session_uid"]})
        observed_min = float(session_data["bci_only_success"].min())
        observed_max = float(session_data["bci_only_success"].max())
        x_grid = np.linspace(observed_min, observed_max, 200)
        x_mean = float(session_data["bci_only_success"].mean())
        group_grid = session_data[["monkey", "task"]].drop_duplicates()
        pred_rows = []
        for _, group in group_grid.iterrows():
            pred_rows.append(pd.DataFrame({
                "monkey": group["monkey"],
                "task": group["task"],
                "x_centered": x_grid - x_mean,
                "x_centered2": (x_grid - x_mean) ** 2,
            }))
        pred = pd.concat(pred_rows, ignore_index=True)
        pred["baseline_grid"] = np.tile(x_grid, len(group_grid))
        prediction = fit.get_prediction(pred).summary_frame(alpha=0.05)
        pred["fit"] = prediction["mean"].to_numpy(float)
        pred["ci_low"] = prediction["mean_ci_lower"].to_numpy(float)
        pred["ci_high"] = prediction["mean_ci_upper"].to_numpy(float)
        curve = (
            pred.groupby("baseline_grid", as_index=False)
            .agg(fit=("fit", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"))
        )
        ax.fill_between(
            100.0 * curve["baseline_grid"],
            100.0 * curve["ci_low"],
            100.0 * curve["ci_high"],
            color="0.55",
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            100.0 * curve["baseline_grid"],
            100.0 * curve["fit"],
            color="0.20",
            lw=2.0,
            zorder=3,
        )
        quadratic_beta, quadratic_p = _load_common_model_stats()
        if not np.isfinite(quadratic_beta):
            quadratic_beta = float(fit.params.get("x_centered2", np.nan))
            quadratic_p = float(fit.pvalues.get("x_centered2", np.nan))
    else:
        x = plot_points["baseline_off_pct"].to_numpy(float)
        y = plot_points["gain_pp"].to_numpy(float)
        coefs = np.polyfit(x, y, deg=2)
        xs = np.linspace(max(0, np.nanmin(x) - 5), min(100, np.nanmax(x) + 5), 200)
        ax.plot(xs, np.polyval(coefs, xs), color="0.20", lw=2.0)
        model = sm.OLS(y, sm.add_constant(np.column_stack([x ** 2, x]))).fit()
        quadratic_beta, quadratic_p = float(model.params[1]), float(model.pvalues[1])

    binned = plot_points.copy()
    bin_specs = [
        ("0-20", 10.0, binned["baseline_off_pct"].ge(0) & binned["baseline_off_pct"].le(20)),
        ("20-40", 30.0, binned["baseline_off_pct"].gt(20) & binned["baseline_off_pct"].le(40)),
        ("40-60", 50.0, binned["baseline_off_pct"].gt(40) & binned["baseline_off_pct"].le(60)),
        ("60-80", 70.0, binned["baseline_off_pct"].gt(60) & binned["baseline_off_pct"].le(80)),
        ("80-100", 90.0, binned["baseline_off_pct"].gt(80) & binned["baseline_off_pct"].le(100)),
    ]
    summary_rows = []
    for label, center, mask in bin_specs:
        values = binned.loc[mask, "gain_pp"].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean = float(values.mean())
        if values.size > 1:
            from scipy.stats import t

            sem = float(values.std(ddof=1) / np.sqrt(values.size))
            ci = float(t.ppf(0.975, values.size - 1) * sem)
        else:
            ci = 0.0
        summary_rows.append({"label": label, "center": center, "mean": mean, "ci": ci, "n": int(values.size)})

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        ax.errorbar(
            summary["center"],
            summary["mean"],
            yerr=summary["ci"],
            fmt="o",
            markersize=7.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            linestyle="none",
            zorder=4,
        )

    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(-40, 105)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(-20, 101, 20))
    ax.set_xlabel("Baseline BCI-only success (%)")
    ax.set_ylabel("Shared-control gain (percentage points)")
    ax.text(
        0.03,
        0.97,
        f"Session-clustered p={quadratic_p:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.20",
    )
    ax.grid(color="0.9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if summary_rows:
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        for row in summary_rows:
            y = row["mean"] + row["ci"] + 0.035 * yspan
            ax.text(
                row["center"],
                y,
                f"n={row['n']}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.20",
                zorder=5,
            )

    fig.tight_layout()
    save_plot(fig, "Performance-dependent AI gains", "AI gain with baseline")
    plt.close(fig)

    print(f"Figure 3C plotted {len(plot_points)} target x session observations.")
    if not session_data.empty:
        print(f"Model statistics are from {len(session_data)} target x session observations.")
    print(f"Quadratic term: beta={quadratic_beta:.6g}, session-clustered p={quadratic_p:.6g}")
