import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t, wilcoxon


mpl.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "Variables" / "grand_summary.pkl"
OUT_DIR = ROOT / "figures" / "Figure 4 paired dots"

BCI_COLOR = "#CC79A7"
SHARED_COLOR = "#009E73"
RAW_POINT_SIZE = 30
MEAN_MARKER_SIZE = 82


def load_summary():
    with open(SUMMARY_PATH, "rb") as f:
        return pickle.load(f)


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


def paired_wilcoxon(on, off, alternative):
    on = np.asarray(on, dtype=float)
    off = np.asarray(off, dtype=float)
    ok = np.isfinite(on) & np.isfinite(off)
    on = on[ok]
    off = off[ok]
    if off.size < 2:
        return np.nan
    if np.allclose(on, off, equal_nan=False):
        return 1.0
    try:
        return float(wilcoxon(on, off, alternative=alternative, zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def mean_interval(values, interval="sem"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, np.nan
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    if interval == "ci95":
        return mean, sem * float(t.ppf(0.975, values.size - 1))
    return mean, sem


def horizontal_jitter(n, width):
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([-0.35 * width], dtype=float)

    offsets = np.arange(n, dtype=float) - (n - 1) / 2.0
    offsets[offsets >= 0] += 0.5
    offsets[offsets < 0] -= 0.5
    return offsets * (width / np.max(np.abs(offsets)))


def remove_errorbar_container(container):
    for item in container.lines:
        if isinstance(item, tuple):
            for artist in item:
                artist.remove()
        elif item is not None:
            item.remove()


def paired_dot_panel(
    categories,
    labels,
    vals_off,
    vals_on,
    ylabel,
    title,
    output_name,
    alternatives,
    ylim=None,
    interval="sem",
    ax=None,
    show_legend=True,
):
    x = np.arange(len(categories), dtype=float)
    offset = 0.16
    jitter_width = 0.055

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.4, 1.45 * len(categories)), 4.2))
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    all_finite = []
    pvals = []
    for i, cat in enumerate(categories):
        off = np.asarray(vals_off[cat], dtype=float)
        on = np.asarray(vals_on[cat], dtype=float)
        ok = np.isfinite(off) & np.isfinite(on)
        off = off[ok]
        on = on[ok]
        n = off.size
        if n == 0:
            pvals.append(np.nan)
            continue

        jitter = horizontal_jitter(n, jitter_width)
        x_off = np.full(n, x[i] - offset) + jitter
        x_on = np.full(n, x[i] + offset) + jitter

        ax.scatter(
            x_off,
            off,
            s=RAW_POINT_SIZE,
            marker="o",
            color=BCI_COLOR,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.88,
            label="BCI-only" if i == 0 else None,
            zorder=3,
        )
        ax.scatter(
            x_on,
            on,
            s=RAW_POINT_SIZE,
            marker="o",
            color=SHARED_COLOR,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.88,
            label="Shared-control" if i == 0 else None,
            zorder=3,
        )

        for xpos, values, color in ((x[i] - offset, off, BCI_COLOR), (x[i] + offset, on, SHARED_COLOR)):
            mean, err = mean_interval(values, interval=interval)
            if np.isfinite(err):
                ax.errorbar(
                    [xpos],
                    [mean],
                    yerr=[err],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.1,
                    capsize=4,
                    capthick=1.1,
                    zorder=5,
                )
            ax.scatter(
                [xpos],
                [mean],
                s=MEAN_MARKER_SIZE,
                marker="D",
                color=color,
                edgecolor="black",
                linewidth=0.9,
                zorder=6,
            )

        all_finite.extend(off[np.isfinite(off)])
        all_finite.extend(on[np.isfinite(on)])
        pvals.append(paired_wilcoxon(on, off, alternatives.get(cat, "two-sided")))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        mean_sem_handle = ax.errorbar(
            [0],
            [0],
            yerr=[[0.1], [0.1]],
            fmt="D",
            color="black",
            ecolor="black",
            elinewidth=1.1,
            capsize=4,
            capthick=1.1,
            markersize=np.sqrt(MEAN_MARKER_SIZE),
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            label="Mean ± s.e.m.",
        )
        remove_errorbar_container(mean_sem_handle)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=np.sqrt(RAW_POINT_SIZE),
                markerfacecolor=BCI_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label="BCI-only individual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=np.sqrt(RAW_POINT_SIZE),
                markerfacecolor=SHARED_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label="Shared-control individual",
            ),
            mean_sem_handle,
        ]
        ax.legend(handles=handles, frameon=False)

    if ylim is None:
        ymax = max(all_finite) * 1.28 if all_finite else 1.0
        ylim = (0, max(ymax, 1.0))
    ax.set_ylim(*ylim)

    h = (ylim[1] - ylim[0]) * 0.035
    for i, (cat, p) in enumerate(zip(categories, pvals)):
        stars = p_to_stars(p)
        if not stars:
            continue
        off = np.asarray(vals_off[cat], dtype=float)
        on = np.asarray(vals_on[cat], dtype=float)
        y_base = np.nanmax(np.concatenate([off, on]))
        y = min(y_base + h, ylim[1] - 3 * h)
        x1 = x[i] - offset
        x2 = x[i] + offset
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
        ax.text((x1 + x2) / 2, y + h * 1.08, stars, ha="center", va="bottom", fontsize=10)

    if output_name:
        if standalone:
            fig.tight_layout()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / output_name
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        if standalone:
            plt.close(fig)
        print(f"Saved: {out_path}")

    return ax


def paired_failure_values(rows, categories, compute):
    units = {}
    for row in rows:
        condition = row.get("condition")
        if condition not in ("AI", "No AI"):
            continue
        units[(row.get("monkey"), row.get("experiment"), condition)] = row

    pair_keys = sorted({
        (monkey, experiment)
        for monkey, experiment, _ in units
        if (monkey, experiment, "AI") in units and (monkey, experiment, "No AI") in units
    })

    vals_off = {cat: [] for cat in categories}
    vals_on = {cat: [] for cat in categories}
    for monkey, experiment in pair_keys:
        off = compute(units[(monkey, experiment, "No AI")])
        on = compute(units[(monkey, experiment, "AI")])
        for cat in categories:
            vals_off[cat].append(off[cat] * 100.0)
            vals_on[cat].append(on[cat] * 100.0)
    return vals_off, vals_on


def f(row, key):
    return float(row.get(key, 0.0) or 0.0)


def make_figure4a(summary, ax=None, show_legend=True, output_name="figure4A_paired_execution_failure_categories.svg"):
    categories = ["blocked", "overshoot", "proximity", "dwell"]
    labels = ["Blocked", "Target\novershoot", "Insufficient\nproximity", "Insufficient\ndwell"]

    def compute(row):
        return {
            "blocked": f(row, "stuck_obstacle"),
            "overshoot": f(row, "overshoot"),
            "proximity": f(row, "not_close_to_true_z"),
            "dwell": f(row, "not_long_enough"),
        }

    vals_off, vals_on = paired_failure_values(summary["failure_props_per_unit"], categories, compute)
    paired_dot_panel(
        categories,
        labels,
        vals_off,
        vals_on,
        ylabel="Percentage of all trials (%)",
        title="Figure 4A - execution failure categories",
        output_name=output_name,
        alternatives={cat: "less" for cat in categories},
        ax=ax,
        show_legend=show_legend,
    )


def make_figure4b(summary, ax=None, show_legend=True, output_name="figure4B_paired_overall_outcome_classes.svg"):
    categories = ["execution", "decoder_direction", "other"]
    labels = ["Execution\nerrors", "Decoder-direction\nerrors", "Other"]

    def compute(row):
        execution = (
            f(row, "stuck_obstacle")
            + f(row, "overshoot")
            + f(row, "not_long_enough")
            + f(row, "not_close_to_true_z")
        )
        decoder_direction = (
            f(row, "wrong_target_choice")
            + f(row, "neighbor_choice")
            + f(row, "ambiguous_choice")
            + f(row, "ai_bci_failure_target")
            + f(row, "wrong_target_other")
            + f(row, "ambiguous_choice_unknown")
        )
        return {
            "execution": execution,
            "decoder_direction": decoder_direction,
            "other": f(row, "other"),
        }

    vals_off, vals_on = paired_failure_values(summary["failure_props_per_unit"], categories, compute)
    paired_dot_panel(
        categories,
        labels,
        vals_off,
        vals_on,
        ylabel="Percentage of all trials (%)",
        title="Figure 4B - overall outcome classes",
        output_name=output_name,
        alternatives={"execution": "less", "decoder_direction": "two-sided", "other": "two-sided"},
        ax=ax,
        show_legend=show_legend,
    )


def make_figure4c(summary, ax=None, show_legend=True, output_name="figure4C_paired_success_by_choice_state.svg"):
    states = ["correct_choice", "neighbor_choice", "ambiguous_choice", "wrong_choice"]
    labels = ["Correct\nchoice", "Adjacent\nchoice", "Ambiguous\nchoice", "Incorrect\nchoice"]
    vals_off = {state: [] for state in states}
    vals_on = {state: [] for state in states}

    for state in states:
        state_rows = [row for row in summary["success_state_per_unit"] if row.get("state") == state]
        for row in state_rows:
            vals_off[state].append(float(row.get("p_success_off", np.nan)))
            vals_on[state].append(float(row.get("p_success_on", np.nan)))

    paired_dot_panel(
        states,
        labels,
        vals_off,
        vals_on,
        ylabel="P(success | choice state)",
        title="Figure 4C - success by choice state",
        output_name=output_name,
        alternatives={state: "greater" for state in states},
        ylim=(0, 1.05),
        ax=ax,
        show_legend=show_legend,
    )


def make_figure4_combined(summary):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 12.2))
    make_figure4a(summary, ax=axes[0], show_legend=True, output_name=None)
    make_figure4b(summary, ax=axes[1], show_legend=False, output_name=None)
    make_figure4c(summary, ax=axes[2], show_legend=False, output_name=None)
    fig.tight_layout(h_pad=2.1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "figure4_combined_vertical_paired_dots.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    summary = load_summary()
    make_figure4a(summary)
    make_figure4b(summary)
    make_figure4c(summary)
    make_figure4_combined(summary)


if __name__ == "__main__":
    main()
