"""Make task- and monkey-stratified versions of the original Figure 3C plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "tables" / "Figure 3C original session audit" / "figure3c_original_plotted_target_points.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "Extended Data Figure 3" / "Figure 3C stratified"
TABLE_DIR = PROJECT_ROOT / "tables" / "Figure 3C stratified"

TASK_LABELS = {
    "AI Obstacle": "Fixed obstacle",
    "AI Appearing Obstacle": "Appearing obstacle",
    "AI Appearing Obstacle 2": "Follow-up appearing",
}
MARKERS = {
    "AI Obstacle": "o",
    "AI Appearing Obstacle": "s",
    "AI Appearing Obstacle 2": "^",
}
COLORS = {
    "Monkey 1": "#E69F00",
    "Monkey 2": "#0072B2",
}
TARGET_MARKERS = {
    "left": "o",
    "slight_left": "s",
    "straight": "^",
    "slight_right": "D",
    "right": "P",
}


def load_points(input_csv: Path = INPUT_CSV) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    data = data[data["audit"].eq("mapped_manuscript_source_paths")].copy()
    data = data[np.isfinite(data["baseline_off_pct"]) & np.isfinite(data["gain_pp_mean_of_session_gains"])]
    data["task_label"] = data["experiment"].map(TASK_LABELS).fillna(data["experiment"])
    data = data.rename(columns={"gain_pp_mean_of_session_gains": "gain_pp"})
    return data


def fit_quadratic(sub: pd.DataFrame) -> dict:
    x = sub["baseline_off_pct"].to_numpy(float)
    y = sub["gain_pp"].to_numpy(float)
    if len(sub) < 4 or len(np.unique(x)) < 3:
        return {
            "n": len(sub),
            "quadratic_beta": np.nan,
            "linear_beta": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "model_p": np.nan,
            "quadratic_p": np.nan,
        }
    design = sm.add_constant(np.column_stack([x ** 2, x]))
    fit = sm.OLS(y, design).fit()
    return {
        "n": len(sub),
        "quadratic_beta": float(fit.params[1]),
        "linear_beta": float(fit.params[2]),
        "intercept": float(fit.params[0]),
        "r2": float(fit.rsquared),
        "model_p": float(fit.f_pvalue),
        "quadratic_p": float(fit.pvalues[1]),
    }


def _draw_fit(ax, sub: pd.DataFrame, color: str = "0.2") -> dict:
    fit = fit_quadratic(sub)
    if np.isfinite(fit["quadratic_beta"]):
        x = sub["baseline_off_pct"].to_numpy(float)
        xs = np.linspace(max(0, np.nanmin(x) - 5), min(100, np.nanmax(x) + 5), 200)
        ys = fit["quadratic_beta"] * xs ** 2 + fit["linear_beta"] * xs + fit["intercept"]
        ax.plot(xs, ys, color=color, linewidth=1.8)
        ax.text(
            0.03,
            0.95,
            f"R2={fit['r2']:.2f}, p={fit['model_p']:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=color,
        )
    return fit


def _style_axes(ax) -> None:
    ax.axhline(0, color="0.65", linewidth=0.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 90)
    ax.grid(color="0.9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _safe_slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def plot_single_strata(data: pd.DataFrame) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for task, sub in data.groupby("experiment", sort=False):
        label = TASK_LABELS.get(task, task)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        for monkey, monkey_sub in sub.groupby("monkey"):
            ax.scatter(
                monkey_sub["baseline_off_pct"],
                monkey_sub["gain_pp"],
                color=COLORS.get(monkey, "0.35"),
                edgecolor="black",
                linewidth=0.5,
                s=62,
                alpha=0.9,
                label=monkey,
            )
        _draw_fit(ax, sub)
        _style_axes(ax)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("No-AI success per target (%)")
        ax.set_ylabel("AI gain per target (pp)")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        for ext in ("svg", "png", "pdf"):
            path = OUT_DIR / f"figure3c_task_{_safe_slug(label)}.{ext}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)

    for monkey, sub in data.groupby("monkey", sort=False):
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        for task, task_sub in sub.groupby("experiment"):
            ax.scatter(
                task_sub["baseline_off_pct"],
                task_sub["gain_pp"],
                marker=MARKERS.get(task, "o"),
                color=COLORS.get(monkey, "0.35"),
                edgecolor="black",
                linewidth=0.5,
                s=62,
                alpha=0.9,
                label=TASK_LABELS.get(task, task),
            )
        _draw_fit(ax, sub)
        _style_axes(ax)
        ax.set_title(monkey, fontsize=10)
        ax.set_xlabel("No-AI success per target (%)")
        ax.set_ylabel("AI gain per target (pp)")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        fig.tight_layout()
        for ext in ("svg", "png", "pdf"):
            path = OUT_DIR / f"figure3c_monkey_{_safe_slug(monkey)}.{ext}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)
    return paths


def plot_by_task(data: pd.DataFrame) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in TASK_LABELS if task in set(data["experiment"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), sharex=True, sharey=True, squeeze=False)
    rows = []
    for ax, task in zip(axes[0], tasks):
        sub = data[data["experiment"].eq(task)]
        for monkey, monkey_sub in sub.groupby("monkey"):
            ax.scatter(
                monkey_sub["baseline_off_pct"],
                monkey_sub["gain_pp"],
                color=COLORS.get(monkey, "0.35"),
                edgecolor="black",
                linewidth=0.5,
                s=58,
                alpha=0.9,
                label=monkey,
            )
        fit = _draw_fit(ax, sub)
        rows.append({"stratification": "task", "group": TASK_LABELS.get(task, task), **fit})
        ax.set_title(TASK_LABELS.get(task, task), fontsize=10)
        ax.set_xlabel("No-AI success per target (%)")
        _style_axes(ax)
    axes[0, 0].set_ylabel("AI gain per target (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT_DIR / f"figure3c_stratified_by_task.{ext}", dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_by_monkey(data: pd.DataFrame) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monkeys = list(data["monkey"].drop_duplicates())
    fig, axes = plt.subplots(1, len(monkeys), figsize=(4.5 * len(monkeys), 3.7), sharex=True, sharey=True, squeeze=False)
    rows = []
    for ax, monkey in zip(axes[0], monkeys):
        sub = data[data["monkey"].eq(monkey)]
        for task, task_sub in sub.groupby("experiment"):
            ax.scatter(
                task_sub["baseline_off_pct"],
                task_sub["gain_pp"],
                marker=MARKERS.get(task, "o"),
                color=COLORS.get(monkey, "0.35"),
                edgecolor="black",
                linewidth=0.5,
                s=62,
                alpha=0.9,
                label=TASK_LABELS.get(task, task),
            )
        fit = _draw_fit(ax, sub)
        rows.append({"stratification": "monkey", "group": monkey, **fit})
        ax.set_title(monkey, fontsize=10)
        ax.set_xlabel("No-AI success per target (%)")
        _style_axes(ax)
    axes[0, 0].set_ylabel("AI gain per target (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT_DIR / f"figure3c_stratified_by_monkey.{ext}", dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows)


def write_report(data: pd.DataFrame, fits: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    report = TABLE_DIR / "figure3c_stratified_plot_report.md"
    counts = (
        data.groupby(["monkey", "task_label"], dropna=False)
        .agg(n_points=("target", "size"), min_baseline=("baseline_off_pct", "min"), median_baseline=("baseline_off_pct", "median"), max_baseline=("baseline_off_pct", "max"))
        .reset_index()
    )
    lines = [
        "# Figure 3C Stratified Plots",
        "",
        "These plots use the same target-summary points as the original Figure 3C provenance audit: one point per monkey x task x target from `mapped_manuscript_source_paths`.",
        "",
        "## Point Counts",
        "",
        counts.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Quadratic Fits",
        "",
        fits.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Outputs",
        "",
        str(OUT_DIR / "figure3c_stratified_by_task.svg"),
        str(OUT_DIR / "figure3c_stratified_by_monkey.svg"),
        str(OUT_DIR / "figure3c_task_fixed_obstacle.svg"),
        str(OUT_DIR / "figure3c_task_appearing_obstacle.svg"),
        str(OUT_DIR / "figure3c_task_follow_up_appearing.svg"),
        str(OUT_DIR / "figure3c_monkey_monkey_1.svg"),
        str(OUT_DIR / "figure3c_monkey_monkey_2.svg"),
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(input_csv: Path = INPUT_CSV) -> dict[str, Path]:
    data = load_points(input_csv)
    if data.empty:
        raise RuntimeError(f"No mapped manuscript Figure 3C points found in {input_csv}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(TABLE_DIR / "figure3c_stratified_points.csv", index=False)
    fits = pd.concat([plot_by_task(data), plot_by_monkey(data)], ignore_index=True)
    plot_single_strata(data)
    fits.to_csv(TABLE_DIR / "figure3c_stratified_quadratic_fits.csv", index=False)
    report = write_report(data, fits)
    return {
        "points": TABLE_DIR / "figure3c_stratified_points.csv",
        "fits": TABLE_DIR / "figure3c_stratified_quadratic_fits.csv",
        "by_task_svg": OUT_DIR / "figure3c_stratified_by_task.svg",
        "by_monkey_svg": OUT_DIR / "figure3c_stratified_by_monkey.svg",
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    args = parser.parse_args()
    for name, path in run(args.input_csv).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
