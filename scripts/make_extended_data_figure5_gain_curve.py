"""
Create the Extended Data Figure 2 gain-curve supplement.

The figure is built from target x session observations:
    x = BCI-only success
    y = shared-control success - BCI-only success

Outputs:
    figures/Extended Data Figure 2/Gain curve extension/
    tables/Extended Data Table 5/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from run_gain_curve_pooling_audit import (
    BASELINE_BIN_LABELS,
    DEFAULT_BASE_DIR,
    DEFAULT_DATASETS,
    PROJECT_ROOT,
    collect_target_session_rows,
    coverage_tables,
    model_comparisons,
    separate_quadratic_fits,
)


def _load_figure_dir() -> Path:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        figure_dir = (config.get("gain_curve_extension") or {}).get("figure_dir")
        if figure_dir:
            return PROJECT_ROOT / figure_dir
    return PROJECT_ROOT / "figures" / "Extended Data Figure 2" / "Gain curve extension"


FIGURE_DIR = _load_figure_dir()
TABLE_DIR = PROJECT_ROOT / "tables" / "Extended Data Table 5"
AUDIT_DIR = PROJECT_ROOT / "tables" / "Gain curve pooling audit"

PANEL_ROWS = [
    ("Monkey 1", "Fixed Obstacle", "Monkey 1 Fixed Obstacle"),
    ("Monkey 2", "Fixed Obstacle", "Monkey 2 Fixed Obstacle"),
    ("Monkey 1", "Appearing Obstacle", "Monkey 1 Appearing Obstacle"),
    ("Monkey 2", "Appearing Obstacle", "Monkey 2 Appearing Obstacle"),
    ("Monkey 1", "Appearing Obstacle follow-up", "Monkey 1 Appearing Obstacle follow-up"),
]

MONKEY_COLORS = {
    "Monkey 1": "#3B6EA8",
    "Monkey 2": "#C85A3D",
}

TASK_COLORS = {
    ("Monkey 1", "Appearing Obstacle"): "#3B6EA8",
    ("Monkey 1", "Appearing Obstacle follow-up"): "#3B6EA8",
    ("Monkey 1", "Fixed Obstacle"): "#3B6EA8",
    ("Monkey 2", "Appearing Obstacle"): "#C85A3D",
    ("Monkey 2", "Fixed Obstacle"): "#C85A3D",
}


def _fit_quadratic(sub: pd.DataFrame):
    return smf.ols("shared_control_gain ~ x_centered + x_centered2", data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["session_uid"]},
    )


def _prediction_frame(fit, x_grid: np.ndarray, x_mean: float) -> pd.DataFrame:
    pred_data = pd.DataFrame({
        "x_centered": x_grid - x_mean,
        "x_centered2": (x_grid - x_mean) ** 2,
    })
    design = patsy.build_design_matrices(
        [fit.model.data.design_info],
        pred_data,
        return_type="dataframe",
    )[0]
    pred = np.asarray(design @ fit.params, dtype=float)
    cov = np.asarray(fit.cov_params(), dtype=float)
    se = np.sqrt(np.maximum(np.sum((np.asarray(design) @ cov) * np.asarray(design), axis=1), 0.0))
    return pd.DataFrame({
        "baseline_success": x_grid,
        "fit": pred,
        "ci_low": pred - 1.96 * se,
        "ci_high": pred + 1.96 * se,
    })


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0:
        return "0.00"
    if value >= 0.001:
        return f"{value:.3f}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f}×10$^{{{exponent}}}$"


def _format_beta(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    magnitude = abs(value)
    if magnitude == 0:
        return "0.00"
    if magnitude < 1:
        return f"{value:.3f}"
    if magnitude < 10:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _annotate_fit(ax, n: int, beta: float, p_value: float) -> None:
    text = f"n = {n}\n" + rf"$\beta_{{\mathrm{{quad}}}}$ = {_format_beta(beta)}" + f"\np={_format_p(p_value)}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        linespacing=1.25,
    )


def _draw_scatter_fit(
    ax,
    sub: pd.DataFrame,
    x_mean: float,
    color: str,
    title: str,
    fit_row: pd.Series,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    ax.axhline(0.0, color="0.72", linewidth=0.8, zorder=1)
    ax.scatter(
        100.0 * sub["bci_only_success"],
        100.0 * sub["shared_control_gain"],
        s=18,
        color=color,
        alpha=0.72,
        linewidth=0.25,
        edgecolor="white",
        zorder=3,
    )
    fit = _fit_quadratic(sub)
    x_min = float(sub["bci_only_success"].min())
    x_max = float(sub["bci_only_success"].max())
    x_grid = np.linspace(x_min, x_max, 160)
    pred = _prediction_frame(fit, x_grid, x_mean)
    ax.fill_between(
        100.0 * pred["baseline_success"].to_numpy(dtype=float),
        100.0 * pred["ci_low"].to_numpy(dtype=float),
        100.0 * pred["ci_high"].to_numpy(dtype=float),
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        100.0 * pred["baseline_success"],
        100.0 * pred["fit"],
        color="0.08",
        linewidth=1.4,
        zorder=4,
    )
    _annotate_fit(
        ax,
        int(fit_row["n"]),
        float(fit_row["quadratic_beta"]),
        float(fit_row["quadratic_p_clustered"]),
    )
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-35, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([-25, 0, 25, 50, 75, 100])
    ax.grid(color="0.90", linewidth=0.7, zorder=0)
    if show_xlabel:
        ax.set_xlabel("Baseline BCI-only success (%)", fontsize=8.5)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Shared-control gain (pp)", fontsize=8.5)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _coverage_matrix(data: pd.DataFrame) -> pd.DataFrame:
    count_data = data.copy()
    count_data["dataset_label"] = ""
    for monkey, task, label in PANEL_ROWS:
        mask = count_data["monkey"].eq(monkey) & count_data["task"].eq(task)
        count_data.loc[mask, "dataset_label"] = label
    counts = (
        count_data.groupby(["dataset_label", "baseline_bin"], observed=False)
        .size()
        .unstack("baseline_bin", fill_value=0)
        .reindex([label for _, _, label in PANEL_ROWS])
        .reindex(columns=BASELINE_BIN_LABELS, fill_value=0)
    )
    return counts.astype(int)


def _ordered_separate_fits(separate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for monkey in ["Monkey 1", "Monkey 2"]:
        sub = separate[separate["grouping"].eq("monkey") & separate["monkey"].eq(monkey)]
        if not sub.empty:
            rows.append(sub.iloc[0])
    for monkey, task, _label in PANEL_ROWS:
        sub = separate[
            separate["grouping"].eq("monkey x task")
            & separate["monkey"].eq(monkey)
            & separate["task"].eq(task)
        ]
        if not sub.empty:
            rows.append(sub.iloc[0])
    if not rows:
        return separate
    return pd.DataFrame(rows).reset_index(drop=True)


def _ordered_baseline_distribution(dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for monkey, task, _label in PANEL_ROWS:
        sub = dist[dist["monkey"].eq(monkey) & dist["task"].eq(task)]
        if not sub.empty:
            rows.append(sub.iloc[0])
    if not rows:
        return dist
    return pd.DataFrame(rows).reset_index(drop=True)


def make_figure(data: pd.DataFrame, separate: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x_mean = float(data["bci_only_success"].mean())

    fig = plt.figure(figsize=(10.8, 10.6), dpi=300)
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.0, 0.9, 0.9, 0.92],
        hspace=0.58,
        wspace=0.70,
    )

    monkey_axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
    ]
    for ax, monkey in zip(monkey_axes, ["Monkey 1", "Monkey 2"]):
        sub = data[data["monkey"].eq(monkey)].copy()
        fit_row = separate[(separate["grouping"].eq("monkey")) & (separate["monkey"].eq(monkey))].iloc[0]
        _draw_scatter_fit(
            ax,
            sub,
            x_mean,
            MONKEY_COLORS[monkey],
            monkey,
            fit_row,
            show_ylabel=(monkey == "Monkey 1"),
            show_xlabel=True,
        )

    task_slots = [
        (1, 0, 3),
        (1, 3, 6),
        (2, 0, 2),
        (2, 2, 4),
        (2, 4, 6),
    ]
    for (monkey, task, label), (row, start, stop) in zip(PANEL_ROWS, task_slots):
        ax = fig.add_subplot(gs[row, start:stop])
        sub = data[data["monkey"].eq(monkey) & data["task"].eq(task)].copy()
        fit_row = separate[
            separate["grouping"].eq("monkey x task")
            & separate["monkey"].eq(monkey)
            & separate["task"].eq(task)
        ].iloc[0]
        _draw_scatter_fit(
            ax,
            sub,
            x_mean,
            TASK_COLORS[(monkey, task)],
            label,
            fit_row,
            show_ylabel=(row == 1 and start == 0),
            show_xlabel=True,
        )

    ax_hm = fig.add_subplot(gs[3, 1:5])
    counts = _coverage_matrix(data)
    cmap = LinearSegmentedColormap.from_list("count_teal", ["#F7FBF9", "#7BB7A5", "#1F6F64"])
    ax_hm.imshow(counts.to_numpy(dtype=float), cmap=cmap, aspect="auto")
    ax_hm.set_xticks(np.arange(len(BASELINE_BIN_LABELS)))
    ax_hm.set_xticklabels(BASELINE_BIN_LABELS, fontsize=8)
    ax_hm.set_yticks(np.arange(len(counts.index)))
    ax_hm.set_yticklabels(counts.index, fontsize=8)
    ax_hm.set_xlabel("Baseline BCI-only success bin", fontsize=8.5)
    ax_hm.set_ylabel("Dataset", fontsize=8.5)
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            value = int(counts.iat[row, col])
            ax_hm.text(col, row, str(value), ha="center", va="center", fontsize=8.3, color="black")
    ax_hm.tick_params(length=0)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    fig.text(0.018, 0.975, "a", fontsize=12, fontweight="bold")
    fig.text(0.018, 0.725, "b", fontsize=12, fontweight="bold")
    fig.text(0.018, 0.298, "c", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.95, top=0.955, bottom=0.06)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(FIGURE_DIR / f"extended_data_figure5_gain_curve_extension.{ext}", bbox_inches="tight")
    plt.close(fig)
    return FIGURE_DIR / "extended_data_figure5_gain_curve_extension.svg"


def extended_cross_validated_mse(data: pd.DataFrame) -> pd.DataFrame:
    formulas = {
        "common_quadratic": "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task)",
        "monkey_specific": "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task) + C(monkey):x_centered + C(monkey):x_centered2",
        "task_specific": "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task) + C(task):x_centered + C(task):x_centered2",
        "monkey_task_specific": "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task) + C(monkey):x_centered + C(monkey):x_centered2 + C(task):x_centered + C(task):x_centered2",
    }
    groups = data["session_uid"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return pd.DataFrame()
    rows = []
    for name, formula in formulas.items():
        fold_mse = []
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(data, groups=groups):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            try:
                fit = smf.ols(formula, data=train).fit()
                pred = fit.predict(test)
                fold_mse.append(mean_squared_error(test["shared_control_gain"], pred))
            except Exception:
                fold_mse.append(np.nan)
        rows.append({
            "model": name,
            "n_folds": n_splits,
            "mean_cv_mse": float(np.nanmean(fold_mse)),
            "median_cv_mse": float(np.nanmedian(fold_mse)),
        })
    return pd.DataFrame(rows)


def make_extended_table(fits: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    names = {
        "common_quadratic_monkey_task_intercepts": ("Common-shape", "—", "common_quadratic"),
        "monkey_specific_quadratic": ("Monkey-specific", "Monkey", "monkey_specific"),
        "task_specific_quadratic": ("Task-specific", "Task", "task_specific"),
        "full_monkey_task_shape": ("Monkey × task-specific", "Monkey × task", "monkey_task_specific"),
    }
    rows = []
    cv_lookup = cv.set_index("model")["mean_cv_mse"].to_dict() if not cv.empty else {}
    for model_name in names:
        row = fits[fits["model"].eq(model_name)].iloc[0]
        label, shape, cv_key = names[model_name]
        rows.append({
            "Model": label,
            "Shape allowed to vary by": shape,
            "df": int(row["df_model"]),
            "AIC": f"{float(row['aic']):.2f}",
            "BIC": f"{float(row['bic']):.2f}",
            "R²": f"{float(row['r2']):.3f}",
            "LR test vs common": (
                "—"
                if not np.isfinite(row["lr_vs_common_p"])
                else f"p={_format_p(float(row['lr_vs_common_p']))}"
            ),
            "Mean CV MSE": (
                "—"
                if cv_key is None or not np.isfinite(float(cv_lookup.get(cv_key, np.nan)))
                else f"{float(cv_lookup[cv_key]):.5f}"
            ),
        })
    return pd.DataFrame(rows)


def write_caption_files(data: pd.DataFrame, figure_path: Path, table: pd.DataFrame) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    figure_caption = TABLE_DIR / "extended_data_figure5_gain_curve_caption.md"
    table_caption = TABLE_DIR / "extended_data_table5_model_comparison_caption.md"

    def table_text(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```csv\n" + df.to_csv(index=False) + "```"

    figure_caption.write_text(
        "\n".join([
            "# Extended Data Figure 2 | Group-specific gain curves and baseline-performance coverage",
            "",
            "Each point is one target in one session. The x-axis shows baseline BCI-only success and the y-axis shows shared-control gain, defined as shared-control success minus BCI-only success; pp denotes percentage points. Quadratic fits were estimated from the same target × session observations with session-clustered standard errors; shaded bands show 95% confidence intervals derived from the session-clustered covariance matrix. Fit lines and confidence bands are drawn only over the observed baseline-performance range for the corresponding monkey or monkey × task dataset, while all scatter panels retain common 0-100% x-axis limits. Panel a shows separate fits by monkey. Panel b shows separate fits for each monkey × task dataset. Panel c shows the number of observations in each baseline-performance bin.",
            "",
            f"Source data: {len(data)} target × session observations from {data['session_uid'].nunique()} sessions.",
            f"Figure file: `{figure_path}`",
        ]) + "\n",
        encoding="utf-8-sig",
    )
    table_caption.write_text(
        "\n".join([
            "# Extended Data Table 5 | Comparison of shared and group-specific models relating baseline BCI performance to shared-control gain",
            "",
            f"Models were fitted to {len(data)} target × session observations from {data['session_uid'].nunique()} sessions. The common-shape model included shared linear and quadratic effects of baseline BCI performance with monkey and task intercepts. Monkey-specific and task-specific models additionally allowed the linear and quadratic terms to vary by monkey or task, respectively. Models were compared with likelihood-ratio tests relative to the common-shape model, AIC, BIC and grouped cross-validation by session.",
            "",
            table_text(table),
        ]) + "\n",
        encoding="utf-8-sig",
    )
    return {"figure_caption": figure_caption, "table_caption": table_caption}


def run(base_dir: str = DEFAULT_BASE_DIR) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = DEFAULT_DATASETS
    if base_dir is None or datasets is None:
        from run_gain_curve_pooling_audit import load_gain_curve_config

        config_base_dir, config_datasets = load_gain_curve_config()
        if base_dir is None:
            base_dir = config_base_dir
        if datasets is None:
            datasets = config_datasets

    data = collect_target_session_rows(base_dir, datasets)
    if data.empty:
        raise RuntimeError("No target-session rows were collected.")

    coverage, dist = coverage_tables(data)
    fits = model_comparisons(data)
    cv = extended_cross_validated_mse(data)
    separate = _ordered_separate_fits(separate_quadratic_fits(data))
    figure_path = make_figure(data, separate)
    counts = _coverage_matrix(data)
    ext_table = make_extended_table(fits, cv)

    outputs = {
        "target_session_rows": AUDIT_DIR / "gain_curve_target_session_rows.csv",
        "separate_fits": TABLE_DIR / "extended_data_figure5_separate_quadratic_fits.csv",
        "baseline_bin_counts": TABLE_DIR / "extended_data_figure5_baseline_bin_counts.csv",
        "baseline_distribution": TABLE_DIR / "extended_data_figure5_baseline_distribution.csv",
        "model_comparison": TABLE_DIR / "extended_data_table5_model_comparison.csv",
        "model_comparison_excel_semicolon": TABLE_DIR / "extended_data_table5_model_comparison_excel_semicolon.csv",
        "figure": figure_path,
    }
    data.to_csv(outputs["target_session_rows"], index=False)
    separate.to_csv(outputs["separate_fits"], index=False)
    counts.to_csv(outputs["baseline_bin_counts"])
    _ordered_baseline_distribution(dist).to_csv(outputs["baseline_distribution"], index=False)
    ext_table.to_csv(outputs["model_comparison"], index=False, encoding="utf-8-sig")
    ext_table.to_csv(outputs["model_comparison_excel_semicolon"], index=False, sep=";", encoding="utf-8-sig")
    outputs.update(write_caption_files(data, figure_path, ext_table))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None)
    args = parser.parse_args()
    outputs = run(args.base_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
