"""
Shared-control adaptation analysis.

This script tests whether decoded-command metrics, and optionally neural
target-decoding evidence, change as exposure to shared control accumulates.
Inference is session-level: linear effects are estimated within each session
and then summarized across sessions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "tables" / "Shared-control adaptation"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Shared-control adaptation"
DEFAULT_COMMAND_METRICS = [
    "angular_error_to_target_deg_mean",
    "decoded_speed_mean",
    "lateral_command_abs_mean",
    "command_heading_variability_deg",
    "command_speed_sd",
]
RECENT_EXPOSURE_VARS = [
    "previous_trial_shared",
    "preceding_3_shared_fraction",
    "preceding_5_shared_fraction",
    "preceding_shared_fraction",
]
BEHAVIOR_METRICS = [
    "success",
    "angular_error_to_target_deg_mean",
    "collision_indicator",
    "obstacle_clearance_m",
    "path_length_efficiency",
]
EXPOSURE_ASSOCIATED_COMMAND_METRICS = ["decoded_speed_mean", "command_speed_sd"]


def _load_command_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"task_family", "session_uid", "trial_id", "condition", "shared_control"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def _load_behavior_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    keep = [
        "monkey",
        "experiment",
        "session_index",
        "session_file",
        "trial_id",
        "obstacle_clearance_m",
        "path_length_efficiency",
        "path_length_ratio_to_straight",
        "mean_executed_speed_mps",
        "smoothness_mean_squared_jerk",
    ]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].copy()
    if "obstacle_clearance_m" in out.columns:
        clearance = pd.to_numeric(out["obstacle_clearance_m"], errors="coerce")
        out["collision_indicator"] = np.where(clearance.notna(), (clearance < 0).astype(float), np.nan)
    return out


def _load_neural_evidence(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    features = pd.read_csv(path)
    feature_cols = [col for col in features.columns if col.startswith("feat_")]
    required = {"task_family", "session_uid", "trial_id", "condition", "target"}
    if features.empty or not feature_cols or not required.issubset(features.columns):
        return pd.DataFrame()

    rows = []
    for session_uid, session in features.groupby("session_uid"):
        counts = session["target"].value_counts()
        usable_targets = counts[counts >= 2].index
        session = session[session["target"].isin(usable_targets)].copy()
        if len(usable_targets) < 2:
            continue
        n_splits = int(min(5, counts[counts >= 2].min()))
        if n_splits < 2:
            continue
        x = session[feature_cols].fillna(0).to_numpy(dtype=float)
        y = session["target"].to_numpy()
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        try:
            proba = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")
            classes = list(clf.steps[-1][1].classes_) if hasattr(clf.steps[-1][1], "classes_") else sorted(usable_targets)
        except Exception:
            # cross_val_predict does not leave a fitted final estimator in every sklearn version.
            try:
                preds = cross_val_predict(clf, x, y, cv=cv)
                evidence = (preds == y).astype(float)
            except Exception:
                continue
        else:
            class_to_col = {label: idx for idx, label in enumerate(classes)}
            evidence = np.array([proba[i, class_to_col[label]] for i, label in enumerate(y)], dtype=float)
        meta = session[
            [
                "monkey",
                "experiment",
                "task_family",
                "session_index",
                "session_uid",
                "session_file",
                "trial_id",
                "condition",
                "shared_control",
                "success",
                "target",
            ]
        ].copy()
        meta["neural_target_evidence"] = evidence
        rows.append(meta)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _add_session_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trial_id_numeric"] = pd.to_numeric(out["trial_id"], errors="coerce")
    out = out.sort_values(["session_uid", "trial_id_numeric", "trial_id"]).reset_index(drop=True)
    out["trial_order"] = out.groupby("session_uid").cumcount()
    counts = out.groupby("session_uid")["trial_order"].transform("max").replace(0, np.nan)
    out["trial_progress"] = (out["trial_order"] / counts).fillna(0.0)
    out["preceding_shared_count"] = out.groupby("session_uid")["shared_control"].cumsum() - out["shared_control"]
    out["preceding_trial_count"] = out.groupby("session_uid").cumcount()
    denom = out["preceding_trial_count"].replace(0, np.nan)
    out["preceding_shared_fraction"] = (out["preceding_shared_count"] / denom).fillna(0.0)
    out["condition_x_trial_progress"] = out["shared_control"] * out["trial_progress"]
    out["previous_trial_shared"] = (
        out.groupby("session_uid")["shared_control"].shift(1).fillna(0).astype(float)
    )
    for window in (3, 5):
        values = []
        for _, session in out.groupby("session_uid", sort=False):
            shared = session["shared_control"].to_numpy(float)
            frac = np.zeros(len(shared), dtype=float)
            for i in range(len(shared)):
                start = max(0, i - window)
                prior = shared[start:i]
                frac[i] = float(np.mean(prior)) if len(prior) else 0.0
            values.append(pd.Series(frac, index=session.index))
        out[f"preceding_{window}_shared_fraction"] = pd.concat(values).sort_index()
    return out


def _target_adjusted_design(session: pd.DataFrame, exposure_col: str | None = None) -> tuple[np.ndarray, list[str]]:
    cols = [
        np.ones(len(session)),
        session["trial_progress"].to_numpy(float),
    ]
    names = ["intercept", "trial_progress"]
    if exposure_col is not None:
        cols.append(session[exposure_col].to_numpy(float))
        names.append(exposure_col)
    if "target" in session.columns:
        dummies = pd.get_dummies(session["target"], prefix="target", drop_first=True, dtype=float)
        for col in dummies.columns:
            cols.append(dummies[col].to_numpy(float))
            names.append(col)
    return np.column_stack(cols), names


def _ols_coefficients(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, float] | None:
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = y[finite]
    x = x[finite]
    if len(y) <= x.shape[1] + 2:
        return None
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        return None
    return {name: float(value) for name, value in zip(names, beta)}


def fit_condition_progress_slopes(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for keys, session in df.groupby(["task_family", "monkey", "experiment", "session_index", "session_uid"]):
            task_family, monkey, experiment, session_index, session_uid = keys
            y = pd.to_numeric(session[metric], errors="coerce").to_numpy(float)
            x = np.column_stack(
                [
                    np.ones(len(session)),
                    session["shared_control"].to_numpy(float),
                    session["trial_progress"].to_numpy(float),
                    session["condition_x_trial_progress"].to_numpy(float),
                ]
            )
            coeffs = _ols_coefficients(
                y,
                x,
                ["intercept", "shared_control", "trial_progress", "condition_x_trial_progress"],
            )
            if coeffs is None:
                continue
            rows.append(
                {
                    "analysis": "condition_by_trial_progress",
                    "metric": metric,
                    "task_family": task_family,
                    "monkey": monkey,
                    "experiment": experiment,
                    "session_index": int(session_index),
                    "session_uid": session_uid,
                    "n_trials": int(np.isfinite(y).sum()),
                    **coeffs,
                }
            )
    return pd.DataFrame(rows)


def fit_bci_only_carryover_slopes(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    bci = df[df["condition"].eq("BCI-only")].copy()
    for metric in metrics:
        if metric not in bci.columns:
            continue
        for exposure_col in RECENT_EXPOSURE_VARS:
            if exposure_col not in bci.columns:
                continue
            for keys, session in bci.groupby(["task_family", "monkey", "experiment", "session_index", "session_uid"]):
                task_family, monkey, experiment, session_index, session_uid = keys
                y = pd.to_numeric(session[metric], errors="coerce").to_numpy(float)
                x, names = _target_adjusted_design(session, exposure_col)
                coeffs = _ols_coefficients(y, x, names)
                if coeffs is None:
                    continue
                rows.append(
                    {
                        "analysis": f"bci_only_{exposure_col}",
                        "metric": metric,
                        "task_family": task_family,
                        "monkey": monkey,
                        "experiment": experiment,
                        "session_index": int(session_index),
                        "session_uid": session_uid,
                        "n_trials": int(np.isfinite(y).sum()),
                        **coeffs,
                    }
                )
    return pd.DataFrame(rows)


def diagnose_bci_only_carryover_collinearity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bci = df[df["condition"].eq("BCI-only")].copy()
    for exposure_col in RECENT_EXPOSURE_VARS:
        if exposure_col not in bci.columns:
            continue
        for keys, session in bci.groupby(["task_family", "monkey", "experiment", "session_index", "session_uid"]):
            task_family, monkey, experiment, session_index, session_uid = keys
            x1 = session["trial_progress"].to_numpy(float)
            x2 = session[exposure_col].to_numpy(float)
            finite = np.isfinite(x1) & np.isfinite(x2)
            x1 = x1[finite]
            x2 = x2[finite]
            if len(x1) < 4 or np.nanstd(x1) <= 1e-12 or np.nanstd(x2) <= 1e-12:
                corr = np.nan
                vif = np.nan
                condition_number = np.nan
                status = "skipped_insufficient_variation"
            else:
                corr = float(np.corrcoef(x1, x2)[0, 1])
                r2 = corr ** 2
                vif = float(1.0 / max(1.0 - r2, 1e-12))
                x, _names = _target_adjusted_design(session.loc[finite].copy(), exposure_col)
                try:
                    condition_number = float(np.linalg.cond(x))
                except Exception:
                    condition_number = np.nan
                status = "ok"
            rows.append(
                {
                    "status": status,
                    "task_family": task_family,
                    "monkey": monkey,
                    "experiment": experiment,
                    "session_index": int(session_index),
                    "session_uid": session_uid,
                    "exposure_variable": exposure_col,
                    "n_bci_only_trials": int(len(x1)),
                    "corr_trial_progress_exposure": corr,
                    "vif_trial_progress_or_exposure": vif,
                    "target_adjusted_design_condition_number": condition_number,
                }
            )
    diagnostics = pd.DataFrame(rows)
    summary_rows = []
    ok = diagnostics[diagnostics["status"].eq("ok")].copy()
    for keys, group in ok.groupby(["task_family", "exposure_variable"]):
        task_family, exposure_variable = keys
        abs_corr = group["corr_trial_progress_exposure"].abs()
        summary_rows.append(
            {
                "task_family": task_family,
                "exposure_variable": exposure_variable,
                "n_sessions": int(len(group)),
                "median_abs_corr": float(abs_corr.median()),
                "p90_abs_corr": float(abs_corr.quantile(0.90)),
                "max_abs_corr": float(abs_corr.max()),
                "median_vif": float(group["vif_trial_progress_or_exposure"].median()),
                "p90_vif": float(group["vif_trial_progress_or_exposure"].quantile(0.90)),
                "max_vif": float(group["vif_trial_progress_or_exposure"].max()),
                "median_condition_number": float(group["target_adjusted_design_condition_number"].median()),
                "p90_condition_number": float(group["target_adjusted_design_condition_number"].quantile(0.90)),
                "max_condition_number": float(group["target_adjusted_design_condition_number"].max()),
                "n_sessions_vif_gt_5": int((group["vif_trial_progress_or_exposure"] > 5).sum()),
                "n_sessions_vif_gt_10": int((group["vif_trial_progress_or_exposure"] > 10).sum()),
                "n_sessions_condition_number_gt_30": int((group["target_adjusted_design_condition_number"] > 30).sum()),
            }
        )
    return diagnostics, pd.DataFrame(summary_rows)


def summarize_slopes(slopes: pd.DataFrame, coefficient: str) -> pd.DataFrame:
    rows = []
    if slopes.empty:
        return pd.DataFrame()
    for keys, group in slopes.groupby(["analysis", "task_family", "metric"]):
        analysis, task_family, metric = keys
        vals = group[coefficient].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        try:
            wil = stats.wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
            w_stat, p_value = float(wil.statistic), float(wil.pvalue)
        except Exception:
            w_stat, p_value = np.nan, np.nan
        rows.append(
            {
                "analysis": analysis,
                "task_family": task_family,
                "metric": metric,
                "coefficient": coefficient,
                "n_sessions": int(len(vals)),
                "mean_slope": float(np.mean(vals)),
                "median_slope": float(np.median(vals)),
                "n_positive": int(np.sum(vals > 0)),
                "n_negative": int(np.sum(vals < 0)),
                "wilcoxon_two_sided_statistic": w_stat,
                "wilcoxon_two_sided_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def summarize_exposure_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for exposure_col in RECENT_EXPOSURE_VARS:
        if exposure_col in slopes.columns:
            summaries.append(summarize_slopes(slopes, exposure_col))
    return pd.concat([s for s in summaries if not s.empty], ignore_index=True) if summaries else pd.DataFrame()


def compare_task_interactions(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if slopes.empty:
        return pd.DataFrame()
    exposure_slopes = slopes[slopes["analysis"].str.startswith("bci_only_", na=False)].copy()
    for exposure_col in RECENT_EXPOSURE_VARS:
        if exposure_col not in exposure_slopes.columns:
            continue
        for metric, group in exposure_slopes.groupby("metric"):
            vals_by_task = {
                task: task_group[exposure_col].dropna().to_numpy(float)
                for task, task_group in group.groupby("task_family")
            }
            vals_by_task = {task: vals for task, vals in vals_by_task.items() if len(vals) > 0}
            if len(vals_by_task) >= 2:
                try:
                    kruskal = stats.kruskal(*vals_by_task.values())
                    h_stat, h_p = float(kruskal.statistic), float(kruskal.pvalue)
                except Exception:
                    h_stat, h_p = np.nan, np.nan
            else:
                h_stat, h_p = np.nan, np.nan
            appearing = vals_by_task.get("Appearing Obstacle")
            for comparison_task in ("Fixed Obstacle", "Appearing Obstacle follow-up"):
                other = vals_by_task.get(comparison_task)
                if appearing is None or other is None:
                    u_stat, u_p = np.nan, np.nan
                    diff = np.nan
                    n_appearing = 0 if appearing is None else len(appearing)
                    n_other = 0 if other is None else len(other)
                else:
                    try:
                        u = stats.mannwhitneyu(appearing, other, alternative="two-sided")
                        u_stat, u_p = float(u.statistic), float(u.pvalue)
                    except Exception:
                        u_stat, u_p = np.nan, np.nan
                    diff = float(np.median(appearing) - np.median(other))
                    n_appearing = len(appearing)
                    n_other = len(other)
                rows.append(
                    {
                        "metric": metric,
                        "exposure_variable": exposure_col,
                        "kruskal_task_effect_statistic": h_stat,
                        "kruskal_task_effect_p": h_p,
                        "comparison": f"Appearing Obstacle vs {comparison_task}",
                        "n_appearing_sessions": int(n_appearing),
                        "n_comparison_sessions": int(n_other),
                        "appearing_minus_comparison_median_slope": diff,
                        "mannwhitney_statistic": u_stat,
                        "mannwhitney_two_sided_p": u_p,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr_q_values(out.rename(columns={"mannwhitney_two_sided_p": "wilcoxon_two_sided_p"}))
        out = out.rename(
            columns={
                "wilcoxon_two_sided_p": "mannwhitney_two_sided_p",
                "wilcoxon_two_sided_q_bh": "mannwhitney_two_sided_q_bh",
            }
        )
    return out


def fit_bci_only_command_behavior_associations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bci = df[df["condition"].eq("BCI-only")].copy()
    for outcome in BEHAVIOR_METRICS:
        if outcome not in bci.columns:
            continue
        for command_metric in EXPOSURE_ASSOCIATED_COMMAND_METRICS:
            if command_metric not in bci.columns:
                continue
            for keys, session in bci.groupby(["task_family", "monkey", "experiment", "session_index", "session_uid"]):
                task_family, monkey, experiment, session_index, session_uid = keys
                y = pd.to_numeric(session[outcome], errors="coerce").to_numpy(float)
                base_x, base_names = _target_adjusted_design(session, None)
                command_values = pd.to_numeric(session[command_metric], errors="coerce").to_numpy(float)
                x = np.column_stack([base_x, command_values])
                names = base_names + [command_metric]
                coeffs = _ols_coefficients(y, x, names)
                if coeffs is None:
                    continue
                rows.append(
                    {
                        "analysis": "bci_only_command_behavior_association",
                        "outcome": outcome,
                        "command_metric": command_metric,
                        "task_family": task_family,
                        "monkey": monkey,
                        "experiment": experiment,
                        "session_index": int(session_index),
                        "session_uid": session_uid,
                        "n_trials": int(np.isfinite(y).sum()),
                        **coeffs,
                    }
                )
    return pd.DataFrame(rows)


def summarize_command_behavior_associations(associations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if associations.empty:
        return pd.DataFrame()
    for keys, group in associations.groupby(["task_family", "outcome", "command_metric"]):
        task_family, outcome, command_metric = keys
        vals = group[command_metric].dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        try:
            wil = stats.wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
            w_stat, p_value = float(wil.statistic), float(wil.pvalue)
        except Exception:
            w_stat, p_value = np.nan, np.nan
        rows.append(
            {
                "task_family": task_family,
                "outcome": outcome,
                "command_metric": command_metric,
                "n_sessions": int(len(vals)),
                "mean_slope": float(np.mean(vals)),
                "median_slope": float(np.median(vals)),
                "n_positive": int(np.sum(vals > 0)),
                "n_negative": int(np.sum(vals < 0)),
                "wilcoxon_two_sided_statistic": w_stat,
                "wilcoxon_two_sided_p": p_value,
            }
        )
    return add_fdr_q_values(pd.DataFrame(rows))


def add_fdr_q_values(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "wilcoxon_two_sided_p" not in summary.columns:
        return summary
    out = summary.copy()
    out["wilcoxon_two_sided_q_bh"] = np.nan
    group_iter = out.groupby("analysis") if "analysis" in out.columns else [(None, out)]
    for _analysis, group in group_iter:
        p = pd.to_numeric(group["wilcoxon_two_sided_p"], errors="coerce")
        valid = p.notna()
        if not valid.any():
            continue
        values = p[valid].to_numpy(float)
        order = np.argsort(values)
        ranked = values[order]
        m = len(ranked)
        q_sorted = ranked * m / np.arange(1, m + 1)
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0, 1)
        q = np.empty_like(q_sorted)
        q[order] = q_sorted
        out.loc[p[valid].index, "wilcoxon_two_sided_q_bh"] = q
    return out


def plot_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in DEFAULT_COMMAND_METRICS if m in set(summary["metric"])]
    plot_df = summary[
        summary["analysis"].eq("condition_by_trial_progress")
        & summary["coefficient"].eq("condition_x_trial_progress")
        & summary["metric"].isin(metrics)
    ].copy()
    if plot_df.empty:
        return
    tasks = ["Fixed Obstacle", "Appearing Obstacle", "Appearing Obstacle follow-up"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 1.55 * len(metrics)), dpi=220, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(tasks))
    for ax, metric in zip(axes, metrics):
        vals = []
        labels = []
        for task in tasks:
            row = plot_df[plot_df["metric"].eq(metric) & plot_df["task_family"].eq(task)]
            if row.empty:
                vals.append(np.nan)
                labels.append("")
            else:
                vals.append(float(row.iloc[0]["median_slope"]))
                labels.append(f"p={float(row.iloc[0]['wilcoxon_two_sided_p']):.3g}")
        ax.bar(x, vals, color="#7b8da8")
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_ylabel(metric.replace("_", "\n"), fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        for xi, yi, lab in zip(x, vals, labels):
            if np.isfinite(yi):
                ax.text(xi, yi, lab, ha="center", va="bottom" if yi >= 0 else "top", fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(x, tasks, rotation=25, ha="right", fontsize=7)
    fig.suptitle("Shared-control-specific within-session adaptation", fontsize=10)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(FIGURE_DIR / f"shared_control_adaptation_summary.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    collinearity_summary: pd.DataFrame,
    task_interactions: pd.DataFrame,
    behavior_association_summary: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Shared-control Adaptation Analysis",
        "",
        "Within each session, decoded-command metrics were modeled as condition + trial progress + condition x trial progress. A separate BCI-only carryover analysis tested whether BCI-only trial metrics changed with the fraction of preceding trials that were shared-control trials.",
        "",
        "## Condition-specific within-session drift",
        "",
    ]
    cond = summary[
        summary["analysis"].eq("condition_by_trial_progress")
        & summary["coefficient"].eq("condition_x_trial_progress")
    ]
    for _, row in cond.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['metric']}: median interaction slope "
            f"{row['median_slope']:.4g}; {int(row['n_positive'])}/{int(row['n_sessions'])} positive; "
            f"two-sided Wilcoxon p={row['wilcoxon_two_sided_p']:.4g}, "
            f"BH q={row['wilcoxon_two_sided_q_bh']:.4g}."
        )
    lines.extend(["", "## BCI-only carryover across exposure timescales", ""])
    carry = summary[summary["analysis"].str.startswith("bci_only_", na=False)]
    for _, row in carry.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['metric']}, {row['coefficient']}: median slope "
            f"{row['median_slope']:.4g}; {int(row['n_positive'])}/{int(row['n_sessions'])} positive; "
            f"two-sided Wilcoxon p={row['wilcoxon_two_sided_p']:.4g}, "
            f"BH q={row['wilcoxon_two_sided_q_bh']:.4g}."
        )
    lines.extend(["", "## BCI-only carryover collinearity diagnostic", ""])
    for _, row in collinearity_summary.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['exposure_variable']}: median |r|={row['median_abs_corr']:.3f}, "
            f"p90 |r|={row['p90_abs_corr']:.3f}, max |r|={row['max_abs_corr']:.3f}; "
            f"median VIF={row['median_vif']:.2f}, p90 VIF={row['p90_vif']:.2f}, "
            f"max VIF={row['max_vif']:.2f}; median condition number="
            f"{row['median_condition_number']:.2f}, max={row['max_condition_number']:.2f}. "
            f"Sessions with VIF>5: {int(row['n_sessions_vif_gt_5'])}/{int(row['n_sessions'])}; "
            f"condition number>30: {int(row['n_sessions_condition_number_gt_30'])}/{int(row['n_sessions'])}."
        )
    lines.extend(["", "## Formal task interaction checks", ""])
    if task_interactions.empty:
        lines.append("No task-interaction rows were available.")
    else:
        top = task_interactions.sort_values("mannwhitney_two_sided_p").head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['metric']}, {row['exposure_variable']}, {row['comparison']}: "
                f"Appearing-minus-comparison median slope difference "
                f"{row['appearing_minus_comparison_median_slope']:.4g}; "
                f"Mann-Whitney p={row['mannwhitney_two_sided_p']:.4g}, "
                f"BH q={row['mannwhitney_two_sided_q_bh']:.4g}."
            )
    lines.extend(["", "## Do exposure-associated commands predict BCI-only behavior?", ""])
    if behavior_association_summary.empty:
        lines.append("No command-behavior association rows were available.")
    else:
        hits = behavior_association_summary[
            pd.to_numeric(behavior_association_summary["wilcoxon_two_sided_q_bh"], errors="coerce") < 0.1
        ].copy()
        if hits.empty:
            lines.append("No command-behavior associations survived BH q<0.1.")
        else:
            for _, row in hits.sort_values(["task_family", "outcome", "command_metric"]).iterrows():
                lines.append(
                    f"- {row['task_family']}, {row['outcome']} vs {row['command_metric']}: "
                    f"median slope {row['median_slope']:.4g}; "
                    f"{int(row['n_positive'])}/{int(row['n_sessions'])} positive; "
                    f"p={row['wilcoxon_two_sided_p']:.4g}, q={row['wilcoxon_two_sided_q_bh']:.4g}."
                )
    lines.extend(
        [
            "",
            "## Suggested reviewer wording",
            "",
            "We tested for progressive strategic adaptation by asking whether raw decoded-command metrics changed differently for shared-control versus BCI-only trials over the course of a session, whether BCI-only commands changed after recent shared-control exposure, whether the exposure slopes differed formally across tasks, and whether exposure-associated command metrics predicted BCI-only behavior. Recent one-trial and 3-5-trial exposure effects were weak after multiple-comparison correction, whereas the cumulative original Appearing Obstacle effect on BCI-only command-speed variability remained significant. Thus the data support a cautious carryover/adaptation interpretation on a slower within-session timescale rather than a strong immediate trial-to-trial effect. Formal task-interaction tests did not survive correction, so apparent task specificity should be described as descriptive.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    command_metrics_path: str | None = None,
    neural_features_path: str | None = None,
    behavior_metrics_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    command_path = Path(command_metrics_path) if command_metrics_path else (
        PROJECT_ROOT / "tables" / "Decoded command comparison" / "decoded_command_trial_metrics.csv"
    )
    neural_path = Path(neural_features_path) if neural_features_path else (
        PROJECT_ROOT / "tables" / "Neural target generalization" / "neural_target_feature_rows.csv"
    )
    behavior_path = Path(behavior_metrics_path) if behavior_metrics_path else (
        PROJECT_ROOT / "tables" / "obstacle_session_metric_trials.csv"
    )
    command = _add_session_order(_load_command_metrics(command_path))
    behavior = _load_behavior_metrics(behavior_path)
    if not behavior.empty:
        command = command.merge(
            behavior,
            on=["monkey", "experiment", "session_index", "session_file", "trial_id"],
            how="left",
        )
    neural = _load_neural_evidence(neural_path)
    datasets = [command]
    metrics = list(DEFAULT_COMMAND_METRICS)
    if not neural.empty:
        datasets.append(neural[["session_uid", "trial_id", "neural_target_evidence"]])
        metrics.append("neural_target_evidence")
    data = command.merge(datasets[1], on=["session_uid", "trial_id"], how="left") if len(datasets) > 1 else command

    condition_slopes = fit_condition_progress_slopes(data, metrics)
    carryover_metrics = list(dict.fromkeys(metrics + [m for m in BEHAVIOR_METRICS if m in data.columns]))
    carryover_slopes = fit_bci_only_carryover_slopes(data, carryover_metrics)
    collinearity, collinearity_summary = diagnose_bci_only_carryover_collinearity(data)
    task_interactions = compare_task_interactions(carryover_slopes)
    behavior_associations = fit_bci_only_command_behavior_associations(data)
    behavior_association_summary = summarize_command_behavior_associations(behavior_associations)
    slopes = pd.concat([condition_slopes, carryover_slopes], ignore_index=True)
    summaries = [
        summarize_slopes(condition_slopes, "condition_x_trial_progress"),
        summarize_exposure_slopes(carryover_slopes),
    ]
    summary = pd.concat([s for s in summaries if not s.empty], ignore_index=True)
    summary = add_fdr_q_values(summary)

    data.to_csv(TABLE_DIR / "shared_control_adaptation_trial_metrics.csv", index=False)
    slopes.to_csv(TABLE_DIR / "shared_control_adaptation_session_slopes.csv", index=False)
    summary.to_csv(TABLE_DIR / "shared_control_adaptation_summary.csv", index=False)
    collinearity.to_csv(TABLE_DIR / "shared_control_adaptation_collinearity_by_session.csv", index=False)
    collinearity_summary.to_csv(TABLE_DIR / "shared_control_adaptation_collinearity_summary.csv", index=False)
    task_interactions.to_csv(TABLE_DIR / "shared_control_adaptation_task_interactions.csv", index=False)
    behavior_associations.to_csv(TABLE_DIR / "shared_control_adaptation_command_behavior_session_slopes.csv", index=False)
    behavior_association_summary.to_csv(TABLE_DIR / "shared_control_adaptation_command_behavior_summary.csv", index=False)
    plot_summary(summary)
    write_report(
        summary,
        collinearity_summary,
        task_interactions,
        behavior_association_summary,
        TABLE_DIR / "shared_control_adaptation_report.md",
    )
    return slopes, summary, collinearity_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-metrics-path", default=None)
    parser.add_argument("--neural-features-path", default=None)
    parser.add_argument("--behavior-metrics-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, summary, collinearity_summary = run_analysis(
        args.command_metrics_path,
        args.neural_features_path,
        args.behavior_metrics_path,
    )
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))
        print("")
        print(collinearity_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
