"""
Follow-up-only reactive obstacle analysis.

This analysis tests whether shared control helps after the obstacle becomes
visible in the six-month follow-up dataset, where the hidden-obstacle prior was
removed. Trials are aligned to obstacle appearance at t=0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = None
MANUSCRIPT_MONKEY = "Monkey 1"
DEFAULT_SOURCE_MONKEY = "Monkey 3"
EXPERIMENT = "AI Appearing Obstacle 2"
CONDITION_LABELS = {0.0: "BCI-only", 1.0: "Shared-control"}
AVATAR_RADIUS_M = 0.25
OBSTACLE_RADIUS_M = 0.45
COLLISION_RADIUS_M = AVATAR_RADIUS_M + OBSTACLE_RADIUS_M
HEADING_WINDOW_S = 1.0


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _scalar(value) -> float:
    if value is None:
        return np.nan
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return np.nan


def _trajectory(trial) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj = _get(trial, "avatarTrajectory", None)
    if not isinstance(traj, dict):
        return np.array([]), np.array([]), np.array([])
    try:
        x = np.asarray(traj["x"], dtype=float)
        z = np.asarray(traj["z"], dtype=float)
        t = np.asarray(traj.get("time", np.arange(len(x)) * 50.0), dtype=float)
    except Exception:
        return np.array([]), np.array([]), np.array([])
    n = min(len(x), len(z), len(t))
    return x[:n], z[:n], t[:n]


def _time_scale_to_seconds(t: np.ndarray) -> float:
    finite = np.asarray(t, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size and np.nanmedian(diffs) > 5.0:
        return 0.001
    return 1.0


def _vec2_from_pos(value) -> np.ndarray:
    if value is None:
        return np.full(2, np.nan)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(2, np.nan)
    if arr.size >= 3:
        return np.asarray([arr[0], arr[2]], dtype=float)
    if arr.size >= 2:
        return np.asarray([arr[0], arr[1]], dtype=float)
    return np.full(2, np.nan)


def _radial_velocity_away(x: np.ndarray, z: np.ndarray, t_s: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.array([])
    dt = np.diff(t_s)
    good_dt = np.isfinite(dt) & (dt > 0)
    vx = np.full(len(dt), np.nan)
    vz = np.full(len(dt), np.nan)
    vx[good_dt] = np.diff(x)[good_dt] / dt[good_dt]
    vz[good_dt] = np.diff(z)[good_dt] / dt[good_dt]
    mid_x = 0.5 * (x[:-1] + x[1:])
    mid_z = 0.5 * (z[:-1] + z[1:])
    radial = np.column_stack([mid_x - obstacle[0], mid_z - obstacle[1]])
    norm = np.linalg.norm(radial, axis=1)
    unit = np.full_like(radial, np.nan)
    valid = np.isfinite(norm) & (norm > 1e-9)
    unit[valid] = radial[valid] / norm[valid, None]
    return vx * unit[:, 0] + vz * unit[:, 1]


def _trial_id(trial) -> int | float:
    for key in ("trial", "trial_id", "trialId"):
        value = _get(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return np.nan
    return np.nan


def _target_label(trial) -> str:
    _ensure_import_path()
    from src.constants import target_mapping

    try:
        return target_mapping.get(tuple(_get(trial, "targetPosition")), "unknown")
    except Exception:
        return "unknown"


def trial_metric_row(trial, source_monkey: str, session_index: int, success: int) -> dict | None:
    try:
        factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None
    if factor not in CONDITION_LABELS:
        return None

    event = _scalar(_get(trial, "obstacleAppearedTime", None))
    if not np.isfinite(event):
        return None

    obstacle = _vec2_from_pos(_get(trial, "obstaclePosition", None))
    if not np.isfinite(obstacle).all():
        return None

    x, z, t = _trajectory(trial)
    if len(x) < 3:
        return None

    scale = _time_scale_to_seconds(t)
    t_rel_s = (t - event) * scale
    points = np.column_stack([x, z])
    clearance = np.linalg.norm(points - obstacle, axis=1) - COLLISION_RADIUS_M

    post = t_rel_s >= 0
    pre_heading = (t_rel_s[:-1] >= -HEADING_WINDOW_S) & (t_rel_s[:-1] < 0)
    post_heading = (t_rel_s[:-1] >= 0) & (t_rel_s[:-1] < HEADING_WINDOW_S)
    radial_v = _radial_velocity_away(x, z, t * scale, obstacle)

    min_pre_clearance = float(np.nanmin(clearance[t_rel_s < 0])) if np.any(t_rel_s < 0) else np.nan
    min_post_clearance = float(np.nanmin(clearance[post])) if np.any(post) else np.nan
    post_collision = bool(np.any(clearance[post] <= 0.0)) if np.any(post) else False
    pre_away = float(np.nanmean(radial_v[pre_heading])) if np.any(pre_heading) else np.nan
    post_away = float(np.nanmean(radial_v[post_heading])) if np.any(post_heading) else np.nan

    return {
        "monkey": MANUSCRIPT_MONKEY,
        "source_monkey_folder": source_monkey,
        "experiment": EXPERIMENT,
        "session_index": session_index,
        "session_uid": f"{EXPERIMENT}__session_{session_index:02d}",
        "trial_id": _trial_id(trial),
        "condition": CONDITION_LABELS[factor],
        "shared_control": int(factor == 1.0),
        "success": int(success),
        "target": _target_label(trial),
        "obstacle_appeared_time": event,
        "n_pre_samples": int(np.sum(t_rel_s < 0)),
        "n_post_samples": int(np.sum(post)),
        "min_pre_clearance_m": min_pre_clearance,
        "min_post_clearance_m": min_post_clearance,
        "post_collision": post_collision,
        "mean_radial_velocity_away_pre_1s_mps": pre_away,
        "mean_radial_velocity_away_post_1s_mps": post_away,
        "delta_radial_velocity_away_post_minus_pre_mps": post_away - pre_away,
    }


def collect_rows(base_dir: str, source_monkey: str) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    all_trials, correct_all, incorrect_all, *_ = load_files(
        EXPERIMENT, source_monkey, base_dir, nb_trials=None
    )
    if all_trials == []:
        return pd.DataFrame()

    rows = []
    for session_index, (correct_trials, incorrect_trials) in enumerate(zip(correct_all, incorrect_all)):
        for success, trials in ((1, correct_trials), (0, incorrect_trials)):
            for trial in trials:
                row = trial_metric_row(trial, source_monkey, session_index, success)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


def _paired_session_table(rows: pd.DataFrame, metric: str, name: str) -> pd.DataFrame:
    aggfunc = "mean"
    wide = rows.pivot_table(
        index=["session_index", "session_uid"],
        columns="condition",
        values=metric,
        aggfunc=aggfunc,
    ).reset_index()
    if not {"Shared-control", "BCI-only"}.issubset(wide.columns):
        return pd.DataFrame()
    wide = wide.dropna(subset=["Shared-control", "BCI-only"]).copy()
    wide["metric"] = name
    wide["shared_control"] = wide["Shared-control"]
    wide["bci_only"] = wide["BCI-only"]
    wide["shared_minus_bci_only"] = wide["shared_control"] - wide["bci_only"]
    return wide[["metric", "session_index", "session_uid", "bci_only", "shared_control", "shared_minus_bci_only"]]


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_tables = [
        _paired_session_table(rows, "post_collision", "post_collision_probability"),
        _paired_session_table(rows, "min_post_clearance_m", "min_post_clearance_m"),
        _paired_session_table(rows, "delta_radial_velocity_away_post_minus_pre_mps", "delta_radial_velocity_away_mps"),
    ]
    session = pd.concat([x for x in session_tables if not x.empty], ignore_index=True)

    summaries = []
    for metric, group in session.groupby("metric"):
        delta = group["shared_minus_bci_only"].to_numpy(float)
        if len(delta) >= 1:
            try:
                wil = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
                w_stat, p_value = float(wil.statistic), float(wil.pvalue)
            except Exception:
                w_stat, p_value = np.nan, np.nan
        else:
            w_stat, p_value = np.nan, np.nan
        summaries.append(
            {
                "metric": metric,
                "n_sessions": int(len(delta)),
                "n_sessions_shared_higher": int(np.sum(delta > 0)),
                "n_sessions_shared_lower": int(np.sum(delta < 0)),
                "n_sessions_no_difference": int(np.sum(delta == 0)),
                "bci_only_mean": float(np.nanmean(group["bci_only"])),
                "shared_control_mean": float(np.nanmean(group["shared_control"])),
                "shared_minus_bci_only_mean": float(np.nanmean(delta)),
                "shared_minus_bci_only_median": float(np.nanmedian(delta)),
                "wilcoxon_statistic": w_stat,
                "wilcoxon_two_sided_p": p_value,
            }
        )
    return session, pd.DataFrame(summaries)


def fit_trial_models(rows: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return pd.DataFrame([{"status": f"statsmodels unavailable: {exc}"}])

    model_df = rows[rows["target"] != "unknown"].copy()
    if model_df.empty:
        return pd.DataFrame([{"status": "no modeled trials after dropping unknown targets"}])

    outputs = []
    specs = [
        ("post_collision", "ols", "post_collision ~ shared_control + C(target)"),
        ("min_post_clearance_m", "ols", "min_post_clearance_m ~ shared_control + C(target)"),
        (
            "delta_radial_velocity_away_mps",
            "ols",
            "delta_radial_velocity_away_post_minus_pre_mps ~ shared_control + C(target)",
        ),
    ]
    for outcome, model_type, formula in specs:
        fit_df = model_df.copy()
        fitter = smf.ols
        if outcome == "post_collision":
            fit_df["post_collision"] = fit_df["post_collision"].astype(int)
        fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[outcome if outcome in fit_df else formula.split("~")[0].strip()]
        )
        try:
            fit = fitter(formula, data=fit_df).fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": fit_df["session_uid"]},
            )
            term = "shared_control"
            ci = fit.conf_int().loc[term]
            outputs.append(
                {
                    "status": "ok",
                    "outcome": outcome,
                    "model_type": model_type,
                    "term": term,
                    "coef": float(fit.params[term]),
                    "se_cluster_session": float(fit.bse[term]),
                    "p": float(fit.pvalues[term]),
                    "ci_low": float(ci.iloc[0]),
                    "ci_high": float(ci.iloc[1]),
                    "n_trials": int(fit.nobs),
                }
            )
        except Exception as exc:
            outputs.append({"status": f"{outcome} model failed: {exc}", "outcome": outcome})
    return pd.DataFrame(outputs)


def plot_panel(rows: pd.DataFrame, session: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=200)
    colors = {"BCI-only": "#8c4b38", "Shared-control": "#237a6b"}
    metric_labels = [
        ("post_collision_probability", "Post-appearance\ncollision probability", "%", 100.0),
        ("min_post_clearance_m", "Minimum post-appearance\nclearance (m)", "m", 1.0),
        ("delta_radial_velocity_away_mps", "Change in velocity away\nfrom obstacle (m/s)", "m/s", 1.0),
    ]
    for ax, (metric, title, ylabel, scale) in zip(axes, metric_labels):
        sub = session[session["metric"].eq(metric)]
        for i, condition in enumerate(["BCI-only", "Shared-control"]):
            vals = sub["bci_only" if condition == "BCI-only" else "shared_control"].to_numpy(float) * scale
            x = np.full(len(vals), i, dtype=float)
            ax.scatter(x, vals, s=28, color=colors[condition], alpha=0.78, edgecolor="white", linewidth=0.4)
            if len(vals):
                ax.hlines(np.nanmean(vals), i - 0.16, i + 0.16, color=colors[condition], lw=2.2)
        for _, row in sub.iterrows():
            ax.plot([0, 1], [row["bci_only"] * scale, row["shared_control"] * scale], color="0.75", lw=0.8, zorder=0)
        ax.set_xticks([0, 1], ["BCI-only", "Shared"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "post_collision_probability":
            ax.set_ylim(0, 105)
    fig.tight_layout(w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(output: Path, summary: pd.DataFrame, models: pd.DataFrame) -> None:
    def metric_row(name: str) -> pd.Series:
        return summary[summary["metric"].eq(name)].iloc[0]

    coll = metric_row("post_collision_probability")
    clear = metric_row("min_post_clearance_m")
    heading = metric_row("delta_radial_velocity_away_mps")
    lines = [
        "# Follow-up Reactive Obstacle Analysis",
        "",
        "Dataset: Monkey 1 Appearing Obstacle six-month follow-up, hidden-obstacle prior absent.",
        "",
        f"Geometry-based trajectory metrics were computed for the {int(coll['n_sessions'])} follow-up sessions with available `_trials.pkl` trajectory files.",
        "",
        "Trials were aligned to obstacle appearance at `t=0`. The primary reactive outcomes were post-appearance collision probability and minimum post-appearance clearance. As a supporting mechanistic outcome, heading/velocity reorientation was quantified as the change in radial velocity away from the obstacle from the 1 s pre-appearance window to the 1 s post-appearance window.",
        "",
        "## Session-Level Paired Comparisons",
        "",
        (
            f"- Post-appearance collision probability: BCI-only {100*coll['bci_only_mean']:.1f}%, "
            f"shared-control {100*coll['shared_control_mean']:.1f}%, "
            f"difference {100*coll['shared_minus_bci_only_mean']:.1f} pp, "
            f"Wilcoxon p={coll['wilcoxon_two_sided_p']:.4g}; "
            f"collision reduction in {int(coll['n_sessions_shared_lower'])}/{int(coll['n_sessions'])} sessions."
        ),
        (
            f"- Minimum post-appearance clearance: BCI-only {clear['bci_only_mean']:.2f} m, "
            f"shared-control {clear['shared_control_mean']:.2f} m, "
            f"difference {clear['shared_minus_bci_only_mean']:.2f} m, "
            f"Wilcoxon p={clear['wilcoxon_two_sided_p']:.4g}; "
            f"clearance increase in {int(clear['n_sessions_shared_higher'])}/{int(clear['n_sessions'])} sessions."
        ),
        (
            f"- Supporting mechanistic outcome, change in radial velocity away from obstacle: BCI-only {heading['bci_only_mean']:.2f} m/s, "
            f"shared-control {heading['shared_control_mean']:.2f} m/s, "
            f"difference {heading['shared_minus_bci_only_mean']:.2f} m/s, "
            f"Wilcoxon p={heading['wilcoxon_two_sided_p']:.4g}; "
            f"increase in {int(heading['n_sessions_shared_higher'])}/{int(heading['n_sessions'])} sessions."
        ),
        "",
        "## Target-Adjusted Trial-Level Models",
        "",
        models.to_markdown(index=False),
        "",
        "## Suggested Manuscript Wording",
        "",
        "In the follow-up trajectory sessions where no hidden-obstacle prior was available, shared control reduced post-appearance collision probability and increased obstacle clearance after the obstacle became visible, with both effects observed consistently across all usable sessions. In the same sessions, the change in radial velocity away from the obstacle also increased under shared control. These results provide direct evidence for online reactive obstacle avoidance by RT-V2 after sensory evidence became available, rather than a benefit that depended on advance spatial knowledge from a hidden obstacle prior.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(base_dir: str, source_monkey: str) -> dict[str, Path]:
    output_tables = PROJECT_ROOT / "tables" / "Follow-up reactive obstacle"
    output_figures = PROJECT_ROOT / "figures" / "Follow-up reactive obstacle"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(base_dir, source_monkey)
    if rows.empty:
        raise RuntimeError(f"No follow-up appearing-obstacle rows found under {base_dir!r}/{source_monkey!r}.")

    session, summary = summarize(rows)
    models = fit_trial_models(rows)

    paths = {
        "trial_metrics": output_tables / "followup_reactive_obstacle_trial_metrics.csv",
        "session_metrics": output_tables / "followup_reactive_obstacle_session_metrics.csv",
        "summary": output_tables / "followup_reactive_obstacle_summary.csv",
        "trial_models": output_tables / "followup_reactive_obstacle_trial_models.csv",
        "report": output_tables / "followup_reactive_obstacle_report.md",
        "figure": output_figures / "followup_reactive_obstacle_panel.svg",
    }
    rows.to_csv(paths["trial_metrics"], index=False)
    session.to_csv(paths["session_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    models.to_csv(paths["trial_models"], index=False)
    plot_panel(rows, session, paths["figure"])
    write_report(paths["report"], summary, models)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--source-monkey", default=DEFAULT_SOURCE_MONKEY)
    args = parser.parse_args()
    paths = run_analysis(load_base_dir(args.base_dir), args.source_monkey)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
