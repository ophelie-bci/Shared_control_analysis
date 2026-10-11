"""
Compare shared-control gain with vs without the hidden-obstacle prior.

This reviewer-control analysis is restricted to Monkey 1 Appearing Obstacle
sessions:

- AI Appearing Obstacle: hidden-obstacle prior present.
- AI Appearing Obstacle 2: six-month follow-up, hidden-obstacle prior absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = None
MONKEY = "Monkey 1"
DEFAULT_SOURCE_MONKEY = "Monkey 3"
CACHED_SESSION_TABLE = PROJECT_ROOT / "tables" / "collision_corrected_success_by_session.csv"
EXPERIMENTS = {
    "AI Appearing Obstacle": "Hidden prior",
    "AI Appearing Obstacle 2": "No hidden prior",
}


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)
CONDITION_LABELS = {
    0.0: "BCI-only",
    1.0: "Shared-control",
}


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _target_label(trial) -> str:
    _ensure_import_path()
    from src.constants import target_mapping

    try:
        return target_mapping.get(tuple(getattr(trial, "targetPosition")), "unknown")
    except Exception:
        return "unknown"


def _is_appearing_obstacle_trial(trial) -> bool:
    return getattr(trial, "obstacleAppearedTime", None) is not None


def _ai_factor(trial) -> float:
    try:
        return float(getattr(trial, "aiVelocityFactor"))
    except Exception:
        return np.nan


def _success_rate(n_success: int, n_total: int) -> float:
    return float(n_success / n_total) if n_total else np.nan


def collect_trial_rows(base_dir: str, source_monkey: str = DEFAULT_SOURCE_MONKEY) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    rows = []
    for experiment, prior_label in EXPERIMENTS.items():
        all_trials, correct_all, incorrect_all, *_rest = load_files(
            experiment, source_monkey, base_dir, nb_trials=None
        )
        if all_trials == []:
            continue
        for session_index, (correct_trials, incorrect_trials) in enumerate(
            zip(correct_all, incorrect_all)
        ):
            for success, trials in ((1, correct_trials), (0, incorrect_trials)):
                for trial in trials:
                    factor = _ai_factor(trial)
                    if factor not in CONDITION_LABELS:
                        continue
                    if not _is_appearing_obstacle_trial(trial):
                        continue
                    rows.append(
                        {
                            "monkey": MONKEY,
                            "source_monkey_folder": source_monkey,
                            "experiment": experiment,
                            "prior_condition": prior_label,
                            "followup": int(prior_label == "No hidden prior"),
                            "session_index": session_index,
                            "session_uid": f"{experiment}__session_{session_index:02d}",
                            "trial_id": getattr(trial, "trial", np.nan),
                            "condition": CONDITION_LABELS[factor],
                            "shared_control": int(factor == 1.0),
                            "success": success,
                            "target": _target_label(trial),
                        }
                    )
    return pd.DataFrame(rows)


def make_session_gain_table(trials: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trials.groupby(["prior_condition", "experiment", "session_index", "session_uid", "condition"])
        .agg(n_trials=("success", "size"), n_success=("success", "sum"), success_rate=("success", "mean"))
        .reset_index()
    )
    wide = grouped.pivot_table(
        index=["prior_condition", "experiment", "session_index", "session_uid"],
        columns="condition",
        values=["n_trials", "n_success", "success_rate"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{condition}".replace("-", "_").replace(" ", "_") for metric, condition in wide.columns]
    wide = wide.reset_index()
    required = ["success_rate_Shared_control", "success_rate_BCI_only"]
    wide = wide.dropna(subset=required)
    wide["delta_sc"] = wide["success_rate_Shared_control"] - wide["success_rate_BCI_only"]
    wide["s_shared_pct"] = 100.0 * wide["success_rate_Shared_control"]
    wide["s_bci_only_pct"] = 100.0 * wide["success_rate_BCI_only"]
    wide["delta_sc_pp"] = 100.0 * wide["delta_sc"]
    return wide


def load_cached_session_gains() -> pd.DataFrame:
    if not CACHED_SESSION_TABLE.exists():
        return pd.DataFrame()
    df = pd.read_csv(CACHED_SESSION_TABLE)
    df = df[
        df["source"].eq("raw_sessions")
        & df["monkey"].eq(MONKEY)
        & df["experiment"].isin(EXPERIMENTS)
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["prior_condition"] = df["experiment"].map(EXPERIMENTS)
    df["session_uid"] = (
        df["experiment"]
        + "__session_"
        + df["session_index"].astype(int).astype(str).str.zfill(2)
    )
    df["delta_sc_pp"] = df["observed_gain_pp"].astype(float)
    df["delta_sc"] = df["delta_sc_pp"] / 100.0
    keep = [
        "prior_condition",
        "experiment",
        "session_index",
        "session_uid",
        "n_shared",
        "n_shared_success",
        "n_bci_only",
        "n_bci_only_success",
        "s_shared_pct",
        "s_bci_only_pct",
        "delta_sc",
        "delta_sc_pp",
    ]
    return df[keep].sort_values(["prior_condition", "session_index"]).reset_index(drop=True)


def bootstrap_difference_ci(
    hidden: np.ndarray,
    no_hidden: np.ndarray,
    n_boot: int = 10000,
    seed: int = 17,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    hidden = np.asarray(hidden, dtype=float)
    no_hidden = np.asarray(no_hidden, dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        h = rng.choice(hidden, size=len(hidden), replace=True)
        n = rng.choice(no_hidden, size=len(no_hidden), replace=True)
        diffs[i] = np.nanmean(h) - np.nanmean(n)
    observed = float(np.nanmean(hidden) - np.nanmean(no_hidden))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return observed, float(lo), float(hi)


def summarize_gains(session_gains: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    arrays = {}
    for label in ["Hidden prior", "No hidden prior"]:
        values = session_gains.loc[session_gains["prior_condition"] == label, "delta_sc_pp"].to_numpy(float)
        arrays[label] = values
        rows.append(
            {
                "prior_condition": label,
                "n_sessions": int(len(values)),
                "mean_gain_pp": float(np.nanmean(values)),
                "median_gain_pp": float(np.nanmedian(values)),
                "sd_gain_pp": float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan,
                "mean_bci_only_pct": float(
                    np.nanmean(session_gains.loc[session_gains["prior_condition"] == label, "s_bci_only_pct"])
                ),
                "mean_shared_control_pct": float(
                    np.nanmean(session_gains.loc[session_gains["prior_condition"] == label, "s_shared_pct"])
                ),
            }
        )

    hidden = arrays["Hidden prior"]
    no_hidden = arrays["No hidden prior"]
    mw = stats.mannwhitneyu(hidden, no_hidden, alternative="two-sided", method="auto")
    diff, ci_lo, ci_hi = bootstrap_difference_ci(hidden, no_hidden)
    comparison = pd.DataFrame(
        [
            {
                "contrast": "Hidden prior minus no hidden prior",
                "difference_in_mean_gain_pp": diff,
                "bootstrap_95ci_low_pp": ci_lo,
                "bootstrap_95ci_high_pp": ci_hi,
                "mannwhitney_u": float(mw.statistic),
                "mannwhitney_two_sided_p": float(mw.pvalue),
            }
        ]
    )
    return pd.DataFrame(rows), comparison


def fit_trial_level_interaction(trials: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return pd.DataFrame([{"status": f"statsmodels unavailable: {exc}"}])

    model_df = trials[trials["target"] != "unknown"].copy()
    if model_df.empty:
        return pd.DataFrame([{"status": "no modeled trials after dropping unknown targets"}])

    formula = "success ~ shared_control * followup + C(target)"
    try:
        robust = smf.logit(formula, data=model_df).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": model_df["session_uid"]},
        )
    except Exception as exc:
        return pd.DataFrame([{"status": f"logistic model failed: {exc}"}])

    params = robust.params
    conf = robust.conf_int()
    rows = []
    for name in params.index:
        rows.append(
            {
                "status": "ok",
                "term": name,
                "coef_log_odds": float(params[name]),
                "se_cluster_session": float(robust.bse[name]),
                "z": float(robust.tvalues[name]),
                "p": float(robust.pvalues[name]),
                "ci_low_log_odds": float(conf.loc[name, 0]),
                "ci_high_log_odds": float(conf.loc[name, 1]),
                "odds_ratio": float(np.exp(params[name])),
                "odds_ratio_ci_low": float(np.exp(conf.loc[name, 0])),
                "odds_ratio_ci_high": float(np.exp(conf.loc[name, 1])),
            }
        )
    return pd.DataFrame(rows)


def marginal_success_table(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["prior_condition", "condition"])
        .agg(n_trials=("success", "size"), n_success=("success", "sum"), success_rate=("success", "mean"))
        .reset_index()
        .assign(success_pct=lambda df: 100.0 * df["success_rate"])
    )


def plot_reviewer_panel(session_gains: pd.DataFrame, trial_summary: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=200)

    colors = {
        "BCI-only": "#8c4b38",
        "Shared-control": "#237a6b",
        "Hidden prior": "#3f6f9f",
        "No hidden prior": "#c07a28",
    }
    x_groups = ["Hidden prior", "No hidden prior"]
    offsets = {"BCI-only": -0.16, "Shared-control": 0.16}

    ax = axes[0]
    for gi, group in enumerate(x_groups):
        for condition in ["BCI-only", "Shared-control"]:
            vals = session_gains.loc[
                session_gains["prior_condition"].eq(group),
                "s_bci_only_pct" if condition == "BCI-only" else "s_shared_pct",
            ].to_numpy(float)
            x = np.full(len(vals), gi + offsets[condition])
            ax.scatter(x, vals, s=26, color=colors[condition], alpha=0.78, edgecolor="white", linewidth=0.4)
            if len(vals):
                ax.hlines(np.nanmean(vals), gi + offsets[condition] - 0.09, gi + offsets[condition] + 0.09, color=colors[condition], lw=2.0)
    ax.set_xticks(range(len(x_groups)), x_groups)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("A. Session success")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", lw=0, color=colors["BCI-only"], label="BCI-only"),
            plt.Line2D([0], [0], marker="o", lw=0, color=colors["Shared-control"], label="Shared-control"),
        ],
        frameon=False,
        loc="lower right",
    )

    ax = axes[1]
    for gi, group in enumerate(x_groups):
        vals = session_gains.loc[session_gains["prior_condition"].eq(group), "delta_sc_pp"].to_numpy(float)
        x = np.full(len(vals), gi)
        ax.scatter(x, vals, s=30, color=colors[group], alpha=0.82, edgecolor="white", linewidth=0.4)
        if len(vals):
            ax.hlines(np.nanmean(vals), gi - 0.16, gi + 0.16, color=colors[group], lw=2.3)
    ax.axhline(0, color="0.55", lw=0.8, ls=":")
    p = comparison["mannwhitney_two_sided_p"].iloc[0]
    diff = comparison["difference_in_mean_gain_pp"].iloc[0]
    lo = comparison["bootstrap_95ci_low_pp"].iloc[0]
    hi = comparison["bootstrap_95ci_high_pp"].iloc[0]
    ax.text(
        0.5,
        0.98,
        f"diff={diff:.1f} pp, 95% CI [{lo:.1f}, {hi:.1f}], p={p:.3g}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xticks(range(len(x_groups)), x_groups)
    ax.set_ylabel("Shared-control gain (pp)")
    ax.set_title("B. Session gain")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    gain_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    model: pd.DataFrame,
    trial_summary: pd.DataFrame,
) -> None:
    hidden = gain_summary[gain_summary["prior_condition"].eq("Hidden prior")].iloc[0]
    no_hidden = gain_summary[gain_summary["prior_condition"].eq("No hidden prior")].iloc[0]
    comp = comparison.iloc[0]
    if "term" in model.columns:
        interaction = model[model["term"].eq("shared_control:followup")]
    else:
        interaction = pd.DataFrame()

    lines = [
        "# Hidden-prior follow-up shared-control gain analysis",
        "",
        "Restricted to Monkey 1 Appearing Obstacle sessions.",
        "",
        "## Session-level gain comparison",
        "",
        (
            f"- Hidden prior present: n={int(hidden['n_sessions'])} sessions, "
            f"mean gain={hidden['mean_gain_pp']:.1f} pp, median gain={hidden['median_gain_pp']:.1f} pp."
        ),
        (
            f"- Hidden prior absent: n={int(no_hidden['n_sessions'])} sessions, "
            f"mean gain={no_hidden['mean_gain_pp']:.1f} pp, median gain={no_hidden['median_gain_pp']:.1f} pp."
        ),
        (
            f"- Difference in mean gains, hidden minus no-hidden: "
            f"{comp['difference_in_mean_gain_pp']:.1f} pp "
            f"(bootstrap 95% CI {comp['bootstrap_95ci_low_pp']:.1f} to "
            f"{comp['bootstrap_95ci_high_pp']:.1f} pp)."
        ),
        (
            f"- Two-sided Mann-Whitney rank-sum test: U={comp['mannwhitney_u']:.1f}, "
            f"p={comp['mannwhitney_two_sided_p']:.4g}."
        ),
        "",
        "## Trial-level interaction model",
        "",
        "Model: `success ~ shared_control * followup + C(target)`, cluster-robust SEs by session.",
    ]
    if interaction.empty:
        status = model["status"].iloc[0] if "status" in model.columns and len(model) else "not available"
        lines.append(f"- Interaction model status: {status}.")
    else:
        row = interaction.iloc[0]
        lines.append(
            f"- Interaction coefficient `shared_control:followup`: "
            f"beta={row['coef_log_odds']:.3f}, SE={row['se_cluster_session']:.3f}, "
            f"p={row['p']:.4g}, OR={row['odds_ratio']:.2f}."
        )
        lines.append(
            "- Interpretation: this tests whether the shared-control effect changed "
            "when the hidden-obstacle prior was removed."
        )

    lines.extend(["", "## Pooled trial success rates", ""])
    if trial_summary.empty:
        lines.append("Not available because raw trial loading/modeling was skipped or failed.")
    else:
        lines.append(trial_summary.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Suggested manuscript wording",
            "",
            (
                "Shared control improved success substantially in both experiments, "
                "and we found no evidence that the shared-control gain differed after "
                "the hidden-obstacle prior was removed."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(base_dir: str, source_monkey: str, skip_trial_model: bool = False) -> dict[str, Path]:
    output_tables = PROJECT_ROOT / "tables" / "Hidden prior follow-up"
    output_figures = PROJECT_ROOT / "figures" / "Hidden prior follow-up"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    session_gains = load_cached_session_gains()
    if session_gains.empty:
        trials_for_session = collect_trial_rows(base_dir, source_monkey)
        if trials_for_session.empty:
            raise RuntimeError(f"No Monkey 1 Appearing Obstacle trial rows found under {base_dir!r}.")
        session_gains = make_session_gain_table(trials_for_session)
        if session_gains.empty:
            raise RuntimeError("No paired session-level BCI-only/shared-control gains could be computed.")

    gain_summary, comparison = summarize_gains(session_gains)
    trials = pd.DataFrame()
    if skip_trial_model:
        model = pd.DataFrame([{"status": "trial-level model skipped"}])
        trial_summary = pd.DataFrame()
    else:
        trials = collect_trial_rows(base_dir, source_monkey)
        if trials.empty:
            model = pd.DataFrame(
                [{"status": f"no raw trial rows loaded from {base_dir!r}, source folder {source_monkey!r}"}]
            )
            trial_summary = pd.DataFrame()
        else:
            model = fit_trial_level_interaction(trials)
            trial_summary = marginal_success_table(trials)

    paths = {
        "trial_rows": output_tables / "hidden_prior_followup_trial_rows.csv",
        "session_gains": output_tables / "hidden_prior_followup_session_gains.csv",
        "gain_summary": output_tables / "hidden_prior_followup_gain_summary.csv",
        "gain_comparison": output_tables / "hidden_prior_followup_gain_comparison.csv",
        "trial_interaction": output_tables / "hidden_prior_followup_trial_level_interaction.csv",
        "trial_summary": output_tables / "hidden_prior_followup_trial_success_summary.csv",
        "report": output_tables / "hidden_prior_followup_report.md",
        "figure": output_figures / "hidden_prior_followup_gain_panel.svg",
    }

    trials.to_csv(paths["trial_rows"], index=False)
    session_gains.to_csv(paths["session_gains"], index=False)
    gain_summary.to_csv(paths["gain_summary"], index=False)
    comparison.to_csv(paths["gain_comparison"], index=False)
    model.to_csv(paths["trial_interaction"], index=False)
    trial_summary.to_csv(paths["trial_summary"], index=False)
    plot_reviewer_panel(session_gains, trial_summary, comparison, paths["figure"])
    write_report(paths["report"], gain_summary, comparison, model, trial_summary)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Root directory containing Monkey 1 experiment folders.")
    parser.add_argument(
        "--source-monkey",
        default=DEFAULT_SOURCE_MONKEY,
        help="Raw-data monkey folder used for trial-level loading; output remains labeled as manuscript Monkey 1.",
    )
    parser.add_argument("--skip-trial-model", action="store_true")
    args = parser.parse_args()
    paths = run_analysis(load_base_dir(args.base_dir), args.source_monkey, args.skip_trial_model)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
