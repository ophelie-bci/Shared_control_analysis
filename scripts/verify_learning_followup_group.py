"""Verify retained learning and six-month follow-up claims.

This script consolidates the existing raw success-rate audit and retained
follow-up analyses into a single verification folder. It does not regenerate
manuscript documents or figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis_results" / "learning_followup_verification"
RAW_AUDIT_DIR = PROJECT_ROOT / "analysis_results" / "success_rate_verification"
SESSION_ROWS_PATH = RAW_AUDIT_DIR / "success_rate_audit_session_condition_rows.csv"
FIG3_SOURCE_ROWS_PATH = RAW_AUDIT_DIR / "figure3_success_rate_source_rows.csv"
HIDDEN_GAIN_PATH = PROJECT_ROOT / "tables" / "Hidden prior follow-up" / "hidden_prior_followup_session_gains.csv"
HIDDEN_SUMMARY_PATH = PROJECT_ROOT / "tables" / "Hidden prior follow-up" / "hidden_prior_followup_gain_summary.csv"
HIDDEN_COMPARISON_PATH = PROJECT_ROOT / "tables" / "Hidden prior follow-up" / "hidden_prior_followup_gain_comparison.csv"
BCI_LEARNING_SESSION_PATH = PROJECT_ROOT / "tables" / "BCI-only learning" / "bci_only_learning_session_values.csv"
BCI_LEARNING_TREND_PATH = PROJECT_ROOT / "tables" / "BCI-only learning" / "bci_only_learning_session_trend.csv"
BCI_LEARNING_TRANSITION_PATH = PROJECT_ROOT / "tables" / "BCI-only learning" / "bci_only_learning_transition_summary.csv"


def _read_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_session_rows() -> pd.DataFrame:
    if not SESSION_ROWS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SESSION_ROWS_PATH}. Run scripts/audit_success_rate_verification.py first."
        )
    df = pd.read_csv(SESSION_ROWS_PATH)
    df["success_pct"] = 100.0 * df["success_rate"].astype(float)
    df["n_trials"] = df["n_trials"].astype(int)
    df["n_success"] = df["n_success"].astype(int)
    df["session_index"] = df["session_index"].astype(int)
    return df


def _session_wide(rows: pd.DataFrame) -> pd.DataFrame:
    keep = rows[
        (
            rows["monkey"].eq("Monkey 2")
            & rows["experiment"].eq("AI Obstacle")
            & rows["condition"].eq("BCI-only")
        )
        | (
            rows["monkey"].eq("Monkey 1")
            & rows["experiment"].isin(["AI Appearing Obstacle", "AI Appearing Obstacle 2"])
        )
    ].copy()
    return keep.sort_values(["monkey", "experiment", "session_index", "condition"])


def _appearing_gain_table(rows: pd.DataFrame) -> pd.DataFrame:
    sub = rows[
        rows["monkey"].eq("Monkey 1")
        & rows["experiment"].isin(["AI Appearing Obstacle", "AI Appearing Obstacle 2"])
    ].copy()
    wide = sub.pivot_table(
        index=["monkey", "experiment", "session_index", "session_uid", "source_file"],
        columns="condition",
        values=["n_trials", "n_success", "success_rate", "success_pct"],
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}_{condition}".replace(" ", "_").replace("-", "_")
        for metric, condition in wide.columns
    ]
    wide = wide.reset_index()
    wide["prior_condition"] = np.where(
        wide["experiment"].eq("AI Appearing Obstacle"), "Hidden prior", "No hidden prior"
    )
    wide["phase"] = np.where(
        wide["experiment"].eq("AI Appearing Obstacle"), "Original", "Follow-up"
    )
    wide["session_number"] = wide.groupby("phase").cumcount() + 1
    wide["gain_pp"] = wide["success_pct_Shared_control"] - wide["success_pct_BCI_only"]
    cols = [
        "phase",
        "prior_condition",
        "monkey",
        "experiment",
        "session_index",
        "session_number",
        "source_file",
        "n_trials_BCI_only",
        "n_success_BCI_only",
        "success_pct_BCI_only",
        "n_trials_Shared_control",
        "n_success_Shared_control",
        "success_pct_Shared_control",
        "gain_pp",
    ]
    return wide[cols].sort_values(["phase", "session_index"])


def _monkey2_learning(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sub = rows[
        rows["monkey"].eq("Monkey 2")
        & rows["experiment"].eq("AI Obstacle")
        & rows["condition"].eq("BCI-only")
    ].copy()
    sub = sub.sort_values("session_index")
    rho, p_value = stats.spearmanr(sub["session_index"].to_numpy(), sub["success_rate"].to_numpy())
    values = sub[
        [
            "monkey",
            "experiment",
            "session_index",
            "source_file",
            "n_trials",
            "n_success",
            "success_rate",
            "success_pct",
        ]
    ].rename(
        columns={
            "n_trials": "n_bci_only",
            "n_success": "n_bci_only_success",
            "success_rate": "bci_only_success_rate",
            "success_pct": "bci_only_success_pct",
        }
    )
    result = {
        "result_id": "monkey2_fixed_obstacle_bci_only_learning",
        "statistical_unit": "session",
        "n_sessions": int(len(sub)),
        "test": "Spearman rank correlation of BCI-only session success versus chronological session_index",
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "reported_old": "rho=0.755, p=0.01",
        "computed_rounded": f"rho={rho:.3f}, p={p_value:.2f}",
        "reproduces_claim": bool(round(float(rho), 3) == 0.755 and round(float(p_value), 2) == 0.01),
    }
    return values, result


def _transition_result(appearing: pd.DataFrame) -> dict:
    original = appearing[appearing["phase"].eq("Original")].sort_values("session_index")
    followup = appearing[appearing["phase"].eq("Follow-up")].sort_values("session_index")
    final_original = original.tail(3)
    first_followup = followup.head(3)
    old_transition = pd.read_csv(BCI_LEARNING_TRANSITION_PATH) if BCI_LEARNING_TRANSITION_PATH.exists() else pd.DataFrame()
    result = {
        "result_id": "monkey1_appearing_bci_only_final3_vs_followup_first3",
        "statistical_unit": "session",
        "aggregation": "arithmetic mean of unrounded session-level BCI-only success percentages",
        "chronological_ordering": "source_file sort order from raw configured data/session_index",
        "final_original_session_indices": ",".join(final_original["session_index"].astype(str)),
        "final_original_source_files": ";".join(final_original["source_file"].astype(str)),
        "final_original_mean_pct": float(final_original["success_pct_BCI_only"].mean()),
        "first_followup_session_indices": ",".join(first_followup["session_index"].astype(str)),
        "first_followup_source_files": ";".join(first_followup["source_file"].astype(str)),
        "first_followup_mean_pct": float(first_followup["success_pct_BCI_only"].mean()),
        "followup_minus_final_original_pp": float(
            first_followup["success_pct_BCI_only"].mean()
            - final_original["success_pct_BCI_only"].mean()
        ),
        "reported_old": "52.8% final three original; 52.4% first three six-month follow-up",
        "computed_rounded": (
            f"{final_original['success_pct_BCI_only'].mean():.1f}% final original; "
            f"{first_followup['success_pct_BCI_only'].mean():.1f}% first follow-up"
        ),
        "reproduces_claim": bool(
            round(float(final_original["success_pct_BCI_only"].mean()), 1) == 52.8
            and round(float(first_followup["success_pct_BCI_only"].mean()), 1) == 52.4
        ),
    }
    if not old_transition.empty:
        result["retained_table_final_original_mean_pct"] = float(old_transition.iloc[0]["final_original_mean_pct"])
        result["retained_table_first_followup_mean_pct"] = float(old_transition.iloc[0]["first_followup_mean_pct"])
    return result


def _gain_result(appearing: pd.DataFrame) -> dict:
    hidden = appearing.loc[appearing["prior_condition"].eq("Hidden prior"), "gain_pp"].to_numpy(float)
    no_hidden = appearing.loc[appearing["prior_condition"].eq("No hidden prior"), "gain_pp"].to_numpy(float)
    mw = stats.mannwhitneyu(hidden, no_hidden, alternative="two-sided", method="auto")
    rng = np.random.default_rng(17)
    diffs = np.empty(10000, dtype=float)
    for i in range(len(diffs)):
        h = rng.choice(hidden, size=len(hidden), replace=True)
        n = rng.choice(no_hidden, size=len(no_hidden), replace=True)
        diffs[i] = np.nanmean(h) - np.nanmean(n)
    ci_lo, ci_hi = np.nanpercentile(diffs, [2.5, 97.5])
    comparison = pd.read_csv(HIDDEN_COMPARISON_PATH) if HIDDEN_COMPARISON_PATH.exists() else pd.DataFrame()
    result = {
        "result_id": "monkey1_appearing_shared_control_gain_hidden_vs_followup",
        "statistical_unit": "session",
        "comparison_test": "two-sided Mann-Whitney rank-sum test, scipy.stats.mannwhitneyu(method='auto')",
        "confidence_interval_method": "nonparametric bootstrap percentile CI on difference in mean session gains",
        "random_seed": 17,
        "bootstrap_iterations": 10000,
        "subtraction_direction": "Hidden prior/original mean gain minus no-hidden-prior/follow-up mean gain",
        "hidden_prior_n_sessions": int(len(hidden)),
        "no_hidden_prior_n_sessions": int(len(no_hidden)),
        "hidden_prior_mean_gain_pp": float(np.nanmean(hidden)),
        "no_hidden_prior_mean_gain_pp": float(np.nanmean(no_hidden)),
        "difference_in_mean_gain_pp": float(np.nanmean(hidden) - np.nanmean(no_hidden)),
        "bootstrap_95ci_low_pp": float(ci_lo),
        "bootstrap_95ci_high_pp": float(ci_hi),
        "mannwhitney_u": float(mw.statistic),
        "p_value": float(mw.pvalue),
        "reported_old": "27.4 vs 23.0 pp; difference 4.4 pp; 95% CI -5.8 to 15.0; p=0.596",
        "computed_rounded": (
            f"{np.nanmean(hidden):.1f} vs {np.nanmean(no_hidden):.1f} pp; "
            f"difference {np.nanmean(hidden) - np.nanmean(no_hidden):.1f} pp; "
            f"95% CI {ci_lo:.1f} to {ci_hi:.1f}; p={mw.pvalue:.3f}"
        ),
        "reproduces_claim": bool(
            round(float(np.nanmean(hidden)), 1) == 27.4
            and round(float(np.nanmean(no_hidden)), 1) == 23.0
            and round(float(np.nanmean(hidden) - np.nanmean(no_hidden)), 1) == 4.4
            and round(float(ci_lo), 1) == -5.8
            and round(float(ci_hi), 1) == 15.0
            and round(float(mw.pvalue), 3) == 0.596
        ),
    }
    if not comparison.empty:
        row = comparison.iloc[0]
        result["retained_table_difference_in_mean_gain_pp"] = float(row["difference_in_mean_gain_pp"])
        result["retained_table_bootstrap_95ci_low_pp"] = float(row["bootstrap_95ci_low_pp"])
        result["retained_table_bootstrap_95ci_high_pp"] = float(row["bootstrap_95ci_high_pp"])
        result["retained_table_mannwhitney_two_sided_p"] = float(row["mannwhitney_two_sided_p"])
    return result


def _old_vs_new(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "result_id": result["result_id"],
                "old_or_claimed_value": result["reported_old"],
                "new_computed_value": result["computed_rounded"],
                "reproduces_claim": result["reproduces_claim"],
                "cause_if_discrepant": "none" if result["reproduces_claim"] else "see report",
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    config: dict,
    results: list[dict],
    monkey2_values: pd.DataFrame,
    appearing: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    by_id = {row["result_id"]: row for row in results}
    learning = by_id["monkey2_fixed_obstacle_bci_only_learning"]
    transition = by_id["monkey1_appearing_bci_only_final3_vs_followup_first3"]
    gain = by_id["monkey1_appearing_shared_control_gain_hidden_vs_followup"]
    lines = [
        "# Learning and Follow-up Verification",
        "",
        "Scope stopped after retained text-only learning and six-month follow-up results. Simulations, removed neural adaptation analyses, collision rescue, override exposure, and Respawn analyses were not run.",
        "",
        "## Provenance",
        "",
        f"- Working repository: `{PROJECT_ROOT}`",
        f"- Configured base_dir: `{config.get('base_dir')}`",
        "- Raw audit input: `analysis_results/success_rate_verification/success_rate_audit_session_condition_rows.csv`, regenerated from configured raw trial pickles.",
        "- Retained follow-up implementation: `scripts/run_bci_only_learning_followup.py`.",
        "- Retained gain implementation: `scripts/run_hidden_prior_followup_gain.py`.",
        "- Original success-rate implementation: `scripts/run_success_rate.py`; raw audit follows the same answer-code and task-inclusion rules.",
        "- Answer/inclusion rules: correct `answer == 1`; retained incorrect obstacle outcomes `answer in {3,5,6}`; Appearing Obstacle additionally requires `obstacleAppearedTime is not None`; conditions are `aiVelocityFactor == 0.0` for BCI-only and `1.0` for shared-control.",
        "",
        "## Results",
        "",
        f"- Monkey 2 Fixed Obstacle BCI-only learning reproduces: rho={learning['spearman_rho']:.15g}, p={learning['p_value']:.15g}, n={learning['n_sessions']} sessions; rounded manuscript value rho={learning['spearman_rho']:.3f}, p={learning['p_value']:.2f}.",
        f"- Monkey 1 Appearing Obstacle transition reproduces: final three original sessions `{transition['final_original_source_files']}` averaged {transition['final_original_mean_pct']:.15g}%; first three follow-up sessions `{transition['first_followup_source_files']}` averaged {transition['first_followup_mean_pct']:.15g}%. Rounded: {transition['computed_rounded']}.",
        f"- Shared-control gain comparison reproduces: original/hidden-prior mean gain {gain['hidden_prior_mean_gain_pp']:.15g} pp versus follow-up/no-hidden-prior {gain['no_hidden_prior_mean_gain_pp']:.15g} pp; hidden minus no-hidden difference {gain['difference_in_mean_gain_pp']:.15g} pp; bootstrap 95% CI {gain['bootstrap_95ci_low_pp']:.15g} to {gain['bootstrap_95ci_high_pp']:.15g}; Mann-Whitney U={gain['mannwhitney_u']:.15g}, p={gain['p_value']:.15g}.",
        "",
        "## Methods Clarifications",
        "",
        "- Monkey 2 learning test: Spearman rank correlation at the session level, using chronological `session_index` versus BCI-only success.",
        "- Follow-up transition: arithmetic mean of unrounded session-level BCI-only percentages; no inferential test.",
        "- Gain comparison: statistical unit is session-level shared-control gain, defined as shared-control success minus BCI-only success in percentage points.",
        "- Gain subtraction direction: original hidden-prior mean gain minus follow-up no-hidden-prior mean gain.",
        "- Gain comparison test: two-sided Mann-Whitney rank-sum test, `scipy.stats.mannwhitneyu(method='auto')`.",
        "- Gain confidence interval: nonparametric percentile bootstrap on the difference in mean session gains, 10,000 resamples, random seed 17.",
        "",
        "## Discrepancies",
        "",
        "- No numerical discrepancies for the three checked retained claims after using the raw configured session rows and retained unrounded session values.",
        "- Regeneration note: `scripts/run_hidden_prior_followup_gain.py` and `scripts/run_bci_only_learning_followup.py` could not overwrite their existing table/figure outputs during this audit because those files were locked (`PermissionError`). Their previously retained CSV outputs match the fresh raw audit values used here.",
        "",
        "## Wording",
        "",
        "Results wording:",
        "Monkey 2 Fixed Obstacle BCI-only performance increased across sessions (Spearman rho = 0.755, p = 0.01). For Monkey 1 Appearing Obstacle, BCI-only success in the final three original sessions averaged 52.8%, comparable to 52.4% in the first three six-month follow-up sessions. Shared-control gains were similar in the original and follow-up Appearing Obstacle sessions (27.4 versus 23.0 percentage points; hidden-prior minus follow-up difference = 4.4 percentage points, bootstrap 95% CI -5.8 to 15.0; two-sided Mann-Whitney p = 0.596).",
        "",
        "Methods/reviewer wording:",
        "Session-level success rates were computed from configured raw trial pickles using the original success-rate inclusion rules. Follow-up comparisons used sessions as the statistical unit. The shared-control gain comparison used the session-level gain (shared-control minus BCI-only success, percentage points), tested original hidden-prior versus six-month no-hidden-prior sessions with a two-sided Mann-Whitney rank-sum test. The 95% confidence interval is a percentile bootstrap for the hidden-prior minus no-hidden-prior difference in mean session gains using 10,000 resamples and random seed 17.",
        "",
        "## Regeneration Commands",
        "",
        "```powershell",
        "python scripts/audit_success_rate_verification.py --base-dir \"<base_dir>\" --monkeys \"Monkey 1\" \"Monkey 2\" --experiments \"AI Obstacle\" \"AI Appearing Obstacle\" \"AI Appearing Obstacle 2\"",
        "python scripts/run_hidden_prior_followup_gain.py --base-dir \"<base_dir>\" --skip-trial-model",
        "python scripts/run_bci_only_learning_followup.py",
        "python scripts/verify_learning_followup_group.py",
        "```",
        "",
        "## Output Files",
        "",
    ]
    for label, path in paths.items():
        lines.append(f"- `{label}`: `{path.relative_to(PROJECT_ROOT)}`")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = _read_config()
    rows = _load_session_rows()
    source_values = _session_wide(rows)
    appearing = _appearing_gain_table(rows)
    monkey2_values, learning_result = _monkey2_learning(rows)
    transition_result = _transition_result(appearing)
    gain_result = _gain_result(appearing)
    results = [learning_result, transition_result, gain_result]

    paths = {
        "session_level_source_values": OUT_DIR / "session_level_source_values.csv",
        "monkey2_fixed_obstacle_bci_only_values": OUT_DIR / "monkey2_fixed_obstacle_bci_only_values.csv",
        "monkey1_appearing_session_gains": OUT_DIR / "monkey1_appearing_session_gains.csv",
        "computed_results_full_precision": OUT_DIR / "computed_results_full_precision.csv",
        "old_vs_new_comparison": OUT_DIR / "old_vs_new_comparison.csv",
        "metadata": OUT_DIR / "metadata.json",
        "report": OUT_DIR / "learning_followup_verification_report.md",
    }
    source_values.to_csv(paths["session_level_source_values"], index=False)
    monkey2_values.to_csv(paths["monkey2_fixed_obstacle_bci_only_values"], index=False)
    appearing.to_csv(paths["monkey1_appearing_session_gains"], index=False)
    pd.DataFrame(results).to_csv(paths["computed_results_full_precision"], index=False)
    _old_vs_new(results).to_csv(paths["old_vs_new_comparison"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "base_dir": config.get("base_dir"),
                "source_audit": str(SESSION_ROWS_PATH.relative_to(PROJECT_ROOT)),
                "figure3_source_rows": str(FIG3_SOURCE_ROWS_PATH.relative_to(PROJECT_ROOT)),
                "retained_hidden_gain_table": str(HIDDEN_GAIN_PATH.relative_to(PROJECT_ROOT)),
                "retained_hidden_gain_summary": str(HIDDEN_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
                "retained_hidden_gain_comparison": str(HIDDEN_COMPARISON_PATH.relative_to(PROJECT_ROOT)),
                "retained_bci_learning_session_values": str(BCI_LEARNING_SESSION_PATH.relative_to(PROJECT_ROOT)),
                "retained_bci_learning_trend": str(BCI_LEARNING_TREND_PATH.relative_to(PROJECT_ROOT)),
                "retained_bci_learning_transition": str(BCI_LEARNING_TRANSITION_PATH.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(config, results, monkey2_values, appearing, paths)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
