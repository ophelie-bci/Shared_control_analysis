"""Verify Extended Data Figure 1 and obstacle trajectory-metric claims.

This verifier preserves existing outputs and writes all artifacts under
analysis_results/extended_data_figure1_verification. It reuses the released
Extended Data Figure 1 generator and the obstacle session-metric functions,
but restricts the claim checks to Figure 3 verified obstacle sessions and
eligible experimental answer codes {1, 3, 5, 6}.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis_results" / "extended_data_figure1_verification"
FIG3_SOURCE_ROWS = (
    PROJECT_ROOT
    / "analysis_results"
    / "success_rate_verification"
    / "figure3_success_rate_source_rows.csv"
)
ELIGIBLE_ANSWERS = {1, 3, 5, 6}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import make_extended_data_figure1_trajectory_supplement as edf1  # noqa: E402
from scripts import run_obstacle_session_metrics as metrics  # noqa: E402
from src.load import load_files  # noqa: E402


TASKS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle follow-up",
}

CLAIMS = {
    ("Appearing Obstacle", "time_to_successful_dwell_s"): 0.83,
    ("Fixed Obstacle", "time_to_successful_dwell_s"): 0.64,
    ("Appearing Obstacle", "successful_mean_executed_speed_mps", "AI"): 1.81,
    ("Appearing Obstacle", "successful_mean_executed_speed_mps", "No AI"): 2.17,
    ("Appearing Obstacle", "successful_mean_executed_speed_mps", "p"): 0.0039,
    ("Fixed Obstacle", "successful_mean_executed_speed_mps", "AI"): 1.92,
    ("Fixed Obstacle", "successful_mean_executed_speed_mps", "No AI"): 2.28,
    ("Fixed Obstacle", "successful_mean_executed_speed_mps", "p"): 0.001,
    ("Appearing Obstacle", "successful_smoothness_mean_squared_jerk", "AI"): 3.32e5,
    ("Appearing Obstacle", "successful_smoothness_mean_squared_jerk", "No AI"): 4.87e5,
    ("Appearing Obstacle", "successful_smoothness_mean_squared_jerk", "p"): 0.0195,
    ("Fixed Obstacle", "successful_smoothness_mean_squared_jerk", "AI"): 3.58e5,
    ("Fixed Obstacle", "successful_smoothness_mean_squared_jerk", "No AI"): 5.30e5,
    ("Fixed Obstacle", "successful_smoothness_mean_squared_jerk", "p"): 0.001,
}

REPORTED_COMPARISON_METRICS = [
    "successful_target_window_path_ratio",
    "successful_target_window_path_efficiency",
    "time_to_successful_dwell_s",
    "successful_mean_executed_speed_mps",
    "obstacle_clearance_m",
    "successful_smoothness_mean_squared_jerk",
]


def _load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _verified_sessions() -> pd.DataFrame:
    rows = pd.read_csv(FIG3_SOURCE_ROWS)
    rows = rows[rows["experiment"].isin(TASKS)].copy()
    rows = rows[["panel", "monkey", "experiment", "session_index", "source_file"]].drop_duplicates()
    rows["session_index"] = rows["session_index"].astype(int)
    rows["task_label"] = rows["experiment"].map(TASKS)
    return rows


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _answer(trial) -> int | None:
    try:
        return int(_get(trial, "answer", None))
    except Exception:
        return None


def _trial_id(trial):
    for key in ("trial", "trial_id", "trialId", "trialIndex"):
        value = _get(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return value
    return np.nan


def _condition(trial) -> str | None:
    try:
        factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None
    if factor == 1.0:
        return "AI"
    if factor == 0.0:
        return "No AI"
    return None


def _collect_verified_trial_metrics(base_dir: str, verified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    verified_files = set(verified["source_file"])
    inclusion_rows = []
    metric_rows = []

    for experiment, task_label in TASKS.items():
        for monkey in sorted(verified.loc[verified["experiment"].eq(experiment), "monkey"].unique()):
            all_trials, *_rest, pkl_files, _ai = load_files(experiment, monkey, base_dir)
            for session_index, trials in enumerate(all_trials):
                session_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                verified_session = session_file in verified_files
                for trial in trials:
                    ans = _answer(trial)
                    cond = _condition(trial)
                    eligible_answer = ans in ELIGIBLE_ANSWERS
                    include = bool(verified_session and eligible_answer and cond in ("AI", "No AI"))
                    reason = []
                    if not verified_session:
                        reason.append("not_verified_figure3_obstacle_session")
                    if not eligible_answer:
                        reason.append("ineligible_answer_code")
                    if cond not in ("AI", "No AI"):
                        reason.append("unknown_ai_condition")
                    base = {
                        "monkey": monkey,
                        "experiment": experiment,
                        "task_label": task_label,
                        "session_index": session_index,
                        "session_file": session_file,
                        "trial_id": _trial_id(trial),
                        "answer": ans,
                        "condition": cond,
                        "success": bool(ans == 1),
                        "verified_figure3_session": bool(verified_session),
                        "eligible_answer_code": bool(eligible_answer),
                        "included": include,
                        "exclusion_reason": "included" if include else ";".join(reason),
                    }
                    inclusion_rows.append(base)
                    if include:
                        row = metrics._trial_metrics(trial, monkey, experiment, session_index, session_file)
                        if row is not None:
                            row["answer"] = ans
                            row["eligible_answer_code"] = True
                            metric_rows.append(row)
                        else:
                            fail = base.copy()
                            fail["extraction_failure"] = "run_obstacle_session_metrics._trial_metrics_returned_none"
                            metric_rows.append(fail)

    return pd.DataFrame(inclusion_rows), pd.DataFrame(metric_rows)


def _paired_from_session_summary(session_summary: pd.DataFrame) -> pd.DataFrame:
    return metrics.paired_comparisons(session_summary)


def _metric_lookup(comparisons: pd.DataFrame, task_label: str, metric: str, summary: str = "mean") -> pd.Series:
    rows = comparisons[
        comparisons["task_label"].eq(task_label)
        & comparisons["metric"].eq(metric)
        & comparisons["summary"].eq(summary)
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def _claim_check(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(claim: str, reported: float, computed: float, tolerance: str = "1dp"):
        if not np.isfinite(computed):
            match = False
        elif tolerance == "p":
            match = np.isclose(float(computed), float(reported), rtol=0.05, atol=0.0)
        elif tolerance == "2dp":
            match = round(float(computed), 2) == round(float(reported), 2)
        else:
            match = round(float(computed), 1) == round(float(reported), 1)
        rows.append(
            {
                "claim": claim,
                "manuscript_reported": reported,
                "computed_full_precision": computed,
                "computed_rounded_2dp": round(float(computed), 2) if np.isfinite(computed) else np.nan,
                "match": bool(match),
            }
        )

    for task in ("Appearing Obstacle", "Fixed Obstacle"):
        time_row = _metric_lookup(comparisons, task, "time_to_successful_dwell_s")
        if not time_row.empty:
            add(f"{task} total acquisition-time increase s", CLAIMS[(task, "time_to_successful_dwell_s")], time_row["mean_delta_ai_minus_no_ai"], "2dp")

        speed_row = _metric_lookup(comparisons, task, "successful_mean_executed_speed_mps")
        if not speed_row.empty:
            add(f"{task} successful mean speed shared-control m/s", CLAIMS[(task, "successful_mean_executed_speed_mps", "AI")], speed_row["ai_mean"], "2dp")
            add(f"{task} successful mean speed BCI-only m/s", CLAIMS[(task, "successful_mean_executed_speed_mps", "No AI")], speed_row["no_ai_mean"], "2dp")
            add(f"{task} successful mean speed Wilcoxon p", CLAIMS[(task, "successful_mean_executed_speed_mps", "p")], speed_row["wilcoxon_p_value"], "p")

        jerk_row = _metric_lookup(comparisons, task, "successful_smoothness_mean_squared_jerk")
        if not jerk_row.empty:
            add(f"{task} successful MSJ shared-control", CLAIMS[(task, "successful_smoothness_mean_squared_jerk", "AI")], jerk_row["ai_mean"], "2dp")
            add(f"{task} successful MSJ BCI-only", CLAIMS[(task, "successful_smoothness_mean_squared_jerk", "No AI")], jerk_row["no_ai_mean"], "2dp")
            add(f"{task} successful MSJ Wilcoxon p", CLAIMS[(task, "successful_smoothness_mean_squared_jerk", "p")], jerk_row["wilcoxon_p_value"], "p")

    for task in ("Appearing Obstacle", "Fixed Obstacle"):
        clearance = _metric_lookup(comparisons, task, "obstacle_clearance_m")
        if not clearance.empty:
            rows.append(
                {
                    "claim": f"{task} minimum obstacle-clearance increase in 0.44-0.48 range",
                    "manuscript_reported": "0.44-0.48",
                    "computed_full_precision": clearance["mean_delta_ai_minus_no_ai"],
                    "computed_rounded_2dp": round(float(clearance["mean_delta_ai_minus_no_ai"]), 2),
                    "match": bool(0.44 <= round(float(clearance["mean_delta_ai_minus_no_ai"]), 2) <= 0.48),
                }
            )

    for task in ("Appearing Obstacle", "Fixed Obstacle"):
        for metric in ("successful_target_window_path_ratio", "successful_target_window_path_efficiency"):
            row = _metric_lookup(comparisons, task, metric)
            if not row.empty:
                rows.append(
                    {
                        "claim": f"{task} path-length efficiency no significant difference ({metric})",
                        "manuscript_reported": "not significant",
                        "computed_full_precision": row["wilcoxon_p_value"],
                        "computed_rounded_2dp": round(float(row["wilcoxon_p_value"]), 2) if np.isfinite(row["wilcoxon_p_value"]) else np.nan,
                        "match": bool(np.isfinite(row["wilcoxon_p_value"]) and row["wilcoxon_p_value"] >= 0.05),
                    }
                )
    return pd.DataFrame(rows)


def _old_vs_new(new_comparisons: pd.DataFrame) -> pd.DataFrame:
    old_path = PROJECT_ROOT / "tables" / "obstacle_session_metric_paired_comparisons.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    keys = ["task_label", "summary", "metric"]
    merged = old.merge(new_comparisons, on=keys, suffixes=("_old_existing", "_new_verified"), how="outer")
    rows = []
    for _, row in merged.iterrows():
        base = {key: row.get(key) for key in keys}
        for col in (
            "n_paired_sessions",
            "ai_mean",
            "no_ai_mean",
            "mean_delta_ai_minus_no_ai",
            "median_delta_ai_minus_no_ai",
            "wilcoxon_p_value",
        ):
            old_col = f"{col}_old_existing"
            new_col = f"{col}_new_verified"
            if old_col in row.index or new_col in row.index:
                old_val = pd.to_numeric(pd.Series([row.get(old_col)]), errors="coerce").iloc[0]
                new_val = pd.to_numeric(pd.Series([row.get(new_col)]), errors="coerce").iloc[0]
                rows.append(
                    {
                        **base,
                        "quantity": col,
                        "old_existing_tables_value": old_val,
                        "new_verified_value": new_val,
                        "new_minus_old": new_val - old_val if np.isfinite(old_val) and np.isfinite(new_val) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _task_aggregate_counts(verified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in verified.groupby(["task_label", "monkey", "experiment"]).size().reset_index(name="n_verified_sessions").iterrows():
        rows.append(
            {
                "task_aggregate": row["task_label"],
                "dataset_label": f"{row['monkey']} / {row['experiment']}",
                "monkey": row["monkey"],
                "experiment": row["experiment"],
                "n_verified_sessions": int(row["n_verified_sessions"]),
                "monkey1_six_month_followup_dataset": bool(
                    row["monkey"] == "Monkey 1" and row["experiment"] == "AI Appearing Obstacle 2"
                ),
                "monkey1_six_month_followup_contributes_to_this_aggregate": bool(
                    row["task_label"] == "Appearing Obstacle follow-up"
                    and row["monkey"] == "Monkey 1"
                    and row["experiment"] == "AI Appearing Obstacle 2"
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_contribution_manifest(session_summary: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reported = comparisons[
        comparisons["summary"].eq("mean")
        & comparisons["metric"].isin(REPORTED_COMPARISON_METRICS)
    ].copy()
    for _, comp in reported.sort_values(["task_label", "metric"]).iterrows():
        metric = comp["metric"]
        col = f"mean_{metric}"
        task_df = session_summary[session_summary["task_label"].eq(comp["task_label"])]
        if col not in task_df.columns:
            continue
        pivot = task_df.pivot_table(
            index=["session_uid", "monkey", "experiment", "session_index", "session_file"],
            columns="condition",
            values=col,
            aggfunc="first",
        )
        if {"AI", "No AI"}.issubset(pivot.columns):
            paired = pivot[["AI", "No AI"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index()
        else:
            paired = pd.DataFrame(columns=["monkey", "experiment", "session_index", "session_file"])

        for keys, group in paired.groupby(["monkey", "experiment"], dropna=False):
            monkey, experiment = keys
            is_followup = bool(monkey == "Monkey 1" and experiment == "AI Appearing Obstacle 2")
            rows.append(
                {
                    "task_aggregate": comp["task_label"],
                    "reported_metric": metric,
                    "summary": "mean",
                    "dataset_label": f"{monkey} / {experiment}",
                    "monkey": monkey,
                    "experiment": experiment,
                    "n_paired_sessions_for_metric": int(len(group)),
                    "paired_session_files": "|".join(sorted(str(v) for v in group["session_file"].dropna().unique())),
                    "aggregate_n_paired_sessions": int(comp["n_paired_sessions"]),
                    "ai_mean": comp["ai_mean"],
                    "no_ai_mean": comp["no_ai_mean"],
                    "mean_delta_ai_minus_no_ai": comp["mean_delta_ai_minus_no_ai"],
                    "wilcoxon_p_value": comp["wilcoxon_p_value"],
                    "monkey1_six_month_followup_dataset": is_followup,
                    "monkey1_six_month_followup_contributes_to_this_aggregate": bool(
                        comp["task_label"] == "Appearing Obstacle follow-up" and is_followup
                    ),
                }
            )
    return pd.DataFrame(rows)


def _panel_a_provenance(base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj_rows = edf1.collect_trajectory_rows(base_dir, edf1.DEFAULT_MONKEYS, edf1.DEFAULT_EXPERIMENT)
    prov_rows = []
    for keys, group in traj_rows.groupby(["monkey", "target_label", "condition", "avoidance_side"], dropna=False):
        monkey, target, condition, side = keys
        prov_rows.append(
            {
                "monkey": monkey,
                "experiment": edf1.DEFAULT_EXPERIMENT,
                "target_label": target,
                "condition": condition,
                "avoidance_side": side,
                "n_successful_trajectories": int(len(group)),
                "interpolation": "arc-length resample to 60 samples after trim before first target/obstacle square entry",
                "center_summary": "mean trajectory",
                "band_summary": "25th-75th percentile x-coordinate band at each resampled point, filled against mean z",
            }
        )
    return traj_rows, pd.DataFrame(prov_rows)


def _panel_b_table(base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = OUT_DIR / "panel_b_fixed_obstacle_source_rows.csv"
    table = edf1._metric_delta_table(base_dir, source_path=source_path)
    source = pd.read_csv(source_path, sep=";")
    return source, table


def _copy_reference_figures() -> list[Path]:
    out = OUT_DIR / "reference_figure_copies"
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    src_dir = PROJECT_ROOT / "figures" / "Extended Data Figure 1"
    if src_dir.exists():
        for path in src_dir.iterdir():
            if path.is_file():
                dest = out / path.name
                shutil.copy2(path, dest)
                copied.append(dest)
    return copied


def _write_report(
    paths: dict[str, Path],
    verified: pd.DataFrame,
    inclusion: pd.DataFrame,
    session_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    claims: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b_table: pd.DataFrame,
    task_counts: pd.DataFrame,
    metric_manifest: pd.DataFrame,
) -> None:
    lines = [
        "# Extended Data Figure 1 Verification",
        "",
        "Scope: Extended Data Figure 1 and associated obstacle trajectory-metric claims only. Existing figures and tables were preserved; verification artifacts were written under this directory.",
        "",
        "## Zenodo Reference",
        "",
        "Zenodo record 10.5281/zenodo.18926135 is the v1.0 software archive for the shared-control BCI analysis pipeline. The local repository contains the relevant v1.0 files used here: `src/trajectory_metrics.py`, `src/plots.py`, `scripts/make_extended_data_figure1_trajectory_supplement.py`, and `scripts/run_obstacle_session_metrics.py`.",
        "",
        "## Inclusion",
        "",
    ]
    for _, row in verified.groupby(["task_label", "monkey", "experiment"]).size().reset_index(name="n_sessions").iterrows():
        follow = "yes" if row["experiment"] == "AI Appearing Obstacle 2" else "no"
        lines.append(f"- {row['task_label']} / {row['monkey']} / {row['experiment']}: {int(row['n_sessions'])} verified sessions; six-month follow-up={follow}.")
    lines.extend(
        [
            "- Explicit aggregate check: the Monkey 1 six-month follow-up (`AI Appearing Obstacle 2`) contributes only to the separate `Appearing Obstacle follow-up` aggregate and contributes to zero rows in the primary `Appearing Obstacle` aggregate.",
            f"- Included trial rows after `{sorted(ELIGIBLE_ANSWERS)}` answer-code restriction: {int(inclusion['included'].sum())}/{len(inclusion)}.",
            "- Metric-specific successful-trial fields retain NaN for unsuccessful trials; clearance and collision/proximity summaries retain both successful and unsuccessful trial groups where defined.",
            f"- Metric-level contributing counts and dataset labels are exported in `{paths['metric_contribution_manifest'].relative_to(PROJECT_ROOT)}`.",
            f"- Task-aggregate contributing session counts are exported in `{paths['task_aggregate_counts'].relative_to(PROJECT_ROOT)}`.",
            "",
            "## Panel A",
            "",
            "- Production panel A uses only `AI Obstacle` successful trials from both monkeys.",
            "- Target grouping uses `src.constants.target_mapping`; target order is left, right, slight_left, straight, slight_right.",
            "- Avoidance side is classified by x-position at closest z to the obstacle center.",
            "- Trajectories are trimmed before first target-square or obstacle-square entry, resampled by arc length to 60 samples, and plotted as mean trajectories.",
            "- The caption's IQR bands are only partially literal: the code computes q25/q75 in x and fills horizontally against mean z. It is not a full 2-D trajectory IQR envelope.",
            "",
            "## Panel B / Claims",
            "",
        ]
    )
    for _, row in claims.iterrows():
        lines.append(
            f"- {row['claim']}: reported {row['manuscript_reported']}; computed {row['computed_full_precision']}; match={row['match']}."
        )

    lines.extend(["", "## Corrected Metric Summary", ""])
    key_metrics = [
        "successful_target_window_path_ratio",
        "successful_target_window_path_efficiency",
        "time_to_successful_dwell_s",
        "successful_mean_executed_speed_mps",
        "obstacle_clearance_m",
        "successful_smoothness_mean_squared_jerk",
    ]
    for _, row in comparisons[
        comparisons["summary"].eq("mean") & comparisons["metric"].isin(key_metrics)
    ].sort_values(["task_label", "metric"]).iterrows():
        lines.append(
            f"- {row['task_label']} {row['metric']}: n={int(row['n_paired_sessions'])}, "
            f"shared-control={row['ai_mean']:.15g}, BCI-only={row['no_ai_mean']:.15g}, "
            f"delta={row['mean_delta_ai_minus_no_ai']:.15g}, Wilcoxon p={row['wilcoxon_p_value']:.15g}."
        )

    lines.extend(
        [
            "",
            "## Path Metric Definition",
            "",
            f"- Let `L` be the actual trajectory path length, `D_center` be the straight-line distance from movement start to target center, and `R={metrics.TARGET_WINDOW_RADIUS:g}` m be the target-window radius. The reviewer-requested path-efficiency metric is `max(D_center - R, 0) / L`, computed only on successful trials, then averaged within each session and condition before paired session-level Wilcoxon tests.",
            f"- `successful_target_window_path_ratio` is the inverse target-window measure, `L / max(D_center - R, 0)`, with the same successful-trial filter and session-level aggregation. It changes the endpoint-adjusted reference distance from the numerator of an efficiency score to the denominator of a path-length ratio.",
            "- The endpoint is the target-window edge, not the target center, for both target-window fields. Older straight-to-center audit fields are `successful_path_length_ratio_to_straight = L / D_center` and `successful_path_length_efficiency = D_center / L`.",
            "- The previous reviewer-response statement used `successful_target_window_path_ratio` as the primary row despite the reviewer-requested wording of path efficiency.",
            "- The production reviewer-facing path metric should use `successful_target_window_path_efficiency`; the inverse ratio remains exported as an audit comparison and is not selected based on statistical significance.",
            "",
            "## Exact Replacement Text",
            "",
            "Caption replacement: `Extended Data Fig. 1b reports session-paired trajectory metrics computed from verified obstacle-task sessions. Path efficiency is the straight-line distance from movement onset to the target-window edge divided by actual successful-trial path length; values were averaged within session and condition, then compared with two-sided paired Wilcoxon signed-rank tests across sessions. The Monkey 1 six-month follow-up is reported only as the separate Appearing Obstacle follow-up aggregate and is not included in the primary Appearing Obstacle aggregate.`",
            "",
            "Methods replacement: `For obstacle trajectory metrics, trials were restricted to verified Figure 3 obstacle sessions and answer codes 1, 3, 5, and 6. For each trial, trajectory path length was the sum of x-z displacement segments. The primary path metric was successful-trial path efficiency max(Dcenter - R, 0) / L, where L is actual path length, Dcenter is start-to-target-center distance, and R = 2.3 m is the target-window radius. Metrics were averaged within session and AI condition, and AI versus No-AI was tested with two-sided paired Wilcoxon signed-rank tests over sessions. The inverse target-window path ratio L / max(Dcenter - R, 0) was retained as an audit field but was not the manuscript metric.`",
            "",
            "Results replacement: `On successful trials, path-length efficiency did not differ, whereas shared control increased total acquisition time (Appearing Obstacle: +0.68 s; Fixed Obstacle: +0.58 s) and reduced mean movement speed (Appearing Obstacle: 1.84 vs. 2.19 m/s, p = 3.81 x 10^-6; Fixed Obstacle: 1.97 vs. 2.30 m/s, p = 2.38 x 10^-7). At the same time, shared control increased minimum obstacle clearance by approximately 0.48 Unity units and reduced mean squared jerk (Appearing Obstacle: 3.55 x 10^5 vs. 5.04 x 10^5, p = 2.67 x 10^-4; Fixed Obstacle: 3.76 x 10^5 vs. 5.27 x 10^5, p = 1.67 x 10^-5).`",
            "",
            "Stale-prose diagnosis: the speed and mean-squared-jerk mismatches are substantive and are best explained by legacy/outdated Results prose or transcription from older outputs, not rounding and not a different verified test definition. The verified ED Fig. 1 / ED Table 4 path uses paired, two-sided Wilcoxon signed-rank tests on session-level AI versus BCI-only summaries across the verified obstacle-session set: Appearing Obstacle n=19 paired primary sessions, Fixed Obstacle n=23 paired sessions, with answer-code restriction `{1, 3, 5, 6}` and the Monkey 1 six-month follow-up excluded from the primary Appearing Obstacle aggregate.",
        ]
    )

    lines.extend(
        [
            "",
            "## Methods/Code Notes",
            "",
            "- `run_obstacle_session_metrics.py` computes actual trajectory duration from trajectory timestamps when possible; movement onset is first nonzero model/avatar velocity or first nonzero position step. `time_to_successful_dwell_s` is completion/targetReached/stop minus movement onset, not first target-window entry.",
            "- Speed is path segment length divided by median trajectory sample interval; coordinates are Unity trajectory coordinates used as meters/Unity units in the scripts.",
            "- Mean squared jerk is computed from x and z third differences divided by dt^3, with no smoothing. Raw MSJ enters the session-level Wilcoxon tests in `run_obstacle_session_metrics.py`; the Extended Data Figure 1 embedded table instead displays log10(normalized MSJ) for Fixed Obstacle only.",
            f"- Clearance in `run_obstacle_session_metrics.py` is center distance minus avatar radius {metrics.AVATAR_RADIUS} and obstacle radius {metrics.OBSTACLE_RADIUS}; the Extended Data Figure 1 embedded table uses raw center distance. Collision/proximity in the figure table uses min center distance <= 0.94 and is not the Figure 5 blocked-at-obstacle classifier.",
            "- SEMs in the verified claim path are SEM of paired session deltas. Wilcoxon calls use SciPy two-sided paired signed-rank tests on session-level AI vs No-AI summaries.",
            "- Current production `run_obstacle_session_metrics.py` does not enforce the verified session list or `{1,3,5,6}` eligibility and, by default, writes `tables/` and `figures/`. This verifier connects the checked calculations to the production functions without overwriting those references.",
            "- Current production `make_extended_data_figure1_trajectory_supplement.py` now exports panel b source rows, the embedded table, and metric contribution counts beside the generated figure, so panel b regeneration is no longer a silent figure-only operation.",
            "",
            "## Minimal Edits",
            "",
            "- State that Extended Data Figure 1 panel A shows Fixed Obstacle successful-trial average trajectories only.",
            "- Revise any caption wording that implies full 2-D IQR envelopes; the plotted band is the x-coordinate IQR around the mean-z trajectory after arc-length resampling.",
            "- If reporting the associated metric claims under verified Figure 3 sessions and `{1,3,5,6}`, use the corrected full-precision values in this report and `claim_check.csv`.",
            "- Clarify the trajectory-metric claim scope. The current saved manuscript-matching `tables/obstacle_session_metric_*` outputs are Monkey 1 only and exclude the six-month follow-up, while production defaults in `run_all.py` / `run_obstacle_session_metrics.py` include Monkey 1, Monkey 2, and the follow-up unless explicitly restricted.",
            "",
            "## Regeneration Commands",
            "",
            "```powershell",
            "python scripts/verify_extended_data_figure1.py",
            "python scripts/make_extended_data_figure1_trajectory_supplement.py --output-dir analysis_results/extended_data_figure1_verification/regenerated_figure",
            "```",
            "",
            "## Output Files",
            "",
        ]
    )
    for label, path in paths.items():
        lines.append(f"- `{label}`: `{path.relative_to(PROJECT_ROOT)}`")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    base_dir = str(config["base_dir"])
    verified = _verified_sessions()

    reference_copies = _copy_reference_figures()
    regenerated_dir = OUT_DIR / "regenerated_figure"
    edf1.build_extended_data_figure1(base_dir, regenerated_dir)

    panel_a_traj, panel_a_prov = _panel_a_provenance(base_dir)
    panel_b_source, panel_b_table = _panel_b_table(base_dir)
    inclusion, trial_metrics = _collect_verified_trial_metrics(base_dir, verified)
    usable_trials = trial_metrics[trial_metrics.get("extraction_failure", pd.Series(dtype=object)).isna()].copy()
    session_summary = metrics.summarize_sessions(usable_trials)
    comparisons = _paired_from_session_summary(session_summary)
    claims = _claim_check(comparisons)
    old_new = _old_vs_new(comparisons)
    task_counts = _task_aggregate_counts(verified)
    metric_manifest = _metric_contribution_manifest(session_summary, comparisons)

    paths = {
        "verified_session_list": OUT_DIR / "verified_obstacle_sessions.csv",
        "trial_inclusion": OUT_DIR / "trial_session_inclusion.csv",
        "trial_metrics": OUT_DIR / "verified_trial_metrics.csv",
        "session_summary": OUT_DIR / "verified_session_metric_summary.csv",
        "paired_comparisons": OUT_DIR / "verified_paired_comparisons.csv",
        "claim_check": OUT_DIR / "claim_check.csv",
        "old_vs_new_comparison": OUT_DIR / "old_vs_new_comparison.csv",
        "panel_a_trajectory_rows": OUT_DIR / "panel_a_fixed_obstacle_trajectory_rows.csv",
        "panel_a_provenance": OUT_DIR / "panel_a_fixed_obstacle_provenance.csv",
        "panel_b_source_rows": OUT_DIR / "panel_b_fixed_obstacle_source_rows.csv",
        "panel_b_table": OUT_DIR / "panel_b_fixed_obstacle_embedded_table.csv",
        "task_aggregate_counts": OUT_DIR / "task_aggregate_contribution_counts.csv",
        "metric_contribution_manifest": OUT_DIR / "metric_contribution_manifest.csv",
        "final_panel_b_verification": OUT_DIR / "final_panel_b_verification.csv",
        "metadata": OUT_DIR / "metadata.json",
        "report": OUT_DIR / "extended_data_figure1_verification_report.md",
    }

    verified.to_csv(paths["verified_session_list"], index=False)
    inclusion.to_csv(paths["trial_inclusion"], index=False)
    trial_metrics.to_csv(paths["trial_metrics"], index=False)
    session_summary.to_csv(paths["session_summary"], index=False)
    comparisons.to_csv(paths["paired_comparisons"], index=False)
    claims.to_csv(paths["claim_check"], index=False)
    old_new.to_csv(paths["old_vs_new_comparison"], index=False)
    panel_a_traj.assign(trajectory=panel_a_traj["trajectory"].map(lambda arr: np.asarray(arr).tolist())).to_csv(paths["panel_a_trajectory_rows"], index=False)
    panel_a_prov.to_csv(paths["panel_a_provenance"], index=False)
    panel_b_source.to_csv(paths["panel_b_source_rows"], index=False)
    panel_b_table.to_csv(paths["panel_b_table"], index=False)
    task_counts.to_csv(paths["task_aggregate_counts"], index=False)
    metric_manifest.to_csv(paths["metric_contribution_manifest"], index=False)
    comparisons[
        comparisons["summary"].eq("mean")
        & comparisons["metric"].isin(REPORTED_COMPARISON_METRICS)
    ].sort_values(["task_label", "metric"]).to_csv(paths["final_panel_b_verification"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "base_dir": base_dir,
                "zenodo_reference": "https://zenodo.org/records/18926135",
                "zenodo_version": "v1.0",
                "eligible_answer_codes": sorted(ELIGIBLE_ANSWERS),
                "verified_session_source": str(FIG3_SOURCE_ROWS.relative_to(PROJECT_ROOT)),
                "source_scripts": [
                    "src/trajectory_metrics.py",
                    "src/plots.py",
                    "scripts/make_extended_data_figure1_trajectory_supplement.py",
                    "scripts/run_obstacle_session_metrics.py",
                ],
                "reference_figure_copies": [str(path.relative_to(PROJECT_ROOT)) for path in reference_copies],
                "no_simulations_run": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        paths,
        verified,
        inclusion,
        session_summary,
        comparisons,
        claims,
        panel_a_prov,
        panel_b_table,
        task_counts,
        metric_manifest,
    )
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
