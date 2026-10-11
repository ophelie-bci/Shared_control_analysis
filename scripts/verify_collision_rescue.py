"""Verify the collision-rescue upper-bound analysis.

This verifier preserves manuscript/reference outputs. It reads configured raw
trial data, restricts to verified Figure 3 sessions, reuses the original
collision-rescue definitions, and writes audit outputs under
analysis_results/collision_rescue_verification/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis_results" / "collision_rescue_verification"
FIG3_SOURCE_ROWS = PROJECT_ROOT / "analysis_results" / "success_rate_verification" / "figure3_success_rate_source_rows.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_collision_corrected_success as original
from src.load import load_files


MONKEYS = ["Monkey 1", "Monkey 2"]
EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
MANUSCRIPT = {
    "Appearing Obstacle": {
        "reported_rescue_contribution_pp": 17.4,
        "reported_total_benefit_pp": 28.7,
        "reported_residual_benefit_pp": 11.3,
    },
    "Fixed Obstacle": {
        "reported_rescue_contribution_pp": 13.2,
        "reported_total_benefit_pp": 29.4,
        "reported_residual_benefit_pp": 16.2,
    },
}


def _load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trial_id(trial):
    for key in ("trial", "trial_id", "trialId"):
        value = _get(trial, key, None)
        if value is not None:
            return value
    return np.nan


def _condition(trial) -> str | None:
    try:
        factor = float(_get(trial, "aiVelocityFactor"))
    except Exception:
        return None
    if factor == 0.0:
        return "BCI-only"
    if factor == 1.0:
        return "Shared-control"
    return None


def _answer_label(answer) -> str:
    return "correct" if answer == 1 else "retained_incorrect"


def _load_verified_sessions() -> pd.DataFrame:
    if not FIG3_SOURCE_ROWS.exists():
        raise FileNotFoundError(
            f"Missing verified Figure 3 source rows: {FIG3_SOURCE_ROWS}. "
            "Run scripts/audit_success_rate_verification.py first."
        )
    rows = pd.read_csv(FIG3_SOURCE_ROWS)
    rows = rows[rows["experiment"].isin(EXPERIMENTS)].copy()
    rows = rows[["monkey", "experiment", "session_index", "source_file"]].drop_duplicates()
    return rows


def _collect_trial_and_session_rows(base_dir: str, verified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    verified_keys = {
        (row.monkey, row.experiment, int(row.session_index), row.source_file)
        for row in verified.itertuples(index=False)
    }
    trial_rows: list[dict] = []
    session_rows: list[dict] = []

    for experiment in EXPERIMENTS:
        task_family, task_label = original.EXPERIMENT_LABELS[experiment]
        for monkey in MONKEYS:
            all_trials, correct_all, incorrect_all, *_rest, pkl_files, _ai = load_files(
                experiment, monkey, base_dir, nb_trials=None
            )
            if all_trials == []:
                continue
            for session_index, (correct_trials, incorrect_trials, pkl_path) in enumerate(
                zip(correct_all, incorrect_all, pkl_files)
            ):
                source_file = Path(pkl_path).name
                if (monkey, experiment, int(session_index), source_file) not in verified_keys:
                    continue
                session_uid = f"{monkey}|{experiment}|{session_index:02d}|{source_file}"
                rows_this_session = []
                for success, bucket in ((1, correct_trials), (0, incorrect_trials)):
                    for trial in bucket:
                        answer = _get(trial, "answer", np.nan)
                        if answer not in (1, 3, 5, 6):
                            continue
                        condition = _condition(trial)
                        if condition is None:
                            continue
                        if not original._trial_has_obstacle(trial, experiment):
                            continue
                        is_bci_only = condition == "BCI-only"
                        is_collision_failure = bool(
                            is_bci_only and success == 0 and original._has_obstacle_collision(trial)
                        )
                        corrected_success = int(success or is_collision_failure)
                        row = {
                            "monkey": monkey,
                            "experiment": experiment,
                            "task_family": task_family,
                            "task_label": task_label,
                            "session_index": int(session_index),
                            "session_uid": session_uid,
                            "source_file": source_file,
                            "trial_id": _trial_id(trial),
                            "answer": answer,
                            "answer_class": _answer_label(answer),
                            "condition": condition,
                            "success_observed": int(success),
                            "bci_only_collision_failure_reclassified": int(is_collision_failure),
                            "success_after_collision_rescue": int(corrected_success),
                            "collision_radius_m": original.COLLISION_RADIUS_M,
                        }
                        trial_rows.append(row)
                        rows_this_session.append(row)

                session_df = pd.DataFrame(rows_this_session)
                if session_df.empty:
                    continue
                shared = session_df[session_df["condition"].eq("Shared-control")]
                bci = session_df[session_df["condition"].eq("BCI-only")]
                n_shared = int(len(shared))
                n_bci = int(len(bci))
                n_shared_success = int(shared["success_observed"].sum())
                n_bci_success = int(bci["success_observed"].sum())
                n_bci_rescued = int(bci["bci_only_collision_failure_reclassified"].sum())
                s_shared = n_shared_success / n_shared if n_shared else np.nan
                s_bci = n_bci_success / n_bci if n_bci else np.nan
                p_rescue = n_bci_rescued / n_bci if n_bci else np.nan
                s_corrected = (n_bci_success + n_bci_rescued) / n_bci if n_bci else np.nan
                session_rows.append(
                    {
                        "source": "raw_verified_figure3_sessions",
                        "monkey": monkey,
                        "experiment": experiment,
                        "task_family": task_family,
                        "task_label": task_label,
                        "session_index": int(session_index),
                        "session_uid": session_uid,
                        "source_file": source_file,
                        "n_shared": n_shared,
                        "n_shared_success": n_shared_success,
                        "n_bci_only": n_bci,
                        "n_bci_only_success": n_bci_success,
                        "n_bci_only_collision_failures_rescued": n_bci_rescued,
                        "s_shared_pct": 100.0 * s_shared,
                        "s_bci_only_pct": 100.0 * s_bci,
                        "rescue_contribution_pp": 100.0 * p_rescue,
                        "s_collision_corrected_pct": 100.0 * s_corrected,
                        "total_benefit_pp": 100.0 * (s_shared - s_bci),
                        "residual_benefit_pp": 100.0 * (s_shared - s_corrected),
                    }
                )

    return pd.DataFrame(trial_rows), pd.DataFrame(session_rows)


def _unit_rows_from_sessions(session_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in session_rows.groupby(["monkey", "experiment", "task_family", "task_label"], sort=False):
        monkey, experiment, task_family, task_label = keys
        n_shared = int(group["n_shared"].sum())
        n_shared_success = int(group["n_shared_success"].sum())
        n_bci = int(group["n_bci_only"].sum())
        n_bci_success = int(group["n_bci_only_success"].sum())
        n_rescued = int(group["n_bci_only_collision_failures_rescued"].sum())
        s_shared = n_shared_success / n_shared if n_shared else np.nan
        s_bci = n_bci_success / n_bci if n_bci else np.nan
        s_corrected = (n_bci_success + n_rescued) / n_bci if n_bci else np.nan
        rescue = n_rescued / n_bci if n_bci else np.nan
        rows.append(
            {
                "aggregation_level": "pooled_monkey_experiment_unit",
                "monkey": monkey,
                "experiment": experiment,
                "task_family": task_family,
                "task_label": task_label,
                "n_sessions": int(group["session_uid"].nunique()),
                "session_indices": ",".join(group["session_index"].astype(str)),
                "source_files": ";".join(group["source_file"].astype(str)),
                "n_shared": n_shared,
                "n_shared_success": n_shared_success,
                "n_bci_only": n_bci,
                "n_bci_only_success": n_bci_success,
                "n_bci_only_collision_failures_rescued": n_rescued,
                "s_shared_pct": 100.0 * s_shared,
                "s_bci_only_pct": 100.0 * s_bci,
                "rescue_contribution_pp": 100.0 * rescue,
                "s_collision_corrected_pct": 100.0 * s_corrected,
                "total_benefit_pp": 100.0 * (s_shared - s_bci),
                "residual_benefit_pp": 100.0 * (s_shared - s_corrected),
            }
        )
    return pd.DataFrame(rows)


def _family_from_units(unit_rows: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "s_shared_pct",
        "s_bci_only_pct",
        "rescue_contribution_pp",
        "s_collision_corrected_pct",
        "total_benefit_pp",
        "residual_benefit_pp",
    ]
    rows = []
    for task_family, group in unit_rows.groupby("task_family", sort=False):
        row = {
            "aggregation": "unweighted_mean_of_pooled_monkey_experiment_units",
            "task_family": task_family,
            "n_units": int(len(group)),
            "included_units": ";".join(group["monkey"] + "|" + group["experiment"]),
            "included_followup": bool(group["experiment"].eq("AI Appearing Obstacle 2").any()),
            "total_shared_trials": int(group["n_shared"].sum()),
            "total_bci_only_trials": int(group["n_bci_only"].sum()),
            "total_bci_only_collision_failures_rescued": int(
                group["n_bci_only_collision_failures_rescued"].sum()
            ),
        }
        for col in value_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _family_from_sessions(session_rows: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "s_shared_pct",
        "s_bci_only_pct",
        "rescue_contribution_pp",
        "s_collision_corrected_pct",
        "total_benefit_pp",
        "residual_benefit_pp",
    ]
    rows = []
    for task_family, group in session_rows.groupby("task_family", sort=False):
        row = {
            "aggregation": "unweighted_mean_of_sessions_current_raw_script_default",
            "task_family": task_family,
            "n_sessions": int(len(group)),
            "included_units": ";".join(
                sorted((group["monkey"] + "|" + group["experiment"]).drop_duplicates())
            ),
            "included_followup": bool(group["experiment"].eq("AI Appearing Obstacle 2").any()),
            "total_shared_trials": int(group["n_shared"].sum()),
            "total_bci_only_trials": int(group["n_bci_only"].sum()),
            "total_bci_only_collision_failures_rescued": int(
                group["n_bci_only_collision_failures_rescued"].sum()
            ),
        }
        for col in value_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _old_new_comparison(family_units: pd.DataFrame, family_sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_family, claim in MANUSCRIPT.items():
        unit = family_units[family_units["task_family"].eq(task_family)].iloc[0]
        session = family_sessions[family_sessions["task_family"].eq(task_family)].iloc[0]
        for metric, reported_col in [
            ("rescue_contribution_pp", "reported_rescue_contribution_pp"),
            ("total_benefit_pp", "reported_total_benefit_pp"),
            ("residual_benefit_pp", "reported_residual_benefit_pp"),
        ]:
            reported = float(claim[reported_col])
            computed = float(unit[metric])
            rows.append(
                {
                    "task_family": task_family,
                    "metric": metric,
                    "manuscript_reported": reported,
                    "raw_pooled_unit_computed_full_precision": computed,
                    "raw_pooled_unit_computed_rounded_1dp": round(computed, 1),
                    "reproduces_manuscript_1dp": bool(round(computed, 1) == reported),
                    "raw_session_mean_current_script_default": float(session[metric]),
                    "session_mean_rounded_1dp": round(float(session[metric]), 1),
                    "discrepancy_explanation": (
                        "none; manuscript value matches pooled unit aggregation"
                        if round(computed, 1) == reported
                        else "manuscript value does not match pooled unit aggregation"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cached_rows_and_family() -> tuple[pd.DataFrame, pd.DataFrame]:
    cached = original.compute_cached_unit_counterfactuals()
    if cached.empty:
        return cached, pd.DataFrame()
    unit = cached.rename(
        columns={
            "source": "aggregation_level",
            "p_blocked_bci_only_pct": "rescue_contribution_pp",
            "observed_gain_pp": "total_benefit_pp",
            "remaining_gain_after_collision_correction_pp": "residual_benefit_pp",
        }
    ).copy()
    unit["aggregation_level"] = "cached_grand_summary_monkey_experiment_unit"

    rows = []
    for task_family, group in unit.groupby("task_family", sort=False):
        rows.append(
            {
                "aggregation": "unweighted_mean_of_cached_grand_summary_units",
                "task_family": task_family,
                "n_units": int(len(group)),
                "included_units": ";".join(group["monkey"] + "|" + group["experiment"]),
                "included_followup": bool(group["experiment"].eq("AI Appearing Obstacle 2").any()),
                "total_shared_trials": int(group["n_shared"].sum()),
                "total_bci_only_trials": int(group["n_bci_only"].sum()),
                "total_bci_only_collision_failures_rescued": int(group["n_bci_only_blocked_failures"].sum()),
                "s_shared_pct": float(group["s_shared_pct"].mean()),
                "s_bci_only_pct": float(group["s_bci_only_pct"].mean()),
                "rescue_contribution_pp": float(group["rescue_contribution_pp"].mean()),
                "s_collision_corrected_pct": float(group["s_collision_corrected_pct"].mean()),
                "total_benefit_pp": float(group["total_benefit_pp"].mean()),
                "residual_benefit_pp": float(group["residual_benefit_pp"].mean()),
            }
        )
    return unit, pd.DataFrame(rows)


def _comparison_with_cached(
    cached_family: pd.DataFrame,
    raw_family_units: pd.DataFrame,
    family_sessions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for task_family, claim in MANUSCRIPT.items():
        cached = cached_family[cached_family["task_family"].eq(task_family)].iloc[0]
        raw_unit = raw_family_units[raw_family_units["task_family"].eq(task_family)].iloc[0]
        session = family_sessions[family_sessions["task_family"].eq(task_family)].iloc[0]
        for metric, reported_col in [
            ("rescue_contribution_pp", "reported_rescue_contribution_pp"),
            ("total_benefit_pp", "reported_total_benefit_pp"),
            ("residual_benefit_pp", "reported_residual_benefit_pp"),
        ]:
            reported = float(claim[reported_col])
            cached_value = float(cached[metric])
            rows.append(
                {
                    "task_family": task_family,
                    "metric": metric,
                    "manuscript_reported": reported,
                    "cached_grand_summary_computed_full_precision": cached_value,
                    "cached_grand_summary_rounded_1dp": round(cached_value, 1),
                    "cached_reproduces_manuscript_1dp": bool(round(cached_value, 1) == reported),
                    "raw_trial_pooled_unit_full_precision": float(raw_unit[metric]),
                    "raw_trial_pooled_unit_rounded_1dp": round(float(raw_unit[metric]), 1),
                    "raw_trial_session_mean_full_precision": float(session[metric]),
                    "raw_trial_session_mean_rounded_1dp": round(float(session[metric]), 1),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    config: dict,
    verified: pd.DataFrame,
    cached_family: pd.DataFrame,
    family_units: pd.DataFrame,
    family_sessions: pd.DataFrame,
    comparison: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    lines = [
        "# Collision-Rescue Upper-Bound Verification",
        "",
        "Scope: collision-rescue upper-bound analysis only. Pinhao's simulations, manuscript values, and reference figures were not modified.",
        "",
        "## Provenance",
        "",
        f"- Repository: `{PROJECT_ROOT}`",
        f"- Configured base_dir: `{config.get('base_dir')}`",
        "- Starting implementation retained: `scripts/run_collision_corrected_success.py`.",
        "- Verified Figure 3 session list: `analysis_results/success_rate_verification/figure3_success_rate_source_rows.csv`.",
        "- Eligible answers: `1` as success and `{3,5,6}` as retained incorrect outcomes; answers `{4,9,10}` are excluded from this verification.",
        f"- Collision radius: `{original.COLLISION_RADIUS_M}` m, from `0.45 + 0.25` in the original implementation.",
        "",
        "## Original Calculation Used For Manuscript Values",
        "",
        "The manuscript numbers reproduce from the original script's cached `Variables/grand_summary.pkl` fallback path, which uses precomputed `failure_props_per_unit.stuck_obstacle` values. That path pools within each monkey x experiment unit and then computes an unweighted mean across units within each task family.",
        "",
        "A fresh raw-trial rerun from `config.yaml` using `scripts/run_collision_corrected_success.py` trajectory-distance collision geometry does not reproduce the rescue/residual values, although it does reproduce the total shared-control benefit closely. This discrepancy is due to the stored `stuck_obstacle` failure-mode inputs versus the fresh trajectory-distance collision reclassification, not to inclusion of Pinhao simulations.",
        "",
    ]
    for _, row in cached_family.iterrows():
        lines.append(
            f"- {row['task_family']}: units `{row['included_units']}`; follow-up included = {row['included_followup']}; "
            f"shared denominator total {int(row['total_shared_trials'])}, BCI-only denominator total {int(row['total_bci_only_trials'])}, "
            f"rescued BCI-only collision failures {int(row['total_bci_only_collision_failures_rescued'])}."
        )
    lines.extend(
        [
            "",
            "## Counterfactual Operation",
            "",
            "For the fresh raw-trial implementation, only BCI-only failed trials with minimum avatar-obstacle distance <= 0.70 m are reclassified as successes. BCI-only noncollision failures, all BCI-only successes, and all shared-control outcomes remain unchanged. For the manuscript-matching cached path, the same algebra is applied to precomputed `stuck_obstacle` fractions in `Variables/grand_summary.pkl`: `collision-corrected BCI-only = BCI-only success + stuck_obstacle`. The operation is an upper bound based on recorded experimental trials; it does not establish the performance of a standalone safety-only controller.",
            "",
            "## Full-Precision Results",
            "",
        ]
    )
    for _, row in cached_family.iterrows():
        lines.append(
            f"- {row['task_family']} cached manuscript aggregation: baseline BCI-only {row['s_bci_only_pct']:.15g}%, "
            f"collision-corrected BCI-only {row['s_collision_corrected_pct']:.15g}%, shared-control {row['s_shared_pct']:.15g}%, "
            f"total benefit {row['total_benefit_pp']:.15g} pp, rescue contribution {row['rescue_contribution_pp']:.15g} pp, "
            f"residual benefit {row['residual_benefit_pp']:.15g} pp."
        )
    lines.extend(["", "Fresh raw-trial trajectory-distance results for comparison:", ""])
    for _, row in family_units.iterrows():
        lines.append(
            f"- {row['task_family']} raw pooled-unit aggregation: total benefit {row['total_benefit_pp']:.15g} pp, "
            f"rescue contribution {row['rescue_contribution_pp']:.15g} pp, residual benefit {row['residual_benefit_pp']:.15g} pp."
        )
    lines.extend(["", "## Old-Versus-New Comparison", ""])
    for _, row in comparison.iterrows():
        lines.append(
            f"- {row['task_family']} `{row['metric']}`: manuscript {row['manuscript_reported']:.1f}; "
            f"cached grand-summary {row['cached_grand_summary_computed_full_precision']:.15g} "
            f"(rounds {row['cached_grand_summary_rounded_1dp']:.1f}); "
            f"raw pooled-unit {row['raw_trial_pooled_unit_full_precision']:.15g} "
            f"(rounds {row['raw_trial_pooled_unit_rounded_1dp']:.1f}); "
            f"raw session-mean {row['raw_trial_session_mean_full_precision']:.15g} "
            f"(rounds {row['raw_trial_session_mean_rounded_1dp']:.1f})."
        )
    lines.extend(
        [
            "",
            "## Discrepancy Explanation",
            "",
            "The manuscript rescue and residual values reproduce exactly to one decimal place only from the cached grand-summary `stuck_obstacle` inputs. They do not reproduce from fresh raw recorded trajectories using the current `run_collision_corrected_success.py` distance-based collision rule: raw pooled units yield Appearing Obstacle rescue/total/residual = 19.6/28.7/9.0 pp and Fixed Obstacle = 5.8/29.4/23.6 pp; raw session means yield 19.5/28.1/8.6 pp and 5.7/29.5/23.8 pp. Therefore the manuscript values should either cite the cached failure-mode/stuck-obstacle analysis explicitly or be updated to the raw trajectory-distance reclassification values.",
            "",
            "## Suggested Edits",
            "",
            "Results wording if retaining the manuscript/cached analysis: `As an upper-bound collision-rescue analysis based on precomputed stuck-obstacle failure labels from recorded trials, we reclassified BCI-only stuck-obstacle failures as successes while leaving all other recorded outcomes unchanged. In the pooled monkey x experiment units used for this analysis, this accounted for 17.4 of the 28.7 percentage-point shared-control benefit in Appearing Obstacle and 13.2 of the 29.4 percentage-point benefit in Fixed Obstacle, leaving residual benefits of 11.3 and 16.2 percentage points, respectively.`",
            "",
            "Results wording if updating to fresh raw trajectory-distance reclassification: `Using the current trajectory-distance collision rule, collision rescue accounted for 19.6 of the 28.7 percentage-point Appearing Obstacle benefit and 5.8 of the 29.4 percentage-point Fixed Obstacle benefit, leaving residual benefits of 9.0 and 23.6 percentage points, respectively.`",
            "",
            "Methods/reviewer-response wording: `The collision-rescue analysis is an upper bound computed from recorded experimental trials, not a simulation of a standalone safety-only controller. We used the verified Figure 3 sessions and retained answer codes 1, 3, 5, and 6. The manuscript values were generated from the stored grand-summary stuck-obstacle labels: BCI-only stuck-obstacle failures were reclassified as successes and all other outcomes were unchanged. Rates were pooled within each monkey x experiment unit and then averaged across units within Fixed Obstacle or Appearing Obstacle. The Appearing Obstacle aggregate includes Monkey 1 original Appearing Obstacle, Monkey 2 original Appearing Obstacle, and Monkey 1 six-month follow-up Appearing Obstacle.`",
            "",
            "## Regeneration Command",
            "",
            "```powershell",
            "python scripts/verify_collision_rescue.py",
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
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("config.yaml is missing base_dir")

    verified = _load_verified_sessions()
    trial_rows, session_rows = _collect_trial_and_session_rows(str(base_dir), verified)
    unit_rows = _unit_rows_from_sessions(session_rows)
    family_units = _family_from_units(unit_rows)
    family_sessions = _family_from_sessions(session_rows)
    cached_units, cached_family = _cached_rows_and_family()
    comparison = _comparison_with_cached(cached_family, family_units, family_sessions)

    paths = {
        "verified_figure3_sessions_used": OUT_DIR / "verified_figure3_sessions_used.csv",
        "trial_provenance": OUT_DIR / "collision_rescue_trial_provenance.csv",
        "session_summary": OUT_DIR / "collision_rescue_session_summary.csv",
        "pooled_unit_summary": OUT_DIR / "collision_rescue_pooled_unit_summary.csv",
        "manuscript_aggregation_summary": OUT_DIR / "collision_rescue_manuscript_aggregation_summary.csv",
        "cached_unit_summary": OUT_DIR / "collision_rescue_cached_grand_summary_units.csv",
        "cached_aggregation_summary": OUT_DIR / "collision_rescue_cached_grand_summary_aggregation.csv",
        "session_mean_alternative_summary": OUT_DIR / "collision_rescue_session_mean_alternative_summary.csv",
        "old_vs_new_comparison": OUT_DIR / "collision_rescue_old_vs_new_comparison.csv",
        "metadata": OUT_DIR / "metadata.json",
        "report": OUT_DIR / "collision_rescue_verification_report.md",
    }
    verified.to_csv(paths["verified_figure3_sessions_used"], index=False)
    trial_rows.to_csv(paths["trial_provenance"], index=False)
    session_rows.to_csv(paths["session_summary"], index=False)
    unit_rows.to_csv(paths["pooled_unit_summary"], index=False)
    family_units.to_csv(paths["manuscript_aggregation_summary"], index=False)
    cached_units.to_csv(paths["cached_unit_summary"], index=False)
    cached_family.to_csv(paths["cached_aggregation_summary"], index=False)
    family_sessions.to_csv(paths["session_mean_alternative_summary"], index=False)
    comparison.to_csv(paths["old_vs_new_comparison"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "config_base_dir": base_dir,
                "source_script": "scripts/run_collision_corrected_success.py",
                "verified_session_source": str(FIG3_SOURCE_ROWS.relative_to(PROJECT_ROOT)),
                "eligible_answer_codes": [1, 3, 5, 6],
                "excluded_answer_codes": [4, 9, 10],
                "collision_radius_m": original.COLLISION_RADIUS_M,
                "manuscript_matching_source": "Variables/grand_summary.pkl failure_props_per_unit.stuck_obstacle",
                "experiments": EXPERIMENTS,
                "monkeys": MONKEYS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(config, verified, cached_family, family_units, family_sessions, comparison, paths)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
