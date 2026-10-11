"""Resolve collision-rescue provenance with the verified Figure 5 classifier.

This script recomputes raw Figure 5 failure classifications, uses the recovered
`stuck_obstacle` / "Blocked at obstacle" values, and applies the original cached
collision-rescue aggregation without modifying manuscript/reference outputs.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis_results" / "collision_rescue_verification" / "blocked_obstacle_rescue"
CACHE_PATH = PROJECT_ROOT / "Variables" / "grand_summary.pkl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_figure5_failure_modes import (  # noqa: E402
    AVATAR_RADIUS,
    ELIGIBLE_ANSWERS,
    EXCLUDED_ANSWERS,
    MIN_OBSTACLE_FRAMES,
    OBSTACLE_RADIUS,
    STUCK_MARGIN,
    collect_raw_trials,
    summarize_failure_props,
)


MONKEYS = ["Monkey 1", "Monkey 2"]
EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
TASK_FAMILY = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle",
}
TASK_LABEL = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle (6 months later)",
}
CONDITION_MAP = {"AI": "Shared-control", "No AI": "BCI-only"}
MANUSCRIPT = {
    "Appearing Obstacle": {
        "rescue_contribution_pp": 17.4,
        "total_benefit_pp": 28.7,
        "residual_benefit_pp": 11.3,
    },
    "Fixed Obstacle": {
        "rescue_contribution_pp": 13.2,
        "total_benefit_pp": 29.4,
        "residual_benefit_pp": 16.2,
    },
}


def _load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_cache_failure_props() -> pd.DataFrame:
    with CACHE_PATH.open("rb") as f:
        summary = pickle.load(f)
    return pd.DataFrame(summary["failure_props_per_unit"])


def _unit_collision_rows(failure: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        for monkey in MONKEYS:
            sub = failure[failure["monkey"].eq(monkey) & failure["experiment"].eq(experiment)]
            if sub.empty:
                continue
            shared = sub[sub["condition"].eq("AI")]
            bci = sub[sub["condition"].eq("No AI")]
            if shared.empty or bci.empty:
                continue
            shared = shared.iloc[0]
            bci = bci.iloc[0]
            s_shared = float(shared["success"])
            s_bci = float(bci["success"])
            blocked = float(bci["stuck_obstacle"])
            corrected = min(1.0, s_bci + blocked)
            rows.append(
                {
                    "source": source,
                    "monkey": monkey,
                    "experiment": experiment,
                    "task_family": TASK_FAMILY[experiment],
                    "task_label": TASK_LABEL[experiment],
                    "condition_source_for_rescue": "BCI-only/No AI stuck_obstacle",
                    "n_shared": int(shared["n_trials"]),
                    "n_shared_success": int(round(s_shared * float(shared["n_trials"]))),
                    "n_bci_only": int(bci["n_trials"]),
                    "n_bci_only_success": int(round(s_bci * float(bci["n_trials"]))),
                    "n_bci_only_blocked_at_obstacle": int(round(blocked * float(bci["n_trials"]))),
                    "s_shared_pct": 100.0 * s_shared,
                    "s_bci_only_pct": 100.0 * s_bci,
                    "blocked_at_obstacle_pp": 100.0 * blocked,
                    "s_blocked_rescue_corrected_pct": 100.0 * corrected,
                    "total_benefit_pp": 100.0 * (s_shared - s_bci),
                    "rescue_contribution_pp": 100.0 * blocked,
                    "residual_benefit_pp": 100.0 * (s_shared - corrected),
                }
            )
    return pd.DataFrame(rows)


def _task_aggregation(unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_family, group in unit_rows.groupby("task_family", sort=False):
        rows.append(
            {
                "source": group["source"].iloc[0],
                "aggregation": "unweighted_mean_of_monkey_experiment_units",
                "task_family": task_family,
                "n_units": int(len(group)),
                "included_units": ";".join(group["monkey"] + "|" + group["experiment"]),
                "includes_six_month_followup": bool(group["experiment"].eq("AI Appearing Obstacle 2").any()),
                "total_shared_trials": int(group["n_shared"].sum()),
                "total_bci_only_trials": int(group["n_bci_only"].sum()),
                "total_bci_only_blocked_at_obstacle": int(group["n_bci_only_blocked_at_obstacle"].sum()),
                "s_shared_pct": float(group["s_shared_pct"].mean()),
                "s_bci_only_pct": float(group["s_bci_only_pct"].mean()),
                "blocked_at_obstacle_pp": float(group["blocked_at_obstacle_pp"].mean()),
                "s_blocked_rescue_corrected_pct": float(group["s_blocked_rescue_corrected_pct"].mean()),
                "total_benefit_pp": float(group["total_benefit_pp"].mean()),
                "rescue_contribution_pp": float(group["rescue_contribution_pp"].mean()),
                "residual_benefit_pp": float(group["residual_benefit_pp"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _agreement(raw_failure: pd.DataFrame, cache_failure: pd.DataFrame) -> pd.DataFrame:
    keys = ["monkey", "experiment", "condition"]
    cols = ["n_trials", "success", "stuck_obstacle"]
    merged = raw_failure[keys + cols].merge(
        cache_failure[keys + cols], on=keys, how="outer", suffixes=("_raw", "_cache"), indicator=True
    )
    for col in cols:
        merged[f"{col}_delta"] = pd.to_numeric(merged[f"{col}_raw"], errors="coerce") - pd.to_numeric(
            merged[f"{col}_cache"], errors="coerce"
        )
    return merged


def _trial_rescue_inputs(trials: pd.DataFrame) -> pd.DataFrame:
    included = trials[trials["included"].eq(1)].copy()
    if included.empty:
        return included
    included["is_bci_only"] = included["condition"].eq("No AI")
    included["is_shared_control"] = included["condition"].eq("AI")
    included["is_failure"] = included["success"].eq(0)
    included["blocked_at_obstacle_raw_classifier"] = included["failure_reason"].eq("stuck_obstacle")
    included["used_as_collision_rescue_numerator"] = (
        included["is_bci_only"]
        & included["is_failure"]
        & included["blocked_at_obstacle_raw_classifier"]
    )
    included["success_after_blocked_obstacle_rescue"] = (
        included["success"].eq(1) | included["used_as_collision_rescue_numerator"]
    )
    included["trial_level_cache_label_available"] = False
    included["classification_agreement_note"] = (
        "Trial-level raw labels are from the verified Figure 5 classifier; "
        "the original cache stores unit-level proportions, not trial-level labels."
    )
    cols = [
        "monkey",
        "experiment",
        "session_index",
        "session_file",
        "trial_id",
        "answer",
        "condition",
        "target_label",
        "success",
        "failure_reason",
        "is_bci_only",
        "is_shared_control",
        "is_failure",
        "blocked_at_obstacle_raw_classifier",
        "used_as_collision_rescue_numerator",
        "success_after_blocked_obstacle_rescue",
        "max_consecutive_obstacle_frames",
        "passed_obstacle",
        "min_obstacle_frames",
        "obstacle_radius_m",
        "avatar_radius_m",
        "stuck_margin_m",
        "trial_level_cache_label_available",
        "classification_agreement_note",
    ]
    return included[[c for c in cols if c in included.columns]]


def _weighting_rows(unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_family, group in unit_rows.groupby("task_family", sort=False):
        weight = 1.0 / len(group) if len(group) else np.nan
        for _, row in group.iterrows():
            rows.append(
                {
                    "task_family": task_family,
                    "monkey": row["monkey"],
                    "experiment": row["experiment"],
                    "task_label": row["task_label"],
                    "unit_weight_in_task_family_mean": weight,
                    "n_shared": int(row["n_shared"]),
                    "n_shared_success": int(row["n_shared_success"]),
                    "n_bci_only": int(row["n_bci_only"]),
                    "n_bci_only_success": int(row["n_bci_only_success"]),
                    "n_bci_only_blocked_at_obstacle": int(row["n_bci_only_blocked_at_obstacle"]),
                    "s_shared_pct": float(row["s_shared_pct"]),
                    "s_bci_only_pct": float(row["s_bci_only_pct"]),
                    "blocked_at_obstacle_pp": float(row["blocked_at_obstacle_pp"]),
                    "total_benefit_pp": float(row["total_benefit_pp"]),
                    "rescue_contribution_pp": float(row["rescue_contribution_pp"]),
                    "residual_benefit_pp": float(row["residual_benefit_pp"]),
                    "aggregation_note": (
                        "Rates are computed within each monkey x experiment unit; "
                        "task-family values are unweighted means of these units."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _manuscript_comparison(raw_agg: pd.DataFrame, cache_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_family, metrics in MANUSCRIPT.items():
        raw = raw_agg[raw_agg["task_family"].eq(task_family)].iloc[0]
        cache = cache_agg[cache_agg["task_family"].eq(task_family)].iloc[0]
        for metric, reported in metrics.items():
            raw_value = float(raw[metric])
            cache_value = float(cache[metric])
            rows.append(
                {
                    "task_family": task_family,
                    "metric": metric,
                    "manuscript_reported": float(reported),
                    "raw_figure5_classifier_full_precision": raw_value,
                    "raw_figure5_classifier_rounded_1dp": round(raw_value, 1),
                    "raw_reproduces_manuscript_1dp": bool(round(raw_value, 1) == float(reported)),
                    "cache_full_precision": cache_value,
                    "cache_rounded_1dp": round(cache_value, 1),
                    "raw_minus_cache": raw_value - cache_value,
                }
            )
    return pd.DataFrame(rows)


def _session_inclusion(trials: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    included = trials[trials["included"].eq(1)].copy()
    by_session = (
        included.groupby(["monkey", "experiment", "session_index", "session_file", "condition"], sort=False)
        .agg(
            n_included_trials=("included", "size"),
            n_success=("success", "sum"),
            n_stuck_obstacle=("failure_reason", lambda x: int(np.sum(x == "stuck_obstacle"))),
        )
        .reset_index()
    )
    return by_session.merge(
        sessions,
        left_on=["monkey", "experiment", "session_index", "session_file"],
        right_on=["monkey", "experiment", "session_index", "session_file"],
        how="left",
    )


def _write_report(
    raw_agg: pd.DataFrame,
    cache_agg: pd.DataFrame,
    comparison: pd.DataFrame,
    agreement: pd.DataFrame,
    paths: dict[str, Path],
    base_dir: str,
) -> None:
    max_stuck_delta = float(np.nanmax(np.abs(agreement["stuck_obstacle_delta"].to_numpy(float))))
    lines = [
        "# Collision-Rescue Provenance Resolution",
        "",
        "Scope: collision-rescue provenance only. Manuscript values and reference outputs were preserved.",
        "",
        "## Raw Reproduction",
        "",
        f"- Raw data source: `{base_dir}` from `config.yaml`.",
        "- Entry point: `scripts/resolve_collision_rescue_provenance.py`.",
        "- Classifier source: `scripts/verify_figure5_failure_modes.py` raw trial classifier.",
        f"- Eligible answers: `{sorted(ELIGIBLE_ANSWERS)}`; excluded answers: `{sorted(EXCLUDED_ANSWERS)}`.",
        f"- Blocked at obstacle definition: at least {MIN_OBSTACLE_FRAMES} consecutive frames inside the obstacle square inflated by obstacle radius {OBSTACLE_RADIUS:g} m, avatar radius {AVATAR_RADIUS:g} m, and margin {STUCK_MARGIN:g} m, with no pass beyond the obstacle.",
        "",
        "## Agreement Check",
        "",
        f"- Raw `stuck_obstacle` agreement with cached unit proportions, maximum absolute delta: `{max_stuck_delta:.12g}`.",
        "- This cache agreement is a consistency check only; the reproduction claim above comes from recomputing trial-level labels from raw data before applying the cached aggregation.",
        "",
        "## Aggregation",
        "",
        "The original cached collision-rescue calculation pools trials within each monkey x experiment unit, computes BCI-only `success + stuck_obstacle`, and then averages unit-level percentages equally within each task family. Appearing Obstacle includes Monkey 1 original Appearing Obstacle, Monkey 2 original Appearing Obstacle, and Monkey 1 six-month follow-up Appearing Obstacle.",
        "",
        "## Full-Precision Raw Results",
        "",
    ]
    for _, row in raw_agg.iterrows():
        lines.append(
            f"- {row['task_family']}: rescue {row['rescue_contribution_pp']:.15g} pp, "
            f"total benefit {row['total_benefit_pp']:.15g} pp, residual {row['residual_benefit_pp']:.15g} pp; "
            f"units `{row['included_units']}`."
        )
    lines.extend(["", "## Manuscript Check", ""])
    for _, row in comparison.iterrows():
        lines.append(
            f"- {row['task_family']} `{row['metric']}`: manuscript {row['manuscript_reported']:.1f}; "
            f"raw Figure 5 classifier {row['raw_figure5_classifier_full_precision']:.15g} "
            f"(rounds {row['raw_figure5_classifier_rounded_1dp']:.1f}); "
            f"raw-cache delta {row['raw_minus_cache']:.15g}."
        )
    lines.extend(
        [
            "",
            "## Wording Assessment",
            "",
            "`Blocked at obstacle` is not the same operation as the current trajectory-distance collision rule in `run_collision_corrected_success.py`. The manuscript-matching operation reclassifies BCI-only trials categorized by the verified Figure 5 execution-failure classifier as `stuck_obstacle` / Blocked at obstacle. The trajectory-distance rule simply asks whether a failed BCI-only trajectory ever comes within 0.70 m of the obstacle center; it lacks the 10-frame dwell, inflated-square, and no-pass criteria.",
            "",
            "Therefore manuscript wording should avoid saying only that trials `touched` or `collided with` the obstacle if that implies the current distance-only rule. More accurate wording is `blocked-at-obstacle failures` or `stuck-obstacle failures` based on the verified Figure 5 classifier.",
            "",
            "## Reproducible Entry Point",
            "",
            "```powershell",
            "python scripts/resolve_collision_rescue_provenance.py",
            "```",
            "",
            "This script calls the verified Figure 5 raw trial classifier and then applies the original cached collision-rescue aggregation.",
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

    trials, sessions = collect_raw_trials(base_dir, MONKEYS, EXPERIMENTS)
    raw_failure = summarize_failure_props(trials)
    cache_failure = _load_cache_failure_props()

    raw_units = _unit_collision_rows(raw_failure, "raw_recomputed_figure5_classifier")
    cache_units = _unit_collision_rows(cache_failure, "cache_failure_props_per_unit")
    raw_agg = _task_aggregation(raw_units)
    cache_agg = _task_aggregation(cache_units)
    agreement = _agreement(raw_failure, cache_failure)
    trial_rescue_inputs = _trial_rescue_inputs(trials)
    weighting = _weighting_rows(raw_units)
    comparison = _manuscript_comparison(raw_agg, cache_agg)
    session_inclusion = _session_inclusion(trials, sessions)

    paths = {
        "trial_classification": OUT_DIR / "blocked_obstacle_trial_classification.csv",
        "trial_rescue_inputs": OUT_DIR / "blocked_obstacle_trial_rescue_inputs.csv",
        "session_inclusion": OUT_DIR / "blocked_obstacle_session_inclusion.csv",
        "raw_failure_props_per_unit": OUT_DIR / "blocked_obstacle_raw_failure_props_per_unit.csv",
        "classification_agreement": OUT_DIR / "blocked_obstacle_classification_agreement.csv",
        "raw_unit_rescue": OUT_DIR / "blocked_obstacle_raw_unit_rescue.csv",
        "denominators_and_weighting": OUT_DIR / "blocked_obstacle_denominators_and_weighting.csv",
        "raw_task_aggregation": OUT_DIR / "blocked_obstacle_raw_task_aggregation.csv",
        "cache_task_aggregation": OUT_DIR / "blocked_obstacle_cache_task_aggregation.csv",
        "manuscript_comparison": OUT_DIR / "blocked_obstacle_manuscript_comparison.csv",
        "metadata": OUT_DIR / "metadata.json",
        "report": OUT_DIR / "blocked_obstacle_collision_rescue_report.md",
    }
    trials.to_csv(paths["trial_classification"], index=False)
    trial_rescue_inputs.to_csv(paths["trial_rescue_inputs"], index=False)
    session_inclusion.to_csv(paths["session_inclusion"], index=False)
    raw_failure.to_csv(paths["raw_failure_props_per_unit"], index=False)
    agreement.to_csv(paths["classification_agreement"], index=False)
    raw_units.to_csv(paths["raw_unit_rescue"], index=False)
    weighting.to_csv(paths["denominators_and_weighting"], index=False)
    raw_agg.to_csv(paths["raw_task_aggregation"], index=False)
    cache_agg.to_csv(paths["cache_task_aggregation"], index=False)
    comparison.to_csv(paths["manuscript_comparison"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "base_dir": base_dir,
                "classifier_source": "scripts.verify_figure5_failure_modes.collect_raw_trials",
                "aggregation_source": "original cached collision-rescue aggregation",
                "script": "scripts/resolve_collision_rescue_provenance.py",
                "eligible_answers": sorted(ELIGIBLE_ANSWERS),
                "excluded_answers": sorted(EXCLUDED_ANSWERS),
                "experiments": EXPERIMENTS,
                "monkeys": MONKEYS,
                "blocked_at_obstacle": {
                    "min_obstacle_frames": MIN_OBSTACLE_FRAMES,
                    "obstacle_radius_m": OBSTACLE_RADIUS,
                    "avatar_radius_m": AVATAR_RADIUS,
                    "stuck_margin_m": STUCK_MARGIN,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(raw_agg, cache_agg, comparison, agreement, paths, base_dir)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
