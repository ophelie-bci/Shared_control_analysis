"""Audit success-rate analyses against revised manuscript claims.

This script is intentionally read-only with respect to the source data and the
existing manuscript artifact folders. It writes verification outputs under
analysis_results/success_rate_verification/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_results" / "success_rate_verification"

DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
    "AI Respawn",
]
OBSTACLE_GAIN_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
]

CONDITION_LABELS = {
    0.0: "BCI-only",
    1.0: "Shared-control",
}

MANUSCRIPT_CLAIMS = [
    {
        "claim_id": "fig3a_fixed_obstacle_monkey1",
        "document_location": "Manuscript para 39 / Figure 3A",
        "monkey": "Monkey 1",
        "experiment": "AI Obstacle",
        "reported_shared_control_pct": None,
        "reported_bci_only_pct": None,
        "reported_median_delta": 0.313,
        "reported_p_two_sided": 4.88e-4,
    },
    {
        "claim_id": "fig3a_fixed_obstacle_monkey2",
        "document_location": "Manuscript para 39 / Figure 3A",
        "monkey": "Monkey 2",
        "experiment": "AI Obstacle",
        "reported_shared_control_pct": None,
        "reported_bci_only_pct": None,
        "reported_median_delta": 0.268,
        "reported_p_two_sided": 9.77e-4,
    },
    {
        "claim_id": "fig3b_appearing_obstacle_monkey1",
        "document_location": "Manuscript para 40 / Figure 3B",
        "monkey": "Monkey 1",
        "experiment": "AI Appearing Obstacle",
        "reported_shared_control_pct": 67.0,
        "reported_bci_only_pct": 39.0,
        "reported_median_delta": 0.267,
        "reported_p_two_sided": 3.91e-3,
        "known_alternate_shared_control_pct": 69.0,
        "known_alternate_p_one_sided": 1.95e-3,
    },
    {
        "claim_id": "fig3b_appearing_obstacle_monkey2",
        "document_location": "Manuscript para 40 / Figure 3B",
        "monkey": "Monkey 2",
        "experiment": "AI Appearing Obstacle",
        "reported_shared_control_pct": 56.0,
        "reported_bci_only_pct": 22.0,
        "reported_median_delta": 0.356,
        "reported_p_two_sided": 3.91e-3,
    },
    {
        "claim_id": "fig3c_appearing_obstacle_followup_monkey1",
        "document_location": "Manuscript para 41 / Figure 3C",
        "monkey": "Monkey 1",
        "experiment": "AI Appearing Obstacle 2",
        "reported_shared_control_pct": 80.0,
        "reported_bci_only_pct": 57.0,
        "reported_gain_pct": 22.0,
    },
    {
        "claim_id": "fig3d_respawn_monkey1",
        "document_location": "Manuscript para 42 / Figure 3D",
        "monkey": "Monkey 1",
        "experiment": "AI Respawn",
        "reported_shared_control_pct": 67.0,
        "reported_bci_only_pct": 80.0,
        "reported_median_delta": -0.114,
        "reported_p_two_sided": 0.01,
    },
]


def ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_base_dir(cli_base_dir: str | None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def get_attr(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def trial_id(trial) -> int | float:
    for key in ("trial", "trial_id", "trialId"):
        value = get_attr(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return value
    return np.nan


def ai_factor(trial) -> float:
    try:
        return float(get_attr(trial, "aiVelocityFactor"))
    except Exception:
        return np.nan


def target_label(trial) -> str:
    ensure_import_path()
    from src.constants import target_mapping

    try:
        return target_mapping.get(tuple(get_attr(trial, "targetPosition")), "unknown")
    except Exception:
        return "unknown"


def condition_included(experiment: str, trial) -> bool:
    if experiment in ("AI Obstacle", "House"):
        return True
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return get_attr(trial, "obstacleAppearedTime", None) is not None
    if experiment == "AI Respawn":
        target_jump = get_attr(trial, "targetJumpPosition", None)
        try:
            return not np.isnan(target_jump).any()
        except Exception:
            return target_jump is not None
    if experiment == "House Obstacle":
        obstacle = get_attr(trial, "obstaclePosition", None)
        try:
            return not np.all(np.isnan(obstacle))
        except Exception:
            return obstacle is not None
    return True


def wilcoxon_rows(on: Iterable[float], off: Iterable[float]) -> dict[str, float]:
    on = np.asarray(list(on), dtype=float)
    off = np.asarray(list(off), dtype=float)
    mask = np.isfinite(on) & np.isfinite(off)
    on = on[mask]
    off = off[mask]
    delta = on - off
    out = {
        "n_sessions": int(len(delta)),
        "n_zero_deltas": int(np.sum(delta == 0)),
        "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
        "median_delta": float(np.median(delta)) if len(delta) else np.nan,
    }
    if np.sum(delta != 0) == 0:
        out.update(
            {
                "wilcoxon_two_sided_stat": np.nan,
                "wilcoxon_two_sided_p": np.nan,
                "wilcoxon_greater_stat": np.nan,
                "wilcoxon_greater_p": np.nan,
                "wilcoxon_less_stat": np.nan,
                "wilcoxon_less_p": np.nan,
            }
        )
        return out
    for alternative in ("two-sided", "greater", "less"):
        result = stats.wilcoxon(on, off, alternative=alternative, zero_method="wilcox", mode="auto")
        out[f"wilcoxon_{alternative.replace('-', '_')}_stat"] = float(result.statistic)
        out[f"wilcoxon_{alternative.replace('-', '_')}_p"] = float(result.pvalue)
    return out


def collect_rows(base_dir: str, monkeys: list[str], experiments: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_import_path()
    from src.load import _root_name, load_data

    session_rows = []
    target_rows = []
    trial_rows = []

    for experiment in experiments:
        for monkey in monkeys:
            data_dir = Path(base_dir) / monkey / experiment
            if not data_dir.exists():
                print(f"Skipping {monkey} for {experiment} (directory not found).")
                continue
            pkl_files = sorted(data_dir.glob("*_trials.pkl"))
            if not pkl_files:
                continue
            ai_candidates = []
            for ai_dir_name in ("AIFiles", "aiLog"):
                ai_dir = data_dir / ai_dir_name
                if ai_dir.is_dir():
                    ai_candidates.extend(glob.glob(str(ai_dir / "*_aitrials.pkl")))
            ai_map = {_root_name(fp): fp for fp in ai_candidates}

            for session_index, pkl_file in enumerate(pkl_files):
                trials = load_data(str(pkl_file))[0]
                correct = [t for t in trials if get_attr(t, "answer", None) == 1]
                incorrect = [t for t in trials if get_attr(t, "answer", None) in (3, 5, 6)]
                source_file = pkl_file.name
                ai_path = ai_map.get(_root_name(str(pkl_file)))
                ai_count = 0
                if ai_path is not None:
                    try:
                        import pickle

                        with open(ai_path, "rb") as f:
                            obj = pickle.load(f)
                        ai_trials = obj[1] if isinstance(obj, (list, tuple)) and len(obj) >= 2 else obj
                        ai_count = len(ai_trials) if isinstance(ai_trials, list) else 0
                    except Exception:
                        ai_count = np.nan
                session_uid = f"{monkey}|{experiment}|{session_index:02d}|{source_file}"
                current_session_rows = []

                for success, bucket in ((1, correct), (0, incorrect)):
                    for trial in bucket:
                        factor = ai_factor(trial)
                        if factor not in CONDITION_LABELS:
                            continue
                        included = condition_included(experiment, trial)
                        row = {
                            "monkey": monkey,
                            "experiment": experiment,
                            "session_index": session_index,
                            "session_uid": session_uid,
                            "source_file": source_file,
                            "trial_id": trial_id(trial),
                            "target": target_label(trial),
                            "condition": CONDITION_LABELS[factor],
                            "ai_velocity_factor": factor,
                            "success": success,
                            "included_in_success_analysis": int(included),
                            "answer": get_attr(trial, "answer", np.nan),
                        }
                        trial_rows.append(row)
                        current_session_rows.append(row)

                included_df = pd.DataFrame(
                    [row for row in current_session_rows if row["included_in_success_analysis"]]
                )
                base = {
                    "monkey": monkey,
                    "experiment": experiment,
                    "session_index": session_index,
                    "session_uid": session_uid,
                    "source_file": source_file,
                    "n_loaded_trials": len(trials),
                    "n_correct_trials_loaded": len(correct),
                    "n_incorrect_trials_loaded": len(incorrect),
                    "n_ai_log_trials": ai_count,
                }
                if included_df.empty:
                    session_rows.append(base)
                    continue

                for condition, sub in included_df.groupby("condition"):
                    successes = int(sub["success"].sum())
                    total = int(len(sub))
                    session_rows.append(
                        {
                            **base,
                            "condition": condition,
                            "n_trials": total,
                            "n_success": successes,
                            "success_rate": successes / total if total else np.nan,
                        }
                    )

                for (condition, target), sub in included_df.groupby(["condition", "target"]):
                    successes = int(sub["success"].sum())
                    total = int(len(sub))
                    target_rows.append(
                        {
                            "monkey": monkey,
                            "experiment": experiment,
                            "session_index": session_index,
                            "session_uid": session_uid,
                            "source_file": source_file,
                            "condition": condition,
                            "target": target,
                            "n_trials": total,
                            "n_success": successes,
                            "success_rate": successes / total if total else np.nan,
                        }
                    )

    return pd.DataFrame(trial_rows), pd.DataFrame(session_rows), pd.DataFrame(target_rows)


def summarize_sessions(session_rows: pd.DataFrame) -> pd.DataFrame:
    if session_rows.empty:
        return pd.DataFrame()
    pivot = session_rows.pivot_table(
        index=["monkey", "experiment", "session_index", "session_uid", "source_file"],
        columns="condition",
        values=["success_rate", "n_trials", "n_success"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{condition}".replace(" ", "_").replace("-", "_") for metric, condition in pivot.columns]
    pivot = pivot.reset_index()

    rows = []
    for (monkey, experiment), sub in pivot.groupby(["monkey", "experiment"], sort=False):
        on_col = "success_rate_Shared_control"
        off_col = "success_rate_BCI_only"
        stats_row = wilcoxon_rows(sub.get(on_col, []), sub.get(off_col, []))
        row = {
            "monkey": monkey,
            "experiment": experiment,
            **stats_row,
        }
        for condition, prefix in (("Shared_control", "shared_control"), ("BCI_only", "bci_only")):
            rate_col = f"success_rate_{condition}"
            n_col = f"n_trials_{condition}"
            s_col = f"n_success_{condition}"
            rates = sub[rate_col].astype(float) if rate_col in sub else pd.Series(dtype=float)
            total_trials = int(sub[n_col].sum()) if n_col in sub else 0
            total_success = int(sub[s_col].sum()) if s_col in sub else 0
            row[f"{prefix}_mean_pct"] = float(100.0 * rates.mean()) if len(rates) else np.nan
            row[f"{prefix}_sd_pct"] = float(100.0 * rates.std(ddof=1)) if len(rates) > 1 else np.nan
            row[f"{prefix}_pooled_pct"] = float(100.0 * total_success / total_trials) if total_trials else np.nan
            row[f"{prefix}_n_trials"] = total_trials
            row[f"{prefix}_n_success"] = total_success
        row["mean_gain_pp"] = row["shared_control_mean_pct"] - row["bci_only_mean_pct"]
        row["pooled_gain_pp"] = row["shared_control_pooled_pct"] - row["bci_only_pooled_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_target_session_rows(target_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    pivot = target_rows.pivot_table(
        index=["monkey", "experiment", "session_index", "session_uid", "source_file", "target"],
        columns="condition",
        values=["success_rate", "n_trials", "n_success"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{condition}".replace(" ", "_").replace("-", "_") for metric, condition in pivot.columns]
    pivot = pivot.reset_index()
    for col in ("success_rate_Shared_control", "success_rate_BCI_only"):
        if col not in pivot:
            pivot[col] = np.nan
    pivot["baseline_bci_only_pct"] = 100.0 * pivot["success_rate_BCI_only"]
    pivot["shared_control_gain_pp"] = 100.0 * (
        pivot["success_rate_Shared_control"] - pivot["success_rate_BCI_only"]
    )
    pivot["has_both_conditions"] = (
        pivot["success_rate_Shared_control"].notna() & pivot["success_rate_BCI_only"].notna()
    ).astype(int)

    summaries = []
    inclusion_sets = {
        "all_obstacle_experiments": OBSTACLE_GAIN_EXPERIMENTS,
        "fixed_plus_original_appearing": ["AI Obstacle", "AI Appearing Obstacle"],
        "appearing_only_original_and_followup": ["AI Appearing Obstacle", "AI Appearing Obstacle 2"],
        "all_experiments_including_respawn": DEFAULT_EXPERIMENTS,
    }
    for label, experiments in inclusion_sets.items():
        sub = pivot[pivot["experiment"].isin(experiments) & pivot["has_both_conditions"].eq(1)]
        summaries.append(
            {
                "inclusion_set": label,
                "experiments": ";".join(experiments),
                "n_target_session_rows": int(len(sub)),
                "n_unique_sessions": int(sub["session_uid"].nunique()),
                "n_target_labels": int(sub["target"].nunique()) if len(sub) else 0,
                "matches_240_rows": bool(len(sub) == 240),
                "matches_48_sessions": bool(sub["session_uid"].nunique() == 48),
            }
        )
    return pivot, pd.DataFrame(summaries)


def compare_claims(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for claim in MANUSCRIPT_CLAIMS:
        match = summary[
            summary["monkey"].eq(claim["monkey"])
            & summary["experiment"].eq(claim["experiment"])
        ]
        if match.empty:
            rows.append({**claim, "status": "missing computed row"})
            continue
        row = match.iloc[0].to_dict()
        computed = {
            "computed_shared_control_mean_pct": row.get("shared_control_mean_pct", np.nan),
            "computed_bci_only_mean_pct": row.get("bci_only_mean_pct", np.nan),
            "computed_shared_control_pooled_pct": row.get("shared_control_pooled_pct", np.nan),
            "computed_bci_only_pooled_pct": row.get("bci_only_pooled_pct", np.nan),
            "computed_mean_gain_pp": row.get("mean_gain_pp", np.nan),
            "computed_pooled_gain_pp": row.get("pooled_gain_pp", np.nan),
            "computed_median_delta": row.get("median_delta", np.nan),
            "computed_wilcoxon_two_sided_p": row.get("wilcoxon_two_sided_p", np.nan),
            "computed_wilcoxon_greater_p": row.get("wilcoxon_greater_p", np.nan),
            "computed_wilcoxon_less_p": row.get("wilcoxon_less_p", np.nan),
            "n_sessions": row.get("n_sessions", np.nan),
        }
        notes = []
        reported_sc = claim.get("reported_shared_control_pct")
        if reported_sc is not None and math.isfinite(computed["computed_shared_control_mean_pct"]):
            notes.append(
                f"reported SC {reported_sc:.1f} vs mean {computed['computed_shared_control_mean_pct']:.1f} vs pooled {computed['computed_shared_control_pooled_pct']:.1f}"
            )
        reported_off = claim.get("reported_bci_only_pct")
        if reported_off is not None and math.isfinite(computed["computed_bci_only_mean_pct"]):
            notes.append(
                f"reported BCI-only {reported_off:.1f} vs mean {computed['computed_bci_only_mean_pct']:.1f} vs pooled {computed['computed_bci_only_pooled_pct']:.1f}"
            )
        reported_p = claim.get("reported_p_two_sided")
        if reported_p is not None and math.isfinite(computed["computed_wilcoxon_two_sided_p"]):
            notes.append(
                f"reported two-sided p {reported_p:.5g} vs computed two-sided {computed['computed_wilcoxon_two_sided_p']:.5g}; one-sided greater {computed['computed_wilcoxon_greater_p']:.5g}; one-sided less {computed['computed_wilcoxon_less_p']:.5g}"
            )
        rows.append(
            {
                **claim,
                **computed,
                "status": "computed",
                "audit_note": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def write_report(paths: dict[str, Path], summary: pd.DataFrame, claims: pd.DataFrame, denom: pd.DataFrame) -> None:
    lines = [
        "# Success-rate verification audit",
        "",
        "Scope: read raw trial pickles from `config.yaml` base_dir and verify success-rate/session-inclusion quantities without regenerating manuscript figures.",
        "",
        "## Key checks",
        "",
    ]
    if not claims.empty:
        for _, row in claims.iterrows():
            note = row.get("audit_note", "")
            lines.append(f"- `{row['claim_id']}`: {note or row.get('status', '')}")
    if not denom.empty:
        lines.extend(["", "## Target x session denominator candidates", ""])
        for _, row in denom.iterrows():
            lines.append(
                f"- `{row['inclusion_set']}`: {row['n_target_session_rows']} rows from {row['n_unique_sessions']} sessions; matches 240/48 = {row['matches_240_rows'] and row['matches_48_sessions']}."
            )
    lines.extend(
        [
            "",
            "## Output files",
            "",
        ]
    )
    for label, path in paths.items():
        if label != "report":
            lines.append(f"- `{label}`: `{path.relative_to(PROJECT_ROOT)}`")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(base_dir: str, monkeys: list[str], experiments: list[str]) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_rows, session_rows, target_rows = collect_rows(base_dir, monkeys, experiments)
    summary = summarize_sessions(session_rows)
    target_session_rows, denominator_summary = summarize_target_session_rows(target_rows)
    claim_comparison = compare_claims(summary)

    paths = {
        "trial_rows": OUTPUT_DIR / "success_rate_audit_trial_rows.csv",
        "session_condition_rows": OUTPUT_DIR / "success_rate_audit_session_condition_rows.csv",
        "session_summary": OUTPUT_DIR / "success_rate_audit_session_summary.csv",
        "target_condition_rows": OUTPUT_DIR / "success_rate_audit_target_condition_rows.csv",
        "target_session_rows": OUTPUT_DIR / "success_rate_audit_target_session_rows.csv",
        "target_session_denominator_summary": OUTPUT_DIR / "success_rate_target_session_denominator_summary.csv",
        "manuscript_claim_comparison": OUTPUT_DIR / "success_rate_manuscript_claim_comparison.csv",
        "metadata": OUTPUT_DIR / "success_rate_audit_metadata.json",
        "report": OUTPUT_DIR / "success_rate_verification_report.md",
    }

    trial_rows.to_csv(paths["trial_rows"], index=False)
    session_rows.to_csv(paths["session_condition_rows"], index=False)
    summary.to_csv(paths["session_summary"], index=False)
    target_rows.to_csv(paths["target_condition_rows"], index=False)
    target_session_rows.to_csv(paths["target_session_rows"], index=False)
    denominator_summary.to_csv(paths["target_session_denominator_summary"], index=False)
    claim_comparison.to_csv(paths["manuscript_claim_comparison"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "base_dir": base_dir,
                "monkeys": monkeys,
                "experiments": experiments,
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(paths, summary, claim_comparison, denominator_summary)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args()

    base_dir = load_base_dir(args.base_dir)
    paths = run(base_dir, args.monkeys, args.experiments)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
