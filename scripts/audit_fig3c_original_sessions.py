"""Audit which sessions contribute to the original Figure 3C gain scatter.

The original Figure 3C path in run_success_rate.py creates one point per
monkey x experiment x target, after pooling BCI-only baseline trials across
sessions and averaging per-session target gains. This script records which
session files are available and which actually contain AI and No-AI trials for
each target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = None
FIG3C_ALLOWED_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
]


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)
ANSWER_CODES = {1, 5, 3, 6, 9, 10, 4}
DEFAULT_CODE_DATASETS = [
    ("Monkey 1", "Monkey 1", exp, exp) for exp in FIG3C_ALLOWED_EXPERIMENTS
] + [
    ("Monkey 2", "Monkey 2", exp, exp) for exp in FIG3C_ALLOWED_EXPERIMENTS
]
MAPPED_MANUSCRIPT_DATASETS = [
    ("Monkey 1", "Monkey 3", "AI Obstacle", "AI Obstacle"),
    ("Monkey 1", "Monkey 3", "AI Appearing Obstacle", "AI Appearing Obstacle"),
    ("Monkey 1", "Monkey 3", "AI Appearing Obstacle 2", "AI Appearing Obstacle 2"),
    ("Monkey 2", "Monkey 1", "AI Obstacle", "AI Obstacle"),
    ("Monkey 2", "Monkey 1", "AI Appearing Obstacle", "AI Appearing Obstacle"),
    ("Monkey 2", "Monkey 1", "AI Appearing Obstacle 2", "AI Appearing Obstacle 2"),
]


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _target_label(trial) -> str:
    _ensure_import_path()
    from src.constants import target_mapping

    try:
        return target_mapping.get(tuple(_get(trial, "targetPosition")), "unknown")
    except Exception:
        return "unknown"


def _include_trial(experiment: str, trial) -> bool:
    if _get(trial, "answer", None) not in ANSWER_CODES:
        return False
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return _get(trial, "obstacleAppearedTime", None) is not None
    if experiment == "AI Respawn":
        try:
            return not np.isnan(np.asarray(_get(trial, "targetJumpPosition")).astype(float)).any()
        except Exception:
            return False
    return True


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


def _load_trials(pkl_file: Path):
    _ensure_import_path()
    from src.load import LoadPkl

    loaded = LoadPkl(str(pkl_file))
    return loaded[1] if isinstance(loaded, tuple) else loaded


def audit_datasets(base_dir: str, datasets: list[tuple[str, str, str, str]], audit_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_rows = []
    target_rows = []
    for monkey_label, source_monkey, experiment, plotted_experiment in datasets:
        data_dir = Path(base_dir) / source_monkey / experiment
        pkl_files = sorted(data_dir.glob("*_trials.pkl")) if data_dir.exists() else []
        per_target_gains: dict[str, list[float]] = {}
        pooled_no_ai: dict[str, list[int]] = {}
        pooled_ai: dict[str, list[int]] = {}
        for session_index, pkl_file in enumerate(pkl_files):
            trials = _load_trials(pkl_file)
            counts: dict[tuple[str, str], list[int]] = {}
            for trial in trials:
                if not _include_trial(experiment, trial):
                    continue
                condition = _condition(trial)
                if condition is None:
                    continue
                target = _target_label(trial)
                if target == "unknown":
                    continue
                counts.setdefault((target, condition), []).append(int(_get(trial, "answer", None) == 1))
            targets = sorted({target for target, _ in counts})
            for target in targets:
                ai = counts.get((target, "AI"), [])
                no_ai = counts.get((target, "No AI"), [])
                for values, destination in [(ai, pooled_ai), (no_ai, pooled_no_ai)]:
                    destination.setdefault(target, []).extend(values)
                has_both = bool(ai) and bool(no_ai)
                if has_both:
                    gain = float(np.mean(ai) - np.mean(no_ai))
                    per_target_gains.setdefault(target, []).append(gain)
                session_rows.append({
                    "audit": audit_name,
                    "monkey": monkey_label,
                    "source_monkey_folder": source_monkey,
                    "experiment": plotted_experiment,
                    "source_experiment_folder": experiment,
                    "session_index": session_index,
                    "session_file": pkl_file.name,
                    "target": target,
                    "n_ai_trials": len(ai),
                    "n_no_ai_trials": len(no_ai),
                    "has_both_conditions": has_both,
                    "session_gain_used_if_both": gain if has_both else np.nan,
                })
        for target in sorted(set(pooled_ai) | set(pooled_no_ai)):
            ai = pooled_ai.get(target, [])
            no_ai = pooled_no_ai.get(target, [])
            contributing_sessions = [
                row["session_file"]
                for row in session_rows
                if row["audit"] == audit_name
                and row["monkey"] == monkey_label
                and row["source_experiment_folder"] == experiment
                and row["target"] == target
                and row["has_both_conditions"]
            ]
            target_rows.append({
                "audit": audit_name,
                "monkey": monkey_label,
                "source_monkey_folder": source_monkey,
                "experiment": plotted_experiment,
                "source_experiment_folder": experiment,
                "target": target,
                "n_available_session_files": len(pkl_files),
                "n_sessions_with_target_any_condition": int(sum(
                    1
                    for row in session_rows
                    if row["audit"] == audit_name
                    and row["monkey"] == monkey_label
                    and row["source_experiment_folder"] == experiment
                    and row["target"] == target
                )),
                "n_sessions_with_both_conditions": len(contributing_sessions),
                "n_ai_trials_pooled": len(ai),
                "n_no_ai_trials_pooled": len(no_ai),
                "baseline_off_pct": 100.0 * float(np.mean(no_ai)) if no_ai else np.nan,
                "gain_pp_mean_of_session_gains": 100.0 * float(np.mean(per_target_gains.get(target, [np.nan]))),
                "contributing_session_files": "; ".join(contributing_sessions),
            })
    return pd.DataFrame(session_rows), pd.DataFrame(target_rows)


def run(base_dir: str) -> dict[str, Path]:
    out = PROJECT_ROOT / "tables" / "Figure 3C original session audit"
    out.mkdir(parents=True, exist_ok=True)
    default_sessions, default_targets = audit_datasets(base_dir, DEFAULT_CODE_DATASETS, "default_run_success_rate_paths")
    mapped_sessions, mapped_targets = audit_datasets(base_dir, MAPPED_MANUSCRIPT_DATASETS, "mapped_manuscript_source_paths")
    sessions = pd.concat([default_sessions, mapped_sessions], ignore_index=True)
    targets = pd.concat([default_targets, mapped_targets], ignore_index=True)
    experiment_summary = (
        targets.groupby(["audit", "monkey", "source_monkey_folder", "experiment", "source_experiment_folder"], dropna=False)
        .agg(
            n_plotted_target_points=("target", "size"),
            n_available_session_files=("n_available_session_files", "max"),
            min_sessions_with_both_conditions=("n_sessions_with_both_conditions", "min"),
            median_sessions_with_both_conditions=("n_sessions_with_both_conditions", "median"),
            max_sessions_with_both_conditions=("n_sessions_with_both_conditions", "max"),
        )
        .reset_index()
    )

    paths = {
        "session_target_audit": out / "figure3c_original_session_target_audit.csv",
        "plotted_target_points": out / "figure3c_original_plotted_target_points.csv",
        "experiment_summary": out / "figure3c_original_experiment_summary.csv",
        "report": out / "figure3c_original_session_audit_report.md",
    }
    sessions.to_csv(paths["session_target_audit"], index=False)
    targets.to_csv(paths["plotted_target_points"], index=False)
    experiment_summary.to_csv(paths["experiment_summary"], index=False)

    default_nonempty = experiment_summary[
        (experiment_summary["audit"] == "default_run_success_rate_paths")
        & (experiment_summary["n_plotted_target_points"] > 0)
    ]
    mapped_nonempty = experiment_summary[
        (experiment_summary["audit"] == "mapped_manuscript_source_paths")
        & (experiment_summary["n_plotted_target_points"] > 0)
    ]
    lines = [
        "# Figure 3C Original Session Audit",
        "",
        "The original `plot_summary_scatter()` allows only `AI Obstacle`, `AI Appearing Obstacle`, and `AI Appearing Obstacle 2`; `AI Respawn` is not plotted in Figure 3C.",
        "The original Figure 3C path creates one plotted point per monkey x experiment x target, not one point per target x session.",
        "",
        "## Default run_success_rate paths found under base_dir",
        "",
        default_nonempty.to_markdown(index=False) if not default_nonempty.empty else "No default-code AI experiment paths produced plotted target points.",
        "",
        "## Mapped manuscript-source paths found under base_dir",
        "",
        mapped_nonempty.to_markdown(index=False) if not mapped_nonempty.empty else "No mapped manuscript paths produced plotted target points.",
        "",
        "## Important Interpretation",
        "",
        "A plotted target point is based on pooled No-AI target success for the x-axis and the mean of session-level target gains for the y-axis.",
        "The audit reports `n_sessions_with_both_conditions` because a target-session can only provide an empirical session gain when both AI and No-AI trials are present for that target.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    args = parser.parse_args()
    args.base_dir = load_base_dir(args.base_dir)
    for name, path in run(args.base_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
