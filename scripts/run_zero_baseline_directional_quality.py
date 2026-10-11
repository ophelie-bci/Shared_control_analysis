"""Audit exact-zero BCI-only baseline rows and decoded directional quality.

This analysis separates target x session observations with exactly 0% BCI-only
success from low-but-nonzero observations, and checks whether BCI-only trials in
the exact-zero rows still contain decoded velocity aligned with the cued target.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_TABLE = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_target_session_rows.csv"
OUT_DIR = PROJECT_ROOT / "tables" / "Gain curve zero baseline directional quality"
ANSWER_CODES = {1, 3, 5, 6}


def load_gain_curve_base_dir() -> str:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    gain_config = config.get("gain_curve_pooling") or {}
    base_dir = gain_config.get("base_dir")
    if not base_dir:
        raise ValueError("Set gain_curve_pooling.base_dir in config.yaml.")
    return str(base_dir)


DEFAULT_BASE_DIR = None


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_array(value):
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _series_from_dict(container, key: str) -> np.ndarray:
    if not isinstance(container, dict):
        return np.asarray([], dtype=float)
    return _as_array(container.get(key, []))


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


def _is_bci_only(trial) -> bool:
    try:
        return float(_get(trial, "aiVelocityFactor", np.nan)) == 0.0
    except Exception:
        return False


def _directional_quality(trial, min_speed: float = 1e-9) -> dict:
    model_velocity = _get(trial, "modelVelocity", {})
    avatar_trajectory = _get(trial, "avatarTrajectory", {})
    target = _as_array(_get(trial, "targetPosition", []))
    if target.size < 3:
        return {"n_samples": 0, "mean_cosine": np.nan, "mean_angle_deg": np.nan}

    vx = _series_from_dict(model_velocity, "vx")
    vz = _series_from_dict(model_velocity, "vz")
    x = _series_from_dict(avatar_trajectory, "x")
    z = _series_from_dict(avatar_trajectory, "z")
    time_values = _series_from_dict(model_velocity, "time")
    n = min(vx.size, vz.size, x.size, z.size)
    if n == 0:
        return {"n_samples": 0, "mean_cosine": np.nan, "mean_angle_deg": np.nan}

    vx = vx[:n]
    vz = vz[:n]
    x = x[:n]
    z = z[:n]
    decoded = np.column_stack([vx, vz])
    to_target = np.column_stack([target[0] - x, target[2] - z])
    decoded_norm = np.linalg.norm(decoded, axis=1)
    target_norm = np.linalg.norm(to_target, axis=1)
    valid = np.isfinite(decoded).all(axis=1) & np.isfinite(to_target).all(axis=1)
    valid &= decoded_norm > min_speed
    valid &= target_norm > min_speed

    target_onset = _get(trial, "targetOnset", np.nan)
    if time_values.size >= n and np.isfinite(target_onset):
        valid &= time_values[:n] >= float(target_onset)

    if not np.any(valid):
        return {"n_samples": 0, "mean_cosine": np.nan, "mean_angle_deg": np.nan}

    cosine = np.sum(decoded[valid] * to_target[valid], axis=1) / (decoded_norm[valid] * target_norm[valid])
    cosine = np.clip(cosine, -1.0, 1.0)
    return {
        "n_samples": int(cosine.size),
        "mean_cosine": float(np.mean(cosine)),
        "mean_angle_deg": float(np.degrees(np.arccos(np.mean(cosine)))),
    }


def _baseline_group(value: float) -> str:
    if value == 0:
        return "exact_0"
    if 0 < value <= 0.10:
        return "gt0_to_10"
    if 0.10 < value <= 0.20:
        return "gt10_to_20"
    return "gt20"


def _mean_ci(values) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, 0
    mean = float(values.mean())
    if values.size < 2:
        return mean, np.nan, int(values.size)
    ci = float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return mean, ci, int(values.size)


def _table_text(df: pd.DataFrame, **kwargs) -> str:
    try:
        return df.to_markdown(index=False, **kwargs)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def collect_directional_rows(data: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import LoadPkl

    wanted = data[data["baseline_group"].isin(["exact_0", "gt0_to_10"])].copy()
    rows = []
    for _, obs in wanted.iterrows():
        path = Path(base_dir) / obs["source_monkey_folder"] / obs["experiment"] / obs["session_file"]
        loaded = LoadPkl(str(path))
        _params, trials = loaded if isinstance(loaded, tuple) else (None, loaded)
        for trial in trials:
            if not _include_trial(obs["experiment"], trial):
                continue
            if not _is_bci_only(trial):
                continue
            if _target_label(trial) != obs["target"]:
                continue
            quality = _directional_quality(trial)
            rows.append({
                "baseline_group": obs["baseline_group"],
                "monkey": obs["monkey"],
                "source_monkey_folder": obs["source_monkey_folder"],
                "experiment": obs["experiment"],
                "task": obs["task"],
                "session_index": obs["session_index"],
                "session_file": obs["session_file"],
                "session_uid": obs["session_uid"],
                "target": obs["target"],
                "baseline_pct": obs["baseline_pct"],
                "trial_id": _get(trial, "trial", np.nan),
                "answer": _get(trial, "answer", np.nan),
                "n_samples": quality["n_samples"],
                "mean_cosine": quality["mean_cosine"],
                "mean_angle_deg": quality["mean_angle_deg"],
            })
    return pd.DataFrame(rows)


def summarize_directional(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trials.empty:
        return pd.DataFrame(rows)
    valid_trials = trials[trials["n_samples"].gt(0) & np.isfinite(trials["mean_cosine"])].copy()
    for group, sub in valid_trials.groupby("baseline_group", observed=False):
        mean_cos, ci_cos, n_trials = _mean_ci(sub["mean_cosine"])
        mean_ang, ci_ang, _ = _mean_ci(sub["mean_angle_deg"])
        rows.append({
            "baseline_group": group,
            "n_target_session_rows": int(trials[trials["baseline_group"].eq(group)]["session_uid"].astype(str).str.cat(trials[trials["baseline_group"].eq(group)]["target"].astype(str), sep="|").nunique()),
            "n_bci_only_trials_with_velocity": n_trials,
            "mean_trial_cosine": mean_cos,
            "ci95_trial_cosine": ci_cos,
            "mean_trial_angle_deg": mean_ang,
            "ci95_trial_angle_deg": ci_ang,
            "median_trial_cosine": float(sub["mean_cosine"].median()) if not sub.empty else np.nan,
        })
    return pd.DataFrame(rows)


def run_analysis(base_dir: str | None = DEFAULT_BASE_DIR) -> dict[str, Path]:
    if base_dir is None:
        base_dir = load_gain_curve_base_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_TABLE)
    data["baseline_group"] = data["bci_only_success"].map(_baseline_group)

    counts = data.copy()
    counts["success_step_pct"] = 100.0 / counts["n_bci_only_trials"]
    counts_out = counts[[
        "baseline_group",
        "monkey",
        "experiment",
        "task",
        "session_index",
        "session_file",
        "target",
        "n_bci_only_trials",
        "bci_only_successes",
        "bci_only_success",
        "baseline_pct",
        "success_step_pct",
        "shared_control_gain",
        "gain_pp",
    ]].copy()

    group_summary_rows = []
    for group, sub in counts.groupby("baseline_group", observed=False):
        group_summary_rows.append({
            "baseline_group": group,
            "n_target_session_rows": int(len(sub)),
            "mean_n_bci_only_trials": float(sub["n_bci_only_trials"].mean()),
            "median_n_bci_only_trials": float(sub["n_bci_only_trials"].median()),
            "min_n_bci_only_trials": int(sub["n_bci_only_trials"].min()),
            "max_n_bci_only_trials": int(sub["n_bci_only_trials"].max()),
            "median_success_step_pct": float(sub["success_step_pct"].median()),
        })
    group_summary = pd.DataFrame(group_summary_rows)

    value_freq = (
        counts.assign(baseline_pct_rounded=counts["baseline_pct"].round(6))
        .groupby("baseline_pct_rounded", observed=False)
        .agg(
            n_target_session_rows=("baseline_pct_rounded", "size"),
            median_n_bci_only_trials=("n_bci_only_trials", "median"),
        )
        .reset_index()
        .sort_values("baseline_pct_rounded")
    )

    directional = collect_directional_rows(data, base_dir)
    directional_summary = summarize_directional(directional)

    outputs = {
        "target_session_trial_counts": OUT_DIR / "target_session_bci_only_trial_counts.csv",
        "baseline_group_summary": OUT_DIR / "baseline_group_trial_count_summary.csv",
        "baseline_value_frequencies": OUT_DIR / "baseline_value_frequencies.csv",
        "directional_quality_trials": OUT_DIR / "zero_and_low_baseline_directional_quality_trials.csv",
        "directional_quality_summary": OUT_DIR / "zero_and_low_baseline_directional_quality_summary.csv",
        "report": OUT_DIR / "zero_baseline_directional_quality_report.md",
    }
    counts_out.to_csv(outputs["target_session_trial_counts"], index=False)
    group_summary.to_csv(outputs["baseline_group_summary"], index=False)
    value_freq.to_csv(outputs["baseline_value_frequencies"], index=False)
    directional.to_csv(outputs["directional_quality_trials"], index=False)
    directional_summary.to_csv(outputs["directional_quality_summary"], index=False)

    exact = directional_summary[directional_summary["baseline_group"].eq("exact_0")]
    low = directional_summary[directional_summary["baseline_group"].eq("gt0_to_10")]
    lines = [
        "# Exact-Zero Baseline Directional Quality Audit",
        "",
        "BCI-only baseline success was separated into exact 0%, >0-10%, >10-20%, and >20% target x session rows.",
        "",
        "## BCI-only Trial Counts",
        "",
        _table_text(group_summary, floatfmt=".3f"),
        "",
        "## Directional Quality",
        "",
        _table_text(directional_summary, floatfmt=".3f") if not directional_summary.empty else "No directional-quality rows were available.",
        "",
    ]
    if not exact.empty:
        row = exact.iloc[0]
        lines.extend([
            "## Interpretation",
            "",
            (
                f"Exact-zero baseline rows contributed {int(row['n_bci_only_trials_with_velocity'])} "
                f"BCI-only trials with usable decoded velocity. Mean trial-level cosine alignment was "
                f"{row['mean_trial_cosine']:.3f} +/- {row['ci95_trial_cosine']:.3f} (95% CI), "
                f"corresponding to a mean angular error of {row['mean_trial_angle_deg']:.1f} deg."
            ),
        ])
        if not low.empty:
            low_row = low.iloc[0]
            lines.append(
                f"For >0-10% rows, mean trial-level cosine alignment was {low_row['mean_trial_cosine']:.3f} "
                f"+/- {low_row['ci95_trial_cosine']:.3f} across {int(low_row['n_bci_only_trials_with_velocity'])} trials."
            )
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None)
    args = parser.parse_args()
    outputs = run_analysis(args.base_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
