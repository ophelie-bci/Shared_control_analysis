"""Verify experimental safety-override exposure from raw obstacle-task trials.

This audit preserves the original implementation in
scripts/run_obstacle_session_metrics.py. It applies the verified Figure 3
obstacle-session inclusion list and eligible answer codes, then exports
provenance under analysis_results/override_exposure_verification/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis_results" / "override_exposure_verification"
FIG3_SOURCE_ROWS = (
    PROJECT_ROOT
    / "analysis_results"
    / "success_rate_verification"
    / "figure3_success_rate_source_rows.csv"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_obstacle_session_metrics as original  # noqa: E402
from src.load import load_files  # noqa: E402


MONKEYS = ["Monkey 1", "Monkey 2"]
EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
ELIGIBLE_ANSWERS = {1, 3, 5, 6}
TASK_FAMILY = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle",
}
TASK_LABEL = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle follow-up",
}
MANUSCRIPT_CLAIMS = {
    "overall_override_percent": 9.0,
    "median_150ms_equivalents": 3.3,
    "iqr_150ms_equivalents_low": 0.0,
    "iqr_150ms_equivalents_high": 5.3,
    "median_override_duration_s": 0.50,
    "fixed_obstacle_override_percent": 6.2,
    "appearing_obstacle_override_percent": 12.1,
}


def _load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _verified_sessions() -> pd.DataFrame:
    rows = pd.read_csv(FIG3_SOURCE_ROWS)
    rows = rows[rows["experiment"].isin(EXPERIMENTS)].copy()
    rows = rows[["panel", "monkey", "experiment", "session_index", "source_file"]].drop_duplicates()
    rows["session_index"] = rows["session_index"].astype(int)
    rows["task_family"] = rows["experiment"].map(TASK_FAMILY)
    rows["task_label"] = rows["experiment"].map(TASK_LABEL)
    rows["includes_followup_session"] = rows["experiment"].eq("AI Appearing Obstacle 2")
    return rows


def _answer_code(trial) -> int | None:
    value = original._get(trial, "answer", None)
    try:
        return int(value)
    except Exception:
        return None


def _condition(trial) -> str | None:
    try:
        factor = float(original._get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None
    if factor == 1.0:
        return "AI"
    if factor == 0.0:
        return "No AI"
    return None


def _raw_event_values(events) -> list[float | str]:
    out = []
    for ev in original._finite_events(events):
        if isinstance(ev, dict):
            for key in ("time", "timestamp", "ts", "t", "output_ts", "unitytime", "unityrealtime", "idx", "index"):
                if key in ev:
                    value = original._as_scalar(ev[key])
                    out.append(float(value) if np.isfinite(value) else json.dumps(ev, sort_keys=True))
                    break
            else:
                out.append(json.dumps(ev, sort_keys=True))
        else:
            value = original._as_scalar(ev)
            out.append(float(value) if np.isfinite(value) else str(ev))
    return out


def _override_segments(trial, output_ts: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n = len(output_ts)
    scale_s = original._timestamp_scale_to_seconds(output_ts) if n else np.nan
    scale_ms = original._timestamp_scale_to_ms(output_ts) if n else np.nan
    finite = output_ts[np.isfinite(output_ts)] if n else np.array([])
    dt_s = float(np.nanmedian(np.diff(np.sort(finite))) * scale_s) if finite.size >= 2 else np.nan

    on_events = original._finite_events(original._get(trial, "aiControlOn", []))
    off_events = original._finite_events(original._get(trial, "aiControlOff", []))
    on_idx = original._events_to_indices(on_events, output_ts, n)
    off_idx = original._events_to_indices(off_events, output_ts, n)

    # Match the original interval pairing: discard off events before first on,
    # then pair each on with the next later off or the final sample.
    off_work = list(off_idx)
    while off_work and (not on_idx or off_work[0] < on_idx[0]):
        off_work.pop(0)
    j = 0
    segments = []
    event_mask = np.zeros(n, dtype=bool)
    for start in sorted(on_idx):
        while j < len(off_work) and off_work[j] <= start:
            j += 1
        end = off_work[j] if j < len(off_work) else n - 1
        if 0 <= start < n and 0 <= end < n:
            event_mask[start : end + 1] = True
            segments.append((start, end))
        if j < len(off_work):
            j += 1

    sample_rows = []
    for idx in np.flatnonzero(event_mask):
        sample_rows.append(
            {
                "override_sample_index": int(idx),
                "output_timestamp_raw": float(output_ts[idx]) if np.isfinite(output_ts[idx]) else np.nan,
                "output_timestamp_s_relative": (
                    float((output_ts[idx] - output_ts[0]) * scale_s)
                    if n and np.isfinite(output_ts[idx]) and np.isfinite(output_ts[0]) and np.isfinite(scale_s)
                    else np.nan
                ),
            }
        )

    segment_rows = []
    for episode_index, (start, end) in enumerate(segments, start=1):
        n_samples = int(end - start + 1)
        segment_rows.append(
            {
                "actual_override_episode_index": episode_index,
                "start_sample_index": int(start),
                "end_sample_index": int(end),
                "n_override_samples": n_samples,
                "duration_from_samples_s": float(n_samples * dt_s) if np.isfinite(dt_s) else np.nan,
                "duration_150ms_equivalents": float(n_samples / 3.0),
                "start_timestamp_raw": float(output_ts[start]) if np.isfinite(output_ts[start]) else np.nan,
                "end_timestamp_raw": float(output_ts[end]) if np.isfinite(output_ts[end]) else np.nan,
                "start_timestamp_s_relative": (
                    float((output_ts[start] - output_ts[0]) * scale_s)
                    if n and np.isfinite(output_ts[start]) and np.isfinite(output_ts[0]) and np.isfinite(scale_s)
                    else np.nan
                ),
                "end_timestamp_s_relative": (
                    float((output_ts[end] - output_ts[0]) * scale_s)
                    if n and np.isfinite(output_ts[end]) and np.isfinite(output_ts[0]) and np.isfinite(scale_s)
                    else np.nan
                ),
            }
        )

    diagnostics = {
        "n_ai_control_on_events": len(on_events),
        "n_ai_control_off_events": len(off_events),
        "n_mapped_on_indices": len(on_idx),
        "n_mapped_off_indices": len(off_idx),
        "actual_override_episode_count_from_segments": len(segments),
        "velocity_log_dt_s": dt_s,
        "timestamp_scale_to_seconds": scale_s,
        "timestamp_scale_to_ms": scale_ms,
        "ai_control_on_raw": json.dumps(_raw_event_values(on_events)),
        "ai_control_off_raw": json.dumps(_raw_event_values(off_events)),
    }
    return pd.DataFrame(segment_rows), pd.DataFrame(sample_rows), diagnostics


def _collect_verified_trials(base_dir: str, verified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verified_keys = {
        (row.monkey, row.experiment, int(row.session_index), row.source_file)
        for row in verified.itertuples(index=False)
    }
    trial_rows = []
    session_rows = []
    segment_frames = []
    sample_frames = []

    for experiment in EXPERIMENTS:
        for monkey in MONKEYS:
            all_trials, *_rest, pkl_files, _ai_trials = load_files(experiment, monkey, base_dir)
            if all_trials == []:
                continue
            for session_index, trials in enumerate(all_trials):
                source_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                key = (monkey, experiment, int(session_index), source_file)
                session_uid = f"{monkey}|{experiment}|{session_index}|{source_file}"
                in_verified = key in verified_keys
                answer_counts = {}
                included_count = 0
                for trial in trials:
                    answer = _answer_code(trial)
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1
                    if not in_verified:
                        continue
                    if answer not in ELIGIBLE_ANSWERS:
                        continue
                    if not original._include_trial(experiment, trial):
                        continue
                    row = original._trial_metrics(trial, monkey, experiment, int(session_index), source_file)
                    if row is None:
                        continue
                    included_count += 1
                    row["answer"] = answer
                    row["task_family"] = TASK_FAMILY[experiment]
                    row["task_label"] = TASK_LABEL[experiment]
                    row["figure3_verified_session"] = True
                    row["eligible_answer_code"] = True
                    row["followup_session"] = experiment == "AI Appearing Obstacle 2"
                    row["movement_time_denominator_source"] = "trial_duration_s from original _duration_s"
                    row["override_numerator_source"] = "explicit aiControlOn/aiControlOff samples on aiVelocities OutputTimestamp"
                    row["n_actual_override_episodes"] = 0
                    row["velocity_log_dt_s"] = np.nan
                    row["ai_control_on_raw"] = "[]"
                    row["ai_control_off_raw"] = "[]"

                    output_ts = original._velocity_log_timestamps(trial)
                    if row["condition"] == "AI":
                        seg, samp, diag = _override_segments(trial, output_ts)
                        for k, v in diag.items():
                            if k == "actual_override_episode_count_from_segments":
                                row["n_actual_override_episodes"] = v
                            else:
                                row[k] = v
                        id_cols = {
                            "monkey": monkey,
                            "experiment": experiment,
                            "task_family": TASK_FAMILY[experiment],
                            "task_label": TASK_LABEL[experiment],
                            "session_index": int(session_index),
                            "session_file": source_file,
                            "session_uid": session_uid,
                            "trial_id": row["trial_id"],
                        }
                        if not seg.empty:
                            segment_frames.append(seg.assign(**id_cols))
                        if not samp.empty:
                            sample_frames.append(samp.assign(**id_cols))
                    trial_rows.append(row)

                session_rows.append(
                    {
                        "monkey": monkey,
                        "experiment": experiment,
                        "task_family": TASK_FAMILY.get(experiment, experiment),
                        "task_label": TASK_LABEL.get(experiment, experiment),
                        "session_index": int(session_index),
                        "session_file": source_file,
                        "session_uid": session_uid,
                        "figure3_verified_session": bool(in_verified),
                        "included_in_override_verification": bool(in_verified and included_count > 0),
                        "followup_session": experiment == "AI Appearing Obstacle 2",
                        "n_loaded_trials": int(len(trials)),
                        "n_included_trials": int(included_count),
                        "n_answer_1": int(answer_counts.get(1, 0)),
                        "n_answer_3": int(answer_counts.get(3, 0)),
                        "n_answer_5": int(answer_counts.get(5, 0)),
                        "n_answer_6": int(answer_counts.get(6, 0)),
                        "n_excluded_other_answers": int(
                            sum(count for ans, count in answer_counts.items() if ans not in ELIGIBLE_ANSWERS)
                        ),
                    }
                )

    trials_df = pd.DataFrame(trial_rows)
    sessions_df = pd.DataFrame(session_rows)
    segments_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    samples_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return trials_df, sessions_df, segments_df, samples_df


def _summary_rows(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ai = trials[trials["condition"].eq("AI")].copy()
    overall_rows = []
    task_rows = []
    unit_rows = []
    if ai.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def summarize_group(name: str, group: pd.DataFrame) -> dict:
        q25, q75 = np.nanpercentile(group["n_150ms_override_events"], [25, 75])
        return {
            "group": name,
            "n_trials": int(len(group)),
            "n_sessions": int(group["session_uid"].nunique()),
            "n_actual_override_episodes": int(group["n_actual_override_episodes"].sum()),
            "n_ai_control_on_events": int(group["n_override_events"].sum()),
            "n_override_samples": int(group["n_override_samples"].sum()),
            "n_150ms_override_equivalents_sum": float(group["n_150ms_override_events"].sum()),
            "total_override_duration_s": float(group["override_duration_s"].sum()),
            "total_movement_time_denominator_s": float(group["trial_duration_s"].sum()),
            "override_percent_of_movement_time": float(
                100.0 * group["override_duration_s"].sum() / group["trial_duration_s"].sum()
            ),
            "median_150ms_equivalents_per_trial": float(group["n_150ms_override_events"].median()),
            "iqr_150ms_equivalents_low": float(q25),
            "iqr_150ms_equivalents_high": float(q75),
            "median_override_duration_s_per_trial": float(group["override_duration_s"].median()),
            "mean_override_duration_s_per_trial": float(group["override_duration_s"].mean()),
            "fraction_trials_with_override": float(group["has_override"].mean()),
        }

    overall_rows.append(summarize_group("All obstacle shared-control trials", ai))
    for task_family, group in ai.groupby("task_family", sort=False):
        task_rows.append(summarize_group(task_family, group))
    task_denominators = ai.groupby("task_family")["trial_duration_s"].sum().to_dict()
    for (task_family, monkey, experiment), group in ai.groupby(["task_family", "monkey", "experiment"], sort=False):
        row = summarize_group(f"{monkey}|{experiment}", group)
        row["task_family"] = task_family
        row["monkey"] = monkey
        row["experiment"] = experiment
        row["task_equal_unit_weight_not_used_for_time_pooled_claim"] = 1.0 / ai[ai["task_family"].eq(task_family)][
            ["monkey", "experiment"]
        ].drop_duplicates().shape[0]
        row["task_time_pooling_weight"] = float(
            group["trial_duration_s"].sum() / task_denominators[task_family]
        )
        unit_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(task_rows), pd.DataFrame(unit_rows)


def _reference_output_trace() -> pd.DataFrame:
    path = PROJECT_ROOT / "tables" / "obstacle_session_metric_trials.csv"
    if not path.exists():
        return pd.DataFrame()
    trials = pd.read_csv(path)
    ai = trials[trials["condition"].eq("AI")].copy()
    if ai.empty:
        return pd.DataFrame()

    rows = []

    def summarize(label: str, group: pd.DataFrame) -> dict:
        q25, q75 = np.nanpercentile(group["n_150ms_override_events"], [25, 75])
        return {
            "reference_group": label,
            "n_trials": int(len(group)),
            "n_sessions": int(group["session_uid"].nunique()),
            "included_units": ";".join(
                sorted((group["monkey"].astype(str) + "|" + group["experiment"].astype(str)).unique())
            ),
            "includes_followup": bool(group["experiment"].eq("AI Appearing Obstacle 2").any()),
            "override_percent_of_movement_time": float(
                100.0 * group["override_duration_s"].sum() / group["trial_duration_s"].sum()
            ),
            "median_150ms_equivalents_per_trial": float(group["n_150ms_override_events"].median()),
            "iqr_150ms_equivalents_low": float(q25),
            "iqr_150ms_equivalents_high": float(q75),
            "median_override_duration_s_per_trial": float(group["override_duration_s"].median()),
            "total_override_duration_s": float(group["override_duration_s"].sum()),
            "total_movement_time_denominator_s": float(group["trial_duration_s"].sum()),
        }

    rows.append(summarize("reference_table_all_ai_trials", ai))
    for task_label, group in ai.groupby("task_label", sort=False):
        rows.append(summarize(f"reference_table_{task_label}", group))
    return pd.DataFrame(rows)


def _claim_check(overall: pd.DataFrame, task: pd.DataFrame) -> pd.DataFrame:
    all_row = overall.iloc[0]
    fixed = task[task["group"].eq("Fixed Obstacle")].iloc[0]
    appearing = task[task["group"].eq("Appearing Obstacle")].iloc[0]
    checks = [
        (
            "overall_override_percent",
            MANUSCRIPT_CLAIMS["overall_override_percent"],
            all_row["override_percent_of_movement_time"],
            "round_1dp",
        ),
        (
            "median_150ms_equivalents",
            MANUSCRIPT_CLAIMS["median_150ms_equivalents"],
            all_row["median_150ms_equivalents_per_trial"],
            "round_1dp",
        ),
        (
            "iqr_150ms_equivalents_low",
            MANUSCRIPT_CLAIMS["iqr_150ms_equivalents_low"],
            all_row["iqr_150ms_equivalents_low"],
            "round_1dp",
        ),
        (
            "iqr_150ms_equivalents_high",
            MANUSCRIPT_CLAIMS["iqr_150ms_equivalents_high"],
            all_row["iqr_150ms_equivalents_high"],
            "round_1dp",
        ),
        (
            "median_override_duration_s",
            MANUSCRIPT_CLAIMS["median_override_duration_s"],
            all_row["median_override_duration_s_per_trial"],
            "approx_0.01s",
        ),
        (
            "fixed_obstacle_override_percent",
            MANUSCRIPT_CLAIMS["fixed_obstacle_override_percent"],
            fixed["override_percent_of_movement_time"],
            "round_1dp",
        ),
        (
            "appearing_obstacle_override_percent",
            MANUSCRIPT_CLAIMS["appearing_obstacle_override_percent"],
            appearing["override_percent_of_movement_time"],
            "round_1dp",
        ),
    ]
    rows = []
    for claim, reported, computed, criterion in checks:
        if criterion == "round_1dp":
            ok = round(float(computed), 1) == float(reported)
        else:
            ok = abs(float(computed) - float(reported)) <= 0.01
        rows.append(
            {
                "claim": claim,
                "manuscript_reported": float(reported),
                "computed_full_precision": float(computed),
                "computed_rounded_1dp": round(float(computed), 1),
                "match": bool(ok),
                "criterion": criterion,
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    paths: dict[str, Path],
    overall: pd.DataFrame,
    task: pd.DataFrame,
    units: pd.DataFrame,
    reference_trace: pd.DataFrame,
    claims: pd.DataFrame,
    sessions: pd.DataFrame,
    base_dir: str,
) -> None:
    all_row = overall.iloc[0]
    fixed = task[task["group"].eq("Fixed Obstacle")].iloc[0]
    appearing = task[task["group"].eq("Appearing Obstacle")].iloc[0]
    followup_included = bool(sessions["followup_session"].eq(True).any())
    n_followup_sessions = int(
        sessions[sessions["included_in_override_verification"] & sessions["followup_session"]]["session_uid"].nunique()
    )
    lines = [
        "# Override Exposure Verification",
        "",
        "Scope: experimental safety-override exposure only. Reference tables and manuscript values were not modified.",
        "",
        "## Source Calculation",
        "",
        "- Starting implementation: `scripts/run_obstacle_session_metrics.py`.",
        "- Override detection uses explicit `aiControlOn`/`aiControlOff` events mapped to `aiVelocities` `OutputTimestamp` samples.",
        "- Consecutive override samples inside an on/off interval are counted sample by sample for duration; they are not collapsed to a single 150 ms event.",
        "- Actual override episodes are retained separately as paired on/off segments and raw `aiControlOn` counts.",
        "- Trial boundaries are hard boundaries: on/off pairing and sample masks are computed independently within each trial, with no carryover across trials.",
        "- Movement-time denominator for the manuscript exposure percentage is the original `trial_duration_s` per shared-control trial.",
        "",
        "## Inclusion",
        "",
        f"- Raw data source: `{base_dir}` from `config.yaml`.",
        "- Sessions: verified Figure 3 obstacle-task sessions from `analysis_results/success_rate_verification/figure3_success_rate_source_rows.csv`.",
        "- Eligible answer codes: `{1,3,5,6}`.",
        f"- Six-month follow-up included: `{followup_included}` ({n_followup_sessions} included follow-up sessions).",
        "",
        "## Full-Precision Results",
        "",
        f"- Overall: {all_row['override_percent_of_movement_time']:.15g}% of shared-control movement time; "
        f"median {all_row['median_150ms_equivalents_per_trial']:.15g} 150 ms equivalents/trial "
        f"(IQR {all_row['iqr_150ms_equivalents_low']:.15g}-{all_row['iqr_150ms_equivalents_high']:.15g}); "
        f"median duration {all_row['median_override_duration_s_per_trial']:.15g} s/trial.",
        f"- Fixed Obstacle pooled exposure: {fixed['override_percent_of_movement_time']:.15g}%.",
        f"- Appearing Obstacle pooled exposure: {appearing['override_percent_of_movement_time']:.15g}%.",
        f"- Actual override episodes from paired segments: {int(all_row['n_actual_override_episodes'])}; "
        f"duration/150 ms equivalents: {all_row['n_150ms_override_equivalents_sum']:.15g}.",
        "",
        "## Reference Output Trace",
        "",
        "The existing manuscript-facing reference table was preserved and was not regenerated. It reproduces the manuscript values from `tables/obstacle_session_metric_trials.csv`, but that table contains only Monkey 1 original Fixed/Appearing Obstacle sessions, not Monkey 2 or the six-month follow-up.",
        "",
    ]
    if not reference_trace.empty:
        for _, row in reference_trace.iterrows():
            lines.append(
                f"- {row['reference_group']}: units `{row['included_units']}`, "
                f"follow-up included = {row['includes_followup']}, "
                f"n={int(row['n_trials'])}, exposure={row['override_percent_of_movement_time']:.15g}%, "
                f"median equivalents={row['median_150ms_equivalents_per_trial']:.15g}."
            )
    lines.extend(
        [
            "",
        "## Claim Check",
        "",
        ]
    )
    for _, row in claims.iterrows():
        lines.append(
            f"- `{row['claim']}`: reported {row['manuscript_reported']:.3g}; "
            f"computed {row['computed_full_precision']:.15g}; match = {row['match']}."
        )
    lines.extend(["", "## Pooling Weights", ""])
    for _, row in units.iterrows():
        lines.append(
            f"- {row['task_family']} `{row['monkey']}|{row['experiment']}`: "
            f"n={int(row['n_trials'])}, time-pooling weight={row['task_time_pooling_weight']:.15g}, "
            f"raw time-pooled exposure={row['override_percent_of_movement_time']:.15g}%."
        )
    lines.extend(
        [
            "",
            "## Wording",
            "",
            "Under the requested verified Figure 3 obstacle-session inclusion, the manuscript numbers should be updated. The exact replacement wording is:",
            "",
            f"`Across verified obstacle-task shared-control trials, explicit safety override intervals occupied {all_row['override_percent_of_movement_time']:.1f}% of total movement time. Per trial, the median cumulative override exposure was {all_row['median_150ms_equivalents_per_trial']:.1f} equivalents of a 150 ms override interval (IQR {all_row['iqr_150ms_equivalents_low']:.1f}-{all_row['iqr_150ms_equivalents_high']:.1f}), corresponding to a median cumulative duration of approximately {all_row['median_override_duration_s_per_trial']:.2f} s. Time-pooled exposure was {fixed['override_percent_of_movement_time']:.1f}% in Fixed Obstacle and {appearing['override_percent_of_movement_time']:.1f}% in Appearing Obstacle; the Appearing Obstacle aggregate includes the six-month follow-up sessions.`",
            "",
            "If the original manuscript values are retained, the necessary qualifier is that they come from the preserved reference output table containing only Monkey 1 original Fixed/Appearing Obstacle sessions. They should not be described as the verified Figure 3 obstacle-task aggregate.",
            "",
            "Use `150 ms override-duration equivalents`, not `override events`, for `n_override_samples / 3`; actual explicit override episodes are reported separately.",
            "",
            "## Regeneration Command",
            "",
            "```powershell",
            "python scripts/verify_override_exposure.py",
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
    base_dir = str(config.get("base_dir", ""))
    if not base_dir:
        raise ValueError("config.yaml is missing base_dir")

    verified = _verified_sessions()
    trials, sessions, segments, samples = _collect_verified_trials(base_dir, verified)
    if trials.empty:
        raise RuntimeError("No verified override-exposure trials were collected.")
    overall, task, units = _summary_rows(trials)
    reference_trace = _reference_output_trace()
    claims = _claim_check(overall, task)

    rules = pd.DataFrame(
        [
            {
                "rule": "source_script",
                "value": "scripts/run_obstacle_session_metrics.py",
            },
            {
                "rule": "override_interval_source",
                "value": "explicit aiControlOn/aiControlOff events mapped to aiVelocities OutputTimestamp",
            },
            {
                "rule": "duration",
                "value": "n_override_samples multiplied by median velocity-log dt within the trial",
            },
            {
                "rule": "150ms_equivalents",
                "value": "n_override_samples / 3; this is not the actual episode count",
            },
            {
                "rule": "movement_time_denominator",
                "value": "trial_duration_s from original _duration_s, summed over shared-control trials",
            },
            {
                "rule": "trial_boundaries",
                "value": "on/off pairing is computed independently within each trial",
            },
            {
                "rule": "eligible_answers",
                "value": ",".join(str(x) for x in sorted(ELIGIBLE_ANSWERS)),
            },
        ]
    )

    paths = {
        "trial_inputs": OUT_DIR / "override_trial_inputs.csv",
        "session_inclusion": OUT_DIR / "override_session_inclusion.csv",
        "override_segments": OUT_DIR / "override_episode_segments.csv",
        "override_sample_timestamps": OUT_DIR / "override_sample_timestamps.csv",
        "detection_rules": OUT_DIR / "override_detection_rules.csv",
        "overall_results": OUT_DIR / "override_overall_results.csv",
        "task_pooled_results": OUT_DIR / "override_task_pooled_results.csv",
        "pooling_weights": OUT_DIR / "override_pooling_weights.csv",
        "reference_output_trace": OUT_DIR / "override_reference_output_trace.csv",
        "claim_check": OUT_DIR / "override_claim_check.csv",
        "metadata": OUT_DIR / "metadata.json",
        "report": OUT_DIR / "override_exposure_verification_report.md",
    }
    trials.to_csv(paths["trial_inputs"], index=False)
    sessions.to_csv(paths["session_inclusion"], index=False)
    segments.to_csv(paths["override_segments"], index=False)
    samples.to_csv(paths["override_sample_timestamps"], index=False)
    rules.to_csv(paths["detection_rules"], index=False)
    overall.to_csv(paths["overall_results"], index=False)
    task.to_csv(paths["task_pooled_results"], index=False)
    units.to_csv(paths["pooling_weights"], index=False)
    reference_trace.to_csv(paths["reference_output_trace"], index=False)
    claims.to_csv(paths["claim_check"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "base_dir": base_dir,
                "verified_session_source": str(FIG3_SOURCE_ROWS.relative_to(PROJECT_ROOT)),
                "source_script": "scripts/run_obstacle_session_metrics.py",
                "verification_script": "scripts/verify_override_exposure.py",
                "eligible_answer_codes": sorted(ELIGIBLE_ANSWERS),
                "experiments": EXPERIMENTS,
                "followup_included": bool(
                    sessions["included_in_override_verification"].eq(True).any()
                    and sessions["followup_session"].eq(True).any()
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(paths, overall, task, units, reference_trace, claims, sessions, base_dir)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
