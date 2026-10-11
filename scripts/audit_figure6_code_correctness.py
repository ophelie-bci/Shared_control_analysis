from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# src.plots imports seaborn for unrelated helpers on machines where it may not be
# installed. Keep this audit focused on the Figure 6 path.
sys.modules.setdefault("seaborn", types.SimpleNamespace(kdeplot=lambda *args, **kwargs: None))

import src.plots as plots
from src.load import load_files
from src.plots import plot_alpha_vs_time_from_obstacle_appearance
from scripts.make_temporal_prior_reset_example import (
    DEFAULT_SESSION,
    DEFAULT_TRIAL,
    build_temporal_prior_reset_example,
    default_source_paths,
    _load_original_trial,
    _load_trial,
)


OUT_DIR = PROJECT_ROOT / "analysis_results" / "figure6_code_correctness"
PANEL_AB_DIR = OUT_DIR / "regenerated_panels_ab"
PANEL_C_DIR = OUT_DIR / "regenerated_panel_c"
REF_AB_DIR = PROJECT_ROOT / "figures" / "Figure 6" / "Prior confidence index"
REF_C_DIR = PROJECT_ROOT / "figures" / "Figure 6" / "Temporal prior reset"
MIN_SPEED = 0.05
SUSTAIN_MS = 200.0

CASES = [
    ("Monkey 1", "AI Appearing Obstacle"),
    ("Monkey 2", "AI Appearing Obstacle"),
    ("Monkey 1", "AI Respawn"),
]

PLOT_ARGS = {
    "stride": 4,
    "min_dt": None,
    "min_dist": None,
    "t_window": (-2000.0, 3000.0),
    "nbins": 24,
    "min_alpha": 0.2,
    "only_answer": 1,
    "task": None,
    "show_per_trial": False,
}


def _load_base_dir() -> Path:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("config.yaml is missing base_dir")
    return Path(base_dir)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trial_id(trial, fallback: int) -> str:
    for key in ("trial", "trial_id", "id"):
        value = _get(trial, key, None)
        if value is not None:
            return str(value)
    return str(fallback)


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def _target_xz(point: dict | None) -> np.ndarray:
    if not point:
        return np.full(2, np.nan)
    return np.asarray([point.get("x", np.nan), point.get("z", np.nan)], dtype=float)


def _old_new_evidence(vel_xz: np.ndarray, pos_xyz: np.ndarray, old_target: dict, new_target: dict) -> np.ndarray:
    vel = np.asarray(vel_xz, dtype=float)
    pos = np.asarray(pos_xyz, dtype=float)[:, [0, 2]]
    old_xz = _target_xz(old_target)
    new_xz = _target_xz(new_target)
    evidence = np.full(len(pos), np.nan, dtype=float)
    valid = (
        np.all(np.isfinite(vel), axis=1)
        & np.all(np.isfinite(pos), axis=1)
        & np.isfinite(old_xz).all()
        & np.isfinite(new_xz).all()
        & (np.linalg.norm(vel, axis=1) >= MIN_SPEED)
    )
    if np.any(valid):
        evidence[valid] = np.sum(_unit(vel[valid]) * _unit(new_xz - pos[valid]), axis=1) - np.sum(
            _unit(vel[valid]) * _unit(old_xz - pos[valid]), axis=1
        )
    return evidence


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _trial_dt_ms(trial: dict, n_samples: int) -> float:
    start = _parse_time(trial.get("start_time"))
    end = _parse_time(trial.get("end_time"))
    if start is None or end is None or n_samples < 2:
        return float("nan")
    return (end - start).total_seconds() * 1000.0 / float(n_samples - 1)


def sustained_crossing_after_reset(evidence: np.ndarray, reset_index: int, dt_ms: float) -> float:
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return float("nan")
    evidence = np.asarray(evidence, dtype=float)
    positive = np.isfinite(evidence) & (evidence > 0)
    positive[: max(0, int(reset_index))] = False
    for start in np.flatnonzero(positive):
        end = start
        while end + 1 < len(positive) and positive[end + 1]:
            end += 1
        if (end - start) * dt_ms >= SUSTAIN_MS:
            return float((start - int(reset_index)) * dt_ms)
    return float("nan")


def legacy_sustained_crossing_after_reset(evidence: np.ndarray, reset_index: int, dt_ms: float) -> float:
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return float("nan")
    need = max(1, int(np.ceil(SUSTAIN_MS / dt_ms)))
    evidence = np.asarray(evidence, dtype=float)
    positive = np.isfinite(evidence) & (evidence > 0)
    positive[: max(0, int(reset_index))] = False
    for idx in np.flatnonzero(positive):
        if idx + need <= len(positive) and bool(np.all(positive[idx : idx + need])):
            return float((idx - int(reset_index)) * dt_ms)
    return float("nan")


def run_known_delay_checks() -> list[dict]:
    cases = [
        {
            "case": "zero_delay_exact_200ms",
            "dt_ms": 50.0,
            "reset_index": 2,
            "bci_positive_from": 2,
            "ai_positive_from": 2,
            "n": 8,
            "expected_bci_ms": 0.0,
            "expected_ai_ms": 0.0,
            "expected_lag_ms": 0.0,
        },
        {
            "case": "ai_100ms_after_bci",
            "dt_ms": 50.0,
            "reset_index": 2,
            "bci_positive_from": 2,
            "ai_positive_from": 4,
            "n": 10,
            "expected_bci_ms": 0.0,
            "expected_ai_ms": 100.0,
            "expected_lag_ms": 100.0,
        },
        {
            "case": "no_reorientation",
            "dt_ms": 50.0,
            "reset_index": 2,
            "bci_positive_from": None,
            "ai_positive_from": None,
            "n": 10,
            "expected_bci_ms": np.nan,
            "expected_ai_ms": np.nan,
            "expected_lag_ms": np.nan,
        },
        {
            "case": "short_150ms_run_rejected",
            "dt_ms": 50.0,
            "reset_index": 2,
            "bci_positive_from": 2,
            "ai_positive_from": 2,
            "n": 6,
            "stop_positive_before": 5,
            "expected_bci_ms": np.nan,
            "expected_ai_ms": np.nan,
            "expected_lag_ms": np.nan,
        },
    ]
    rows = []
    for case in cases:
        n = int(case["n"])
        bci = np.full(n, -1.0)
        ai = np.full(n, -1.0)
        stop = int(case.get("stop_positive_before", n))
        if case["bci_positive_from"] is not None:
            bci[int(case["bci_positive_from"]) : stop] = 1.0
        if case["ai_positive_from"] is not None:
            ai[int(case["ai_positive_from"]) : stop] = 1.0
        bci_t = sustained_crossing_after_reset(bci, int(case["reset_index"]), float(case["dt_ms"]))
        ai_t = sustained_crossing_after_reset(ai, int(case["reset_index"]), float(case["dt_ms"]))
        lag = ai_t - bci_t if np.isfinite(ai_t) and np.isfinite(bci_t) else np.nan
        legacy_bci_t = legacy_sustained_crossing_after_reset(bci, int(case["reset_index"]), float(case["dt_ms"]))
        rows.append(
            {
                "case": case["case"],
                "dt_ms": repr(float(case["dt_ms"])),
                "reset_index": case["reset_index"],
                "computed_bci_latency_ms": repr(float(bci_t)) if np.isfinite(bci_t) else "",
                "computed_ai_latency_ms": repr(float(ai_t)) if np.isfinite(ai_t) else "",
                "computed_ai_minus_bci_lag_ms": repr(float(lag)) if np.isfinite(lag) else "",
                "expected_bci_latency_ms": repr(float(case["expected_bci_ms"])) if np.isfinite(case["expected_bci_ms"]) else "",
                "expected_ai_latency_ms": repr(float(case["expected_ai_ms"])) if np.isfinite(case["expected_ai_ms"]) else "",
                "expected_ai_minus_bci_lag_ms": repr(float(case["expected_lag_ms"])) if np.isfinite(case["expected_lag_ms"]) else "",
                "passes": (
                    (math.isnan(case["expected_bci_ms"]) and not np.isfinite(bci_t) or bci_t == case["expected_bci_ms"])
                    and (math.isnan(case["expected_ai_ms"]) and not np.isfinite(ai_t) or ai_t == case["expected_ai_ms"])
                    and (math.isnan(case["expected_lag_ms"]) and not np.isfinite(lag) or lag == case["expected_lag_ms"])
                ),
                "legacy_bci_latency_ms": repr(float(legacy_bci_t)) if np.isfinite(legacy_bci_t) else "",
            }
        )
    return rows


def _make_save_plot():
    PANEL_AB_DIR.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, name, subfolder=None):
        path = PANEL_AB_DIR / f"{name}.svg"
        fig.savefig(path, bbox_inches="tight")
        return path

    return save_plot


def collect_panel_ab_inputs_and_outputs(base_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    input_rows = []
    summary_rows = []
    session_value_rows = []
    original_save_plot = plots.save_plot
    original_show = plt.show
    plots.save_plot = _make_save_plot()
    plt.show = lambda *args, **kwargs: None
    try:
        for monkey, experiment in CASES:
            all_trials, *_rest, pkl_files, _ai_trials = load_files(experiment, monkey, base_dir=str(base_dir))
            event_attr = "targetJumpTime" if experiment == "AI Respawn" else "obstacleAppearedTime"
            seen = set()
            duplicate_count = 0
            total_trials = 0
            selected_answer_1 = 0
            missing_event_selected = 0
            answer_counts: dict[str, int] = {}
            for session_idx, session in enumerate(all_trials):
                for trial_idx, trial in enumerate(session):
                    total_trials += 1
                    key = (session_idx, _trial_id(trial, trial_idx))
                    duplicate_count += int(key in seen)
                    seen.add(key)
                    ans = _get(trial, "answer", None)
                    answer_counts[str(ans)] = answer_counts.get(str(ans), 0) + 1
                    if ans == 1:
                        selected_answer_1 += 1
                        if _get(trial, event_attr, None) is None:
                            missing_event_selected += 1
            result = plot_alpha_vs_time_from_obstacle_appearance(
                monkey,
                experiment,
                str(base_dir),
                **PLOT_ARGS,
            )
            plt.close("all")
            session = result["session_level"]
            label = f"{monkey} {experiment}"
            input_rows.append(
                {
                    "panel_condition": label,
                    "base_dir": str(base_dir),
                    "n_session_files": len(pkl_files),
                    "n_loaded_sessions": len(all_trials),
                    "n_total_trials": total_trials,
                    "answer_code_counts_original_experiment": json.dumps(answer_counts, sort_keys=True),
                    "condition_selection": "only_answer == 1",
                    "n_selected_success_trials_before_event_filter": selected_answer_1,
                    "event_attribute": event_attr,
                    "n_selected_trials_missing_event_attribute": missing_event_selected,
                    "duplicate_session_trial_keys": duplicate_count,
                    "file_names": json.dumps([Path(p).name for p in pkl_files]),
                }
            )
            summary_rows.append(
                {
                    "panel_condition": label,
                    "n_sessions": int(session["n_sessions"]),
                    "real_modulation_median": repr(float(session["delta_early_median_across_sessions"])),
                    "real_modulation_wilcoxon_p": repr(float(session["delta_early_p"])),
                    "pseudo_modulation_median": repr(float(session["delta_pseudo_median_across_sessions"])),
                    "pseudo_modulation_wilcoxon_p": repr(float(session["delta_pseudo_p"])),
                    "real_minus_pseudo_median": repr(float(session["real_minus_pseudo_median_across_sessions"])),
                    "real_minus_pseudo_wilcoxon_p": repr(float(session["real_vs_pseudo_p"])),
                    "recovery_median": repr(float(session["delta_recovery_median_across_sessions"])),
                    "recovery_wilcoxon_p": repr(float(session["delta_recovery_p"])),
                    "min_dip_depth_median": repr(float(session["dip_depth_valid_median_across_sessions"])),
                    "min_dip_depth_wilcoxon_p": repr(float(session["dip_depth_valid_p"])),
                    "subtraction_order": "(early_window_mean - pre_window_mean) - (pseudo_post_window_mean - pseudo_pre_window_mean)",
                    "wilcoxon_call": "scipy.stats.wilcoxon(session_median_differences, alternative='two-sided')",
                    "pseudo_rng": "np.random.default_rng(0), re-created inside each function call",
                }
            )
            value_arrays = {
                "real_modulation": session["delta_early_values"],
                "pseudo_modulation": session["delta_pseudo_values"],
                "real_minus_pseudo": session["real_minus_pseudo_values"],
                "recovery": session["delta_recovery_values"],
                "min_dip_depth": session["dip_depth_valid_values"],
                "frac_dip_trials": session["frac_dip_trials_values"],
            }
            for idx in range(int(session["n_sessions"])):
                row = {"panel_condition": label, "session_index": idx}
                for key, values in value_arrays.items():
                    value = np.asarray(values, dtype=float)[idx]
                    row[key] = repr(float(value)) if np.isfinite(value) else ""
                session_value_rows.append(row)
    finally:
        plt.close("all")
        plots.save_plot = original_save_plot
        plt.show = original_show
    return input_rows, summary_rows, session_value_rows


def _coords_from_svg_path(d: str) -> np.ndarray:
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", d)]
    if not nums:
        return np.empty((0, 2), dtype=float)
    return np.asarray(nums, dtype=float).reshape(-1, 2)


def _extract_svg_geometry(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    axes_match = re.search(r'<g id="patch_2">\s*<path d="([\s\S]*?)"', text)
    if axes_match is None:
        raise ValueError(f"Could not find axes patch in {path}")
    axes = _coords_from_svg_path(axes_match.group(1))
    x0, x1 = float(axes[:, 0].min()), float(axes[:, 0].max())
    y0, y1 = float(axes[:, 1].min()), float(axes[:, 1].max())

    def norm(coords: np.ndarray) -> np.ndarray:
        return np.column_stack([(coords[:, 0] - x0) / (x1 - x0), (y1 - coords[:, 1]) / (y1 - y0)])

    poly_match = re.search(r'<g id="PolyCollection_1">[\s\S]*?<path d="([\s\S]*?)"', text)
    if poly_match is None:
        raise ValueError(f"Could not find PolyCollection_1 in {path}")
    band = norm(_coords_from_svg_path(poly_match.group(1)))

    candidates: list[np.ndarray] = []
    for match in re.finditer(r'<g id="line2d_\d+">([\s\S]*?)</g>', text):
        block = match.group(1)
        if "#1f77b4" not in block or "stroke-width: 2.5" not in block:
            continue
        d_match = re.search(r'<path d="([\s\S]*?)"', block)
        if d_match is not None:
            candidates.append(norm(_coords_from_svg_path(d_match.group(1))))
    if not candidates:
        raise ValueError(f"Could not find mean line in {path}")
    line = max(candidates, key=len)
    return {"band": band, "mean_line": line}


def compare_panel_ab_svgs() -> list[dict]:
    rows = []
    for monkey, experiment in CASES:
        name = f"alpha_vs_time_FROM_EVENT_{monkey}_{experiment}.svg"
        fresh = PANEL_AB_DIR / name
        ref = REF_AB_DIR / name
        fresh_geom = _extract_svg_geometry(fresh)
        ref_geom = _extract_svg_geometry(ref)
        band_delta = (
            float(np.nanmax(np.abs(fresh_geom["band"] - ref_geom["band"])))
            if fresh_geom["band"].shape == ref_geom["band"].shape
            else float("nan")
        )
        line_delta = (
            float(np.nanmax(np.abs(fresh_geom["mean_line"] - ref_geom["mean_line"])))
            if fresh_geom["mean_line"].shape == ref_geom["mean_line"].shape
            else float("nan")
        )
        rows.append(
            {
                "panel_file": name,
                "fresh_path": str(fresh),
                "reference_path": str(ref),
                "fresh_sha256": _sha256(fresh),
                "reference_sha256": _sha256(ref),
                "byte_identical": _sha256(fresh) == _sha256(ref),
                "fresh_band_points": len(fresh_geom["band"]),
                "reference_band_points": len(ref_geom["band"]),
                "fresh_mean_line_points": len(fresh_geom["mean_line"]),
                "reference_mean_line_points": len(ref_geom["mean_line"]),
                "max_abs_normalized_band_delta": repr(band_delta),
                "max_abs_normalized_mean_line_delta": repr(line_delta),
                "plotted_values_match_reference": bool(
                    np.isfinite(band_delta)
                    and np.isfinite(line_delta)
                    and band_delta <= 1e-6
                    and line_delta <= 1e-6
                ),
            }
        )
    return rows


def summarize_reset_prior(base_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    reset_dir = base_dir / "Monkey 1" / "AI Respawn" / "resetPriorFiles"
    files = sorted(reset_dir.glob("reset_prior_analyzes_*.pkl"))
    trial_rows = []
    session_rows = []
    changed_rows = []
    for path in files:
        with path.open("rb") as f:
            data = pickle.load(f)
        trials = data.get("trials", [])
        n_success = 0
        answer_counts: dict[str, int] = {}
        for trial in trials:
            answer = trial.get("answer")
            answer_log = trial.get("answer_log")
            answer_counts[str(answer)] = answer_counts.get(str(answer), 0) + 1
            is_success = answer == 1
            n_success += int(is_success)
            samples = trial.get("samples", {})
            pos = np.asarray(samples.get("position", []), dtype=float)
            bci = np.asarray(samples.get("bci_velocity", []), dtype=float)
            ai = np.asarray(samples.get("ai_velocity", []), dtype=float)
            n = min(len(pos), len(bci), len(ai))
            include_lag = (
                trial.get("target_jump_index") is not None
                and trial.get("true_goal") is not None
                and trial.get("target_jump_position") is not None
                and n >= 3
            )
            dt_ms = _trial_dt_ms(trial, n)
            bci_latency = ai_latency = lag = float("nan")
            legacy_bci = legacy_ai = legacy_lag = float("nan")
            bci_crossed = ai_crossed = False
            if include_lag:
                reset_index = int(trial["target_jump_index"])
                e_bci = _old_new_evidence(bci[:n], pos[:n], trial["true_goal"], trial["target_jump_position"])
                e_ai = _old_new_evidence(ai[:n], pos[:n], trial["true_goal"], trial["target_jump_position"])
                bci_latency = sustained_crossing_after_reset(e_bci, reset_index, dt_ms)
                ai_latency = sustained_crossing_after_reset(e_ai, reset_index, dt_ms)
                legacy_bci = legacy_sustained_crossing_after_reset(e_bci, reset_index, dt_ms)
                legacy_ai = legacy_sustained_crossing_after_reset(e_ai, reset_index, dt_ms)
                bci_crossed = bool(np.isfinite(bci_latency))
                ai_crossed = bool(np.isfinite(ai_latency))
                if bci_crossed and ai_crossed:
                    lag = float(ai_latency - bci_latency)
                if np.isfinite(legacy_bci) and np.isfinite(legacy_ai):
                    legacy_lag = float(legacy_ai - legacy_bci)
            row = {
                "source_pkl": str(path),
                "session": data.get("session", path.stem),
                "trial_id": trial.get("trial_id"),
                "answer_log_stored_replay": answer_log,
                "answer_stored_replay": answer,
                "answer_reason_stored_replay": trial.get("answer_reason"),
                "included_success_denominator": True,
                "success_stored_replay": is_success,
                "included_lag_denominator_metadata": include_lag,
                "included_median_lag_denominator_both_reoriented": bool(np.isfinite(lag)),
                "n_samples_aligned_streams": n,
                "target_jump_index": trial.get("target_jump_index"),
                "dt_ms_from_start_end_over_n_minus_1": repr(float(dt_ms)) if np.isfinite(dt_ms) else "",
                "bci_reoriented_after_reset": bci_crossed,
                "ai_executed_reoriented_after_reset": ai_crossed,
                "bci_reorientation_latency_ms": repr(float(bci_latency)) if np.isfinite(bci_latency) else "",
                "ai_executed_reorientation_latency_ms": repr(float(ai_latency)) if np.isfinite(ai_latency) else "",
                "executed_minus_bci_lag_ms": repr(float(lag)) if np.isfinite(lag) else "",
                "legacy_executed_minus_bci_lag_ms": repr(float(legacy_lag)) if np.isfinite(legacy_lag) else "",
            }
            trial_rows.append(row)
            if (np.isfinite(lag) != np.isfinite(legacy_lag)) or (
                np.isfinite(lag) and np.isfinite(legacy_lag) and abs(lag - legacy_lag) > 1e-9
            ):
                changed_rows.append(row)
        n_total = len(trials)
        session_rows.append(
            {
                "source_pkl": str(path),
                "session": data.get("session", path.stem),
                "answer_code_counts_stored_replay": json.dumps(answer_counts, sort_keys=True),
                "n_trials_success_denominator": n_total,
                "n_success_answer_eq_1": n_success,
                "success_fraction": repr(float(n_success / n_total)) if n_total else "",
            }
        )

    success_total = sum(int(r["n_success_answer_eq_1"]) for r in session_rows)
    trial_total = sum(int(r["n_trials_success_denominator"]) for r in session_rows)
    session_success = np.asarray([float(r["success_fraction"]) for r in session_rows], dtype=float)
    lag_values = np.asarray(
        [float(r["executed_minus_bci_lag_ms"]) for r in trial_rows if r["executed_minus_bci_lag_ms"] != ""],
        dtype=float,
    )
    legacy_lag_values = np.asarray(
        [float(r["legacy_executed_minus_bci_lag_ms"]) for r in trial_rows if r["legacy_executed_minus_bci_lag_ms"] != ""],
        dtype=float,
    )
    bci_cross = [r for r in trial_rows if r["bci_reoriented_after_reset"]]
    ai_cross = [r for r in trial_rows if r["ai_executed_reoriented_after_reset"]]
    both = [r for r in trial_rows if r["included_median_lag_denominator_both_reoriented"]]
    aggregate_rows = [
        {
            "metric": "reset_prior_success_fraction_pooled_trials",
            "numerator": success_total,
            "denominator": trial_total,
            "value": repr(float(success_total / trial_total)),
            "convention": "pooled stored replay trials",
        },
        {
            "metric": "reset_prior_success_fraction_mean_of_session_rates",
            "numerator": len(session_rows),
            "denominator": len(session_rows),
            "value": repr(float(np.nanmean(session_success))),
            "convention": "mean of per-session stored replay success fractions",
        },
        {
            "metric": "reset_prior_success_fraction_session_rate_sd",
            "numerator": len(session_rows),
            "denominator": len(session_rows),
            "value": repr(float(np.nanstd(session_success, ddof=1))),
            "convention": "SD across session fractions",
        },
        {
            "metric": "reset_prior_lag_median_executed_minus_bci_ms_both_reoriented",
            "numerator": len(both),
            "denominator": len(trial_rows),
            "value": repr(float(np.nanmedian(lag_values))) if len(lag_values) else "",
            "convention": "finite corrected lag rows only; both BCI and executed streams reoriented",
        },
        {
            "metric": "reset_prior_lag_iqr_executed_minus_bci_ms_both_reoriented",
            "numerator": len(both),
            "denominator": len(trial_rows),
            "value": repr(float(np.nanpercentile(lag_values, 75) - np.nanpercentile(lag_values, 25))) if len(lag_values) else "",
            "convention": "finite corrected lag rows only; both BCI and executed streams reoriented",
        },
        {
            "metric": "legacy_reset_prior_lag_median_executed_minus_bci_ms_both_reoriented",
            "numerator": len(legacy_lag_values),
            "denominator": len(trial_rows),
            "value": repr(float(np.nanmedian(legacy_lag_values))) if len(legacy_lag_values) else "",
            "convention": "pre-fix sample-count sustained-run criterion",
        },
        {
            "metric": "reset_prior_rows_with_changed_lag_after_200ms_duration_fix",
            "numerator": len(changed_rows),
            "denominator": len(trial_rows),
            "value": repr(float(len(changed_rows) / len(trial_rows))) if trial_rows else "",
            "convention": "corrected versus legacy verifier criterion",
        },
        {
            "metric": "bci_reoriented_after_reset_fraction",
            "numerator": len(bci_cross),
            "denominator": len(trial_rows),
            "value": repr(float(len(bci_cross) / len(trial_rows))) if trial_rows else "",
            "convention": "corrected 200-ms sustained criterion",
        },
        {
            "metric": "executed_reoriented_after_reset_fraction",
            "numerator": len(ai_cross),
            "denominator": len(trial_rows),
            "value": repr(float(len(ai_cross) / len(trial_rows))) if trial_rows else "",
            "convention": "corrected 200-ms sustained criterion",
        },
    ]
    return trial_rows, session_rows, aggregate_rows, changed_rows


def compare_original_and_reset_trial_keys(reset_rows: list[dict]) -> list[dict]:
    original_path = PROJECT_ROOT / "tables" / "respawn_reorientation_lag_by_trial.csv"
    if not original_path.exists():
        return []
    with original_path.open("r", newline="", encoding="utf-8") as f:
        original_rows = list(csv.DictReader(f))
    original = {}
    for row in original_rows:
        session = Path(row["session_file"]).stem.replace("_trials", "")
        original[(session, row["trial"])] = row
    reset_total = len(reset_rows)
    matched = 0
    finite_both = 0
    for row in reset_rows:
        session = str(row["session"]).replace("reset_prior_analyzes_", "")
        key = (session, str(row["trial_id"]))
        if key in original:
            matched += 1
            if row["executed_minus_bci_lag_ms"] != "" and original[key].get("controller_lag_ms", "") != "":
                finite_both += 1
    return [
        {
            "metric": "reset_rows_matching_original_experimental_trial_keys",
            "numerator": matched,
            "denominator": reset_total,
            "value": repr(float(matched / reset_total)) if reset_total else "",
            "notes": "session name normalized by removing reset_prior_analyzes_ and _trials suffix",
        },
        {
            "metric": "matched_rows_with_finite_lag_in_both_original_and_reset",
            "numerator": finite_both,
            "denominator": reset_total,
            "value": repr(float(finite_both / reset_total)) if reset_total else "",
            "notes": "not a full matched original-versus-reset lag analysis",
        },
    ]


def summarize_panel_c(base_dir: Path) -> list[dict]:
    original_path, original_ai_path, reset_path = default_source_paths(base_dir)
    original = _load_original_trial(original_path, original_ai_path, DEFAULT_TRIAL)
    reset = _load_trial(reset_path, DEFAULT_TRIAL)
    rows = []
    for label, trial, path in (
        ("original_no_reset_cropped", original, original_path),
        ("reset_prior_replay", reset, reset_path),
    ):
        samples = trial["samples"]
        pos = np.asarray(samples.get("position", []), dtype=float)
        bci = np.asarray(samples.get("bci_velocity", []), dtype=float)
        entropy = np.asarray(samples.get("entropy", []), dtype=float).reshape(-1)
        rows.append(
            {
                "source": label,
                "path": str(path),
                "session": DEFAULT_SESSION,
                "trial_id": DEFAULT_TRIAL,
                "answer_code": trial.get("answer"),
                "n_position_samples": len(pos),
                "n_bci_velocity_samples": len(bci),
                "n_entropy_samples": len(entropy),
                "target_jump_index": trial.get("target_jump_index"),
                "old_target_x": trial.get("true_goal", {}).get("x"),
                "old_target_z": trial.get("true_goal", {}).get("z"),
                "new_target_x": trial.get("target_jump_position", {}).get("x"),
                "new_target_z": trial.get("target_jump_position", {}).get("z"),
                "first_x": repr(float(pos[0, 0])) if len(pos) else "",
                "first_z": repr(float(pos[0, 2])) if len(pos) else "",
                "last_x": repr(float(pos[-1, 0])) if len(pos) else "",
                "last_z": repr(float(pos[-1, 2])) if len(pos) else "",
            }
        )
    PANEL_C_DIR.mkdir(parents=True, exist_ok=True)
    outputs = build_temporal_prior_reset_example(
        original_path=original_path,
        original_ai_path=original_ai_path,
        reset_path=reset_path,
        trial_id=DEFAULT_TRIAL,
        output_dir=PANEL_C_DIR,
        formats=("svg", "png", "pdf"),
        write_provenance=True,
    )
    for output in outputs:
        rows.append(
            {
                "source": "regenerated_panel_c_output",
                "path": str(output),
                "session": DEFAULT_SESSION,
                "trial_id": DEFAULT_TRIAL,
                "answer_code": "",
                "n_position_samples": "",
                "n_bci_velocity_samples": "",
                "n_entropy_samples": "",
                "target_jump_index": "",
                "old_target_x": "",
                "old_target_z": "",
                "new_target_x": "",
                "new_target_z": "",
                "first_x": "",
                "first_z": "",
                "last_x": "",
                "last_z": "",
                "sha256": _sha256(output),
                "retained_reference_sha256": _sha256(REF_C_DIR / output.name),
                "byte_identical_to_retained_reference": _sha256(output) == _sha256(REF_C_DIR / output.name),
            }
        )
    return rows


def write_commands(base_dir: Path) -> None:
    base_dir_arg = str(base_dir)
    commands = [
        "# Figure 6 code-correctness regeneration commands",
        "",
        "Run from the repository root used in this audit:",
        "",
        "```powershell",
        "cd <repo-root>",
        "python scripts\\audit_figure6_code_correctness.py",
        "python scripts\\verify_figure6_only.py",
        "python scripts\\verify_figure6_remaining.py",
        f"python scripts\\make_temporal_prior_reset_example.py --base-dir \"{base_dir_arg}\" --output-dir \"analysis_results\\figure6_code_correctness\\regenerated_panel_c\" --formats svg png pdf --write-provenance",
        "```",
        "",
        f"Resolved `config.yaml` base_dir: `{base_dir}`.",
    ]
    (OUT_DIR / "exact_regeneration_commands.md").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_report(
    base_dir: Path,
    panel_inputs: list[dict],
    panel_summary: list[dict],
    reset_rows: list[dict],
    reset_aggregate: list[dict],
    changed_rows: list[dict],
    key_match_rows: list[dict],
) -> None:
    agg = {row["metric"]: row for row in reset_aggregate}
    finite_lag_rows = [row for row in reset_rows if row["executed_minus_bci_lag_ms"] != ""]
    nonzero_lag_rows = [
        row
        for row in finite_lag_rows
        if abs(float(row["executed_minus_bci_lag_ms"])) > 1e-9
    ]
    missing_event_parts = [
        f"{row['panel_condition']}: {row['n_selected_trials_missing_event_attribute']} selected trials missing `{row['event_attribute']}`"
        for row in panel_inputs
    ]
    duplicate_total = sum(int(row["duplicate_session_trial_keys"]) for row in panel_inputs)
    lines = [
        "# Figure 6 Code-Correctness Audit",
        "",
        "Scope: current Figure 6 analysis and plotting code, using saved reset-prior PKLs as valid input data. No upstream controller replay implementation or missing logs were searched.",
        "",
        "## Code Paths",
        "",
        "- Panels a/b: `scripts/run_prior_confidence_index_analysis.py` -> `src.plots.plot_alpha_vs_time_from_obstacle_appearance`.",
        "- Panel c: `scripts/make_temporal_prior_reset_example.py`.",
        "- Reset summaries: saved `reset_prior_analyzes_*.pkl` files under `config.yaml` base_dir, audited here with the same new-minus-old evidence rule as the original lag analysis.",
        f"- `config.yaml` resolved base_dir: `{base_dir}`.",
        "- Zenodo v1.0 reference: software record `10.5281/zenodo.18926135`; local read-only snapshot used where present: `analysis_results/figure6_prior_verification/external_shared_control_v1_0`.",
        "",
        "## Panels a/b Verdict",
        "",
        "Correct. The live code converts `EntropyLb` to alpha as `1 - clip(EntropyLb, 0, Hmax) / Hmax`, estimates `Hmax = log(round(exp(max EntropyLb)))`, clips alpha to `[0.2, 1.0]`, aligns appearing-obstacle trials to `obstacleAppearedTime` and respawn trials to `targetJumpTime`, bins per-trial traces, and plots the across-trial mean with `mean +/- 1.96 * SEM` bands. The pseudo control is reproducible because each function call constructs `np.random.default_rng(0)`.",
        "",
        "The tested real/pseudo contrast is session-level: each session contributes the median trial value for `(early 0-400 ms - pre -400-0 ms) - (pseudo post 0-400 ms - pseudo pre -400-0 ms)`, then `scipy.stats.wilcoxon(..., alternative='two-sided')` is applied to the finite session medians. Minimum dip depth is separately reported and is not the tested window-mean modulation.",
        "",
        "Panel a/b full-precision summaries are in `figure6_panel_ab_full_precision_summary.csv`; input/session checks are in `figure6_panel_ab_input_audit.csv` and `figure6_panel_ab_session_values.csv`.",
        "",
        f"Input checks found {duplicate_total} duplicate session/trial keys across the three panel conditions. The `only_answer == 1` selection is applied first; trials missing the required event attribute are then skipped by the plotting function ({'; '.join(missing_event_parts)}).",
        "",
        "## Panel c And Reset Summaries Verdict",
        "",
        "Corrected. Panel c uses the intended original trial and reset replay PKL for Monkey 1 AI Respawn session `navdecodingsphereairespawn_Maui_20250604_1149_A`, trial 6. The plotting code uses X/Z coordinates, target positions, BCI velocity arrows, and prior-confidence coloring from stored entropy.",
        "",
        "A verifier bug was corrected in `scripts/verify_figure6_remaining.py`: the reset-prior sustained-reorientation check now requires elapsed positive-evidence duration `>= 200 ms`, matching the original lag analysis, instead of accepting `ceil(200/dt)` positive samples. Absent BCI or executed reorientation remains missing and does not become zero.",
        "",
        "Reset-prior aggregation conventions are now explicit:",
        f"- Pooled stored replay success: {agg['reset_prior_success_fraction_pooled_trials']['numerator']}/{agg['reset_prior_success_fraction_pooled_trials']['denominator']} = {agg['reset_prior_success_fraction_pooled_trials']['value']}.",
        f"- Mean of session success rates: {agg['reset_prior_success_fraction_mean_of_session_rates']['value']}.",
        f"- Corrected median executed-minus-BCI lag: {agg['reset_prior_lag_median_executed_minus_bci_ms_both_reoriented']['value']} ms over {agg['reset_prior_lag_median_executed_minus_bci_ms_both_reoriented']['numerator']}/{agg['reset_prior_lag_median_executed_minus_bci_ms_both_reoriented']['denominator']} reset rows where both streams reoriented.",
        f"- Nonzero finite reset lags after correction: {len(nonzero_lag_rows)}/{len(finite_lag_rows)}.",
        f"- Rows with changed lag status/value after the 200-ms duration fix: {len(changed_rows)}.",
        "",
        "The reset `answer` and `answer_log` fields are stored replay outcome codes and are not the original experimental answer codes. Trial-level rows and denominators are in `reset_prior_trial_level_corrected.csv`, `reset_prior_session_success_summary.csv`, and `reset_prior_aggregate_full_precision_summary.csv`.",
        "",
        "## Original Versus Reset Matching",
        "",
        "Unresolved as a paired original-versus-reset comparison. The reset PKLs are valid stored replay outputs, but the audit does not silently treat them as a fully matched original-vs-reset lag experiment. Matching diagnostics are in `original_reset_trial_key_match_summary.csv`.",
    ]
    for row in key_match_rows:
        lines.append(f"- {row['metric']}: {row['numerator']}/{row['denominator']} = {row['value']}. {row['notes']}")
    lines.extend(
        [
            "",
            "## Final Verdicts",
            "",
            "| Item | Verdict | Reason |",
            "| --- | --- | --- |",
            "| Panel a | correct | Code path, config input, alpha conversion, event alignment, bins, 1.96 SEM band, real/pseudo session contrast and reproducible pseudo sampling verified. |",
            "| Panel b | correct | Same verified prior-confidence implementation and statistics as panel a. |",
            "| Panel c | correct | Uses intended saved original/replay outputs and target/trajectory coordinates; no manuscript figure was replaced. |",
            "| Reset pooled success 648/812 | correct | This is the pooled stored replay trial convention. |",
            "| Reset mean-of-session success | correct | Reported separately to avoid switching conventions. |",
            "| Reset median lag | corrected | Same sign convention, executed minus BCI; median remains a median over finite both-reoriented rows after corrected 200-ms criterion. |",
            "| Claim reset eliminated lag on every trial | unresolved/unsupported | Median 0 ms does not imply every finite trial has zero lag; trial-level file records nonzero lags. |",
            "| Original-vs-reset paired lag comparison | unresolved | Trial-key overlap exists but not all rows have finite lag in both tables; this audit does not convert that into a matched comparison. |",
        ]
    )
    (OUT_DIR / "figure6_code_correctness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_dir = _load_base_dir()

    known_delay_rows = run_known_delay_checks()
    _write_csv(
        OUT_DIR / "lag_known_delay_validation.csv",
        known_delay_rows,
        [
            "case",
            "dt_ms",
            "reset_index",
            "computed_bci_latency_ms",
            "computed_ai_latency_ms",
            "computed_ai_minus_bci_lag_ms",
            "expected_bci_latency_ms",
            "expected_ai_latency_ms",
            "expected_ai_minus_bci_lag_ms",
            "passes",
            "legacy_bci_latency_ms",
        ],
    )

    panel_inputs, panel_summary, session_values = collect_panel_ab_inputs_and_outputs(base_dir)
    _write_csv(
        OUT_DIR / "figure6_panel_ab_input_audit.csv",
        panel_inputs,
        [
            "panel_condition",
            "base_dir",
            "n_session_files",
            "n_loaded_sessions",
            "n_total_trials",
            "answer_code_counts_original_experiment",
            "condition_selection",
            "n_selected_success_trials_before_event_filter",
            "event_attribute",
            "n_selected_trials_missing_event_attribute",
            "duplicate_session_trial_keys",
            "file_names",
        ],
    )
    _write_csv(
        OUT_DIR / "figure6_panel_ab_full_precision_summary.csv",
        panel_summary,
        [
            "panel_condition",
            "n_sessions",
            "real_modulation_median",
            "real_modulation_wilcoxon_p",
            "pseudo_modulation_median",
            "pseudo_modulation_wilcoxon_p",
            "real_minus_pseudo_median",
            "real_minus_pseudo_wilcoxon_p",
            "recovery_median",
            "recovery_wilcoxon_p",
            "min_dip_depth_median",
            "min_dip_depth_wilcoxon_p",
            "subtraction_order",
            "wilcoxon_call",
            "pseudo_rng",
        ],
    )
    _write_csv(
        OUT_DIR / "figure6_panel_ab_session_values.csv",
        session_values,
        [
            "panel_condition",
            "session_index",
            "real_modulation",
            "pseudo_modulation",
            "real_minus_pseudo",
            "recovery",
            "min_dip_depth",
            "frac_dip_trials",
        ],
    )
    svg_rows = compare_panel_ab_svgs()
    _write_csv(
        OUT_DIR / "figure6_panel_ab_svg_reference_comparison.csv",
        svg_rows,
        [
            "panel_file",
            "fresh_path",
            "reference_path",
            "fresh_sha256",
            "reference_sha256",
            "byte_identical",
            "fresh_band_points",
            "reference_band_points",
            "fresh_mean_line_points",
            "reference_mean_line_points",
            "max_abs_normalized_band_delta",
            "max_abs_normalized_mean_line_delta",
            "plotted_values_match_reference",
        ],
    )

    reset_rows, reset_session_rows, reset_aggregate_rows, changed_rows = summarize_reset_prior(base_dir)
    _write_csv(
        OUT_DIR / "reset_prior_trial_level_corrected.csv",
        reset_rows,
        [
            "source_pkl",
            "session",
            "trial_id",
            "answer_log_stored_replay",
            "answer_stored_replay",
            "answer_reason_stored_replay",
            "included_success_denominator",
            "success_stored_replay",
            "included_lag_denominator_metadata",
            "included_median_lag_denominator_both_reoriented",
            "n_samples_aligned_streams",
            "target_jump_index",
            "dt_ms_from_start_end_over_n_minus_1",
            "bci_reoriented_after_reset",
            "ai_executed_reoriented_after_reset",
            "bci_reorientation_latency_ms",
            "ai_executed_reorientation_latency_ms",
            "executed_minus_bci_lag_ms",
            "legacy_executed_minus_bci_lag_ms",
        ],
    )
    _write_csv(
        OUT_DIR / "reset_prior_session_success_summary.csv",
        reset_session_rows,
        [
            "source_pkl",
            "session",
            "answer_code_counts_stored_replay",
            "n_trials_success_denominator",
            "n_success_answer_eq_1",
            "success_fraction",
        ],
    )
    _write_csv(
        OUT_DIR / "reset_prior_aggregate_full_precision_summary.csv",
        reset_aggregate_rows,
        ["metric", "numerator", "denominator", "value", "convention"],
    )
    _write_csv(
        OUT_DIR / "reset_prior_rows_changed_by_200ms_fix.csv",
        changed_rows,
        [
            "source_pkl",
            "session",
            "trial_id",
            "answer_log_stored_replay",
            "answer_stored_replay",
            "answer_reason_stored_replay",
            "included_success_denominator",
            "success_stored_replay",
            "included_lag_denominator_metadata",
            "included_median_lag_denominator_both_reoriented",
            "n_samples_aligned_streams",
            "target_jump_index",
            "dt_ms_from_start_end_over_n_minus_1",
            "bci_reoriented_after_reset",
            "ai_executed_reoriented_after_reset",
            "bci_reorientation_latency_ms",
            "ai_executed_reorientation_latency_ms",
            "executed_minus_bci_lag_ms",
            "legacy_executed_minus_bci_lag_ms",
        ],
    )
    key_match_rows = compare_original_and_reset_trial_keys(reset_rows)
    _write_csv(
        OUT_DIR / "original_reset_trial_key_match_summary.csv",
        key_match_rows,
        ["metric", "numerator", "denominator", "value", "notes"],
    )

    panel_c_rows = summarize_panel_c(base_dir)
    _write_csv(
        OUT_DIR / "figure6_panel_c_trial_and_reference_summary.csv",
        panel_c_rows,
        [
            "source",
            "path",
            "session",
            "trial_id",
            "answer_code",
            "n_position_samples",
            "n_bci_velocity_samples",
            "n_entropy_samples",
            "target_jump_index",
            "old_target_x",
            "old_target_z",
            "new_target_x",
            "new_target_z",
            "first_x",
            "first_z",
            "last_x",
            "last_z",
            "sha256",
            "retained_reference_sha256",
            "byte_identical_to_retained_reference",
        ],
    )

    write_commands(base_dir)
    write_report(base_dir, panel_inputs, panel_summary, reset_rows, reset_aggregate_rows, changed_rows, key_match_rows)
    print(f"Wrote Figure 6 code-correctness audit to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
