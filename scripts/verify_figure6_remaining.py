from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis_results" / "figure6_prior_verification" / "remaining_checks"
FRESH_AB = PROJECT_ROOT / "analysis_results" / "figure6_prior_verification" / "final_verification" / "fresh_panels_ab"
REF_AB = PROJECT_ROOT / "figures" / "Figure 6" / "Prior confidence index"
MIN_SPEED = 0.05
SUSTAIN_MS = 200.0

PANEL_FILES = [
    "alpha_vs_time_FROM_EVENT_Monkey 1_AI Appearing Obstacle.svg",
    "alpha_vs_time_FROM_EVENT_Monkey 2_AI Appearing Obstacle.svg",
    "alpha_vs_time_FROM_EVENT_Monkey 1_AI Respawn.svg",
]


def _load_base_dir() -> Path:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return Path(config["base_dir"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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

    candidates: list[tuple[str, np.ndarray]] = []
    for match in re.finditer(r'<g id="(line2d_\d+)">([\s\S]*?)</g>', text):
        gid, block = match.group(1), match.group(2)
        if "#1f77b4" not in block or "stroke-width: 2.5" not in block:
            continue
        d_match = re.search(r'<path d="([\s\S]*?)"', block)
        if d_match is not None:
            candidates.append((gid, norm(_coords_from_svg_path(d_match.group(1)))))
    if not candidates:
        raise ValueError(f"Could not find mean line in {path}")
    line_id, line = max(candidates, key=lambda item: len(item[1]))
    return {
        "axes_bbox": [x0, y0, x1, y1],
        "band": band,
        "mean_line": line,
        "mean_line_id": line_id,
    }


def _max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def compare_panels() -> None:
    rows = []
    point_rows = []
    preview_dir = OUT_DIR / "panel_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name in PANEL_FILES:
        ref_path = REF_AB / name
        fresh_path = FRESH_AB / name
        ref = _extract_svg_geometry(ref_path)
        fresh = _extract_svg_geometry(fresh_path)

        band_delta = _max_abs_delta(ref["band"], fresh["band"])
        line_delta = _max_abs_delta(ref["mean_line"], fresh["mean_line"])
        rows.append(
            {
                "panel_file": name,
                "reference_sha256": _sha256(ref_path),
                "fresh_sha256": _sha256(fresh_path),
                "byte_identical": _sha256(ref_path) == _sha256(fresh_path),
                "reference_axes_bbox_svg_units": json.dumps(ref["axes_bbox"]),
                "fresh_axes_bbox_svg_units": json.dumps(fresh["axes_bbox"]),
                "reference_band_points": len(ref["band"]),
                "fresh_band_points": len(fresh["band"]),
                "reference_mean_line_id": ref["mean_line_id"],
                "fresh_mean_line_id": fresh["mean_line_id"],
                "reference_mean_line_points": len(ref["mean_line"]),
                "fresh_mean_line_points": len(fresh["mean_line"]),
                "max_abs_normalized_band_delta": repr(band_delta),
                "max_abs_normalized_mean_line_delta": repr(line_delta),
                "scientific_difference_detected": bool(band_delta > 1e-6 or line_delta > 1e-6),
                "interpretation": "geometry matches; byte differences are layout/metadata/id serialization"
                if band_delta <= 1e-6 and line_delta <= 1e-6
                else "geometry differs",
            }
        )
        for source, geom in (("reference", ref), ("fresh", fresh)):
            for kind in ("band", "mean_line"):
                for idx, (x, y) in enumerate(geom[kind]):
                    point_rows.append(
                        {
                            "panel_file": name,
                            "source": source,
                            "geometry": kind,
                            "point_index": idx,
                            "normalized_x": repr(float(x)),
                            "normalized_y": repr(float(y)),
                        }
                    )

        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        for label, geom, color, alpha in (
            ("reference", ref, "#1f77b4", 0.18),
            ("fresh", fresh, "#d95f02", 0.12),
        ):
            band = geom["band"]
            line = geom["mean_line"]
            half = len(band) // 2
            ax.fill(band[:, 0], band[:, 1], color=color, alpha=alpha, label=f"{label} band")
            ax.plot(line[:, 0], line[:, 1], color=color, lw=1.6, label=f"{label} mean")
            if half:
                ax.scatter(band[:half, 0], band[:half, 1], s=4, color=color, alpha=0.25)
        ax.set_title(name.replace(".svg", ""), fontsize=8)
        ax.set_xlabel("normalized x within axes")
        ax.set_ylabel("normalized y within axes")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(preview_dir / f"{Path(name).stem}_normalized_overlay.png", dpi=220)
        plt.close(fig)

    _write_csv(
        OUT_DIR / "figure6_panel_ab_svg_geometry_comparison.csv",
        rows,
        [
            "panel_file",
            "reference_sha256",
            "fresh_sha256",
            "byte_identical",
            "reference_axes_bbox_svg_units",
            "fresh_axes_bbox_svg_units",
            "reference_band_points",
            "fresh_band_points",
            "reference_mean_line_id",
            "fresh_mean_line_id",
            "reference_mean_line_points",
            "fresh_mean_line_points",
            "max_abs_normalized_band_delta",
            "max_abs_normalized_mean_line_delta",
            "scientific_difference_detected",
            "interpretation",
        ],
    )
    _write_csv(
        OUT_DIR / "figure6_panel_ab_svg_geometry_points.csv",
        point_rows,
        ["panel_file", "source", "geometry", "point_index", "normalized_x", "normalized_y"],
    )


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
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _trial_dt_ms(trial: dict, n_samples: int) -> float:
    start = _parse_time(trial.get("start_time"))
    end = _parse_time(trial.get("end_time"))
    if start is None or end is None or n_samples < 2:
        return float("nan")
    return (end - start).total_seconds() * 1000.0 / float(n_samples - 1)


def _sustained_crossing_after_reset(evidence: np.ndarray, reset_index: int, dt_ms: float) -> float:
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


def summarize_reset_prior_pkls() -> None:
    base_dir = _load_base_dir()
    input_dir = base_dir / "Monkey 1" / "AI Respawn" / "resetPriorFiles"
    files = sorted(input_dir.glob("reset_prior_analyzes_*.pkl"))
    trial_rows = []
    session_rows = []
    lag_rows = []
    for path in files:
        with path.open("rb") as f:
            data = pickle.load(f)
        trials = data.get("trials", [])
        n_success = 0
        for trial in trials:
            answer = trial.get("answer")
            is_success = answer == 1
            n_success += int(is_success)
            samples = trial.get("samples", {})
            pos = np.asarray(samples.get("position", []), dtype=float)
            bci = np.asarray(samples.get("bci_velocity", []), dtype=float)
            ai = np.asarray(samples.get("ai_velocity", []), dtype=float)
            n = min(len(pos), len(bci), len(ai))
            reset_index = trial.get("target_jump_index")
            include_lag = (
                reset_index is not None
                and trial.get("true_goal") is not None
                and trial.get("target_jump_position") is not None
                and n >= 3
            )
            dt_ms = _trial_dt_ms(trial, n)
            bci_latency = ai_latency = lag = float("nan")
            bci_crossed = ai_crossed = False
            if include_lag:
                reset_index = int(reset_index)
                e_bci = _old_new_evidence(bci[:n], pos[:n], trial["true_goal"], trial["target_jump_position"])
                e_ai = _old_new_evidence(ai[:n], pos[:n], trial["true_goal"], trial["target_jump_position"])
                bci_latency = _sustained_crossing_after_reset(e_bci, reset_index, dt_ms)
                ai_latency = _sustained_crossing_after_reset(e_ai, reset_index, dt_ms)
                bci_crossed = bool(np.isfinite(bci_latency))
                ai_crossed = bool(np.isfinite(ai_latency))
                if bci_crossed and ai_crossed:
                    lag = float(ai_latency - bci_latency)
            trial_rows.append(
                {
                    "source_pkl": str(path),
                    "session": data.get("session", path.stem),
                    "trial_id": trial.get("trial_id"),
                    "answer_log": trial.get("answer_log"),
                    "answer": answer,
                    "answer_reason": trial.get("answer_reason"),
                    "included_success_denominator": True,
                    "success": is_success,
                    "included_lag_denominator": include_lag,
                    "n_samples": n,
                    "target_jump_index": trial.get("target_jump_index"),
                    "dt_ms_from_start_end_over_n_minus_1": repr(float(dt_ms)) if np.isfinite(dt_ms) else "",
                    "bci_reoriented_after_reset": bci_crossed,
                    "ai_reoriented_after_reset": ai_crossed,
                    "bci_reorientation_latency_ms": repr(float(bci_latency)) if np.isfinite(bci_latency) else "",
                    "ai_reorientation_latency_ms": repr(float(ai_latency)) if np.isfinite(ai_latency) else "",
                    "ai_minus_bci_lag_ms": repr(float(lag)) if np.isfinite(lag) else "",
                }
            )
            lag_rows.append(trial_rows[-1])
        n_total = len(trials)
        session_rows.append(
            {
                "source_pkl": str(path),
                "session": data.get("session", path.stem),
                "n_trials_success_denominator": n_total,
                "n_success": n_success,
                "success_fraction": repr(float(n_success / n_total)) if n_total else "",
            }
        )
    lag_values = np.asarray(
        [float(r["ai_minus_bci_lag_ms"]) for r in lag_rows if r["ai_minus_bci_lag_ms"] != ""],
        dtype=float,
    )
    bci_cross = [r for r in lag_rows if r["bci_reoriented_after_reset"]]
    ai_cross = [r for r in lag_rows if r["ai_reoriented_after_reset"]]
    both = [r for r in lag_rows if r["ai_minus_bci_lag_ms"] != ""]
    total_trials = sum(int(r["n_trials_success_denominator"]) for r in session_rows)
    total_success = sum(int(r["n_success"]) for r in session_rows)
    aggregate_rows = [
        {
            "metric": "reset_prior_success_fraction_all_trials",
            "numerator": total_success,
            "denominator": total_trials,
            "value": repr(float(total_success / total_trials)),
            "interpretation": "stored reset-prior replay output success; approximately 80%",
        },
        {
            "metric": "reset_prior_lag_median_ai_minus_bci_ms_both_reoriented",
            "numerator": len(both),
            "denominator": len(lag_rows),
            "value": repr(float(np.nanmedian(lag_values))) if len(lag_values) else "",
            "interpretation": "stored reset-prior replay output lag; median is zero when both streams reorient",
        },
        {
            "metric": "reset_prior_lag_iqr_ai_minus_bci_ms_both_reoriented",
            "numerator": len(both),
            "denominator": len(lag_rows),
            "value": repr(float(np.nanpercentile(lag_values, 75) - np.nanpercentile(lag_values, 25)))
            if len(lag_values)
            else "",
            "interpretation": "IQR of stored reset-prior replay lag",
        },
        {
            "metric": "bci_reoriented_after_reset_fraction",
            "numerator": len(bci_cross),
            "denominator": len(lag_rows),
            "value": repr(float(len(bci_cross) / len(lag_rows))) if lag_rows else "",
            "interpretation": "sustained positive BCI evidence after reset index",
        },
        {
            "metric": "ai_reoriented_after_reset_fraction",
            "numerator": len(ai_cross),
            "denominator": len(lag_rows),
            "value": repr(float(len(ai_cross) / len(lag_rows))) if lag_rows else "",
            "interpretation": "sustained positive AI/executed evidence after reset index",
        },
    ]
    _write_csv(
        OUT_DIR / "reset_prior_trial_inclusion_success_and_lag.csv",
        trial_rows,
        [
            "source_pkl",
            "session",
            "trial_id",
            "answer_log",
            "answer",
            "answer_reason",
            "included_success_denominator",
            "success",
            "included_lag_denominator",
            "n_samples",
            "target_jump_index",
            "dt_ms_from_start_end_over_n_minus_1",
            "bci_reoriented_after_reset",
            "ai_reoriented_after_reset",
            "bci_reorientation_latency_ms",
            "ai_reorientation_latency_ms",
            "ai_minus_bci_lag_ms",
        ],
    )
    _write_csv(
        OUT_DIR / "reset_prior_session_success_summary.csv",
        session_rows,
        ["source_pkl", "session", "n_trials_success_denominator", "n_success", "success_fraction"],
    )
    _write_csv(
        OUT_DIR / "reset_prior_aggregate_success_lag_summary.csv",
        aggregate_rows,
        ["metric", "numerator", "denominator", "value", "interpretation"],
    )

    log_dir = base_dir / "Monkey 1" / "AI Respawn" / "Reset_prior_analyze"
    provenance = {
        "input_reset_prior_pkl_dir": str(input_dir),
        "n_reset_prior_pkls": len(files),
        "success_source": "trial['answer'] in reset_prior_analyzes_*.pkl, generated by recovered log parser/evaluator",
        "lag_source": "stored samples['bci_velocity'] and samples['ai_velocity'] in reset_prior_analyzes_*.pkl",
        "lag_timebase": "PKLs lack per-sample timestamps; dt_ms inferred as (end_time - start_time)/(n_samples - 1)",
        "lag_crossing_rule": {
            "evidence": "unit velocity projection to post-respawn target minus projection to pre-respawn target",
            "minimum_speed": MIN_SPEED,
            "sustained_positive_evidence_ms": SUSTAIN_MS,
            "latency_zero": "target_jump_index/reset index from stored PKL",
        },
        "source_log_dir_expected_by_parser": str(log_dir),
        "source_log_dir_exists": log_dir.exists(),
        "n_source_logs_found": len(list(log_dir.glob('*.log'))) if log_dir.exists() else 0,
        "recovered_parser": str((PROJECT_ROOT.parent / "AI" / "analyse_reset_prior.py").resolve()),
        "end_to_end_replay_implementation_recovered": False,
    }
    (OUT_DIR / "reset_prior_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def write_extended_data_figure3_band_sources() -> None:
    rows = [
        {
            "manuscript_panel_or_file": "figures/Extended Data Figure 3/BCI influence/time_resolved_bci_influence.svg; upper axis",
            "generator": "scripts/run_time_resolved_bci_influence.py",
            "source_table": "tables/time_resolved_bci_influence_by_progress.csv",
            "plotted_center": "mean_directional_alignment",
            "band_low": "mean_directional_alignment - sem_directional_alignment",
            "band_high": "mean_directional_alignment + sem_directional_alignment",
            "calculation": "pandas group SEM of directional_alignment by task_family/progress_bin",
            "is_mean_plus_minus_1p96_sem": False,
            "is_bootstrap": False,
            "code_lines_checked": "summarize_by_progress stores sem_directional_alignment; plot_influence fill_between uses +/- sem",
        },
        {
            "manuscript_panel_or_file": "figures/Extended Data Figure 3/BCI influence/time_resolved_bci_influence.svg; lower axis",
            "generator": "scripts/run_time_resolved_bci_influence.py",
            "source_table": "tables/time_resolved_bci_influence_by_progress.csv",
            "plotted_center": "beta_current_bci",
            "band_low": "ci_low",
            "band_high": "ci_high",
            "calculation": "statsmodels OLS confidence interval for bci_z coefficient with cluster-robust covariance by session_uid",
            "is_mean_plus_minus_1p96_sem": False,
            "is_bootstrap": False,
            "code_lines_checked": "fit.conf_int() after OLS(...).fit(cov_type='cluster'); plot_influence fill_between uses ci_low/ci_high",
        },
    ]
    _write_csv(
        OUT_DIR / "extended_data_figure3_actual_panel_band_sources.csv",
        rows,
        [
            "manuscript_panel_or_file",
            "generator",
            "source_table",
            "plotted_center",
            "band_low",
            "band_high",
            "calculation",
            "is_mean_plus_minus_1p96_sem",
            "is_bootstrap",
            "code_lines_checked",
        ],
    )


def write_recovery_status() -> None:
    base_dir = _load_base_dir()
    log_dir = base_dir / "Monkey 1" / "AI Respawn" / "Reset_prior_analyze"
    lines = [
        "# Figure 6 Reset-Prior Recovery Status",
        "",
        "Recovered and used:",
        "",
        f"- Stored reset-prior replay PKLs in `{base_dir / 'Monkey 1' / 'AI Respawn' / 'resetPriorFiles'}`.",
        "- Recovered log-to-PKL parser/evaluator at `..\\AI\\analyse_reset_prior.py`.",
        "- Figure redraw code at `scripts\\make_temporal_prior_reset_example.py`.",
        "",
        "Not recovered:",
        "",
        f"- Source reset-prior logs: expected directory `{log_dir}`; exists = `{log_dir.exists()}`; `*.log` count = `{len(list(log_dir.glob('*.log'))) if log_dir.exists() else 0}`.",
        "- The original controller/replay implementation that applied temporal-prior reset before logging.",
        "",
        "Reproducibility status:",
        "",
        "- Stored-output reproducibility is supported: aggregate success and stored BCI/AI lag can be recomputed from the reset-prior PKLs.",
        "- End-to-end replay reproducibility is not supported in this checkout: the source logs and controller replay implementation are absent.",
    ]
    (OUT_DIR / "reset_prior_recovery_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_panels()
    summarize_reset_prior_pkls()
    write_extended_data_figure3_band_sources()
    write_recovery_status()
    print(f"Wrote remaining Figure 6 checks to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
