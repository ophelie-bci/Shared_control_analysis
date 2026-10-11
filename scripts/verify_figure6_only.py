from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# src.plots imports seaborn for unrelated helpers on some installations.
sys.modules.setdefault("seaborn", types.SimpleNamespace(kdeplot=lambda *args, **kwargs: None))

import src.plots as plots
from src.plots import plot_alpha_vs_time_from_obstacle_appearance


OUT_DIR = PROJECT_ROOT / "analysis_results" / "figure6_prior_verification" / "final_verification"
PANEL_DIR = OUT_DIR / "fresh_panels_ab"
REF_DIR = PROJECT_ROOT / "figures" / "Figure 6" / "Prior confidence index"

CASES = [
    {
        "monkey": "Monkey 1",
        "experiment": "AI Appearing Obstacle",
        "manuscript_real_minus_pseudo": -0.041,
        "manuscript_real_vs_pseudo_p": 0.011,
    },
    {
        "monkey": "Monkey 2",
        "experiment": "AI Appearing Obstacle",
        "manuscript_real_minus_pseudo": -0.040,
        "manuscript_real_vs_pseudo_p": 0.012,
    },
    {
        "monkey": "Monkey 1",
        "experiment": "AI Respawn",
        "manuscript_real_minus_pseudo": -0.016,
        "manuscript_real_vs_pseudo_p": 0.028,
    },
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

WINDOW_ARGS = {
    "pre_window_ms": [-400.0, 0.0],
    "early_window_ms": [0.0, 400.0],
    "late_window_ms": [1600.0, 2400.0],
    "dip_window_ms": [0.0, 800.0],
    "plateau_window_ms": [1600.0, 2600.0],
}

TEST_ARGS = {
    "library": "scipy.stats.wilcoxon",
    "level": "session",
    "input": "finite session medians for each metric",
    "call": "wilcoxon(x, alternative='two-sided').pvalue",
    "null": "median session-level value equals 0",
    "zero_method": "scipy default",
    "correction": "scipy default",
    "mode": "scipy default",
}


def _load_base_dir() -> str:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("config.yaml is missing base_dir")
    return str(base_dir)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_save_plot():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, name, subfolder=None):
        path = PANEL_DIR / f"{name}.svg"
        fig.savefig(path, bbox_inches="tight")
        return path

    return save_plot


def _float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_dir = _load_base_dir()

    original_save_plot = plots.save_plot
    original_show = plt.show
    plots.save_plot = _make_save_plot()
    plt.show = lambda *args, **kwargs: None
    try:
        summary_rows = []
        session_rows = []
        figure_rows = []

        for case in CASES:
            result = plot_alpha_vs_time_from_obstacle_appearance(
                case["monkey"],
                case["experiment"],
                base_dir,
                **PLOT_ARGS,
            )
            plt.close("all")
            session = result["session_level"]
            label = f'{case["monkey"]} {case["experiment"]}'

            values = {
                "real_modulation": np.asarray(session["delta_early_values"], float),
                "pseudo_modulation": np.asarray(session["delta_pseudo_values"], float),
                "real_minus_pseudo": np.asarray(session["real_minus_pseudo_values"], float),
                "recovery": np.asarray(session["delta_recovery_values"], float),
                "dip_depth": np.asarray(session["dip_depth_valid_values"], float),
            }

            summary_rows.append(
                {
                    "condition": label,
                    "monkey": case["monkey"],
                    "experiment": case["experiment"],
                    "n_sessions": int(session["n_sessions"]),
                    "real_modulation_median": repr(_float(session["delta_early_median_across_sessions"])),
                    "real_modulation_p": repr(_float(session["delta_early_p"])),
                    "pseudo_modulation_median": repr(_float(session["delta_pseudo_median_across_sessions"])),
                    "pseudo_modulation_p": repr(_float(session["delta_pseudo_p"])),
                    "real_minus_pseudo_median": repr(_float(session["real_minus_pseudo_median_across_sessions"])),
                    "real_vs_pseudo_p": repr(_float(session["real_vs_pseudo_p"])),
                    "manuscript_real_minus_pseudo": repr(float(case["manuscript_real_minus_pseudo"])),
                    "manuscript_real_vs_pseudo_p": repr(float(case["manuscript_real_vs_pseudo_p"])),
                    "matches_manuscript_3dp": (
                        round(_float(session["real_minus_pseudo_median_across_sessions"]), 3)
                        == round(float(case["manuscript_real_minus_pseudo"]), 3)
                        and round(_float(session["real_vs_pseudo_p"]), 3)
                        == round(float(case["manuscript_real_vs_pseudo_p"]), 3)
                    ),
                    "recovery_median": repr(_float(session["delta_recovery_median_across_sessions"])),
                    "recovery_p": repr(_float(session["delta_recovery_p"])),
                    "dip_depth_median": repr(_float(session["dip_depth_valid_median_across_sessions"])),
                    "dip_depth_p": repr(_float(session["dip_depth_valid_p"])),
                    "test_call": TEST_ARGS["call"],
                }
            )

            n_sessions = int(session["n_sessions"])
            for idx in range(n_sessions):
                row = {
                    "condition": label,
                    "monkey": case["monkey"],
                    "experiment": case["experiment"],
                    "session_index": idx,
                }
                for key, arr in values.items():
                    row[key] = repr(float(arr[idx])) if idx < len(arr) and np.isfinite(arr[idx]) else ""
                session_rows.append(row)

            fresh_name = f'alpha_vs_time_FROM_EVENT_{case["monkey"]}_{case["experiment"]}.svg'
            fresh = PANEL_DIR / fresh_name
            ref = REF_DIR / fresh_name
            figure_rows.append(
                {
                    "panel_file": fresh_name,
                    "fresh_path": str(fresh),
                    "reference_path": str(ref),
                    "fresh_exists": fresh.exists(),
                    "reference_exists": ref.exists(),
                    "fresh_sha256": _sha256(fresh),
                    "reference_sha256": _sha256(ref),
                    "byte_identical": fresh.exists() and ref.exists() and _sha256(fresh) == _sha256(ref),
                    "fresh_size_bytes": fresh.stat().st_size if fresh.exists() else "",
                    "reference_size_bytes": ref.stat().st_size if ref.exists() else "",
                }
            )

        _write_csv(
            OUT_DIR / "figure6_full_precision_session_summary.csv",
            summary_rows,
            [
                "condition",
                "monkey",
                "experiment",
                "n_sessions",
                "real_modulation_median",
                "real_modulation_p",
                "pseudo_modulation_median",
                "pseudo_modulation_p",
                "real_minus_pseudo_median",
                "real_vs_pseudo_p",
                "manuscript_real_minus_pseudo",
                "manuscript_real_vs_pseudo_p",
                "matches_manuscript_3dp",
                "recovery_median",
                "recovery_p",
                "dip_depth_median",
                "dip_depth_p",
                "test_call",
            ],
        )
        _write_csv(
            OUT_DIR / "figure6_session_level_values.csv",
            session_rows,
            [
                "condition",
                "monkey",
                "experiment",
                "session_index",
                "real_modulation",
                "pseudo_modulation",
                "real_minus_pseudo",
                "recovery",
                "dip_depth",
            ],
        )
        _write_csv(
            OUT_DIR / "figure6_fresh_panel_svg_comparison.csv",
            figure_rows,
            [
                "panel_file",
                "fresh_path",
                "reference_path",
                "fresh_exists",
                "reference_exists",
                "fresh_sha256",
                "reference_sha256",
                "byte_identical",
                "fresh_size_bytes",
                "reference_size_bytes",
            ],
        )
        (OUT_DIR / "figure6_exact_arguments.json").write_text(
            json.dumps(
                {
                    "base_dir": base_dir,
                    "plot_function": "src.plots.plot_alpha_vs_time_from_obstacle_appearance",
                    "plot_args": PLOT_ARGS,
                    "window_args": WINDOW_ARGS,
                    "test_args": TEST_ARGS,
                    "manuscript_values_supplied_for_comparison": CASES,
                    "fresh_panel_output_dir": str(PANEL_DIR),
                    "reference_panel_dir": str(REF_DIR),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        plt.close("all")
        plots.save_plot = original_save_plot
        plt.show = original_show

    print(f"Wrote Figure 6 verification outputs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
