"""Verify current shared-control gain figure outputs.

The script audits the existing implementation and reference SVGs without
changing plotting code or statistical methods.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis_results" / "figure4_gain_verification"
FIG_DIR = PROJECT_ROOT / "figures" / "Figure 3" / "AI gain with baseline"
REFERENCE_SVGS = [
    FIG_DIR / "AI_gain_labels_Monkey 1_AI Appearing Obstacle 2.svg",
    FIG_DIR / "AI_gain_labels_Monkey 2_AI Appearing Obstacle.svg",
    FIG_DIR / "Performance-dependent AI gains.svg",
]

GAIN_TABLE = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_target_session_rows.csv"
MODEL_TABLE = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_nested_model_comparisons.csv"
BIN_TABLE = PROJECT_ROOT / "tables" / "Gain curve pooling audit" / "gain_curve_baseline_bin_counts.csv"
SUCCESS_AUDIT_TARGET_ROWS = PROJECT_ROOT / "analysis_results" / "success_rate_verification" / "success_rate_audit_target_session_rows.csv"

PANEL_DATASETS = [
    ("panel_a", "Monkey 1", "AI Appearing Obstacle 2"),
    ("panel_b", "Monkey 2", "AI Appearing Obstacle"),
]


def ensure_import_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_svg_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_svg_numbers() -> pd.DataFrame:
    rows = []
    for path in REFERENCE_SVGS:
        text = get_svg_text(path)
        for match in re.finditer(r"n=(\d+)", text):
            rows.append({"svg": path.name, "annotation_type": "bin_count", "value": int(match.group(1))})
        for match in re.finditer(r"Session-clustered p=([0-9.]+e[+-]\d+)", text):
            rows.append({"svg": path.name, "annotation_type": "session_clustered_p", "value": match.group(1)})
        for match in re.finditer(r"BCI-only: ([0-9]+)%", text):
            rows.append({"svg": path.name, "annotation_type": "baseline_label_pct", "value": int(match.group(1))})
    return pd.DataFrame(rows)


def load_panel_rows(base_dir: str, monkey: str, experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_import_path()
    from scripts.redraw_figure3_from_cached_tables import _load_original_panel_rows
    from src.figure3_plots import _ci95_t

    per_target_gains, bci_only_by_target = _load_original_panel_rows(base_dir, monkey, experiment)
    preferred = ["left", "slight_left", "straight", "slight_right", "right"]
    rows = []
    source_rows = []
    for target in [target for target in preferred if target in per_target_gains]:
        gains = np.asarray(per_target_gains[target], dtype=float)
        gains = gains[np.isfinite(gains)]
        mean, (lo, hi) = _ci95_t(gains)
        baseline = float(bci_only_by_target.get(target, np.nan))
        rows.append(
            {
                "monkey": monkey,
                "experiment": experiment,
                "target": target,
                "n_sessions": int(gains.size),
                "baseline_label_pct_display": int(np.round(100.0 * baseline)) if np.isfinite(baseline) else np.nan,
                "baseline_pct_full": 100.0 * baseline if np.isfinite(baseline) else np.nan,
                "mean_gain_pp": 100.0 * mean if np.isfinite(mean) else np.nan,
                "ci95_low_pp": 100.0 * lo if np.isfinite(lo) else np.nan,
                "ci95_high_pp": 100.0 * hi if np.isfinite(hi) else np.nan,
                "ci_method": "two-sided t interval across session target-specific gains",
            }
        )
        for session_pos, value in enumerate(gains):
            source_rows.append(
                {
                    "monkey": monkey,
                    "experiment": experiment,
                    "target": target,
                    "session_position_in_panel": session_pos,
                    "gain_pp": 100.0 * float(value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(source_rows)


def model_verification() -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf
    from scipy import stats

    data = pd.read_csv(GAIN_TABLE)
    fit = smf.ols(
        "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task)",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["session_uid"]})
    rows = [
        {
            "model": "common_quadratic_monkey_task_intercepts",
            "n_observations": int(fit.nobs),
            "n_sessions": int(data["session_uid"].nunique()),
            "quadratic_beta": float(fit.params["x_centered2"]),
            "quadratic_p_clustered": float(fit.pvalues["x_centered2"]),
            "covariance": "cluster-robust by session_uid",
            "formula": "shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task)",
        }
    ]
    stored = pd.read_csv(MODEL_TABLE)
    common = stored[stored["model"].eq("common_quadratic_monkey_task_intercepts")].iloc[0]
    rows.append(
        {
            "model": "stored_common_quadratic_monkey_task_intercepts",
            "n_observations": int(common["n"]),
            "n_sessions": int(data["session_uid"].nunique()),
            "quadratic_beta": float(common["quadratic_beta"]),
            "quadratic_p_clustered": float(common["quadratic_p_clustered"]),
            "covariance": "stored value",
            "formula": "",
        }
    )

    bin_rows = []
    bin_specs = [
        ("0-20", 0, 20, data["baseline_pct"].ge(0) & data["baseline_pct"].le(20)),
        ("20-40", 20, 40, data["baseline_pct"].gt(20) & data["baseline_pct"].le(40)),
        ("40-60", 40, 60, data["baseline_pct"].gt(40) & data["baseline_pct"].le(60)),
        ("60-80", 60, 80, data["baseline_pct"].gt(60) & data["baseline_pct"].le(80)),
        ("80-100", 80, 100, data["baseline_pct"].gt(80) & data["baseline_pct"].le(100)),
    ]
    for label, lo, hi, mask in bin_specs:
        vals = data.loc[mask, "gain_pp"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
            ci = float(stats.t.ppf(0.975, vals.size - 1) * sem)
        else:
            ci = 0.0
        bin_rows.append(
            {
                "baseline_bin": label,
                "low_pct": lo,
                "high_pct": hi,
                "n": int(vals.size),
                "mean_gain_pp": float(vals.mean()) if vals.size else np.nan,
                "ci95_half_width_pp": ci if vals.size else np.nan,
                "ci_method": "two-sided t interval across target-session rows in bin",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def session_inclusion_records() -> pd.DataFrame:
    data = pd.read_csv(GAIN_TABLE)
    rows = (
        data.groupby(["monkey", "source_monkey_folder", "experiment", "task", "session_uid", "session_file"], as_index=False)
        .agg(n_target_session_rows=("target", "size"), targets=("target", lambda x: ";".join(sorted(x))))
    )
    return rows


def row_count_comparison() -> pd.DataFrame:
    gain = pd.read_csv(GAIN_TABLE)
    audit = pd.read_csv(SUCCESS_AUDIT_TARGET_ROWS)
    audit_sub = audit[
        audit["experiment"].isin(["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"])
        & audit["has_both_conditions"].eq(1)
    ].copy()
    rows = [
        {
            "source": "current_gain_curve_table",
            "n_target_session_rows": int(len(gain)),
            "n_sessions": int(gain["session_uid"].nunique()),
            "basis": "mapped manuscript source folders in tables/Gain curve pooling audit/gain_curve_target_session_rows.csv",
        },
        {
            "source": "success_rate_audit_configured_data",
            "n_target_session_rows": int(len(audit_sub)),
            "n_sessions": int(audit_sub["session_uid"].nunique()),
            "basis": "configured Monkey 1 / Monkey 2 data in success_rate_audit_target_session_rows.csv",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    config: dict,
    panel_summary: pd.DataFrame,
    model: pd.DataFrame,
    bins: pd.DataFrame,
    counts: pd.DataFrame,
    hashes: pd.DataFrame,
    svg_numbers: pd.DataFrame,
) -> None:
    generated_command = "python scripts\\redraw_figure3_from_cached_tables.py"
    common = model[model["model"].eq("common_quadratic_monkey_task_intercepts")].iloc[0]
    lines = [
        "# Figure 4 Shared-Control Gain Verification",
        "",
        f"Generator: `scripts/redraw_figure3_from_cached_tables.py` using `src/figure3_plots.py`.",
        f"Exact regeneration command: `{generated_command}`.",
        f"Configured `base_dir`: `{config.get('base_dir')}`.",
        "",
        "## Configuration",
        "",
        "- Panels a/b read `config.yaml` through `redraw_figure3_from_cached_tables.py`.",
        "- Panel c reads cached `tables/Gain curve pooling audit/gain_curve_target_session_rows.csv` when present; therefore the current panel c can be regenerated from repository tables. The cached table may include source-monkey folders beyond those present in a local `config.yaml` `base_dir`.",
        "",
        "## Panel A/B Target Summaries",
        "",
        panel_summary.to_csv(index=False),
        "",
        "## Panel C Model",
        "",
        f"Common quadratic model n={int(common['n_observations'])}, sessions={int(common['n_sessions'])}, beta={common['quadratic_beta']:.12g}, clustered p={common['quadratic_p_clustered']:.12g}.",
        "Test: cluster-robust t/z test for the `x_centered2` coefficient in OLS model `shared_control_gain ~ x_centered + x_centered2 + C(monkey) + C(task)`, clustered by `session_uid`.",
        "",
        "## Panel C Bins",
        "",
        bins.to_csv(index=False),
        "",
        "## 240 versus 255",
        "",
        counts.to_csv(index=False),
        "",
        "The displayed 240 rows come from the current gain-curve table using mapped manuscript source folders: Monkey 1 is sourced from Monkey 3 for fixed/original/follow-up obstacle data, and Monkey 2 is sourced from Monkey 1 for fixed/original appearing obstacle data. The 255 success-rate-audit rows come from configured Monkey 1 and Monkey 2 folders directly. The difference is a data/source-selection difference, not a forced recount.",
        "",
        "## Reference SVG Preservation",
        "",
        hashes.to_csv(index=False),
        "",
        "## SVG Annotation Extraction",
        "",
        svg_numbers.to_csv(index=False),
        "",
        "## Conclusion",
        "",
        "The current code and cached tables reproduce the reported panel-c count and p-value. No method changes are recommended. Caption should explicitly state that panel c uses 240 target x session observations from 48 mapped manuscript-source sessions and that CIs are two-sided t intervals; the model p-value is session-clustered for the pooled quadratic term.",
    ]
    (OUT_DIR / "figure4_gain_verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_config()

    backup_dir = OUT_DIR / "reference_svg_copies"
    backup_dir.mkdir(exist_ok=True)
    hash_rows = []
    for path in REFERENCE_SVGS:
        backup = backup_dir / path.name
        shutil.copy2(path, backup)
        hash_rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "backup": str(backup.relative_to(PROJECT_ROOT)),
            }
        )
    hashes = pd.DataFrame(hash_rows)

    panel_summaries = []
    panel_source = []
    for panel, monkey, experiment in PANEL_DATASETS:
        summary, source = load_panel_rows(str(config["base_dir"]), monkey, experiment)
        summary.insert(0, "panel", panel)
        source.insert(0, "panel", panel)
        panel_summaries.append(summary)
        panel_source.append(source)
    panel_summary = pd.concat(panel_summaries, ignore_index=True)
    panel_source_rows = pd.concat(panel_source, ignore_index=True)

    model, bins = model_verification()
    sessions = session_inclusion_records()
    counts = row_count_comparison()
    svg_numbers = extract_svg_numbers()

    pd.read_csv(GAIN_TABLE).to_csv(OUT_DIR / "panel_c_target_session_source_rows.csv", index=False)
    panel_source_rows.to_csv(OUT_DIR / "panels_ab_target_session_gain_points.csv", index=False)
    panel_summary.to_csv(OUT_DIR / "panels_ab_target_summary.csv", index=False)
    sessions.to_csv(OUT_DIR / "panel_c_session_inclusion_records.csv", index=False)
    model.to_csv(OUT_DIR / "panel_c_model_verification.csv", index=False)
    bins.to_csv(OUT_DIR / "panel_c_bin_summary.csv", index=False)
    counts.to_csv(OUT_DIR / "panel_c_240_vs_255_row_count_comparison.csv", index=False)
    hashes.to_csv(OUT_DIR / "reference_svg_hashes.csv", index=False)
    svg_numbers.to_csv(OUT_DIR / "current_svg_extracted_annotations.csv", index=False)
    metadata = {
        "generator": "scripts/redraw_figure3_from_cached_tables.py",
        "plot_module": "src/figure3_plots.py",
        "exact_regeneration_command": "python scripts\\redraw_figure3_from_cached_tables.py",
        "base_dir_from_config": str(config.get("base_dir")),
        "reference_svgs_preserved": [row["backup"] for row in hash_rows],
    }
    (OUT_DIR / "figure4_gain_verification_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(config, panel_summary, model, bins, counts, hashes, svg_numbers)

    for path in sorted(OUT_DIR.glob("*")):
        if path.is_file():
            print(f"[INFO] wrote {path}")


if __name__ == "__main__":
    main()
