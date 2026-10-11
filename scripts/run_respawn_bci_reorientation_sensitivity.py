"""
Sensitivity and survival-style summaries for Respawn decoded BCI reorientation.

The primary latency extractor in run_respawn_bci_reorientation_by_outcome.py is
reused with a 3 s post-jump window, then summarized at shorter windows without
changing the original 2 s outputs.

It inherits the final Respawn answer-code inclusion rule from that extractor:
answer 1 is success, unsuccessful outcomes are {3,5,6}, and {4,9,10} are
excluded from the final manuscript pool.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
WINDOWS_MS = [1000.0, 1500.0, 2000.0, 3000.0]
TIME_GRID_MS = np.arange(0.0, 3000.0 + 50.0, 50.0)
SUSTAIN_MS = 200.0
GROUP_ORDER = [
    ("AI", "correct", "Shared control correct"),
    ("AI", "incorrect", "Shared control incorrect"),
    ("No AI", "correct", "BCI-only correct"),
    ("No AI", "incorrect", "BCI-only incorrect"),
]


def _load_reorientation_module():
    path = PROJECT_ROOT / "scripts" / "run_respawn_bci_reorientation_by_outcome.py"
    spec = importlib.util.spec_from_file_location("respawn_bci_reorientation_by_outcome", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_base_dir(cli_base_dir: str | None) -> str:
    if cli_base_dir:
        return cli_base_dir
    import yaml

    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def _latencies_for(group: pd.DataFrame, window_ms: float) -> pd.Series:
    lat = pd.to_numeric(group["t_bci_modelvelocity_ms"], errors="coerce")
    max_onset_ms = window_ms - SUSTAIN_MS
    return lat[(lat >= 0.0) & (lat <= max_onset_ms)].dropna()


def _fisher_p(correct: pd.DataFrame, incorrect: pd.DataFrame, window_ms: float) -> float:
    c_lat = pd.to_numeric(correct["t_bci_modelvelocity_ms"], errors="coerce")
    i_lat = pd.to_numeric(incorrect["t_bci_modelvelocity_ms"], errors="coerce")
    max_onset_ms = window_ms - SUSTAIN_MS
    c_cross = int(((c_lat >= 0.0) & (c_lat <= max_onset_ms)).sum())
    i_cross = int(((i_lat >= 0.0) & (i_lat <= max_onset_ms)).sum())
    table = [[c_cross, len(correct) - c_cross], [i_cross, len(incorrect) - i_cross]]
    if min(sum(table[0]), sum(table[1])) == 0:
        return np.nan
    return float(stats.fisher_exact(table, alternative="two-sided").pvalue)


def _logrank_p(group_a: pd.DataFrame, group_b: pd.DataFrame, window_ms: float) -> float:
    times_a, events_a = _survival_inputs(group_a, window_ms)
    times_b, events_b = _survival_inputs(group_b, window_ms)
    event_times = np.sort(np.unique(np.concatenate([times_a[events_a], times_b[events_b]])))
    observed_minus_expected = 0.0
    variance = 0.0
    for t in event_times:
        risk_a = np.sum(times_a >= t)
        risk_b = np.sum(times_b >= t)
        events_at_t_a = np.sum((times_a == t) & events_a)
        events_at_t_b = np.sum((times_b == t) & events_b)
        risk_total = risk_a + risk_b
        events_total = events_at_t_a + events_at_t_b
        if risk_total <= 1 or events_total == 0:
            continue
        expected_a = events_total * risk_a / risk_total
        var_a = (
            risk_a
            * risk_b
            * events_total
            * (risk_total - events_total)
            / (risk_total * risk_total * (risk_total - 1))
        )
        observed_minus_expected += events_at_t_a - expected_a
        variance += var_a
    if variance <= 0:
        return np.nan
    chi2 = observed_minus_expected * observed_minus_expected / variance
    return float(stats.chi2.sf(chi2, df=1))


def _survival_inputs(group: pd.DataFrame, window_ms: float) -> tuple[np.ndarray, np.ndarray]:
    lat = pd.to_numeric(group["t_bci_modelvelocity_ms"], errors="coerce").to_numpy(dtype=float)
    max_onset_ms = window_ms - SUSTAIN_MS
    events = np.isfinite(lat) & (lat >= 0.0) & (lat <= max_onset_ms)
    times = np.where(events, lat, window_ms)
    return times, events


def collect_rows(monkeys: list[str], base_dir: str) -> pd.DataFrame:
    module = _load_reorientation_module()
    module.WINDOW_MS = (-1000.0, 3000.0)
    return module.collect_rows(monkeys, base_dir)


def summarize_sensitivity(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    test_rows = []
    for window_ms in WINDOWS_MS:
        for condition, outcome, label in GROUP_ORDER:
            group = rows[(rows["condition"] == condition) & (rows["outcome"] == outcome)]
            lat = _latencies_for(group, window_ms)
            summary_rows.append({
                "window_ms": window_ms,
                "condition": "Shared control" if condition == "AI" else "BCI-only",
                "outcome": outcome,
                "n_trials": int(len(group)),
                "n_bci_reoriented": int(len(lat)),
                "pct_bci_reoriented": 100.0 * len(lat) / len(group) if len(group) else np.nan,
                "median_bci_latency_ms": float(np.nanmedian(lat)) if len(lat) else np.nan,
                "iqr_bci_latency_ms": (
                    float(np.nanpercentile(lat, 75) - np.nanpercentile(lat, 25)) if len(lat) else np.nan
                ),
                "label": label,
            })

        for condition, condition_label in (("AI", "Shared control"), ("No AI", "BCI-only")):
            correct = rows[(rows["condition"] == condition) & (rows["outcome"] == "correct")]
            incorrect = rows[(rows["condition"] == condition) & (rows["outcome"] == "incorrect")]
            c_lat = _latencies_for(correct, window_ms)
            i_lat = _latencies_for(incorrect, window_ms)
            mw_p = (
                float(stats.mannwhitneyu(c_lat, i_lat, alternative="two-sided").pvalue)
                if len(c_lat) and len(i_lat) else np.nan
            )
            test_rows.append({
                "window_ms": window_ms,
                "comparison": f"{condition_label} incorrect vs correct",
                "test_family": "within_condition",
                "n_correct_reoriented": int(len(c_lat)),
                "n_incorrect_reoriented": int(len(i_lat)),
                "correct_median_ms": float(np.nanmedian(c_lat)) if len(c_lat) else np.nan,
                "incorrect_median_ms": float(np.nanmedian(i_lat)) if len(i_lat) else np.nan,
                "incorrect_minus_correct_median_ms": (
                    float(np.nanmedian(i_lat) - np.nanmedian(c_lat)) if len(c_lat) and len(i_lat) else np.nan
                ),
                "latency_mannwhitney_p": mw_p,
                "crossing_rate_fisher_p": _fisher_p(correct, incorrect, window_ms),
                "survival_logrank_p": _logrank_p(correct, incorrect, window_ms),
            })

        test_rows.append(_interaction_row(rows, window_ms))
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def _interaction_row(rows: pd.DataFrame, window_ms: float) -> dict:
    subset = rows.copy()
    lat = pd.to_numeric(subset["t_bci_modelvelocity_ms"], errors="coerce")
    max_onset_ms = window_ms - SUSTAIN_MS
    subset = subset[(lat >= 0.0) & (lat <= max_onset_ms)].copy()
    subset["latency_ms"] = pd.to_numeric(subset["t_bci_modelvelocity_ms"], errors="coerce")
    subset["shared_control"] = (subset["condition"] == "AI").astype(int)
    subset["incorrect"] = (subset["outcome"] == "incorrect").astype(int)
    fit = smf.ols("latency_ms ~ shared_control * incorrect", data=subset).fit(cov_type="HC3")
    med = subset.groupby(["condition", "outcome"])["latency_ms"].median()
    ai_penalty = med[("AI", "incorrect")] - med[("AI", "correct")]
    no_ai_penalty = med[("No AI", "incorrect")] - med[("No AI", "correct")]
    return {
        "window_ms": window_ms,
        "comparison": "Shared-control penalty vs BCI-only penalty",
        "test_family": "interaction",
        "n_correct_reoriented": int((subset["outcome"] == "correct").sum()),
        "n_incorrect_reoriented": int((subset["outcome"] == "incorrect").sum()),
        "correct_median_ms": np.nan,
        "incorrect_median_ms": np.nan,
        "incorrect_minus_correct_median_ms": float(ai_penalty - no_ai_penalty),
        "latency_mannwhitney_p": np.nan,
        "crossing_rate_fisher_p": np.nan,
        "survival_logrank_p": np.nan,
        "robust_ols_interaction_coefficient_ms": float(fit.params["shared_control:incorrect"]),
        "robust_ols_interaction_p": float(fit.pvalues["shared_control:incorrect"]),
    }


def cumulative_probability(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    lat_all = pd.to_numeric(rows["t_bci_modelvelocity_ms"], errors="coerce")
    rows = rows.copy()
    rows["latency_ms"] = lat_all
    for condition, outcome, label in GROUP_ORDER:
        group = rows[(rows["condition"] == condition) & (rows["outcome"] == outcome)]
        lat = group["latency_ms"].to_numpy(dtype=float)
        for t in TIME_GRID_MS:
            max_onset_ms = t - SUSTAIN_MS
            records.append({
                "time_ms": t,
                "condition": "Shared control" if condition == "AI" else "BCI-only",
                "outcome": outcome,
                "label": label,
                "n_trials": int(len(group)),
                "cumulative_reorientation_probability": (
                    float(np.mean(np.isfinite(lat) & (lat >= 0.0) & (lat <= max_onset_ms))) if len(lat) else np.nan
                ),
            })
    return pd.DataFrame(records)


def plot_cumulative(cumulative: pd.DataFrame) -> None:
    out_dir = PROJECT_ROOT / "figures" / "Extended Data Figure 3" / "Respawn BCI reorientation"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Shared control correct": "#23746f",
        "Shared control incorrect": "#b24b45",
        "BCI-only correct": "#4b66a3",
        "BCI-only incorrect": "#9a6a30",
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for label, group in cumulative.groupby("label", sort=False):
        group = group.sort_values("time_ms")
        ax.step(
            group["time_ms"],
            100.0 * group["cumulative_reorientation_probability"],
            where="post",
            label=label,
            color=colors.get(label),
            linewidth=1.8,
        )
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Time after target jump (ms)")
    ax.set_ylabel("Cumulative BCI reorientation (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(out_dir / f"respawn_bci_reorientation_cumulative_probability.{ext}", dpi=300)
    plt.close(fig)


def write_statement(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "tables" / "respawn_bci_reorientation_sensitivity_statement.md"
    primary = summary[summary["window_ms"] == 2000.0]
    lines = [
        "# Respawn BCI Reorientation Sensitivity",
        "",
        "Decoded BCI reorientation was summarized as cumulative probability over 0-3 s and at 1.0, 1.5, 2.0, and 3.0 s windows.",
        "A reorientation counted for a given window only when its full 200 ms sustained-positive evidence segment fit inside that window.",
        "",
        "## Primary 2 s window",
        "",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- {row['condition']} {row['outcome']}: {row['pct_bci_reoriented']:.1f}% reoriented "
            f"({int(row['n_bci_reoriented'])}/{int(row['n_trials'])}); median latency {row['median_bci_latency_ms']:.1f} ms."
        )
    lines.extend(["", "## Tests", ""])
    for _, row in tests[tests["window_ms"] == 2000.0].iterrows():
        if row["test_family"] == "interaction":
            lines.append(
                f"- {row['comparison']}: median penalty contrast {row['incorrect_minus_correct_median_ms']:.1f} ms; "
                f"robust OLS interaction p={row['robust_ols_interaction_p']:.3g}."
            )
        else:
            lines.append(
                f"- {row['comparison']}: latency p={row['latency_mannwhitney_p']:.3g}; "
                f"crossing-rate p={row['crossing_rate_fisher_p']:.3g}; log-rank p={row['survival_logrank_p']:.3g}."
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(monkeys: list[str], base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(monkeys, base_dir)
    rows.to_csv(tables_dir / "respawn_bci_reorientation_3s_trials.csv", index=False)
    sensitivity, tests = summarize_sensitivity(rows)
    cumulative = cumulative_probability(rows)
    sensitivity.to_csv(tables_dir / "respawn_bci_reorientation_window_sensitivity.csv", index=False)
    tests.to_csv(tables_dir / "respawn_bci_reorientation_window_tests.csv", index=False)
    cumulative.to_csv(tables_dir / "respawn_bci_reorientation_cumulative_probability.csv", index=False)
    plot_cumulative(cumulative)
    write_statement(sensitivity, tests)
    return sensitivity, tests, cumulative


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Respawn BCI reorientation cumulative and sensitivity analyses.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    args = parser.parse_args(argv)
    run(args.monkeys, load_base_dir(args.base_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
