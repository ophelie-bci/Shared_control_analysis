"""
Quantify why the same prior-confidence index can yield different assistance.

For shared-control samples, assistance magnitude is

    ||v_exec - v_BCI||

This script bins samples into narrow alpha bins and summarizes the distribution
of assistance magnitude inside each bin. Explicit safety-override windows plus
a 150-ms buffer are excluded from the primary analysis because those samples
follow an intentionally prior-only control regime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
    "AI Respawn",
]
ALPHA_FLOOR = 0.2
MIN_SPEED = 1e-6
ALPHA_BIN_WIDTH = 0.05
OVERRIDE_PAD_MS = 150.0


def _ensure_import_path() -> None:
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


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trial_id(trial) -> int | float:
    for key in ("trial", "trial_id", "trialId", "trialIndex"):
        value = _get(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return np.nan


def _as_scalar(value) -> float:
    if value is None:
        return np.nan
    try:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return np.nan


def _timestamp_scale_to_ms(timestamps: np.ndarray) -> float:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 3:
        return 1.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size and np.nanmedian(diffs) < 1.0:
        return 1000.0
    return 1.0


def _finite_events(events) -> list:
    if events is None:
        return []
    if not isinstance(events, (list, tuple, np.ndarray)):
        events = [events]
    out = []
    for event in events:
        if isinstance(event, dict):
            out.append(event)
            continue
        value = _as_scalar(event)
        if np.isfinite(value):
            out.append(event)
    return out


def _events_to_indices(events, output_ts: np.ndarray, n: int) -> list[int]:
    indices: list[int] = []
    if not events:
        return indices
    has_ts = np.isfinite(output_ts).any()
    for event in events:
        if isinstance(event, (int, np.integer)) and 0 <= int(event) < n:
            indices.append(int(event))
            continue
        if isinstance(event, dict):
            used = False
            for key in ("idx", "index", "i", "frame"):
                if key in event:
                    try:
                        index = int(event[key])
                        if 0 <= index < n:
                            indices.append(index)
                            used = True
                    except Exception:
                        pass
                    break
            if used:
                continue
            for key in ("time", "timestamp", "ts", "t", "output_ts", "unitytime", "unityrealtime"):
                if key in event:
                    value = _as_scalar(event[key])
                    if np.isfinite(value) and has_ts:
                        indices.append(int(np.nanargmin(np.abs(output_ts - value))))
                    break
            continue
        value = _as_scalar(event)
        if np.isfinite(value) and has_ts:
            indices.append(int(np.nanargmin(np.abs(output_ts - value))))
    return sorted(set(i for i in indices if 0 <= i < n))


def _explicit_override_mask(trial, output_ts: np.ndarray, pad_ms: float = OVERRIDE_PAD_MS) -> np.ndarray:
    n = len(output_ts)
    mask = np.zeros(n, dtype=bool)
    on_events = _finite_events(_get(trial, "aiControlOn", []))
    off_events = _finite_events(_get(trial, "aiControlOff", []))
    if not on_events and not off_events:
        return mask

    on_idx = _events_to_indices(on_events, output_ts, n)
    off_idx = _events_to_indices(off_events, output_ts, n)
    if not on_idx and not off_idx:
        return mask

    while off_idx and (not on_idx or off_idx[0] < on_idx[0]):
        off_idx.pop(0)

    event_mask = np.zeros(n, dtype=bool)
    j = 0
    segments: list[tuple[int, int]] = []
    for start in sorted(on_idx):
        while j < len(off_idx) and off_idx[j] <= start:
            j += 1
        end = off_idx[j] if j < len(off_idx) else n - 1
        segments.append((start, end))
        event_mask[start : end + 1] = True
        if j < len(off_idx):
            j += 1

    if np.isfinite(output_ts).any():
        scale = _timestamp_scale_to_ms(output_ts)
        pad_units = pad_ms / scale
        for start, end in segments:
            left = output_ts[start] - pad_units
            right = output_ts[end] + pad_units
            if np.isfinite(left) and np.isfinite(right):
                mask |= (output_ts >= left) & (output_ts <= right)
    else:
        mask = event_mask
    return mask


def _vec3(value) -> np.ndarray:
    if value is None:
        return np.full(3, np.nan, dtype=float)
    if isinstance(value, dict):
        for keys in (("vx", "vy", "vz"), ("x", "y", "z"), ("Item1", "Item2", "Item3")):
            if all(k in value for k in keys):
                return np.asarray([value[keys[0]], value[keys[1]], value[keys[2]]], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(3, np.nan, dtype=float)
    out = np.full(3, np.nan, dtype=float)
    out[: min(3, arr.size)] = arr[:3]
    return out


def _robust_alpha(entropy: np.ndarray, floor: float = ALPHA_FLOOR) -> np.ndarray:
    entropy = np.asarray(entropy, dtype=float)
    if entropy.size == 0 or np.all(np.isnan(entropy)):
        return np.full_like(entropy, np.nan, dtype=float)
    hmax_obs = np.nanmax(entropy)
    if not np.isfinite(hmax_obs) or hmax_obs <= 0:
        hmax_obs = np.nanpercentile(entropy, 95)
    k_est = max(2, int(np.round(np.exp(hmax_obs))))
    hmax = float(np.log(k_est))
    h = np.clip(entropy, 0.0, hmax)
    alpha = 1.0 - (h / hmax)
    alpha = np.nan_to_num(alpha, nan=float(floor), posinf=float(floor), neginf=1.0)
    return np.clip(alpha, float(floor), 1.0)


def _alpha_bin(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bins = int(round((1.0 - ALPHA_FLOOR) / ALPHA_BIN_WIDTH))
    edges = ALPHA_FLOOR + np.arange(n_bins + 1) * ALPHA_BIN_WIDTH
    idx = np.floor((alpha - ALPHA_FLOOR) / ALPHA_BIN_WIDTH).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    starts = edges[idx]
    ends = edges[idx + 1]
    labels = np.asarray([f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(starts, ends)], dtype=object)
    return labels, starts, ends


def collect_samples(monkeys: list[str], experiments: list[str], base_dir: str) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    rows = []
    for experiment in experiments:
        print(f"\n[INFO] Processing {experiment}")
        for monkey in monkeys:
            all_trials, _correct, _incorrect, _training, _channels, _nb_channels, pkl_files, _ai = load_files(
                experiment,
                monkey,
                base_dir,
            )
            for session_index, trials in enumerate(all_trials, start=1):
                session_file = Path(pkl_files[session_index - 1]).name if session_index - 1 < len(pkl_files) else ""
                session_uid = f"{monkey}|{experiment}|{session_index}|{session_file}"
                n_session = 0
                for trial in trials:
                    try:
                        ai_factor = float(_get(trial, "aiVelocityFactor", np.nan))
                    except Exception:
                        ai_factor = np.nan
                    if ai_factor != 1.0:
                        continue
                    recs = _get(trial, "aiVelocities", []) or []
                    if len(recs) < 3:
                        continue
                    inputs = np.vstack([_vec3(rec.get("Input")) for rec in recs])
                    outputs = np.vstack([_vec3(rec.get("Output")) for rec in recs])
                    output_ts = np.asarray([_as_scalar(rec.get("OutputTimestamp")) for rec in recs], dtype=float)
                    entropy = np.asarray([_as_scalar(rec.get("EntropyLb")) for rec in recs], dtype=float)
                    alpha = _robust_alpha(entropy)
                    bci = inputs[:, [0, 2]]
                    executed = outputs[:, [0, 2]]
                    assistance_vec = executed - bci
                    assistance_mag = np.linalg.norm(assistance_vec, axis=1)
                    bci_speed = np.linalg.norm(bci, axis=1)
                    exec_speed = np.linalg.norm(executed, axis=1)
                    override_mask = _explicit_override_mask(trial, output_ts)
                    valid = (
                        np.isfinite(alpha)
                        & np.isfinite(bci).all(axis=1)
                        & np.isfinite(executed).all(axis=1)
                        & (alpha > 0)
                        & (bci_speed > MIN_SPEED)
                        & (exec_speed > MIN_SPEED)
                    )
                    if not np.any(valid):
                        continue
                    labels, starts, ends = _alpha_bin(alpha)
                    idxs = np.where(valid)[0]
                    for i in idxs:
                        rows.append({
                            "monkey": monkey,
                            "experiment": experiment,
                            "session_index": session_index,
                            "session_file": session_file,
                            "session_uid": session_uid,
                            "trial_id": _trial_id(trial),
                            "sample_index": int(i),
                            "answer": _get(trial, "answer", np.nan),
                            "alpha": float(alpha[i]),
                            "alpha_bin": str(labels[i]),
                            "alpha_bin_start": float(starts[i]),
                            "alpha_bin_end": float(ends[i]),
                            "assistance_magnitude": float(assistance_mag[i]),
                            "is_safety_override_plus_pad": bool(override_mask[i]),
                            "bci_speed": float(bci_speed[i]),
                            "executed_speed": float(exec_speed[i]),
                        })
                        n_session += 1
                if n_session:
                    print(f"  {monkey} session {session_index}: {n_session} samples")
    return pd.DataFrame(rows)


def summarize_bins(samples: pd.DataFrame, min_samples: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows = []
    example_rows = []
    for alpha_bin, group in samples.groupby("alpha_bin", sort=True):
        if len(group) < min_samples:
            continue
        q = group["assistance_magnitude"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        bin_rows.append({
            "alpha_bin": alpha_bin,
            "alpha_bin_start": float(group["alpha_bin_start"].iloc[0]),
            "alpha_bin_end": float(group["alpha_bin_end"].iloc[0]),
            "n_samples": int(len(group)),
            "n_trials": int(group[["session_uid", "trial_id"]].drop_duplicates().shape[0]),
            "n_sessions": int(group["session_uid"].nunique()),
            "mean_alpha": float(group["alpha"].mean()),
            "mean_assistance_magnitude": float(group["assistance_magnitude"].mean()),
            "sd_assistance_magnitude": float(group["assistance_magnitude"].std(ddof=1)),
            "p10_assistance_magnitude": float(q.loc[0.10]),
            "p25_assistance_magnitude": float(q.loc[0.25]),
            "median_assistance_magnitude": float(group["assistance_magnitude"].median()),
            "p75_assistance_magnitude": float(q.loc[0.75]),
            "p90_assistance_magnitude": float(q.loc[0.90]),
            "p90_minus_p10_assistance_magnitude": float(q.loc[0.90] - q.loc[0.10]),
            "iqr_assistance_magnitude": float(q.loc[0.75] - q.loc[0.25]),
        })
        ranked = group.copy()
        ranked["assistance_tertile"] = pd.qcut(
            ranked["assistance_magnitude"],
            q=3,
            labels=["low", "middle", "high"],
            duplicates="drop",
        )
        for tertile, sub in ranked.groupby("assistance_tertile", observed=True):
            example_rows.append({
                "alpha_bin": alpha_bin,
                "assistance_tertile": str(tertile),
                "n_samples": int(len(sub)),
                "mean_alpha": float(sub["alpha"].mean()),
                "p10_assistance_magnitude": float(sub["assistance_magnitude"].quantile(0.10)),
                "median_assistance_magnitude": float(sub["assistance_magnitude"].median()),
                "p90_assistance_magnitude": float(sub["assistance_magnitude"].quantile(0.90)),
                "mean_assistance_magnitude": float(sub["assistance_magnitude"].mean()),
            })
    return pd.DataFrame(bin_rows), pd.DataFrame(example_rows)


def summarize_variability(bins: pd.DataFrame, label: str) -> pd.DataFrame:
    valid = bins.copy()
    valid = valid[pd.to_numeric(valid["n_samples"], errors="coerce") > 0]
    return pd.DataFrame([{
        "analysis": label,
        "n_alpha_bins": int(len(valid)),
        "median_bin_p90_minus_p10": float(valid["p90_minus_p10_assistance_magnitude"].median()),
        "min_bin_p90_minus_p10": float(valid["p90_minus_p10_assistance_magnitude"].min()),
        "max_bin_p90_minus_p10": float(valid["p90_minus_p10_assistance_magnitude"].max()),
        "median_bin_iqr": float(valid["iqr_assistance_magnitude"].median()),
        "min_bin_iqr": float(valid["iqr_assistance_magnitude"].min()),
        "max_bin_iqr": float(valid["iqr_assistance_magnitude"].max()),
    }])


def plot_example(bins: pd.DataFrame) -> None:
    out_dir = PROJECT_ROOT / "figures" / "Extended Data Figure 3" / "Same alpha assistance"
    out_dir.mkdir(parents=True, exist_ok=True)
    if bins.empty:
        return
    plot_df = bins.sort_values("alpha_bin_start")
    x = plot_df["mean_alpha"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.fill_between(
        x,
        plot_df["p10_assistance_magnitude"].to_numpy(dtype=float),
        plot_df["p90_assistance_magnitude"].to_numpy(dtype=float),
        color="#8fb6b2",
        alpha=0.35,
        linewidth=0,
        label="10th-90th percentile",
    )
    ax.fill_between(
        x,
        plot_df["p25_assistance_magnitude"].to_numpy(dtype=float),
        plot_df["p75_assistance_magnitude"].to_numpy(dtype=float),
        color="#23746f",
        alpha=0.32,
        linewidth=0,
        label="IQR",
    )
    ax.plot(x, plot_df["median_assistance_magnitude"], color="#1f4f4b", linewidth=2.0, label="Median")
    ax.set_xlim(ALPHA_FLOOR, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["alpha_bin"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Prior-confidence bin")
    ax.set_ylabel("Assistance magnitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(out_dir / f"same_alpha_assistance_distribution_by_alpha.{ext}", dpi=300)
    plt.close(fig)


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0.0:
        return "<1e-300"
    return f"{value:.3g}"


def write_statement(samples: pd.DataFrame, bins: pd.DataFrame, including_override_bins: pd.DataFrame, variability: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "tables" / "same_alpha_assistance_statement.md"
    var_primary = variability[variability["analysis"] == "excluding safety override"].iloc[0]
    var_all = variability[variability["analysis"] == "including safety override"].iloc[0]
    override_fraction = float(samples["is_safety_override_plus_pad"].mean()) if "is_safety_override_plus_pad" in samples else np.nan
    lines = [
        "# Same Alpha, Different Assistance",
        "",
        "Assistance magnitude was computed as ||v_exec - v_BCI|| for shared-control samples.",
        "The primary analysis excludes explicit safety-override samples plus a 150-ms buffer because those samples intentionally suppress BCI influence.",
        "",
        f"The collected table contains {len(samples):,} valid shared-control samples from {samples[['session_uid', 'trial_id']].drop_duplicates().shape[0]:,} trials and {samples['session_uid'].nunique()} sessions.",
        f"Safety-override plus pad samples were {100.0 * override_fraction:.1f}% of valid samples and were excluded from the primary distribution summary.",
        f"After exclusion, {int(bins['n_samples'].sum()):,} samples remained across {len(bins)} populated alpha bins of width {ALPHA_BIN_WIDTH:.2f}.",
        f"Across alpha bins, the median within-bin 90th-10th percentile range of assistance magnitude was {var_primary['median_bin_p90_minus_p10']:.3f} (range {var_primary['min_bin_p90_minus_p10']:.3f}-{var_primary['max_bin_p90_minus_p10']:.3f}).",
        f"Including safety-override samples gave the same qualitative result: median within-bin 90th-10th percentile range {var_all['median_bin_p90_minus_p10']:.3f}.",
        "",
        "Reviewer-facing summary:",
        "",
        "Even within narrow prior-confidence bins, assistance magnitude was broadly distributed. Thus alpha alone does not determine the amount of correction: alpha indexes prior concentration, while the executed correction also depends on the relationship between the decoded BCI command and the controller-favored command.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    base_dir: str,
    monkeys: list[str],
    experiments: list[str],
    samples_input: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(samples_input) if samples_input else collect_samples(monkeys, experiments, base_dir)
    if samples.empty:
        raise RuntimeError("No shared-control aiVelocities samples were found.")
    labels, starts, ends = _alpha_bin(pd.to_numeric(samples["alpha"], errors="coerce").to_numpy(dtype=float))
    samples["alpha_bin"] = labels
    samples["alpha_bin_start"] = starts
    samples["alpha_bin_end"] = ends
    if "is_safety_override_plus_pad" not in samples:
        samples["is_safety_override_plus_pad"] = False
    samples["is_safety_override_plus_pad"] = samples["is_safety_override_plus_pad"].astype(str).str.lower().isin(["true", "1", "yes"])
    primary_samples = samples[~samples["is_safety_override_plus_pad"]].copy()
    bins, tertiles = summarize_bins(primary_samples)
    including_override_bins, including_override_tertiles = summarize_bins(samples)
    variability = pd.concat(
        [
            summarize_variability(bins, "excluding safety override"),
            summarize_variability(including_override_bins, "including safety override"),
        ],
        ignore_index=True,
    )
    samples.to_csv(tables_dir / "same_alpha_assistance_samples.csv", index=False)
    primary_samples.to_csv(tables_dir / "same_alpha_assistance_samples_excluding_override.csv", index=False)
    bins.to_csv(tables_dir / "same_alpha_assistance_by_alpha_bin.csv", index=False)
    including_override_bins.to_csv(tables_dir / "same_alpha_assistance_by_alpha_bin_including_override.csv", index=False)
    tertiles.to_csv(tables_dir / "same_alpha_assistance_by_assistance_tertile.csv", index=False)
    including_override_tertiles.to_csv(tables_dir / "same_alpha_assistance_by_assistance_tertile_including_override.csv", index=False)
    variability.to_csv(tables_dir / "same_alpha_assistance_variability_summary.csv", index=False)
    plot_example(bins)
    write_statement(samples, bins, including_override_bins, variability)
    return samples, bins, variability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantify same-alpha/different-assistance examples.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--samples-input", default=None, help="Optional existing same_alpha_assistance_samples.csv path.")
    args = parser.parse_args(argv)
    run_analysis(load_base_dir(args.base_dir), args.monkeys, args.experiments, samples_input=args.samples_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
