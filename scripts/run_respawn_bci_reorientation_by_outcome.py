"""
Compare post-jump decoded BCI-velocity reorientation across Respawn outcomes.

Unlike run_respawn_reorientation_lag.py, which uses the shared-control
aiVelocities input/output log and is therefore restricted to AI trials, this
script uses trial.modelVelocity plus avatarTrajectory. That makes the decoded
BCI-command reorientation latency comparable across:

- correct shared-control trials
- incorrect shared-control trials
- correct BCI-only trials
- incorrect BCI-only trials

Final manuscript Respawn analyses explicitly use answer code 1 for successful
trials and codes {3,5,6} for unsuccessful experimental outcomes. Codes {4,9,10}
from the broader loader pool are excluded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
EXPERIMENT = "AI Respawn"
WINDOW_MS = (-1000.0, 2000.0)
SUSTAIN_MS = 200.0
MIN_SPEED = 0.05


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


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


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_scalar(value) -> float:
    if value is None:
        return np.nan
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return np.nan


def _trial_id(trial) -> int | float:
    for key in ("trial", "trial_id", "trialId", "trialIndex"):
        value = _get(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return np.nan
    return np.nan


def _vec3(value) -> np.ndarray:
    if value is None:
        return np.full(3, np.nan)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(3, np.nan)
    out = np.full(3, np.nan)
    out[: min(3, arr.size)] = arr[:3]
    return out


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(norm, eps, None)


def _to_relative_ms(sample_times: np.ndarray, event_time: float) -> np.ndarray:
    rel = np.asarray(sample_times, dtype=float) - float(event_time)
    finite = sample_times[np.isfinite(sample_times)]
    if finite.size >= 3:
        diffs = np.diff(np.sort(finite))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size and np.nanmedian(diffs) < 1.0:
            rel *= 1000.0
    return rel


def _old_new_evidence_2d(vel_xz: np.ndarray, pos_xz: np.ndarray, old_target, new_target) -> np.ndarray:
    old_xz = _vec3(old_target)[[0, 2]]
    new_xz = _vec3(new_target)[[0, 2]]
    vel_xz = np.asarray(vel_xz, dtype=float)
    pos_xz = np.asarray(pos_xz, dtype=float)
    out = np.full(len(vel_xz), np.nan)
    valid = (
        np.isfinite(old_xz).all()
        & np.isfinite(new_xz).all()
        & np.isfinite(vel_xz).all(axis=1)
        & np.isfinite(pos_xz).all(axis=1)
    )
    if not np.any(valid):
        return out
    vhat = _unit(vel_xz[valid])
    u_old = _unit(old_xz[None, :] - pos_xz[valid])
    u_new = _unit(new_xz[None, :] - pos_xz[valid])
    out[valid] = np.sum(vhat * u_new, axis=1) - np.sum(vhat * u_old, axis=1)
    speed = np.linalg.norm(vel_xz, axis=1)
    out[speed < MIN_SPEED] = np.nan
    return out


def _sustained_crossing_time(t_rel_ms: np.ndarray, evidence: np.ndarray, sustain_ms: float = SUSTAIN_MS) -> float:
    valid = np.isfinite(t_rel_ms) & np.isfinite(evidence) & (t_rel_ms >= 0)
    if not np.any(valid):
        return np.nan
    t = np.asarray(t_rel_ms[valid], dtype=float)
    e = np.asarray(evidence[valid], dtype=float)
    order = np.argsort(t)
    t = t[order]
    e = e[order]
    ok = e > 0
    for start in np.flatnonzero(ok):
        end = start
        while end + 1 < len(ok) and ok[end + 1]:
            end += 1
        if t[end] - t[start] >= sustain_ms:
            return float(t[start])
    return np.nan


def _window_mean(t_rel_ms: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    mask = (t_rel_ms >= lo) & (t_rel_ms < hi) & np.isfinite(values)
    return float(np.nanmean(values[mask])) if np.any(mask) else np.nan


def extract_modelvelocity_reorientation(trial, monkey: str, session_index: int, session_file: str) -> dict | None:
    jump_time = _as_scalar(_get(trial, "targetJumpTime"))
    old_target = _get(trial, "targetPosition")
    new_target = _get(trial, "targetJumpPosition")
    mv = _get(trial, "modelVelocity", None)
    traj = _get(trial, "avatarTrajectory", None)
    if not np.isfinite(jump_time) or mv is None or traj is None:
        return None

    try:
        vx = np.asarray(mv["vx"], dtype=float)
        vz = np.asarray(mv["vz"], dtype=float)
        xs = np.asarray(traj["x"], dtype=float)
        zs = np.asarray(traj["z"], dtype=float)
        ts = np.asarray(traj.get("time", np.arange(len(xs)) * 0.05), dtype=float)
        ai_factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None

    n = min(len(vx), len(vz), len(xs), len(zs), len(ts))
    if n < 3:
        return None
    vx, vz, xs, zs, ts = vx[:n], vz[:n], xs[:n], zs[:n], ts[:n]
    t_rel_ms = _to_relative_ms(ts, jump_time)
    keep = (t_rel_ms >= WINDOW_MS[0]) & (t_rel_ms <= WINDOW_MS[1])
    if not np.any(keep):
        return None

    vel = np.column_stack([vx[keep], vz[keep]])
    pos = np.column_stack([xs[keep], zs[keep]])
    t_rel_ms = t_rel_ms[keep]
    evidence = _old_new_evidence_2d(vel, pos, old_target, new_target)
    t_bci = _sustained_crossing_time(t_rel_ms, evidence)
    answer = _get(trial, "answer", np.nan)
    condition = "AI" if ai_factor == 1.0 else "No AI" if ai_factor == 0.0 else "unknown"
    outcome = "correct" if answer == 1 else "incorrect"

    return {
        "monkey": monkey,
        "experiment": EXPERIMENT,
        "session_index": session_index,
        "session_file": session_file,
        "session_uid": f"{monkey}|{EXPERIMENT}|{session_index}|{session_file}",
        "trial": _trial_id(trial),
        "answer": answer,
        "condition": condition,
        "outcome": outcome,
        "group": f"{condition}_{outcome}".replace(" ", "_").lower(),
        "n_samples": int(len(t_rel_ms)),
        "t_bci_modelvelocity_ms": t_bci,
        "bci_crossed_within_window": bool(np.isfinite(t_bci)),
        "e_bci_early_0_500": _window_mean(t_rel_ms, evidence, 0.0, 500.0),
        "e_bci_early_200_800": _window_mean(t_rel_ms, evidence, 200.0, 800.0),
    }


def collect_rows(monkeys: list[str], base_dir: str) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files
    from src.respawn_inclusion import (
        RESPAWN_ELIGIBLE_ANSWERS,
        iter_final_respawn_trials,
    )

    rows = []
    for monkey in monkeys:
        all_trials, _all_correct, _all_incorrect, _training, _channels, _nb_channels, pkl_files, _ai = load_files(
            EXPERIMENT,
            monkey,
            base_dir,
        )
        last_session = None
        session_start_count = len(rows)
        for trial, session_index, session_file in iter_final_respawn_trials(all_trials, pkl_files, monkey, EXPERIMENT):
            if last_session is not None and session_index != last_session:
                print(f"{monkey} session {last_session}: collected {len(rows) - session_start_count} eligible rows")
                session_start_count = len(rows)
            last_session = session_index
            row = extract_modelvelocity_reorientation(trial, monkey, session_index, session_file)
            if row is not None:
                row["eligible_answer_code"] = True
                row["included_answer_codes"] = "{" + ",".join(str(x) for x in sorted(RESPAWN_ELIGIBLE_ANSWERS)) + "}"
                rows.append(row)
        if last_session is not None:
            print(f"{monkey} session {last_session}: collected {len(rows) - session_start_count} eligible rows")
    return pd.DataFrame(rows)


def summarize_groups(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    order = [
        ("AI", "correct"),
        ("AI", "incorrect"),
        ("No AI", "correct"),
        ("No AI", "incorrect"),
    ]
    for condition, outcome in order:
        group = rows[(rows["condition"] == condition) & (rows["outcome"] == outcome)].copy()
        lat = group["t_bci_modelvelocity_ms"].astype(float).dropna()
        crossed = group["bci_crossed_within_window"].astype(bool) if not group.empty else pd.Series(dtype=bool)
        records.append({
            "condition": condition,
            "outcome": outcome,
            "n_trials": int(len(group)),
            "n_crossed": int(crossed.sum()) if len(crossed) else 0,
            "crossing_fraction": float(crossed.mean()) if len(crossed) else np.nan,
            "failure_to_cross_fraction": float((~crossed).mean()) if len(crossed) else np.nan,
            "median_t_bci_modelvelocity_ms": float(np.nanmedian(lat)) if len(lat) else np.nan,
            "iqr_t_bci_modelvelocity_ms": (
                float(np.nanpercentile(lat, 75) - np.nanpercentile(lat, 25)) if len(lat) else np.nan
            ),
            "mean_e_bci_early_200_800": float(group["e_bci_early_200_800"].astype(float).mean()) if len(group) else np.nan,
        })
    return pd.DataFrame(records)


def compare_correct_incorrect(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for condition, group in rows.groupby("condition", dropna=False):
        correct = group.loc[group["outcome"] == "correct", "t_bci_modelvelocity_ms"].astype(float).dropna()
        incorrect = group.loc[group["outcome"] == "incorrect", "t_bci_modelvelocity_ms"].astype(float).dropna()
        if len(correct) and len(incorrect):
            test = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
            p_value = float(test.pvalue)
            stat = float(test.statistic)
        else:
            p_value = np.nan
            stat = np.nan
        records.append({
            "condition": condition,
            "n_correct_crossed": int(len(correct)),
            "n_incorrect_crossed": int(len(incorrect)),
            "correct_median_ms": float(np.nanmedian(correct)) if len(correct) else np.nan,
            "incorrect_median_ms": float(np.nanmedian(incorrect)) if len(incorrect) else np.nan,
            "incorrect_minus_correct_median_ms": (
                float(np.nanmedian(incorrect) - np.nanmedian(correct)) if len(correct) and len(incorrect) else np.nan
            ),
            "mannwhitney_u": stat,
            "p_value": p_value,
        })
    return pd.DataFrame(records)


def write_statement(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "tables" / "respawn_bci_reorientation_by_outcome_statement.md"
    lines = [
        "# Respawn Decoded BCI Reorientation by Outcome",
        "",
        "Post-jump decoded BCI-velocity reorientation latency was computed from `modelVelocity` for both shared-control and BCI-only Respawn trials.",
        f"Latency was defined as the first post-jump time at which old-vs-new target evidence remained positive for at least {SUSTAIN_MS:g} ms.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['condition']} {row['outcome']}: n={int(row['n_trials'])}, "
            f"crossed={100.0 * row['crossing_fraction']:.1f}%, "
            f"median latency={row['median_t_bci_modelvelocity_ms']:.1f} ms."
        )
    lines.append("")
    for _, row in comparison.iterrows():
        p_txt = "NA" if not np.isfinite(row["p_value"]) else f"{row['p_value']:.3g}"
        lines.append(
            f"- {row['condition']}: incorrect-minus-correct median latency difference = "
            f"{row['incorrect_minus_correct_median_ms']:.1f} ms; Mann-Whitney p={p_txt}."
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: pd.DataFrame) -> None:
    out_dir = PROJECT_ROOT / "figures" / "Extended Data Figure 3" / "Respawn BCI reorientation"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.condition}\n{r.outcome}" for r in summary.itertuples()]
    y = summary["median_t_bci_modelvelocity_ms"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.bar(np.arange(len(labels)), y, color=["#2f7f78", "#8f4f4a", "#4f6f9f", "#9a6a30"], width=0.7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Decoded BCI reorientation latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(out_dir / f"respawn_bci_reorientation_by_outcome.{ext}", dpi=300)
    plt.close(fig)


def run_respawn_bci_reorientation_by_outcome(monkeys: list[str], base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(monkeys, base_dir)
    if rows.empty:
        raise RuntimeError("No Respawn modelVelocity rows were collected.")
    summary = summarize_groups(rows)
    comparison = compare_correct_incorrect(rows)
    rows.to_csv(tables_dir / "respawn_bci_reorientation_by_outcome_trials.csv", index=False)
    summary.to_csv(tables_dir / "respawn_bci_reorientation_by_outcome_summary.csv", index=False)
    comparison.to_csv(tables_dir / "respawn_bci_reorientation_correct_vs_incorrect.csv", index=False)
    write_statement(summary, comparison)
    plot_summary(summary)
    return rows, summary, comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Respawn decoded BCI reorientation by AI condition and outcome.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    args = parser.parse_args(argv)
    base_dir = load_base_dir(args.base_dir)
    run_respawn_bci_reorientation_by_outcome(args.monkeys, base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
