"""
Audit the fraction of decoded-choice time steps excluded near obstacles.

The choice-state analysis excludes obstacle-task frames when the avatar center
is within 1.5 m of the obstacle center in the x-z plane. This script reports how
large that exclusion is per trial and per session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
OBSTACLE_WINDOW_RADIUS = 1.5
SPEED_THRESH = 0.1


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


def _trial_id(trial) -> int | float:
    for key in ("trial", "trial_id", "trialId"):
        value = _get(trial, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return np.nan
    return np.nan


def _include_trial(experiment: str, trial) -> bool:
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return _get(trial, "obstacleAppearedTime", None) is not None
    return True


def _trial_exclusion_row(
    trial,
    monkey: str,
    experiment: str,
    session_index: int,
    session_file: str,
    obstacle_radius: float,
    speed_thresh: float,
) -> dict | None:
    if not _include_trial(experiment, trial):
        return None

    mv = _get(trial, "modelVelocity", None)
    traj = _get(trial, "avatarTrajectory", None)
    obst = _get(trial, "obstaclePosition", None)
    if mv is None or traj is None or obst is None:
        return None

    try:
        vx = np.asarray(mv["vx"], dtype=float)
        vz = np.asarray(mv["vz"], dtype=float)
        xs = np.asarray(traj["x"], dtype=float)
        zs = np.asarray(traj["z"], dtype=float)
        ox, oz = float(obst[0]), float(obst[2])
        ai_factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None

    n = min(len(vx), len(vz), len(xs), len(zs))
    if n < 1:
        return None

    vx = vx[:n]
    vz = vz[:n]
    xs = xs[:n]
    zs = zs[:n]
    speed = np.sqrt(vx ** 2 + vz ** 2)
    finite_model = np.isfinite(speed) & np.isfinite(xs) & np.isfinite(zs)
    speed_valid = finite_model & (speed > speed_thresh)
    dist_obs = np.sqrt((xs - ox) ** 2 + (zs - oz) ** 2)
    excluded = speed_valid & np.isfinite(dist_obs) & (dist_obs <= obstacle_radius)

    n_all = int(finite_model.sum())
    n_speed_valid = int(speed_valid.sum())
    n_excluded = int(excluded.sum())
    condition = "AI" if ai_factor == 1.0 else "No AI" if ai_factor == 0.0 else "unknown"

    return {
        "monkey": monkey,
        "experiment": experiment,
        "session_index": session_index,
        "session_file": session_file,
        "session_uid": f"{monkey}|{experiment}|{session_index}|{session_file}",
        "trial_id": _trial_id(trial),
        "condition": condition,
        "success": int(_get(trial, "answer", None) == 1),
        "obstacle_window_radius_m": float(obstacle_radius),
        "speed_threshold": float(speed_thresh),
        "n_model_time_steps": n_all,
        "n_speed_valid_time_steps_before_obstacle_exclusion": n_speed_valid,
        "n_excluded_near_obstacle_time_steps": n_excluded,
        "fraction_excluded_of_speed_valid_time_steps": (
            float(n_excluded / n_speed_valid) if n_speed_valid > 0 else np.nan
        ),
        "fraction_excluded_of_all_model_time_steps": (
            float(n_excluded / n_all) if n_all > 0 else np.nan
        ),
    }


def collect_exclusion_rows(
    monkeys: list[str],
    experiments: list[str],
    base_dir: str,
    obstacle_radius: float = OBSTACLE_WINDOW_RADIUS,
    speed_thresh: float = SPEED_THRESH,
) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    rows = []
    for experiment in experiments:
        print(f"\n[INFO] Processing {experiment}")
        for monkey in monkeys:
            all_trials, *_rest, pkl_files, _ai_trials = load_files(experiment, monkey, base_dir)
            for session_index, trials in enumerate(all_trials):
                session_file = Path(pkl_files[session_index]).name if session_index < len(pkl_files) else ""
                for trial in trials:
                    row = _trial_exclusion_row(
                        trial,
                        monkey,
                        experiment,
                        session_index,
                        session_file,
                        obstacle_radius,
                        speed_thresh,
                    )
                    if row is not None:
                        rows.append(row)
                print(f"  {monkey} session {session_index}: collected {len(rows)} total rows")
    return pd.DataFrame(rows)


def summarize_sessions(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["monkey", "experiment", "session_index", "session_file", "session_uid", "condition"]
    for keys, group in trials.groupby(group_cols, dropna=False):
        n_before = int(group["n_speed_valid_time_steps_before_obstacle_exclusion"].sum())
        n_excl = int(group["n_excluded_near_obstacle_time_steps"].sum())
        n_all = int(group["n_model_time_steps"].sum())
        row = dict(zip(group_cols, keys))
        row.update({
            "n_trials": int(len(group)),
            "n_model_time_steps": n_all,
            "n_speed_valid_time_steps_before_obstacle_exclusion": n_before,
            "n_excluded_near_obstacle_time_steps": n_excl,
            "fraction_excluded_of_speed_valid_time_steps": (
                float(n_excl / n_before) if n_before > 0 else np.nan
            ),
            "fraction_excluded_of_all_model_time_steps": (
                float(n_excl / n_all) if n_all > 0 else np.nan
            ),
            "median_trial_fraction_excluded_of_speed_valid_time_steps": float(
                group["fraction_excluded_of_speed_valid_time_steps"].median()
            ),
            "p90_trial_fraction_excluded_of_speed_valid_time_steps": float(
                group["fraction_excluded_of_speed_valid_time_steps"].quantile(0.90)
            ),
            "max_trial_fraction_excluded_of_speed_valid_time_steps": float(
                group["fraction_excluded_of_speed_valid_time_steps"].max()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def write_statement(trials: pd.DataFrame, sessions: pd.DataFrame) -> None:
    out_path = PROJECT_ROOT / "tables" / "obstacle_choice_exclusion_statement.md"
    n_before = int(trials["n_speed_valid_time_steps_before_obstacle_exclusion"].sum())
    n_excl = int(trials["n_excluded_near_obstacle_time_steps"].sum())
    frac = n_excl / n_before if n_before > 0 else np.nan
    lines = [
        "# Obstacle Choice-State Exclusion Audit",
        "",
        f"The decoded-choice analysis excluded time steps when the avatar center was within {OBSTACLE_WINDOW_RADIUS:g} m of the obstacle center in the x-z plane, after applying the model-velocity speed threshold of {SPEED_THRESH:g}.",
        "",
        f"Across included obstacle trials, {n_excl:,} of {n_before:,} speed-valid model-velocity time steps were excluded ({100.0 * frac:.2f}%)." if np.isfinite(frac) else "No speed-valid model-velocity time steps were available.",
        f"Median trial-level excluded fraction: {100.0 * trials['fraction_excluded_of_speed_valid_time_steps'].median():.2f}%.",
        f"Median session-level excluded fraction: {100.0 * sessions['fraction_excluded_of_speed_valid_time_steps'].median():.2f}%.",
        f"Maximum session-level excluded fraction: {100.0 * sessions['fraction_excluded_of_speed_valid_time_steps'].max():.2f}%.",
        "",
        "See `obstacle_choice_exclusion_by_trial.csv` and `obstacle_choice_exclusion_by_session.csv` for the per-trial and per-session values.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_obstacle_exclusion_audit(
    monkeys: list[str],
    experiments: list[str],
    base_dir: str,
    obstacle_radius: float = OBSTACLE_WINDOW_RADIUS,
    speed_thresh: float = SPEED_THRESH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    trials = collect_exclusion_rows(monkeys, experiments, base_dir, obstacle_radius, speed_thresh)
    if trials.empty:
        raise RuntimeError("No obstacle exclusion rows were collected.")
    sessions = summarize_sessions(trials)
    trials.to_csv(tables_dir / "obstacle_choice_exclusion_by_trial.csv", index=False)
    sessions.to_csv(tables_dir / "obstacle_choice_exclusion_by_session.csv", index=False)
    write_statement(trials, sessions)
    return trials, sessions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit near-obstacle time-step exclusion fractions.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--obstacle-radius", type=float, default=OBSTACLE_WINDOW_RADIUS)
    parser.add_argument("--speed-thresh", type=float, default=SPEED_THRESH)
    args = parser.parse_args(argv)
    base_dir = load_base_dir(args.base_dir)
    run_obstacle_exclusion_audit(
        args.monkeys,
        args.experiments,
        base_dir,
        obstacle_radius=args.obstacle_radius,
        speed_thresh=args.speed_thresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
