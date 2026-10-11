"""Build revised manuscript Figure 3 success-rate panels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures" / "Figure 3"
TABLE_DIR = PROJECT_ROOT / "analysis_results" / "success_rate_verification"
SESSION_ROWS = TABLE_DIR / "success_rate_audit_session_condition_rows.csv"
COMPARISON_CSV = TABLE_DIR / "figure3_success_rate_comparison.csv"
SOURCE_CSV = TABLE_DIR / "figure3_success_rate_source_rows.csv"
CHANCE_CSV = TABLE_DIR / "figure3_success_rate_chance_estimates.csv"
CHANCE_METADATA_JSON = TABLE_DIR / "figure3_success_rate_chance_metadata.json"

SHARED_COLOR = "#009E73"
BCI_COLOR = "#CC79A7"
CHANCE_COLOR = "0.45"

PANELS = [
    {
        "panel": "a",
        "title": "Fixed Obstacle",
        "experiment": "AI Obstacle",
        "monkeys": ["Monkey 1", "Monkey 2"],
    },
    {
        "panel": "b",
        "title": "Appearing Obstacle",
        "experiment": "AI Appearing Obstacle",
        "monkeys": ["Monkey 1", "Monkey 2"],
    },
    {
        "panel": "c",
        "title": "Appearing Obstacle - 6 months follow-up",
        "experiment": "AI Appearing Obstacle 2",
        "monkeys": ["Monkey 1"],
    },
    {
        "panel": "d",
        "title": "Respawn",
        "experiment": "AI Respawn",
        "monkeys": ["Monkey 1"],
    },
]

MANUSCRIPT_ANNOTATIONS = {
    ("Monkey 1", "AI Obstacle"): {"shared_mean": 81, "shared_sd": 8, "bci_mean": 49, "bci_sd": 10, "chance": 14},
    ("Monkey 2", "AI Obstacle"): {"shared_mean": 61, "shared_sd": 9, "bci_mean": 34, "bci_sd": 8, "chance": 18},
    ("Monkey 1", "AI Appearing Obstacle"): {"shared_mean": 67, "shared_sd": 16, "bci_mean": 39, "bci_sd": 15, "chance": 18},
    ("Monkey 2", "AI Appearing Obstacle"): {"shared_mean": 56, "shared_sd": 9, "bci_mean": 22, "bci_sd": 10, "chance": 21},
    ("Monkey 1", "AI Appearing Obstacle 2"): {"shared_mean": 80, "shared_sd": 10, "bci_mean": 57, "bci_sd": 12, "chance": 17},
    ("Monkey 1", "AI Respawn"): {"shared_mean": 67, "shared_sd": 10, "bci_mean": 80, "bci_sd": 8, "chance": 24},
}


def ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_dir(cli_base_dir: str | None, config: dict | None = None) -> str:
    if cli_base_dir:
        return cli_base_dir
    if config is None:
        config = load_config()
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def config_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{key} must be an integer in config.yaml.") from exc


def get_attr(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def trial_id(trial):
    for key in ("trial", "trial_id", "trialId"):
        value = get_attr(trial, key, None)
        if value is not None:
            return value
    return np.nan


def ai_factor(trial) -> float:
    try:
        return float(get_attr(trial, "aiVelocityFactor"))
    except Exception:
        return np.nan


def condition_included(experiment: str, trial) -> bool:
    if experiment == "AI Obstacle":
        return True
    if experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
        return get_attr(trial, "obstacleAppearedTime", None) is not None
    if experiment == "AI Respawn":
        target_jump = get_attr(trial, "targetJumpPosition", None)
        try:
            return not np.isnan(target_jump).any()
        except Exception:
            return target_jump is not None
    return True


def target_position_2d(trial) -> tuple[float, float] | None:
    target = get_attr(trial, "targetPosition", None)
    if target is None:
        return None
    arr = np.asarray(target, dtype=float)
    if arr.size < 3:
        return None
    return float(arr[0]), float(arr[2])


def trajectory_2d(trial) -> np.ndarray | None:
    traj = get_attr(trial, "avatarTrajectory", None)
    if traj is None:
        return None
    try:
        x = np.asarray(traj["x"], dtype=float)
        z = np.asarray(traj["z"], dtype=float)
    except Exception:
        return None
    if x.size == 0 or z.size == 0 or x.size != z.size:
        return None
    return np.column_stack([x, z])


def obstacle_2d(trial) -> tuple[float, float] | None:
    obstacle = get_attr(trial, "obstaclePosition", None)
    if obstacle is None:
        return None
    arr = np.asarray(obstacle, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[[0, 2]])):
        return None
    return float(arr[0]), float(arr[2])


def dwell_success(traj: np.ndarray, target: tuple[float, float], window_size: float = 4.2, min_consecutive: int = 10) -> bool:
    half = window_size / 2.0
    count = 0
    tx, tz = target
    for x, z in traj:
        if tx - half <= x <= tx + half and tz - half <= z <= tz + half:
            count += 1
            if count >= min_consecutive:
                return True
        else:
            count = 0
    return False


def collision(traj: np.ndarray, obstacle: tuple[float, float] | None, radius: float = 0.7, min_consecutive: int = 5) -> bool:
    if obstacle is None:
        return False
    center = np.asarray(obstacle, dtype=float)
    dist2 = np.sum((traj - center) * (traj - center), axis=1)
    inside = dist2 < radius * radius
    if min_consecutive <= 1:
        return bool(np.any(inside))
    run = 0
    for value in inside:
        run = run + 1 if value else 0
        if run >= min_consecutive:
            return True
    return False


def chance_level_for_trials(trials: list, n_permutations: int, seed: int) -> dict:
    """Estimate chance by reassigning observed joint target-obstacle pairs.

    Each included trial keeps its recorded trajectory. For every permutation, it
    receives a random target-obstacle pair sampled from the same task/monkey
    condition. Success requires 10 consecutive samples inside the 4.2-unit
    square target window and, when an obstacle exists, no 5-consecutive-sample
    collision with the assigned obstacle.
    """
    rows = []
    for trial in trials:
        traj = trajectory_2d(trial)
        target = target_position_2d(trial)
        if traj is None or target is None:
            continue
        rows.append(
            {
                "trajectory": traj,
                "target": target,
                "obstacle": obstacle_2d(trial),
                "answer_success": get_attr(trial, "answer", None) == 1,
            }
        )
    if not rows:
        return {
            "chance_rate": np.nan,
            "chance_sd_across_permutations": np.nan,
            "observed_recomputed_rate": np.nan,
            "answer_success_rate": np.nan,
            "n_trials": 0,
            "n_target_obstacle_pairs": 0,
            "n_unique_targets": 0,
            "n_unique_obstacles": 0,
        }

    targets = [row["target"] for row in rows]
    pairs = [(row["target"], row["obstacle"]) for row in rows]
    unique_targets = sorted(set(targets))
    target_to_col = {target: j for j, target in enumerate(unique_targets)}
    dwell_by_target = np.zeros((len(rows), len(unique_targets)), dtype=bool)
    for i, row in enumerate(rows):
        for j, target in enumerate(unique_targets):
            dwell_by_target[i, j] = dwell_success(row["trajectory"], target)

    if n_permutations <= 0:
        raise ValueError("--chance-permutations / figure3_chance_permutations must be positive.")

    unique_obstacles = sorted(
        {obstacle for obstacle in (row["obstacle"] for row in rows) if obstacle is not None}
    )
    obstacle_to_col = {obstacle: j for j, obstacle in enumerate(unique_obstacles)}
    collision_by_obstacle = np.zeros((len(rows), len(unique_obstacles)), dtype=bool)
    for i, row in enumerate(rows):
        for j, obstacle in enumerate(unique_obstacles):
            collision_by_obstacle[i, j] = collision(row["trajectory"], obstacle)

    success_by_pair = np.zeros((len(rows), len(pairs)), dtype=bool)
    for pair_index, (target, obstacle) in enumerate(pairs):
        target_success = dwell_by_target[:, target_to_col[target]]
        if obstacle is None:
            success_by_pair[:, pair_index] = target_success
        else:
            success_by_pair[:, pair_index] = target_success & ~collision_by_obstacle[:, obstacle_to_col[obstacle]]

    original_pair_success = np.array(
        [
            success_by_pair[i, i]
            for i in range(len(rows))
        ],
        dtype=bool,
    )

    rng = np.random.default_rng(seed)
    sampled_pair_indices = rng.integers(0, len(pairs), size=(len(rows), n_permutations))
    trial_indices = np.arange(len(rows))[:, None]
    null_success = success_by_pair[trial_indices, sampled_pair_indices]
    null_rates = null_success.mean(axis=0)

    return {
        "chance_rate": float(null_rates.mean()),
        "chance_sd_across_permutations": float(null_rates.std(ddof=1)) if n_permutations > 1 else np.nan,
        "observed_recomputed_rate": float(original_pair_success.mean()),
        "answer_success_rate": float(np.mean([row["answer_success"] for row in rows])),
        "n_trials": int(len(rows)),
        "n_target_obstacle_pairs": int(len(pairs)),
        "n_unique_targets": int(len(unique_targets)),
        "n_unique_obstacles": int(len(unique_obstacles)),
    }


def load_trials_for_condition(base_dir: str, monkey: str, experiment: str) -> list:
    ensure_import_path()
    from src.load import load_data

    data_dir = Path(base_dir) / monkey / experiment
    if not data_dir.exists():
        return []
    trials = []
    for pkl_file in sorted(data_dir.glob("*_trials.pkl")):
        loaded = load_data(str(pkl_file))[0]
        for trial in loaded:
            if ai_factor(trial) in (0.0, 1.0) and condition_included(experiment, trial):
                trials.append(trial)
    return trials


def ensure_success_audit(base_dir: str) -> None:
    metadata_path = TABLE_DIR / "success_rate_audit_metadata.json"
    if SESSION_ROWS.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        if str(metadata.get("base_dir")) == str(base_dir):
            return
    elif SESSION_ROWS.exists():
        return
    ensure_import_path()
    from scripts.audit_success_rate_verification import run

    run(
        base_dir,
        ["Monkey 1", "Monkey 2"],
        ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2", "AI Respawn"],
    )


def session_rates(session_rows: pd.DataFrame, monkey: str, experiment: str) -> pd.DataFrame:
    sub = session_rows[
        session_rows["monkey"].eq(monkey)
        & session_rows["experiment"].eq(experiment)
    ].copy()
    pivot = sub.pivot_table(
        index=["session_index", "source_file"],
        columns="condition",
        values="success_rate",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.sort_values("session_index").reset_index(drop=True)
    return pivot


def rounded_mean_sd(values: np.ndarray) -> tuple[float, float, int, int]:
    values = np.asarray(values, dtype=float)
    mean_pct = float(np.nanmean(values) * 100)
    sd_pct = float(np.nanstd(values, ddof=1) * 100)
    return mean_pct, sd_pct, int(np.round(mean_pct)), int(np.round(sd_pct))


def plot_panel(ax, rows: pd.DataFrame, chance: float, title: str, show_xlabel: bool = True) -> dict:
    x = np.arange(1, len(rows) + 1)
    shared = rows["Shared-control"].to_numpy(dtype=float)
    bci = rows["BCI-only"].to_numpy(dtype=float)
    shared_mean_full, shared_sd_full, shared_mean, shared_sd = rounded_mean_sd(shared)
    bci_mean_full, bci_sd_full, bci_mean, bci_sd = rounded_mean_sd(bci)
    chance_pct = int(np.round(chance * 100)) if np.isfinite(chance) else np.nan

    ax.plot(x, shared * 100, marker="o", color=SHARED_COLOR, linewidth=2.0, markersize=5.5)
    ax.plot(x, bci * 100, marker="o", color=BCI_COLOR, linewidth=2.0, markersize=5.5)
    ax.axhline(shared_mean, color=SHARED_COLOR, linestyle="--", linewidth=1.2, alpha=0.55)
    ax.axhline(bci_mean, color=BCI_COLOR, linestyle="--", linewidth=1.2, alpha=0.55)
    if np.isfinite(chance):
        ax.axhline(chance * 100, color=CHANCE_COLOR, linestyle=":", linewidth=1.2)

    ax.set_title(title, fontsize=10, weight="bold", pad=8)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, len(rows) + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(i)) for i in x], fontsize=8)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Success rate (%)", fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Session", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("0.45")

    labels = [
        f"Shared-control: {shared_mean} \u00b1 {shared_sd}%",
        f"BCI-only: {bci_mean} \u00b1 {bci_sd}%",
    ]
    if np.isfinite(chance):
        labels.append(f"Chance level: {chance_pct}%")
    handles = [
        plt.Line2D([0], [0], color=SHARED_COLOR, marker="o", linewidth=2, markersize=5.5),
        plt.Line2D([0], [0], color=BCI_COLOR, marker="o", linewidth=2, markersize=5.5),
    ]
    if np.isfinite(chance):
        handles.append(plt.Line2D([0], [0], color=CHANCE_COLOR, linestyle=":", linewidth=1.2))
    ax.legend(handles, labels, loc="lower right" if bci_mean < 65 else "upper right", fontsize=8, frameon=True)

    return {
        "n_sessions": int(len(rows)),
        "aggregation_method": "mean_and_sample_sd_of_session_success_rates",
        "rounding_rule": "numpy_round_to_nearest_even_after_percent_conversion",
        "sd_ddof": 1,
        "shared_mean_pct_full": shared_mean_full,
        "shared_mean": shared_mean,
        "shared_sd_pct_full": shared_sd_full,
        "shared_sd": shared_sd,
        "bci_mean_pct_full": bci_mean_full,
        "bci_mean": bci_mean,
        "bci_sd_pct_full": bci_sd_full,
        "bci_sd": bci_sd,
        "chance_pct_full": float(chance * 100) if np.isfinite(chance) else np.nan,
        "chance": chance_pct,
        "shared_values_pct": ";".join(f"{v:.6g}" for v in shared * 100),
        "bci_values_pct": ";".join(f"{v:.6g}" for v in bci * 100),
    }


def build_figure(base_dir: str, n_permutations: int, seed: int) -> dict[str, Path]:
    ensure_success_audit(base_dir)
    session_rows = pd.read_csv(SESSION_ROWS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(10.4, 10.2), dpi=180)
    fig.subplots_adjust(hspace=0.47, wspace=0.27, left=0.08, right=0.98, top=0.96, bottom=0.07)

    panel_axes = {
        ("a", "Monkey 1"): axes[0, 0],
        ("a", "Monkey 2"): axes[0, 1],
        ("b", "Monkey 1"): axes[1, 0],
        ("b", "Monkey 2"): axes[1, 1],
        ("c", "Monkey 1"): axes[2, 0],
        ("d", "Monkey 1"): axes[2, 1],
    }

    comparison_rows = []
    chance_rows = []
    source_rows = []
    for panel in PANELS:
        left_ax = panel_axes[(panel["panel"], panel["monkeys"][0])]
        left_ax.text(
            -0.105,
            1.115,
            f"{panel['panel']} {panel['title']}",
            transform=left_ax.transAxes,
            fontsize=11,
            weight="bold",
            ha="left",
            va="bottom",
        )
        for monkey in panel["monkeys"]:
            ax = panel_axes[(panel["panel"], monkey)]
            rows = session_rates(session_rows, monkey, panel["experiment"])
            if rows.empty:
                ax.axis("off")
                continue
            trials = load_trials_for_condition(base_dir, monkey, panel["experiment"])
            chance_result = chance_level_for_trials(trials, n_permutations=n_permutations, seed=seed)
            chance = chance_result["chance_rate"]
            computed = plot_panel(ax, rows, chance, monkey)
            key = (monkey, panel["experiment"])
            manuscript = MANUSCRIPT_ANNOTATIONS.get(key, {})
            chance_rows.append(
                {
                    "panel": panel["panel"],
                    "monkey": monkey,
                    "experiment": panel["experiment"],
                    "method": "joint_target_obstacle_monte_carlo",
                    "success_definition": "10 consecutive samples in 4.2-unit target window and no 5-consecutive-sample obstacle collision within radius 0.7",
                    "included_trials": "aiVelocityFactor in {0,1} plus task-specific inclusion rule",
                    "n_permutations": int(n_permutations),
                    "seed": int(seed),
                    **chance_result,
                    "chance_pct_full": float(chance_result["chance_rate"] * 100)
                    if np.isfinite(chance_result["chance_rate"])
                    else np.nan,
                    "chance_pct_display": computed["chance"],
                }
            )
            comparison = {
                "panel": panel["panel"],
                "panel_title": panel["title"],
                "monkey": monkey,
                "experiment": panel["experiment"],
                "chance_source": "joint_target_obstacle_monte_carlo",
                "chance_permutations": int(n_permutations),
                "chance_seed": int(seed),
                **computed,
            }
            for field, value in manuscript.items():
                comparison[f"manuscript_{field}"] = value
                comparison[f"delta_{field}"] = computed.get(field, np.nan) - value
            comparison_rows.append(comparison)
            out_rows = rows.copy()
            out_rows.insert(0, "panel", panel["panel"])
            out_rows.insert(1, "monkey", monkey)
            out_rows.insert(2, "experiment", panel["experiment"])
            source_rows.append(out_rows)

    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"figure3_success_rates_revised.{ext}", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(comparison_rows).to_csv(COMPARISON_CSV, index=False)
    pd.DataFrame(chance_rows).to_csv(CHANCE_CSV, index=False)
    CHANCE_METADATA_JSON.write_text(
        json.dumps(
            {
                "base_dir": base_dir,
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "chance_method": "joint target-obstacle Monte Carlo over included raw trials",
                "chance_permutations": int(n_permutations),
                "chance_seed": int(seed),
                "legacy_reference": (
                    "src.stats_tests.compute_chance_level shuffles targets in the "
                    "legacy success-rate pipeline; the revised Figure 3 builder uses "
                    "joint target-obstacle reassignment and records this file."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.concat(source_rows, ignore_index=True).to_csv(SOURCE_CSV, index=False)
    return {
        "svg": OUTPUT_DIR / "figure3_success_rates_revised.svg",
        "pdf": OUTPUT_DIR / "figure3_success_rates_revised.pdf",
        "png": OUTPUT_DIR / "figure3_success_rates_revised.png",
        "comparison": COMPARISON_CSV,
        "chance": CHANCE_CSV,
        "chance_metadata": CHANCE_METADATA_JSON,
        "source_rows": SOURCE_CSV,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--chance-permutations", type=int, default=None)
    parser.add_argument("--chance-seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    n_permutations = (
        int(args.chance_permutations)
        if args.chance_permutations is not None
        else config_int(config, "figure3_chance_permutations", 1000)
    )
    seed = (
        int(args.chance_seed)
        if args.chance_seed is not None
        else config_int(config, "figure3_chance_seed", 42)
    )
    paths = build_figure(load_base_dir(args.base_dir, config), n_permutations, seed)
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
