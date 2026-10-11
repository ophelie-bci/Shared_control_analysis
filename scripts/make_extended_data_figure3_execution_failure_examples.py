"""
Build representative execution-failure trajectory examples for Extended Data
Figure 3.

The examples are selected from raw trial files and checked against the
canonical execution-failure criteria used in the trajectory analysis:
stuck at obstacle, target overshoot, insufficient target proximity, and
insufficient dwell time.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import target_mapping, target_to_obstacle_mapping


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "Extended Data Figure 3"
DEFAULT_OUTPUT_STEM = "extended_data_figure3_execution_failure_examples"
PUBLICATION_OUTPUT_STEM = "extended_data_figure3_execution_failure_examples_pub_layout"
TARGET_HALF_WIDTH = 2.10
DRAW_TARGET_WIDTH = 1.50
OBSTACLE_WIDTH = 0.90
AVATAR_RADIUS = 0.25
OBSTACLE_RADIUS = 0.45
STUCK_MARGIN = 0.20
MIN_DWELL_FRAMES = 10

TARGETS = {
    label: np.asarray(pos, dtype=float)[[0, 2]]
    for pos, label in target_mapping.items()
}


@dataclass(frozen=True)
class ExampleSpec:
    panel: str
    monkey: str
    experiment: str
    session_index: int
    session_file: str
    trial_id: int


@dataclass(frozen=True)
class TrialRecord:
    panel: str
    monkey: str
    experiment: str
    session_index: int
    session_file: str
    trial_id: int
    condition: str
    outcome: str
    target_label: str
    wrong_target_label: str
    points: np.ndarray
    obstacle_center: tuple[float, float]
    min_obstacle_distance_m: float
    final_distance_true_target_m: float
    final_distance_nearest_wrong_target_m: float
    duration_s: float
    selection_score: float
    max_consecutive_target_frames: int
    max_z_minus_target_z_m: float
    canonical_reason: str


EXAMPLES = [
    ExampleSpec(
        "Correct target acquisition",
        "Monkey 1",
        "AI Obstacle",
        3,
        "navdecodingsphereaiobstacle_Maui_20241113_1015_B_trials.pkl",
        24,
    ),
    ExampleSpec(
        "Blocked",
        "Monkey 1",
        "AI Appearing Obstacle",
        5,
        "navdecodingsphereaiappearingobstacle_Maui_20241121_1018_A_trials.pkl",
        80,
    ),
    ExampleSpec(
        "Target overshoot",
        "Monkey 1",
        "AI Obstacle",
        1,
        "navdecodingsphereaiobstacle_Maui_20241108_1039_B_trials.pkl",
        30,
    ),
    ExampleSpec(
        "Insufficient target proximity",
        "Monkey 2",
        "AI Appearing Obstacle",
        0,
        "navdecodingsphereaiappearingobstacle_Loki_20241209_1007_A_trials.pkl",
        86,
    ),
    ExampleSpec(
        "Insufficient dwell time",
        "Monkey 1",
        "AI Obstacle",
        3,
        "navdecodingsphereaiobstacle_Maui_20241113_1015_B_trials.pkl",
        40,
    ),
]

PANEL_TITLES = {
    "Correct target acquisition": "Successful acquisition",
    "Blocked": "Blocked at obstacle",
    "Target overshoot": "Target overshoot",
    "Insufficient target proximity": "Insufficient proximity",
    "Insufficient dwell time": "Insufficient dwell",
}

TRAJECTORY_COLOR = "#08509b"
TARGET_COLOR = "#008000"
OTHER_TARGET_COLOR = "#d9d9d9"
OTHER_TARGET_EDGE = "#a8a8a8"
OBSTACLE_COLOR = "#d95f02"


def load_base_dir(cli_base_dir: str | None = None) -> Path:
    if cli_base_dir:
        return Path(cli_base_dir)
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        base_dir = (yaml.safe_load(f) or {}).get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return Path(base_dir)


def _get(trial, key: str, default=None):
    return trial.get(key, default) if isinstance(trial, dict) else getattr(trial, key, default)


def _trial_id(trial) -> int | str:
    value = _get(trial, "trial")
    try:
        return int(value)
    except Exception:
        return "unknown" if value is None else str(value)


def _load_trials(path: Path) -> list:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data[1] if isinstance(data, tuple) else data


def _condition_label(trial) -> str:
    factor = _get(trial, "aiVelocityFactor", np.nan)
    if factor == 1:
        return "Shared-control"
    if factor == 0:
        return "BCI-only"
    return f"AI factor {factor:g}"


def _outcome_label(trial) -> str:
    return "success" if _get(trial, "answer") == 1 else "failure"


def _trial_xz(trial) -> np.ndarray:
    traj = _get(trial, "avatarTrajectory")
    x = np.asarray(traj["x"], dtype=float)
    z = np.asarray(traj["z"], dtype=float)
    n = min(len(x), len(z))
    points = np.column_stack([x[:n], z[:n]])
    return points[np.isfinite(points).all(axis=1)]


def _target_label(trial) -> str:
    pos = np.asarray(_get(trial, "targetPosition"), dtype=float)[[0, 2]]
    label, dist = min(
        ((label, float(np.linalg.norm(pos - center))) for label, center in TARGETS.items()),
        key=lambda item: item[1],
    )
    if dist >= 0.35:
        raise ValueError(f"Unknown target position {pos}")
    return label


def _obstacle_center(target_label: str, trial) -> tuple[float, float]:
    obstacle_pos = _get(trial, "obstaclePosition")
    if obstacle_pos is not None:
        obstacle = np.asarray([obstacle_pos[0], obstacle_pos[2]], dtype=float)
        if np.isfinite(obstacle).all():
            return float(obstacle[0]), float(obstacle[1])
    target = TARGETS[target_label]
    mapped = target_to_obstacle_mapping[(round(float(target[0]), 1), round(float(target[1]), 1))]
    return float(mapped[0]), float(mapped[1])


def _target_window_mask(points: np.ndarray, target_label: str) -> np.ndarray:
    target = TARGETS[target_label]
    return (
        (np.abs(points[:, 0] - target[0]) <= TARGET_HALF_WIDTH)
        & (np.abs(points[:, 1] - target[1]) <= TARGET_HALF_WIDTH)
    )


def _max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _wrong_target_label(points: np.ndarray, target_label: str) -> str:
    end = points[-1]
    wrong = [(label, center) for label, center in TARGETS.items() if label != target_label]
    label, _dist = min(
        ((label, float(np.linalg.norm(end - center))) for label, center in wrong),
        key=lambda item: item[1],
    )
    return label


def _classify_execution_reason(
    points: np.ndarray,
    target_label: str,
    obstacle_center: tuple[float, float],
) -> str:
    obstacle = np.asarray(obstacle_center, dtype=float)
    inflated_half = OBSTACLE_RADIUS + AVATAR_RADIUS + STUCK_MARGIN
    inside_obstacle = (
        (np.abs(points[:, 0] - obstacle[0]) < inflated_half)
        & (np.abs(points[:, 1] - obstacle[1]) < inflated_half)
    )
    passed_obstacle = bool(np.any(points[:, 1] > obstacle[1] + OBSTACLE_RADIUS))
    if _max_consecutive(inside_obstacle) >= 1 and not passed_obstacle:
        return "stuck_obstacle"

    inside_true = _target_window_mask(points, target_label)
    ever_in_true = bool(np.any(inside_true))
    target_z = float(TARGETS[target_label][1])
    if ever_in_true:
        enter_idx = int(np.argmax(inside_true))
        if bool(np.any(points[enter_idx:, 1] > target_z + TARGET_HALF_WIDTH)):
            return "overshoot"

    longest_true = _max_consecutive(inside_true)
    if ever_in_true and longest_true < MIN_DWELL_FRAMES:
        return "not_long_enough"

    if (not ever_in_true) and float(np.nanmax(points[:, 1])) < target_z - TARGET_HALF_WIDTH:
        return "not_close_to_true_z"

    return "other"


def _is_correct_acquisition(points: np.ndarray, target_label: str, trial) -> bool:
    return _get(trial, "answer") == 1 and _max_consecutive(_target_window_mask(points, target_label)) >= MIN_DWELL_FRAMES


def _find_trial(base_dir: Path, spec: ExampleSpec):
    path = base_dir / spec.monkey / spec.experiment / spec.session_file
    if not path.exists():
        raise FileNotFoundError(path)
    trials = _load_trials(path)
    for trial in trials:
        if _trial_id(trial) == spec.trial_id:
            return trial
    raise ValueError(f"Trial {spec.trial_id} not found in {path}")


def _record_from_spec(base_dir: Path, spec: ExampleSpec) -> TrialRecord:
    trial = _find_trial(base_dir, spec)
    points = _trial_xz(trial)
    target_label = _target_label(trial)
    obstacle = _obstacle_center(target_label, trial)
    wrong_label = _wrong_target_label(points, target_label)
    target = TARGETS[target_label]
    wrong_target = TARGETS[wrong_label]
    inside_true = _target_window_mask(points, target_label)
    reason = _classify_execution_reason(points, target_label, obstacle)
    if spec.panel == "Correct target acquisition":
        if not _is_correct_acquisition(points, target_label, trial):
            raise ValueError(
                f"{spec.panel} example {spec.session_file} trial {spec.trial_id} "
                "does not satisfy the correct acquisition criterion."
            )
        reason = "correct"
    else:
        expected = {
            "Blocked": "stuck_obstacle",
            "Target overshoot": "overshoot",
            "Insufficient target proximity": "not_close_to_true_z",
            "Insufficient dwell time": "not_long_enough",
        }[spec.panel]
        if reason != expected:
            raise ValueError(
                f"{spec.panel} example {spec.session_file} trial {spec.trial_id} "
                f"classified as {reason}, expected {expected}."
            )
    return TrialRecord(
        panel=spec.panel,
        monkey=spec.monkey,
        experiment=spec.experiment,
        session_index=spec.session_index,
        session_file=spec.session_file,
        trial_id=spec.trial_id,
        condition=_condition_label(trial),
        outcome=_outcome_label(trial),
        target_label=target_label,
        wrong_target_label=wrong_label,
        points=points,
        obstacle_center=obstacle,
        min_obstacle_distance_m=float(np.nanmin(np.linalg.norm(points - np.asarray(obstacle), axis=1))),
        final_distance_true_target_m=float(np.linalg.norm(points[-1] - target)),
        final_distance_nearest_wrong_target_m=float(np.linalg.norm(points[-1] - wrong_target)),
        duration_s=float(len(points) / 20.0),
        selection_score=_selection_score(spec.panel, points, target_label, obstacle),
        max_consecutive_target_frames=_max_consecutive(inside_true),
        max_z_minus_target_z_m=float(np.nanmax(points[:, 1]) - target[1]),
        canonical_reason=reason,
    )


def _selection_score(
    panel: str,
    points: np.ndarray,
    target_label: str,
    obstacle_center: tuple[float, float],
) -> float:
    target = TARGETS[target_label]
    inside_true = _target_window_mask(points, target_label)
    if panel == "Blocked":
        return float(np.nanmin(np.linalg.norm(points - np.asarray(obstacle_center), axis=1)))
    if panel == "Target overshoot":
        enter_idx = int(np.argmax(inside_true))
        return float(np.nanmax(points[enter_idx:, 1]) - (target[1] + TARGET_HALF_WIDTH))
    if panel == "Insufficient target proximity":
        return float(np.linalg.norm(points[-1] - target))
    if panel == "Insufficient dwell time":
        return float(_max_consecutive(inside_true))
    if panel == "Correct target acquisition":
        return float(np.linalg.norm(points[-1] - target))
    return float("nan")


def _plot_record(
    ax,
    record: TrialRecord,
    panel_letter: str | None = None,
    title: str | None = None,
    publication_layout: bool = False,
) -> None:
    points = record.points
    target = TARGETS[record.target_label]
    obstacle = np.asarray(record.obstacle_center, dtype=float)

    for label, center in TARGETS.items():
        if label == record.target_label:
            continue
        ax.add_patch(
            Rectangle(
                (center[0] - DRAW_TARGET_WIDTH / 2, center[1] - DRAW_TARGET_WIDTH / 2),
                DRAW_TARGET_WIDTH,
                DRAW_TARGET_WIDTH,
                facecolor=OTHER_TARGET_COLOR,
                edgecolor=OTHER_TARGET_EDGE,
                lw=0.8 if publication_layout else 0.8,
                alpha=1.0,
                zorder=0,
            )
        )

    ax.plot(points[:, 0], points[:, 1], color=TRAJECTORY_COLOR, lw=1.9 if publication_layout else 1.6, alpha=0.95)
    ax.scatter(
        points[0, 0],
        points[0, 1],
        s=44 if publication_layout else 36,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

    ax.add_patch(
        Rectangle(
            (obstacle[0] - OBSTACLE_WIDTH / 2, obstacle[1] - OBSTACLE_WIDTH / 2),
            OBSTACLE_WIDTH,
            OBSTACLE_WIDTH,
            facecolor=OBSTACLE_COLOR,
            edgecolor=OBSTACLE_COLOR,
            lw=1.0,
            alpha=0.65,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (target[0] - TARGET_HALF_WIDTH, target[1] - TARGET_HALF_WIDTH),
            2 * TARGET_HALF_WIDTH,
            2 * TARGET_HALF_WIDTH,
            facecolor="none",
            edgecolor=TARGET_COLOR,
            lw=1.0 if publication_layout else 0.9,
            ls=(0, (3, 2)),
            alpha=0.75,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (target[0] - DRAW_TARGET_WIDTH / 2, target[1] - DRAW_TARGET_WIDTH / 2),
            DRAW_TARGET_WIDTH,
            DRAW_TARGET_WIDTH,
            facecolor=TARGET_COLOR,
            edgecolor=TARGET_COLOR,
            lw=1.2 if publication_layout else 1.0,
            zorder=3,
        )
    )

    if publication_layout:
        ax.set_title(title or PANEL_TITLES.get(record.panel, record.panel), fontsize=9.5, pad=5)
        if panel_letter is not None:
            ax.text(
                -0.09,
                1.08,
                panel_letter,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )
    else:
        ax.set_title(record.panel, fontsize=8.5, pad=4)
        ax.text(
            0.02,
            0.98,
            f"{record.condition}; target: {record.target_label}\n"
            f"{record.monkey}, {record.experiment}\n"
            f"trial {record.trial_id}; {record.canonical_reason}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.6,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-1.0, 14.5)
    ax.grid(True, color="#dddddd", lw=0.5 if publication_layout else 0.45, alpha=0.8)
    ax.tick_params(labelsize=7.5 if publication_layout else 7, length=2.5 if publication_layout else 2)


def _write_provenance(records: list[TrialRecord], output_dir: Path, output_stem: str = DEFAULT_OUTPUT_STEM) -> list[Path]:
    csv_path = output_dir / f"{output_stem}_provenance.csv"
    txt_path = output_dir / f"{output_stem}_provenance.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "panel",
                "monkey",
                "experiment",
                "session_index",
                "session_file",
                "trial_id",
                "condition",
                "outcome",
                "target_label",
                "wrong_target_label",
                "min_obstacle_distance_m",
                "final_distance_true_target_m",
                "final_distance_nearest_wrong_target_m",
                "duration_s",
                "selection_score",
                "max_consecutive_target_frames",
                "max_z_minus_target_z_m",
                "canonical_reason",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for r in records:
            writer.writerow({name: getattr(r, name) for name in writer.fieldnames})

    txt_path.write_text(
        "\n".join(
            [
                "Data-driven representative trajectory examples for execution-failure intuition.",
                "Categories match canonical execution-failure fields: stuck_obstacle, overshoot, not_close_to_true_z, and not_long_enough.",
                "Correct target acquisition example: successful trial with at least 10 consecutive frames inside the true-target window.",
                "Blocked example: failed trial that enters the inflated obstacle square and does not pass the obstacle.",
                "Target overshoot rule: failed trial enters the true-target window, then later has z > target_z + 2.10 m before satisfying the 10-frame dwell criterion.",
                "Insufficient target proximity example: failed trial that does not enter the true-target window and does not pass the target z coordinate.",
                "Insufficient dwell time example: failed trial that enters the true-target window but stays there for fewer than 10 consecutive frames.",
                f"Collision radius shown: {OBSTACLE_RADIUS + AVATAR_RADIUS:.2f} m.",
                f"Obstacle-proximity selection radius: {OBSTACLE_RADIUS + AVATAR_RADIUS + STUCK_MARGIN + 0.04:.2f} m.",
                f"Stuck selection inflated obstacle half-width: {OBSTACLE_RADIUS + AVATAR_RADIUS + STUCK_MARGIN:.2f} m.",
                "Stuck selection minimum consecutive frames: 1.",
                f"Wrong-target/timeout endpoint window half-width: {TARGET_HALF_WIDTH:.2f} m.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [csv_path, txt_path]


def _legend_handles(line_width: float = 1.6) -> list:
    return [
        Line2D([0], [0], color=TRAJECTORY_COLOR, lw=line_width, label="Trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            lw=0,
            markersize=5.5,
            label="Start",
        ),
        Patch(facecolor=TARGET_COLOR, edgecolor=TARGET_COLOR, label="Cued target"),
        Patch(facecolor=OTHER_TARGET_COLOR, edgecolor=OTHER_TARGET_EDGE, label="Other targets"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, alpha=0.65, label="Obstacle"),
        Patch(facecolor="none", edgecolor=TARGET_COLOR, linestyle="--", linewidth=1.0, label="Target window"),
    ]


def build_figure(base_dir: Path, output_dir: Path, output_stem: str = DEFAULT_OUTPUT_STEM) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [_record_from_spec(base_dir, spec) for spec in EXAMPLES]

    fig, axes = plt.subplots(1, len(records), figsize=(4.35 * len(records), 4.7), sharex=True, sharey=True)
    for ax, record in zip(axes, records):
        _plot_record(ax, record)
    axes[0].set_ylabel("Z position (m)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("X position (m)", fontsize=8)

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), w_pad=1.0)

    outputs = []
    for ext in ("svg", "png", "pdf"):
        path = output_dir / f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    outputs.extend(_write_provenance(records, output_dir, output_stem))
    return outputs


def build_publication_figure(base_dir: Path, output_dir: Path, output_stem: str = PUBLICATION_OUTPUT_STEM) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [_record_from_spec(base_dir, spec) for spec in EXAMPLES]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=False)
    axes = [
        fig.add_axes([0.175, 0.585, 0.285, 0.262]),
        fig.add_axes([0.540, 0.585, 0.285, 0.262]),
        fig.add_axes([0.070, 0.205, 0.250, 0.230]),
        fig.add_axes([0.375, 0.205, 0.250, 0.230]),
        fig.add_axes([0.680, 0.205, 0.250, 0.230]),
    ]

    for letter, ax, record in zip("abcde", axes, records):
        _plot_record(
            ax,
            record,
            panel_letter=letter,
            title=PANEL_TITLES.get(record.panel, record.panel),
            publication_layout=True,
        )

    for ax in axes:
        ax.set_xlabel("X position (m)", fontsize=8)
    axes[0].set_ylabel("Z position (m)", fontsize=8)
    axes[2].set_ylabel("Z position (m)", fontsize=8)

    fig.suptitle(
        "Representative trajectory examples illustrating execution-related failure categories quantified in Figure 5",
        x=0.070,
        y=0.965,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.legend(
        handles=_legend_handles(line_width=1.9),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7.4,
        bbox_to_anchor=(0.535, 0.035),
        handlelength=1.6,
        columnspacing=1.25,
        handletextpad=0.5,
    )

    outputs = []
    for ext in ("svg", "png", "pdf"):
        path = output_dir / f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    outputs.extend(_write_provenance(records, output_dir, output_stem))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build execution-failure examples for Extended Data Figure 3.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--publication-layout", action="store_true")
    args = parser.parse_args(argv)

    if args.publication_layout:
        outputs = build_publication_figure(
            load_base_dir(args.base_dir),
            Path(args.output_dir),
            args.output_stem or PUBLICATION_OUTPUT_STEM,
        )
    else:
        outputs = build_figure(
            load_base_dir(args.base_dir),
            Path(args.output_dir),
            args.output_stem or DEFAULT_OUTPUT_STEM,
        )
    for path in outputs:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
