"""
Recreate the temporal-prior reset example panel for the AI paper.

The figure compares one respawn trial as originally run to the same trial
replayed with temporal prior reset enabled. The original trajectory is loaded
from the Method-paper trial PKL; the reset trajectory is loaded from the
matching reset-prior replay PKL.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONKEY = "Monkey 1"
DEFAULT_EXPERIMENT = "AI Respawn"
DEFAULT_SESSION = "navdecodingsphereairespawn_Maui_20250604_1149_A"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "Figure 5" / "Temporal prior reset"
DEFAULT_TRIAL = 6
DEFAULT_STRIDE = 5

ALL_TARGET_POSITIONS = [
    (-7.0, 6.0),
    (-3.5, 8.5),
    (0.0, 9.2),
    (3.5, 8.5),
    (7.0, 6.0),
]


def load_base_dir(cli_base_dir: str | None = None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def default_source_paths(base_dir: str | Path) -> tuple[Path, Path, Path]:
    session_dir = Path(base_dir) / DEFAULT_MONKEY / DEFAULT_EXPERIMENT
    original_path = session_dir / f"{DEFAULT_SESSION}_trials.pkl"
    original_ai_path = session_dir / "AIFiles" / f"{DEFAULT_SESSION}_aitrials.pkl"
    reset_path = session_dir / "resetPriorFiles" / f"reset_prior_analyzes_{DEFAULT_SESSION}.pkl"
    return original_path, original_ai_path, reset_path


def _load_trial(path: Path, trial_id: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing reset-prior data file: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    for trial in data.get("trials", []):
        if int(trial.get("trial_id", -1)) == int(trial_id):
            return trial
    raise ValueError(f"Trial {trial_id} not found in {path}")


def _load_original_trial(path: Path, ai_path: Path, trial_id: int, crop_to_old_target: bool = True) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing original trial data file: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    trials = data[1] if isinstance(data, tuple) and len(data) > 1 else data
    raw_trial = None
    for trial in trials:
        if int(trial.get("trial", -1)) == int(trial_id):
            raw_trial = trial
            break
    if raw_trial is None:
        raise ValueError(f"Trial {trial_id} not found in {path}")

    trajectory = raw_trial["avatarTrajectory"]
    times = np.asarray(trajectory["time"], float).reshape(-1)
    x = np.asarray(trajectory["x"], float).reshape(-1)
    z = np.asarray(trajectory["z"], float).reshape(-1)
    y = np.asarray(trajectory.get("y", np.full_like(x, 0.75)), float).reshape(-1)
    n = min(len(times), len(x), len(y), len(z))
    times = times[:n]
    position = np.column_stack([x[:n], y[:n], z[:n]])

    velocity = raw_trial.get("modelVelocity") or raw_trial.get("avatarVelocity") or {}
    vx = np.asarray(velocity.get("vx", np.full(n, np.nan)), float).reshape(-1)
    vz = np.asarray(velocity.get("vz", np.full(n, np.nan)), float).reshape(-1)
    bci_n = min(n, len(vx), len(vz))
    bci_velocity = np.full((n, 2), np.nan)
    bci_velocity[:bci_n, 0] = vx[:bci_n]
    bci_velocity[:bci_n, 1] = vz[:bci_n]

    entropy = _load_entropy_for_original_trial(ai_path, trial_id, times)
    target_jump_time = np.asarray(raw_trial.get("targetJumpTime", []), float).reshape(-1)
    target_jump_index = None
    if len(target_jump_time) and len(times):
        target_jump_index = int(np.nanargmin(np.abs(times - target_jump_time[0])))

    if crop_to_old_target:
        old_target = _point_from_xyz(raw_trial.get("targetPosition"))
        crop_index = _nearest_target_index(position, old_target)
        if crop_index is not None and crop_index > 1:
            stop = crop_index + 1
            position = position[:stop]
            bci_velocity = bci_velocity[:stop]
            entropy = entropy[:stop]

    return {
        "trial_id": trial_id,
        "answer": raw_trial.get("answer"),
        "answer_log": raw_trial.get("answer"),
        "true_goal": _point_from_xyz(raw_trial.get("targetPosition")),
        "target_jump_position": _point_from_xyz(raw_trial.get("targetJumpPosition")),
        "target_jump_index": target_jump_index,
        "reset_iters": [target_jump_index] if target_jump_index is not None else [],
        "samples": {
            "position": position,
            "entropy": entropy,
            "bci_velocity": bci_velocity,
        },
    }


def _nearest_target_index(position: np.ndarray, target: dict | None) -> int | None:
    if target is None or len(position) == 0:
        return None
    target_xz = np.asarray([target["x"], target["z"]], float)
    pos_xz = np.asarray(position[:, [0, 2]], float)
    finite = np.isfinite(pos_xz).all(axis=1)
    if not finite.any():
        return None
    distances = np.full(len(position), np.inf)
    distances[finite] = np.linalg.norm(pos_xz[finite] - target_xz, axis=1)
    return int(np.nanargmin(distances))


def _point_from_xyz(values) -> dict | None:
    if values is None:
        return None
    arr = np.asarray(values, float).reshape(-1)
    if len(arr) < 3:
        return None
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}


def _load_entropy_for_original_trial(ai_path: Path, trial_id: int, sample_times: np.ndarray) -> np.ndarray:
    if not ai_path.exists():
        return np.full(len(sample_times), np.nan)
    with ai_path.open("rb") as f:
        data = pickle.load(f)
    trials = data[1] if isinstance(data, tuple) and len(data) > 1 else data
    ai_trial = None
    for trial in trials:
        if int(trial.get("trial", -1)) == int(trial_id):
            ai_trial = trial
            break
    if ai_trial is None:
        return np.full(len(sample_times), np.nan)

    ai_times = []
    ai_entropy = []
    for sample in ai_trial.get("aiVelocities", []):
        timestamp = sample.get("OutputTimestamp", sample.get("InputTimestamp"))
        entropy = sample.get("EntropyLb")
        if timestamp is None or entropy is None:
            continue
        timestamp = float(np.asarray(timestamp).reshape(-1)[0])
        entropy = float(np.asarray(entropy).reshape(-1)[0])
        if np.isfinite(timestamp) and np.isfinite(entropy):
            ai_times.append(timestamp)
            ai_entropy.append(entropy)
    if not ai_times:
        return np.full(len(sample_times), np.nan)

    ai_times = np.asarray(ai_times)
    ai_entropy = np.asarray(ai_entropy)
    order = np.argsort(ai_times)
    ai_times = ai_times[order]
    ai_entropy = ai_entropy[order]
    nearest = np.searchsorted(ai_times, sample_times)
    nearest = np.clip(nearest, 1, len(ai_times) - 1)
    left = nearest - 1
    right = nearest
    take_right = np.abs(ai_times[right] - sample_times) < np.abs(sample_times - ai_times[left])
    indices = np.where(take_right, right, left)
    return ai_entropy[indices]


def _xz(point: dict | None) -> tuple[float, float] | None:
    if point is None:
        return None
    return float(point["x"]), float(point["z"])


def _add_square(ax, xz: tuple[float, float], width: float, **kwargs) -> Rectangle:
    x, z = xz
    patch = Rectangle((x - width / 2, z - width / 2), width, width, **kwargs)
    ax.add_patch(patch)
    return patch


def _prior_confidence_index(entropy: np.ndarray, floor: float = 0.2) -> np.ndarray:
    """Match the paper plotting code: Prior Confidence Index = 1 - EntropyLb / Hmax."""
    entropy = np.asarray(entropy, dtype=float)
    if entropy.size == 0 or np.all(np.isnan(entropy)):
        return np.full_like(entropy, float(floor), dtype=float)

    hmax_obs = np.nanmax(entropy)
    if not np.isfinite(hmax_obs) or hmax_obs <= 0:
        hmax_obs = np.nanpercentile(entropy, 95)

    k_est = int(np.round(np.exp(hmax_obs)))
    if k_est < 2:
        k_est = 2
    hmax = float(np.log(k_est))
    h = np.clip(entropy, 0.0, hmax)
    confidence = 1.0 - (h / hmax)
    confidence = np.nan_to_num(confidence, nan=float(floor), posinf=float(floor), neginf=1.0)
    return np.clip(confidence, float(floor), 1.0)


def _plot_trial(ax, trial: dict, title: str, show_legend: bool = False, stride: int | None = DEFAULT_STRIDE):
    samples = trial["samples"]
    pos = np.asarray(samples["position"], float)
    entropy = np.asarray(samples.get("entropy", np.full(len(pos), np.nan)), float).reshape(-1)
    bci = np.asarray(samples.get("bci_velocity", np.empty((0, 2))), float)

    n = min(len(pos), len(entropy))
    pos = pos[:n]
    entropy = entropy[:n]
    if bci.ndim == 2 and bci.shape[0] >= n:
        bci = bci[:n]
    else:
        bci = np.full((n, 2), np.nan)

    finite_pos = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 2])
    pos = pos[finite_pos]
    entropy = entropy[finite_pos]
    bci = bci[finite_pos]
    if stride is not None and stride > 1 and len(pos) > 1:
        keep = np.arange(0, len(pos), int(stride))
        pos = pos[keep]
        entropy = entropy[keep]
        bci = bci[keep]
    alpha = _prior_confidence_index(entropy)

    old_target = _xz(trial.get("true_goal"))
    new_target = _xz(trial.get("target_jump_position"))

    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 12)
    ax.axhline(2, color="#8E63CE", lw=1.0, zorder=0)

    for target in ALL_TARGET_POSITIONS:
        if old_target is not None and np.allclose(target, old_target):
            continue
        if new_target is not None and np.allclose(target, new_target):
            continue
        _add_square(ax, target, 1.5, facecolor="none", edgecolor="0.55", linestyle="--", linewidth=0.9)

    if old_target is not None:
        _add_square(ax, old_target, 1.5, facecolor="0.5", edgecolor="0.5", linewidth=1.0)
    if new_target is not None:
        _add_square(ax, new_target, 1.5, facecolor="green", edgecolor="green", linewidth=1.0)

    if len(pos) >= 2:
        confidence = np.ma.masked_invalid(alpha)
        scatter = ax.scatter(
            pos[:, 0],
            pos[:, 2],
            c=confidence,
            cmap="Blues",
            vmin=0,
            vmax=1,
            s=13,
            edgecolors="white",
            linewidths=0.25,
            zorder=5,
        )
        ax.plot(pos[:, 0], pos[:, 2], color="0.2", linewidth=0.8, alpha=0.35, zorder=3)
        ax.scatter(pos[0, 0], pos[0, 2], s=28, facecolor="white", edgecolor="black", linewidth=0.8, zorder=8)

        step = max(1, len(pos) // 18)
        for i in range(0, len(pos), step):
            vx, vz = bci[i]
            if not np.isfinite(vx) or not np.isfinite(vz):
                continue
            norm = np.hypot(vx, vz)
            if norm <= 1e-9:
                continue
            ax.arrow(
                pos[i, 0],
                pos[i, 2],
                0.35 * vx / norm,
                0.35 * vz / norm,
                color="black",
                alpha=0.8,
                head_width=0.10,
                length_includes_head=True,
                linewidth=0.6,
                zorder=9,
            )
    else:
        scatter = None

    if show_legend:
        handles = [
            Line2D([0], [0], color="#8E63CE", lw=1.0, label="Respawn boundary (z=2)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="Start Position"),
            Rectangle((0, 0), 1, 1, facecolor="green", edgecolor="green", label="Target after respawn"),
            Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="0.5", label="Target before respawn"),
            Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="0.55", linestyle="--", label="Other candidate targets"),
            Line2D([0], [0], color="black", marker=">", lw=0, label="BCI velocity"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=True)

    return scatter


def build_temporal_prior_reset_example(
    original_path: Path,
    original_ai_path: Path,
    reset_path: Path,
    trial_id: int = DEFAULT_TRIAL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    formats: tuple[str, ...] = ("svg",),
    write_provenance: bool = False,
    stride: int | None = DEFAULT_STRIDE,
) -> list[Path]:
    no_reset = _load_original_trial(original_path, original_ai_path, trial_id)
    reset = _load_trial(reset_path, trial_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    sc0 = _plot_trial(axes[0], no_reset, "No temporal prior reset", show_legend=True, stride=stride)
    sc1 = _plot_trial(axes[1], reset, "Temporal prior reset", show_legend=False, stride=stride)

    for ax, sc in zip(axes, [sc0, sc1]):
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.015)
            cbar.set_label("Prior Confidence Index", fontsize=8)
            cbar.set_ticks([0, 0.5, 1.0])
            cbar.ax.tick_params(labelsize=7)

    outputs = []
    stem = (
        "temporal_prior_reset_example_"
        f"Monkey1_{DEFAULT_SESSION}_"
        f"trial{trial_id:03d}"
    )
    save_kwargs = {
        "svg": {"bbox_inches": "tight"},
        "png": {"dpi": 300, "bbox_inches": "tight"},
        "pdf": {"bbox_inches": "tight"},
    }
    for ext in formats:
        out_path = output_dir / f"{stem}.{ext}"
        fig.savefig(out_path, **save_kwargs[ext])
        outputs.append(out_path)
        print(f"Saved {out_path}")
    plt.close(fig)

    if write_provenance:
        provenance_path = output_dir / f"{stem}_provenance.txt"
        provenance_path.write_text(
            "\n".join(
                [
                    "Temporal prior reset example",
                    f"Original no-reset source: {original_path}",
                    f"Original AI entropy source: {original_ai_path}",
                    f"Reset replay source: {reset_path}",
                    f"Trial: {trial_id}",
                    f"Trajectory sampling stride: {stride}",
                    "Old target / new target transition: (-3.5, 8.5) -> (0.0, 9.2).",
                    "Original no-reset panel is cropped at closest approach to the pre-respawn target.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(provenance_path)
        print(f"Wrote {provenance_path}")
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the temporal-prior reset example figure.")
    parser.add_argument("--base-dir", default=None, help="Root data directory. Defaults to config.yaml.")
    parser.add_argument("--original-path", default=None)
    parser.add_argument("--original-ai-path", default=None)
    parser.add_argument("--reset-path", default=None)
    parser.add_argument("--trial-id", type=int, default=DEFAULT_TRIAL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--formats", nargs="+", choices=("svg", "png", "pdf"), default=["svg"])
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--write-provenance", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    default_original, default_original_ai, default_reset = default_source_paths(load_base_dir(args.base_dir))
    return build_temporal_prior_reset_example(
        original_path=Path(args.original_path) if args.original_path else default_original,
        original_ai_path=Path(args.original_ai_path) if args.original_ai_path else default_original_ai,
        reset_path=Path(args.reset_path) if args.reset_path else default_reset,
        trial_id=args.trial_id,
        output_dir=Path(args.output_dir),
        formats=tuple(args.formats),
        write_provenance=args.write_provenance,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
