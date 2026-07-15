#!/usr/bin/env python
"""Make the Figure 2 Fixed Obstacle example trial panel.

This recreates the single-trial example that was previously generated from:

    Loki_20241223_1106_E, trial 058

The manuscript-era plotting code is the commented/overwritten
``plot_individual_trials`` block in ``src/plots.py``. This standalone script
reproduces that plotting recipe for the selected trial using the released
  source under the configured ``base_dir``.

Source trace:
  ai_ibci_analysis/scripts/run_AI_trials_analysis.py marks
  ``plot_individual_trials(monkey, experiment, base_dir)`` as the Figure 2-style
  visualization. That function reads recorded positions from
  ``aiVelocities[i]["AvatarPosition"]`` and plots confidence over time. This
  paper script uses the same project loader path, ``src.load.load_files()``, to
  pair the base trial file with the matching AI trial log before drawing the
  cleaned manuscript panel.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow, FancyArrowPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AI_metrics import ai_on_indices_from_trial  # noqa: E402
from src.load import load_files  # noqa: E402

MONKEY = "Monkey 2"
EXPERIMENT = "AI Obstacle"
SESSION_STEM = "navdecodingsphereaiobstacle_Loki_20241223_1106_E"
TRIAL_ID = 58
SOURCE_TRACE = (
    "ai_ibci_analysis/scripts/run_AI_trials_analysis.py -> "
    "plot_individual_trials(monkey, experiment, base_dir) marked 'THIS ONE!!!!'; "
    "later ai_ibci_analysis/src/plots.py::plot_individual_trials uses the "
    "time-from-trial-start panel with AI-override bands; this script ports "
    "that visual recipe and selected manuscript trial from the configured base_dir."
)

TARGETS = np.array(
    [
        [-7.0, 0.75, 6.0],
        [-3.5, 0.75, 8.5],
        [0.0, 0.75, 9.2],
        [3.5, 0.75, 8.5],
        [7.0, 0.75, 6.0],
    ],
    dtype=float,
)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_base_dir(cli_base_dir: str | None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def robust_alpha(entropy: np.ndarray, floor: float = 0.2) -> np.ndarray:
    """Convert entropy to the prior-confidence index used by the time-panel code."""
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
    alpha = 1.0 - (h / hmax)
    alpha = np.nan_to_num(alpha, nan=float(floor), posinf=float(floor), neginf=1.0)
    return np.clip(alpha, float(floor), 1.0)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trial_id(obj) -> int | None:
    for key in ("trial", "trial_id", "trialId"):
        value = _get(obj, key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _source_ai_path(base_file: str | Path) -> Path:
    base_file = Path(base_file)
    return base_file.parent / "AIFiles" / base_file.name.replace("_trials.pkl", "_aitrials.pkl")


def _configured_ai_log_path(base_dir: str | Path) -> Path:
    return (
        Path(base_dir)
        / MONKEY
        / EXPERIMENT
        / "AIFiles"
        / f"{SESSION_STEM}_aitrials.pkl"
    )


def _as_array_field(trial, field: str) -> np.ndarray:
    value = _get(trial, field)
    if value is None:
        raise ValueError(f"Selected trial is missing {field}.")
    return np.asarray(value, dtype=float)


def load_trial(base_dir: str, trial_id: int) -> dict:
    configured_ai_log = _configured_ai_log_path(base_dir)
    if configured_ai_log.exists():
        with configured_ai_log.open("rb") as f:
            obj = pickle.load(f)
        trials = obj[1] if isinstance(obj, tuple) else obj
        for trial in trials:
            if _trial_id(trial) == int(trial_id):
                trial = dict(trial)
                trial["source_base_file"] = None
                trial["source_ai_file"] = str(configured_ai_log)
                trial["source_session_index"] = None
                trial["source_trace"] = SOURCE_TRACE
                return trial
        raise ValueError(f"Trial {trial_id} not found in configured AI log: {configured_ai_log}")

    all_trials, _, _, _, _, _, pkl_files, _ = load_files(EXPERIMENT, MONKEY, base_dir)

    for session_idx, base_file in enumerate(pkl_files):
        base_file = Path(base_file)
        if SESSION_STEM not in base_file.stem:
            continue

        for trial in all_trials[session_idx]:
            if _trial_id(trial) != int(trial_id):
                continue

            ai_velocities = _get(trial, "aiVelocities", []) or []
            if len(ai_velocities) < 3:
                raise ValueError(
                    f"Trial {trial_id} in {base_file} has no matched aiVelocities; "
                    f"expected paired AI file at {_source_ai_path(base_file)}."
                )

            return {
                "trial": int(trial_id),
                "answer": _get(trial, "answer"),
                "targetPosition": _as_array_field(trial, "targetPosition"),
                "obstaclePosition": _as_array_field(trial, "obstaclePosition"),
                "aiVelocities": ai_velocities,
                "source_base_file": str(base_file),
                "source_ai_file": str(_source_ai_path(base_file)),
                "source_session_index": session_idx,
                "source_trace": SOURCE_TRACE,
            }

    searched = Path(base_dir) / MONKEY / EXPERIMENT
    raise ValueError(f"Trial {trial_id} from session {SESSION_STEM} not found under {searched}.")


def _position_from_record(rec: dict) -> tuple[float, float] | None:
    pos = rec.get("AvatarPosition")
    if pos is None:
        return None
    if isinstance(pos, (float, int, np.floating, np.integer)):
        return None
    if isinstance(pos, dict):
        x, z = pos.get("x"), pos.get("z")
    else:
        try:
            if len(pos) < 3:
                return None
        except TypeError:
            return None
        x, z = pos[0], pos[2]
    if x is None or z is None:
        return None
    x, z = float(x), float(z)
    if not np.isfinite(x) or not np.isfinite(z):
        return None
    return x, z


def _thin_trajectory(pos, keep_idx, alpha_vals, ai_on_mask, ai_recs, stride=None, min_dt=None, min_dist=None):
    idx = list(range(len(keep_idx)))

    if stride and stride > 1:
        idx = idx[::stride]

    if min_dt is not None:
        ts_vals = []
        for k in keep_idx:
            t = ai_recs[k].get("OutputTimestamp", ai_recs[k].get("InputTimestamp"))
            if hasattr(t, "timestamp"):
                t = t.timestamp()
            ts_vals.append(float(t) if t is not None else np.nan)

        kept = []
        last_t = None
        for j in idx:
            tj = ts_vals[j]
            if np.isnan(tj):
                kept.append(j)
                last_t = tj
                continue
            if last_t is None or (tj - last_t) >= float(min_dt):
                kept.append(j)
                last_t = tj
        idx = kept

    if min_dist is not None and len(idx) > 1:
        kept = [idx[0]]
        last_p = pos[idx[0]]
        for j in idx[1:]:
            if np.linalg.norm(pos[j] - last_p) >= float(min_dist):
                kept.append(j)
                last_p = pos[j]
        idx = kept

    pos_th = pos[idx]
    keep_idx_th = [keep_idx[j] for j in idx]
    alpha_th = alpha_vals[idx]
    ai_on_th = ai_on_mask[idx]
    return pos_th, keep_idx_th, alpha_th, ai_on_th


def extract_trial_series(trial: dict, stride: int | None = 5, min_dt=None, min_dist=None) -> dict:
    ai_recs = trial.get("aiVelocities", []) or []
    xs, zs, keep_idx, times = [], [], [], []
    for i, rec in enumerate(ai_recs):
        pos = _position_from_record(rec)
        if pos is None:
            continue
        t = rec.get("OutputTimestamp", rec.get("InputTimestamp", np.nan))
        if hasattr(t, "timestamp"):
            t = t.timestamp()
        try:
            t = float(np.asarray(t, dtype=float).ravel()[0])
        except Exception:
            t = np.nan
        xs.append(pos[0])
        zs.append(pos[1])
        keep_idx.append(i)
        times.append(t)

    if len(keep_idx) < 3:
        raise ValueError("Selected trial has too few AI velocity samples.")

    ent_raw = []
    for rec in ai_recs:
        entropy = rec.get("EntropyLb")
        if isinstance(entropy, (list, tuple, np.ndarray)):
            arr = np.asarray(entropy, dtype=float).ravel()
            entropy = arr[0] if arr.size else np.nan
        entropy = np.nan if entropy is None else float(entropy)
        ent_raw.append(entropy)

    entropy = np.asarray(ent_raw, dtype=float)
    alpha_all = robust_alpha(entropy)
    alpha_vals = alpha_all[keep_idx]
    alpha_vals = np.clip(np.nan_to_num(alpha_vals, nan=1.0, posinf=1.0, neginf=0.2), 0.2, 1.0)

    ai_idxs, inputs, outputs, _time_info = ai_on_indices_from_trial(trial)
    ai_on_set = set(ai_idxs)
    ai_on_mask = np.array([(k in ai_on_set) for k in keep_idx], dtype=bool)

    pos = np.column_stack([np.asarray(xs), np.asarray(zs)])
    time_by_keep_idx = {idx: float(t) for idx, t in zip(keep_idx, times)}
    pos, keep_idx, alpha_vals, ai_on_mask = _thin_trajectory(
        pos, keep_idx, alpha_vals, ai_on_mask, ai_recs, stride=stride, min_dt=min_dt, min_dist=min_dist
    )
    times_th = np.asarray([time_by_keep_idx.get(k, np.nan) for k in keep_idx], dtype=float)

    return {
        "sample_index": np.asarray(keep_idx, dtype=int),
        "time": times_th,
        "pos": pos,
        "x": pos[:, 0],
        "z": pos[:, 1],
        "entropy": entropy[keep_idx],
        "alpha": alpha_vals,
        "ai_on_mask": ai_on_mask,
        "inputs": inputs,
        "outputs": outputs,
    }


def add_square(ax, center, width=1.5, color="k", fill=False, lw=2, label=None, linestyle=None):
    if center is None:
        return None
    x, _, z = center
    patch = Rectangle(
        (x - width / 2.0, z - width / 2.0),
        width,
        width,
        edgecolor=color,
        facecolor=(color if fill else "none"),
        linestyle=("--" if linestyle is None and not fill else linestyle or "-"),
        linewidth=lw,
        label=label,
    )
    ax.add_patch(patch)
    return patch


def _as_tuple(center):
    if center is None:
        return None
    if isinstance(center, dict):
        return (center.get("x"), center.get("y", 0.0), center.get("z"))
    return tuple(center)


def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrow(
        xdescent,
        ydescent + height / 2.0,
        width,
        0.0,
        length_includes_head=True,
        head_width=0.6 * height,
        head_length=0.35 * width,
        color="black",
    )


def _legend_square(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    size = 1.2 * min(width, height)
    x = xdescent + (width - size) / 2.0
    y = ydescent + (height - size) / 2.0
    sq = Rectangle((x, y), size, size)
    sq.set_facecolor(orig_handle.get_facecolor())
    sq.set_edgecolor(orig_handle.get_edgecolor())
    sq.set_linestyle(orig_handle.get_linestyle())
    sq.set_linewidth(orig_handle.get_linewidth() or 1.5)
    return sq


def make_figure(
    trial: dict,
    out_dir: Path,
    stride: int | None = 5,
    formats: tuple[str, ...] = ("svg",),
    write_provenance: bool = False,
) -> list[Path]:
    series = extract_trial_series(trial, stride=stride)
    pos = series["pos"]
    alpha = series["alpha"]
    ai_on_mask = series["ai_on_mask"]
    keep_idx = series["sample_index"].tolist()
    times_th = np.asarray(series["time"], dtype=float)
    inputs = series["inputs"]

    fig, (ax_path, ax_alpha) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.set_xlabel("X")
    ax_path.set_ylabel("Z")

    add_square(ax_path, trial.get("targetPosition"), width=1.5, color="g", fill=True, lw=1.5)
    add_square(ax_path, trial.get("obstaclePosition"), width=0.9, color="r", fill=True, lw=1.5)

    curr_t = _as_tuple(trial.get("targetPosition"))
    other_targets_labeled = False
    for candidate in TARGETS:
        ct = _as_tuple(candidate)
        if curr_t is not None and np.allclose([ct[0], ct[2]], [curr_t[0], curr_t[2]], atol=1e-6):
            continue
        label = "Other candidate targets" if not other_targets_labeled else "_nolegend_"
        add_square(ax_path, ct, width=1.5, color="#7f7f7f", fill=False, lw=1.5, label=label)
        other_targets_labeled = True

    color_on = "#D81B60"
    color_off = "#6C6E6F"

    base = plt.cm.Blues(np.linspace(0.30, 0.95, 256))
    base[:, 3] = 1.0
    cmap = ListedColormap(base)
    vmin, vmax = np.nanpercentile(alpha, [5, 95])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    colors = cmap(norm(alpha))

    for i in range(len(pos) - 1):
        ax_path.plot(
            pos[i : i + 2, 0],
            pos[i : i + 2, 1],
            color=color_off,
            alpha=alpha[i],
            linewidth=2,
        )

    on_idx = np.flatnonzero(ai_on_mask)
    off_idx = np.flatnonzero(~ai_on_mask)
    ax_path.scatter(
        pos[on_idx, 0],
        pos[on_idx, 1],
        s=70,
        c=color_on,
        edgecolors="black",
        marker="D",
        linewidths=1,
        alpha=0.7,
        zorder=6,
        label="AI override",
    )
    ax_path.scatter(
        pos[off_idx, 0],
        pos[off_idx, 1],
        s=70,
        c=colors[off_idx],
        edgecolors="white",
        linewidths=0.7,
        alpha=1.0,
        zorder=3,
    )
    ax_path.scatter(pos[0, 0], pos[0, 1], s=60, facecolor="white", edgecolor="k", zorder=4, label="Start Position")

    if inputs is not None and len(inputs) > 0 and len(inputs) > max(keep_idx, default=-1):
        v_all = inputs[:, [0, 2]]
        v = v_all[keep_idx]
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        v = (v / n) * 0.4
        for j in range(len(pos)):
            ax_path.arrow(
                pos[j, 0],
                pos[j, 1],
                v[j, 0],
                v[j, 1],
                head_width=0.11,
                length_includes_head=True,
                color="black",
                alpha=1,
                zorder=10,
            )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_path, fraction=0.046, pad=0.02)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"])

    arrow_handle = FancyArrowPatch((0, 0), (1.0, 0.0), arrowstyle="->", mutation_scale=16, color="black", linewidth=2, label="BCI velocity")
    square_active = Rectangle((0, 0), 1, 1, facecolor="g", edgecolor="g", label="Active target")
    square_obst = Rectangle((0, 0), 1, 1, facecolor="r", edgecolor="r", label="Obstacle")
    square_other = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="k", linestyle="--", label="Other candidate targets")
    handles, _ = ax_path.get_legend_handles_labels()
    handles += [square_active, square_obst, square_other, arrow_handle]
    by_label = {h.get_label(): h for h in handles}
    ax_path.legend(
        by_label.values(),
        by_label.keys(),
        loc="best",
        frameon=True,
        handler_map={
            FancyArrowPatch: HandlerPatch(patch_func=_legend_arrow),
            Rectangle: HandlerPatch(patch_func=_legend_square),
        },
    )
    ax_path.margins(0.05)
    ax_path.set_xlim(-10, 10)
    ax_path.set_ylim(-1, 11)

    if np.isfinite(times_th).sum() >= 3:
        mask_ts = np.isfinite(times_th)
        t_rel = times_th - times_th[mask_ts][0]

        is_on = ai_on_mask & np.isfinite(t_rel)
        if np.any(is_on):
            idx_on = np.where(is_on)[0]
            start = idx_on[0]
            prev = idx_on[0]
            segments = []
            for idx in idx_on[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    segments.append((start, prev))
                    start = idx
                    prev = idx
            segments.append((start, prev))
            for start, end in segments:
                ax_alpha.axvspan(t_rel[start], t_rel[end], color="#E78FB3", alpha=0.45, zorder=0)

        m = np.isfinite(t_rel) & np.isfinite(alpha)
        t_rel_valid = t_rel[m]
        alpha_valid = alpha[m]
        ax_alpha.scatter(t_rel_valid, alpha_valid, s=14, color="0.5", alpha=0.35, edgecolors="none", label="Samples")

        if t_rel_valid.size >= 4:
            qbins = 8
            edges = np.quantile(t_rel_valid, np.linspace(0, 1, qbins + 1))
            idx = np.digitize(t_rel_valid, edges[1:-1], right=True)
            bin_x, bin_y = [], []
            for b in range(qbins):
                mask_b = idx == b
                if mask_b.sum() == 0:
                    continue
                bin_x.append(t_rel_valid[mask_b].mean())
                bin_y.append(alpha_valid[mask_b].mean())
            ax_alpha.plot(np.asarray(bin_x), np.asarray(bin_y), "-o", color="k", lw=2, ms=4, label="Binned mean")

        try:
            from scipy.stats import pearsonr

            if t_rel_valid.size >= 3:
                r, p = pearsonr(t_rel_valid, alpha_valid)
                ax_alpha.text(
                    0.02,
                    0.98,
                    f"r = {r:.2f}, p = {p:.3g}",
                    transform=ax_alpha.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                )
        except Exception:
            pass

        ai_band_patch = Rectangle((0, 0), 1, 1, facecolor="#E78FB3", edgecolor="none", alpha=0.45, label="AI override")
        handles_r, _ = ax_alpha.get_legend_handles_labels()
        handles_r.append(ai_band_patch)
        ax_alpha.legend(handles_r, [h.get_label() for h in handles_r], loc="lower right", frameon=True)
        ax_alpha.set_ylim(0, 1)
        ax_alpha.set_xlim(t_rel_valid.min() - 0.1, t_rel_valid.max() + 0.1)
        ax_alpha.set_xlabel("Time from trial start (ms)")
        ax_alpha.set_ylabel("Prior Confidence Index")
        ax_alpha.grid(True, alpha=0.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"figure2_fixed_obstacle_Monkey2_{SESSION_STEM}_trial{int(trial['trial']):03d}"
    outputs = []
    plt.tight_layout()
    save_kwargs = {
        "svg": {"bbox_inches": "tight"},
        "pdf": {"bbox_inches": "tight"},
        "png": {"dpi": 600, "bbox_inches": "tight"},
    }
    for ext in formats:
        kwargs = save_kwargs[ext]
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)

    if write_provenance:
        provenance = out_dir / f"{stem}_provenance.txt"
        provenance.write_text(
            "\n".join(
                [
                    "Figure 2 Fixed Obstacle example",
                    f"Monkey: {MONKEY}",
                    f"Experiment: {EXPERIMENT}",
                    f"Session: {SESSION_STEM}",
                    f"Trial: {int(trial['trial']):03d}",
                    f"Base trial file: {trial.get('source_base_file')}",
                    f"Matched AI file: {trial.get('source_ai_file')}",
                    f"Source trace: {trial.get('source_trace')}",
                    "Plotted data: recorded AvatarPosition samples from aiVelocities; "
                    "Prior Confidence Index is 1 - EntropyLb/Hmax with the Hmax estimate "
                    "used in the time-panel plotting code.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(provenance)
    return outputs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the Figure 2 Fixed Obstacle example trial.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--trial-id", type=int, default=TRIAL_ID)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "figures" / "Figure 2"))
    parser.add_argument("--formats", nargs="+", choices=("svg", "pdf", "png"), default=["svg"])
    parser.add_argument("--write-provenance", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    base_dir = load_base_dir(args.base_dir)
    trial = load_trial(base_dir, args.trial_id)
    outputs = make_figure(
        trial,
        Path(args.output_dir),
        formats=tuple(args.formats),
        write_provenance=args.write_provenance,
    )
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
