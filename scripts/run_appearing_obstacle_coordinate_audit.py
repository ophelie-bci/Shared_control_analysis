"""
Audit nominal, controller-side, and actually appeared obstacle coordinates.

The Appearing Obstacle task has three related coordinates in the logs:

- ShowTarget.obstaclePosition: nominal/configured trial metadata.
- Obstacle0: obstacle sent to the AI controller obstacle list.
- obstacleAppeared.obstaclePosition: rendered obstacle position emitted when
  the cursor crosses the appearance threshold.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = None

DATASETS = [
    ("Original Loki", "Monkey 1", "AI Appearing Obstacle"),
    ("Original Maui", "Monkey 3", "AI Appearing Obstacle"),
    ("Follow-up Maui", "Monkey 3", "AI Appearing Obstacle 2"),
]


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)


def _literal_dict(text: str) -> dict:
    safe = re.sub(r"\bnan\b", "None", text)
    return ast.literal_eval(safe)


def _vec3(value) -> tuple[float, float, float]:
    if value is None:
        return (math.nan, math.nan, math.nan)
    try:
        arr = list(value)
    except TypeError:
        return (math.nan, math.nan, math.nan)
    if len(arr) < 3:
        return (math.nan, math.nan, math.nan)
    out = []
    for item in arr[:3]:
        out.append(float(item) if item is not None else math.nan)
    return tuple(out)


def _finite_vec(vec: tuple[float, float, float]) -> bool:
    return all(math.isfinite(x) for x in vec)


def _xz_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    if not (_finite_vec(a) and _finite_vec(b)):
        return math.nan
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))


def _parse_log(path: Path, dataset: str, monkey_folder: str, experiment: str) -> list[dict]:
    rows = []
    current = None
    trial_index = -1
    show_re = re.compile(r"recv \[ShowTarget\]: (\{.*\})")
    appeared_re = re.compile(r"recv \[obstacleAppeared\]: (\{.*\})")
    obstacle0_re = re.compile(r"Obstacle0:\[([^\]]+)\]")

    def flush() -> None:
        if current is not None:
            rows.append(current)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            show_match = show_re.search(line)
            if show_match:
                flush()
                trial_index += 1
                payload = _literal_dict(show_match.group(1))
                current = {
                    "dataset": dataset,
                    "source_monkey_folder": monkey_folder,
                    "experiment": experiment,
                    "log_file": path.name,
                    "trial_index_in_log": trial_index,
                    "showtarget_line": line_number,
                    "target_x": _vec3(payload.get("targetPosition"))[0],
                    "target_y": _vec3(payload.get("targetPosition"))[1],
                    "target_z": _vec3(payload.get("targetPosition"))[2],
                    "show_obstacle_x": _vec3(payload.get("obstaclePosition"))[0],
                    "show_obstacle_y": _vec3(payload.get("obstaclePosition"))[1],
                    "show_obstacle_z": _vec3(payload.get("obstaclePosition"))[2],
                    "controller_obstacle_x": math.nan,
                    "controller_obstacle_y": math.nan,
                    "controller_obstacle_z": math.nan,
                    "controller_obstacle_line": math.nan,
                    "appeared_obstacle_x": math.nan,
                    "appeared_obstacle_y": math.nan,
                    "appeared_obstacle_z": math.nan,
                    "appeared_obstacle_line": math.nan,
                }
                continue

            if current is None:
                continue

            obstacle0_match = obstacle0_re.search(line)
            if obstacle0_match:
                vals = [float(x.strip()) for x in obstacle0_match.group(1).split(",")]
                current["controller_obstacle_x"] = vals[0]
                current["controller_obstacle_y"] = vals[1]
                current["controller_obstacle_z"] = vals[2]
                current["controller_obstacle_line"] = line_number
                continue

            appeared_match = appeared_re.search(line)
            if appeared_match:
                payload = _literal_dict(appeared_match.group(1))
                appeared = _vec3(payload.get("obstaclePosition"))
                current["appeared_obstacle_x"] = appeared[0]
                current["appeared_obstacle_y"] = appeared[1]
                current["appeared_obstacle_z"] = appeared[2]
                current["appeared_obstacle_line"] = line_number

    flush()
    return rows


def run_analysis(base_dir: str = DEFAULT_BASE_DIR) -> dict[str, Path]:
    rows = []
    for dataset, monkey_folder, experiment in DATASETS:
        log_dir = Path(base_dir) / monkey_folder / experiment / "aiLog"
        for log_path in sorted(log_dir.glob("*.log")):
            rows.extend(_parse_log(log_path, dataset, monkey_folder, experiment))

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No log rows were parsed.")

    show_cols = ["show_obstacle_x", "show_obstacle_y", "show_obstacle_z"]
    ctrl_cols = ["controller_obstacle_x", "controller_obstacle_y", "controller_obstacle_z"]
    app_cols = ["appeared_obstacle_x", "appeared_obstacle_y", "appeared_obstacle_z"]
    data["has_show_obstacle"] = data[show_cols].notna().all(axis=1)
    data["has_controller_obstacle"] = data[ctrl_cols].notna().all(axis=1)
    data["has_appeared_obstacle"] = data[app_cols].notna().all(axis=1)
    data["target_midpoint_x_from_origin"] = data["target_x"] / 2.0
    data["target_midpoint_z_from_origin"] = data["target_z"] / 2.0
    midpoint = data[["target_midpoint_x_from_origin", "target_y", "target_midpoint_z_from_origin"]].to_numpy()
    data["controller_to_show_xz_distance_m"] = [
        _xz_distance(tuple(c), tuple(s)) for c, s in zip(data[ctrl_cols].to_numpy(), data[show_cols].to_numpy())
    ]
    data["appeared_to_show_xz_distance_m"] = [
        _xz_distance(tuple(a), tuple(s)) for a, s in zip(data[app_cols].to_numpy(), data[show_cols].to_numpy())
    ]
    data["appeared_to_controller_xz_distance_m"] = [
        _xz_distance(tuple(a), tuple(c)) for a, c in zip(data[app_cols].to_numpy(), data[ctrl_cols].to_numpy())
    ]
    data["show_to_target_midpoint_xz_distance_m"] = [
        _xz_distance(tuple(s), tuple(m)) for s, m in zip(data[show_cols].to_numpy(), midpoint)
    ]
    data["controller_to_target_midpoint_xz_distance_m"] = [
        _xz_distance(tuple(c), tuple(m)) for c, m in zip(data[ctrl_cols].to_numpy(), midpoint)
    ]
    data["appeared_to_target_midpoint_xz_distance_m"] = [
        _xz_distance(tuple(a), tuple(m)) for a, m in zip(data[app_cols].to_numpy(), midpoint)
    ]

    out_dir = PROJECT_ROOT / "tables" / "Hidden prior follow-up"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "appearing_obstacle_log_coordinate_audit.csv"
    summary_path = out_dir / "appearing_obstacle_log_coordinate_audit_summary.csv"
    note_path = out_dir / "appearing_obstacle_log_coordinate_audit.md"

    data.to_csv(csv_path, index=False)

    summaries = []
    for dataset, sub in data.groupby("dataset", sort=False):
        appeared = sub[sub["has_appeared_obstacle"]]
        controller = sub[sub["has_controller_obstacle"]]
        both = sub[sub["has_appeared_obstacle"] & sub["has_controller_obstacle"]]
        summaries.append(
            {
                "dataset": dataset,
                "n_showtarget_trials": len(sub),
                "n_show_obstacle_metadata": int(sub["has_show_obstacle"].sum()),
                "n_controller_obstacle0": int(sub["has_controller_obstacle"].sum()),
                "n_obstacle_appeared_events": int(sub["has_appeared_obstacle"].sum()),
                "median_controller_to_show_xz_m": float(controller["controller_to_show_xz_distance_m"].median())
                if len(controller)
                else math.nan,
                "median_appeared_to_show_xz_m": float(appeared["appeared_to_show_xz_distance_m"].median())
                if len(appeared)
                else math.nan,
                "median_appeared_to_controller_xz_m": float(both["appeared_to_controller_xz_distance_m"].median())
                if len(both)
                else math.nan,
                "max_show_to_target_midpoint_xz_m": float(
                    sub.loc[sub["has_show_obstacle"], "show_to_target_midpoint_xz_distance_m"].max()
                )
                if int(sub["has_show_obstacle"].sum())
                else math.nan,
                "max_controller_to_target_midpoint_xz_m": float(
                    sub.loc[sub["has_controller_obstacle"], "controller_to_target_midpoint_xz_distance_m"].max()
                )
                if int(sub["has_controller_obstacle"].sum())
                else math.nan,
                "median_appeared_to_target_midpoint_xz_m": float(
                    appeared["appeared_to_target_midpoint_xz_distance_m"].median()
                )
                if len(appeared)
                else math.nan,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_path, index=False)

    lines = [
        "# Appearing-obstacle log coordinate audit",
        "",
        "Date: 2026-08-31",
        "",
        "## Result",
        "",
        "The earlier interpretation was too strong. The original logs show that the controller-side `Obstacle0` entry matches the nominal `ShowTarget.obstaclePosition`, but the actually rendered obstacle is logged separately as `recv [obstacleAppeared]` and is not the same coordinate.",
        "",
        "The nominal `ShowTarget.obstaclePosition` is exactly halfway in x/z from the origin/start position to the target position in all non-NaN cases. In original sessions, the controller-side `Obstacle0` matched that same nominal midpoint exactly. The actually appeared obstacle was not exactly at the midpoint.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Example",
        "",
        "Original Maui, first appearing-obstacle trial:",
        "",
        "```text",
        "ShowTarget.obstaclePosition = [-3.5, 0, 3]",
        "Obstacle0 = [-3.5, 0.0, 3.0]",
        "obstacleAppeared.obstaclePosition = [-2.6586, 0.3000, 2.6782]",
        "targetPosition = [-7.0, 0.75, 6.0]",
        "target midpoint from origin = [-3.5, 0.75, 3.0]",
        "```",
        "",
        "Thus the hidden controller obstacle was at the nominal midpoint location, not at the exact rendered location that appeared later.",
        "",
        "## Corrected wording",
        "",
        "Use:",
        "",
        "> In the original Appearing Obstacle sessions, when an appearing-obstacle trial was configured, the shared-control controller received a non-rendered `Obstacle0` entry at the nominal `ShowTarget.obstaclePosition`, which was exactly halfway between the start position and target in x/z. The obstacle later rendered to the monkey was logged separately as `obstacleAppeared.obstaclePosition` and was not exactly at this midpoint, consistent with the rendered obstacle being computed from cursor position at the moment of appearance. Thus the hidden controller obstacle provided advance information about the upcoming obstacle region, but not the exact trial-specific appeared coordinate. In the six-month follow-up logs, the nominal future `obstaclePosition` remained in the trial metadata, but no `Obstacle0` controller-list entries were found.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"csv": csv_path, "summary": summary_path, "note": note_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    args = parser.parse_args()
    args.base_dir = load_base_dir(args.base_dir)
    outputs = run_analysis(args.base_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
