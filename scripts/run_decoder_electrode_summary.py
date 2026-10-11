"""
Summarize decoder-contributing electrodes for the AI-paper dataset.

Decoder-contributing electrodes are defined from each online-session CCF as
`elec*` channels whose spike threshold level divided by 4 is greater than -60,
matching the threshold rule used by src.load.load_files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Appearing Obstacle 2",
    "AI Respawn",
]
AREAS = ["M1", "PMv", "PMd"]


def load_base_dir(cli_base_dir: str | None) -> str:
    if cli_base_dir:
        return cli_base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config.get("base_dir")
    if not base_dir:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(base_dir)


def _subject_from_filename(path: Path) -> str:
    match = re.search(r"_(Maui|Loki)_", path.name, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else ""


def _electrode_number(label: str) -> int | None:
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _area_for_electrode(monkey: str, subject: str, electrode_number: int) -> str:
    # Historical maps in the surrounding analysis code name the animals as
    # Maui and Loki. The dataset folders expose them as Monkey 1 and Monkey 2.
    animal = subject or monkey
    if animal in ("Maui", "Monkey 1"):
        if 1 <= electrode_number <= 64 or 97 <= electrode_number <= 128:
            return "M1"
        if 65 <= electrode_number <= 96 or 129 <= electrode_number <= 192:
            return "PMv"
        if 193 <= electrode_number <= 297:
            return "PMd"
    elif animal in ("Loki", "Monkey 2"):
        if 65 <= electrode_number <= 96 or 129 <= electrode_number <= 192:
            return "M1"
        if 1 <= electrode_number <= 64 or 97 <= electrode_number <= 128:
            return "PMv"
        if 193 <= electrode_number <= 297:
            return "PMd"
    return "unmapped"


def _decoder_electrodes_from_ccf(path: Path) -> list[str]:
    root = ET.fromstring(path.read_text(encoding="utf-8", errors="ignore"))
    electrodes = []
    for chan_info in root.findall(".//ChanInfo_item"):
        label_node = chan_info.find("label")
        threshold_node = chan_info.find(".//spike/threshold/level")
        if label_node is None or threshold_node is None:
            continue
        label = (label_node.text or "").strip()
        threshold_text = (threshold_node.text or "").strip()
        if not label.startswith("elec"):
            continue
        try:
            threshold = int(threshold_text) / 4.0
        except ValueError:
            continue
        if threshold > -60.0:
            electrodes.append(label)
    return electrodes


def collect_session_rows(base_dir: str, monkeys: list[str], experiments: list[str]) -> pd.DataFrame:
    rows = []
    base_path = Path(base_dir)
    for monkey in monkeys:
        for experiment in experiments:
            experiment_dir = base_path / monkey / experiment
            if not experiment_dir.exists():
                print(f"Skipping {monkey} {experiment}: directory not found.")
                continue
            ccf_files = sorted(experiment_dir.glob("*.ccf"))
            for ccf_path in ccf_files:
                subject = _subject_from_filename(ccf_path)
                electrodes = _decoder_electrodes_from_ccf(ccf_path)
                counts = {area: 0 for area in AREAS}
                unmapped = 0
                for label in electrodes:
                    number = _electrode_number(label)
                    if number is None:
                        unmapped += 1
                        continue
                    area = _area_for_electrode(monkey, subject, number)
                    if area in counts:
                        counts[area] += 1
                    else:
                        unmapped += 1
                rows.append({
                    "monkey": monkey,
                    "subject": subject,
                    "experiment": experiment,
                    "ccf_file": str(ccf_path.relative_to(base_path)),
                    "session": ccf_path.stem,
                    "n_decoder_electrodes": len(electrodes),
                    "n_M1": counts["M1"],
                    "n_PMv": counts["PMv"],
                    "n_PMd": counts["PMd"],
                    "n_unmapped": unmapped,
                })
    return pd.DataFrame(rows)


def _summary_stats(values: pd.Series) -> dict:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "sem": float(vals.sem()) if len(vals) > 1 else 0.0,
        "median": float(vals.median()),
        "min": int(vals.min()),
        "max": int(vals.max()),
    }


def summarize_by_monkey(session_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    area_rows = []
    for monkey, group in session_rows.groupby("monkey", sort=False):
        total = _summary_stats(group["n_decoder_electrodes"])
        row = {
            "monkey": monkey,
            "subject": ", ".join(sorted(set(group["subject"].dropna()))),
            "n_sessions": int(len(group)),
            "mean_decoder_electrodes": total["mean"],
            "sd_decoder_electrodes": total["sd"],
            "sem_decoder_electrodes": total["sem"],
            "median_decoder_electrodes": total["median"],
            "min_decoder_electrodes": total["min"],
            "max_decoder_electrodes": total["max"],
        }
        for area in AREAS:
            stats = _summary_stats(group[f"n_{area}"])
            row[f"mean_{area}_electrodes"] = stats["mean"]
            row[f"sd_{area}_electrodes"] = stats["sd"]
            area_rows.append({
                "monkey": monkey,
                "subject": row["subject"],
                "area": area,
                "n_sessions": int(len(group)),
                "mean_electrodes": stats["mean"],
                "sd_electrodes": stats["sd"],
                "sem_electrodes": stats["sem"],
                "median_electrodes": stats["median"],
                "min_electrodes": stats["min"],
                "max_electrodes": stats["max"],
            })
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(area_rows)


def write_statement(summary: pd.DataFrame, area_summary: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "tables" / "decoder_electrode_summary_statement.md"
    by_monkey = {row["monkey"]: row for _, row in summary.iterrows()}
    monkey_1 = by_monkey.get("Monkey 1")
    monkey_2 = by_monkey.get("Monkey 2")
    lines = [
        "# Decoder Electrode Summary",
        "",
        "Decoder-contributing electrodes were defined from online-session CCF files as `elec*` channels with spike threshold level / 4 greater than -60.",
        "",
    ]
    if monkey_1 is not None and monkey_2 is not None:
        lines.extend([
            "Sentence-ready wording:",
            "",
            (
                "Neural activity was recorded from primary motor cortex (M1), dorsal premotor cortex (PMd), "
                "and ventral premotor cortex (PMv), with "
                f"{monkey_1['mean_decoder_electrodes']:.1f} +/- {monkey_1['sd_decoder_electrodes']:.1f} "
                f"electrodes contributing to the decoder in Monkey 1 and "
                f"{monkey_2['mean_decoder_electrodes']:.1f} +/- {monkey_2['sd_decoder_electrodes']:.1f} "
                "in Monkey 2."
            ),
            "",
        ])
    lines.append("Area counts are summarized separately because the sentence above reports total decoder-contributing electrodes.")
    lines.append("")
    for _, row in area_summary.iterrows():
        lines.append(
            f"- {row['monkey']} {row['area']}: {row['mean_electrodes']:.1f} +/- {row['sd_electrodes']:.1f} electrodes "
            f"across {int(row['n_sessions'])} sessions."
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_decoder_electrode_summary(
    base_dir: str,
    monkeys: list[str],
    experiments: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    session_rows = collect_session_rows(base_dir, monkeys, experiments)
    if session_rows.empty:
        raise RuntimeError("No online-session CCF files found for the requested dataset.")
    summary, area_summary = summarize_by_monkey(session_rows)
    session_rows.to_csv(tables_dir / "decoder_electrode_summary_by_session.csv", index=False)
    summary.to_csv(tables_dir / "decoder_electrode_summary_by_monkey.csv", index=False)
    area_summary.to_csv(tables_dir / "decoder_electrode_summary_by_area.csv", index=False)
    write_statement(summary, area_summary)
    return session_rows, summary, area_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize decoder-contributing electrodes for the AI-paper dataset.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args(argv)
    run_decoder_electrode_summary(load_base_dir(args.base_dir), args.monkeys, args.experiments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
