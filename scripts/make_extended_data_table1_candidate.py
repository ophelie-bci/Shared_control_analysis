"""Build an audit candidate for Extended Data Table 1.

This script preserves the previous Extended Data Table 1 file and writes
separate candidate/comparison outputs. It uses reproduced session-level success
summaries for success-rate statistics and retains the previous table's chance
values as legacy values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "tables" / "Extended Data Table 1"
AUDIT_DIR = PROJECT_ROOT / "analysis_results" / "success_rate_verification"

PREVIOUS_TABLE = TABLE_DIR / "extended_data_table_1_success_rate_by_task_excel_semicolon.csv"
SESSION_SUMMARY = AUDIT_DIR / "success_rate_audit_session_summary.csv"
FIGURE3_COMPARISON = AUDIT_DIR / "figure3_success_rate_comparison.csv"

CANDIDATE_EDITABLE = TABLE_DIR / "extended_data_table_1_success_rate_by_task_candidate_editable_semicolon.csv"
CANDIDATE_MACHINE = TABLE_DIR / "extended_data_table_1_success_rate_by_task_candidate_machine_readable.csv"
DIFFERENCES_CSV = TABLE_DIR / "extended_data_table_1_success_rate_by_task_candidate_cell_differences.csv"
METADATA_JSON = TABLE_DIR / "extended_data_table_1_success_rate_by_task_candidate_metadata.json"
REPORT_MD = TABLE_DIR / "extended_data_table_1_success_rate_by_task_candidate_report.md"

ORDER = [
    ("Monkey 1", "AI Obstacle"),
    ("Monkey 1", "AI Appearing Obstacle"),
    ("Monkey 1", "AI Appearing Obstacle 2"),
    ("Monkey 1", "AI Respawn"),
    ("Monkey 2", "AI Obstacle"),
    ("Monkey 2", "AI Appearing Obstacle"),
]

TASK_LABELS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle (6 months later)",
    "AI Respawn": "Respawn",
}

FIGURE3_FIELDS = {
    "BCI-only (mean +/- SD)": ("manuscript_bci_mean", "manuscript_bci_sd"),
    "Shared-control (mean +/- SD)": ("manuscript_shared_mean", "manuscript_shared_sd"),
    "Legacy chance level": ("manuscript_chance",),
}


def fmt_mean_sd(mean_pct: float, sd_pct: float) -> str:
    return f"{mean_pct:.0f} +/- {sd_pct:.0f}"


def fmt_gain(gain_pp: float) -> str:
    return f"{gain_pp:.1f}"


def fmt_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value == 0:
        return "<1e-300"
    exponent = int(np.floor(np.log10(abs(p_value))))
    mantissa = p_value / (10 ** exponent)
    return f"{mantissa:.2f} x 10^{exponent}"


def normalize_previous_columns(previous: pd.DataFrame) -> pd.DataFrame:
    out = previous.copy()
    if "Chance level" in out.columns:
        out = out.rename(columns={"Chance level": "Legacy chance level"})
    return out


def previous_lookup(previous: pd.DataFrame) -> dict[tuple[str, str], dict]:
    return {
        (str(row["Monkey"]), str(row["Task"])): row.to_dict()
        for _, row in previous.iterrows()
    }


def figure3_lookup(figure3: pd.DataFrame) -> dict[tuple[str, str], dict]:
    return {
        (str(row["monkey"]), str(row["experiment"])): row.to_dict()
        for _, row in figure3.iterrows()
    }


def build_candidate(previous: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_by_row = previous_lookup(previous)
    rows = []
    machine_rows = []
    for monkey, experiment in ORDER:
        task = TASK_LABELS[experiment]
        match = summary[summary["monkey"].eq(monkey) & summary["experiment"].eq(experiment)]
        if match.empty:
            raise ValueError(f"Missing session summary for {monkey} / {experiment}")
        row = match.iloc[0]
        legacy = legacy_by_row.get((monkey, task), {})
        legacy_chance = legacy.get("Legacy chance level", "")

        n_sessions = int(row["n_sessions"])
        bci_mean = float(row["bci_only_mean_pct"])
        bci_sd = float(row["bci_only_sd_pct"])
        shared_mean = float(row["shared_control_mean_pct"])
        shared_sd = float(row["shared_control_sd_pct"])
        gain = float(row["mean_gain_pp"])
        p_value = float(row["wilcoxon_two_sided_p"])

        rows.append(
            {
                "Monkey": monkey,
                "Task": task,
                "Number of sessions": n_sessions,
                "BCI-only (mean +/- SD)": fmt_mean_sd(bci_mean, bci_sd),
                "Shared-control (mean +/- SD)": fmt_mean_sd(shared_mean, shared_sd),
                "Legacy chance level": legacy_chance,
                "Shared-control Gain (pp)": fmt_gain(gain),
                "p-value (paired Wilcoxon)": fmt_p_value(p_value),
            }
        )
        machine_rows.append(
            {
                "monkey": monkey,
                "task": task,
                "experiment": experiment,
                "n_sessions": n_sessions,
                "bci_only_mean_pct": bci_mean,
                "bci_only_sample_sd_pct": bci_sd,
                "shared_control_mean_pct": shared_mean,
                "shared_control_sample_sd_pct": shared_sd,
                "mean_paired_gain_pp": gain,
                "wilcoxon_two_sided_p": p_value,
                "legacy_chance_level_pct": pd.to_numeric(pd.Series([legacy_chance]), errors="coerce").iloc[0],
                "legacy_chance_source": "previous Extended Data Table 1 CSV and legacy success-rate SVG annotations",
                "gain_definition": "mean of paired session-level shared-control minus BCI-only success rates, percentage points",
                "sd_ddof": 1,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(machine_rows)


def compare_cells(candidate: pd.DataFrame, previous: pd.DataFrame, figure3: pd.DataFrame) -> pd.DataFrame:
    previous = normalize_previous_columns(previous)
    prev = previous_lookup(previous)
    fig3 = figure3_lookup(figure3)
    rows = []
    for _, cand_row in candidate.iterrows():
        monkey = str(cand_row["Monkey"])
        task = str(cand_row["Task"])
        experiment = {v: k for k, v in TASK_LABELS.items()}[task]
        row_key = f"{monkey} | {task}"

        for column in candidate.columns:
            cand_value = cand_row[column]
            old_value = prev.get((monkey, task), {}).get(column, "")
            rows.append(
                {
                    "row": row_key,
                    "column": column,
                    "comparison_source": "previous_csv",
                    "candidate_value": cand_value,
                    "reference_value": old_value,
                    "difference_type": "none" if str(cand_value) == str(old_value) else "computational_or_formatting_difference",
                    "note": "" if str(cand_value) == str(old_value) else "Candidate differs from previous Extended Data Table 1 CSV.",
                }
            )

            manuscript_note = "Latest manuscript embedded Extended Data Table 1 was not found in this repository."
            rows.append(
                {
                    "row": row_key,
                    "column": column,
                    "comparison_source": "latest_manuscript_embedded_table",
                    "candidate_value": cand_value,
                    "reference_value": "",
                    "difference_type": "unresolved_missing_reference",
                    "note": manuscript_note,
                }
            )

            if column not in FIGURE3_FIELDS:
                rows.append(
                    {
                        "row": row_key,
                        "column": column,
                        "comparison_source": "figure3_annotations",
                        "candidate_value": cand_value,
                        "reference_value": "",
                        "difference_type": "not_applicable",
                        "note": "Figure 3 annotations do not contain this table cell.",
                    }
                )
                continue

            fig_row = fig3.get((monkey, experiment), {})
            field_names = FIGURE3_FIELDS[column]
            if len(field_names) == 2:
                reference = f"{int(fig_row.get(field_names[0], np.nan))} +/- {int(fig_row.get(field_names[1], np.nan))}"
            else:
                value = fig_row.get(field_names[0], "")
                reference = "" if value == "" or (isinstance(value, float) and math.isnan(value)) else str(int(float(value)))
            difference = "none" if str(cand_value) == reference else "manuscript_annotation_difference"
            rows.append(
                {
                    "row": row_key,
                    "column": column,
                    "comparison_source": "figure3_annotations",
                    "candidate_value": cand_value,
                    "reference_value": reference,
                    "difference_type": difference,
                    "note": "" if difference == "none" else "Candidate table cell differs from copied Figure 3 annotation.",
                }
            )
    return pd.DataFrame(rows)


def write_report(candidate: pd.DataFrame, differences: pd.DataFrame) -> None:
    diff_rows = differences[differences["difference_type"].ne("none")]
    computational = diff_rows[
        diff_rows["difference_type"].eq("computational_or_formatting_difference")
        & diff_rows["comparison_source"].eq("previous_csv")
    ]
    figure = diff_rows[diff_rows["difference_type"].eq("manuscript_annotation_difference")]
    manuscript_missing = diff_rows[diff_rows["comparison_source"].eq("latest_manuscript_embedded_table")]

    lines = [
        "# Extended Data Table 1 Candidate Audit",
        "",
        "This candidate uses reproduced session-level summaries for session counts, means, sample SDs, paired mean gains, and paired two-sided Wilcoxon p-values.",
        "Chance values are retained from the previous Extended Data Table 1 and are marked as legacy values.",
        "",
        "## Candidate table",
        "",
        markdown_table(candidate),
        "",
        "## Previous CSV comparison",
        "",
    ]
    if computational.empty:
        lines.append("No cell differences from the previous CSV after renaming `Chance level` to `Legacy chance level`.")
    else:
        lines.extend(
            f"- {row['row']} / {row['column']}: previous `{row['reference_value']}` vs candidate `{row['candidate_value']}`"
            for _, row in computational.iterrows()
        )
    lines.extend(["", "## Latest manuscript embedded table comparison", ""])
    if manuscript_missing.empty:
        lines.append("No manuscript-table differences detected.")
    else:
        lines.append("The latest manuscript embedded Extended Data Table 1 was not found inside this repository, so every manuscript-table cell comparison is unresolved.")
    lines.extend(["", "## Figure 3 annotation comparison", ""])
    if figure.empty:
        lines.append("No differences for cells that also appear in Figure 3 annotations.")
    else:
        lines.extend(
            f"- {row['row']} / {row['column']}: Figure 3 annotation `{row['reference_value']}` vs candidate `{row['candidate_value']}`"
            for _, row in figure.iterrows()
        )
    lines.extend(
        [
            "",
            "## Original chance procedure",
            "",
            "Legacy chance values come from the original success-rate outputs. The original helper performs target permutations using successful recorded trajectories, with the obstacle list fixed in recorded-trial order.",
            "",
            "Suggested Methods wording: Random-target chance was estimated by replaying successful recorded trajectories while randomly permuting the empirical target labels across trials; obstacle positions remained fixed to the recorded trials, and success was scored with the same target-entry and obstacle-collision criteria used by the original success-rate analysis.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in cols]
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    previous = normalize_previous_columns(pd.read_csv(PREVIOUS_TABLE, sep=";"))
    summary = pd.read_csv(SESSION_SUMMARY)
    figure3 = pd.read_csv(FIGURE3_COMPARISON)

    candidate, machine = build_candidate(previous, summary)
    differences = compare_cells(candidate, previous, figure3)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(CANDIDATE_EDITABLE, sep=";", index=False)
    machine.to_csv(CANDIDATE_MACHINE, index=False)
    differences.to_csv(DIFFERENCES_CSV, index=False)
    metadata = {
        "previous_table": str(PREVIOUS_TABLE.relative_to(PROJECT_ROOT)),
        "session_summary": str(SESSION_SUMMARY.relative_to(PROJECT_ROOT)),
        "figure3_comparison": str(FIGURE3_COMPARISON.relative_to(PROJECT_ROOT)),
        "candidate_editable": str(CANDIDATE_EDITABLE.relative_to(PROJECT_ROOT)),
        "candidate_machine_readable": str(CANDIDATE_MACHINE.relative_to(PROJECT_ROOT)),
        "legacy_chance_source": "previous Extended Data Table 1 CSV and legacy success-rate SVG annotations",
        "legacy_chance_procedure": "target permutations using successful recorded trajectories with obstacles fixed in recorded-trial order",
        "new_joint_pair_chance_estimates_used": False,
        "latest_manuscript_embedded_table_source": "not found in repository",
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(candidate, differences)

    for path in (CANDIDATE_EDITABLE, CANDIDATE_MACHINE, DIFFERENCES_CSV, METADATA_JSON, REPORT_MD):
        print(f"[INFO] wrote {path}")


if __name__ == "__main__":
    main()
