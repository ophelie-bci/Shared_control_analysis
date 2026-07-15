"""
Stage the Figure 5C innovation overlay panel for the AI paper artifact set.

This panel was already generated for a manuscript/poster figure set outside the
Github AI paper dataset. The original analysis scripts live in the
Neurophysiology_paper script folder and are comparatively broad; this wrapper
copies an explicitly supplied Figure 5C artifact into this paper's figures
folder and writes source provenance beside it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "Figure 5"
DEFAULT_OUTPUT_NAME = "figure5C_innovation_correct_vs_incorrect_overlay.png"

SOURCE_NOTES = [
    "Staged source artifact:",
    "",
    "",
    "Likely generating/related scripts:",
    "navbcidecode/bcidecode/scripts/Neurophysiology_paper/scripts/poster_innovation_figures.py",
    "  - fig_correct_vs_incorrect(traces_collapsed, title, out_path=None)",
    "navbcidecode/bcidecode/scripts/Neurophysiology_paper/scripts/run_innovation_analysis.py",
    "  - related session-equal innovation Figure C helper",
    "",
    "Panel content:",
    "Baseline-corrected innovation magnitude aligned to perturbation time, split by",
    "correct/incorrect outcome and AI_OFF/AI_ON condition.",
]


def build_figure5c(source: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    if not source.exists():
        raise FileNotFoundError(
            f"Figure 5C source artifact was not found: {source}. "
            "Regenerate it with the Neurophysiology_paper innovation scripts or pass --source."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_NAME
    shutil.copy2(source, output_path)

    provenance_path = output_dir / "figure5C_innovation_correct_vs_incorrect_overlay_provenance.txt"
    notes = SOURCE_NOTES.copy()
    notes[1] = str(source)
    provenance_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Saved Figure 5C: {output_path}")
    print(f"Wrote provenance: {provenance_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the manuscript Figure 5C innovation overlay panel.")
    parser.add_argument("--source", required=True, help="Existing generated Figure 5C PNG.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Destination figure directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return build_figure5c(Path(args.source), Path(args.output_dir))


if __name__ == "__main__":
    main()
