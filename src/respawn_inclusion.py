"""Respawn-specific trial inclusion rules for final manuscript analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG3_SOURCE_ROWS = (
    PROJECT_ROOT
    / "analysis_results"
    / "success_rate_verification"
    / "figure3_success_rate_source_rows.csv"
)

# Final Respawn manuscript analyses use the experimental answer-code pool below.
# answer 1 = successful trial. Unsuccessful Respawn analyses include only
# intended experimental outcome codes {3,5,6}. Codes {4,9,10} are excluded:
# they belong to the broader loader/obstacle-outcome pool rather than the final
# Respawn experimental failure set; in the verified Respawn sessions, code 4
# trials lack usable target-jump/condition information, and codes 9/10 are not
# present.
RESPAWN_ELIGIBLE_ANSWERS = {1, 3, 5, 6}
RESPAWN_UNSUCCESSFUL_ANSWERS = {3, 5, 6}
RESPAWN_EXCLUDED_ANSWERS = {4, 9, 10}


def get_value(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def answer_code(trial: Any) -> int | None:
    try:
        return int(get_value(trial, "answer", None))
    except Exception:
        return None


def is_respawn_eligible_answer(trial: Any) -> bool:
    return answer_code(trial) in RESPAWN_ELIGIBLE_ANSWERS


def respawn_outcome_label(trial: Any, success_label: str = "correct", failure_label: str = "incorrect") -> str:
    return success_label if answer_code(trial) == 1 else failure_label


def verified_respawn_source_files(monkey: str = "Monkey 1", experiment: str = "AI Respawn") -> set[str] | None:
    """Return verified Figure 3 Respawn session filenames, or None if unavailable."""
    if not FIG3_SOURCE_ROWS.exists():
        return None
    rows = pd.read_csv(FIG3_SOURCE_ROWS)
    rows = rows[(rows["monkey"].eq(monkey)) & (rows["experiment"].eq(experiment))]
    return set(rows["source_file"].dropna().astype(str))


def iter_final_respawn_trials(
    all_trials: Iterable[Iterable[Any]],
    pkl_files: Iterable[str],
    monkey: str,
    experiment: str = "AI Respawn",
):
    """Yield trials in the final Respawn analysis pool with their session metadata."""
    verified_files = verified_respawn_source_files(monkey, experiment)
    pkl_files_list = list(pkl_files)
    for session_index, session_trials in enumerate(all_trials, start=1):
        session_file = Path(pkl_files_list[session_index - 1]).name if session_index - 1 < len(pkl_files_list) else ""
        if verified_files is not None and session_file not in verified_files:
            continue
        for trial in session_trials:
            ans = answer_code(trial)
            if ans not in RESPAWN_ELIGIBLE_ANSWERS:
                continue
            jump_time = get_value(trial, "targetJumpTime", None)
            try:
                has_jump = np.isfinite(float(np.asarray(jump_time).reshape(-1)[0]))
            except Exception:
                has_jump = False
            if not has_jump:
                continue
            yield trial, session_index, session_file
