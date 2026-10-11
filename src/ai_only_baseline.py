"""AI-only and candidate-set-size baseline helpers.

The strict AI-only baseline must be computed from autonomous-controller replay
outputs with no decoded BCI command and no BCI-derived history. This module
therefore evaluates replay rows; it does not infer AI-only performance from
recorded shared-control trajectories.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITION_AI_ONLY = "ai_only"
CONDITION_SHARED = "shared_control"
CONDITION_ORACLE = "oracle_autonomy"

DEFAULT_TARGETS = (
    "left",
    "slight_left",
    "straight",
    "slight_right",
    "right",
)


def normalize_target(value) -> str:
    """Convert labels or coordinate-like values into stable string keys."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        return text.replace(" ", "")
    return text


def parse_bool(value) -> bool:
    """Parse common boolean encodings used in CSV exports."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "success", "hit", "collision", "failed"}


def canonical_condition(value: str, k: int | None = None) -> str:
    """Normalize condition names while labeling K=1 AI-only as oracle autonomy."""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ai": CONDITION_AI_ONLY,
        "ai_only": CONDITION_AI_ONLY,
        "autonomy": CONDITION_AI_ONLY,
        "autonomous": CONDITION_AI_ONLY,
        "bci_ai": CONDITION_SHARED,
        "bci+ai": CONDITION_SHARED,
        "shared": CONDITION_SHARED,
        "shared_control": CONDITION_SHARED,
        "oracle": CONDITION_ORACLE,
        "goal_revealed": CONDITION_ORACLE,
        "goal_revealed_oracle": CONDITION_ORACLE,
        "oracle_autonomy": CONDITION_ORACLE,
    }
    condition = aliases.get(text, text)
    if condition == CONDITION_AI_ONLY and k == 1:
        return CONDITION_ORACLE
    return condition


def enumerate_candidate_sets(true_target: str, all_targets: Iterable[str] = DEFAULT_TARGETS) -> dict[int, list[tuple[str, ...]]]:
    """Enumerate all candidate sets of size K that include the true target."""
    true_target = normalize_target(true_target)
    targets = [normalize_target(t) for t in all_targets]
    distractors = [t for t in targets if t != true_target]
    out: dict[int, list[tuple[str, ...]]] = {}
    for k in range(1, len(targets) + 1):
        out[k] = [
            tuple([true_target, *combo])
            for combo in combinations(distractors, k - 1)
        ]
    return out


def load_replay_rows(path: Path) -> pd.DataFrame:
    """Load replay rows and normalize the schema used by the summaries."""
    raw = pd.read_csv(path)
    return normalize_replay_rows(raw)


def normalize_replay_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized replay table with required analysis columns."""
    df = raw.copy()
    lower_to_original = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name.lower() in lower_to_original:
                return lower_to_original[name.lower()]
        return None

    true_col = pick("true_target", "cued_target", "target", "target_label")
    reached_col = pick("reached_target", "selected_target", "ai_reached_target", "outcome_target")
    condition_col = pick("condition", "controller", "mode")
    k_col = pick("k", "candidate_set_size", "n_candidates", "num_candidates")

    missing = [
        name
        for name, col in {
            "condition": condition_col,
            "candidate_set_size": k_col,
            "true_target": true_col,
            "reached_target/selected_target": reached_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(
            "Replay CSV is missing required columns: "
            + ", ".join(missing)
            + ". See tables/ai_only_candidate_replay_rows_template.csv."
        )

    out = pd.DataFrame()
    for optional in ("monkey", "experiment", "session", "trial", "candidate_targets"):
        col = pick(optional)
        out[optional] = df[col] if col is not None else ""

    out["candidate_set_size"] = df[k_col].astype(int)
    out["true_target"] = df[true_col].map(normalize_target)
    out["reached_target"] = df[reached_col].map(normalize_target)
    out["condition"] = [
        canonical_condition(cond, k)
        for cond, k in zip(df[condition_col], out["candidate_set_size"])
    ]

    success_col = pick("success", "hit_true_target", "reached_true_target")
    if success_col is None:
        out["success"] = out["reached_target"] == out["true_target"]
    else:
        out["success"] = df[success_col].map(parse_bool)

    collision_col = pick("collision", "collided", "hit_obstacle")
    out["collision"] = df[collision_col].map(parse_bool) if collision_col else False

    failure_col = pick("failure", "failed")
    if failure_col:
        out["failure"] = df[failure_col].map(parse_bool)
    else:
        out["failure"] = ~out["success"] | out["collision"]

    return out


def summarize_replay_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize success, reached target, failure/collision, and the critical K=5 baseline."""
    group_cols = ["condition", "candidate_set_size", "monkey", "experiment"]
    summary = (
        rows.groupby(group_cols, dropna=False)
        .agg(
            n_trials=("success", "size"),
            n_success=("success", "sum"),
            success_rate=("success", "mean"),
            n_collision=("collision", "sum"),
            collision_rate=("collision", "mean"),
            n_failure=("failure", "sum"),
            failure_rate=("failure", "mean"),
        )
        .reset_index()
    )
    summary["random_target_chance"] = 1.0 / summary["candidate_set_size"].astype(float)

    reached = (
        rows.groupby(group_cols + ["reached_target"], dropna=False)
        .size()
        .rename("n_reached")
        .reset_index()
    )
    reached["reached_fraction"] = reached["n_reached"] / reached.groupby(group_cols)["n_reached"].transform("sum")

    critical_k5 = _critical_k5_comparison(summary)
    paired = _shared_minus_ai_only(summary)
    return summary, reached, critical_k5, paired


def _critical_k5_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the actual-experiment AI-only K=5 comparison against random-target chance."""
    k5 = summary[
        (summary["candidate_set_size"] == 5)
        & (summary["condition"].isin([CONDITION_AI_ONLY, CONDITION_SHARED]))
    ].copy()
    if k5.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "monkey",
                "experiment",
                "n_trials",
                "success_rate",
                "random_target_chance",
                "success_minus_random_target_chance",
                "collision_rate",
                "failure_rate",
            ]
        )
    k5["success_minus_random_target_chance"] = k5["success_rate"] - k5["random_target_chance"]
    return k5[
        [
            "condition",
            "monkey",
            "experiment",
            "n_trials",
            "success_rate",
            "random_target_chance",
            "success_minus_random_target_chance",
            "collision_rate",
            "failure_rate",
        ]
    ]


def _shared_minus_ai_only(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute optional shared-control advantage over strict AI-only when paired replay rows exist."""
    key_cols = ["candidate_set_size", "monkey", "experiment"]
    wide = summary.pivot_table(
        index=key_cols,
        columns="condition",
        values="success_rate",
        aggfunc="mean",
    ).reset_index()
    if CONDITION_SHARED not in wide or CONDITION_AI_ONLY not in wide:
        return pd.DataFrame(columns=key_cols + ["shared_minus_ai_only"])
    wide["shared_minus_ai_only"] = wide[CONDITION_SHARED] - wide[CONDITION_AI_ONLY]
    return wide[key_cols + [CONDITION_AI_ONLY, CONDITION_SHARED, "shared_minus_ai_only"]]


def plot_candidate_set_success(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot success as a function of candidate-set size for each condition."""
    import matplotlib.pyplot as plt

    if summary.empty:
        return

    pooled = (
        summary.groupby(["condition", "candidate_set_size"], dropna=False)
        .apply(lambda g: pd.Series({
            "success_rate": np.average(g["success_rate"], weights=g["n_trials"]),
            "n_trials": g["n_trials"].sum(),
        }))
        .reset_index()
    )

    labels = {
        CONDITION_AI_ONLY: "AI-only",
        CONDITION_SHARED: "BCI+AI shared control (optional K sweep)",
        CONDITION_ORACLE: "Goal-revealed/oracle autonomy",
    }
    colors = {
        CONDITION_AI_ONLY: "#4C78A8",
        CONDITION_SHARED: "#F58518",
        CONDITION_ORACLE: "#54A24B",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for condition, sub in pooled.groupby("condition"):
        sub = sub.sort_values("candidate_set_size")
        ax.plot(
            sub["candidate_set_size"],
            sub["success_rate"] * 100.0,
            marker="o",
            linewidth=2.0,
            label=labels.get(condition, condition),
            color=colors.get(condition),
        )

    ks = np.arange(1, int(pooled["candidate_set_size"].max()) + 1)
    ax.plot(ks, 100.0 / ks, color="#777777", linestyle="--", linewidth=1.5, label="Random-target chance")
    ax.set_xlabel("Candidate targets, K")
    ax.set_ylabel("Success at cued target (%)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_template(path: Path) -> None:
    """Write a replay-row CSV template documenting the required schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame([
        {
            "monkey": "Monkey 1",
            "experiment": "AI Respawn",
            "session": "session_id",
            "trial": 1,
            "condition": "ai_only",
            "candidate_set_size": 5,
            "candidate_targets": "left|slight_left|straight|slight_right|right",
            "true_target": "straight",
            "reached_target": "straight",
            "success": True,
            "collision": False,
            "failure": False,
        }
    ])
    template.to_csv(path, index=False)


def write_analysis_note(path: Path) -> None:
    """Write manuscript-facing analysis notes for the strict baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# AI-only baseline and candidate-set-size analysis

## Definition

The strict AI-only baseline is defined as: environment, candidate targets, and
the autonomous controller, with no decoded BCI command and no BCI-derived command
history. It must not retain a history derived from previous BCI commands, because
that would leak neural information into the autonomous condition.

For each replayed trial, measure:

- success at the hidden cued target;
- which candidate target the autonomous controller reaches;
- collision and failure rate.

The critical strict-baseline result is AI-only at K = 5, where all five targets
are available, the cued target is hidden, and there is no neural command or
history. This is the number to compare against random-target chance, BCI-only,
and the actual shared-control system.

The existing permutation-derived baseline should be reported as
random-target chance, not simply chance. It tests random target selection, not
autonomous-controller performance.

## Candidate-set-size sweep

For every trial, evaluate candidate-set size K = 1, 2, 3, 4, 5. Always include
the true cued target and include K - 1 distractors from the remaining targets.
Because this task has five targets, all distractor combinations can be evaluated
exhaustively.

K = 1 is not ordinary AI-only; it is goal-revealed/oracle autonomy, because the
controller is given the true target. K = 5 corresponds to the actual experimental
condition.

The main curve to generate first is P_AI-only(K). A shared-control K-sweep,
P_shared(K), is useful only if replaying the actual BCI commands under modified
candidate sets is explicitly performed; it is a separate counterfactual from the
strict AI-only baseline.

If P_AI-only is close to random-target chance at K = 5, then geometry and
candidate-target knowledge are not sufficient to solve the task. If P_AI-only is
above random-target chance, that structure should be reported explicitly, and
the key actual-condition contrast becomes shared-control performance at K = 5
minus AI-only performance at K = 5.

## Interpretation table

| Condition | Target information | Neural input | Main purpose |
| --- | --- | --- | --- |
| Random-target chance | none | none | statistical/random-selection reference |
| AI-only, K = 5 | 5 candidates, cued target hidden | none | autonomous baseline |
| BCI-only | cued target encoded only through neural command | yes | neural-control baseline |
| Shared control | 5 candidates, cued target hidden | yes | actual proposed system |
| AI-only, K = 1 | true target effectively revealed | none | autonomous upper bound |

This analysis tests sensitivity to increasing goal ambiguity within the present
five-target task; it should not be described as demonstrating scaling to
arbitrarily complex real-world environments.
"""
    path.write_text(text, encoding="utf-8")
