"""
BCI-only learning across original Appearing Obstacle sessions and follow-up.

This reviewer-control analysis separates the six-month follow-up comparison
from within-experiment learning by asking whether Monkey 1 BCI-only performance
already improved across the original Appearing Obstacle sessions, and whether
the first follow-up sessions resemble the end of the original experiment.

It also includes an exploratory trial-level test of whether BCI-only trials
were more successful after recent exposure to shared-control trials. That
analysis is descriptive/supporting only because recent exposure is entangled
with general task practice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "tables" / "Hidden prior follow-up"
OUTPUT_DIR = PROJECT_ROOT / "tables" / "BCI-only learning"
FIGURE_DIR = PROJECT_ROOT / "figures" / "BCI-only learning"
CACHED_SESSION_TABLE = PROJECT_ROOT / "tables" / "collision_corrected_success_by_session.csv"
DEFAULT_BASE_DIR = None
MONKEY = "Monkey 1"
ORIGINAL = "AI Appearing Obstacle"
FOLLOWUP = "AI Appearing Obstacle 2"
EXPERIMENTS = {
    ORIGINAL: "Hidden prior",
    FOLLOWUP: "No hidden prior",
}


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)
CONDITION_LABELS = {
    0.0: "BCI-only",
    1.0: "Shared-control",
}


def _ensure_import_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _target_label(trial) -> str:
    _ensure_import_path()
    from src.constants import target_mapping

    try:
        return target_mapping.get(tuple(_get(trial, "targetPosition")), "unknown")
    except Exception:
        return "unknown"


def _ai_factor(trial) -> float:
    try:
        return float(_get(trial, "aiVelocityFactor"))
    except Exception:
        return np.nan


def collect_trial_rows(base_dir: str, monkey: str = MONKEY) -> pd.DataFrame:
    _ensure_import_path()
    from src.load import load_files

    rows = []
    for experiment, prior_condition in EXPERIMENTS.items():
        all_trials, correct_all, incorrect_all, *_rest = load_files(
            experiment,
            monkey,
            base_dir,
            nb_trials=None,
        )
        if all_trials == []:
            continue
        for session_index, (correct_trials, incorrect_trials) in enumerate(zip(correct_all, incorrect_all)):
            for success, trials in ((1, correct_trials), (0, incorrect_trials)):
                for trial in trials:
                    factor = _ai_factor(trial)
                    if factor not in CONDITION_LABELS:
                        continue
                    rows.append(
                        {
                            "monkey": monkey,
                            "experiment": experiment,
                            "prior_condition": prior_condition,
                            "followup": int(experiment == FOLLOWUP),
                            "phase": "Original" if experiment == ORIGINAL else "Follow-up",
                            "session_index": session_index,
                            "session_uid": f"{monkey}|{experiment}__session_{session_index:02d}",
                            "trial_id": _get(trial, "trial", np.nan),
                            "condition": CONDITION_LABELS[factor],
                            "shared_control": int(factor == 1.0),
                            "success": success,
                            "target": _target_label(trial),
                        }
                    )
    return pd.DataFrame(rows)


def _read_inputs(input_dir: Path = INPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_path = input_dir / "hidden_prior_followup_trial_rows.csv"
    session_path = input_dir / "hidden_prior_followup_session_gains.csv"
    if not trial_path.exists() or not session_path.exists():
        raise FileNotFoundError(
            "Run run_hidden_prior_followup_gain.py first; expected "
            f"{trial_path} and {session_path}."
        )
    return pd.read_csv(trial_path), pd.read_csv(session_path)


def _session_table_from_trials(trials: pd.DataFrame) -> pd.DataFrame:
    bci = trials[trials["condition"].eq("BCI-only")].copy()
    grouped = (
        bci.groupby(["phase", "prior_condition", "experiment", "session_index", "session_uid"])
        .agg(
            n_bci_only=("success", "size"),
            n_bci_only_success=("success", "sum"),
            bci_only_success_rate=("success", "mean"),
        )
        .reset_index()
        .sort_values(["phase", "session_index"])
    )
    grouped["s_bci_only_pct"] = 100.0 * grouped["bci_only_success_rate"]
    grouped["session_number"] = grouped.groupby("phase").cumcount() + 1
    grouped = grouped.sort_values(["experiment", "session_index"]).reset_index(drop=True)
    grouped["global_session_number"] = np.arange(1, len(grouped) + 1)
    return grouped[
        [
            "phase",
            "prior_condition",
            "experiment",
            "session_index",
            "session_uid",
            "session_number",
            "global_session_number",
            "n_bci_only",
            "n_bci_only_success",
            "bci_only_success_rate",
            "s_bci_only_pct",
        ]
    ]


def _session_table_from_collision_summary(path: Path = CACHED_SESSION_TABLE, monkey: str = MONKEY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw-session summary table: {path}")
    rows = pd.read_csv(path)
    rows = rows[
        rows["source"].eq("raw_sessions")
        & rows["monkey"].eq(monkey)
        & rows["experiment"].isin([ORIGINAL, FOLLOWUP])
    ].copy()
    if rows.empty:
        raise RuntimeError(f"No raw-session rows for {monkey!r} in {path}")
    rows["phase"] = np.where(rows["experiment"].eq(ORIGINAL), "Original", "Follow-up")
    rows["prior_condition"] = rows["experiment"].map(EXPERIMENTS)
    rows["session_uid"] = rows["monkey"] + "|" + rows["experiment"] + "__session_" + rows["session_index"].astype(int).astype(str).str.zfill(2)
    rows["bci_only_success_rate"] = rows["s_bci_only_pct"] / 100.0
    rows = rows.sort_values(["experiment", "session_index"]).reset_index(drop=True)
    rows["session_number"] = rows.groupby("phase").cumcount() + 1
    rows["global_session_number"] = np.arange(1, len(rows) + 1)
    return rows[
        [
            "phase",
            "prior_condition",
            "experiment",
            "session_index",
            "session_uid",
            "session_number",
            "global_session_number",
            "n_bci_only",
            "n_bci_only_success",
            "bci_only_success_rate",
            "s_bci_only_pct",
        ]
    ]


def _bci_session_table_from_hidden_summary(session_gains: pd.DataFrame) -> pd.DataFrame:
    rows = session_gains[
        session_gains["experiment"].isin([ORIGINAL, FOLLOWUP])
    ].copy()
    rows["phase"] = np.where(rows["experiment"].eq(ORIGINAL), "Original", "Follow-up")
    rows["bci_only_success_rate"] = rows["s_bci_only_pct"] / 100.0
    rows["session_number"] = rows.groupby("phase").cumcount() + 1
    rows["global_session_number"] = np.arange(1, len(rows) + 1)
    return rows[
        [
            "phase",
            "prior_condition",
            "experiment",
            "session_index",
            "session_uid",
            "session_number",
            "global_session_number",
            "n_bci_only",
            "n_bci_only_success",
            "bci_only_success_rate",
            "s_bci_only_pct",
        ]
    ]


def _fit_logistic(formula: str, df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return pd.DataFrame([{"status": f"statsmodels unavailable: {exc}", "formula": formula}])
    try:
        model = smf.logit(formula, data=df).fit(disp=False)
        fit = model.get_robustcov_results(cov_type="cluster", groups=df[cluster_col].astype(str))
    except Exception:
        try:
            fit = smf.logit(formula, data=df).fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": df[cluster_col].astype(str)},
            )
        except Exception as exc:
            return pd.DataFrame([{"status": f"model failed: {exc}", "formula": formula}])

    rows = []
    ci = fit.conf_int()
    for term in fit.params.index:
        rows.append(
            {
                "status": "ok",
                "formula": formula,
                "term": term,
                "beta": float(fit.params[term]),
                "se_cluster_session": float(fit.bse[term]),
                "p": float(fit.pvalues[term]),
                "or": float(np.exp(fit.params[term])),
                "or_ci_low": float(np.exp(ci.loc[term].iloc[0])),
                "or_ci_high": float(np.exp(ci.loc[term].iloc[1])),
                "n_trials": int(fit.nobs),
                "n_session_clusters": int(df[cluster_col].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _session_trend(session_table: pd.DataFrame) -> pd.DataFrame:
    original = session_table[session_table["phase"].eq("Original")].copy()
    rho, p_value = stats.spearmanr(original["session_number"], original["bci_only_success_rate"])
    slope, intercept, r_value, lin_p, stderr = stats.linregress(
        original["session_number"], original["s_bci_only_pct"]
    )
    return pd.DataFrame(
        [
            {
                "analysis": "original_bci_only_session_trend",
                "n_sessions": int(len(original)),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
                "linear_slope_pp_per_session": float(slope),
                "linear_slope_p": float(lin_p),
                "first_session_pct": float(original["s_bci_only_pct"].iloc[0]),
                "last_session_pct": float(original["s_bci_only_pct"].iloc[-1]),
                "mean_pct": float(original["s_bci_only_pct"].mean()),
            }
        ]
    )


def _transition_summary(session_table: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    original = session_table[session_table["phase"].eq("Original")].copy()
    followup = session_table[session_table["phase"].eq("Follow-up")].copy()
    final_original = original.tail(window)
    first_followup = followup.head(window)
    return pd.DataFrame(
        [
            {
                "window_n_sessions": window,
                "final_original_sessions": ",".join(final_original["session_number"].astype(str)),
                "first_followup_sessions": ",".join(first_followup["session_number"].astype(str)),
                "final_original_mean_pct": float(final_original["s_bci_only_pct"].mean()),
                "final_original_median_pct": float(final_original["s_bci_only_pct"].median()),
                "first_followup_mean_pct": float(first_followup["s_bci_only_pct"].mean()),
                "first_followup_median_pct": float(first_followup["s_bci_only_pct"].median()),
                "followup_minus_final_original_mean_pp": float(
                    first_followup["s_bci_only_pct"].mean() - final_original["s_bci_only_pct"].mean()
                ),
            }
        ]
    )


def _trial_rows_with_order(trials: pd.DataFrame) -> pd.DataFrame:
    rows = trials[trials["experiment"].isin([ORIGINAL, FOLLOWUP])].copy()
    rows["phase"] = np.where(rows["experiment"].eq(ORIGINAL), "Original", "Follow-up")
    rows = rows.sort_values(["experiment", "session_index", "trial_id"]).reset_index(drop=True)
    rows["trial_order_in_session"] = rows.groupby(["experiment", "session_index"]).cumcount() + 1
    rows["session_number"] = rows.groupby("phase")["session_index"].transform(lambda x: pd.factorize(x)[0] + 1)

    recent_counts = []
    recent_any = []
    for _, group in rows.groupby(["experiment", "session_index"], sort=False):
        shared = group["shared_control"].to_numpy(int)
        for i in range(len(group)):
            previous = shared[max(0, i - 10) : i]
            recent_counts.append(int(previous.sum()))
            recent_any.append(int(previous[-5:].sum() > 0) if previous.size else 0)
    rows["recent_shared_count_10"] = recent_counts
    rows["recent_shared_any_5"] = recent_any
    return rows


def _trial_models(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame(
            [
                {
                    "status": "skipped: raw trial rows not loaded for this session-table analysis",
                    "model": "original_bci_only_learning",
                    "term": "",
                },
                {
                    "status": "skipped: raw trial rows not loaded for this session-table analysis",
                    "model": "exploratory_recent_shared_exposure",
                    "term": "",
                },
            ]
        )
    ordered = _trial_rows_with_order(trials)
    original_bci = ordered[
        ordered["phase"].eq("Original")
        & ordered["condition"].eq("BCI-only")
    ].copy()
    original_bci["session_number_z"] = (
        original_bci["session_number"] - original_bci["session_number"].mean()
    ) / original_bci["session_number"].std(ddof=0)
    original_bci["trial_order_z"] = (
        original_bci["trial_order_in_session"] - original_bci["trial_order_in_session"].mean()
    ) / original_bci["trial_order_in_session"].std(ddof=0)
    original_bci["recent_shared_count_10_z"] = (
        original_bci["recent_shared_count_10"] - original_bci["recent_shared_count_10"].mean()
    ) / original_bci["recent_shared_count_10"].std(ddof=0)

    trend = _fit_logistic(
        "success ~ session_number_z + C(target)",
        original_bci.dropna(subset=["session_number_z"]),
        "session_uid",
    )
    trend["model"] = "original_bci_only_learning"

    exposure = _fit_logistic(
        "success ~ session_number_z + trial_order_z + recent_shared_count_10_z + C(target)",
        original_bci.dropna(subset=["session_number_z", "trial_order_z", "recent_shared_count_10_z"]),
        "session_uid",
    )
    exposure["model"] = "exploratory_recent_shared_exposure"
    return pd.concat([trend, exposure], ignore_index=True)


def plot_sessions(session_table: pd.DataFrame, transition: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=220)
    colors = {"Original": "#7b5aa6", "Follow-up": "#237a6b"}
    gap = 1.5
    original = session_table[session_table["phase"].eq("Original")].copy()
    followup = session_table[session_table["phase"].eq("Follow-up")].copy()
    original["x"] = original["session_number"]
    followup["x"] = followup["session_number"] + original["session_number"].max() + gap
    for phase, sub in (("Original", original), ("Follow-up", followup)):
        ax.plot(sub["x"], sub["s_bci_only_pct"], marker="o", color=colors[phase], label=phase)

    row = transition.iloc[0]
    final_x = original.tail(int(row["window_n_sessions"]))["x"]
    first_x = followup.head(int(row["window_n_sessions"]))["x"]
    ax.hlines(row["final_original_mean_pct"], final_x.min(), final_x.max(), color=colors["Original"], lw=3)
    ax.hlines(row["first_followup_mean_pct"], first_x.min(), first_x.max(), color=colors["Follow-up"], lw=3)
    ax.axvline(original["session_number"].max() + gap / 2, color="0.65", lw=1, linestyle=":")
    ax.text(original["session_number"].max() + gap / 2 + 0.1, 8, "6 months", fontsize=8, color="0.35")
    ax.set_ylabel("BCI-only success (%)")
    ax.set_xlabel("Session order")
    ax.set_ylim(0, 100)
    ax.set_xticks(
        list(original["x"]) + list(followup["x"]),
        [str(int(x)) for x in original["session_number"]] + [str(int(x)) for x in followup["session_number"]],
        fontsize=7,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(FIGURE_DIR / f"bci_only_learning_followup.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    session_table: pd.DataFrame,
    trend: pd.DataFrame,
    transition: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    trend_row = trend.iloc[0]
    trans = transition.iloc[0]
    logit_session = models[
        models["model"].eq("original_bci_only_learning")
        & models["term"].eq("session_number_z")
    ]
    exposure = models[
        models["model"].eq("exploratory_recent_shared_exposure")
        & models["term"].eq("recent_shared_count_10_z")
    ]

    lines = [
        "# BCI-only Learning Across Original and Follow-up Appearing Obstacle Sessions",
        "",
        "This analysis uses Monkey 1 BCI-only trials from the original Appearing Obstacle experiment and the six-month follow-up without the hidden-obstacle prior.",
        "",
        "## Original-session BCI-only trend",
        "",
        (
            f"Across the original sessions, BCI-only success increased with session number "
            f"(Spearman rho={trend_row['spearman_rho']:.3f}, p={trend_row['spearman_p']:.4g}; "
            f"linear slope={trend_row['linear_slope_pp_per_session']:.2f} pp/session)."
        ),
    ]
    if not logit_session.empty and logit_session.iloc[0]["status"] == "ok":
        row = logit_session.iloc[0]
        lines.append(
            f"A trial-level logistic model with target covariates and cluster-robust SEs by session gave OR={row['or']:.2f} per 1 SD session-number increase (p={row['p']:.4g})."
        )
    else:
        lines.append(
            "The primary inference is session-level here; the trial-level clustered logistic model was skipped because this run used the raw session summary table rather than reloading the multi-GB trial pickles."
        )
    lines.extend(
        [
            "",
            "## End-of-original to follow-up transition",
            "",
            (
                f"The final {int(trans['window_n_sessions'])} original sessions averaged "
                f"{trans['final_original_mean_pct']:.1f}% BCI-only success; the first "
                f"{int(trans['window_n_sessions'])} follow-up sessions averaged "
                f"{trans['first_followup_mean_pct']:.1f}% "
                f"(follow-up minus final-original={trans['followup_minus_final_original_mean_pp']:.1f} pp)."
            ),
            "",
            "Individual BCI-only session values are saved in `bci_only_learning_session_values.csv`.",
        ]
    )
    if not exposure.empty and exposure.iloc[0]["status"] == "ok":
        lines.extend(
            [
                "",
                "## Exploratory recent shared-control exposure",
                "",
            ]
        )
        row = exposure.iloc[0]
        lines.append(
            f"Among original-experiment BCI-only trials, recent shared-control exposure over the preceding 10 trials had OR={row['or']:.2f} per 1 SD increase after accounting for session number, target, and within-session trial order (cluster-robust p={row['p']:.4g}). This exploratory model should not be interpreted causally."
        )
    lines.extend(
        [
            "",
            "## Suggested reviewer wording",
            "",
            (
                "The apparent increase from the original experiment mean to the six-month follow-up mean partly reflects learning already present within the original Appearing Obstacle sessions. "
                f"BCI-only success showed a positive, non-significant trend across the original sessions (Spearman rho={trend_row['spearman_rho']:.2f}, p={trend_row['spearman_p']:.3g}), and the first follow-up sessions began near the level reached at the end of the original experiment "
                f"(final three original sessions {trans['final_original_mean_pct']:.1f}% vs first three follow-up sessions {trans['first_followup_mean_pct']:.1f}%). "
                "Thus, the 39-40% original mean should not be interpreted as the BCI-only level immediately before the six-month gap."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    base_dir: str | None = DEFAULT_BASE_DIR,
    monkey: str = MONKEY,
    input_dir: Path = INPUT_DIR,
    use_cached_session_table: bool = True,
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if use_cached_session_table:
        trials = pd.DataFrame()
        session_table = _session_table_from_collision_summary(CACHED_SESSION_TABLE, monkey)
    elif base_dir:
        trials = collect_trial_rows(base_dir, monkey)
        if trials.empty:
            raise RuntimeError(f"No raw BCI-only learning rows found for {monkey!r} under {base_dir!r}.")
        session_table = _session_table_from_trials(trials)
    else:
        trials, session_gains = _read_inputs(input_dir)
        session_table = _bci_session_table_from_hidden_summary(session_gains)
    trend = _session_trend(session_table)
    transition = _transition_summary(session_table, window=3)
    models = _trial_models(trials)
    plot_sessions(session_table, transition)

    paths = {
        "trial_rows": OUTPUT_DIR / "bci_only_learning_trial_rows.csv",
        "session_values": OUTPUT_DIR / "bci_only_learning_session_values.csv",
        "session_trend": OUTPUT_DIR / "bci_only_learning_session_trend.csv",
        "transition": OUTPUT_DIR / "bci_only_learning_transition_summary.csv",
        "trial_models": OUTPUT_DIR / "bci_only_learning_trial_models.csv",
        "report": OUTPUT_DIR / "bci_only_learning_report.md",
    }
    trials.to_csv(paths["trial_rows"], index=False)
    session_table.to_csv(paths["session_values"], index=False)
    trend.to_csv(paths["session_trend"], index=False)
    transition.to_csv(paths["transition"], index=False)
    models.to_csv(paths["trial_models"], index=False)
    write_report(paths["report"], session_table, trend, transition, models)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--monkey", default=MONKEY)
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument(
        "--from-hidden-summary",
        action="store_true",
        help="Use existing hidden-prior summary tables instead of loading raw Monkey 1 data.",
    )
    parser.add_argument(
        "--from-raw-trials",
        action="store_true",
        help="Load raw trial pickles instead of the raw-session collision summary table.",
    )
    args = parser.parse_args()
    paths = run_analysis(
        base_dir=None if args.from_hidden_summary else load_base_dir(args.base_dir),
        monkey=args.monkey,
        input_dir=Path(args.input_dir),
        use_cached_session_table=not args.from_hidden_summary and not args.from_raw_trials,
    )
    for label, path in paths.items():
        print(f"[INFO] wrote {label}: {path}")


if __name__ == "__main__":
    main()
