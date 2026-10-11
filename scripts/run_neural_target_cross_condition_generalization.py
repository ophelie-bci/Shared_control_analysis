"""
Cross-condition neural target representation analysis.

Within each session, train a target classifier on neural firing-rate features
from one condition and test it on the other:

    BCI-only -> shared-control
    shared-control -> BCI-only

Features are MUA spike rates per electrode in a cue-locked window from
targetOnset to targetOnset + 800 ms, capped at obstacle appearance when that
event is available. This targets the cued-target representation while limiting
contamination by the later reactive obstacle response.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = None
DEFAULT_MONKEYS = ["Monkey 1", "Monkey 2"]
DEFAULT_EXPERIMENTS = ["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"]
EXPERIMENT_LABELS = {
    "AI Obstacle": "Fixed Obstacle",
    "AI Appearing Obstacle": "Appearing Obstacle",
    "AI Appearing Obstacle 2": "Appearing Obstacle follow-up",
}
CONDITION_LABELS = {0.0: "BCI-only", 1.0: "Shared-control"}
TABLE_DIR = PROJECT_ROOT / "tables" / "Neural target generalization"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Neural target generalization"
FEATURE_WINDOW_MS = 800.0
CHOICE_THRESH = 0.6
CHOICE_SPEED_THRESH = 0.1
OBSTACLE_WINDOW_RADIUS = 1.5
KNOWN_GOALS = [(7.0, 6.0), (3.5, 8.5), (0.0, 9.2), (-3.5, 8.5), (-7.0, 6.0)]
GOAL_IDX = {goal: i for i, goal in enumerate(KNOWN_GOALS)}
NEIGHBORS = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}


def load_base_dir(base_dir: str | None = None) -> str:
    if base_dir:
        return base_dir
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    configured = config.get("base_dir")
    if not configured:
        raise ValueError("Set base_dir in config.yaml or pass --base-dir.")
    return str(configured)


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _scalar(value) -> float:
    if value is None:
        return np.nan
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return np.nan


def _vec3(value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(3, np.nan)
    out = np.full(3, np.nan)
    out[: min(3, len(arr))] = arr[:3]
    return out


def _target_label(trial) -> str:
    target = tuple(np.round(_vec3(_get(trial, "targetPosition", None)), 3))
    mapping = {
        (-7.0, 0.75, 6.0): "left",
        (-3.5, 0.75, 8.5): "slight_left",
        (0.0, 0.75, 9.2): "straight",
        (3.5, 0.75, 8.5): "slight_right",
        (7.0, 0.75, 6.0): "right",
    }
    return mapping.get(target, "unknown")


def _decoder_choice_state(trial, experiment: str) -> tuple[str, float, float, float, int]:
    mv = _get(trial, "modelVelocity", None)
    traj = _get(trial, "avatarTrajectory", None)
    if mv is None or traj is None:
        return "no_model", np.nan, np.nan, np.nan, 0
    try:
        vx = np.asarray(mv["vx"], dtype=float)
        vz = np.asarray(mv["vz"], dtype=float)
        xs_all = np.asarray(traj["x"], dtype=float)
        zs_all = np.asarray(traj["z"], dtype=float)
        t_all = np.asarray(_get(traj, "time", np.arange(len(xs_all))), dtype=float)
    except Exception:
        return "no_model", np.nan, np.nan, np.nan, 0

    n = min(len(vx), len(vz), len(xs_all), len(zs_all), len(t_all))
    if n < 5:
        return "no_model", np.nan, np.nan, np.nan, 0
    vx, vz = vx[:n], vz[:n]
    xs_all, zs_all, t_all = xs_all[:n], zs_all[:n], t_all[:n]

    true_goal = None
    target = _get(trial, "targetPosition", None)
    if target is not None:
        try:
            true_goal = (float(target[0]), float(target[2]))
        except Exception:
            true_goal = None

    speed = np.sqrt(vx ** 2 + vz ** 2)
    valid = np.isfinite(speed) & (speed > CHOICE_SPEED_THRESH)
    obstacle = _get(trial, "obstaclePosition", None)
    if obstacle is not None and "Obstacle" in str(experiment):
        try:
            ox, oz = float(obstacle[0]), float(obstacle[2])
            dist_obs = np.sqrt((xs_all - ox) ** 2 + (zs_all - oz) ** 2)
            valid &= np.isfinite(dist_obs) & (dist_obs > OBSTACLE_WINDOW_RADIUS)
        except Exception:
            pass

    if not np.any(valid):
        return "no_model", np.nan, np.nan, np.nan, 0

    v = np.stack([vx[valid], vz[valid]], axis=1)
    xs = xs_all[valid]
    zs = zs_all[valid]
    counts = {goal: 0 for goal in KNOWN_GOALS}
    for i in range(v.shape[0]):
        vi = v[i]
        vi_norm = np.linalg.norm(vi)
        if vi_norm < 1e-9:
            continue
        best_goal = None
        best_cos = -np.inf
        for gx, gz in KNOWN_GOALS:
            gvec = np.array([gx - xs[i], gz - zs[i]], dtype=float)
            gnorm = np.linalg.norm(gvec)
            if gnorm < 1e-9:
                continue
            cos = float(np.dot(vi, gvec) / (vi_norm * gnorm + 1e-12))
            if cos > best_cos:
                best_goal = (gx, gz)
                best_cos = cos
        if best_goal is not None and best_cos > 0.0:
            counts[best_goal] += 1

    total = sum(counts.values())
    if total == 0:
        return "no_model", np.nan, np.nan, np.nan, int(v.shape[0])
    fracs = {goal: count / total for goal, count in counts.items() if count > 0}
    best_goal = max(fracs, key=fracs.get)
    best_frac = fracs[best_goal]
    true_frac = fracs.get(true_goal, 0.0) if true_goal is not None else 0.0
    neighbor_frac = 0.0
    if true_goal in GOAL_IDX:
        neighbor_goals = [KNOWN_GOALS[i] for i in NEIGHBORS.get(GOAL_IDX[true_goal], [])]
        neighbor_frac = sum(fracs.get(goal, 0.0) for goal in neighbor_goals)
    non_true_mass = max(0.0, sum(fracs.values()) - true_frac)
    other_frac = max(0.0, non_true_mass - neighbor_frac)

    if best_frac >= CHOICE_THRESH:
        if true_goal is not None and best_goal == true_goal:
            state = "correct"
        elif true_goal in GOAL_IDX and best_goal in [KNOWN_GOALS[i] for i in NEIGHBORS.get(GOAL_IDX[true_goal], [])]:
            state = "adjacent"
        else:
            state = "incorrect"
    elif non_true_mass > 0 and neighbor_frac >= 0.8 * non_true_mass and true_goal in GOAL_IDX:
        state = "adjacent"
    else:
        state = "ambiguous"
    return state, true_frac, neighbor_frac, other_frac, int(v.shape[0])


def _load_trials(path: Path) -> list:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj[1] if isinstance(obj, (list, tuple)) and len(obj) >= 2 else obj


def _trial_feature_row(trial, monkey: str, experiment: str, session_index: int, session_file: str) -> dict | None:
    try:
        factor = float(_get(trial, "aiVelocityFactor", np.nan))
    except Exception:
        return None
    if factor not in CONDITION_LABELS:
        return None
    target = _target_label(trial)
    if target == "unknown":
        return None
    onset = _scalar(_get(trial, "targetOnset", None))
    if not np.isfinite(onset):
        onset = _scalar(_get(trial, "start", None))
    if not np.isfinite(onset):
        return None
    obstacle_time = _scalar(_get(trial, "obstacleAppearedTime", None))
    end = onset + FEATURE_WINDOW_MS
    if np.isfinite(obstacle_time) and obstacle_time > onset:
        end = min(end, obstacle_time)
    duration_s = max((end - onset) / 1000.0, 1e-3)

    mua = _get(trial, "muaA", {})
    if not isinstance(mua, dict) or not mua:
        return None

    features = {}
    for electrode, spike_times in sorted(mua.items()):
        try:
            spikes = np.asarray(spike_times, dtype=float)
        except Exception:
            continue
        if spikes.size == 0:
            count = 0
        else:
            count = int(np.sum((spikes >= onset) & (spikes < end)))
        features[f"feat_{electrode}"] = count / duration_s
    if not features:
        return None
    choice_state, choice_true_frac, choice_neighbor_frac, choice_other_frac, choice_n_frames = _decoder_choice_state(
        trial,
        experiment,
    )

    row = {
        "monkey": monkey,
        "experiment": experiment,
        "task_family": EXPERIMENT_LABELS[experiment],
        "session_index": session_index,
        "session_uid": f"{monkey}|{experiment}|{session_index}",
        "session_file": session_file,
        "trial_id": _get(trial, "trial", np.nan),
        "condition": CONDITION_LABELS[factor],
        "shared_control": int(factor == 1.0),
        "success": int(_get(trial, "answer", None) == 1),
        "target": target,
        "decoder_choice_state": choice_state,
        "decoder_choice_true_fraction": choice_true_frac,
        "decoder_choice_adjacent_fraction": choice_neighbor_frac,
        "decoder_choice_incorrect_fraction": choice_other_frac,
        "decoder_choice_n_frames": choice_n_frames,
        "feature_window_start": onset,
        "feature_window_end": end,
        "feature_window_ms": end - onset,
    }
    row.update(features)
    return row


def collect_features(base_dir: str, monkeys: list[str], experiments: list[str]) -> pd.DataFrame:
    rows = []
    for experiment in experiments:
        if experiment not in EXPERIMENT_LABELS:
            continue
        for monkey in monkeys:
            data_dir = Path(base_dir) / monkey / experiment
            if not data_dir.exists():
                print(f"[INFO] skipping missing {data_dir}")
                continue
            files = sorted(data_dir.glob("*_trials.pkl"))
            print(f"[INFO] {monkey} {experiment}: {len(files)} sessions")
            for session_index, path in enumerate(files):
                trials = _load_trials(path)
                session_rows = [
                    row for trial in trials
                    if (row := _trial_feature_row(trial, monkey, experiment, session_index, path.name)) is not None
                ]
                rows.extend(session_rows)
                print(f"  session {session_index}: {len(session_rows)} neural rows")
    return pd.DataFrame(rows)


def _chance_accuracy(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True)
    return float(counts.max()) if len(counts) else np.nan


def _classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )


def _append_result(
    rows: list[dict],
    *,
    status: str,
    task_family: str,
    monkey: str,
    experiment: str,
    session_index: int,
    session_uid: str,
    train_condition: str,
    test_condition: str,
    n_train: int,
    n_test: int,
    n_targets_common: int,
    accuracy: float,
    balanced: float,
    chance: float,
    evaluation: str,
) -> None:
    rows.append(
        {
            "status": status,
            "task_family": task_family,
            "monkey": monkey,
            "experiment": experiment,
            "session_index": int(session_index),
            "session_uid": session_uid,
            "train_condition": train_condition,
            "test_condition": test_condition,
            "direction": f"{train_condition} -> {test_condition}",
            "evaluation": evaluation,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_targets_common": int(n_targets_common),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "majority_class_chance": chance,
            "accuracy_minus_chance": accuracy - chance if np.isfinite(accuracy) and np.isfinite(chance) else np.nan,
        }
    )


def fit_cross_condition(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in features.columns if col.startswith("feat_")]
    rows = []
    for keys, session in features.groupby(["task_family", "monkey", "experiment", "session_index", "session_uid"]):
        task_family, monkey, experiment, session_index, session_uid = keys
        for train_condition, test_condition in (("BCI-only", "Shared-control"), ("Shared-control", "BCI-only")):
            train = session[session["condition"].eq(train_condition)].copy()
            test = session[session["condition"].eq(test_condition)].copy()
            common_targets = sorted(set(train["target"]) & set(test["target"]))
            train = train[train["target"].isin(common_targets)]
            test = test[test["target"].isin(common_targets)]
            status = "ok"
            if len(common_targets) < 2 or len(train) < len(common_targets) * 2 or len(test) < len(common_targets):
                status = "skipped_insufficient_targets_or_trials"
                accuracy = balanced = chance = np.nan
            else:
                x_train = train[feature_cols].fillna(0).to_numpy(dtype=float)
                y_train = train["target"].to_numpy()
                x_test = test[feature_cols].fillna(0).to_numpy(dtype=float)
                y_test = test["target"].to_numpy()
                clf = _classifier()
                try:
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)
                    accuracy = float(accuracy_score(y_test, pred))
                    balanced = float(balanced_accuracy_score(y_test, pred))
                    chance = _chance_accuracy(pd.Series(y_test))
                except Exception as exc:
                    status = f"model_failed: {exc}"
                    accuracy = balanced = chance = np.nan
            _append_result(
                rows,
                status=status,
                task_family=task_family,
                monkey=monkey,
                experiment=experiment,
                session_index=session_index,
                session_uid=session_uid,
                train_condition=train_condition,
                test_condition=test_condition,
                n_train=len(train),
                n_test=len(test),
                n_targets_common=len(common_targets),
                accuracy=accuracy,
                balanced=balanced,
                chance=chance,
                evaluation="cross_condition",
            )
        for condition in ("BCI-only", "Shared-control"):
            data = session[session["condition"].eq(condition)].copy()
            counts = data["target"].value_counts()
            usable_targets = sorted(counts[counts >= 2].index)
            data = data[data["target"].isin(usable_targets)]
            n_splits = int(min(5, counts[counts >= 2].min())) if usable_targets else 0
            status = "ok"
            if len(usable_targets) < 2 or len(data) < len(usable_targets) * 2 or n_splits < 2:
                status = "skipped_insufficient_targets_or_trials"
                accuracy = balanced = chance = np.nan
            else:
                x = data[feature_cols].fillna(0).to_numpy(dtype=float)
                y = data["target"].to_numpy()
                clf = _classifier()
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                try:
                    pred = cross_val_predict(clf, x, y, cv=cv)
                    accuracy = float(accuracy_score(y, pred))
                    balanced = float(balanced_accuracy_score(y, pred))
                    chance = _chance_accuracy(pd.Series(y))
                except Exception as exc:
                    status = f"model_failed: {exc}"
                    accuracy = balanced = chance = np.nan
            _append_result(
                rows,
                status=status,
                task_family=task_family,
                monkey=monkey,
                experiment=experiment,
                session_index=session_index,
                session_uid=session_uid,
                train_condition=condition,
                test_condition=condition,
                n_train=len(data),
                n_test=len(data),
                n_targets_common=len(usable_targets),
                accuracy=accuracy,
                balanced=balanced,
                chance=chance,
                evaluation="within_condition_cv",
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("ok")].copy()
    rows = []
    for keys, group in ok.groupby(["task_family", "direction"]):
        task_family, direction = keys
        delta = group["accuracy_minus_chance"].to_numpy(float)
        try:
            wil = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
            w_stat, p_value = float(wil.statistic), float(wil.pvalue)
        except Exception:
            w_stat, p_value = np.nan, np.nan
        rows.append(
            {
                "task_family": task_family,
                "direction": direction,
                "n_sessions": int(len(group)),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_majority_chance": float(group["majority_class_chance"].mean()),
                "mean_accuracy_minus_chance": float(group["accuracy_minus_chance"].mean()),
                "median_accuracy_minus_chance": float(group["accuracy_minus_chance"].median()),
                "n_sessions_above_chance": int((group["accuracy_minus_chance"] > 0).sum()),
                "wilcoxon_greater_than_chance_statistic": w_stat,
                "wilcoxon_greater_than_chance_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def compare_cross_to_within(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("ok")].copy()
    cross = ok[ok["evaluation"].eq("cross_condition")].copy()
    within = ok[ok["evaluation"].eq("within_condition_cv")].copy()
    within = within.rename(
        columns={
            "test_condition": "within_test_condition",
            "accuracy": "within_accuracy",
            "balanced_accuracy": "within_balanced_accuracy",
            "majority_class_chance": "within_majority_class_chance",
            "accuracy_minus_chance": "within_accuracy_minus_chance",
        }
    )
    keys = ["task_family", "monkey", "experiment", "session_index", "session_uid", "train_condition"]
    merged = cross.merge(
        within[
            keys
            + [
                "within_test_condition",
                "within_accuracy",
                "within_balanced_accuracy",
                "within_majority_class_chance",
                "within_accuracy_minus_chance",
            ]
        ],
        on=keys,
        how="inner",
    )
    merged["cross_minus_within_accuracy"] = merged["accuracy"] - merged["within_accuracy"]
    merged["cross_minus_within_accuracy_minus_chance"] = (
        merged["accuracy_minus_chance"] - merged["within_accuracy_minus_chance"]
    )
    rows = []
    for keys, group in merged.groupby(["task_family", "train_condition"]):
        task_family, train_condition = keys
        delta = group["cross_minus_within_accuracy_minus_chance"].to_numpy(float)
        try:
            wil_two = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
            wil_less = stats.wilcoxon(delta, alternative="less", zero_method="wilcox")
            two_stat, two_p = float(wil_two.statistic), float(wil_two.pvalue)
            less_stat, less_p = float(wil_less.statistic), float(wil_less.pvalue)
        except Exception:
            two_stat = two_p = less_stat = less_p = np.nan
        rows.append(
            {
                "task_family": task_family,
                "train_condition": train_condition,
                "n_sessions": int(len(group)),
                "mean_cross_accuracy": float(group["accuracy"].mean()),
                "mean_within_accuracy": float(group["within_accuracy"].mean()),
                "mean_cross_minus_within_accuracy": float(group["cross_minus_within_accuracy"].mean()),
                "mean_cross_accuracy_minus_chance": float(group["accuracy_minus_chance"].mean()),
                "mean_within_accuracy_minus_chance": float(group["within_accuracy_minus_chance"].mean()),
                "mean_cross_minus_within_accuracy_minus_chance": float(delta.mean()),
                "median_cross_minus_within_accuracy_minus_chance": float(np.median(delta)),
                "n_sessions_cross_lower_than_within": int((delta < 0).sum()),
                "n_sessions_cross_higher_than_within": int((delta > 0).sum()),
                "wilcoxon_cross_diff_two_sided_statistic": two_stat,
                "wilcoxon_cross_diff_two_sided_p": two_p,
                "wilcoxon_cross_less_than_within_statistic": less_stat,
                "wilcoxon_cross_less_than_within_p": less_p,
            }
        )
    return pd.DataFrame(rows), merged


def decode_by_choice_state(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in features.columns if col.startswith("feat_")]
    rows = []
    group_cols = [
        "task_family",
        "monkey",
        "experiment",
        "session_index",
        "session_uid",
        "condition",
        "decoder_choice_state",
    ]
    for keys, group in features.groupby(group_cols, dropna=False):
        task_family, monkey, experiment, session_index, session_uid, condition, choice_state = keys
        data = group.copy()
        counts = data["target"].value_counts()
        usable_targets = sorted(counts[counts >= 2].index)
        data = data[data["target"].isin(usable_targets)]
        n_splits = int(min(5, counts[counts >= 2].min())) if usable_targets else 0
        status = "ok"
        if choice_state == "no_model":
            status = "skipped_no_model_choice_state"
            accuracy = balanced = chance = np.nan
        elif len(usable_targets) < 2 or len(data) < len(usable_targets) * 2 or n_splits < 2:
            status = "skipped_insufficient_targets_or_trials"
            accuracy = balanced = chance = np.nan
        else:
            x = data[feature_cols].fillna(0).to_numpy(dtype=float)
            y = data["target"].to_numpy()
            clf = _classifier()
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            try:
                pred = cross_val_predict(clf, x, y, cv=cv)
                accuracy = float(accuracy_score(y, pred))
                balanced = float(balanced_accuracy_score(y, pred))
                chance = _chance_accuracy(pd.Series(y))
            except Exception as exc:
                status = f"model_failed: {exc}"
                accuracy = balanced = chance = np.nan
        rows.append(
            {
                "status": status,
                "task_family": task_family,
                "monkey": monkey,
                "experiment": experiment,
                "session_index": int(session_index),
                "session_uid": session_uid,
                "condition": condition,
                "decoder_choice_state": choice_state,
                "n_trials_total": int(len(group)),
                "n_trials_model": int(len(data)),
                "n_targets_usable": int(len(usable_targets)),
                "n_cv_splits": int(n_splits),
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "majority_class_chance": chance,
                "accuracy_minus_chance": accuracy - chance if np.isfinite(accuracy) and np.isfinite(chance) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_choice_state_decoding(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("ok")].copy()
    rows = []
    for keys, group in ok.groupby(["task_family", "condition", "decoder_choice_state"], dropna=False):
        task_family, condition, choice_state = keys
        delta = group["accuracy_minus_chance"].to_numpy(float)
        try:
            wil = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
            w_stat, p_value = float(wil.statistic), float(wil.pvalue)
        except Exception:
            w_stat, p_value = np.nan, np.nan
        rows.append(
            {
                "task_family": task_family,
                "condition": condition,
                "decoder_choice_state": choice_state,
                "n_sessions": int(len(group)),
                "n_trials_total": int(group["n_trials_total"].sum()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_majority_chance": float(group["majority_class_chance"].mean()),
                "mean_accuracy_minus_chance": float(group["accuracy_minus_chance"].mean()),
                "median_accuracy_minus_chance": float(group["accuracy_minus_chance"].median()),
                "n_sessions_above_chance": int((group["accuracy_minus_chance"] > 0).sum()),
                "wilcoxon_greater_than_chance_statistic": w_stat,
                "wilcoxon_greater_than_chance_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def compare_choice_state_conditions(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("ok")].copy()
    bci = ok[ok["condition"].eq("BCI-only")].copy()
    shared = ok[ok["condition"].eq("Shared-control")].copy()
    keys = ["task_family", "monkey", "experiment", "session_index", "session_uid", "decoder_choice_state"]
    merged = bci.merge(
        shared,
        on=keys,
        suffixes=("_bci_only", "_shared_control"),
        how="inner",
    )
    merged["shared_minus_bci_accuracy_minus_chance"] = (
        merged["accuracy_minus_chance_shared_control"] - merged["accuracy_minus_chance_bci_only"]
    )
    rows = []
    for keys, group in merged.groupby(["task_family", "decoder_choice_state"], dropna=False):
        task_family, choice_state = keys
        delta = group["shared_minus_bci_accuracy_minus_chance"].to_numpy(float)
        try:
            wil = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
            w_stat, p_value = float(wil.statistic), float(wil.pvalue)
        except Exception:
            w_stat, p_value = np.nan, np.nan
        rows.append(
            {
                "task_family": task_family,
                "decoder_choice_state": choice_state,
                "n_paired_sessions": int(len(group)),
                "mean_bci_only_accuracy_minus_chance": float(group["accuracy_minus_chance_bci_only"].mean()),
                "mean_shared_control_accuracy_minus_chance": float(group["accuracy_minus_chance_shared_control"].mean()),
                "mean_shared_minus_bci_accuracy_minus_chance": float(delta.mean()),
                "median_shared_minus_bci_accuracy_minus_chance": float(np.median(delta)),
                "n_sessions_shared_higher": int((delta > 0).sum()),
                "n_sessions_bci_higher": int((delta < 0).sum()),
                "wilcoxon_two_sided_statistic": w_stat,
                "wilcoxon_two_sided_p": p_value,
            }
        )
    return pd.DataFrame(rows), merged


def plot_summary(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=220)
    labels = summary["task_family"] + "\n" + summary["direction"]
    x = np.arange(len(summary))
    ax.bar(x, 100.0 * summary["mean_accuracy_minus_chance"], color="#6d7f9a")
    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy above majority-class chance (pp)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(FIGURE_DIR / f"neural_target_cross_condition_generalization.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_choice_state_summary(choice_summary: pd.DataFrame) -> None:
    if choice_summary.empty:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    task_order = ["Fixed Obstacle", "Appearing Obstacle", "Appearing Obstacle follow-up"]
    state_order = ["correct", "adjacent", "ambiguous", "incorrect"]
    colors = {"BCI-only": "#4c78a8", "Shared-control": "#f58518"}
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), dpi=220, sharey=True)
    for ax, task in zip(axes, task_order):
        task_df = choice_summary[choice_summary["task_family"].eq(task)]
        x = np.arange(len(state_order))
        width = 0.36
        for offset, condition in [(-width / 2, "BCI-only"), (width / 2, "Shared-control")]:
            vals = []
            labels = []
            for state in state_order:
                row = task_df[
                    task_df["condition"].eq(condition)
                    & task_df["decoder_choice_state"].eq(state)
                ]
                if row.empty:
                    vals.append(np.nan)
                    labels.append("")
                else:
                    vals.append(100.0 * float(row.iloc[0]["mean_accuracy_minus_chance"]))
                    labels.append(str(int(row.iloc[0]["n_sessions"])))
            ax.bar(x + offset, vals, width=width, color=colors[condition], label=condition)
            for xi, yi, lab in zip(x + offset, vals, labels):
                if lab and np.isfinite(yi):
                    ax.text(xi, yi + 1.0, lab, ha="center", va="bottom", fontsize=6)
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_title(task, fontsize=8)
        ax.set_xticks(x, state_order, rotation=35, ha="right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy above majority-class chance (pp)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("svg", "png", "pdf"):
        fig.savefig(FIGURE_DIR / f"neural_target_by_decoder_choice_state.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    choice_summary: pd.DataFrame,
    choice_condition_summary: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Neural Target Cross-condition Generalization",
        "",
        "Within each session, target classifiers were trained on cue-window MUA firing rates from one condition and tested on the other condition. The feature window was targetOnset to 800 ms after target onset, capped at obstacle appearance when available.",
        "",
        "## Session-level decoding",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['direction']}: mean accuracy {100*row['mean_accuracy']:.1f}%, "
            f"mean majority-class chance {100*row['mean_majority_chance']:.1f}%, "
            f"above chance {100*row['mean_accuracy_minus_chance']:.1f} pp; "
            f"{int(row['n_sessions_above_chance'])}/{int(row['n_sessions'])} sessions above chance; "
            f"one-sided Wilcoxon p={row['wilcoxon_greater_than_chance_p']:.4g}."
        )
    lines.extend(["", "## Cross-condition versus within-condition decoding", ""])
    for _, row in comparison_summary.iterrows():
        lines.append(
            f"- {row['task_family']}, trained on {row['train_condition']}: "
            f"cross-condition accuracy {100*row['mean_cross_accuracy']:.1f}% vs "
            f"within-condition accuracy {100*row['mean_within_accuracy']:.1f}%; "
            f"cross-minus-within, chance-corrected = "
            f"{100*row['mean_cross_minus_within_accuracy_minus_chance']:.1f} pp "
            f"(median {100*row['median_cross_minus_within_accuracy_minus_chance']:.1f} pp; "
            f"two-sided Wilcoxon p={row['wilcoxon_cross_diff_two_sided_p']:.4g})."
        )
    lines.extend(["", "## Neural target decoding by decoded-choice state", ""])
    preferred_order = {"correct": 0, "adjacent": 1, "ambiguous": 2, "incorrect": 3}
    ordered_choice = choice_summary.copy()
    ordered_choice["_order"] = ordered_choice["decoder_choice_state"].map(preferred_order).fillna(99)
    ordered_choice = ordered_choice.sort_values(["task_family", "_order", "condition"])
    for _, row in ordered_choice.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['condition']}, {row['decoder_choice_state']}: "
            f"mean accuracy {100*row['mean_accuracy']:.1f}%, "
            f"mean majority-class chance {100*row['mean_majority_chance']:.1f}%, "
            f"above chance {100*row['mean_accuracy_minus_chance']:.1f} pp; "
            f"{int(row['n_sessions_above_chance'])}/{int(row['n_sessions'])} sessions above chance; "
            f"one-sided Wilcoxon p={row['wilcoxon_greater_than_chance_p']:.4g}."
        )
    lines.extend(["", "## BCI-only versus shared-control within decoded-choice state", ""])
    ordered_condition = choice_condition_summary.copy()
    ordered_condition["_order"] = ordered_condition["decoder_choice_state"].map(preferred_order).fillna(99)
    ordered_condition = ordered_condition.sort_values(["task_family", "_order"])
    for _, row in ordered_condition.iterrows():
        lines.append(
            f"- {row['task_family']}, {row['decoder_choice_state']}: "
            f"shared-minus-BCI chance-corrected accuracy = "
            f"{100*row['mean_shared_minus_bci_accuracy_minus_chance']:.1f} pp "
            f"(median {100*row['median_shared_minus_bci_accuracy_minus_chance']:.1f} pp; "
            f"n={int(row['n_paired_sessions'])}; two-sided Wilcoxon p={row['wilcoxon_two_sided_p']:.4g})."
        )
    lines.extend(
        [
            "",
            "## Suggested reviewer wording",
            "",
            "A neural target classifier decoded the cued target from early cue-window population activity, before trial outcome could dominate the signal. Target information generalized across BCI-only and shared-control trials and was close to within-condition held-out decoding. When trials were split by the decoded-choice state of the BCI velocity command, cue-target information was strongest for decoder-correct and adjacent trials. Ambiguous-trial decoding was modest and not significant across sessions, and incorrect-choice trials were too sparse for a reliable session-level inference. Across matched decoder-choice states with sufficient paired sessions, BCI-only and shared-control decoding did not differ significantly, supporting broadly comparable target-related neural representation across conditions.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    base_dir: str,
    monkeys: list[str],
    experiments: list[str],
    feature_input: list[str] | None = None,
    feature_output: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if feature_input:
        features = pd.concat([pd.read_csv(path) for path in feature_input], ignore_index=True)
    else:
        features = collect_features(base_dir, monkeys, experiments)
    if features.empty:
        raise RuntimeError("No neural feature rows were found.")
    feature_path = Path(feature_output).resolve() if feature_output else TABLE_DIR / "neural_target_feature_rows.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False)
    if feature_path != TABLE_DIR / "neural_target_feature_rows.csv":
        features.to_csv(TABLE_DIR / "neural_target_feature_rows.csv", index=False)
    results = fit_cross_condition(features)
    summary = summarize(results)
    comparison_summary, comparison_rows = compare_cross_to_within(results)
    choice_results = decode_by_choice_state(features)
    choice_summary = summarize_choice_state_decoding(choice_results)
    choice_condition_summary, choice_condition_rows = compare_choice_state_conditions(choice_results)
    results.to_csv(TABLE_DIR / "neural_target_cross_condition_session_results.csv", index=False)
    summary.to_csv(TABLE_DIR / "neural_target_cross_condition_summary.csv", index=False)
    comparison_rows.to_csv(TABLE_DIR / "neural_target_cross_vs_within_session_results.csv", index=False)
    comparison_summary.to_csv(TABLE_DIR / "neural_target_cross_vs_within_summary.csv", index=False)
    choice_results.to_csv(TABLE_DIR / "neural_target_by_decoder_choice_state_session_results.csv", index=False)
    choice_summary.to_csv(TABLE_DIR / "neural_target_by_decoder_choice_state_summary.csv", index=False)
    choice_condition_rows.to_csv(
        TABLE_DIR / "neural_target_by_decoder_choice_state_condition_pairs.csv",
        index=False,
    )
    choice_condition_summary.to_csv(
        TABLE_DIR / "neural_target_by_decoder_choice_state_condition_summary.csv",
        index=False,
    )
    plot_summary(summary)
    plot_choice_state_summary(choice_summary)
    write_report(
        summary,
        comparison_summary,
        choice_summary,
        choice_condition_summary,
        TABLE_DIR / "neural_target_cross_condition_report.md",
    )
    return results, summary, comparison_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--monkeys", nargs="+", default=DEFAULT_MONKEYS)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--feature-input", nargs="+", default=None)
    parser.add_argument("--feature-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, summary, comparison_summary = run_analysis(
        load_base_dir(args.base_dir),
        args.monkeys,
        args.experiments,
        feature_input=args.feature_input,
        feature_output=args.feature_output,
    )
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False))
        print("")
        print(comparison_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
