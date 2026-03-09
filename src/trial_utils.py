import numpy as np
from typing import Dict, List
from src.constants import target_mapping

def _mean_ignore_nan(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(x.mean())
import random
from collections import defaultdict
import pathlib
import pickle

def balance_ai_on_off_per_target_with_split(
    correct_ai_on, incorrect_ai_on,
    correct_ai_off, incorrect_ai_off,
    key_field='trial', seed=42
):
    """
    Balance AI-On vs AI-Off per target category,
    then split back into correct/incorrect buckets.

    Inputs:
      - Each argument is dict: {target: [trial_dict, ...]}
    Returns:
      bal_correct_ai_on, bal_incorrect_ai_on,
      bal_correct_ai_off, bal_incorrect_ai_off
      (all dicts with per-target lists, balanced per target)
    """
    random.seed(seed)

    # Merge for convenience
    def merge_dicts(d1, d2):
        out = defaultdict(list)
        for d in (d1, d2):
            for tgt, lst in d.items():
                out[tgt].extend(lst)
        return out

    on_by_target  = merge_dicts(correct_ai_on, incorrect_ai_on)
    off_by_target = merge_dicts(correct_ai_off, incorrect_ai_off)

    # Prepare outputs
    bal_correct_ai_on   = defaultdict(list)
    bal_incorrect_ai_on = defaultdict(list)
    bal_correct_ai_off  = defaultdict(list)
    bal_incorrect_ai_off= defaultdict(list)

    # Balance per target
    all_targets = set(on_by_target.keys()).union(off_by_target.keys())
    for tgt in all_targets:
        on_trials  = on_by_target.get(tgt, [])
        off_trials = off_by_target.get(tgt, [])

        n_on, n_off = len(on_trials), len(off_trials)
        if n_on == 0 or n_off == 0:
            # no balance possible
            continue

        target_n = min(n_on, n_off)

        # Downsample
        if n_on > n_off:
            on_trials  = random.sample(on_trials,  target_n)
            off_trials = list(off_trials)
        elif n_off > n_on:
            off_trials = random.sample(off_trials, target_n)
            on_trials  = list(on_trials)

        # Split back into correct/incorrect using trial IDs
        keep_on_ids  = {t.key_field for t in on_trials}
        keep_off_ids = {t[key_field] for t in off_trials}

        bal_correct_ai_on[tgt]   = [t for t in correct_ai_on.get(tgt, [])   if t[key_field] in keep_on_ids]
        bal_incorrect_ai_on[tgt] = [t for t in incorrect_ai_on.get(tgt, []) if t[key_field] in keep_on_ids]
        bal_correct_ai_off[tgt]  = [t for t in correct_ai_off.get(tgt, [])  if t[key_field] in keep_off_ids]
        bal_incorrect_ai_off[tgt]= [t for t in incorrect_ai_off.get(tgt, [])if t[key_field] in keep_off_ids]

    return dict(bal_correct_ai_on), dict(bal_incorrect_ai_on), dict(bal_correct_ai_off), dict(bal_incorrect_ai_off)



def collect_summary_points(summary_list, monkey, experiment,
                           per_target_gains, acc_ai_off):
    """
    Append one record per target to `summary_list` with:
      - baseline_off: pooled No-AI success for that target (0..1)
      - gain_mean: mean AI gain across sessions for that target (0..1)
    """
    for tgt in sorted(set(per_target_gains) | set(acc_ai_off)):
        gain_mean = _mean_ignore_nan(per_target_gains.get(tgt, []))  # proportion
        baseline  = acc_ai_off.get(tgt, np.nan)                       # proportion
        if isinstance(baseline, (list, tuple, np.ndarray)):
            baseline = _mean_ignore_nan(baseline)
        if np.isfinite(gain_mean) and np.isfinite(baseline):
            summary_list.append({
                "monkey": monkey,
                "experiment": experiment,
                "target": tgt,
                "baseline_off_pct": 100.0 * baseline,
                "gain_pp": 100.0 * gain_mean,
            })

def categorize_trials_by_target_and_ai(experiment: str, trials: List, ai_condition: float) -> Dict[str, List]:
    """
    Categorizes trials by target and AI condition (AI On or Off).
    """
    trials_per_target = {key: [] for key in target_mapping.values()}
    
    for trial in trials:
        target_key = target_mapping.get(tuple(trial.targetPosition), 'unknown')
        if target_key == 'unknown':
            continue

        if experiment in ("AI Obstacle", "House"):
            if getattr(trial, 'aiVelocityFactor', 0) == ai_condition:
                trials_per_target[target_key].append(trial)

        elif experiment in ("AI Appearing Obstacle", "AI Appearing Obstacle 2"):
            if trial.obstacleAppearedTime is not None and getattr(trial, 'aiVelocityFactor', 0) == ai_condition:
                trials_per_target[target_key].append(trial)
        
        elif experiment == "House Obstacle":
            if not np.all(np.isnan(trial.obstaclePosition)) and getattr(trial, 'aiVelocityFactor', 0) == ai_condition:
                trials_per_target[target_key].append(trial)

        elif experiment == "AI Respawn":
            if not np.isnan(trial.targetJumpPosition).any() and getattr(trial, 'aiVelocityFactor', 0) == ai_condition:
                trials_per_target[target_key].append(trial)

    return trials_per_target

def compute_target_stats(correct_trials: Dict[str, List], incorrect_trials: Dict[str, List]) -> List[Dict[str, float]]:
    """
    Computes the number of trials and success rate per target.
    """
    results = []
    all_targets = set(correct_trials.keys()) | set(incorrect_trials.keys())

    for target in sorted(all_targets):
        correct = correct_trials.get(target, [])
        incorrect = incorrect_trials.get(target, [])
        total = len(correct) + len(incorrect)
        success = (len(correct) / total * 100) if total > 0 else 0.0

        results.append({
            'target': target,
            'num_trials': total,
            'success_rate (%)': round(success, 2)
        })

    return results

def save_grand_summary(grand_summary, experiments):
    # go from current file -> parent (scripts) -> parent (ai_ibci_analysis)
    root_dir = pathlib.Path(__file__).resolve().parent.parent

    # if you want it directly in ai_ibci_analysis:
    # out_path = root_dir / "grand_summary.pkl"

    # if you prefer the Variables subfolder (from your screenshot):
    out_dir = root_dir / "Variables"
    out_dir.mkdir(exist_ok=True)
    if "AI Respawn" in experiments: 
        out_path = out_dir / "grand_summary_respawn.pkl"
    else:
        out_path = out_dir / "grand_summary.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(grand_summary, f)

    print(f"Saved grand_summary to {out_path}")

def load_grand_summary():
    # go to ai_ibci_analysis (same as in main for config.yaml)
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    path = root_dir / "Variables" / "grand_summary.pkl"

    with open(path, "rb") as f:
        grand_summary = pickle.load(f)

    print(f"Loaded grand_summary from {path}")
    return grand_summary