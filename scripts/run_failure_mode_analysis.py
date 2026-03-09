"""
Trajectory and failure-mode analysis for AI-assisted vs non-assisted BCI navigation.

This script aggregates trial-level trajectory metrics across monkeys and experiments,
with a focus on:

- success vs failure under AI ON and AI OFF conditions
- failure-mode classification for incorrect trials
- intention / choice-state estimates from modelVelocity
- ambiguity metrics from decoded and executed velocities
- per-unit summaries used for manuscript figures

The analysis supports both standard obstacle tasks and Respawn tasks. For Respawn,
post-jump segments are used when appropriate, and targetJumpPosition is treated as
the relevant goal for post-event analyses.

Typical workflow
----------------
1. Load trials with `load_files`.
2. Split into correct / incorrect and AI ON / AI OFF.
3. Compute per-trial metrics:
   - final distances
   - collisions
   - smoothness
   - decoded intention state
   - ambiguity / switchiness
   - executed velocity statistics
4. Aggregate metrics across monkey × experiment units.
5. Plot global summaries from the saved or loaded grand summary.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon
from collections import Counter, defaultdict

# Make project modules importable when running this file from /scripts.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load import load_files
from src.trial_utils import (
    categorize_trials_by_target_and_ai,
    save_grand_summary,
    load_grand_summary,
)
from src.trajectory_metrics import (
    compute_final_distances,
    analyze_per_trial_smoothness,
    compare_smoothness_metrics,
    count_obstacle_collisions,
    determine_avoidance_side,
)
from src.stats_tests import paired_t_test
from src.plots import (
    plot_success_by_choice_state_global_final,
    plot_success_by_choice_state_global,
    plot_failure_modes_nonexec_incorrect,
    plot_failure_modes_all,
    plot_wrong_target_subtypes,
    plot_ambiguity_metric_deltas,
    plot_outcome_and_failure_modes_per_trial,
    plot_failure_modes_global,
    plot_success_by_choice_state_split,
    plot_success_by_choice_state,
    plot_bci_vs_ai_near_target,
    plot_overshoot_hist_last_half,
    plot_overshoot_timecourse,
    plot_pie_chart_failure_modes,
    plot_failure_mode_counts,
    plot_collision_heatmap_per_target,
    plot_avg_trajectories_per_target_and_avoidance,
)
from src.constants import target_mapping, target_to_obstacle_mapping
from src.AI_metrics import (
    categorize_choice_state,
    summarize_behavior_geometry,
    build_behavior_df_from_correct,
    compute_traj_similarity_by_side,
    event_aligned_input_to_target,
    summarize_input_alignment_windows,
    event_aligned_entropy_bounds,
    entropy_bounds_action_correlation,
    summarize_entropy_bounds_windows,
    metrics_summary,
    event_aligned_profiles,
)


def trial_id(tr):
    """
    Robustly extract the trial identifier from either a dict-like trial
    or an object-like trial.

    Parameters
    ----------
    tr : dict or object
        Trial object with a `trial` field/attribute.

    Returns
    -------
    int
        Trial ID.
    """
    try:
        return int(tr["trial"])
    except Exception:
        return int(getattr(tr, "trial"))


def analyze_trajectories(monkeys, experiments, base_dir):
    """
    Run the main trajectory / failure-mode / ambiguity analysis.

    This function loops over all monkey × experiment combinations and builds
    a grand summary dictionary that aggregates:
      - trajectory-distance metrics
      - collision counts
      - failure-mode proportions
      - success by decoded choice state
      - ambiguity metrics from modelVelocity
      - movement statistics from avatarVelocity

    Respawn-specific handling
    -------------------------
    For Respawn experiments:
      - post-jump segments are used where appropriate
      - targetJumpPosition is treated as the true goal
      - targetJumpTime defines the post-event segment

    Parameters
    ----------
    monkeys : list of str
        Monkey identifiers.
    experiments : list of str
        Experiment names.
    base_dir : str
        Root directory containing the data.

    Returns
    -------
    grand : dict
        Grand summary used for downstream statistics and plotting.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import defaultdict, Counter

    # ------------------------------------------------------------------
    # Helper: infer intention / choice state from decoded modelVelocity
    # ------------------------------------------------------------------
    def compute_choice_state_from_model_velocity(
        trial,
        experiment,
        known_goals,
        goal_idx,
        neighbors,
        choice_thresh=0.6,
        speed_thresh=0.1,
        obstacle_window_radius=1.5,
    ):
        """
        Estimate intention from modelVelocity and map it to a choice-state label.

        The decoded velocity vector is compared against vectors pointing from
        the current avatar position to each canonical goal. Each valid frame
        votes for the goal with the highest positive cosine similarity.

        Returns
        -------
        state : str
            Fine-grained state used for plotting:
            'correct_choice', 'neighbor_choice', 'ambiguous_choice',
            'wrong_choice', or 'no_model'.

        choice_label : str
            Coarser label used inside failure classification:
            'correct_choice', 'wrong_choice', 'ambiguous_choice', or 'no_model'.

        true_frac, neighbor_frac, other_frac : float
            Fractions of valid frames assigned to the true goal, neighboring
            goals, or other goals.

        n_rel : int
            Number of valid frames included in the analysis.
        """
        import numpy as np

        # --- Read decoded velocity ---
        mv = getattr(trial, "modelVelocity", None)
        if mv is None:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        try:
            vx = np.asarray(mv["vx"], float)
            vz = np.asarray(mv["vz"], float)
        except Exception:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        # --- Read trajectory and time base ---
        traj = getattr(trial, "avatarTrajectory", None)
        if traj is None:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        try:
            xs_all = np.asarray(traj["x"], float)
            zs_all = np.asarray(traj["z"], float)
        except Exception:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        t_all = np.asarray(traj.get("time", np.arange(len(xs_all))), float)

        T = min(len(vx), len(vz), len(xs_all), len(zs_all), len(t_all))
        if T < 5:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        vx = vx[:T]
        vz = vz[:T]
        xs_all = xs_all[:T]
        zs_all = zs_all[:T]
        t_all = t_all[:T]

        # --- Restrict to post-jump segment for Respawn tasks ---
        is_respawn = ("AI Respawn" in str(experiment))
        if is_respawn:
            event_time = getattr(trial, "targetJumpTime", None)
            if event_time is not None:
                mask_time = t_all >= float(event_time)
            else:
                mask_time = np.ones(T, dtype=bool)

            tgt = getattr(trial, "targetJumpPosition", None)
            if tgt is None:
                tgt = getattr(trial, "targetPosition", None)
        else:
            mask_time = np.ones(T, dtype=bool)
            tgt = getattr(trial, "targetPosition", None)

        true_goal = None
        if tgt is not None:
            try:
                true_goal = (float(tgt[0]), float(tgt[2]))
            except Exception:
                true_goal = None

        # --- Build validity mask ---
        v = np.stack([vx, vz], axis=1)
        speed = np.linalg.norm(v, axis=1)
        valid = np.isfinite(speed) & (speed > speed_thresh) & mask_time

        # Optionally exclude frames close to the obstacle.
        obst = getattr(trial, "obstaclePosition", None)
        if obst is not None and ("Obstacle" in str(experiment)):
            try:
                ox, oz = float(obst[0]), float(obst[2])
                dist_obs = np.sqrt((xs_all - ox) ** 2 + (zs_all - oz) ** 2)
                valid &= (dist_obs > obstacle_window_radius)
            except Exception:
                pass

        if not np.any(valid):
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        v = v[valid]
        xs = xs_all[valid]
        zs = zs_all[valid]
        n_rel = v.shape[0]

        # --- Count best-aligned goal per valid frame ---
        counts = {g: 0 for g in known_goals}

        for i in range(n_rel):
            vi = v[i]
            vi_norm = np.linalg.norm(vi)
            if vi_norm < 1e-9:
                continue

            best_goal = None
            best_cos = -np.inf

            for (ox, oz) in known_goals:
                gvec = np.array([ox - xs[i], oz - zs[i]], float)
                gnorm = np.linalg.norm(gvec)
                if gnorm < 1e-9:
                    continue

                cos = float(np.dot(vi, gvec) / (vi_norm * gnorm + 1e-12))
                if cos > best_cos:
                    best_cos = cos
                    best_goal = (ox, oz)

            if best_goal is not None and best_cos > 0.0:
                counts[best_goal] += 1

        total = sum(counts.values())
        if total == 0:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        fracs = {g: c / total for g, c in counts.items() if c > 0}
        if not fracs:
            return "no_model", "no_model", np.nan, np.nan, np.nan, 0

        best_goal = max(fracs, key=fracs.get)
        best_frac = fracs[best_goal]
        true_frac = fracs.get(true_goal, 0.0) if true_goal is not None else 0.0
        total_mass = sum(fracs.values())

        # --- Partition non-true mass into neighbor vs other ---
        neighbor_frac = 0.0
        other_frac = 0.0
        if true_goal in goal_idx:
            idx_true = goal_idx[true_goal]
            neighbor_idxs = neighbors.get(idx_true, [])
            neighbor_goals = [known_goals[i] for i in neighbor_idxs]
            neighbor_frac = sum(fracs.get(g, 0.0) for g in neighbor_goals)

        non_true_mass = max(0.0, total_mass - true_frac)
        other_frac = max(0.0, non_true_mass - neighbor_frac)

        # --- Convert fractions to discrete labels ---
        if best_frac >= choice_thresh:
            if true_goal is not None and best_goal == true_goal:
                choice_label = "correct_choice"
                state = "correct_choice"
            else:
                choice_label = "wrong_choice"
                state = "wrong_choice"
        else:
            choice_label = "ambiguous_choice"
            if (non_true_mass > 0 and neighbor_frac >= 0.8 * non_true_mass and true_goal in goal_idx):
                state = "neighbor_choice"
            else:
                state = "ambiguous_choice"

        return state, choice_label, true_frac, neighbor_frac, other_frac, n_rel

    # ------------------------------------------------------------------
    # Helper: executed movement statistics from avatarVelocity
    # ------------------------------------------------------------------
    def compute_avatar_velocity_metrics(
        trial,
        experiment,
        speed_thresh=0.02,
        obstacle_window_radius=None,
        dt_ms_fallback=50.0,
    ):
        """
        Compute per-trial metrics from executed avatarVelocity rather than decoded
        modelVelocity.

        Metrics include:
          - duration
          - median / mean / p95 speed
          - fraction of frames above a speed threshold

        The same respawn masking logic is used as in the modelVelocity analyses.
        """
        import numpy as np

        vel = getattr(trial, "avatarVelocity", None)
        traj = getattr(trial, "avatarTrajectory", None)
        if vel is None or traj is None:
            return None

        try:
            if "x" in vel and "z" in vel:
                vx = np.asarray(vel["x"], float)
                vz = np.asarray(vel["z"], float)
            else:
                vx = np.asarray(vel["vx"], float)
                vz = np.asarray(vel["vz"], float)
        except Exception:
            return None

        try:
            xs = np.asarray(traj["x"], float)
            zs = np.asarray(traj["z"], float)
            t = np.asarray(traj.get("time", np.arange(len(xs))), float)
        except Exception:
            return None

        T = min(len(vx), len(vz), len(xs), len(zs), len(t))
        if T < 5:
            return None
        vx, vz, xs, zs, t = vx[:T], vz[:T], xs[:T], zs[:T], t[:T]

        is_respawn = ("AI Respawn" in str(experiment))
        if is_respawn:
            event_time = getattr(trial, "targetJumpTime", None)
            mask_time = (t >= float(event_time)) if event_time is not None else np.ones(T, bool)
        else:
            mask_time = np.ones(T, bool)

        speed = np.sqrt(vx**2 + vz**2)
        valid = np.isfinite(speed) & mask_time

        if obstacle_window_radius is not None:
            obst = getattr(trial, "obstaclePosition", None)
            if obst is not None and ("Obstacle" in str(experiment)):
                try:
                    ox, oz = float(obst[0]), float(obst[2])
                    dist_obs = np.sqrt((xs - ox)**2 + (zs - oz)**2)
                    valid &= (dist_obs > float(obstacle_window_radius))
                except Exception:
                    pass

        if not np.any(valid):
            return None

        speed_v = speed[valid]
        t_v = t[valid]

        duration_s = np.nan
        if t_v.size >= 2 and np.all(np.isfinite(t_v)):
            dt_est = float(np.nanmedian(np.diff(t_v)))
            duration_s = (t_v[-1] - t_v[0]) / 1000.0 if dt_est > 5 else (t_v[-1] - t_v[0])
        if not np.isfinite(duration_s) or duration_s <= 0:
            duration_s = (int(speed_v.size) * dt_ms_fallback) / 1000.0

        return {
            "n_used": int(speed_v.size),
            "duration_s": float(duration_s),
            "median_speed": float(np.nanmedian(speed_v)),
            "mean_speed": float(np.nanmean(speed_v)),
            "p95_speed": float(np.nanpercentile(speed_v, 95)),
            "frac_moving": float(np.mean(speed_v > speed_thresh)),
        }

    # ------------------------------------------------------------------
    # Helper: decoded ambiguity / switchiness from modelVelocity
    # ------------------------------------------------------------------
    def compute_entropy_switch_from_model_velocity(
        trial,
        experiment,
        known_goals,
        speed_thresh=0.1,
        obstacle_window_radius=1.5,
        dt_ms_fallback=50.0,
    ):
        """
        Compute whole-trial ambiguity metrics from modelVelocity.

        Output metrics
        --------------
        H_soft_norm : float
            Soft-evidence entropy normalized to [0, 1].
        H_counts_norm : float
            Counts-based entropy normalized to [0, 1].
        switch_count : int
            Number of switches in the framewise best-goal label.
        switch_rate_hz : float
            Switch count normalized by trial duration.
        n_used : int
            Number of valid frames used.
        """
        import numpy as np

        mv = getattr(trial, "modelVelocity", None)
        traj = getattr(trial, "avatarTrajectory", None)
        if mv is None or traj is None:
            return np.nan, np.nan, 0, np.nan, 0

        try:
            vx = np.asarray(mv["vx"], float)
            vz = np.asarray(mv["vz"], float)
            xs_all = np.asarray(traj["x"], float)
            zs_all = np.asarray(traj["z"], float)
        except Exception:
            return np.nan, np.nan, 0, np.nan, 0

        t_all = np.asarray(traj.get("time", np.arange(len(xs_all))), float)

        T = min(len(vx), len(vz), len(xs_all), len(zs_all), len(t_all))
        if T < 5:
            return np.nan, np.nan, 0, np.nan, 0

        vx, vz = vx[:T], vz[:T]
        xs_all, zs_all, t_all = xs_all[:T], zs_all[:T], t_all[:T]

        is_respawn = ("AI Respawn" in str(experiment))
        if is_respawn:
            event_time = getattr(trial, "targetJumpTime", None)
            if event_time is not None:
                mask_time = t_all >= float(event_time)
            else:
                mask_time = np.ones(T, dtype=bool)
        else:
            mask_time = np.ones(T, dtype=bool)

        v = np.stack([vx, vz], axis=1)
        speed = np.linalg.norm(v, axis=1)
        valid = np.isfinite(speed) & (speed > speed_thresh) & mask_time

        obst = getattr(trial, "obstaclePosition", None)
        if obst is not None and ("Obstacle" in str(experiment)):
            try:
                ox, oz = float(obst[0]), float(obst[2])
                dist_obs = np.sqrt((xs_all - ox) ** 2 + (zs_all - oz) ** 2)
                valid &= (dist_obs > obstacle_window_radius)
            except Exception:
                pass

        if not np.any(valid):
            return np.nan, np.nan, 0, np.nan, 0

        v = v[valid]
        xs = xs_all[valid]
        zs = zs_all[valid]
        tt = t_all[valid]

        n_used = int(v.shape[0])
        G = len(known_goals)
        if n_used < 3 or G < 2:
            return np.nan, np.nan, 0, np.nan, n_used

        # Cosine evidence to all goals for each valid frame.
        vnorm = np.linalg.norm(v, axis=1) + 1e-12
        cos_mat = np.full((n_used, G), np.nan, float)

        for gi, (gx, gz) in enumerate(known_goals):
            gvec = np.stack([gx - xs, gz - zs], axis=1)
            gnorm = np.linalg.norm(gvec, axis=1) + 1e-12
            cos_mat[:, gi] = np.sum(v * gvec, axis=1) / (vnorm * gnorm)

        best_idx = np.nanargmax(cos_mat, axis=1)
        best_cos = cos_mat[np.arange(n_used), best_idx]
        labels = np.where(best_cos > 0.0, best_idx, -1)

        labels_valid = labels[labels >= 0]
        if labels_valid.size < 2:
            return np.nan, np.nan, 0, np.nan, n_used

        counts = np.bincount(labels_valid, minlength=G).astype(float)
        p_counts = counts / np.sum(counts)
        p_counts = p_counts[p_counts > 0]
        H_counts = float(-np.sum(p_counts * np.log(p_counts)))
        H_counts_norm = H_counts / np.log(G)

        w = np.maximum(0.0, cos_mat)
        rs = np.sum(w, axis=1, keepdims=True)
        w = w / np.clip(rs, 1e-12, None)
        p_soft = np.nanmean(w, axis=0)
        p_soft = np.clip(p_soft, 0.0, None)
        if np.sum(p_soft) <= 0:
            H_soft_norm = np.nan
        else:
            p_soft = p_soft / np.sum(p_soft)
            p_soft2 = p_soft[p_soft > 0]
            H_soft = float(-np.sum(p_soft2 * np.log(p_soft2)))
            H_soft_norm = H_soft / np.log(G)

        switch_count = int(np.sum(labels_valid[1:] != labels_valid[:-1]))

        duration_s = np.nan
        if np.all(np.isfinite(tt)) and tt.size >= 2:
            dt_est = float(np.nanmedian(np.diff(tt)))
            if dt_est > 5:
                duration_s = (tt[-1] - tt[0]) / 1000.0
            else:
                duration_s = (tt[-1] - tt[0])
        if not np.isfinite(duration_s) or duration_s <= 0:
            duration_s = (n_used * dt_ms_fallback) / 1000.0

        switch_rate_hz = float(switch_count / duration_s) if duration_s > 0 else np.nan

        return float(H_soft_norm), float(H_counts_norm), switch_count, switch_rate_hz, n_used

    def _softmax(x, tau=0.2):
        """
        Temperature-controlled softmax used for ambiguity estimation.
        """
        x = np.asarray(x, float)
        x = x / max(tau, 1e-9)
        x = x - np.nanmax(x)
        ex = np.exp(x)
        s = np.nansum(ex)
        if not np.isfinite(s) or s <= 0:
            return None
        return ex / s

    def compute_ambiguity_metrics_from_model_velocity(
        trial,
        experiment,
        known_goals,
        speed_thresh=0.1,
        obstacle_window_radius=1.5,
        tau=0.2,
    ):
        """
        Compute per-trial ambiguity metrics from modelVelocity over valid frames.

        This version uses a soft distribution over candidate goals for each frame
        and summarizes:
          - median normalized entropy across frames
          - number of switches in the framewise argmax-goal sequence
          - switch rate in Hz
          - number of valid frames used
        """
        # ... keep your current body unchanged ...
        pass

    def _summarize_ambiguity_rows(rows, monkey, experiment):
        """
        Summarize per-trial ambiguity rows into per-unit medians.

        The output is grouped by:
          - monkey
          - experiment
          - condition (AI / No AI)
          - subset (global / neighbor / all)
          - outcome (all / correct / incorrect)
        """
        # ... keep your current body unchanged ...
        pass

    # ------------------------------------------------------------------
    # Per-unit ambiguity accumulator
    # ------------------------------------------------------------------
    # This stores only intention-level ambiguous trials and is later converted
    # into per-unit summaries.
    amb_acc = {
        "AI":    {"all": [], "global": [], "neighbor": []},
        "No AI": {"all": [], "global": [], "neighbor": []},
    }

    def _log_ambiguity_row(
        trial, condition, success, state, choice_label,
        true_frac, neighbor_frac, other_frac, n_rel,
        reason=None, ambiguity_type=None
    ):
        """
        Append one trial's ambiguity metrics to the grand summary and to the
        per-unit ambiguity accumulator.

        Only trials with choice_label == 'ambiguous_choice' are routed into
        the ambiguity-specific per-unit summaries.
        """
        # ... keep your current body unchanged ...
        pass

    def compute_entropy_switch_from_avatar_velocity(
        trial,
        experiment,
        known_goals,
        speed_thresh=0.1,
        obstacle_window_radius=1.5,
        dt_ms_fallback=50.0,
        tau=0.2,
    ):
        """
        Compute ambiguity metrics from executed avatarVelocity.

        This mirrors the decoded-ambiguity analysis based on modelVelocity,
        but uses the actual executed movement shown on screen.
        """
        # ... keep your current body unchanged ...
        pass

    # ------------------------------------------------------------------
    # Grand accumulators across all monkeys and experiments
    # ------------------------------------------------------------------
    grand = {
        "dist_target_on": [], "dist_target_off": [],
        "dist_obst_on": [],   "dist_obst_off": [],
        "correct_target_on": [], "correct_target_off": [],
        "correct_obst_on": [],   "correct_obst_off": [],

        "df_all_rows": [],

        "collisions": {
            "correct_on": 0, "correct_off": 0,
            "incorrect_on": 0, "incorrect_off": 0
        },
        "n_trials": {
            "correct_on": 0, "correct_off": 0,
            "incorrect_on": 0, "incorrect_off": 0
        },

        "fail_AI_ON": {
            "stuck_obstacle": 0, "wrong_target": 0,
            "wrong_target_monkey": 0, "overshoot": 0,
            "other": 0, "not_long_enough": 0,
            "not_close_to_true_z": 0
        },
        "fail_AI_OFF": {
            "stuck_obstacle": 0, "wrong_target": 0,
            "wrong_target_monkey": 0, "overshoot": 0,
            "other": 0, "not_long_enough": 0,
            "not_close_to_true_z": 0
        },

        "ai_fail_total": 0,
        "ai_fail_obstacle": 0,
        "no_ai_fail_total": 0,
        "no_ai_fail_obstacle": 0,

        "nb_ai_trials": 0,
        "nb_noai_trials": 0,

        # Per monkey × experiment summaries used for global plots
        "failure_props_per_unit": [],
        "success_state_per_unit": [],
        "wrong_target_subtypes_per_unit": [],

        # Per-trial and per-unit ambiguity summaries
        "ambiguity_metrics_per_trial": [],
        "ambiguity_metrics_per_unit": [],
        "ambiguity_trial_rows": [],

        # Executed velocity summaries
        "avatar_vel_trial_rows": [],
        "avatar_vel_per_unit": [],

        # Executed ambiguity summaries (avatarVelocity)
        "avatar_ambiguity_trial_rows": [],
        "avatar_ambiguity_per_unit": [],
    }

    # ------------------------------------------------------------------
    # Main outer loop over experiment × monkey
    # ------------------------------------------------------------------
    for experiment in experiments:
        for monkey in monkeys:
            print(f"\n--- Analyzing trajectories for Experiment: {experiment}, Monkey: {monkey} ---")

            try:
                (
                    all_trials,
                    all_correct,
                    all_incorrect,
                    all_training,
                    all_channels,
                    nb_channels,
                    pkl_files,
                    ai_trials,
                ) = load_files(experiment, monkey, base_dir=base_dir)
                if all_trials == []:
                    continue
            except FileNotFoundError as e:
                print(f"[ERROR] Data not found for {monkey} in {experiment}: {e}")
                continue

            # ------------------------------------------------------------------
            # Per-unit accumulators (this monkey × experiment only)
            # ------------------------------------------------------------------
            dist_target_on, dist_target_off = [], []
            dist_obst_on, dist_obst_off = [], []
            correct_target_on, correct_target_off = [], []
            correct_obst_on, correct_obst_off = [], []

            all_trial_dicts = []
            trial_counter_ai = 0
            trial_counter_noai = 0

            ai_fail_total = 0
            ai_fail_obstacle = 0
            no_ai_fail_total = 0
            no_ai_fail_obstacle = 0

            collisions = {
                "correct_on": 0,
                "correct_off": 0,
                "incorrect_on": 0,
                "incorrect_off": 0,
            }
            n_trials = {
                "correct_on": 0,
                "correct_off": 0,
                "incorrect_on": 0,
                "incorrect_off": 0,
            }

            overshoot_trials = []
            correct_trials_all = []

            ambiguity_trial_rows_unit = []
            avatar_vel_trial_rows_unit = []
            avatar_ambig_trial_rows_unit = []

            # Fine-grained failure categories used for global failure plots.
            category_counts_AI_ON = defaultdict(int)
            category_counts_AI_OFF = defaultdict(int)

            # Coarse failure reasons used for manuscript-level summaries.
            reason_counts_AI_ON = defaultdict(int)
            reason_counts_AI_OFF = defaultdict(int)

            info_reason_counts_AI_ON = defaultdict(Counter)
            info_reason_counts_AI_OFF = defaultdict(Counter)

            # Success by decoded choice state.
            success_by_choice_AI_ON = defaultdict(int)
            total_by_choice_AI_ON = defaultdict(int)
            success_by_choice_AI_OFF = defaultdict(int)
            total_by_choice_AI_OFF = defaultdict(int)

            # "Behavioral rescue" version that excludes execution-only failures.
            behavior_success_by_choice_AI_ON = defaultdict(int)
            behavior_total_by_choice_AI_ON = defaultdict(int)
            behavior_success_by_choice_AI_OFF = defaultdict(int)
            behavior_total_by_choice_AI_OFF = defaultdict(int)

            # Canonical goal geometry and neighbor relationships.
            known_goals = [
                (7.0, 6.0),
                (3.5, 8.5),
                (0.0, 9.2),
                (-3.5, 8.5),
                (-7.0, 6.0),
            ]
            goal_idx = {g: i for i, g in enumerate(known_goals)}
            neighbors = {
                0: [1],
                1: [0, 2],
                2: [1, 3],
                3: [2, 4],
                4: [3],
            }

            # Failure categories treated as execution-level failures.
            execution_reasons = {
                "stuck_obstacle",
                "overshoot",
                "not_long_enough",
                "not_close_to_true_z",
            }

            # Per-trial intention summaries for correct trials.
            true_fracs_correct_on = []
            true_fracs_correct_off = []
            neighbor_fracs_correct_on = []
            neighbor_fracs_correct_off = []
            other_fracs_correct_on = []
            other_fracs_correct_off = []
            choice_labels_correct_on = Counter()
            choice_labels_correct_off = Counter()

            # Optional grouping for trajectory-shape comparisons.
            from collections import defaultdict as _dd
            traj_groups = _dd(lambda: {"AI": [], "No AI": []})

            def map_target_label_from_pos(pos):
                """
                Convert a target position into a stable string label using the
                predefined canonical target mapping.
                """
                key = (
                    round(float(pos[0]), 1),
                    round(float(pos[1]), 2),
                    round(float(pos[2]), 1),
                )
                return target_mapping.get(key, "unknown")

            # Axes used only if obstacle-contact trials are visualized during classification.
            fig, ax = plt.subplots(figsize=(8, 8))
            beh_rows_all_sessions = []

            # ==============================================================
            # Session loop
            # ==============================================================
            for sess_idx, (correct_trials, incorrect_trials) in enumerate(zip(all_correct, all_incorrect)):
                # ... keep your current session-loop body unchanged ...
                pass

            # ==============================================================
            # After all sessions for this monkey × experiment
            # ==============================================================
            ax.set_aspect("equal")
            ax.set_xlabel("X position (m)")
            ax.set_ylabel("Z position (m)")
            ax.set_title("Obstacle-contact No-AI incorrect trials")
            ax.grid(True)
            plt.close(fig)

            # Per-unit ambiguity summaries
            def _nanmed(x):
                """
                Safe nan-median helper.
                """
                x = np.asarray(x, float)
                return float(np.nanmedian(x)) if np.any(np.isfinite(x)) else np.nan

            def _append_amb_summary(condition, subset, L):
                """
                Append one per-unit ambiguity summary to the grand dictionary.
                """
                if len(L) == 0:
                    return
                grand["ambiguity_metrics_per_unit"].append({
                    "monkey": monkey,
                    "experiment": experiment,
                    "condition": condition,
                    "subset": subset,
                    "n_trials": int(len(L)),
                    "median_entropy_soft_norm": _nanmed([d["H"] for d in L]),
                    "median_switch_rate_hz": _nanmed([d["sw_hz"] for d in L]),
                    "median_switch_count": _nanmed([d["swc"] for d in L]),
                    "median_n_used": _nanmed([d["n_used"] for d in L]),
                    "p_success": _nanmed([d["success"] for d in L]),
                })

            for cond in ["AI", "No AI"]:
                _append_amb_summary(cond, "all", amb_acc[cond]["all"])
                _append_amb_summary(cond, "global", amb_acc[cond]["global"])
                _append_amb_summary(cond, "neighbor", amb_acc[cond]["neighbor"])

            def _summarize_avatar_ambig_rows(rows, monkey, experiment):
                """
                Summarize executed ambiguity rows into per-unit medians.
                """
                # ... keep your current body unchanged ...
                pass

            grand["avatar_ambiguity_per_unit"].extend(
                _summarize_avatar_ambig_rows(avatar_ambig_trial_rows_unit, monkey, experiment)
            )

            if "wrong_target_subtypes_per_unit" not in grand:
                grand["wrong_target_subtypes_per_unit"] = []

            def _add_wt_row(condition, n_total, info_counts):
                """
                Append one per-unit wrong-target subtype summary.

                Fractions are expressed relative to all trials in the unit,
                not relative to incorrect trials only.
                """
                # ... keep your current body unchanged ...
                pass

            grand["ambiguity_trial_rows"].extend(ambiguity_trial_rows_unit)

            grand["ambiguity_metrics_per_unit"].extend(
                _summarize_ambiguity_rows(ambiguity_trial_rows_unit, monkey, experiment)
            )

            def _summarize_avatar_vel_rows(rows, monkey, experiment):
                """
                Summarize executed-velocity rows into per-unit medians by
                condition and outcome.
                """
                # ... keep your current body unchanged ...
                pass

            grand["avatar_vel_per_unit"].extend(
                _summarize_avatar_vel_rows(avatar_vel_trial_rows_unit, monkey, experiment)
            )

            # ... keep the rest of your current per-unit aggregation unchanged ...

    return grand


def classify_no_ai_failure_obstacle_contact(
    trial,
    obstacle_half=0.45,
    robot_radius=0.25,
    margin=0.20,
    min_frames_inside=1,
    require_not_pass=True,
    plot_ax=None,
    plot_alpha=0.35,
):
    """
    Detect whether a trial became stuck at the obstacle.

    The criterion is:
      1. The avatar enters an inflated obstacle square.
      2. It remains inside for at least `min_frames_inside` consecutive frames.
      3. Optionally, it never passes the obstacle along z.

    Parameters
    ----------
    trial : object
        Trial with avatarTrajectory and obstaclePosition.
    obstacle_half : float
        Half-size of the physical obstacle square.
    robot_radius : float
        Radius of the avatar / sphere.
    margin : float
        Extra safety inflation for the stuck test.
    min_frames_inside : int
        Minimum consecutive frames inside the inflated square.
    require_not_pass : bool
        If True, discard trials that clearly pass beyond the obstacle in z.
    plot_ax : matplotlib.axes.Axes or None
        Optional axis for visualizing detected stuck trials.

    Returns
    -------
    is_stuck : bool
        Whether the trial is classified as obstacle-stuck.
    info : dict
        Diagnostic information.
    """
    # ... keep your current body unchanged ...
    pass


def classify_no_ai_failure_reason(
    trial,
    all_goal_positions=None,
    target_window_size=4.2,
    min_consecutive_in_true=10,
    pass_margin=0.0,
    z_margin=0.0,
    **stuck_kwargs,
):
    """
    Classify the reason for failure on a single trial.

    Output labels
    -------------
    Returns one of:
      - 'stuck_obstacle'
      - 'overshoot'
      - 'not_long_enough'
      - 'wrong_target'
      - 'wrong_target_monkey'
      - 'not_close_to_true_z'
      - 'other'

    Notes
    -----
    Standard tasks:
      - use the full trial
      - true goal = targetPosition

    Respawn tasks:
      - use the post-jump segment when possible
      - true goal = targetJumpPosition
      - event time = targetJumpTime

    The returned `info` dict also includes intention-based diagnostics derived
    from modelVelocity:
      - choice_label
      - intention fractions
      - ambiguity_type
      - wrong-target subtype information
    """
    # ... keep your current body unchanged ...
    pass


def compute_intention_from_model_velocity_full_trial(
    trial,
    known_goals,
    choice_thresh=0.6,
    speed_thresh=0.1,
):
    """
    Compute intention fractions from modelVelocity over the full trial.

    This is a simpler whole-trial helper than the respawn-aware choice-state
    function used above.

    Returns
    -------
    choice_label : str
        'correct_choice', 'wrong_choice', 'ambiguous_choice', or 'no_model'
    fracs : dict
        Fraction of valid frames assigned to each goal.
    best_goal : tuple or None
        Goal with the largest fraction.
    best_frac : float
        Fraction assigned to the best goal.
    true_frac : float
        Fraction assigned to the actual target of this trial.
    """
    # ... keep your current body unchanged ...
    pass


def is_within_target_window_2D(states, target, window_size, min_consecutive=10):
    """
    Check whether a 2D trajectory enters and remains inside a square target
    window for at least `min_consecutive` frames.

    Parameters
    ----------
    states : iterable of (x, z)
        2D positions.
    target : tuple
        Target center (x, z).
    window_size : float
        Side length of the target window.
    min_consecutive : int
        Number of consecutive frames required.

    Returns
    -------
    bool
        True if the criterion is met, otherwise False.
    """
    half_window = window_size / 2.0
    count = 0
    for x, z in states:
        if (
            target[0] - half_window <= x <= target[0] + half_window
            and target[1] - half_window <= z <= target[1] + half_window
        ):
            count += 1
            if count >= min_consecutive:
                return True
        else:
            count = 0
    return False


def _goal_color(gx, gz, palette=None):
    """
    Deterministically assign a plotting color to a goal location.
    """
    if palette is None:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
    key = (round(gx, 2), round(gz, 2))
    idx = hash(key) % len(palette)
    return palette[idx]


def plot_failure_modes(trials, **kwargs):
    """
    Visualize incorrect-trial trajectories grouped by failure reason.

    Parameters
    ----------
    trials : list or dict of lists
        Trial objects. If a dict is provided, all contained trial lists are flattened.
    kwargs : dict
        Additional keyword arguments passed to `classify_no_ai_failure_reason`.

    Returns
    -------
    dict
        Counts per failure category.
    """
    # ... keep your current body unchanged ...
    pass


def table_avatar_entropy_switch_on_ambiguous_trials(
    grand_summary,
    subset="global",
    outcomes=("all", "correct", "incorrect"),
    metrics=("entropy_soft_norm", "switch_rate_hz", "switch_count"),
    key_trial="avatar_ambiguity_trial_rows",
    key_unit="avatar_ambiguity_per_unit",
):
    """
    Build paired AI vs No-AI summary tables for executed ambiguity metrics.

    The function prefers per-unit summaries if available, otherwise it derives
    them from per-trial rows.

    Returns
    -------
    df_unit : pandas.DataFrame
        Per-unit table.
    df_pair : pandas.DataFrame
        Paired AI vs No-AI comparison table with Wilcoxon p-values.
    """
    # ... keep your current body unchanged ...
    pass


def failure_mode_pvalues(grand_summary, modes=None):
    """
    Compute paired Wilcoxon statistics for failure-mode proportions
    across monkey × experiment units.

    Parameters
    ----------
    grand_summary : dict
        Output of `analyze_trajectories`.
    modes : list of str or None
        Failure modes to test. If None, a default subset is used.

    Returns
    -------
    pandas.DataFrame
        One row per failure mode with medians and p-values.
    """
    # ... keep your current body unchanged ...
    pass


def main():
    """
    Entry point for trajectory / failure-mode analysis.

    If --base_dir is not provided, it is loaded from config.yaml in the
    project root.

    Notes
    -----
    The current default behavior loads a previously saved grand summary and
    generates plots from it. To recompute the summary from raw data, uncomment
    the analysis + save lines below.
    """
    parser = argparse.ArgumentParser(
        description="Analyze AI ON vs AI OFF trajectory and failure metrics."
    )
    parser.add_argument(
        "--monkeys",
        nargs="+",
        default=["Monkey 1", "Monkey 2"],
        help="List of monkeys to analyze",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["AI Obstacle", "AI Appearing Obstacle", "AI Appearing Obstacle 2"],
        help="List of experiment names",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Path to the root data directory",
    )
    args = parser.parse_args()

    # Load base_dir from config.yaml if not provided on the command line.
    if args.base_dir is None:
        import yaml
        import pathlib

        config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        args.base_dir = config.get("base_dir")
        if args.base_dir is None:
            raise ValueError("base_dir must be specified via CLI or in config.yaml")

    # Recompute grand summary from raw data if needed.
    grand_summary = analyze_trajectories(args.monkeys, args.experiments, args.base_dir)
    save_grand_summary(grand_summary, args.experiments)

    # Default behavior: load an existing grand summary and plot from it.
    grand_summary = load_grand_summary()

    plot_success_by_choice_state_global_final(grand_summary, behavioral_only=False)
    plot_failure_modes_all(grand_summary, save_prefix="failure_modes_rebinned")
    plot_failure_modes_global(grand_summary)


if __name__ == "__main__":
    main()