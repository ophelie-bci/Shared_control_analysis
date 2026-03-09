import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

def compute_mean_squared_jerk(position, dt):
    """Compute the mean squared jerk for a 1D trajectory."""
    if len(position) < 4:
        return np.nan
    pos = np.array(position)
    jerk = np.diff(pos, n=3) / (dt ** 3)
    msj = np.mean(jerk ** 2)
    return msj

def determine_avoidance_side(x, z, obstacle_x, obstacle_z):
    """
    Decide avoidance side based on whether the trajectory passed to the left or right
    of the obstacle center (x,z).
    """
    # Find the index of the closest z to the obstacle
    idx_closest = np.argmin(np.abs(z - obstacle_z))
    x_at_closest = x[idx_closest]

    if x_at_closest < obstacle_x:
        return "left"
    else:
        return "right"

def count_obstacle_collisions(trial, obstacle_pos, r_avatar=0.25, r_obstacle=0.45):
    """Count the number of timepoints where the avatar intersects the obstacle."""
    if obstacle_pos is None:
        return 0

    obstacle = np.array([obstacle_pos[0], obstacle_pos[2]])
    traj_x = np.array(trial.avatarTrajectory['x'])
    traj_z = np.array(trial.avatarTrajectory['z'])
    traj = np.stack([traj_x, traj_z], axis=1)

    dists = np.linalg.norm(traj - obstacle, axis=1)
    collision_radius = r_avatar + r_obstacle
    return int(np.sum(dists <= collision_radius))

def analyze_per_trial_smoothness(df, dt=0.05):
    """Compute normalized MSJ and speed metrics for each trajectory."""
    ai_trials = set(df[df['condition'] == 'AI']['trial_id'])
    no_ai_trials = set(df[df['condition'] == 'No AI']['trial_id'])

    target_labels = {
        trial_id: (round(pos[0], 2), round(pos[2], 2))
        for trial_id, pos in enumerate(df['target_positions'])
    }
    df['target'] = df['trial_id'].map(target_labels)

    trial_results = []
    for trial_id in df['trial_id'].unique():
        trial_df = df[df['trial_id'] == trial_id]
        x = trial_df['x'].values
        z = trial_df['z'].values
        vx = trial_df['vx'].values
        vz = trial_df['vz'].values

        nb_time_points = len(trial_df)
        spatial_diffs = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
        traj_length = np.sum(spatial_diffs)

        speed = np.sqrt(vx**2 + vz**2)
        avg_speed = np.mean(speed)
        peak_speed = np.max(speed)
        var_speed = np.var(speed)

        msj_x = compute_mean_squared_jerk(x, dt)
        msj_z = compute_mean_squared_jerk(z, dt)
        avg_msj = np.nanmean([msj_x, msj_z])
        norm_msj = avg_msj / nb_time_points if nb_time_points else np.nan

        condition = 'AI' if trial_id in ai_trials else 'No AI' if trial_id in no_ai_trials else 'Unknown'
        target = trial_df['target'].iloc[0] if len(trial_df) > 0 else (np.nan, np.nan)

        trial_results.append({
            'trial_id': trial_id,
            'condition': condition,
            'length': traj_length,
            'nb_time_points': nb_time_points,
            'msj': avg_msj,
            'norm_msj': norm_msj,
            'target': target,
            'avg_speed': avg_speed,
            'peak_speed': peak_speed,
            'var_speed': var_speed,
        })

    return pd.DataFrame(trial_results)

def compare_smoothness_metrics(df_summary):
    """Run t-test and Mann–Whitney U test on normalized MSJ for AI vs No-AI."""
    ai_group = df_summary[df_summary['condition'] == 'AI']['norm_msj'].dropna()
    no_ai_group = df_summary[df_summary['condition'] == 'No AI']['norm_msj'].dropna()

    t_stat = ttest_ind(ai_group, no_ai_group, equal_var=False)
    u_stat = mannwhitneyu(ai_group, no_ai_group, alternative='two-sided')

    return {
        'ai_on_norm_msj': np.mean(ai_group),
        'no_ai_on_norm_msj': np.mean(no_ai_group),
        't-test': t_stat,
        'Mann–Whitney': u_stat
    }

def compute_final_distances(trial, target_pos, obstacle_pos):
    """
    Compute:
    - Final Euclidean distance from end of trajectory to target
    - Minimum distance to obstacle over the entire trajectory (obstacle clearance)
    """
    # Final distance to target
    x_end = trial.avatarTrajectory['x'][-1]
    z_end = trial.avatarTrajectory['z'][-1]
    end = np.array([x_end, z_end])
    target = np.array([target_pos[0], target_pos[2]])
    d_target = np.linalg.norm(end - target)

    # Obstacle clearance = minimum distance during the trajectory
    if obstacle_pos is not None:
        obstacle = np.array([obstacle_pos[0], obstacle_pos[2]])
        traj_x = np.array(trial.avatarTrajectory['x'])
        traj_z = np.array(trial.avatarTrajectory['z'])
        traj = np.stack([traj_x, traj_z], axis=1)
        dists = np.linalg.norm(traj - obstacle, axis=1)
        d_obstacle = np.min(dists)
    else:
        d_obstacle = np.nan

    return d_target, d_obstacle