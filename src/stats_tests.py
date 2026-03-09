import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from src.constants import target_to_obstacle_mapping

def paired_success_test(success_on, success_off, alternative="two-sided"):
    """
    Paired test across sessions: AI ON vs AI OFF success rates.

    Parameters
    ----------
    success_on, success_off : array-like
        Per-session success rates (floats in [0,1]).
    alternative : {"two-sided","greater","less"}
        "greater" tests ON > OFF.

    Returns
    -------
    out : dict
        Contains wilcoxon statistic/p, n used, deltas, and summary stats.
    """
    import numpy as np
    from scipy import stats

    on  = np.asarray(success_on, dtype=float)
    off = np.asarray(success_off, dtype=float)

    # basic sanity
    if on.shape != off.shape:
        raise ValueError(f"ON and OFF must have same shape. Got {on.shape} vs {off.shape}.")

    # drop NaNs pairwise
    mask = np.isfinite(on) & np.isfinite(off)
    on, off = on[mask], off[mask]

    delta = on - off
    n_total = len(delta)

    # Wilcoxon drops exact zeros; keep track
    n_zero = int(np.sum(delta == 0))
    delta_nz = delta[delta != 0]
    n_used = len(delta_nz)

    out = {
        "n_total": n_total,
        "n_zero": n_zero,
        "n_used": n_used,
        "mean_delta": float(np.mean(delta)) if n_total else np.nan,
        "median_delta": float(np.median(delta)) if n_total else np.nan,
        "deltas": delta,
    }

    if n_used < 1:
        out.update({"W": np.nan, "p": np.nan, "note": "All paired differences are zero or no data."})
        return out

    # scipy Wilcoxon on paired samples (preferred) or on delta
    # Using mode='auto' works across scipy versions for small n.
    res = stats.wilcoxon(on, off, alternative=alternative, zero_method="wilcox", mode="auto")
    out.update({"W": float(res.statistic), "p": float(res.pvalue)})
    return out

def compute_success_rate(correct_trials, incorrect_trials):
    """
    Compute success rate as ratio of correct trials to total trials.
    Returns one value per session.
    """
    success_rates = []
    for correct, incorrect in zip(correct_trials, incorrect_trials):
        total_trials = len(correct) + len(incorrect)
        if total_trials == 0:
            success = np.nan
        else:
            success = len(correct) / total_trials
        success_rates.append(success)
    return success_rates

def paired_t_test(ai_on_rates, ai_off_rates):
    """
    Perform paired t-test between AI ON and AI OFF success rates.
    """
    ai_on = np.array(ai_on_rates)
    ai_off = np.array(ai_off_rates)
    
    if len(ai_on) != len(ai_off):
        raise ValueError("Lists must have the same length for paired t-test")

    t_stat, p_val = stats.ttest_rel(ai_on, ai_off, nan_policy='omit')
    return t_stat, p_val

def permutation_test_success_rate(trials, labels, n_permutations=10000, seed=42):
    """
    Estimate chance level using permutation test.
    Input:
    - trials: list of bool (True if correct, False if incorrect)
    - labels: list of group labels (same length as trials)
    - n_permutations: number of shuffles

    Output:
    - observed success rate
    - null distribution
    - empirical p-value
    """
    rng = np.random.default_rng(seed)
    trials = np.array(trials)
    labels = np.array(labels)
    
    observed = np.mean(trials)
    null_dist = []

    for _ in range(n_permutations):
        shuffled = rng.permutation(trials)
        null_dist.append(np.mean(shuffled))

    null_dist = np.array(null_dist)
    p_value = (np.sum(null_dist >= observed) + 1) / (n_permutations + 1)

    return observed, null_dist, p_value

COLLISION_RADIUS = 0.45 + 0.25  # obstacle radius + sphere radius

def is_within_target_window_2D(states, target, window_size, min_consecutive=10):
    half_window = window_size / 2.0
    count = 0
    for x, z in states:
        if (target[0] - half_window <= x <= target[0] + half_window and
                target[1] - half_window <= z <= target[1] + half_window):
            count += 1
            if count >= min_consecutive:
                return True
        else:
            count = 0
    return False


def hit_obstacle(states_2d, obstacle_pos, radius=0.7, min_consecutive=5, plot_on_hit=False):
    count = 0
    obstacle_pos = np.array(obstacle_pos[0], obstacle_pos[2])
    for x, z in states_2d:
        if np.linalg.norm(np.array([x, z]) - np.array(obstacle_pos)) < radius:
            count += 1
            if count >= min_consecutive:
                if plot_on_hit:
                    # Extract full trajectory for plotting
                    xs, zs = zip(*states_2d)

                    plt.figure(figsize=(5, 5))
                    plt.plot(xs, zs, label='Trajectory', color='blue')
                    plt.scatter(*obstacle_pos, color='red', label='Obstacle', s=100)
                    circle = plt.Circle(obstacle_pos, radius, color='red', fill=False, linestyle='--')
                    plt.gca().add_patch(circle)
                    plt.title("Collision Detected")
                    plt.xlabel("X")
                    plt.ylabel("Z")
                    plt.axis('equal')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                return True
        else:
            count = 0
    return False


def calculate_success(states, targs, obstacles):
    results = []
    for s, t, o in zip(states, targs, obstacles):
        s = np.array(s)[:, [0, 2]]  # x, z only
        t_2d = (round(t[0], 1), round(t[2], 1))
        success = is_within_target_window_2D(s, t_2d, window_size=4.2)
        # obstacle_pos = target_to_obstacle_mapping.get(t_2d, None)
        obstacle_pos = o
        if obstacle_pos is not None and hit_obstacle(s, obstacle_pos):
            success = False
        results.append(success)
    return np.mean(results)


def compute_chance_level(trials, experiment, num_permutations=1000):
    """
    Estimate chance level by shuffling targets and checking success,
    with obstacle collisions accounted for.
    """
    conc_trials = [item for sublist in trials for item in sublist]
    states_data = [list(zip(trial.avatarTrajectory['x'],
                            trial.avatarTrajectory['y'],
                            trial.avatarTrajectory['z'])) for trial in conc_trials]
    targets = [trial.targetPosition for trial in conc_trials]
    obstacles = [trial.obstaclePosition for trial in conc_trials]
    observed = calculate_success(states_data, targets, obstacles)

    null_distribution = [
        calculate_success(states_data, np.random.permutation(targets), obstacles)
        for _ in range(num_permutations)
    ]

    return np.mean(null_distribution), observed, null_distribution
