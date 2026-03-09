import numpy as np
def compute_time_to_target(data):
    """Compute time-to-target (TTT) for trial(s).

    Args:
        data: Single trial or list of sessions (each a list of trials).

    Returns:
        float or list of floats: TTT in ms
    """
    def find_first_non_zero(arr):
        return np.flatnonzero(arr)[0] if np.any(arr) else 0

    if isinstance(data, list) and hasattr(data[0], '__iter__'):
        times = []
        for session in data:
            selected = [t for t in session if getattr(t, "answer", 0) == 1]
            for trial in selected:
                start = find_first_non_zero(trial.avatarTrajectory['z'])
                t = trial.avatarTrajectory['time'][-10] - trial.avatarTrajectory['time'][start]
                times.append(t)
        return times
    else:
        index = find_first_non_zero(data.avatarTrajectory['z'])
        return data.avatarTrajectory['time'][-10] - data.avatarTrajectory['time'][index]