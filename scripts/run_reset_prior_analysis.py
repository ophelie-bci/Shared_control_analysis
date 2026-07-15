import os
import glob
import pickle
import numpy as np
from pathlib import Path
import yaml
import pathlib
import pandas as pd

# -------------------------------------------------------------------------
# User-defined analysis settings
# -------------------------------------------------------------------------
# Monkey, experiment, and subfolder to analyze. The script will look for
# session-level PKL files inside:
#   <base_dir>/<MONKEY>/<EXPERIMENT>/<SUBFOLDER>/
MONKEY = "Monkey 1"
EXPERIMENT = "AI Respawn"
SUBFOLDER = "resetPriorFiles"


def load_base_dir_from_config(config_path="config.yaml"):
    """
    Load the root data directory from config.yaml.

    Parameters
    ----------
    config_path : str, optional
        Ignored in the current implementation. The config file is resolved
        relative to the project root using the location of this script.

    Returns
    -------
    str
        Value of 'base_dir' from config.yaml.

    Notes
    -----
    The config file is expected to live in the project root, i.e. one level
    above the directory containing this script.
    """
    config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["base_dir"]


def compute_session_success_rate(trials):
    """
    Compute success rate for a single session.

    Parameters
    ----------
    trials : list of dict
        List of trial dictionaries loaded from one PKL file.

    Returns
    -------
    success_rate : float
        Fraction of trials with answer == 1.
    n_success : int
        Number of successful trials.
    n_total : int
        Total number of trials in the session.

    Notes
    -----
    This assumes that each trial dictionary contains an 'answer' field where:
      - 1 = success
      - non-1 = failure
    """
    if not trials:
        return np.nan, 0, 0

    n_total = len(trials)
    n_success = sum(tr.get("answer") == 1 for tr in trials)
    success_rate = n_success / n_total
    return success_rate, n_success, n_total


def main(base_dir=None):
    """
    Load reset-prior PKL files for one monkey × experiment condition and
    summarize session-level success rates.

    Workflow
    --------
    1. Load base_dir from config.yaml.
    2. Build the input directory from MONKEY / EXPERIMENT / SUBFOLDER.
    3. Find all files matching:
           reset_prior_analyzes_*.pkl
    4. For each file:
         - load trials
         - compute session success rate
         - print session summary
    5. Print the average success rate across sessions.
    """
    if base_dir is None:
        base_dir = load_base_dir_from_config("config.yaml")
    input_dir = os.path.join(base_dir, MONKEY, EXPERIMENT, SUBFOLDER)

    # Check that the expected data directory exists
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory is not a directory: {input_dir}")

    # Load all reset-prior analysis PKL files in the target folder
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "reset_prior_analyzes_*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No reset_prior_analyzes_*.pkl files found in: {input_dir}")

    session_rates = []
    rows = []

    for pkl_path in pkl_files:
        print(f"\n--- Loading: {pkl_path}")

        # Load session-level PKL file
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Each PKL is expected to store a top-level session name and a list of trials
        trials = data.get("trials", [])
        session_name = data.get("session", Path(pkl_path).stem)

        # Compute per-session success rate
        session_rate, n_success, n_total = compute_session_success_rate(trials)
        session_rates.append(session_rate)
        rows.append(
            {
                "session": session_name,
                "file": os.path.basename(pkl_path),
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": session_rate,
            }
        )

        print(f"{session_name}: {n_success}/{n_total} = {100 * session_rate:.2f}%")

    # Compute average session success rate across all loaded files
    mean_session_success = float(np.nanmean(session_rates))
    print("\n=== Across-session summary ===")
    print(f"Average session success rate: {100 * mean_session_success:.2f}%")
    print(f"Number of sessions: {len(session_rates)}")


if __name__ == "__main__":
    main()
