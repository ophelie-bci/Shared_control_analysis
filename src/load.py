import os
import re
import pickle
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import collections
import numpy as np
import pandas as pd
import glob


class Trial:
    """Simple container for trial data."""
    def __init__(self, **entries):
        for k, v in entries.items():
            setattr(self, k, v)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

def AsBuiltin(obj, callback=None):
    """Convert numpy arrays and scalars to built-in Python types."""
    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 1:
            return [AsBuiltin(o, callback) for o in obj]
        return callback(obj) if callback else obj
    if isinstance(obj, dict):
        return {k: AsBuiltin(v, callback) for k, v in obj.items()}
    if hasattr(obj, 'item'):
        return obj.item()
    return callback(obj) if callback else obj

def LoadPkl(file, useNumPyTypes=False):
    """Load pickled trial data and optionally convert lists to numpy arrays."""
    with open(file, 'rb') as f:
        data = pickle.load(f)

    tnsData, tnsTrials = (data if isinstance(data, tuple) else (None, data))
    tnsTrials = [Trial(**trial) for trial in tnsTrials]

    if useNumPyTypes:
        for trial in tnsTrials:
            for field, value in trial.__dict__.items():
                if isinstance(value, list):
                    setattr(trial, field, np.array(value))

    return tnsTrials if tnsData is None else (tnsData, tnsTrials)

def LoadLastNTrials(pklFiles, n, answerNumbers=[1]):
    """Load the last N correct trials across a list of pickle files."""
    trials = []
    filesUsed = []
    for file in sorted(pklFiles, key=lambda x: TnsDateAndTime(x), reverse=True):
        data = LoadPkl(file)
        taskParameters, onlineTrials = (data if isinstance(data, tuple) else (None, data))
        for trial in reversed(onlineTrials):
            if trial.answer not in answerNumbers:
                continue
            trials.insert(0, trial)
            if len(trials) == n:
                return trials, taskParameters, filesUsed
        filesUsed.append(file)
    return trials, taskParameters, filesUsed

def TnsDateAndTime(pklFile):
    """Extract datetime string from a filename based on TnsId convention."""
    return TnsId(os.path.splitext(os.path.basename(pklFile))[0]).date + TnsId(os.path.splitext(os.path.basename(pklFile))[0]).time

class TnsId:
    """Parser for standardized experiment filenames (Tns format)."""
    def __init__(self, name):
        groups = re.match(self.Regex(), name).groups()
        if groups[1].startswith('dev'):
            self.subject, self.date, self.time, self.run = 'dev', '00010101', '0000', groups[1][3:]
        else:
            self.subject, self.date, self.time = re.match(r"(\w+?)_(\d{8})_(\d{4})", groups[1]).groups()
            self.run = groups[2][1:] if groups[2] else None
        self.config = groups[0][:-1] if groups[0] else None
        self.name = name

    @staticmethod
    def Regex():
        return r"(\w+_)?(\d*[a-zA-Z][a-zA-Z\d]*_\d+_\d+|dev\d+)(_\w+)?"

def load_data(data_path, take_answers=[1, 5, 3, 6, 9, 10, 4]):
    """Wrapper to load all trials with specific answer values."""
    trials, taskparameters, _ = LoadLastNTrials([data_path], 10000, answerNumbers=take_answers)
    return trials, taskparameters

# ---------- helpers ----------
def _root_name(p):
    return re.sub(r'_(?:ai)?trials\.pkl$', '', os.path.basename(p), flags=re.IGNORECASE)

def _trial_id(obj):
    # supports dicts and objects; tries 'trial', 'trial_id', 'trialId'
    if isinstance(obj, dict):
        for k in ("trial", "trial_id", "trialId"):
            if k in obj:
                try: return int(obj[k])
                except Exception: pass
    else:
        for k in ("trial", "trial_id", "trialId"):
            if hasattr(obj, k):
                try: return int(getattr(obj, k))
                except Exception: pass
    return None

def _attach_ai(base_trial, ai_trial):
    """Attach AI fields to base trial (non-destructive)."""
    if ai_trial is None:
        return base_trial
    # prefer attributes when possible; fallback to dict keys
    try:
        setattr(base_trial, "ai", ai_trial)
        if isinstance(ai_trial, dict):
            if "aiVelocities" in ai_trial: setattr(base_trial, "aiVelocities", ai_trial["aiVelocities"])
            if "aiControlOn" in ai_trial:  setattr(base_trial, "aiControlOn",  ai_trial["aiControlOn"])
            if "aiControlOff" in ai_trial: setattr(base_trial, "aiControlOff", ai_trial["aiControlOff"])
        return base_trial
    except Exception:
        pass
    if isinstance(base_trial, dict):
        base_trial["ai"] = ai_trial
        if isinstance(ai_trial, dict):
            if "aiVelocities" in ai_trial: base_trial["aiVelocities"] = ai_trial["aiVelocities"]
            if "aiControlOn" in ai_trial:  base_trial["aiControlOn"]  = ai_trial["aiControlOn"]
            if "aiControlOff" in ai_trial: base_trial["aiControlOff"] = ai_trial["aiControlOff"]
    return base_trial

import numpy as np

def is_incorrect_house_obstacle(trial, z_threshold=9.0):
    """
    House Obstacle correctness rule:
      - answer in {3,5,6,9,10}
      - OR (answer == 4 AND avatar z crosses threshold)
    """

    answer = getattr(trial, "answer", None)

    # Case 1: directly correct answers
    if answer in (3, 5, 6, 9, 10):
        return True

    # Case 2: special rule for answer == 4
    if answer == 4:
        traj = getattr(trial, "avatarTrajectory", None)
        if traj is None:
            return False

        z = traj.get("z", None)
        if z is None:
            return False

        z = np.asarray(z)
        return np.any(z > z_threshold)

    return False

# ---------- main ----------
def load_files(experimentName, monkey, base_dir, nb_trials=None, strict_pairing=False):
    """
   Loads base trials and AI trials, merges AI info into base trials by matching trial id.
   Returns:
     all_trials(enriched per session), all_correct, all_incorrect, all_training,
     all_channels, nb_channels, pkl_files, ai_trials_list (aligned to pkl_files)
    """
    data_dir = os.path.join(base_dir, monkey, experimentName)
    training_dir = os.path.join(data_dir, "trainingFiles")
    if not os.path.exists(data_dir):
        print(f"Skipping {monkey} for {experimentName} (directory not found).")
        return [], [], [], [], [], {'M1': 0.0, 'PMv': 0.0, 'PMd': 0.0}, [], []

    # --- discover base & AI files and pair them by filename root ---
    pkl_files = sorted([os.path.join(data_dir, f)
                        for f in os.listdir(data_dir)
                        if f.endswith("_trials.pkl")])
    ai_data_dir = os.path.join(base_dir, monkey, experimentName, "AIFiles")
    ai_candidates = glob.glob(os.path.join(ai_data_dir, "*_aitrials.pkl")) if os.path.isdir(ai_data_dir) else []
    ai_map = {_root_name(fp): fp for fp in ai_candidates}

    ai_pkl_files_ordered = []
    for base_fp in pkl_files:
        root = _root_name(base_fp)
        ai_fp = ai_map.get(root)
        if ai_fp is None:
            guess = os.path.join(ai_data_dir, os.path.basename(base_fp).replace("_trials.pkl", "_aitrials.pkl"))
            if os.path.isfile(guess):
                ai_fp = guess
        if ai_fp is None:
            msg = f"[warn] No matching AI file for {os.path.basename(base_fp)}"
            if strict_pairing:
                raise FileNotFoundError(msg)
            else:
                print(msg)
        ai_pkl_files_ordered.append(ai_fp)

    # --- load base trials (ordered), split correct/incorrect, and merge AI per session ---
    all_trials, all_correct, all_incorrect = [], [], []
    ai_trials_list = []
    for base_fp, ai_fp in zip(pkl_files, ai_pkl_files_ordered):
        # base trials
        trials = load_data(base_fp)[0]   # your existing loader
        if nb_trials is not None:
            trials = trials[:nb_trials]

        correct = [t for t in trials if getattr(t, "answer", None) == 1]
        if experimentName in ("House","House Obstacle"):
            incorrect = [t for t in trials if is_incorrect_house_obstacle(t)]
        # elif experimentName == "House":
        #     incorrect = [t for t in trials if getattr(t, "answer", None) in (5, 3, 6, 9, 10)]
        else:
            incorrect = [t for t in trials if getattr(t, "answer", None) in (3, 5, 6)]

        # AI trials for this session
        if ai_fp:
            with open(ai_fp, "rb") as f:
                obj = pickle.load(f)
            ai_trials = obj[1] if isinstance(obj, (list, tuple)) and len(obj) >= 2 else (obj if isinstance(obj, list) else [])
        else:
            ai_trials = []

        # index AI trials by trial id
        ai_by_idx = {}
        for ai_tr in ai_trials:
            tid = _trial_id(ai_tr)
            if tid is not None:
                ai_by_idx[tid] = ai_tr

        # merge AI info into each base trial
        for bt in trials:
            tid = _trial_id(bt)
            ai_match = ai_by_idx.get(tid)
            _attach_ai(bt, ai_match)

        # collect
        all_trials.append(trials)
        all_correct.append(correct)
        all_incorrect.append(incorrect)
        ai_trials_list.append(ai_trials)

        # small log
        n_merged = sum(1 for bt in trials if _trial_id(bt) in ai_by_idx)
        # print(f"[merge] {os.path.basename(base_fp)} ↔ {os.path.basename(ai_fp) if ai_fp else 'MISSING'} | "
            #   f"trials={len(trials)} | ai={len(ai_trials)} | merged={n_merged}")

    # --- training files (unchanged) ---
    training_files = [os.path.join(training_dir, f)
                      for f in os.listdir(training_dir)] if os.path.isdir(training_dir) else []
    all_training = [load_data(f, take_answers=[1])[0] for f in training_files if f.endswith(".pkl")]

    # --- channel counts / areas (your original logic) ---
    ccf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.ccf')]
    all_channels, area_counts = [], []
    for file in ccf_files:
        root = ET.fromstring(Path(file).read_text())
        elecs = {
            e.find("label").text.strip(): e.find(".//spike/threshold/level").text.strip()
            for e in root.findall(".//ChanInfo_item")
        }
        filtered = [k for k, v in elecs.items() if k.startswith('elec') and (int(v) / 4) > -60]
        all_channels.append(len(filtered))

        ch_map = {'M1': 0, 'PMv': 0, 'PMd': 0}
        for ch in filtered:
            num = int(''.join(filter(str.isdigit, ch)))
            if monkey == "Monkey 1":
                if 65 <= num <= 96 or 129 <= num <= 192: ch_map['M1'] += 1
                elif 1 <= num <= 64 or 97 <= num <= 128: ch_map['PMv'] += 1
                elif 193 <= num <= 297:                 ch_map['PMd'] += 1
            elif monkey == "Monkey 3":
                if 1 <= num <= 64 or 97 <= num <= 128:  ch_map['M1'] += 1
                elif 65 <= num <= 96 or 129 <= num <= 192: ch_map['PMv'] += 1
                elif 193 <= num <= 297:                 ch_map['PMd'] += 1
            else:
                ch_map['M1'] += 1
        area_counts.append([ch_map['M1'], ch_map['PMv'], ch_map['PMd']])

    if len(area_counts) and len(all_channels):
        perc = np.mean([[c / t for c in count] for count, t in zip(area_counts, all_channels)], axis=0)
        nb_channels = {'M1': perc[0], 'PMv': perc[1], 'PMd': perc[2]}
    else:
        nb_channels = {'M1': 0.0, 'PMv': 0.0, 'PMd': 0.0}

    # aligned returns:
    #   - pkl_files[i]           ↔ all_trials[i], all_correct[i], all_incorrect[i]
    #   - ai_pkl_files_ordered[i]↔ ai_trials_list[i]
    #   - all_trials are enriched with AI fields when matched
    return all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, pkl_files, ai_trials_list
