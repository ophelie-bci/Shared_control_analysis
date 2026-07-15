*This readme file was generated on 2026-03-06 by <Ophelie Saussus>*

# [Dataset for: Stabilization–Responsiveness Trade-offs in Continuous Shared-Control for Invasive Brain–Computer Interfaces]

Dataset DOI: [https://doi.org/10.48804/7KGSQS](https://doi.org/10.48804/7KGSQS)

## General Information

- Dataset title: Stabilization–Responsiveness Trade-offs in Continuous Shared-Control for Invasive Brain–Computer Interfaces
- Description:	
This dataset contains neural and behavioral recordings from two rhesus macaques performing intracortical brain–computer interface (BCI) navigation tasks in a virtual environment. Neural activity was recorded from three 96-channel Utah arrays implanted in ventral premotor cortex (PMv), dorsal premotor cortex (PMd), and primary motor cortex (M1), while animals controlled a virtual sphere using decoded neural activity in closed loop. The dataset includes trial-level behavioral trajectories, decoded velocity commands, and neural spike-rate features used to train and evaluate a preferential subspace identification (PSID) neural decoder and a probabilistic shared-control controller (RT-V2). Experiments were conducted across multiple sessions and tasks designed to probe stabilization–responsiveness trade-offs in neural control, including Fixed Obstacle, Appearing Obstacle, and Respawn navigation tasks. Each session contains time-aligned neural activity (50 ms bins), decoded velocity signals, executed trajectories, task event timestamps, and trial metadata. These data enable reproduction of the behavioral analyses and figures reported in the associated manuscript and provide a benchmark dataset for studying shared autonomy and adaptive control in high-bandwidth invasive BCIs.
- Authors: Ophelie Saussus1, Pinhao Song2, Sofie De Schrijver1,3, Irene Caprara1, Thomas Decramer4, Renaud Detry2,5, Peter Janssen1* Affiliations:
1 Laboratory for Neuro- and Psychophysiology, Department of Neurosciences, KU Leuven and the Leuven Brain Institute, Leuven, Belgium.
2 KU Leuven, Dept. Mechanical Engineering, Research unit Robotics, Automation and Mechatronics.
3 Department of Electrical & Computer Engineering, University of Washington, Seattle, WA, USA.
4 Research group Experimental Neurosurgery and Neuroanatomy, KU Leuven and the Leuven Brain Institute, Leuven, Belgium.
5 KU Leuven, Dept. Electrical Engineering, Research unit Processing Speech and Images.
* Corresponding author: Peter Janssen, peter.janssen@kuleuven.be
- Contributors: Data collectors: Sofie De Schrijver & Irene Caprara, Supervisors: Peter Janssen & Renaud Detry, Researchers: Ophelie Saussus & Pinhao Song, Related person: Thomas Decramer
- Funding information: Fonds Wetenschappelijk onderzoek (FWO) grant G.097422N 
KU Leuven grant C14/18/100 
KU Leuven grant C14/22/134

## Description of the data and file structure

### Files and variables

#### File: AI_paper_data.zip

A compressed archive containing all session data organized by subject (“Monkey 1/2”) and task.


#### Folder organization inside `AI_paper_data.zip`

```
AI_paper_data/
  Monkey 1/   (Maui)
  Monkey 2/   (Loki)
```

Within each Monkey folder, task folders are present only for tasks performed by that animal:

* Monkey 1 (Maui): performed 4 tasks.
* Monkey 2 (Loki): performed 2 tasks (fewer task folders).

``
AI_paper_data/
  Monkey 1/   (Maui)
    AI Obstacle/
    AI Appearing Obstacle/
    AI Appearing Obstacle 2/
    AI Respawn/
  Monkey 2/   (Loki)
    AI Obstacle/
    AI Appearing Obstacle/
```
##### Structure inside each task folder

Each task folder has the following structure:

```
Task_Name/
  AiFiles/
  trainingFiles/
     navtraining_<Subject>_<YYYYMMDD>_<HHMM>_<SessionLetter>_trials.pkl
     <Subject>_<YYYYMMDD>_<HHMM>_<SessionLetter>.ccf
  navdecoding_<Subject>_<YYYYMMDD>_<HHMM>_<SessionLetter>_trials.pkl
  <Subject>_<YYYYMMDD>_<HHMM>_<SessionLetter>.ccf
```

For Monkey 1/AI Respawn/, the following additional folder is present:
```
AI Respawn/
  resetPriorFiles/
     reset_prior_analyzes_navdecodingsphereairespawn_<Subject>_<YYYYMMDD>_<HHMM>_<SessionLetter>.pkl
```

##### AiFiles/
Contains information of the shared-control AI module recorded during each session.

*_aitrials.pkl — trials with AI information

##### trainingFiles/

Contains files required for decoder training:

```
trainingFiles/
  navtraining*_trials.pkl — training trial data (passive observation phase)
  *.ccf channel configuration file (spike thresholds) used during that session
```

##### Decoding files

Located directly inside the task folder:

* navdecoding*_trials.pkl — closed-loop BCI trial data
* *.ccf channel configuration file (spike thresholds) used during that session

There are multiple navdecoding*.pkl, navtraining*.pkl,  and .ccf files within each task folder — one set per recording session.

##### resetPriorFiles/ (AI Respawn only)

Contains offline replay outputs for the Respawn task, generated from session log files to evaluate the effect of resetting the AI temporal prior at target respawn.

```
resetPriorFiles/
  reset_prior_analyzes_navdecodingsphereairespawn_*.pkl — offline replay trial data for reset-prior analyses
```

#### Tasks and corresponding filename prefixes

Each session typically includes:

* a training file (navtraining…) used to train the decoder from passive observation
* a decoding file (navdecoding…) containing closed-loop online BCI trials

Task names used in the paper correspond to these filename prefixes:

| Task (paper)                                     | Decoding file prefix                      | Training file prefix |
| :----------------------------------------------- | :---------------------------------------- | :------------------- |
| AI Obstacle                                      | `navdecodingsphereaiobstacle_`           | `navtrainingsphere_` |
| AI Appearing Obstacle                            | `navdecodingsphereaiappearingobstacle_`  | `navtrainingsphere_` |
| AI Appearing Obstacle 2                          | `navdecodingsphereaiappearingobstacle_`  | `navtrainingsphere_` |
| AI Respawn                                       | `navdecodingsphereairespawn_`             | `navtrainingsphere_` |

#### File naming convention (recommended for reuse)

Most data files follow:

`<prefix>_<Subject>_<YYYYMMDD>_<HHMM>_<SessionLetter>_<suffix>.pkl`

Examples:

* navdecodingsphereaiappearingobstacle_Maui_20241108_1021_A_trials.pkl
* navtrainingsphere_Maui_20241108_0950_A_trials.pkl

Where:

* Subject is one of: Loki, Maui
* YYYYMMDD is the recording date (YearMonthDay)
* HHMM is local start time in 24h format (HourMinute)
* SessionLetter is a session label (A, B, C, …) used internally during acquisition
* suffix indicates content (see below)

#### File types included

**(1) Trial data (.pkl)**

* *_trials.pkl : training or decoding trial structures (core dataset)
* *_aitrials.pkl: trial-aligned shared-control controller variables, including decoded input velocity, AI-adjusted output velocity, entropy measures, and AI control interval timestamps
* reset_prior_analyzes_*.pkl: offline replay outputs for reset-prior analyses in the Respawn task

**(2) Spike threshold configuration (.ccf)**

* *.ccf: spike detection thresholds and electrode metadata used during that session

#### Number of sessions per task and monkey

The number of recording sessions (i.e., number of navdecoding*_trials.pkl files) per task and per animal is:

* AI Obstacle task
  * Monkey 1 (Maui): 12 sessions
  * Monkey 2 (Loki): 11 sessions
* AI Appearing Obstacle task
  * Monkey 1 (Maui): 10 sessions
  * Monkey 2 (Loki): 9 sessions
* AI Appearing Obstacle 2 task
  * Monkey 1 (Maui): 9 sessions
* AI Respawn task
  * Monkey 1 (Maui): 9 sessions

Each decoding session corresponds to one navdecoding*_trials.pkl file and one associated .ccf file.\
Training sessions are stored separately as navtraining*_trials.pkl files. A single training session may be used to train a decoder that is applied to multiple decoding sessions.

This archive contains 60 decoding files, 60 training files, 60 AI trials, 9 reset-prior replay files, and 60 .ccf files. The same .ccf file is stored inside trainingFiles/ and duplicated in the task root directory for convenience.

### Usage notes: how to open and view the files

#### Opening .pkl files (Python)

All .pkl files were generated in Python and can be loaded using:

```
import pickle

with open("filename.pkl", "rb") as f:
    obj = pickle.load(f)
```

The loaded object is typically:

* a **list of trials**, or
* a **dictionary** containing per-trial arrays and metadata (offline analysis).

If you prefer a safer/modern loader, **joblib** may also work depending on how the file was saved, but **pickle** is the default.

#### Viewing .ccf files

.ccf files are XML-based channel configuration files exported by the Blackrock/Cerebus acquisition system. They contain per-channel metadata used during that recording session, including electrode labels, acquisition settings, and spike detection threshold levels.

They can be:

* opened in any text editor (e.g., VS Code, Notepad++, etc.), or
* parsed programmatically using an XML parser in Python (e.g., xml.etree.ElementTree).

Each .ccf file includes:

* Electrode/channel labels (e.g., elec1)
* Channel and bank identifiers
* Analog-to-digital scaling parameters (units typically µV)
* Hardware filter settings
* Spike detection threshold values (stored as integer level codes in the XML)

Key fields relevant for this dataset are described below under “Variables”.

### Variables and data dictionaries (with units)

#### A) Training and decoding *_trials.pkl files

Each *_trials.pkl file is a 2-tuple:

* session_config (dict): task/session parameters (e.g., task, threeDimensions, targetWindowRadius, timeToStayInTargetWindow, dataDirectory, isTraining, …)
* trials *(list of dicts)*: one dictionary per trial

Below describes the per-trial dictionary keys (exact presence depends on task).

##### Session configuration (session_config)

Per-session task parameters exported from Unity. Keys may vary by task/version.

* task (str): task variant (e.g., movingCamera, fixedCamera). "movingCamera" is for the Continuous Navigation and Continuous Navigation (first person perspective) task, the rest is labeled "fixedCamera".
* threeDimensions (bool): if True, movement/control is 3D; if False, movement is constrained (typically planar) even if Unity stores 3D coordinates.
* targetWindowRadius (float, Unity units): radius of the target acceptance region used for success detection.
* timeToStayInTargetWindow (int, ms): dwell time required inside the target window to count as target acquisition.
* curvedPath (bool): whether the task enforces a curved/trajectory constraint during the training (task-specific).
* isTraining (bool): whether the session/trials were run in training mode (as set in Unity).

Logged for reproducibility:

* dataDirectory (str): acquisition path on the recording computer.
* useDebugLogLevel (bool): Unity logging verbosity.

##### Per-trial dictionary

###### 1) Trial metadata

* trial (int): trial index.
* start (float, ms): trial start time relative to the start of the recording file.
* stop (float, ms): trial end time relative to the start of the recording file.
* answer (int): trial outcome code.
  * 1 = correct
  * 3 = incorrect: target went off-screen
  * 5 = incorrect: target not reached within max trial time
  * 6 = incorrect: sphere/avatar went off-screen
  * other values = aborted trials

###### 2) Target / task state

* targetPosition (np.array, shape (3,), Unity units): target position (x, y, z) in the local task coordinate frame, expressed relative to the avatar’s start position at (0, 0, 0) at the beginning of the trial.
* targetOnset (float, ms): time the target appeared.
* photoEvents (np.array, ms): timestamps when the target was visible on screen (photodiode / screen-sync events).

Task-dependent fields:

* targetJumpPosition (np.array, shape (3,), Unity units): respawn/jump target position (Respawn task only; otherwise typically [nan, nan, nan]).
* unityTargetJumpTime (float, ms; often nan): target jump/respawn time recorded from Unity (Respawn task only).
* unityTargetPosition (np.array of dict, dtype=object; Unity units): target position as logged by Unity in absolute world coordinates (often a 1-element array containing {'x','y','z'}).
* obstaclePosition (np.array, shape (3,), Unity units): obstacle position (Obstacle task only; otherwise typically [nan, nan, nan]).

###### 3) Behavior

* avatarTrajectory (dict): sphere/avatar position time series
  * time (np.array, ms)
  * x, y, z (np.array, Unity units) in the same local coordinate frame as targetPosition
  * may include avatarRotation (np.array, shape (T,), degrees): avatar yaw angle around the vertical axis (Unity Euler Y), sampled at the same timestamps as x,y,z.
* avatarVelocity (dict): sphere/avatar velocity time series
  * time (np.array, ms)
  * vx, vy, vz (np.array, Unity units/s)

###### 4) Neural data

* muaA *(dict)*: multiunit threshold crossings per electrode/channel
  * keys like elec1, elec2, …
  * each value is an array of spike timestamps *(ms, relative to recording start)*
* contA *(dict)*: local field potential (LFP) signal per electrode.
  * each channel stores:
    * samplingRate *(float, Hz; typically 1000.0)*
    * data *(np.array,* µV)*: Continuous LFP signal

Optional / task- or session-dependent:

* muaB, contB: same structure as above for a second bank/array.

###### Units note (Unity)

Unity uses arbitrary Unity units for spatial quantities. In this dataset:

* positions are stored in Unity units
* velocities are stored in Unity units/s
* timestamps in trial dicts and behavior time series are in ms, all relative to the start of the recording file.

All timestamps in *_trials.pkl (trial events, behavior samples, and neural event times) are expressed in milliseconds relative to the start of the recording file, enabling direct alignment. Continuous signals (LFP, EMG, eye) are aligned to trials using their samplingRate and the trial start/stop times. Spike times and event timestamps are stored in milliseconds relative to recording start. While some logged Unity fields (e.g., unityTargetPosition) are spatial metadata rather than time-synchronized traces.

#### B) .ccf files (spike threshold configuration)

Files are XML-based Blackrock Cerebus Channel Configuration Files (CCF, Version 12) containing per-channel acquisition parameters used during that recording session.

Each file includes:

* Electrode/channel labels (e.g., elec1)
* Channel ID and bank information
* Analog-to-digital scaling parameters (including units, typically µV)
* Hardware filter settings
* Spike detection threshold levels
* Sampling and display parameters

Spike thresholds are stored in the acquisition system’s digital units, together with scaling parameters that define their relationship to physical units (µV).

These files can be opened with any text editor.

#### C) AI trial files (*_aitrials.pkl)

Files ending in *_aitrials.pkl contain trial-wise shared-control variables extracted from the AI runtime stream and aligned to the behavioral trial structure. These files are intended to provide a structured representation of AI-controller activity during closed-loop sessions, without requiring parsing of raw log files.

Each *_aitrials.pkl file is a list of dictionaries, with one dictionary per trial.

##### Per-trial fields

* trial (int): trial index within the session.
* start (float, ms): trial start time relative to the start of the recording file.
* stop (float, ms): trial stop time relative to the start of the recording file.
* answer (int): trial outcome code, using the same coding as in *_trials.pkl.
* aiControlOn (float array or NaN, ms): timestamp(s) at which the AI controller entered an explicit control block during the trial. If no such block occurred, the value is NaN.
* aiControlOff (float array or NaN, ms): timestamp(s) at which the AI controller exited an explicit control block during the trial. If no such block occurred, the value is NaN.
* aiVelocities (list): time-resolved sequence of AI/controller updates during the trial. Each element corresponds to one controller update step.
* multipleAiControlBlocks (bool): indicates whether more than one explicit AI control block occurred during the trial.
* targetPosition (array-like, length 3, Unity units, or None): cued target position for the trial.
* obstaclePosition (array-like, length 3, Unity units, or None): obstacle position for the trial.

##### Structure of aiVelocities

Each element of aiVelocities is a dictionary with the following fields:

* Input (array-like, length 3): decoded velocity command provided as input to the AI controller at that update step.
* InputTimestamp (float array, ms): timestamp of the controller input.
* Output (tuple, length 3): velocity command output by the AI controller after arbitration.
* OutputTimestamp (float array, ms): timestamp of the controller output.
* EntropyUb (float or NaN): upper-bound entropy measure of the AI action distribution at that update step.
* EntropyLb (float or NaN): lower-bound entropy measure of the AI action distribution at that update step.
* Latency (float or NaN, ms): controller processing latency, when available.

##### Interpretation

Input corresponds to the decoded BCI velocity before AI arbitration, whereas Output corresponds to the velocity command returned by the shared-control controller and sent for execution. EntropyUb and EntropyLb quantify uncertainty in the controller’s action distribution and can be used to derive confidence-related arbitration measures, as described in the associated manuscript. Early samples in a trial may contain NaN entropy values, reflecting initialization or periods before the AI prior became fully defined.

##### Notes

Trials without active shared-control may contain an empty aiVelocities list.

aiControlOn and aiControlOff mark explicit AI control intervals; these may be absent even when aiVelocities are present, depending on the trial type and controller state.

Timestamps are in milliseconds relative to the same session clock used in the corresponding behavioral trial files, allowing direct alignment with task events and trajectories.

#### D) Reset-prior replay files (reset_prior_analyzes_*.pkl)

Files ending in reset_prior_analyzes_*.pkl contain trial-wise offline replay data generated for the AI Respawn task. These files were produced by re-running the AI arbitration offline while resetting the temporal prior at target respawn, allowing evaluation of the effect of prior carry-over on post-respawn behavior.

These files are located in:

AI_paper_data/
  Monkey 1/
    AI Respawn/
      resetPriorFiles/

Each reset_prior_analyzes_*.pkl file corresponds to a single Respawn session and contains replayed trial data aligned to the original session timing.

Each file is a dictionary with the following structure:

{
  "session": <session_name>,
  "trials": [ {trial}, {trial}, ... ]
}

###### Per-trial fields
* trial_id (int): trial index
* directory (str or None): session directory label parsed from the log
* true_goal (dict): original cued goal position with keys x, y, z
* start_time, end_time (str): ISO-formatted timestamps
* n_samples (dict): number of valid aligned samples per stream
* samples (dict): aligned trial time series including:
      * position (N x 3 array): sphere position
      * entropy (N array): AI entropy trace
      * ai_velocity (N x 2 array): AI velocity in XZ
      * bci_velocity (N x 2 array): decoded BCI velocity in XZ
* target_jumps (list): all parsed respawn target positions
* target_jump_position (dict or None): main respawn target position for the trial
* target_jump_index (int or None): iteration index of the target jump
* reset_iters (list): reset events recorded in the log
* aivelocity_factor (int or None): AI velocity blending factor from the log
* answer_log (int or None): original logged trial outcome
* answer (int): re-evaluated trial outcome after offline replay
* answer_reason (str): reason for success/failure assignment

##### Notes
These files are intended for offline replay and boundary-condition analyses of the Respawn task and are not present for the other tasks.

## Missing data, abbreviations, and notes

* Missing data codes: Missing values are typically stored as NaN (e.g., fields not applicable to a task).
* Some modalities are session-dependent:
  * EMG and eye tracking are not available for all animals/sessions.
* Abbreviations:
  * mua= multiunit activity (threshold crossings)
  * fpp = first-person perspective

## Code/Software

* Data files were generated and analyzed using Python.
* To load .pkl files, use Python’s pickle module (example above).
* Unity tasks were implemented in Unity 2021.3.x (used during data collection).
* Related code to reproduce offline analyses and figures: Zenodo [XX](XX)

## Access information

Other publicly accessible locations of the data:

* None

Data was derived from the following sources:

* N/A

## Methodological information

- Data collection & processing methods: Neural and behavioral data were collected from two adult rhesus macaques implanted with three 96-channel Utah microelectrode arrays targeting primary motor cortex (M1), ventral premotor cortex (PMv), and dorsal premotor cortex (PMd). Neural signals were recorded using a Blackrock Cerebus acquisition system with Cereplex M headstages. Signals were filtered and spike events were extracted using manually defined thresholds at the beginning of each recording session.

During experiments, animals controlled a virtual sphere in a three-dimensional Unity environment using an intracortical brain–computer interface (BCI). Neural activity was decoded in real time into velocity commands using a decoder trained during a passive observation phase. The decoder was based on the Preferential Subspace Identification (PSID) framework and produced continuous velocity updates every 50 ms.

During online control, decoded velocities either directly controlled the virtual sphere (BCI-only condition) or were processed by a shared-control AI module that adjusted commands based on environmental context and inferred trajectory intent. Behavioral and neural signals were logged during both training and online decoding phases and exported for offline analysis.

- Instrument-specific information:Neural activity was recorded using a Blackrock Neurotech Cerebus acquisition system with Cereplex M headstages. Each subject was implanted with three 96-channel Utah arrays (Blackrock Neurotech) with electrode lengths of 1–1.5 mm and 400 µm spacing. Neural signals were sampled at 30 kHz and filtered to extract multiunit activity using threshold crossings.

Behavioral experiments were conducted in a Unity-based virtual environment presented on a ViewPixx 3D display (1920 × 1080 resolution, 120 Hz refresh rate). Animals viewed the display through synchronized shutter glasses providing stereoscopic depth perception.
- Quality assurance procedures:Spike detection thresholds were manually verified at the start of each session to ensure reliable multiunit detection across electrodes. Behavioral event timing and neural timestamps were synchronized through the acquisition system to allow alignment of neural activity with task events and movement trajectories.

Trial logs and task parameters were automatically recorded during experiments to ensure reproducibility. Data files were inspected offline to verify correct synchronization of behavioral trajectories, neural signals, and task events before inclusion in the dataset.
- Software:Data collection and task control were implemented using:

Unity 2021.3 LTS (virtual environment and behavioral task control)

Blackrock Central / Cerebus software (neural acquisition)

Data processing and analysis were performed using Python. Custom scripts used for offline analysis and figure generation will be made publicly available in the associated code repository upon publication.
- Ethical approval: All experimental procedures involving animals were approved by the KU Leuven Ethical Committee for Animal Experimentation and complied with the European Directive 2010/63/EU on the protection of animals used for scientific purposes.

## Access and sharing information

- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Restrictions: None. Data are released under CC0 1.0.
- Dataset URL: The dataset is available at: https://doi.org/10.48804/7KGSQS. The dataset will be publicly accessible at this location upon publication of the associated article.
- Related materials:
  - Associated manuscript: Saussus O., Song P., De Schrijver S., Caprara I., Decramer T., Detry R., Janssen P. Stabilization–Responsiveness Trade-offs in Continuous Shared-Control for Invasive Brain–Computer Interfaces. Under review.
  - Related dataset: Saussus O., De Schrijver S., Ramirez J.G., Decramer T., Janssen P. Intracortical brain–computer interface for navigation in virtual reality in macaque monkeys. Science Advances (accepted).

