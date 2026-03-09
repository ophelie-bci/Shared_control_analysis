import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
import glob
import pickle
import xml.etree.ElementTree as ET
from omegaconf import OmegaConf
import logging
import time

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from bcidecode.preprocessing.data import load_data
from bcidecode.preprocessing.ratesTransformer import (
    EpochTransformer, RatesTransformer
)
from bcidecode.online.models import Model
from bcidecode.optimization.axUtils import load_best_config
from bcidecode.modeling.pipeline_builder import PipelineBuilder
from bcidecode.modeling.defaults import (
    FILTERING_SCORERS, PREPROCESSORS, REGRESSORS
)
from bcidecode.kalman.filters import (
    PSID_Decoder, PSID_DecoderPositions, KalmanRegressor, PSID_DecoderVelocities
)
from onlinedecoding import localconfig, tasks, util
from tnsbmi import bintrials, dataconversion, nevdata, modeling
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.load import load_files

def train_model(dest_dir, trials, cfg, opt_parameters, bin_params, available_steps, available_scorers, start, states_data, bin_data, targets):
    ## DEFINE MODEL
    pipeline_builder = PipelineBuilder(available_steps, available_scorers)
    preprocessor = pipeline_builder.fit(
        dict(preprocessor=cfg.pipeline.pipeline_arch.preprocessor)) # tuning_filter, concat_transformer
    regressor = pipeline_builder.fit(dict(regressor=cfg.pipeline.pipeline_arch.regressor)) # nystroem_kernel (sklearn), ridge (sklearn)
    model = Pipeline(
        steps=[("decoder", PSID_DecoderVelocities(
            regressor=regressor, preprocessor=preprocessor, bin_size=bin_params["bin_size"]))]
    )
    
    ## FIT MODEL
    model.set_params(**opt_parameters)
    model = model.fit(bin_data, states_data)

    # Elapsed time for fitting & nb of channels used for fitting
    elapsed = time.time() - start
    time_per_trial = elapsed/len(trials)
    # k_channels = model["decoder"].kinematics_filter.k_ #nb of channels used for decoding
    k_channels = len(bin_data[0])
    logging.info(f"Selected {k_channels} channels") 
    logging.info(f"time for training: {elapsed}, time per trial {time_per_trial}")
    start_position = {"x": 0, "y": 0, "z": 0}
    withLfp = True
    with open(os.path.join(dest_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(dest_dir, "bin_params.pkl"), "wb") as f:
        pickle.dump(bin_params, f)
        config_dict = dict(
        modelCenterCue= start_position,
        modelAxes=["x", "y","z"],
        modelSmoothing=False,
        modelRetrainPeriod=0, #16
        modelCovScaling=0.2,
        modelTaskType="navigation", #cursor
        modelRmax=4,
        withLfp = withLfp,
    )
    config = localconfig.Configuration(**config_dict)
    with open(os.path.join(dest_dir,"config.pkl"), "wb") as f:
        pickle.dump(config, f)

def train_different_models(training_file, neural_file):
    # Functions
    # # Function to extract electrode number
    # def extract_electrode_number(elec_label):
    #     return int(elec_label.replace('elec', ''))

    # # Electrode selection
    # base_dir = os.path.dirname(os.path.dirname(training_file))  # Go up one level from 'trainingFiles'
    # load_path = os.path.join(base_dir, "Electrode_selection", "electrode_selection.pkl")
    # with open(load_path, 'rb') as f:
    #     electrode_selection = pickle.load(f)
    neural_directory = os.path.dirname(neural_file)
    ccf_file = [file for file in os.listdir(neural_directory) if file.endswith(".ccf")]
    ccf_file_path = os.path.join(neural_directory, ccf_file[0])
    ccf_file_path = os.path.normpath(ccf_file_path)
    # Only keep channels with a threshold higher than -60mV
    with open(ccf_file_path, 'rb') as f: 
        data = f.read()
    root = ET.fromstring(data)  # Parse the XML content
    elec_info = {}  # Initialize a dictionary to store the label and threshold level for each elec
    for chan_info in root.findall(".//ChanInfo_item"):  # Find all the ChanInfo_item elements
        label = chan_info.find("label").text.strip()
        spike_threshold = chan_info.find(".//spike/threshold/level").text.strip()
        elec_info[label] = spike_threshold
    filtered_data = {key: value for key, value in elec_info.items() if key.startswith('elec')}
    
    # Extract the session identifier from the filename
    basename = os.path.basename(neural_file)  # 'navdecoding_Maui_20240422_0935_A_trials.pkl'
    session_id = basename.replace("_trials.pkl", "")  # 'Maui_20240422_0935_A'
    # Retrieve electrode selection for this session
    selected_electrodes = filtered_data
     
    #Load data
    trials, taskparameters = load_data(neural_file)
    targets = [np.array([trial.targetPosition[0], trial.targetPosition[1], trial.targetPosition[2]]) for trial in trials]
    neural_directory = os.path.dirname(neural_file)
    ccf_file = [file for file in os.listdir(neural_directory) if file.endswith(".ccf")]
    ccf_file_path = os.path.join(neural_directory, ccf_file[0])
    ccf_file_path = os.path.normpath(ccf_file_path)
    optimization_dir = r"L:\GBW-0039_PJanssen_Backup\0002_Lab\Ophelie_backup\Decoding algorithm\OptimizationResults"
    ## SELECT CHANNELS
    for trial in trials:
        keys = [k for k,v in trial.muaA.items() if 'elec' not in k]
        for key in keys:
            del trial.muaA[key]
    muas = ["muaA"]
    mua_channels = [len(trials[0].muaA)]
    channels = [
        [nevdata.ChannelLabel(i, False) for i in range(1, n_chs + 1)] for n_chs in mua_channels
    ]
    # Only keep channels with a threshold higher than -60mV
    with open(ccf_file_path, 'rb') as f: 
        data = f.read()
    root = ET.fromstring(data)  # Parse the XML content
    elec_info = {}  # Initialize a dictionary to store the label and threshold level for each elec
    for chan_info in root.findall(".//ChanInfo_item"):  # Find all the ChanInfo_item elements
        label = chan_info.find("label").text.strip()
        spike_threshold = chan_info.find(".//spike/threshold/level").text.strip()
        elec_info[label] = spike_threshold
    filtered_data = {key: value for key, value in elec_info.items() if key.startswith('elec')}
    spike_threshold = -60
    all_channels = [[electrode for electrode, value in filtered_data.items()
                    if electrode in filtered_data and (int(value) / 4) > spike_threshold]]
    channels = bintrials.CerebiLabels(*all_channels) 
    # # Group by cortical region
    # M1_channels = [[elec for elec in all_channels[0] if extract_electrode_number(elec) in M1_range_1 or extract_electrode_number(elec) in M1_range_2]]
    # PMv_channels = [[elec for elec in all_channels[0] if extract_electrode_number(elec) in PMv_range_1 or extract_electrode_number(elec) in PMv_range_2]]
    # PMd_channels = [[elec for elec in all_channels[0] if extract_electrode_number(elec) in PMd_range]]
    # Create different training conditions
    channels_conditions = {
        "Online_simulation": channels,
        "All": selected_electrodes['M1'] + selected_electrodes['PMv'] + selected_electrodes['PMd'],
        "M1": selected_electrodes['M1'],
        "PMv": selected_electrodes['PMv'],
        "PMd": selected_electrodes['PMd'],
        "PMv+PMd": selected_electrodes['PMv'] + selected_electrodes['PMd'],
        "M1+PMd": selected_electrodes['M1'] + selected_electrodes['PMd'],
        "M1+PMv": selected_electrodes['M1'] + selected_electrodes['PMv'],
    }

    # Loop over each condition and train a model
    for label, selected_channels in channels_conditions.items():
        # Create directory for this condition
        # Make new dest_dir
        parts = os.path.basename(training_file).split('_')
        date_str = parts[2]      # Extracts date
        dest_dir = os.path.join(os.path.dirname(training_file), date_str)
        condition_dir = os.path.join(dest_dir, label)
        if os.path.exists(condition_dir):
            print(f"Skipping {label} — already exists.")
            continue  # Skip training if directory exists
        os.makedirs(condition_dir, exist_ok=True)

        # Convert to CerebiLabels format
        # channels = bintrials.CerebiLabels(*selected_channels)
        channels = selected_channels
        channels_lfp = []

        cfg_path = os.path.join(optimization_dir, ".hydra", "config.yaml")
        cfg = OmegaConf.load(cfg_path)
        opt_parameters = load_best_config(
        optimization_dir, cfg.obj_name, cfg.minimize)
        opt_parameters['decoder__preprocessor__preprocessor__tuning_filter__k'] = 258
        opt_parameters["decoder__n1"] = 6
        opt_parameters["decoder__tuning_threshold"] = 0
        opt_parameters["decoder__nx"] = opt_parameters["decoder__n1"] 
        neural_bin_size = opt_parameters['decoder__bin_size']
        frequencyBand = [100,200]
        lfpLength = 300
        samplingRate = 1000
        combined = False
        withLfp = False
        pipeline_arch = {
            "preprocessing": ["rates_comp", "tuning_filter", "concat_transformer", "power_transformer"],
        }
        pipeline_params = {
        "preprocessing__rates_comp__t_int": 7000,
        "preprocessing__rates_comp__t_skip": 0,
        "preprocessing__rates_comp__event": "GoCue",
        "preprocessing__rates_comp__stop_flag": "stop",
        "preprocessing__rates_comp__channels": channels,
        "preprocessing__rates_comp__channels_lfp": channels_lfp,
        "preprocessing__rates_comp__bin_size": neural_bin_size,
        "preprocessing__rates_comp__withLfp": withLfp,
        "preprocessing__rates_comp__lfpLength": lfpLength,
        "preprocessing__rates_comp__combined": combined,
        "preprocessing__rates_comp__samplingRate": samplingRate,
        "preprocessing__rates_comp__frequencyBand": frequencyBand,
        }
        bin_params = {
            key.split("__")[-1]: value
            for key, value in pipeline_params.items()
            if "rates_comp" in key
        }
        # Setting model catalogue
        available_steps = {**PREPROCESSORS, **REGRESSORS}
        available_scorers = FILTERING_SCORERS #anova, mi, anovareg
        # Define pipeline
        pipeline_builder = PipelineBuilder(available_steps, available_scorers) #store available steps and available scorers in PipelineBuilder
        pipeline = pipeline_builder.fit(pipeline_arch) #set up pipeline with the different steps of pipeline_arch -> preprocessing: rates_comp, tuning_filter, concat_transformer, power_transformer
        # Start time
        start = time.time() # Current time (s) -> compute time needed to build model

        ## EPOCH DATA IN 50MS BINS
        epocher = EpochTransformer(**bin_params) # EpochTransformer with bin parameters
        data = epocher.fit_transform(trials, task) #x, vx, y, vy, z, vz -> for each trial -> for each 50ms bin
        n_trials = len(trials)
        states_data = [
            np.hstack([values[epoch] for values in data.values()]) 
            for epoch in range(n_trials)
        ]
        ## EXTRACT BINS
        binner = RatesTransformer(**bin_params).fit(trials,task)
        bin_data = binner.transform(trials) #for each trial -> for each electrode -> for each 50ms bin (between start and stop trial and NaN values until max time trial) -> spike rate

        ## TRAIN MODEL
        train_model(condition_dir, trials, cfg, opt_parameters, bin_params, available_steps, available_scorers, start, states_data, bin_data, targets)

def online_decoding(neural_file, condition):
    parts = os.path.basename(neural_file).split('_')
    training_dir = os.path.join(os.path.dirname(neural_file), 'trainingFiles')
    matching_files = glob.glob(os.path.join(training_dir, f"*_{parts[1]}_{parts[2]}_*_trials.pkl"))
    full_path = os.path.join(training_dir, parts[2], condition, "decoding_results.pkl")

    if os.path.exists(full_path):
        print(f"File already exists: {full_path}")
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
        return data['trial_answers'], data['allRewards'], data['allPredictions']
    else:
        # Continue with your analysis or saving
        print("File does not exist, proceeding with processing...")
        ## Setup Configuration and Model 
        # Configuration
        with open(os.path.join(training_dir, parts[2], condition, 'bin_params.pkl'), 'rb') as config_file:
            preloaded_config = pickle.load(config_file)
        with open(os.path.join(training_dir, parts[2], condition, 'config.pkl'), 'rb') as config_file:
            extra_config = pickle.load(config_file)
        output_directory =  os.path.join(training_dir, parts[2], condition)
        configuration = localconfig.Configuration(**preloaded_config)
        configuration.modelDirectory =os.path.join(training_dir, parts[2])
        update_configuration(configuration, extra_config)
        # Model
        model = Model(output_directory, configuration)
        configuration.channels = model.DataConfiguration()['channels']
        configuration.binWidth = model.DataConfiguration()['binWidth']
        configuration.withLfp = model.DataConfiguration()['withLfp']
        configuration.combined = model.DataConfiguration()['combined']
        configuration.lfpLength = model.DataConfiguration()['lfpLength']
        configuration.frequencyBand = model.DataConfiguration()['frequencyBand']
        configuration.samplingRate = model.DataConfiguration()['samplingRate']
        configuration.withSpikes = model.DataConfiguration()['withSpikes']

        ## Bin online decoding data
        pipeline_params = {
                "preprocessing__rates_comp__t_int": 8000,
                "preprocessing__rates_comp__t_skip": 0,
                "preprocessing__rates_comp__event": "GoCue",
                "preprocessing__rates_comp__stop_flag": "stop",
                "preprocessing__rates_comp__channels": configuration.channels,
                "preprocessing__rates_comp__channels_lfp": configuration.channels_lfp,
                "preprocessing__rates_comp__bin_size": configuration.binWidth,
                "preprocessing__rates_comp__withLfp": configuration.withLfp,
                "preprocessing__rates_comp__lfpLength": configuration.lfpLength,
                "preprocessing__rates_comp__combined": configuration.combined,
                "preprocessing__rates_comp__samplingRate": configuration.samplingRate,
                "preprocessing__rates_comp__frequencyBand": configuration.frequencyBand,
            }
        bin_params = {
            key.split("__")[-1]: value
            for key, value in pipeline_params.items()
            if "rates_comp" in key
        }
        binner = RatesTransformer(**bin_params).fit(trials,task)
        timeStamps_spikes, timeStamps_lfps = binner.get_binned_data(trials, configuration.channels_lfp)

        epocher = EpochTransformer(**bin_params) # EpochTransformer with bin parameters
        data = epocher.fit_transform(trials, task) #x, vx, y, vy, z, vz -> for each trial -> for each 50ms bin
        trial_answers = [trial.answer for trial in trials]
        # targetReached = [trial.targetReached for trial in trials]
        
        ## Predictions: For loop over trials and over bins
        allPredictions = []
        allRewards = []
        last_prediction = False
        for trial_index in range(0,len(trials)):
            last_bin = False
            last_trial = False
            if trial_index == len(trials)-1:
                last_trial = True
            model.TrialInit(trials[trial_index])
            trialPredictions = []  
            binWidth = 50
            timeStamps_spikes_trial = timeStamps_spikes[trial_index]
            spikeHistograms = [bintrials.SpikeRates_offline(evt, binWidth, configuration.channels) for evt in timeStamps_spikes_trial]
            spikeHistograms = np.array(np.transpose([np.array(list(histogram.values()), dtype=float) for histogram in spikeHistograms]))
            timeStamps_lfps_trial = []
            lfpPower = []
            # timeStamps_lfps_trial = timeStamps_lfps[trial_index]
            # lfpPower = [bintrials.LfpPowers_offline(evt, configuration.channels_lfp, configuration.frequencyBand, configuration.samplingRate) for evt in timeStamps_lfps_trial]
            # lfpPower = np.array(np.transpose([np.array(list(power.values()), dtype=float) for power in lfpPower]))
            # Compute velocities
            for bin_index in range(0,len(spikeHistograms[1])):
                    if bin_index == len(spikeHistograms[1])-1:
                        last_bin = True
                    spikeHistogram = np.array([histogram[bin_index] for histogram in spikeHistograms])
                    # spikeHistogram = np.array([histogram[bin_index + 4] for histogram in spikeHistograms])
                    lfpFeatures = np.array([lfp[bin_index] for lfp in lfpPower])
                    if last_trial == True and last_bin == True:
                        last_prediction = True
                    predictions = model.Predict(np.array(spikeHistogram), np.array(lfpFeatures), configuration.withSpikes, last_prediction)
                    trialPredictions.append(predictions)
            allPredictions.append(trialPredictions)   
            model.Reset() 

        data_to_save = {
            'trial_answers': trial_answers,
            'allRewards': allRewards,
            'allPredictions': allPredictions
        }

        # Save to file
        with open(full_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        return trial_answers, allRewards, allPredictions


def extract_latent_states(monkeys, experiments, base_dir):

    for experiment in experiments:
        for monkey in monkeys: 
            all_trials, all_correct, all_incorrect, all_training, all_channels, nb_channels, pkl_files, ai_trials_list = load_files(experiment, monkey, base_dir)
            # Directory
            base_directory = r"X:\Experiments"
            path = os.path.join(base_dir, monkey, experiment)

            # Loop over each file
            for trials, neural_file in zip(all_trials, pkl_files):
                # Session id
                basename = os.path.basename(neural_file)  # 'navdecoding_Maui_20240422_0935_A_trials.pkl'
                session_id = basename.replace("_trials.pkl", "") 
                match = re.search(r'_(\d{8})_', basename)
                if match:
                    date_str = match.group(1)
                    print("Date:", date_str)  # Output: 20230825
                # Load variables
                targets = [np.array([trial.targetPosition[0], trial.targetPosition[1], trial.targetPosition[2]]) for trial in trials]
                trial_answers = [trial.answer for trial in trials]

                ## Extract training file
                parts = os.path.basename(neural_file).split('_')
                training_dir = os.path.join(os.path.dirname(neural_file), 'trainingFiles')
                matching_files = glob.glob(os.path.join(training_dir, f"*_{parts[1]}_{parts[2]}_*_trials.pkl"))
                training_file = matching_files[0] if matching_files else None

                ## Train model
                train_different_models(training_file, neural_file)

                ## Online decoding
                conditions = ["Online_simulation"]
                # conditions = ['Online_simulation','All', 'PMd', 'PMv', 'M1', 'PMv+PMd', 'M1+PMd', "M1+PMv"]

                for condition in conditions: 
                    trial_answers_offline_decoding, allRewards, allPredictions = online_decoding(neural_file, condition)
                    # plot_trajectories(trials, allPredictions)
                    allPrediction_path = os.path.join(path, date_str, condition)
                    with open(os.path.join(allPrediction_path,"decoding_results.pkl"), "rb") as f:
                        online_decoding_prediction = pickle.load(f)
                    trial_answers_offline_decoding = online_decoding_prediction['trial_answers']
                    allRewards = online_decoding_prediction['allRewards']
                    allPredictions = online_decoding_prediction['allPredictions']
                    
            print("Done")

def main():
    parser = argparse.ArgumentParser(description="Run success rate analysis for AI vs non-AI BCI trials.")
    parser.add_argument('--monkeys', nargs='+', default=["Monkey 3", "Monkey 1"], help='List of monkeys to analyze')
    parser.add_argument('--experiments', nargs='+', default=["AI Appearing Obstacle 2"], help='List of experiment names') #, "AI Appearing Obstacle", "AI Respawn"
    parser.add_argument('--base_dir', type=str, default=None, help='Path to data directory')

    args = parser.parse_args()

    # Load base_dir from config.yaml if not provided
    if args.base_dir is None:
        import yaml
        import pathlib
        config_path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        args.base_dir = config.get("base_dir")
        if args.base_dir is None:
            raise ValueError("base_dir must be specified via CLI or in config.yaml")
        
    extract_latent_states(args.monkeys, args.experiments, args.base_dir)

if __name__ == "__main__":
    main()