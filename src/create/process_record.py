#!/usr/bin/env python3
"""
MIMIC III Dataset Creator
"""

import json
import os
import time
import wfdb
import numpy as np
import pandas as pd
import logging
import urllib3
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor 
import logging.handlers
import queue
import threading

from preprocessing.windowing import create_windower
from preprocessing.signal_processing import perform_signal_processing
from preprocessing.imputing import is_imputer_that_needs_split

from validation.validation import validate_record, generate_detailed_analysis, analyze_nan_values, save_reports


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance that works in worker processes."""
    if name is None:
        name = __name__
    return logging.getLogger(name)

def extract_subject_id(record_path: str) -> str:
    """Extract subject ID from record path."""
    return record_path.split('/')[1]


def extract_record_id(record_path: str) -> str:
    """Extract record ID from record path."""
    return record_path.split('/')[-1]



def load_record_data_from_wfdb(record_path: str,  offset_start_seconds: int, offset_end_seconds: int, config: Dict[str, Any], logger) -> Tuple[Optional[np.ndarray], List[str], str]:
    """
    Load signal data from a record.
    
    Returns:
        Tuple of (signal_data, channel_names, error_message)
    """
    try:
        database_name = config.get('database_name', 'mimic3wdb-matched/1.0')
        validation_config = config.get('validation', {})
        strict_nan_check = validation_config.get('strict_nan_check', True)
        
        # Get required channels
        input_channels = config.get('input_channels', [])
        output_channels = config.get('output_channels', [])
        required_channels = list(set(input_channels + output_channels))
        
        # Extract directory and record name
        path_parts = record_path.split('/')
        directory = f"{database_name}/{'/'.join(path_parts[:-1])}"
        record_name = path_parts[-1]
        
        # Load the record
        record = wfdb.rdrecord(record_name, pn_dir=directory)
        
        # Get available channels and find indices for required channels
        available_channels = record.sig_name
        channel_indices = []
        found_channels = []
        
        for ch in required_channels:
            if ch in available_channels:
                idx = available_channels.index(ch)
                if idx not in channel_indices:
                    channel_indices.append(idx)
                    found_channels.append(ch)

        if not channel_indices:
            return None, [], f"No required channels found. Required: {required_channels}, Available: {available_channels}"
        
        # Extract only the required channels - this is the key fix
        signal_data = record.p_signal[:, channel_indices]

        # Apply offsets
        if offset_start_seconds is not None and offset_end_seconds is not None:
            start_sample = int(offset_start_seconds * record.fs)
            end_sample = int(offset_end_seconds * record.fs)
            signal_data = signal_data[start_sample:end_sample]

        header = wfdb.rdheader(record_name, pn_dir=directory)

        # Validate data
        if signal_data is None:
            return None, [], "No signal data available in record"
        
        # Check for NaN values only in the required channels
        if strict_nan_check:
            nan_diagnostic = analyze_nan_values(signal_data, found_channels, header)
            return None, found_channels, f"NaN values detected in required channels: {nan_diagnostic}"
            
        return signal_data, found_channels, ""
        
    except Exception as e:
        return None, [], f"Load error: {str(e)}"


def filter_channels(data: np.ndarray, channel_names: List[str], 
                   required_channels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Filter data to only include required channels."""
    if not required_channels:
        return data, channel_names
        
    indices = []
    filtered_names = []
    
    for required_ch in required_channels:
        if required_ch in channel_names:
            idx = channel_names.index(required_ch)
            indices.append(idx)
            filtered_names.append(required_ch)
            
    if not indices:
        return np.array([]), []
        
    filtered_data = data[:, indices]
    return filtered_data, filtered_names


def save_uncutsamples(samples: List[dict], 
                channel_names: List[str], record_id: str, subject_id: str,
                config: Dict[str, Any], output_manager, logger, row_index) -> int:
    """Save training samples to CSV files grouped by subject."""
    try:

        # Create subject directory
        data_dir = output_manager.get_run_directory() / "data"
        data_dir.mkdir(exist_ok=True)
        subject_dir = data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        samples_saved = 0

        # create folders for observation and prediction
        uncut_dir = subject_dir / "uncut"
        uncut_dir.mkdir(exist_ok=True)

        # special case
        if record_id == "p001785-2140-01-07-12-34n" and sample_idx == 326 and row_index == 30:
            pass        
        
        for i in range(len(samples)):
            # save obs_data in a dataframe
            uncut_dict = {}
            for channel_name in channel_names:
                uncut_dict[channel_name] = samples[i][channel_name]
            
                # Save as CSV
                df = pd.DataFrame(uncut_dict)
                filename = f"{record_id}_sample_{i}.csv"
                filepath = uncut_dir / filename
                df.to_csv(filepath, index=False)

            samples_saved += 1
        
        return samples_saved
        
    except Exception as e:
        logger.error(f"Failed to save samples for {record_id}: {e}")
        return 0

def save_windows(snippets: List[dict], 
                channel_names: List[str], record_id: str, subject_id: str,
                config: Dict[str, Any], output_manager, logger, row_index) -> int:
    """Save training samples to CSV files grouped by subject."""
    try:

        # Create subject directory
        data_dir = output_manager.get_run_directory() / "data"
        data_dir.mkdir(exist_ok=True)
        subject_dir = data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        samples_saved = 0

        # create folders for observation and prediction
        obs_dir = subject_dir / "observation"
        obs_dir.mkdir(exist_ok=True)
        pred_dir = subject_dir / "prediction"
        pred_dir.mkdir(exist_ok=True)


        # iterate over number of snippets
        for snippet_idx in range(len(snippets[channel_names[0]])):
            channels_obs = {}
            channels_preds = {}
            for channel_name in channel_names:
                channels_obs[channel_name] = snippets[channel_name][snippet_idx][0]
                channels_preds[channel_name] = snippets[channel_name][snippet_idx][1]

            # Save as CSV
            df = pd.DataFrame(channels_obs)
            filename = f"{record_id}_sample_{snippet_idx:04d}.csv"
            filepath = obs_dir/ filename
            df.to_csv(filepath, index=False)
            df = pd.DataFrame(channels_preds)
            filename = f"{record_id}_sample_{snippet_idx:04d}.csv"
            filepath = pred_dir / filename
            df.to_csv(filepath)
            samples_saved += 1
        
        return samples_saved
        
    except Exception as e:
        logger.error(f"Failed to save samples for {record_id}: {e}")
        return 0



def create_samples_from_record_from_wfdb(record_path: str, offset_start_seconds: int, offset_end_seconds: int, start_at_step:int, until_step: int, config: Dict[str, Any], 
                              output_manager, logger_name: str, row_index) -> Tuple[str, int, str, Dict[str, Any]]:
    """
    Create training samples from a single record.
    
    Args:
        logger_name: String name of logger (not Logger object for pickle compatibility)
    
    Returns:
        Tuple of (record_id, samples_created, status, details)
    """
    # Get logger in worker process
    logger = get_logger(logger_name)
    
    record_id = extract_record_id(record_path)
    subject_id = extract_subject_id(record_path)
    
    start_time = time.time()
    details = {
        'record_id': record_id,
        'subject_id': subject_id,
        'processing_time': 0,
        'channels_found': [],
        'input_channels_available': [],
        'output_channels_available': []
    }
    
    try:
        # Get configuration
        input_channels = config.get('input_channels', [])
        output_channels = config.get('output_channels', [])
        required_channels = list(set(input_channels + output_channels))
        
        signal_configs = config.get('signal_processing', {})
        
        # Validate record
        is_valid, error_msg, metadata = validate_record(record_path, config)
        if not is_valid:
            details.update(metadata)
            return record_id, 0, error_msg, details
        
        details.update(metadata)
        metadata["min_record_duration"] = config.get('validation', {}).get('min_record_duration', 7200)
        
        # Load signal data
        signal_data, channel_names, load_error = load_record_data_from_wfdb(record_path, offset_start_seconds, offset_end_seconds, config, logger)

        # Always capture channel information if available
        if channel_names:
            details['channels_found'] = channel_names
        elif 'available_channels' in details:
            details['channels_found'] = details['available_channels']
            
        if load_error:
            return record_id, 0, load_error, details
        
        # Filter to required channels
        filtered_data, filtered_names = filter_channels(signal_data, channel_names, required_channels)
        if len(filtered_data) == 0:
            return record_id, 0, "No required channels found", details
            
        # Track which channels are available
        details['input_channels_available'] = [ch for ch in input_channels if ch in filtered_names]
        details['output_channels_available'] = [ch for ch in output_channels if ch in filtered_names]
        
        if not details['input_channels_available'] or not details['output_channels_available']:
            return record_id, 0, "Missing input or output channels", details
        
        # Preprocessing Pipeline
        long_nan_removal_config = config.get('long_nan_seq_removal', None)
        processed_data_array, logger_infos = perform_signal_processing(
            filtered_data=filtered_data, 
            filtered_names=filtered_names, 
            signal_processing=signal_configs, 
            start_at_processing_step=start_at_step,
            process_until_step=until_step,
            metadata=metadata, 
            logger=logger  # Pass the logger
        )        

        for logger_info in logger_infos:
            logger.info(f"{logger_info} record_id={record_id}")

        # Check for NaNs introduced during processing
        #validation_config = config.get('validation', {})
        #strict_nan_check = validation_config.get('strict_nan_check', True)
        #if strict_nan_check:
        #    for item_dict in processed_data_array:
        #        for channel_name, data in item_dict.items():
        #            if np.any(np.isnan(data)):
        #                nan_diagnostic = analyze_nan_values(data, channel_name)
        #                return record_id, 0, f"NaN values detected after processing in channel {channel_name}: {nan_diagnostic}", details
        

        # Create windows
        windowing_config = config.get('windowing', {})
        if windowing_config != {}:
            observation_window = windowing_config.get('observation_window', 3600)
            prediction_horizon = windowing_config.get('prediction_horizon', 300)
            prediction_window = windowing_config.get('prediction_window', 1800)
            step = windowing_config.get('step', 300)
            expected_resolution = windowing_config.get('expected_resolution', 1.0)
            windower = create_windower()
            
            logger.info(f"Creating windows for record {record_id}")
            windows = windower.create_windows(
                processed_data_array,
                observation_window,
                prediction_horizon, 
                prediction_window,
                step,
                expected_resolution
            )

            if not windows:
                return record_id, 0, "No valid windows created", details
        
            # Check for NaNs in windows if strict checking is enabled
            #if strict_nan_check:
            #    windows_with_nans = []
            #    for i, (obs_data, pred_data) in enumerate(windows):
            #        if np.any(np.isnan(obs_data)) or np.any(np.isnan(pred_data)):
            #            windows_with_nans.append(i)
                
                #if windows_with_nans:
                #    if len(windows_with_nans) == len(windows):
                #        # All windows have NaNs
                #        combined_sample = np.vstack([windows[0][0], windows[0][1]])
                #        nan_diagnostic = analyze_nan_values(combined_sample, filtered_names)
                #        return record_id, 0, f"NaN values detected in all windows: {nan_diagnostic}", details
                #    else:
                #        # Some windows have NaNs
                #        return record_id, 0, f"NaN values detected in {len(windows_with_nans)}/{len(windows)} windows", details
                    
            # Save windows
            filtered_names = [channel_config["channel"] for channel_config in signal_configs]
            samples_saved = save_windows(windows, filtered_names, record_id, subject_id, 
                                    config, output_manager, logger, row_index)
        else:
            samples_saved = save_uncutsamples(processed_data_array, filtered_names, record_id, subject_id,
                                    config, output_manager, logger, row_index)
        
        details['processing_time'] = time.time() - start_time
        logger.info(f"Successfully processed record {record_id}: {samples_saved} samples created")
        return record_id, samples_saved, "Success", details
        
    except Exception as e:
        details['processing_time'] = time.time() - start_time
        error_msg = f"Processing error: {str(e)}"
        logger.error(f"Error processing record {record_id}: {error_msg}")
        return record_id, 0, error_msg, details


def load_record_data_from_split(record_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[str], str]:  
    try:
        
        observation_df = pd.from_pickle(record_path)
        prediction_df = pd.from_pickle(record_path.replace("observation", "prediction"))

        observation_found_channels = list(observation_df.columns)
        prediction_found_channels = list(prediction_df.columns)

        observation_signal_data = observation_df.to_numpy()
        prediction_signal_data = prediction_df.to_numpy()

            
        return observation_signal_data, prediction_signal_data, observation_found_channels, prediction_found_channels, ""
        
    except Exception as e:
        return None, None, [], [], f"Load error: {str(e)}"


def create_samples_from_record_from_split(split: str, subject: str, start_step: int, end_step: int, config: Dict[str, Any], 
                              output_manager, logger_name: str, subject_index: int) -> Tuple[str, int, str, Dict[str, Any]]:
    """
    Create training samples from a single record.
    
    Args:
        logger_name: String name of logger (not Logger object for pickle compatibility)
    
    Returns:
        Tuple of (record_id, samples_created, status, details)
    """

    # Get logger in worker process
    logger = get_logger(logger_name)
    
    try:
        # Get configuration
        input_channels = config.get('input_channels', [])
        output_channels = config.get('output_channels', [])
        required_channels = list(set(input_channels + output_channels))
        
        signal_configs = config.get('signal_processing', {})
        
        subject_path = os.path.join(split, subject, "observation")
        long_nan_removal_config = config.get('long_nan_seq_removal', None)

        # TODO for now assue that current_sampling_rate is the same for all channels

        config_channel_1 = config.get('signal_processing', {})[0].get("steps", [{}])
        # remove all steps that are not downsampling or are larger than start_step
        # sort entries by "step"
        current_fs = [step.get("downsampling").get("desired_resolution", 1.0) for step in config_channel_1 if step.get("step", 0) < start_step and step.get("downsampling", {}) != {}]
        metadata = {
            "min_record_duration": config.get('validation', {}).get('min_record_duration', 7200),
            "sampling_rate": current_fs[-1] if len(current_fs) > 0 else 1.0,
            "imputer_path": config.get("output", {}).get("base_dir","") + "/data/iterative_imputer_X.pkl"
            }

        # read all file_names in subject_path
        record_files = os.listdir(subject_path)
        for record_file in record_files:
            record_path = os.path.join(subject_path, record_file)

            for type_record_path in [record_path, record_path.replace("observation", "prediction")]:

                # Load signal data
                df = pd.read_csv(type_record_path)
                found_channels = df.columns
                signal_data = df.to_numpy()
    

                # Preprocessing Pipeline
                processed_data_array, logger_infos = perform_signal_processing(
                    filtered_data=signal_data, 
                    filtered_names=found_channels, 
                    signal_processing=signal_configs, 
                    start_at_processing_step=start_step,
                    process_until_step=end_step,
                    metadata=metadata, 
                    logger=logger  # Pass the logger
                )        

                #print(processed_data_array)
                processed_data_df = pd.DataFrame(processed_data_array[0])
                #print(processed_data_df)
                processed_data_df.to_csv(type_record_path.replace(".csv", "_processed.csv"), index=False)
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
