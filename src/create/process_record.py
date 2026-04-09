#!/usr/bin/env python3
"""
MIMIC III Dataset Creator
"""

import os
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from common.pipeline_config import PipelineConfig
from common.processing_context import ProcessingContext
from create.record_loader import RecordLoader, LoadedRecord, create_record_loader
from preprocessing.windowing import create_windower
from preprocessing.signal_processing import perform_signal_processing
from preprocessing.imputing import is_imputer_that_needs_split
from common.signal_data import SignalData, WindowedData
from common.signal_io import get_signal_writer, get_file_extension

import json as _json


def _write_npy_intermediate(filepath: Path, data: np.ndarray, channel_names: List[str]) -> None:
    """Write data as .npy with a JSON sidecar for channel names. Used for fast intermediate storage."""
    np.save(filepath, data)
    sidecar = Path(str(filepath) + '.json')
    with open(sidecar, 'w') as f:
        _json.dump({'channel_names': channel_names}, f)

from validation.validation import validate_record, generate_detailed_analysis, save_reports
from validation.visualize_steps import visualize_windowing_for_record


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance that works in worker processes."""
    if name is None:
        name = __name__
    return logging.getLogger(name)

def extract_subject_id(record_path: str) -> str:
    """Extract subject ID from record path."""
    return Path(record_path).parts[1]


def extract_record_id(record_path: str) -> str:
    """Extract record ID from record path."""
    return Path(record_path).parts[-1]


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


def save_uncutsamples(samples: List[SignalData],
                channel_names: List[str], record_id: str, subject_id: str,
                config: PipelineConfig, output_manager, logger, row_index,
                intermediate: bool = False) -> int:
    """Save training samples grouped by subject.
    
    When intermediate=True, uses NumPy .npy for fast I/O (32x faster than CSV)
    during multi-pass processing. Final output uses the user's chosen format.
    """
    try:

        # Create subject directory
        data_dir = output_manager.get_run_directory() / "data"
        data_dir.mkdir(exist_ok=True)
        subject_dir = data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        samples_saved = 0

        if intermediate:
            ext = '.npy'
        else:
            ext = get_file_extension(config.output.save_format)
        writer = None if intermediate else get_signal_writer(
            config.output.save_format
        )
        fs = 1.0 / config.windowing.expected_resolution if config.windowing.expected_resolution else 1.0

        # create folders for observation and prediction
        uncut_dir = subject_dir / "uncut"
        uncut_dir.mkdir(exist_ok=True)

        for i in range(len(samples)):
            # save obs_data in a dataframe
            uncut_dict = {}
            for channel_name in channel_names:
                uncut_dict[channel_name] = samples[i][channel_name]
            
                data = np.column_stack([uncut_dict[ch] for ch in channel_names])
                filename = f"{record_id}_sample_{i}{ext}"
                filepath = uncut_dir / filename
                if intermediate:
                    _write_npy_intermediate(filepath, data, channel_names)
                else:
                    writer(filepath, data, channel_names, fs)

            samples_saved += 1
        
        return samples_saved
        
    except Exception as e:
        logger.error(f"Failed to save samples for {record_id}: {e}")
        return 0

def save_windows(windows: WindowedData,
                channel_names: List[str], record_id: str, subject_id: str,
                config: PipelineConfig, output_manager, logger, row_index,
                intermediate: bool = False) -> int:
    """Save windowed samples grouped by subject.
    
    When intermediate=True, uses NumPy .npy for fast I/O (32x faster than CSV)
    during multi-pass processing. Final output uses the user's chosen format.
    """
    try:

        data_dir = output_manager.get_run_directory() / "data"
        data_dir.mkdir(exist_ok=True)
        subject_dir = data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        samples_saved = 0

        if intermediate:
            ext = '.npy'
        else:
            ext = get_file_extension(config.output.save_format)
        writer = None if intermediate else get_signal_writer(
            config.output.save_format
        )
        fs = 1.0 / config.windowing.expected_resolution if config.windowing.expected_resolution else 1.0

        obs_dir = subject_dir / "observation"
        obs_dir.mkdir(exist_ok=True)
        pred_dir = subject_dir / "prediction"
        pred_dir.mkdir(exist_ok=True)

        for window_idx in range(windows.n_windows):
            channels_obs = {}
            channels_preds = {}
            for channel_name in channel_names:
                window = windows[channel_name][window_idx]
                channels_obs[channel_name] = window.observation
                channels_preds[channel_name] = window.prediction

            obs_data = np.column_stack([channels_obs[ch] for ch in channel_names])
            filename = f"{record_id}_sample_{window_idx:04d}{ext}"
            filepath = obs_dir / filename
            if intermediate:
                _write_npy_intermediate(filepath, obs_data, channel_names)
            else:
                writer(filepath, obs_data, channel_names, fs)

            pred_data = np.column_stack([channels_preds[ch] for ch in channel_names])
            filename = f"{record_id}_sample_{window_idx:04d}{ext}"
            filepath = pred_dir / filename
            if intermediate:
                _write_npy_intermediate(filepath, pred_data, channel_names)
            else:
                writer(filepath, pred_data, channel_names, fs)

            samples_saved += 1
        
        return samples_saved
        
    except Exception as e:
        logger.error(f"Failed to save samples for {record_id}: {e}")
        return 0



def create_samples_from_record_from_wfdb(record_path: str, offset_start_seconds: int, offset_end_seconds: int, start_at_step:int, until_step: int, config: PipelineConfig, 
                              output_manager, logger_name: str, row_index, records_to_visualize, intermediate: bool = False) -> Tuple[str, int, str, Dict[str, Any]]:
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
        required_channels = config.required_channels
        metadata_base = ProcessingContext(
            min_record_duration=config.validation.min_record_duration,
            record_id=record_id,
            output_path_process_images=str(Path(config.output.base_dir) / "reports" / "process_images"),
            windowing_config=config.windowing,
        )
        
        loader = create_record_loader(
            'wfdb',
            record_path=record_path,
            offset_start=offset_start_seconds,
            offset_end=offset_end_seconds,
            config=config,
            logger=logger,
            metadata_base=metadata_base,
        )
        loaded_records, load_error = loader.load()
        
        samples_saved = 0
        for record in loaded_records:
            if record.channel_names:
                details['channels_found'] = record.channel_names
            elif 'available_channels' in details:
                details['channels_found'] = details['available_channels']
                
            if load_error:
                return record_id, 0, load_error, details

            filtered_data, filtered_names = filter_channels(record.signal_data, record.channel_names, required_channels)
            if len(filtered_data) == 0:
                return record_id, 0, "No required channels found", details
                
            details['input_channels_available'] = [ch for ch in config.input_channels if ch in filtered_names]
            details['output_channels_available'] = [ch for ch in config.output_channels if ch in filtered_names]
            
            if not details['input_channels_available'] or not details['output_channels_available']:
                return record_id, 0, "Missing input or output channels", details
            
            processed_data_array, logger_infos = perform_signal_processing(
                filtered_data=filtered_data, 
                filtered_names=filtered_names, 
                signal_processing=config.signal_processing, 
                start_at_processing_step=start_at_step,
                process_until_step=until_step,
                long_nan_removal_config=config.long_nan_seq_removal,
                metadata=record.metadata, 
                logger=logger,
                records_to_visualize=records_to_visualize
            )        

            for logger_info in logger_infos:
                logger.info(f"{logger_info} record_id={record_id}")
            
            if len(processed_data_array) == 0:
                return record_id, 0, "No valid processed data after preprocessing", details
            
            w = config.windowing
            if config.signal_processing:
                windower = create_windower()
                
                logger.info(f"Creating windows for record {record_id}")

                windows = windower.create_windows(
                    processed_data_array,
                    w.observation_window,
                    w.prediction_horizon, 
                    w.prediction_window,
                    w.step,
                    w.expected_resolution
                )
                if record_id in records_to_visualize:
                    visualize_windowing_for_record(record_id, until_step - 1, windows, processed_data_array, w.observation_window, w.prediction_horizon, w.prediction_window, w.step, w.expected_resolution, output_path=record.metadata.output_path_process_images)

                if not windows:
                    return record_id, 0, "No valid windows created", details
            
                filtered_names = config.channel_names
                samples_saved = save_windows(windows, filtered_names, record_id, subject_id, 
                                        config, output_manager, logger, row_index,
                                        intermediate=intermediate)
            else:
                samples_saved = save_uncutsamples(processed_data_array, filtered_names, record_id, subject_id,
                                        config, output_manager, logger, row_index,
                                        intermediate=intermediate)
            
            details['processing_time'] = time.time() - start_time
            logger.info(f"Successfully processed record {record_id}: {samples_saved} samples created")
        return record_id, samples_saved, "Success", details
        
    except Exception as e:
        details['processing_time'] = time.time() - start_time
        error_msg = f"Processing error: {str(e)}"
        logger.error(f"Error processing record {record_id}: {error_msg}")
        return record_id, 0, error_msg, details


def create_samples_from_record_from_split(split: str, subject: str, start_step: int, end_step: int, config: PipelineConfig, 
                              output_manager, logger_name: str, subject_index: int, records_to_visualize=[]) -> Tuple[str, int, str, Dict[str, Any]]:
    """
    Apply later processing steps to already-split records.

    Reads intermediate .npy files (written by the first pass for speed),
    runs the remaining processing steps, and writes the final output in the
    user's chosen format, removing the intermediate .npy files afterwards.
    
    Args:
        logger_name: String name of logger (not Logger object for pickle compatibility)
    
    Returns:
        Tuple of (record_id, samples_created, status, details)
    """
    logger = get_logger(logger_name)
    
    try:
        subject_path = Path(split) / subject / "observation"

        config_channel_1 = config.signal_processing[0].steps
        current_fs = [s.downsampling.desired_resolution for s in config_channel_1 if s.step < start_step and s.downsampling]
        base_metadata = ProcessingContext(
            min_record_duration=config.validation.min_record_duration,
            sampling_rate=current_fs[-1] if len(current_fs) > 0 else 1.0,
            imputer_path=str(Path(config.output.base_dir) / "data" / "iterative_imputer_X.pkl"),
            output_path_process_images=str(Path(config.output.base_dir) / "reports" / "process_images"),
            windowing_config=config.windowing,
        )

        save_format = config.output.save_format
        writer = get_signal_writer(save_format)
        final_ext = get_file_extension(save_format)
        fs = 1.0 / config.windowing.expected_resolution if config.windowing.expected_resolution else 1.0

        # Only iterate actual data files, not .npy.json sidecars
        record_files = [f for f in os.listdir(str(subject_path))
                        if f.endswith('.npy') or f.endswith('.csv')]
        converted = 0
        for record_file in record_files:
            record_path = subject_path / record_file
            prediction_path = record_path.parent.parent / "prediction" / record_path.name

            for file_path in [record_path, prediction_path]:
                try:
                    file_path_str = str(file_path)
                    record_type = "observation" if "observation" in file_path_str else "prediction"
                    base_metadata.record_id = f"{record_path.stem}_{record_type}"

                    # Detect intermediate format: .npy or .csv
                    if file_path.suffix == '.npy':
                        loader = create_record_loader('npy', file_path=file_path_str, metadata=base_metadata)
                    else:
                        loader = create_record_loader('csv', file_path=file_path_str, metadata=base_metadata)

                    loaded_records, load_error = loader.load()
                    if load_error or not loaded_records:
                        continue

                    rec = loaded_records[0]
                    processed_data_array, logger_infos = perform_signal_processing(
                        filtered_data=rec.signal_data, 
                        filtered_names=rec.channel_names, 
                        signal_processing=config.signal_processing, 
                        start_at_processing_step=start_step,
                        process_until_step=end_step,
                        long_nan_removal_config=config.long_nan_seq_removal,
                        metadata=base_metadata, 
                        logger=logger,
                        records_to_visualize=records_to_visualize
                    )        

                    if not processed_data_array:
                        logger.warning(f"No output from signal processing for {file_path.name}")
                        continue

                    # Write final output in user's chosen format
                    processed_data = dict(processed_data_array[0])
                    channel_names = list(processed_data.keys())
                    data = np.column_stack([processed_data[ch] for ch in channel_names])
                    final_path = file_path.with_suffix(final_ext)
                    writer(final_path, data, channel_names, fs)
                    converted += 1

                    # Remove intermediate .npy + sidecar if present
                    if file_path.suffix == '.npy':
                        file_path.unlink(missing_ok=True)
                        sidecar = Path(file_path_str + '.json')
                        if sidecar.exists():
                            sidecar.unlink()

                except Exception as e:
                    logger.error(f"Error converting {file_path}: {e}")
        
        return subject, converted, "Success", {}
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        return "", 0, error_msg, {}
