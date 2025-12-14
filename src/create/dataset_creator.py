#!/usr/bin/env python3
"""
MIMIC III Dataset Creator

Dataset creation from MIMIC III records.
Creates time-series samples with observation and prediction windows.
"""

import json
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

from preprocessing.windowing import create_windower
from preprocessing.signal_processing import perform_signal_processing

# Suppress unnecessary logging
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('wfdb').setLevel(logging.ERROR)
urllib3.disable_warnings()

for logger_name in ['requests', 'urllib', 'urllib3', 'requests.packages.urllib3']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False



def extract_subject_id(record_path: str) -> str:
    """Extract subject ID from record path."""
    return record_path.split('/')[1]


def extract_record_id(record_path: str) -> str:
    """Extract record ID from record path."""
    return record_path.split('/')[-1]


def load_record_list(config: Dict[str, Any], logger) -> pd.DataFrame:
    """Load record list from input file."""

        
    records_file = Path(config.get('record_list_file', 'inputs/record_list.txt'))
    logger.info("Loading test record list")
            
    if not records_file.exists():
        raise FileNotFoundError(f"Records file not found: {records_file}")

    records_df = pd.read_csv(records_file)
    print(records_df.head())
    records_df["subject_id"] = records_df["record"].apply(lambda x: x.split('-')[0])
    records_df["subject_dir"] = records_df["subject_id"].apply(lambda x: f"p{x[1:3]}")
    records_df["full_path"] = records_df.apply(lambda x: f"{x['subject_dir']}/{x['subject_id']}/{x['record']}", axis=1)

    return records_df


def validate_record(record_path: str, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate a single record for dataset creation.
    
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    try:
        database_name = config.get('database_name', 'mimic3wdb-matched/1.0')
        validation_config = config.get('validation', {})
        min_duration = validation_config.get('min_record_duration', 7200)
        validate_channels = validation_config.get('validate_channels', True)
        
        # Required channels
        input_channels = config.get('input_channels', [])
        output_channels = config.get('output_channels', [])
        required_channels = list(set(input_channels + output_channels))
        
        # Extract directory and record name for wfdb
        path_parts = record_path.split('/')
        directory = f"{database_name}/{'/'.join(path_parts[:-1])}"
        record_name = path_parts[-1]
        
        # Read header only for efficiency
        header = wfdb.rdheader(record_name, pn_dir=directory)
        
        # Check duration
        duration_seconds = header.sig_len / header.fs if header.fs > 0 else 0
        if duration_seconds < min_duration:
            return False, f"Record too short: {duration_seconds:.1f}s < {min_duration}s", {}
        
        # Check channels if validation enabled
        available_channels = header.sig_name
        if validate_channels and required_channels:
            missing_channels = [ch for ch in required_channels if ch not in available_channels]
            if missing_channels:
                return False, f"Missing channels: {missing_channels}", {}
        
        metadata = {
            'duration_seconds': duration_seconds,
            'duration_hours': duration_seconds / 3600,
            'sampling_rate': header.fs,
            'n_channels': header.n_sig,
            'available_channels': available_channels
        }
        
        return True, "Valid", metadata
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", {}


def load_record_data(record_path: str,  offset_start_seconds: int, offset_end_seconds: int, config: Dict[str, Any], logger) -> Tuple[Optional[np.ndarray], List[str], str]:
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


        # Validate data
        if signal_data is None:
            return None, [], "No signal data available in record"
        
        # Check for NaN values only in the required channels
        if strict_nan_check and np.any(np.isnan(signal_data)):
            nan_diagnostic = analyze_nan_values(signal_data, found_channels)
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


def save_samples(windows: List[Tuple[np.ndarray, np.ndarray]], 
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
        
        for sample_idx, (obs_data, pred_data) in enumerate(windows):

            # special case
            if record_id == "p001785-2140-01-07-12-34n" and sample_idx == 326 and row_index == 30:
                pass
            
            # save obs_data in a dataframe
            obs_dict = {}
            pred_dict = {}
            for i, channel in enumerate(channel_names):
                obs_dict[channel] = obs_data[i]
                pred_dict[channel] = pred_data[i]

            if pred_dict == {}:
                pass
            
            # Save as CSV
            df = pd.DataFrame(obs_dict)
            filename = f"{record_id}_sample_{sample_idx:04d}_{row_index}.csv"
            filepath = obs_dir/ filename
            df.to_csv(filepath, index=False)

            df = pd.DataFrame(pred_dict)
            filename = f"{record_id}_sample_{sample_idx:04d}_{row_index}.csv"
            filepath = pred_dir / filename
            df.to_csv(filepath, index=False)

            samples_saved += 1
        
        return samples_saved
        
    except Exception as e:
        logger.error(f"Failed to save samples for {record_id}: {e}")
        return 0





def create_samples_from_record(record_path: str, offset_start_seconds: int, offset_end_seconds: int, config: Dict[str, Any], 
                              output_manager, logger, row_index) -> Tuple[str, int, str, Dict[str, Any]]:
    """
    Create training samples from a single record.
    
    Returns:
        Tuple of (record_id, samples_created, status, details)
    """
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
        
        windowing_config = config.get('windowing', {})
        observation_window = windowing_config.get('observation_window', 3600)
        prediction_horizon = windowing_config.get('prediction_horizon', 300)
        prediction_window = windowing_config.get('prediction_window', 1800)
        step = windowing_config.get('step', 300)
        
        # Validate record
        is_valid, error_msg, metadata = validate_record(record_path, config)
        if not is_valid:
            details.update(metadata)
            return record_id, 0, error_msg, details
        
        details.update(metadata)
        
        # Load signal data
        signal_data, channel_names, load_error = load_record_data(record_path, offset_start_seconds, offset_end_seconds, config, logger)

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
        processed_data, processed_names = perform_signal_processing(filtered_data=filtered_data, filtered_names=filtered_names, signal_processing=signal_configs, metadata=metadata)        

        # Check for NaNs introduced during processing
        validation_config = config.get('validation', {})
        strict_nan_check = validation_config.get('strict_nan_check', True)
        if strict_nan_check and np.any(np.isnan(processed_data)):
            nan_diagnostic = analyze_nan_values(processed_data, filtered_names)
            return record_id, 0, f"NaN values detected after processing: {nan_diagnostic}", details
        

        # Create windows
        expected_resolution = windowing_config.get('expected_resolution', 1.0)
        windower = create_windower()
        windows = windower.create_windows(
            processed_data,
            observation_window,
            prediction_horizon, 
            prediction_window,
            step,
            expected_resolution # just for testing puroses
        )

        if not windows:
            return record_id, 0, "No valid windows created", details
        
        # Check for NaNs in windows if strict checking is enabled
        if strict_nan_check:
            windows_with_nans = []
            for i, (obs_data, pred_data) in enumerate(windows):
                if np.any(np.isnan(obs_data)) or np.any(np.isnan(pred_data)):
                    windows_with_nans.append(i)
            
            if windows_with_nans:
                if len(windows_with_nans) == len(windows):
                    # All windows have NaNs
                    combined_sample = np.vstack([windows[0][0], windows[0][1]])
                    nan_diagnostic = analyze_nan_values(combined_sample, filtered_names)
                    return record_id, 0, f"NaN values detected in all windows: {nan_diagnostic}", details
                else:
                    # Some windows have NaNs
                    return record_id, 0, f"NaN values detected in {len(windows_with_nans)}/{len(windows)} windows", details
        
        # Save samples
        print("filtered_names", filtered_names)
        filtered_names = [channel_config["channel"] for channel_config in signal_configs]
        print("filtered_names", filtered_names)
        samples_saved = save_samples(windows, filtered_names, record_id, subject_id, 
                                   config, output_manager, logger, row_index)
        
        
        details['processing_time'] = time.time() - start_time
        return record_id, samples_saved, "Success", details
        
    except Exception as e:
        details['processing_time'] = time.time() - start_time
        return record_id, 0, f"Processing error: {str(e)}", details


def process_records_parallel(records_df: pd.DataFrame, config: Dict[str, Any], 
                           output_manager, logger) -> List[Tuple[str, int, str, Dict[str, Any]]]:
    """Process multiple records in parallel."""
    max_workers = 1
    
    results = []
    
        # Create process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        # Convert DataFrame rows to process arguments
        process_args = [
            (row['full_path'], row["offset_start_seconds"], row["offset_end_seconds"], config, output_manager, logger, idx)
            for idx, row in records_df.iterrows()
        ]

        
        # Submit all jobs
        future_to_record = {
            executor.submit(create_samples_from_record, *args): args[0]
            for args in process_args
        }
        
        # Collect results
        results = []
        for future in tqdm(as_completed(future_to_record), total=len(future_to_record)):
            record_id = future_to_record[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug(f"Successfully processed record {record_id}")
            except Exception as e:
                logger.error(f"Error processing record {record_id}: {str(e)}")
                results.append((record_id, None, str(e), {}))
    
    return results


def generate_detailed_analysis(results: List[Tuple[str, int, str, Dict[str, Any]]], 
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive analysis of processing results."""
    total_records = len(results)
    successful_records = [r for r in results if r[1] > 0]
    failed_records = [r for r in results if r[1] == 0]
    
    total_samples = sum(r[1] for r in results)
    
    input_channels = config.get('input_channels', [])
    output_channels = config.get('output_channels', [])
    
    # Basic statistics
    analysis = {
        'processing_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_records': total_records,
            'successful_records': len(successful_records),
            'failed_records': len(failed_records),
            'success_rate_percent': (len(successful_records) / total_records * 100) if total_records > 0 else 0,
            'total_samples_created': total_samples
        },
        'record_details': {},
        'failure_analysis': {},
        'channel_analysis': {
            'requested_input_channels': input_channels,
            'requested_output_channels': output_channels
        }
    }
    
    # Sample statistics
    if successful_records:
        sample_counts = [r[1] for r in successful_records]
        analysis['sample_statistics'] = {
            'min_samples': min(sample_counts),
            'max_samples': max(sample_counts),
            'mean_samples': np.mean(sample_counts),
            'median_samples': np.median(sample_counts),
            'std_samples': np.std(sample_counts)
        }
    
    # Detailed record information
    for record_id, samples, status, details in results:
        record_info = {
            'record_id': record_id,
            'samples_created': samples,
            'status': 'success' if samples > 0 else 'failed'
        }
        
        if details:
            record_info.update({
                'duration_hours': details.get('duration_hours', 0),
                'channels_available': details.get('channels_found', []),
                'input_channels_found': details.get('input_channels_available', []),
                'output_channels_found': details.get('output_channels_available', [])
            })
            
            if samples == 0:
                record_info['failure_reason'] = status
        
        analysis['record_details'][record_id] = record_info
    
    # Failure analysis
    failure_reasons = {}
    for record_id, samples, status, details in failed_records:
        # Extract detailed failure reason
        if "NaN values detected" in status:
            # Categorize NaN failures by stage
            if "during loading" in status:
                short_reason = "NaN in source data"
            elif "after downsampling" in status:
                short_reason = "NaN after downsampling"
            elif "in windows" in status or "in all windows" in status:
                short_reason = "NaN in windowing"
            else:
                short_reason = "NaN values detected"
        elif ":" in status:
            short_reason = status.split(":")[0]
        else:
            short_reason = status
            
        if short_reason not in failure_reasons:
            failure_reasons[short_reason] = {
                'count': 0, 
                'records': [],
                'detailed_messages': []
            }
        failure_reasons[short_reason]['count'] += 1
        failure_reasons[short_reason]['records'].append(record_id)
        failure_reasons[short_reason]['detailed_messages'].append(status)
    
    analysis['failure_analysis'] = failure_reasons
    
    return analysis


def generate_markdown_report(results: List[Tuple[str, int, str, Dict[str, Any]]], 
                           processing_time: float, config: Dict[str, Any]) -> str:
    """Generate markdown report content."""
    successful_records = [r for r in results if r[1] > 0]
    failed_records = [r for r in results if r[1] == 0]
    total_samples = sum(r[1] for r in results)
    
    # Sample statistics
    if successful_records:
        sample_counts = [r[1] for r in successful_records]
        min_samples = min(sample_counts)
        max_samples = max(sample_counts)
        avg_samples = total_samples / len(successful_records)
        median_samples = np.median(sample_counts)
    else:
        min_samples = max_samples = avg_samples = median_samples = 0
    
    # Get configuration
    input_channels = config.get('input_channels', [])
    output_channels = config.get('output_channels', [])
    
    windowing_config = config.get('windowing', {})
    observation_window = windowing_config.get('observation_window', 3600)
    prediction_horizon = windowing_config.get('prediction_horizon', 300)
    prediction_window = windowing_config.get('prediction_window', 1800)
    step = windowing_config.get('step', 300)
    target_fs = windowing_config.get('expected_resolution', 1.0)
    
    validation_config = config.get('validation', {})
    min_duration = validation_config.get('min_record_duration', 7200)
    
    report_lines = [
        "# Dataset Creation Report",
        "",
        "## Summary",
        f"- **Total Records Processed**: {len(results)}",
        f"- **Successful Records**: {len(successful_records)}",
        f"- **Failed Records**: {len(failed_records)}",
        f"- **Success Rate**: {(len(successful_records) / len(results) * 100):.1f}%",
        f"- **Total Samples Created**: {total_samples:,}",
        "",
        "## Processing Performance", 
        f"- **Processing Time**: {processing_time:.2f} seconds",
        f"- **Processing Rate**: {len(results) / processing_time:.3f} records/second",
        "",
        "## Sample Distribution",
        f"- **Minimum Samples per Record**: {min_samples}",
        f"- **Maximum Samples per Record**: {max_samples}",
        f"- **Average Samples per Record**: {avg_samples:.1f}",
        f"- **Median Samples per Record**: {median_samples:.1f}",
        ""
    ]
    
    # Add failure analysis if there were failures
    if failed_records:
        report_lines.extend([
            "## Failure Analysis",
            ""
        ])
        
        failure_reasons = {}
        for record_id, samples, status, details in failed_records:
            # Categorize failures better
            if "NaN values detected" in status:
                if "during loading" in status:
                    short_reason = "NaN in source data"
                elif "after downsampling" in status:
                    short_reason = "NaN after downsampling"
                elif "in windows" in status or "in all windows" in status:
                    short_reason = "NaN in windowing"
                else:
                    short_reason = "NaN values detected"
            elif ":" in status:
                short_reason = status.split(":")[0]
            else:
                short_reason = status
                
            if short_reason not in failure_reasons:
                failure_reasons[short_reason] = []
            failure_reasons[short_reason].append(record_id)
        
        for reason, records in failure_reasons.items():
            report_lines.append(f"- **{reason}**: {len(records)} records")
            
        # Add detailed NaN diagnostics section if there were NaN-related failures
        nan_failures = [r for r in failed_records if "NaN" in r[2]]
        if nan_failures:
            report_lines.extend([
                "",
                "### NaN Diagnostics",
                ""
            ])
            
            # Group by failure stage
            loading_failures = [r for r in nan_failures if "during loading" in r[2]]
            downsampling_failures = [r for r in nan_failures if "after downsampling" in r[2]]
            windowing_failures = [r for r in nan_failures if "in windows" in r[2] or "in all windows" in r[2]]
            
            if loading_failures:
                report_lines.append(f"**Source Data NaNs** ({len(loading_failures)} records): NaN values present in original waveform data")
            if downsampling_failures:
                report_lines.append(f"**Downsampling NaNs** ({len(downsampling_failures)} records): NaN values introduced during frequency conversion")
            if windowing_failures:
                report_lines.append(f"**Windowing NaNs** ({len(windowing_failures)} records): NaN values detected during window extraction")
        
        report_lines.append("")
    
    # Add configuration summary
    report_lines.extend([
        "## Configuration Summary",
        f"- **Input Channels**: {', '.join(input_channels)}",
        f"- **Output Channels**: {', '.join(output_channels)}",
        f"- **Target Sampling Rate**: {target_fs} Hz",
        f"- **Observation Window**: {observation_window} seconds",
        f"- **Prediction Horizon**: {prediction_horizon} seconds", 
        f"- **Prediction Window**: {prediction_window} seconds",
        f"- **Step Size**: {step} seconds",
        f"- **Minimum Record Duration**: {min_duration} seconds",
        ""
    ])
    
    return '\n'.join(report_lines)


def save_reports(results: List[Tuple[str, int, str, Dict[str, Any]]], 
                processing_time: float, config: Dict[str, Any], 
                output_manager, logger):
    """Save comprehensive reports and analysis."""
    try:
        # Generate detailed analysis
        analysis = generate_detailed_analysis(results, config)
        
        # Save detailed analysis JSON
        analysis_dir = output_manager.get_analysis_directory()
        analysis_path = analysis_dir / 'detailed_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate and save markdown report
        report_content = generate_markdown_report(results, processing_time, config)
        reports_dir = output_manager.get_reports_directory()
        report_path = reports_dir / 'dataset_creation_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info("Reports saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save reports: {e}")


def run_dataset_creation(config: Dict[str, Any], output_manager, logger):
    """Run the complete dataset creation process."""
    logger.info("Starting dataset creation")
    
    # Log configuration
    input_channels = config.get('input_channels', [])
    output_channels = config.get('output_channels', [])
    signal_config = config.get('signal_processing', [])
    #target_fs = signal_config.get('desired_resolution', 1.0)
    windowing_config = config.get('windowing', {})
    
    logger.info(f"Input channels: {input_channels}")
    logger.info(f"Output channels: {output_channels}")
    #logger.info(f"Target sampling rate: {target_fs} Hz")
    logger.info(f"Windows: obs={windowing_config.get('observation_window')}s, "
               f"horizon={windowing_config.get('prediction_horizon')}s, "
               f"pred={windowing_config.get('prediction_window')}s")
    
    start_time = time.time()
    
    try:
        # Load record list
        records_df = load_record_list(config, logger)
        
        # Process records
        max_workers = 4

        logger.info(f"Processing {len(records_df)} records with {max_workers} workers")
        results = process_records_parallel(records_df, config, output_manager, logger)

        # Calculate metrics
        processing_time = time.time() - start_time
        successful_records = [r for r in results if r[1] > 0]
        total_samples = sum(r[1] for r in results)
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Successfully processed {len(successful_records)}/{len(results)} records")
        logger.info(f"Created {total_samples:,} training samples")
        
        # Save reports
        save_reports(results, processing_time, config, output_manager, logger)
        
        logger.info("Dataset creation completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise

def analyze_nan_values(data: np.ndarray, channel_names: List[str]) -> str:
    """
    Analyze NaN values in signal data and provide detailed diagnostics.
    
    Returns:
        String with detailed NaN analysis
    """
    if data is None or len(data) == 0:
        return "No data to analyze"
    
    total_samples = data.shape[0]
    total_channels = data.shape[1] if len(data.shape) > 1 else 1
    
    # Overall NaN statistics
    total_nans = np.sum(np.isnan(data))
    nan_percentage = (total_nans / (total_samples * total_channels)) * 100
    
    diagnostics = []
    diagnostics.append(f"{total_nans:,} NaNs out of {total_samples * total_channels:,} values ({nan_percentage:.2f}%)")
    
    # Per-channel analysis
    if len(data.shape) > 1:
        channel_issues = []
        for i, channel_name in enumerate(channel_names):
            if i < data.shape[1]:
                channel_data = data[:, i]
                channel_nans = np.sum(np.isnan(channel_data))
                if channel_nans > 0:
                    channel_nan_pct = (channel_nans / total_samples) * 100
                    
                    # Find NaN patterns
                    nan_mask = np.isnan(channel_data)
                    nan_indices = np.where(nan_mask)[0]
                    
                    if len(nan_indices) > 0:
                        # Check if NaNs are at start, end, or scattered
                        if nan_indices[0] == 0:
                            pattern = "starts with NaNs"
                        elif nan_indices[-1] == total_samples - 1:
                            pattern = "ends with NaNs"
                        else:
                            # Check for consecutive blocks
                            consecutive_blocks = []
                            current_block_start = nan_indices[0]
                            current_block_size = 1
                            
                            for j in range(1, len(nan_indices)):
                                if nan_indices[j] == nan_indices[j-1] + 1:
                                    current_block_size += 1
                                else:
                                    consecutive_blocks.append((current_block_start, current_block_size))
                                    current_block_start = nan_indices[j]
                                    current_block_size = 1
                            consecutive_blocks.append((current_block_start, current_block_size))
                            
                            if len(consecutive_blocks) == 1:
                                pattern = f"single block at index {consecutive_blocks[0][0]}"
                            elif len(consecutive_blocks) <= 3:
                                pattern = f"{len(consecutive_blocks)} blocks"
                            else:
                                pattern = "scattered throughout"
                    
                    channel_issues.append(f"{channel_name}: {channel_nans:,} NaNs ({channel_nan_pct:.1f}%, {pattern})")
        
        if channel_issues:
            diagnostics.append("Affected channels: " + "; ".join(channel_issues))
    
    return "; ".join(diagnostics)


