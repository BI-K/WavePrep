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
import logging.handlers
import queue
import threading

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
        
        
        # Check channels if validation enabled
        available_channels = header.sig_name
        if validate_channels and required_channels:
            missing_channels = [ch for ch in required_channels if ch not in available_channels]
            if missing_channels:
                return False, f"Missing channels: {missing_channels}", {}
        
        metadata = {
            'sampling_rate': header.fs,
            'n_channels': header.n_sig,
            'available_channels': available_channels
        }
        
        return True, "Valid", metadata
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", {}
    

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
                'duration_min': details.get('duration_min', 0),
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
        f"- **Standard Deviation of Samples per Record**: {np.std(sample_counts):.1f}",
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


def analyze_nan_values(data: np.ndarray, channel_name: str, header) -> str:
    """
    Analyze NaN values in signal data and provide detailed diagnostics.
    
    Returns:
        String with detailed NaN analysis
    """

    samplig_frequency = header.fs

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
                        # if nan_indices[0] == 0:
                        #    pattern = "starts with NaNs"
                        # elif nan_indices[-1] == total_samples - 1:
                        #    pattern = "ends with NaNs"
                        # else:
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
                                
                            pattern = f"{len(consecutive_blocks)} blocks"

                            min_size = min(size for _, size in consecutive_blocks)
                            max_size = max(size for _, size in consecutive_blocks)
                            median_size = np.median([size for _, size in consecutive_blocks])
                            pattern += f" (size min={min_size}, max={max_size}, median={median_size})"

                            min_in_seconds = min_size / samplig_frequency /60
                            max_in_seconds = max_size / samplig_frequency /60
                            median_in_seconds = median_size / samplig_frequency /60
                            pattern += f" samples (~{min_in_seconds:.1f} min to {max_in_seconds:.1f} min, median ~{median_in_seconds:.1f} min)"

                        
                    channel_issues.append(f"{channel_name}: {channel_nans:,} NaNs ({channel_nan_pct:.1f}%, {pattern})")
            
        if channel_issues:
            diagnostics.append("Affected channels: " + "; ".join(channel_issues))
        
        return "; ".join(diagnostics)