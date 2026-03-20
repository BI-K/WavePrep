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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
import logging.handlers
import queue
import threading
import torch
import argparse
import sys

from common.pipeline_config import PipelineConfig
from create.process_record import create_samples_from_record_from_wfdb, create_samples_from_record_from_split, extract_record_id
from split.mimic_splitter import run_dataset_splitting, create_output_directory

from preprocessing.windowing import create_windower
from preprocessing.signal_processing import perform_signal_processing
from preprocessing.imputing import is_imputer_that_needs_split
from preprocessing.imputing import train_and_save_imputer

from validation.validation import validate_record, generate_detailed_analysis, analyze_nan_values, save_reports
from validation import initialize_visualization, merge_step_visualizations_for_record

# Worker process initialization function
def worker_init(log_file_path=None):
    """Initialize logging for worker processes."""
    handlers = [logging.StreamHandler()]
    
    # Add file handler if log file path is provided
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    # Suppress third-party logging
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('wfdb').setLevel(logging.ERROR)

def worker_init_shared():
    """Initialize logging for worker processes with shared queue."""
    # Workers will inherit the queue handler from the main process
    pass


def load_record_list(config: PipelineConfig, logger) -> pd.DataFrame:
    """Load record list from input file."""

        
    records_file = Path(config.record_list_file)
    logger.info("Loading test record list")
            
    if not records_file.exists():
        raise FileNotFoundError(f"Records file not found: {records_file}")

    records_df = pd.read_csv(records_file)
    records_df["subject_id"] = records_df["record"].apply(lambda x: x.split('-')[0])
    records_df["subject_dir"] = records_df["subject_id"].apply(lambda x: f"p{x[1:3]}")
    records_df["full_path"] = records_df.apply(lambda x: f"{x['subject_dir']}/{x['subject_id']}/{x['record']}", axis=1)

    return records_df


def check_if_imputer_needs_to_be_trained(config: PipelineConfig) -> bool:
    """Check if any imputer in the configuration needs to be trained."""
    for ch_cfg in config.signal_processing:
        for step in ch_cfg.steps:
            if step.imputation and is_imputer_that_needs_split(step.imputation.method):
                return True
    return False

def get_step_for_windowing_and_split(config: PipelineConfig) -> Tuple[int, int, int]:
    """Get start and stop processing steps for intermediate window creation."""
    max_steps = max(len(ch.steps) for ch in config.signal_processing)

    if check_if_imputer_needs_to_be_trained(config):
        for ch_cfg in config.signal_processing:
            for step in ch_cfg.steps:
                if step.imputation and is_imputer_that_needs_split(step.imputation.method):
                    return 0, step.step, max_steps

    return 0, max_steps, max_steps


def process_records_parallel_wfdb(records_df: pd.DataFrame, config: PipelineConfig, start_step: int, end_step: int, max_workers: int,
                           output_manager, logger, log_file_path=None, records_to_visualize = [],
                           intermediate: bool = False) -> List[Tuple[str, int, str, Dict[str, Any]]]:
    """Process multiple records in parallel."""
    
    logger.info(f"Starting parallel processing with {max_workers} workers")
    
    # Create process pool with worker initialization
    with ProcessPoolExecutor(max_workers=max_workers, 
                           initializer=worker_init, 
                           initargs=(log_file_path,)) as executor:
        # Convert DataFrame rows to process arguments
        # Pass logger name instead of logger object
        logger_name = logger.name if logger else 'dataset_creator'

        process_args = [
                (row['full_path'], row["offset_start_seconds"], row["offset_end_seconds"], start_step, end_step,
                config, output_manager, logger_name, idx, records_to_visualize, intermediate)
                for idx, row in records_df.iterrows()
        ]
            
        # Submit all jobs
        future_to_record = {
                executor.submit(create_samples_from_record_from_wfdb, *args): args[0]
                for args in process_args
        }
        
        # Collect results
        results = []
        for future in tqdm(as_completed(future_to_record), total=len(future_to_record)):
            record_path = future_to_record[future]
            record_id = extract_record_id(record_path)
            try:
                result = future.result()
                results.append(result)
                logger.debug(f"Successfully processed record {record_id}")
            except Exception as e:
                logger.error(f"Error processing record {record_id}: {str(e)}")
                results.append((record_id, 0, str(e), {}))
    
    logger.info(f"Completed parallel processing of {len(results)} records")
    return results


def process_records_parallel_split(config: PipelineConfig, start_step: int, end_step: int, max_workers: int,
                           output_manager, logger, log_file_path=None, records_to_visualize=[]) -> List[Tuple[str, int, str, Dict[str, Any]]]:
    """Process multiple records in parallel."""
    
    logger.info(f"Starting parallel processing with {max_workers} workers")

    path = config.output.base_dir

    for split_type in ["train", "test", "validation"]:
        print(f"Processing split: {split_type}")

        # read names pf folders in path/data/split
        split_path = os.path.join(path, "data", split_type)
        split_subjects = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
    
        # Create process pool with worker initialization
        with ProcessPoolExecutor(max_workers=max_workers, 
                            initializer=worker_init, 
                            initargs=(log_file_path,)) as executor:
            # Convert DataFrame rows to process arguments
            # Pass logger name instead of logger object
            logger_name = logger.name if logger else 'dataset_creator'

            process_args = [
                    (split_path, subject, start_step, end_step,
                    config, output_manager, logger_name, idx, records_to_visualize)
                    for idx, subject in enumerate(split_subjects)
            ]
                
            # Submit all jobs
            future_to_record = {
                    executor.submit(create_samples_from_record_from_split, *args): args[0]
                    for args in process_args
            }
            
            # Collect results
            results = []
            for future in tqdm(as_completed(future_to_record), total=len(future_to_record)):
                record_path = future_to_record[future]
                record_id = extract_record_id(record_path)
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Successfully processed record {record_id}")
                except Exception as e:
                    logger.error(f"Error processing record {record_id}: {str(e)}")
                    results.append((record_id, 0, str(e), {}))
    
    logger.info(f"Completed parallel processing of {len(results)} records")
    return results


def split_dataset(config: PipelineConfig, output_manager, logger) -> Dict[str, Any]:

    # Log splitting parameters
    train_ratio = config.splitting.train_ratio
    validation_ratio = config.splitting.validation_ratio
    test_ratio = config.splitting.test_ratio

    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
    # Log subject filtering if configured
    exclude_subjects = config.splitting.exclude_subjects
    include_only_subjects = config.splitting.include_only_subjects
        
    if exclude_subjects:
        logger.info(f"Excluding {len(exclude_subjects)} subjects")
    if include_only_subjects:
        logger.info(f"Including only {len(include_only_subjects)} subjects")
        
    # Run the splitting process
    results = run_dataset_splitting(config, output_manager, logger)
    return results


def create_dataset_pt_process_folder(folder_path: str, folder_name: str) -> List:
    """Process a single folder and return samples."""
    samples = []
    labels = []
    
    obs_dir = os.path.join(folder_path, folder_name, "observation")
    observation_files = [f for f in os.listdir(obs_dir)
                         if f.endswith('.csv') or f.endswith('.npy')
                         and not f.endswith('.npy.json')]
    for file in observation_files:
        file_path = os.path.join(obs_dir, file)
        if file.endswith('.npy'):
            data = np.load(file_path)
            samples.append(data.tolist())
        else:
            samples_df = pd.read_csv(file_path)
            samples.append(samples_df.values.tolist())
    
    return samples


def create_dataset_pt(path: str, is_train: bool):
    """Create dataset dictionary by reading CSV files in parallel."""
    samples = []
    labels = []

    if is_train:
        path = os.path.join(path, 'train')
    else:
        path = os.path.join(path, 'test')
    # read all folders in the path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Parallelize folder processing using ThreadPoolExecutor (I/O-bound task)
    max_workers = min(os.cpu_count() or 1, len(folders))  # Don't spawn more workers than folders
    # max_workers = 1
    print(f"Processing {len(folders)} folders with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_dataset_pt_process_folder, path, folder) for folder in folders]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
            try:
                folder_samples = future.result()
                samples.extend(folder_samples)
            except Exception as e:
                print(f"Error processing folder: {e}")

    print(f"Created dataset with {len(samples)} samples")
    return samples



def train_imputer(config: PipelineConfig, output_manager, logger):
    """Train and save imputer if required."""
    logger.info("Training imputer as per configuration")

    output_path = Path(config.output.base_dir)
    output_path = os.path.join(output_path, "data")
    
    train_samples = create_dataset_pt(path=output_path, is_train=True)

    # train imputer for input features X
    save_imputer_path = os.path.join(output_path, "iterative_imputer_X.pkl")
    # TODO actually load imputer from somewhere
    train_and_save_imputer(train_samples, 'iterative_imputer', save_imputer_path)

    # train imputer for prediction y
    # TODO do we really need two imputers?
    # save_imputer_path = os.path.join(output_path, "iterative_imputer_y.pkl")
    # train_and_save_imputer(train_dict["labels"], 'iterative_imputer', save_imputer_path)



# In run_dataset_creation, replace the ProcessPoolExecutor section:
def _log_phase_completion(logger, phase_name: str, start_time: float):
    """Log the completion of a pipeline phase with formatted runtime."""
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    logger.info("=" * 60)
    logger.info(f"{phase_name}")
    logger.info(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({elapsed:.2f} seconds)")
    logger.info("=" * 60)


def run_dataset_creation(config: PipelineConfig, output_manager, logger, log_file_path=None):
    """Run the complete dataset creation process."""
    logger.info("Starting dataset creation")
    
    w = config.windowing
    
    logger.info(f"Input channels: {config.input_channels}")
    logger.info(f"Output channels: {config.output_channels}")
    logger.info(f"Windows: obs={w.observation_window}s, "
               f"horizon={w.prediction_horizon}s, "
               f"pred={w.prediction_window}s")
    
    start_time = time.time()

    try:
        records_df = load_record_list(config, logger)

        create_output_directory(config, logger)
        records_to_visualize = initialize_visualization(records_df['record'].tolist(), config)
        
        max_workers = os.cpu_count() or 1
        logger.info(f"Processing {len(records_df)} records with {max_workers} workers")

        start_step, end_step, max_steps = get_step_for_windowing_and_split(config)
        has_second_pass = max_steps > end_step

        results = process_records_parallel_wfdb(records_df, config, start_step, end_step, 
                                                max_workers, output_manager, logger, log_file_path, records_to_visualize,
                                                intermediate=has_second_pass)
        _log_phase_completion(logger, "FIRST SIGNAL PROCESSING COMPLETED", start_time)

        split_results = split_dataset(config, output_manager, logger)
        _log_phase_completion(logger, "SPLIT COMPLETED", start_time)

        if end_step < max_steps:
            if check_if_imputer_needs_to_be_trained(config):
                train_imputer(config, output_manager, logger)

        if max_steps > end_step:
            results = process_records_parallel_split(config, end_step, max_steps, max_workers, output_manager, logger, log_file_path, records_to_visualize)
            _log_phase_completion(logger, "SECOND SIGNAL PROCESSING COMPLETED", start_time)

        _log_phase_completion(logger, "DATASET CREATION COMPLETED", start_time)
        
        output_path = Path(config.output.base_dir)
        output_path = output_path / "reports" / "process_images"
        for record in records_to_visualize:
            match = records_df[records_df["record"] == record].head(1)

            if not match.empty:
                start_offset_seconds = float(match["offset_start_seconds"].iloc[0])
                end_offset_seconds = float(match["offset_end_seconds"].iloc[0])
                
                merge_step_visualizations_for_record(record, start_offset_seconds, end_offset_seconds, signal_processing_config=config.signal_processing, output_path=output_path, windowing_config=config.windowing)

        # TODO fix Save reports
        # save_reports(results, processing_time, config, output_manager, logger)
        
        logger.info("Dataset creation completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise


