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
import os
import pandas as pd
import numpy as np
import torch
import argparse
import sys


from create.process_record import create_samples_from_record_from_wfdb, create_samples_from_record_from_split, extract_record_id
from split.mimic_splitter import run_dataset_splitting

from preprocessing.windowing import create_windower
from preprocessing.signal_processing import perform_signal_processing
from preprocessing.imputing import is_imputer_that_needs_split
from preprocessing.imputing import train_and_save_imputer

from validation.validation import validate_record, generate_detailed_analysis, analyze_nan_values, save_reports



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


def check_if_imputer_needs_to_be_trained(config: Dict[str, Any]) -> bool:
    """Check if any imputer in the configuration needs to be trained."""
    signal_configs = config.get('signal_processing', {})
    for channel_config in signal_configs:
        steps = channel_config.get('steps', [])
        for step_config in steps:
            to_imputation = step_config.get("imputation", {})
            if is_imputer_that_needs_split(to_imputation.get('method')):
                return True
    return False

def get_step_for_windowing_and_split(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Get start and stop processing steps for intermediate window creation."""
    signal_configs = config.get('signal_processing', {})
    max_steps = max(len(channel_config.get('steps', [])) for channel_config in signal_configs)

    if check_if_imputer_needs_to_be_trained(config):
        for channel_config in signal_configs:
            steps = channel_config.get('steps', [])
            for step_idx in range(len(steps)):
                to_imputation = steps[step_idx].get("imputation", {})
                if is_imputer_that_needs_split(to_imputation.get('method')):
                    return 0, step_idx, max_steps

    return 0, max_steps, max_steps


def process_records_parallel_wfdb(records_df: pd.DataFrame, config: Dict[str, Any], start_step: int, end_step: int, max_workers: int,
                           output_manager, logger, log_file_path=None) -> List[Tuple[str, int, str, Dict[str, Any]]]:
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
                config, output_manager, logger_name, idx)
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


def process_records_parallel_split(config: Dict[str, Any], start_step: int, end_step: int, max_workers: int,
                           output_manager, logger, log_file_path=None) -> List[Tuple[str, int, str, Dict[str, Any]]]:
    """Process multiple records in parallel."""
    
    logger.info(f"Starting parallel processing with {max_workers} workers")

    path = config.get("output", {}).get("base_dir", "outputs")

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
                    config, output_manager, logger_name, idx)
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


def split_dataset(config: Dict[str, Any], output_manager, logger) -> Dict[str, Any]:

    # Log splitting parameters
    train_ratio = config.get("train_ratio", 0.7)
    validation_ratio = config.get("validation_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.2)

    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:  # Allow for small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
    # Log subject filtering if configured
    exclude_subjects = config.get("exclude_subjects", [])
    include_only_subjects = config.get("include_only_subjects", [])
        
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
    
    observation_files = [f for f in os.listdir(os.path.join(folder_path, folder_name, "observation")) if f.endswith('.csv')]
    for file in observation_files:
        file_path = os.path.join(folder_path, folder_name, "observation", file)
        samples_df = pd.read_csv(file_path)
        labels_df = pd.read_csv(os.path.join(folder_path, folder_name, "prediction", file))
        samples.append(samples_df.values.tolist())
    
    return samples


def create_dataset_pt(path: str, is_train: bool):
    """Create dataset dictionary by reading CSV files in parallel."""
    samples = []
    labels = []

    if is_train:
        path += '/train'
    else:
        path += '/test'
    
    print("Creating pt-dataset from path: ", path)
    # read all folders in the path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Parallelize folder processing using ThreadPoolExecutor (I/O-bound task)
    max_workers = min(os.cpu_count() or 1, len(folders))  # Don't spawn more workers than folders
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



def train_imputer(config: Dict[str, Any], output_manager, logger):
    """Train and save imputer if required."""
    logger.info("Training imputer as per configuration")

    output_path = config.get("output", {}).get("base_dir", "outputs")
    output_path += "/data"
    
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
def run_dataset_creation(config: Dict[str, Any], output_manager, logger, log_file_path=None):
    """Run the complete dataset creation process."""
    logger.info("Starting dataset creation")
    
    # Log configuration
    input_channels = config.get('input_channels', [])
    output_channels = config.get('output_channels', [])
    signal_config = config.get('signal_processing', [])
    windowing_config = config.get('windowing', {})
    
    logger.info(f"Input channels: {input_channels}")
    logger.info(f"Output channels: {output_channels}")
    logger.info(f"Windows: obs={windowing_config.get('observation_window')}s, "
               f"horizon={windowing_config.get('prediction_horizon')}s, "
               f"pred={windowing_config.get('prediction_window')}s")
    
    start_time = time.time()
    
    try:
        # Load record list
        records_df = load_record_list(config, logger)
        
        # Process records with log file path
        max_workers = os.cpu_count() or 1  # Use all available CPU cores, fallback to 8
        logger.info(f"Processing {len(records_df)} records with {max_workers} workers")

        start_step, end_step, max_steps = get_step_for_windowing_and_split(config)
        print(f"Windowing and split between steps {start_step} and {end_step}")

        

        # processing before windowing and split
        results = process_records_parallel_wfdb(records_df, config, start_step, end_step, max_workers, output_manager, logger, log_file_path)
        # Calculate metrics
        processing_time = time.time() - start_time
        successful_records = [r for r in results if r[1] > 0]
        total_samples = sum(r[1] for r in results)
        
        # Log runtime in multiple formats
        hours = int(processing_time // 3600)
        minutes = int((processing_time % 3600) // 60)
        seconds = processing_time % 60
        
        logger.info("="*60)
        logger.info(f"FIRST SIGNAL PROCESSING COMPLETED")
        logger.info(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({processing_time:.2f} seconds)")
        logger.info("="*60)

        # split
        split_results = split_dataset(config, output_manager, logger)
        # Calculate metrics
        processing_time = time.time() - start_time
        successful_records = [r for r in results if r[1] > 0]
        total_samples = sum(r[1] for r in results)
        
        # Log runtime in multiple formats
        hours = int(processing_time // 3600)
        minutes = int((processing_time % 3600) // 60)
        seconds = processing_time % 60
        
        logger.info("="*60)
        logger.info(f"SPLIT COMPLETED")
        logger.info(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({processing_time:.2f} seconds)")
        logger.info("="*60)

        # if imputer should be trained
        if end_step < max_steps:
            if check_if_imputer_needs_to_be_trained(config):
                train_imputer(config, output_manager, logger)

            
        if max_steps > end_step:
            # another round of processing, but load data from disc instead of wfdb
            print("Processing records after split and before windowing")
            results = process_records_parallel_split(config, end_step, max_steps, max_workers, output_manager, logger, log_file_path)

            # Calculate metrics
            processing_time = time.time() - start_time
            successful_records = [r for r in results if r[1] > 0]
            total_samples = sum(r[1] for r in results)
            
            # Log runtime in multiple formats
            hours = int(processing_time // 3600)
            minutes = int((processing_time % 3600) // 60)
            seconds = processing_time % 60
            
            logger.info("="*60)
            logger.info(f"SECOND SIGNAL PROCESSING COMPLETED")
            logger.info(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({processing_time:.2f} seconds)")
            logger.info("="*60)

        # Calculate metrics
        processing_time = time.time() - start_time
        successful_records = [r for r in results if r[1] > 0]
        total_samples = sum(r[1] for r in results)
        
        # Log runtime in multiple formats
        hours = int(processing_time // 3600)
        minutes = int((processing_time % 3600) // 60)
        seconds = processing_time % 60
        
        logger.info("="*60)
        logger.info(f"DATASET CREATION COMPLETED")
        logger.info(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({processing_time:.2f} seconds)")
        logger.info("="*60)
        
        # TODO fix Save reports
        # save_reports(results, processing_time, config, output_manager, logger)
        
        logger.info("Dataset creation completed successfully")

        # for me
        output_path = config.get("output", {}).get("base_dir", "outputs")
        output_path += "/data"
    
        #train_dict = create_dataset_pt(path=output_path, is_train=True)
        #test_dict = create_dataset_pt(path=output_path, is_train=False)
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise


