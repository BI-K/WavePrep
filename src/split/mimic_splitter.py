#!/usr/bin/env python3
"""
MIMIC III Dataset Splitter

Professional group-based dataset splitting for MIMIC III datasets.
Ensures no data leakage between subjects across train/validation/test splits.
"""

import json
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .splitter_exceptions import (
    SplittingError, ConfigurationError, DataValidationError, 
    InsufficientDataError, SplitRatioError
)

# Suppress warnings
warnings.filterwarnings('ignore')


def validate_config(config: Dict[str, Any]):
    """Validate the configuration parameters."""
    required_fields = [
        'train_ratio', 'validation_ratio',
        'test_ratio', 'random_seed'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required configuration field: {field}")
    
    # Validate ratios sum to 1.0
    total_ratio = (config['train_ratio'] + 
                  config['validation_ratio'] + 
                  config['test_ratio'])
    if abs(total_ratio - 1.0) > 0.001:
        raise SplitRatioError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Validate all ratios are positive
    for ratio_name in ['train_ratio', 'validation_ratio', 'test_ratio']:
        ratio = config[ratio_name]
        #if ratio <= 0:
        #    raise SplitRatioError(f"{ratio_name} must be positive, got {ratio}")


def set_random_seeds(config: Dict[str, Any], logger):
    """Set random seeds for reproducibility."""
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")


def resolve_latest_input_path(input_path: str, logger) -> Path:
    """
    Resolve 'latest' in input path to the most recent creation run.
    
    Args:
        input_path: Input path that may contain 'latest'
        
    Returns:
        Resolved path to the actual directory
    """
    path = Path(input_path)
    
    # If path contains 'latest', resolve it
    if 'latest' in str(path):
        # Look for create output directories
        create_dir = Path("outputs/create")
        if not create_dir.exists():
            raise DataValidationError(f"Create output directory not found: {create_dir}")
        
        # Find the most recent create run
        create_runs = [d for d in create_dir.iterdir() if d.is_dir() and d.name.startswith('create_')]
        if not create_runs:
            raise DataValidationError("No create runs found in outputs/create")
        
        # Sort by name (which includes timestamp)
        latest_run = sorted(create_runs)[-1]
        
        # Replace 'latest' with the actual directory name
        resolved_path = Path(str(path).replace('latest', latest_run.name))
        logger.info(f"Resolved 'latest' to: {resolved_path}")
        return resolved_path
    
    return path


def create_output_directory(config: Dict[str, Any], logger) -> Path:
    """Create timestamped output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"split_{timestamp}"
    
    base_dir = Path(config.get('output', {}).get('base_dir', 'outputs/split'))
    output_dir = base_dir / run_name
    
    # Create directory structure from config
    output_config = config.get('output', {})
    directories = output_config.get('directory_structure', ['logs', 'reports', 'split_data', 'splits'])
    
    for dir_name in directories:
        (output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create split data subdirectories - use the actual directory name from config
    split_data_dir = None
    for dir_name in directories:
        if 'data' in dir_name.lower():  # Find the data directory (could be 'data', 'split_data', etc.)
            split_data_dir = dir_name
            break
    
    if not split_data_dir:
        split_data_dir = 'split_data'  # fallback
        (output_dir / split_data_dir).mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'validation', 'test']:
        (output_dir / split_data_dir / split_name).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory structure at: {output_dir}")
    logger.info(f"Using data directory: {split_data_dir}")
    return output_dir


def discover_subjects_and_samples(input_path: Path, logger) -> Dict[str, Dict[str, Any]]:
    """
    Discover all subjects and their samples in the input directory.
    
    Args:
        input_path: Path to the directory containing subject folders with CSV files
        
    Returns:
        Dictionary with subject_id as key and metadata as value
    """
    logger.info("Discovering subjects and samples...")
    
    if not input_path.exists():
        raise DataValidationError(f"Input path does not exist: {input_path}")
    
    subjects_data = {}
    
    # Look for subject directories
    subject_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not subject_dirs:
        raise DataValidationError(f"No subject directories found in: {input_path}")
    
    logger.info(f"Found {len(subject_dirs)} subject directories")
    
    for subject_dir in tqdm(subject_dirs, desc="Scanning subjects", leave=False):
        subject_id = subject_dir.name
        
        # Find CSV files in subject directory
        subject_observation_window_dir = subject_dir / "observation"
        subject_preditction_window_dir = subject_dir / "prediction"
        csv_files = list(subject_observation_window_dir.glob("*.csv"))
        
        if csv_files:
            # Get total file size for this subject
            total_size = sum(f.stat().st_size for f in csv_files)
            
            subjects_data[subject_id] = {
                'subject_id': subject_id,
                'total_samples': len(csv_files),
                'sample_files': [f.name for f in csv_files],
                'total_size_bytes': total_size,
                'directory_path': str(subject_dir)  # Convert to string for JSON serialization
            }
    
    logger.info(f"Discovered {len(subjects_data)} subjects with data")
    
    if len(subjects_data) == 0:
        raise DataValidationError("No subjects with CSV files found")
    
    return subjects_data


def filter_subjects(subjects_data: Dict[str, Dict[str, Any]], config: Dict[str, Any], logger) -> Dict[str, Dict[str, Any]]:
    """
    Filter subjects based on configuration criteria.
    
    Args:
        subjects_data: Dictionary of subject data
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Filtered subjects dictionary
    """
    logger.info("Filtering subjects based on criteria...")
    
    filtered_subjects = {}
    
    # Apply minimum samples filter
    min_samples = config.get('min_samples_per_subject', 1)
    
    # Apply include/exclude filters
    exclude_subjects = set(config.get('exclude_subjects', []))
    include_only = config.get('include_only_subjects', [])
    include_only_set = set(include_only) if include_only else None
    
    excluded_count = 0
    min_samples_filtered = 0
    
    for subject_id, data in subjects_data.items():
        # Check exclude list
        if subject_id in exclude_subjects:
            excluded_count += 1
            continue
        
        # Check include-only list
        if include_only_set and subject_id not in include_only_set:
            excluded_count += 1
            continue
        
        # Check minimum samples
        if data['total_samples'] < min_samples:
            min_samples_filtered += 1
            continue
        
        filtered_subjects[subject_id] = data
    
    logger.info(f"Filtering results:")
    logger.info(f"  • Original subjects: {len(subjects_data)}")
    logger.info(f"  • Excluded by filters: {excluded_count}")
    logger.info(f"  • Below minimum samples: {min_samples_filtered}")
    logger.info(f"  • Final subjects: {len(filtered_subjects)}")
    
    if len(filtered_subjects) < 3:
        raise InsufficientDataError(f"Insufficient subjects for splitting: {len(filtered_subjects)}")
    
    return filtered_subjects


def perform_group_shuffle_split(subjects_data: Dict[str, Dict[str, Any]], config: Dict[str, Any], logger) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Split subjects using scikit-learn's GroupShuffleSplit for group-aware splitting.
    
    This function ensures that all samples from the same subject (group) remain in the 
    same split, which is ideal for medical data where samples from the same patient
    should not be distributed across training and testing sets.
    
    Args:
        subjects_data: Dictionary of subject data
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (split_assignments, split_sample_counts)
    """
    logger.info("Performing group-aware subject splitting using GroupShuffleSplit...")
    
    subjects = list(subjects_data.keys())
    subject_samples = {subj: subjects_data[subj]['total_samples'] for subj in subjects}
    
    # For GroupShuffleSplit, we need to create arrays where each element represents
    # a "sample" but the group identifies which subject it belongs to
    X_indices = []  # Sample indices (not really used, just for sklearn interface)
    groups = []     # Subject IDs for each sample
    
    sample_idx = 0
    for subject_id in subjects:
        num_samples = subject_samples[subject_id]
        X_indices.extend(range(sample_idx, sample_idx + num_samples))
        groups.extend([subject_id] * num_samples)
        sample_idx += num_samples
    
    X_indices = np.array(X_indices)
    groups = np.array(groups)
    
    logger.info(f"Created group arrays: {len(X_indices)} total sample points across {len(subjects)} subjects")
    
    # Calculate test_size for GroupShuffleSplit
    # GroupShuffleSplit's test_size refers to the proportion of groups (subjects), not samples
    test_size = config['validation_ratio'] + config['test_ratio']
    val_size = config['validation_ratio'] / test_size if test_size > 0 else 0
    
    # First split: separate training from (validation + test)
    gss_train_test = GroupShuffleSplit(
        n_splits=1, 
        test_size=test_size,
        random_state=config['random_seed']
    )
    
    train_indices, temp_indices = next(gss_train_test.split(X_indices, groups=groups))
    
    # Get subjects in each split
    train_subjects = list(set(groups[train_indices]))
    temp_subjects = list(set(groups[temp_indices]))
    
    logger.info(f"Initial split: {len(train_subjects)} train subjects, {len(temp_subjects)} temp subjects")
    
    # Second split: separate validation from test within the temp set
    if len(temp_subjects) > 1 and config['validation_ratio'] > 0:
        # Create arrays for temp subjects only
        temp_X = []
        temp_groups = []
        temp_idx = 0
        
        for subject_id in temp_subjects:
            num_samples = subject_samples[subject_id]
            temp_X.extend(range(temp_idx, temp_idx + num_samples))
            temp_groups.extend([subject_id] * num_samples)
            temp_idx += num_samples
        
        temp_X = np.array(temp_X)
        temp_groups = np.array(temp_groups)
        
        # Split temp into validation and test
        gss_val_test = GroupShuffleSplit(
            n_splits=1,
            test_size=1 - val_size,  # test_size as proportion of temp subjects
            random_state=config['random_seed'] + 1
        )
        
        val_indices, test_indices = next(gss_val_test.split(temp_X, groups=temp_groups))
        val_subjects = list(set(temp_groups[val_indices]))
        test_subjects = list(set(temp_groups[test_indices]))
    else:
        # If only one temp subject or no validation needed, assign all to test
        val_subjects = []
        test_subjects = temp_subjects
    
    logger.info(f"Final split: {len(train_subjects)} train, {len(val_subjects)} validation, {len(test_subjects)} test subjects")
    
    # Create final split assignment
    splits = {
        'train': train_subjects,
        'validation': val_subjects,
        'test': test_subjects
    }
    
    # Calculate sample counts
    split_samples = {'train': 0, 'validation': 0, 'test': 0}
    for split_name, subject_list in splits.items():
        split_samples[split_name] = sum(subject_samples[subj] for subj in subject_list)
    
    # Log results
    total_samples = sum(split_samples.values())
    for split_name in ['train', 'validation', 'test']:
        subjects_count = len(splits[split_name])
        samples_count = split_samples[split_name]
        samples_pct = (samples_count / total_samples * 100) if total_samples > 0 else 0
        logger.info(f"{split_name.capitalize()}: {subjects_count} subjects, {samples_count:,} samples ({samples_pct:.1f}%)")
    
    return splits, split_samples


def copy_split_data(splits: Dict[str, List[str]], input_path: Path, output_dir: Path, config: Dict[str, Any], logger):
    """
    Copy actual data files (CSV files) to split directories for direct ML training.
    
    Args:
        splits: Dictionary with split assignments (subject IDs)
        input_path: Path to original data directory
        output_dir: Output directory path
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Copying data files to split directories...")
    
    # Find the data directory name from config
    output_config = config.get('output', {})
    directories = output_config.get('directory_structure', ['logs', 'reports', 'split_data', 'splits'])
    
    split_data_dir_name = None
    for dir_name in directories:
        if 'data' in dir_name.lower():  # Find the data directory
            split_data_dir_name = dir_name
            break
    
    if not split_data_dir_name:
        split_data_dir_name = 'split_data'  # fallback
    
    split_data_dir = output_dir / split_data_dir_name
    
    total_files_copied = 0
    
    for split_name, subjects in splits.items():
        if not subjects:  # Skip empty splits
            continue
            
        split_dir = split_data_dir / split_name
        logger.info(f"Copying {len(subjects)} subjects to {split_name} split...")
        
        for subject_id in tqdm(subjects, desc=f"Copying {split_name}", leave=False):
            source_dir = input_path / subject_id
            target_dir = split_dir / subject_id
            
            if source_dir.exists():
                # Copy entire subject directory
                shutil.move(source_dir, target_dir)
                
                # Count files copied
                csv_files = list(target_dir.glob("*.csv"))
                total_files_copied += len(csv_files)

            else:
                logger.warning(f"Source directory not found: {source_dir}")
    
    logger.info(f"Total files copied: {total_files_copied}")
    logger.info(f"Files copied to: {split_data_dir}")


def save_splits(splits: Dict[str, List[str]], split_samples: Dict[str, int], output_dir: Path, config: Dict[str, Any], logger):
    """Save split assignments to JSON files."""
    splits_dir = output_dir / 'splits'
    
    # Save individual split files
    for split_name, subjects in splits.items():
        split_file = splits_dir / f"{split_name}_subjects.json"
        with open(split_file, 'w') as f:
            json.dump({
                'split_name': split_name,
                'subjects': subjects,
                'num_subjects': len(subjects),
                'num_samples': split_samples[split_name]
            }, f, indent=2)
    
    # Save combined splits file
    combined_file = splits_dir / 'all_splits.json'
    with open(combined_file, 'w') as f:
        json.dump({
            'splits': splits,
            'sample_counts': split_samples,
            'metadata': {
                'total_subjects': sum(len(subjects) for subjects in splits.values()),
                'total_samples': sum(split_samples.values()),
                'split_ratios': {
                    'train_ratio': config['train_ratio'],
                    'validation_ratio': config['validation_ratio'],
                    'test_ratio': config['test_ratio']
                },
                'random_seed': config['random_seed'],
                'created_at': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    logger.info("Split assignments saved successfully")


def generate_markdown_report(splits: Dict[str, List[str]], split_samples: Dict[str, int], subjects_data: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> str:
    """Generate markdown report content."""
    total_subjects = sum(len(subjects) for subjects in splits.values())
    total_samples = sum(split_samples.values())
    
    lines = [
        "# Dataset Splitting Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- **Total Subjects:** {total_subjects}",
        f"- **Total Samples:** {total_samples:,}",
        f"- **Splitting Method:** Group-based shuffle split",
        f"- **Random Seed:** {config['random_seed']}",
        "",
        "## Split Distribution",
        ""
    ]
    
    for split_name in ['train', 'validation', 'test']:
        subjects_count = len(splits[split_name])
        samples_count = split_samples[split_name]
        subject_pct = (subjects_count / total_subjects * 100) if total_subjects > 0 else 0
        sample_pct = (samples_count / total_samples * 100) if total_samples > 0 else 0
        target_pct = config[f'{split_name}_ratio'] * 100
        
        lines.extend([
            f"### {split_name.capitalize()} Set",
            f"- **Subjects:** {subjects_count} ({subject_pct:.1f}% of total)",
            f"- **Samples:** {samples_count:,} ({sample_pct:.1f}% of total)",
            f"- **Target Ratio:** {target_pct:.1f}%",
            f"- **Actual Ratio:** {sample_pct:.1f}%",
            ""
        ])
    
    # Add subject distribution analysis
    lines.extend([
        "## Subject Sample Distribution",
        ""
    ])
    
    subject_sample_counts = []
    for subjects in splits.values():
        for subject_id in subjects:
            if subject_id in subjects_data:
                subject_sample_counts.append(subjects_data[subject_id]['total_samples'])
    
    if subject_sample_counts:
        lines.extend([
            f"- **Mean samples per subject:** {np.mean(subject_sample_counts):.1f}",
            f"- **Median samples per subject:** {np.median(subject_sample_counts):.1f}",
            f"- **Min samples per subject:** {min(subject_sample_counts)}",
            f"- **Max samples per subject:** {max(subject_sample_counts)}",
            ""
        ])
    
    lines.extend([
        "## Configuration",
        f"- **Input Path:** {config.get("output", {}).get("base_dir") + "/data"}",
        f"- **Train Ratio:** {config['train_ratio']}",
        f"- **Validation Ratio:** {config['validation_ratio']}",
        f"- **Test Ratio:** {config['test_ratio']}",
        ""
    ])
    
    return '\n'.join(lines)


def generate_detailed_analysis(splits: Dict[str, List[str]], split_samples: Dict[str, int], subjects_data: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed analysis of the splitting results."""
    # Convert Path objects to strings for JSON serialization
    def convert_paths_to_strings(obj):
        if isinstance(obj, dict):
            return {k: convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    serializable_config = convert_paths_to_strings(config)
    serializable_subjects_data = convert_paths_to_strings(subjects_data)
    
    analysis = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'config': serializable_config,
            'total_subjects': sum(len(subjects) for subjects in splits.values()),
            'total_samples': sum(split_samples.values())
        },
        'splits_summary': {},
        'subjects_data': serializable_subjects_data
    }
    
    for split_name, subjects in splits.items():
        analysis['splits_summary'][split_name] = {
            'subjects': subjects,
            'num_subjects': len(subjects),
            'num_samples': split_samples[split_name],
            'subject_sample_distribution': [
                subjects_data[subj]['total_samples'] 
                for subj in subjects if subj in subjects_data
            ]
        }
    
    return analysis


def generate_reports(splits: Dict[str, List[str]], split_samples: Dict[str, int], subjects_data: Dict[str, Dict[str, Any]], output_dir: Path, config: Dict[str, Any], logger):
    """Generate comprehensive reports about the splitting process."""
    reports_dir = output_dir / 'reports'
    
    # Generate markdown report
    report_content = generate_markdown_report(splits, split_samples, subjects_data, config)
    report_file = reports_dir / 'splitting_report.md'
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    # Generate detailed analysis
    analysis = generate_detailed_analysis(splits, split_samples, subjects_data, config)
    analysis_file = reports_dir / 'detailed_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info("Reports generated successfully")


def split_dataset(config: Dict[str, Any], logger, output_dir: Path = None) -> Dict[str, Any]:
    """
    Perform the complete dataset splitting process.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        output_dir: Optional output directory (if None, will create new one)
        
    Returns:
        Dictionary containing split results and statistics
    """
    logger.info("Starting dataset splitting process...")
    
    # Validate configuration
    validate_config(config)
    
    # Set random seeds for reproducibility
    set_random_seeds(config, logger)
    
    # Resolve input path
    input_path = Path(config.get("output", {}).get("base_dir") + "/data")
    print(f"Input path before resolving latest: {input_path}")
    
    # Discover subjects and samples
    subjects_data = discover_subjects_and_samples(input_path, logger)
    
    # Filter subjects based on criteria
    subjects_data = filter_subjects(subjects_data, config, logger)
    
    # Perform the splitting
    splits, split_samples = perform_group_shuffle_split(subjects_data, config, logger)
    
    # Use provided output directory or create new one
    if output_dir is None:
        output_dir = create_output_directory(config, logger)
    else:
        # Ensure the necessary subdirectories exist in the provided output directory
        output_config = config.get('output', {})
        directories = output_config.get('directory_structure', ['logs', 'reports', 'split_data', 'splits'])
        
        for dir_name in directories:
            (output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create split data subdirectories
        split_data_dir_name = next((d for d in directories if 'data' in d.lower()), 'split_data')
        for split_name in ['train', 'validation', 'test']:
            (output_dir / split_data_dir_name / split_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using existing output directory: {output_dir}")
    
    # Copy data files to split directories
    if not config.get('dry_run', False):
        copy_split_data(splits, input_path, output_dir, config, logger)
    
    # Save split assignments and generate reports
    save_splits(splits, split_samples, output_dir, config, logger)
    generate_reports(splits, split_samples, subjects_data, output_dir, config, logger)
    
    # Return results
    results = {
        'train_subjects': len(splits['train']),
        'validation_subjects': len(splits['validation']),
        'test_subjects': len(splits['test']),
        'train_samples': split_samples['train'],
        'validation_samples': split_samples['validation'],
        'test_samples': split_samples['test'],
        'total_subjects': sum(len(subjects) for subjects in splits.values()),
        'total_samples': sum(split_samples.values()),
        'splits': splits,
        'output_dir': str(output_dir),
        'input_path': str(input_path)
    }
    
    logger.info("Dataset splitting completed successfully")
    return results


def run_dataset_splitting(config: Dict[str, Any], output_manager, logger):
    """
    Run the complete dataset splitting process.
    
    Args:
        config: Configuration dictionary
        output_manager: Output manager instance
        logger: Logger instance
    """
    logger.info("Starting dataset splitting")
    
    # Log configuration
    input_path = config.get("output", {}).get("base_dir") + "/data"
    train_ratio = config.get('train_ratio')
    validation_ratio = config.get('validation_ratio')
    test_ratio = config.get('test_ratio')
    random_seed = config.get('random_seed', 42)
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Split ratios - Train: {train_ratio}, Validation: {validation_ratio}, Test: {test_ratio}")
    logger.info(f"Random seed: {random_seed}")
    
    try:
        # Get the output directory from the output manager
        output_dir = output_manager.run_dir

        # Perform splitting using function-based approach with existing output directory
        results = split_dataset(config, logger, output_dir)
        
        # Log results
        logger.info(f"Split completed successfully:")
        logger.info(f"  - Train: {results['train_subjects']} subjects, {results['train_samples']:,} samples")
        logger.info(f"  - Validation: {results['validation_subjects']} subjects, {results['validation_samples']:,} samples")
        logger.info(f"  - Test: {results['test_subjects']} subjects, {results['test_samples']:,} samples")
        logger.info(f"Output saved to: {results['output_dir']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        raise
