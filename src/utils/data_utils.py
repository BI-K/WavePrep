"""
Data utility functions for preprocessing and handling data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_split_data(split_data_dir: Path) -> Dict[str, List[Path]]:
    """
    Load file paths for each split.
    
    Args:
        split_data_dir: Directory containing split data
        
    Returns:
        Dictionary mapping split names to lists of CSV file paths
    """
    splits = {}
    
    for split in ['train', 'validation', 'test']:
        split_dir = split_data_dir / split
        if split_dir.exists():
            csv_files = []
            for subject_dir in split_dir.iterdir():
                if subject_dir.is_dir():
                    csv_files.extend(list(subject_dir.glob("*.csv")))
            splits[split] = csv_files
        else:
            splits[split] = []
    
    return splits


def normalize_features(data: np.ndarray, 
                      method: str = 'standard',
                      fitted_scaler: Optional[object] = None) -> Tuple[np.ndarray, object]:
    """
    Normalize features using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('standard' or 'minmax')
        fitted_scaler: Pre-fitted scaler (for validation/test sets)
        
    Returns:
        Normalized data and fitted scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if fitted_scaler is not None:
        # Use pre-fitted scaler
        normalized_data = fitted_scaler.transform(data)
        return normalized_data, fitted_scaler
    else:
        # Fit new scaler
        normalized_data = scaler.fit_transform(data)
        return normalized_data, scaler


def create_sequences(data: np.ndarray, 
                    sequence_length: int,
                    target_column_idx: int = -1,
                    prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction.
    
    Args:
        data: Input time-series data
        sequence_length: Length of input sequences
        target_column_idx: Index of target column
        prediction_horizon: How many steps ahead to predict
        
    Returns:
        Sequences and targets
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Input sequence
        sequence = data[i:i + sequence_length]
        # Target value (prediction_horizon steps ahead)
        target = data[i + sequence_length + prediction_horizon - 1, target_column_idx]
        
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def load_and_preprocess_csv(csv_path: Path, 
                           feature_columns: List[str],
                           target_column: str = 'HR') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        feature_columns: List of feature column names
        target_column: Name of target column
        
    Returns:
        Features and targets as numpy arrays
    """
    df = pd.read_csv(csv_path)
    
    # Check if all required columns exist
    missing_cols = set(feature_columns + [target_column]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
    
    # Extract features and target
    features = df[feature_columns].values
    target = df[target_column].values
    
    return features, target


def calculate_data_statistics(data_loaders: Dict) -> Dict[str, Dict]:
    """
    Calculate statistics for each split.
    
    Args:
        data_loaders: Dictionary of data loaders
        
    Returns:
        Dictionary with statistics for each split
    """
    stats = {}
    
    for split_name, loader in data_loaders.items():
        total_samples = len(loader.dataset)
        
        # Calculate feature statistics from first batch
        if total_samples > 0:
            sample_batch = next(iter(loader))
            sequences, targets = sample_batch
            
            stats[split_name] = {
                'total_samples': total_samples,
                'sequence_length': sequences.shape[1],
                'num_features': sequences.shape[2],
                'target_mean': float(targets.mean()),
                'target_std': float(targets.std()),
                'batches': len(loader)
            }
        else:
            stats[split_name] = {
                'total_samples': 0,
                'sequence_length': 0,
                'num_features': 0,
                'target_mean': 0.0,
                'target_std': 0.0,
                'batches': 0
            }
    
    return stats
