"""
Data loading utilities for MIMIC III dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MIMICDataLoader(Dataset):
    """
    PyTorch Dataset for loading MIMIC III time-series data.
    """
    
    def __init__(self, 
                 data_dir: Path, 
                 split: str = 'train',
                 sequence_length: int = 60,
                 feature_columns: List[str] = None,
                 target_column: str = 'HR',
                 normalize: bool = True):
        """
        Initialize the MIMIC data loader.
        
        Args:
            data_dir: Path to the split data directory
            split: Split to load ('train', 'validation', 'test')
            sequence_length: Length of input sequences
            feature_columns: List of feature columns to use
            target_column: Target column for prediction
            normalize: Whether to normalize features
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Default feature columns
        if feature_columns is None:
            self.feature_columns = ['SpO2', 'RESP', 'HR']
        else:
            self.feature_columns = feature_columns
            
        self.target_column = target_column
        
        # Load and preprocess data
        self.sequences, self.targets = self._load_data()
        
        if self.normalize:
            self.scaler = StandardScaler()
            self.sequences = self._normalize_features(self.sequences)
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process CSV files into sequences."""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        all_sequences = []
        all_targets = []
        
        # Iterate through subject directories
        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            # Load all CSV files for this subject
            csv_files = list(subject_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                
                # Extract features and targets
                if all(col in df.columns for col in self.feature_columns):
                    features = df[self.feature_columns].values
                    
                    # Create sequences
                    sequences, targets = self._create_sequences(features)
                    all_sequences.extend(sequences)
                    all_targets.extend(targets)
        
        return np.array(all_sequences), np.array(all_targets)
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[List, List]:
        """Create sequences from time-series data."""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length, self.feature_columns.index(self.target_column)]
            
            sequences.append(sequence)
            targets.append(target)
            
        return sequences, targets
    
    def _normalize_features(self, sequences: np.ndarray) -> np.ndarray:
        """Normalize features using StandardScaler."""
        # Reshape for scaling
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        
        # Fit and transform
        sequences_normalized = self.scaler.fit_transform(sequences_flat)
        
        # Reshape back
        return sequences_normalized.reshape(original_shape)
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and target."""
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        
        return sequence, target


def load_split_data(data_dir: Path, 
                   batch_size: int = 32, 
                   sequence_length: int = 60,
                   feature_columns: List[str] = None,
                   target_column: str = 'HR',
                   device: str = 'cpu') -> Dict[str, torch.utils.data.DataLoader]:
    """
    Load train, validation, and test data loaders.
    
    Args:
        data_dir: Path to the split data directory  
        batch_size: Batch size for data loaders
        sequence_length: Length of input sequences
        feature_columns: List of feature columns to use
        target_column: Target column for prediction
        device: Device to determine pin_memory setting
        
    Returns:
        Dictionary with train, validation, and test data loaders
    """
    # Pin memory only for CUDA devices, not for MPS
    pin_memory = device == 'cuda'
    
    datasets = {}
    data_loaders = {}
    
    for split in ['train', 'validation', 'test']:
        datasets[split] = MIMICDataLoader(
            data_dir=data_dir,
            split=split,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
            target_column=target_column
        )
        
        data_loaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=0,  # Use main process only to avoid hanging
            pin_memory=pin_memory
        )
    
    return data_loaders
