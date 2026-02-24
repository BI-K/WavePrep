#!/usr/bin/env python3
"""
Abstract Base Classes for Signal Preprocessing

This module defines abstract interfaces for signal downsampling and windowing operations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from common.signal_data import SignalData, WindowedData


class AbstractImputer(ABC):
    """Abstract base class for signal imputation operations."""

    @abstractmethod
    def impute(self, processed_data: 'SignalData', channel_name: str, path: str) -> np.ndarray:
        """
        Impute missing values in a signal channel.

        Args:
            processed_data: Signal data containing all channels
            channel_name: Name of the channel to impute
            path: Path to a fitted imputer model (if needed)

        Returns:
            Channel with imputed values
        """
        pass




class AbstractDownsampler(ABC):
    """Abstract base class for signal downsampling operations."""
    
    @abstractmethod
    def downsample(self, channel: np.ndarray, target_fs: float, original_fs: float) -> List[np.ndarray]:
        """
        Downsample channels to target frequency.
        
        Args:
            channels: channel as numpy arrays
            target_fs: Target sampling frequency
            original_fs: Original sampling frequency
            
        Returns:
            List of downsampled channels
        """
        pass



class AbstractWindower(ABC):
    """Abstract base class for signal windowing operations."""

    @abstractmethod
    def create_windows(
        self,
        data: List['SignalData'],
        observation_window: int,
        prediction_horizon: int,
        prediction_window: int,
        step: int,
        sampling_rate: float
    ) -> 'WindowedData':
        """
        Create observation and prediction windows from signal data.

        Args:
            data: List of signal snippets
            observation_window: Observation window length in seconds
            prediction_horizon: Gap between observation and prediction in seconds
            prediction_window: Prediction window length in seconds
            step: Step size between windows in seconds
            sampling_rate: Sampling rate of the data

        Returns:
            WindowedData with observation/prediction pairs per channel
        """
        pass


