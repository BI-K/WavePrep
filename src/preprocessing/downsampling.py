#!/usr/bin/env python3
"""
Signal Downsampling Implementations

This module provides concrete implementations of signal downsampling strategies.
"""

import numpy as np
from typing import List
from scipy.signal import decimate
from preprocessing import AbstractDownsampler


class SkipDownsampler(AbstractDownsampler):
    """Simple downsampling by skipping samples."""
    
    def downsample(self, channel: np.ndarray, target_fs: float, original_fs: float) -> List[np.ndarray]:
        """
        Downsample by skipping samples.
        
        Args:
            channels: List of signal channels
            target_fs: Target sampling frequency  
            original_fs: Original sampling frequency
            
        Returns:
            List of downsampled channels
        """
        if target_fs >= original_fs:
            return channel
            
        downsample_factor = int(original_fs / target_fs)
        
        downsampled_channel = channel[::downsample_factor]
            
        return downsampled_channel

class MedianDownsampler(AbstractDownsampler):
    """Downsampling by taking local median around target points."""
    
    def downsample(self, channel: np.ndarray, target_fs: float, original_fs: float) -> List[np.ndarray]:
        """
        Downsample by taking median of neighboring samples.
        
        Args:
            channels: List of signal channels
            target_fs: Target sampling frequency
            original_fs: Original sampling frequency
            
        Returns:
            List of downsampled channels
        """
        if target_fs >= original_fs:
            return channel
            
        downsample_factor = int(original_fs / target_fs)

        # keep first entry of array
        #first_entry = channel[0]
        channel = channel [1:]
        trim_len = (len(channel) // downsample_factor) * downsample_factor
        trimmed_channel = channel[:trim_len]
        downsampled_array = np.median(trimmed_channel.reshape(-1, downsample_factor), axis=1)
        #downsampled_array = np.concatenate(([first_entry], downsampled_array))  
        return downsampled_array

class MeanDownsampler(AbstractDownsampler):
    """Downsampling by taking local mean around target points."""
    
    def downsample(self, channel: np.ndarray, target_fs: float, original_fs: float) -> List[np.ndarray]:
        """
        Downsample by taking mean of neighboring samples.
        
        Args:
            channels: List of signal channels
            target_fs: Target sampling frequency
            original_fs: Original sampling frequency
            
        Returns:
            List of downsampled channels
        """
        if target_fs >= original_fs:
            return channel
            
        downsample_factor = int(original_fs / target_fs)
        # keep first entry of array
        #first_entry = channel[0]
        channel = channel [1:]
        trim_len = (len(channel) // downsample_factor) * downsample_factor
        trimmed_channel = channel[:trim_len]
        downsampled_array = np.mean(trimmed_channel.reshape(-1, downsample_factor), axis=1)
        downsampled_array = np.concatenate(([first_entry], downsampled_array))  
        return downsampled_array


class DecimateDownsampler(AbstractDownsampler):
    """Downsampling using scipy's decimate function."""
    
    def downsample(self, channel: np.ndarray, target_fs: float, original_fs: float) -> List[np.ndarray]:
        """
        Downsample using scipy decimate with FIR filter.
        
        Args:
            channels: List of signal channels
            target_fs: Target sampling frequency
            original_fs: Original sampling frequency
            
        Returns:
            List of downsampled channels
        """
        if target_fs >= original_fs:
            return channel
            
        downsample_factor = int(original_fs / target_fs)
        channel = channel[::downsample_factor]  
        return channel


def create_downsampler(strategy: str) -> AbstractDownsampler:
    """
    Factory function to create downsampler instances.
    
    Args:
        strategy: Downsampling strategy ('skip', 'mean', 'decimate')
        
    Returns:
        Downsampler instance
    """

    if strategy == 'skip':
        return SkipDownsampler()
    elif strategy == 'mean':
        return MeanDownsampler()
    elif strategy == 'decimate':
        return DecimateDownsampler()
    elif strategy == 'median':
        return MedianDownsampler()
    else:
        raise ValueError(f"Unknown downsampling strategy: {strategy}")
