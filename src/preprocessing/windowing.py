#!/usr/bin/env python3
"""
Signal Windowing Implementation

This module provides windowing functionality for creating time series samples.
"""

import numpy as np
from typing import List
from preprocessing import AbstractWindower
from common.signal_data import SignalData, WindowedData, Window


class TimeWindowProcessor(AbstractWindower):
    """Implementation for creating time-based windows from signals."""

    def create_windows(
        self,
        data: List[SignalData],
        observation_window: int,
        prediction_horizon: int,
        prediction_window: int,
        step: int,
        sampling_rate: float
    ) -> WindowedData:
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
            WindowedData containing observation/prediction pairs per channel
        """
        obs_samples = int(np.ceil(observation_window * sampling_rate))
        horizon_samples = int(np.ceil(prediction_horizon * sampling_rate))
        pred_samples = int(np.ceil(prediction_window * sampling_rate))
        step_samples = int(np.ceil(step * sampling_rate))
        total_window_size = obs_samples + horizon_samples + pred_samples

        windowed = WindowedData()
        channel_names = data[0].channel_names
        for channel_name in channel_names:
            windowed[channel_name] = []
            for snippet in data:
                channel = snippet[channel_name]

                n_samples = len(channel)
                if n_samples < total_window_size:
                    continue

                start_idx = 0
                while start_idx + total_window_size <= n_samples:
                    obs_end = start_idx + obs_samples
                    obs_data = channel[start_idx:obs_end]

                    pred_start = obs_end + horizon_samples
                    pred_end = pred_start + pred_samples
                    pred_data = channel[pred_start:pred_end]

                    windowed[channel_name].append(Window(obs_data, pred_data))

                    start_idx += step_samples

        return windowed
    


def create_windower() -> AbstractWindower:
    """
    Factory function to create windower instance.
    
    Returns:
        Windower instance
    """
    return TimeWindowProcessor()
