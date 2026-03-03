import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from preprocessing.downsampling import create_downsampler
from preprocessing.imputing import create_imputer
from validation.visualize_steps import visualize_step_for_record, visualize_long_nan_removal_for_record
from common.signal_data import SignalData
from common.processing_context import ProcessingContext
from common.pipeline_config import ChannelProcessingConfig, DownsamplingStepConfig
import copy


class DatasetCreationError(Exception):
    """Custom exception for dataset creation errors."""
    pass


def clean_data(channel, lower_threshold, upper_threshold):
    """
    Clean data by removing values outside specified thresholds.
    
    Args:
        channel: Signal channel data as np.ndarray
        lower_threshold: Lower threshold for cleaning
        upper_threshold: Upper threshold for cleaning
        
    Returns:
        Cleaned channel data
    """
    if lower_threshold is not None:
        channel[channel < lower_threshold] = np.nan
    if upper_threshold is not None:
        channel[channel > upper_threshold] = np.nan
    return channel




def merge_intervals(intervals):
    if not intervals:
        return []

    # 1. Sort intervals based on the start index
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]

        # 2. Check for overlap
        # If current start is <= last interval's end, they overlap
        if current_start <= last_end:
            # Merge by updating the end index to the maximum found so far
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # 3. No overlap, just add the interval
            merged.append((current_start, current_end))

    return merged

def remove_long_nan_sequences(step_idx, processed_data_array: List[SignalData], max_consecutive_nans):
    cleaned_processed_data_array = []

    logger_infos = []
    non_nan_sequences_each_data_array = []

    for processed_data in processed_data_array:
        nan_sequences = []

        # get start and end indices of nan sequences longer than max_consecutive_nans
        for channel_name, channel in processed_data.items():
            isnan_mask = np.isnan(channel)
            padded = np.concatenate(([False], isnan_mask, [False]))
            diff = np.diff(padded.astype(int))
            start_indices = np.where(diff == 1)[0]
            # end indices need to be caluclated -1
            end_indices = np.where(diff == -1)[0]
            lengths = end_indices - start_indices
            start_indices = start_indices[lengths > max_consecutive_nans]
            end_indices = end_indices[lengths > max_consecutive_nans]

            if start_indices.size > 0 and end_indices.size > 0:
                nan_sequences.append(zip(start_indices, end_indices))

        sequences = []
        for nan_seq in nan_sequences:
            for start, end in sorted(nan_seq):
                sequences.append([start, end])
        
        merged_sequences = merge_intervals(sequences)

        
        for i in range(len(merged_sequences)):
            logger_infos.append(f"Removed long NaN sequence: step={step_idx}, start={merged_sequences[i][0]}, end={merged_sequences[i][1]}")


        if merged_sequences:
            # caluclate non-nan sequences from merged nan sequences
            non_nan_sequences = [(0, merged_sequences[0][0])]
            for i in range(len(merged_sequences) - 1):
                non_nan_sequences.append((merged_sequences[i][1], merged_sequences[i + 1][0]))
                
            non_nan_sequences.append((merged_sequences[-1][1], processed_data.n_samples))

            non_nan_sequences_each_data_array.append(non_nan_sequences)
            for start, end in non_nan_sequences:
                if end - start <= 0:
                    continue
                channels = {name: data[start:end] for name, data in processed_data.items()}
                cleaned_processed_data_array.append(SignalData(channels=channels))
        else:
            cleaned_processed_data_array.append(processed_data)
            

    return cleaned_processed_data_array, logger_infos, non_nan_sequences_each_data_array

        


def downsample_record(channel: np.ndarray, config: DownsamplingStepConfig, current_fs: float) -> Tuple[np.ndarray, float]:
    downsampler = create_downsampler(config.strategy)
    channel = downsampler.downsample(channel, config.desired_resolution, current_fs)
    return channel, config.desired_resolution


def perform_signal_processing(
        filtered_data: np.ndarray,
        filtered_names: List[str],
        signal_processing: List[ChannelProcessingConfig],
        start_at_processing_step: int,
        process_until_step: int,
        metadata: ProcessingContext,
        long_nan_removal_config: Dict[str, Any] = None,
        logger: logging.Logger = None,
        records_to_visualize = []
    ) -> Tuple[List[SignalData], List[str]]:
    """
    Perform signal processing on the filtered data.

    Args:
        filtered_data: Filtered signal data (samples x channels)
        filtered_names: Names of the channels in the filtered data
        signal_processing: Per-channel processing step configuration
        start_at_processing_step: First processing step index to execute
        process_until_step: Stop before this step index
        metadata: Processing context with sampling rate, record ID, and paths
        long_nan_removal_config: Config for removing long NaN sequences
        logger: Logger instance
        records_to_visualize: Record IDs selected for step visualization

    Returns:
        Tuple of (processed signal snippets, log messages)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger_infos = []

    # Build initial SignalData from the filtered numpy array
    channels = {}
    filtered_names_list = list(filtered_names)
    for item in signal_processing:
        if item.channel not in filtered_names_list:
            raise DatasetCreationError(f"Channel {item.channel} not found in filtered names: {filtered_names_list}")
        channel_idx = filtered_names_list.index(item.channel)
        channels[item.channel] = np.array(filtered_data[:, channel_idx])
    processed_data_array: List[SignalData] = [SignalData(channels=channels)]



    long_nan_removal_config_dict = {}
    for item in long_nan_removal_config or []:
        after_step = item.get("after_step")
        max_consecutive_nans = item.get("max_consecutive_nans")
        long_nan_removal_config_dict[after_step] = max_consecutive_nans


    # process step by step
    max_steps = max(len(ch.steps) for ch in signal_processing)
    current_fs = {ch_name: metadata.sampling_rate for ch_name in filtered_names}

    for step_idx in range(start_at_processing_step, process_until_step):

        processed_data_array_copy = copy.deepcopy(processed_data_array)
        for channel_name in filtered_names:

            ch_cfg = next((item for item in signal_processing if item.channel == channel_name), None)
            if not ch_cfg:
                continue

            current_step = next((s for s in ch_cfg.steps if s.step == step_idx), None)
            if not current_step:
                continue

            # Resolve factories once per step, not per data-array element
            downsampler = create_downsampler(current_step.downsampling.strategy) if current_step.downsampling else None
            imputer = create_imputer(current_step.imputation.method) if current_step.imputation else None

            for i in range(len(processed_data_array)):
                channel = processed_data_array[i][channel_name]

                if current_step.downsampling:
                    channel = downsampler.downsample(channel, current_step.downsampling.desired_resolution, current_fs[channel_name])
                    current_fs[channel_name] = current_step.downsampling.desired_resolution

                if current_step.data_cleaning:
                    channel = clean_data(channel, current_step.data_cleaning.lower_threshold, current_step.data_cleaning.upper_threshold)

                if imputer:
                    channel = imputer.impute(processed_data_array[i], channel_name, metadata.imputer_path)

                processed_data_array[i][channel_name] = channel


        if metadata.record_id in records_to_visualize:
            visualize_step_for_record(
                record_id=metadata.record_id,
                step_id=step_idx,
                processed_data_array_before=processed_data_array_copy,
                processed_data_array_after=processed_data_array,
                signal_processing_config=signal_processing,
                output_path=metadata.output_path_process_images,
                windowing_config=metadata.windowing_config
            )


        # check if a remove long nan sequence step needs to be applied
        if long_nan_removal_config_dict.get(step_idx) is not None:
            processed_data_array_copy = copy.deepcopy(processed_data_array)
            max_consecutive_nans = long_nan_removal_config_dict.get(step_idx)
            processed_data_array, new_logger_infos, non_nan_sequences = remove_long_nan_sequences(step_idx, processed_data_array, max_consecutive_nans)
            logger_infos.extend(new_logger_infos)

            intermediate_processed_data_array = copy.deepcopy(processed_data_array)
            processed_data_array = []
            for i in range(len(intermediate_processed_data_array)):
                processed_data = intermediate_processed_data_array[i]
                if processed_data.n_samples >= current_fs[filtered_names[0]] * metadata.min_record_duration:
                    processed_data_array.append(processed_data)
                else:
                    logger_infos.append(f"Removed processed data due to insufficient duration: {processed_data.n_samples / current_fs[filtered_names[0]]}s")
            

            if metadata.record_id in records_to_visualize:
                visualize_long_nan_removal_for_record(
                    record_id=metadata.record_id,
                    step_id=step_idx,
                    processed_data_array_before=processed_data_array_copy,
                    processed_data_array_after=processed_data_array,
                    non_nan_sequences=non_nan_sequences,
                    min_required_length=metadata.min_record_duration * current_fs[filtered_names[0]],
                    output_path=metadata.output_path_process_images
                )


        if len(processed_data_array) == 0:
            step_idx = max_steps  # to exit the loop

    return processed_data_array, logger_infos