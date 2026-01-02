import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from preprocessing.downsampling import create_downsampler
from preprocessing.imputing import create_imputer
from validation.visualize_steps import visualize_step_for_record, visualize_long_nan_removal_for_record
import copy


class DatasetCreationError(Exception):
    """Custom exception for dataset creation errors."""
    pass


def get_channel_signal_from_array(data, channel_name, channel_names):
    # get channel signal
    if channel_name not in channel_names:
            raise DatasetCreationError(f"Channel {channel_name} not found in filtered names: {channel_names}")

    # channel_index = channel_names.index(channel_name)
    channel_index = next((i for i, name in enumerate(channel_names) if name == channel_name), None)
    if channel_index is None:
        raise DatasetCreationError(f"Channel {channel_name} not found in filtered names: {channel_names}")
    channel = data[:, channel_index]
    # turn into np.array
    channel = np.array(channel)
    return channel


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



def remove_long_nan_sequences(step_idx, processed_data_array, max_consecutive_nans):
    # check consecutive nans in each channel
    # channel, start_idx, end_idx

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

        merged_sequences = []

        for nan_seq in nan_sequences:
            # merge overlapping sequences
            for start, end in sorted(nan_seq):
                merged_sequences.append([start, end])
        
        for i in range(len(merged_sequences)):
            logger_infos.append(f"Removed long NaN sequence: step={step_idx}, start={merged_sequences[i][0]}, end={merged_sequences[i][1]}")


        if merged_sequences:
            # caluclate non-nan sequences from merged nan sequences
            non_nan_sequences = [(0, merged_sequences[0][0])]
            for i in range(len(merged_sequences) - 1):
                non_nan_sequences.append((merged_sequences[i][1], merged_sequences[i + 1][0]))
                
            non_nan_sequences.append((merged_sequences[-1][1], int(len(next(iter(processed_data.values()))))))

            non_nan_sequences_each_data_array.append(non_nan_sequences)
            for start, end in non_nan_sequences:
                if end - start <= 0:
                    continue
                snippet = {}
                for channel_name, channel in processed_data.items():
                    snippet[channel_name] = channel[start:end]

                cleaned_processed_data_array.append(snippet)
        # else no long nan sequences that need to be removed were found, so imply append the original processed data
        else:
            cleaned_processed_data_array.append(processed_data)
            

    return cleaned_processed_data_array, logger_infos, non_nan_sequences_each_data_array

        


def downsample_record(channel, to_downsample, current_fs):

    downsampling_strategy = to_downsample.get('downsampling_strategy', 'decimate')
    desired_resolution = to_downsample.get('desired_resolution', 1.0)
    downsampler = create_downsampler(downsampling_strategy)
    channel = downsampler.downsample(channel, desired_resolution, current_fs)

    return channel, desired_resolution


def perform_signal_processing(
        filtered_data: np.ndarray, 
        filtered_names: List[str], 
        signal_processing: List[Dict[str, Any]], 
        start_at_processing_step: int,
        process_until_step: int,
        metadata: Dict[str, Any],
        long_nan_removal_config: Dict[str, Any] = None,
        logger: logging.Logger = None,
        records_to_visualize = []
    ) -> Tuple[np.ndarray, List[str]]:
    """
    Perform signal processing on the filtered data.
    
    Args:
        filtered_data: Filtered signal data (samples x channels)
        filtered_names: Names of the channels in the filtered data
        signal_processing: Configuration for signal processing
        metadata: Metadata containing sampling rate and other info
        
    Returns:
        Processed signal data and updated channel names
    """
    # Get logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)

    logger_infos = []
    
    processed_data_array = []
    processed_data = {}
    for item in signal_processing:
        channel_name = item.get('channel')
        if channel_name not in filtered_names:
            raise DatasetCreationError(f"Channel {channel_name} not found in filtered names: {filtered_names}")
        else:
            processed_data[channel_name] = get_channel_signal_from_array(filtered_data, channel_name, filtered_names)
    processed_data_array.append(processed_data)


    long_nan_removal_config_dict = {}
    for item in long_nan_removal_config or []:
        after_step = item.get("after_step")
        max_consecutive_nans = item.get("max_consecutive_nans")
        long_nan_removal_config_dict[after_step] = max_consecutive_nans


    # process step by step 
    # get max steps for all channels
    max_steps = max(len(channel_config.get('steps', [])) for channel_config in signal_processing)
    # for donwsampling we need the current fs
    current_fs = {}
    for channel_name in filtered_names:
        current_fs[channel_name] = metadata['sampling_rate']
    for step_idx in range(start_at_processing_step, process_until_step):

        # copy it in a way that we can compare before and after
        processed_data_array_copy = copy.deepcopy(processed_data_array)
        for channel_name in filtered_names:

            # get channel config where channel_name matches "channel" in the list of dicts
            channel_config = next((item for item in signal_processing if item.get('channel') == channel_name), None)
            if channel_config:

            
                steps = channel_config.get('steps', [])
                # get dict from channel_config, where "step" matches step_idx
                current_step = next((step for step in steps if step.get('step') == step_idx), None)

                # print("Current step for channel", channel_name, ":", current_step)
                if current_step:
                    # print(f"Processing step {step_idx} for channel {channel_name}")

                    for i in range(len(processed_data_array)):
                        channel = processed_data_array[i][channel_name]

                        to_downsample = current_step.get("downsampling", {})
                        to_data_cleaning = current_step.get("data_cleaning", {})
                        to_imputation = current_step.get("imputation", {})

                        if to_downsample != {}:
                            #print(f"Downsampling channel {channel_name} at step {step_idx}")
                            channel, current_fs[channel_name] = downsample_record(channel, to_downsample, current_fs[channel_name])

                        if to_data_cleaning != {}:
                            lower_threshold = to_data_cleaning.get('lower_threshold')
                            upper_threshold = to_data_cleaning.get('upper_threshold')
                            channel = clean_data(channel, lower_threshold, upper_threshold)

                        if to_imputation != {}:
                            #print(f"Imputing channel {channel_name} at step {step_idx}")
                            imputation_stategy = to_imputation.get('method', 'mean')
                            imputer = create_imputer(imputation_stategy)
                            #print("Imputer created:", imputer)
                            channel = imputer.impute(processed_data_array[i], channel_name, metadata.get("imputer_path", ""))
                            
                        processed_data_array[i][channel_name] = channel

        if metadata.get("record_id") in records_to_visualize:
            visualize_step_for_record(
                record_id=metadata.get("record_id", "unknown"),
                step_id=step_idx,
                processed_data_array_before=processed_data_array_copy,
                processed_data_array_after=processed_data_array,
                signal_processing_config=signal_processing,
                output_path=metadata.get("output_path_process_images", "outputs/reports/process_images")
            )


        # check if a remove long nan sequence step needs to be applied
        if long_nan_removal_config_dict.get(step_idx) is not None:
            processed_data_array_copy = copy.deepcopy(processed_data_array)
            max_consecutive_nans = long_nan_removal_config_dict.get(step_idx)
            processed_data_array, new_logger_infos, non_nan_sequences = remove_long_nan_sequences(step_idx, processed_data_array, max_consecutive_nans)
            logger_infos.extend(new_logger_infos)

            # remove everything that does not meet the "min_record_duration" requirement
            intermediate_processed_data_array = copy.deepcopy(processed_data_array)
            processed_data_array = []
            print("Long nan removal for record:", metadata.get("record_id", "unknown"))
            print("len(intermediate_processed_data_array):", len(intermediate_processed_data_array))
            for i in range(len(intermediate_processed_data_array)):
                processed_data = intermediate_processed_data_array[i]
                total_length = len(next(iter(processed_data.values())))
                print("Total length after long nan removal:", total_length)
                print("Min Length", metadata.get("min_record_duration", 0) * current_fs[filtered_names[0]])
                # TODO more elegant solution to get current fs
                if total_length >=  current_fs[filtered_names[0]] * metadata.get("min_record_duration", 0):
                    processed_data_array.append(processed_data)
                else:
                    logger_infos.append(f"Removed processed data due to insufficient duration: {total_length / current_fs[filtered_names[0]]}s")
            
            print("len(processed_data_array) after min length removal:", len(processed_data_array))
            visualize_long_nan_removal_for_record(
                record_id=metadata.get("record_id", "unknown"),
                step_id=step_idx,
                processed_data_array_before=processed_data_array_copy,
                processed_data_array_after=processed_data_array,
                non_nan_sequences=non_nan_sequences,
                min_required_length=metadata.get("min_record_duration", 0) * current_fs[filtered_names[0]],
                output_path=metadata.get("output_path_process_images", "outputs/reports/process_images")
            )


        if len(processed_data_array) == 0:
            print(f"No processed data left after step {step_idx}, exiting processing loop.")
            step_idx = max_steps  # to exit the loop

    return processed_data_array, logger_infos