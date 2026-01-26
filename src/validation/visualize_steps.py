from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np
import random
import math
import os
import threading

# Set matplotlib to use non-interactive backend for multiprocessing safety
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

_zoom_snippet = 130
max_snippet_length = 1000
image_width = 20
image_height = 10
number_of_records_to_visualize = 20

# Lock for thread-safe file I/O in multiprocessing context
_visualization_lock = threading.Lock()

    
def initialize_visualization(list_of_all_records, config):
        # choose a random of max 10 records to visualize
        # remove duplicates
        list_of_all_records = list(set(list_of_all_records))

        if len(list_of_all_records) > number_of_records_to_visualize:
            records = random.sample(list_of_all_records, number_of_records_to_visualize)
        else:
            records = list_of_all_records

        seed = config.get('random_seed', 42)
        random.seed(seed)

        records_including_after_split = []
        for record in records:
            records_including_after_split.append(record)
            records_including_after_split.append(record + "_sample_0000_observation")
            records_including_after_split.append(record + "_sample_0000_prediction")

        print(f"Records selected for visualization: {records_including_after_split}")
        return records_including_after_split
        
    
def beautify_axes(axes, channel_idx, data_before, data_after):
    data_before_clean = data_before[~np.isnan(data_before)]
    data_after = np.array(data_after)
    data_after_clean = data_after[~np.isnan(data_after)]
    
    if data_after_clean.size == 0 or data_before_clean.size == 0:
        return axes

    axes[channel_idx * 2].set_ylim(min(np.min(data_before_clean), np.min(data_after_clean)) - 1, max(np.max(data_before_clean), np.max(data_after_clean)) + 1)
    axes[channel_idx * 2 + 1].set_ylim(min(np.min(data_before_clean), np.min(data_after_clean)) - 1, max(np.max(data_before_clean), np.max(data_after_clean)) + 1)

    return axes

def _add_header(img, text, header_h, font_scale):
    """Helper to add header to image."""
    header_img = np.full((header_h, img.shape[1], 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = max(10, (img.shape[1] - text_w) // 2)
    text_y = (header_h + text_h) // 2
    cv2.putText(header_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return cv2.vconcat([header_img, img])

# for step -> for channel -> array of cut records
# downsampling (input_channel, output_channel) -> less data <- done :)
# data cleaning (input_channel, output_channel) -> less data (more np values) <- done :)
# imputing (input_channel, output_channel) -> nan replaced by imputed values 

def _visualize_imputing_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id: str, step_id: int, channel_id, data_before, data_after):

        # plot two lines one for data_before and one for data_after
        axes[channel_id * 2].plot(data_before, label='Before Imputation', alpha=0.7, color='blue')
        axes[channel_id * 2].scatter(range(len(data_before)), data_before, label='Before Imputation', alpha=0.7, color='blue', s=10, marker='x')
        axes[channel_id * 2 + 1].plot(data_after, label='After Imputation', alpha=0.7, color='red')
        axes[channel_id * 2 + 1].scatter(range(len(data_after)), data_after, label='After Imputation', alpha=0.7, color='red', s=10, marker='x')

        # background of values with nan should be highlighted
        nan_indices_before = np.where(np.isnan(data_before))[0]
        nan_indices_after = np.where(np.isnan(data_after))[0]
        nan_indices_union = set(nan_indices_before).union(set(nan_indices_after))
        filled_nan_indices = [idx for idx in nan_indices_union if idx in nan_indices_before and idx not in nan_indices_after]
        

        for nan_index in filled_nan_indices:
            axes[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color='green', alpha=0.5)

        for nan_index in nan_indices_before:
            axes[channel_id * 2].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

        for nan_index in nan_indices_after:
            axes[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

        axes = beautify_axes(axes, channel_id, data_before, data_after)
    

        # zoomed in version - use original data for zoom since it's small
        if (not "prediction" in record_id) and (not "observation" in record_id):
            if len(data_before) < _zoom_snippet:
                data_before_zoom = data_before
                data_after_zoom = data_after
            else:
                data_before_zoom = data_before[:_zoom_snippet]
                data_after_zoom = data_after[:_zoom_snippet]
            # plot two lines one for data_before and one for data_after
            axes_zoom[channel_id * 2].plot(data_before_zoom, label='Before Imputation', alpha=0.7, color='blue')
            axes_zoom[channel_id * 2].scatter(range(len(data_before_zoom)), data_before_zoom, label='Before Imputation', alpha=0.7, color='blue', s=10, marker='x')
            axes_zoom[channel_id * 2 + 1].plot(data_after_zoom, label='After Imputation', alpha=0.7, color='red')
            axes_zoom[channel_id * 2 + 1].scatter(range(len(data_after_zoom)), data_after_zoom, label='After Imputation', alpha=0.7, color='red', s=10, marker='x')

            # background of values with nan should be highlighted
            nan_indices_before = np.where(np.isnan(data_before_zoom))[0]
            nan_indices_after = np.where(np.isnan(data_after_zoom))[0]
            nan_indices_union = set(nan_indices_before).union(set(nan_indices_after))
            filled_nan_indices = [idx for idx in nan_indices_union if idx in nan_indices_before and idx not in nan_indices_after]
            for nan_index in filled_nan_indices:
                axes_zoom[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color='green', alpha=0.5)

            for nan_index in nan_indices_before:
                axes_zoom[channel_id * 2].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

            for nan_index in nan_indices_after:
                axes_zoom[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

            axes_zoom = beautify_axes(axes_zoom, channel_id, data_before_zoom, data_after_zoom)

        return fig, fig_zoom, axes, axes_zoom



def _visualize_data_cleaning_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id: str, step_id: int, channel_id, data_before, data_after):
        
        # plot two lines one for data_before and one for data_after
        axes[channel_id * 2].plot(data_before, label='Before Data Cleaning', alpha=0.7, color='blue')
        axes[channel_id * 2].scatter(range(len(data_before)), data_before, label='Before Data Cleaning', alpha=0.7, color='blue', s=10, marker='x')
        axes[channel_id * 2 + 1].plot(data_after, label='After Data Cleaning', alpha=0.7, color='red')
        axes[channel_id * 2 + 1].scatter(range(len(data_after)), data_after, label='After Data Cleaning', alpha=0.7, color='red', s=10, marker='x')

        # background of values with nan should be highlighted
        nan_indices_before = np.where(np.isnan(data_before))[0]
        for nan_index in nan_indices_before:
            axes[channel_id * 2].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

        nan_indices_after = np.where(np.isnan(data_after))[0]
        for nan_index in nan_indices_after:
            axes[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)
            # axes[channel_id * 2].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

        axes = beautify_axes(axes, channel_id, data_before, data_after)

        if (not "prediction" in record_id) and (not "observation" in record_id):
            # zoomed in version - use original data for zoom since it's small
            if len(data_before) < _zoom_snippet:
                data_before_zoom = data_before
                data_after_zoom = data_after
            else:
                data_before_zoom = data_before[:_zoom_snippet]
                data_after_zoom = data_after[:_zoom_snippet]
            # plot two lines one for data_before and one for data_after
            axes_zoom[channel_id * 2].plot(data_before_zoom, label='Before Data Cleaning', alpha=0.7, color='blue')
            axes_zoom[channel_id * 2].scatter(range(len(data_before_zoom)), data_before_zoom, label='Before Data Cleaning', alpha=0.7, color='blue', s=10, marker='x')
            axes_zoom[channel_id * 2 + 1].plot(data_after_zoom, label='After Data Cleaning', alpha=0.7, color='red')
            axes_zoom[channel_id * 2 + 1].scatter(range(len(data_after_zoom)), data_after_zoom, label='After Data Cleaning', alpha=0.7, color='red', s=10, marker='x')

            # background of values with nan should be highlighted
            nan_indices_before = np.where(np.isnan(data_before_zoom))[0]
            for nan_index in nan_indices_before:
                axes_zoom[channel_id * 2].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

            nan_indices_after = np.where(np.isnan(data_after_zoom))[0]
            for nan_index in nan_indices_after:
                axes_zoom[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)
                # axes_zoom[channel_id * 2].axvspan(nan_index - 0.5, nan_index + 0.5, color='grey', alpha=0.5)

            axes_zoom = beautify_axes(axes_zoom, channel_id, data_before_zoom, data_after_zoom)

        return fig, fig_zoom, axes, axes_zoom


def _visualize_downsampling_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id: str, step_id: int, channel_id, data_before, data_after):

        # calculate downsampling factor based on original data
        downsampling_factor = len(data_before) // len(data_after)


        # fill up data_after to match length of data_before for visualization
        data_after_expanded = []
        #data_after_expanded_with_nans = [np.nan] * (downsampling_factor // 2)
        for data_point in data_after:
            data_after_expanded.extend([data_point] * downsampling_factor)
        #    data_after_expanded_with_nans.extend([data_point] + [np.nan] * (downsampling_factor -1))

        # plot two lines one for data_before and one for data_after
        axes[channel_id * 2].plot(data_before, label='Before Downsampling', alpha=0.7, color='blue')
        axes[channel_id * 2].scatter(range(len(data_before)), data_before, label='Before Downsampling', alpha=0.7, color='blue', s=10, marker='x')
        axes[channel_id * 2 + 1].plot(data_after_expanded, label='After Downsampling', alpha=0.7, color='red')
       # axes[channel_id * 2 + 1].scatter(range(len(data_after_expanded_with_nans)), data_after_expanded_with_nans, label='After Downsampling', alpha=0.7, color='red', s=10, marker='x')
        # add a vline every downsampling_factor
        for i in range(0, len(data_before) + 1, downsampling_factor):
            axes[channel_id * 2].axvline(x=i - 0.5, color='gray', linestyle='-', alpha=0.5)
            axes[channel_id * 2 + 1].axvline(x=i - 0.5, color='gray', linestyle='-', alpha=0.5)

        axes = beautify_axes(axes, channel_id, data_before, data_after_expanded)
        
        # zoomed in version - use original data for zoom since it's small
        # zoomed in version - use original data for zoom since it's small
        if (not "prediction" in record_id) and (not "observation" in record_id):
            if len(data_before) < _zoom_snippet:
                data_before_zoom = data_before
                data_after_zoom = data_after
            else:
                data_before_zoom = data_before[:_zoom_snippet]
                data_after_zoom = data_after_expanded[:_zoom_snippet]

            # plot two lines one for data_before and one for data_after
            axes_zoom[channel_id * 2].plot(data_before_zoom, label='Before Downsampling', alpha=0.7, color='blue')
            axes_zoom[channel_id * 2].scatter(range(len(data_before_zoom)), data_before_zoom, label='Before Downsampling', alpha=0.7, color='blue', s=10, marker='x')
            axes_zoom[channel_id * 2 + 1].plot(data_after_zoom, label='After Downsampling', alpha=0.7, color='red')
            # add a vline every downsampling_factor
            for i in range(0, len(data_before) + 1, downsampling_factor):
                axes_zoom[channel_id * 2].axvline(x=i - 0.5, color='gray', linestyle='-', alpha=0.5)
                axes_zoom[channel_id * 2 + 1].axvline(x=i - 0.5, color='gray', linestyle='-', alpha=0.5)

            axes_zoom = beautify_axes(axes_zoom, channel_id, data_before_zoom, data_after_zoom)

        return fig, fig_zoom, axes, axes_zoom


def _visualize_all_channels_for_record(data_array, number_of_additional_subplots=0, image_width=image_width, image_height=image_height):

    channels = list(data_array.keys())

    fig = plt.figure(figsize=(image_width, image_height / 2))
    gs = fig.add_gridspec(len(channels) + number_of_additional_subplots, hspace=0, figure=fig)
    axes = gs.subplots(sharex=True, sharey=False)
    
    # Ensure axes is always iterable
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    channel_idx = 0
    for channel in channels:
        axes[channel_idx].plot(data_array[channel], label=f'Channel: {channel}', alpha=0.7)
        axes[channel_idx].scatter(range(len(data_array[channel])), data_array[channel], label=f'Channel: {channel}', alpha=0.7, s=10, marker='x')
        axes[channel_idx].set_ylabel(f'{channel}')
        channel_idx += 1
               
    return fig, axes

def visualize_long_nan_removal_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after, non_nan_sequences, min_required_length, output_path="outputs/img/"):
    
    data_array = processed_data_array_before[0] # only first entry
    fig, axes = _visualize_all_channels_for_record(data_array, number_of_additional_subplots=0, image_width=image_width, image_height=image_height)
    channels = list(data_array.keys())
    
    # Ensure axes is always iterable
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # no sequences removed
    if len(non_nan_sequences) == 0:
        for ax in axes:
            ax.axvline(x=-0.5, color='red', linestyle='-', alpha=0.7)
            ax.axvline(x=len(data_array[channels[0]]) + 0.5, color='red', linestyle='-', alpha=0.7)
            ax.axvspan(- 0.5, len(data_array[channels[0]]) + 0.5, color='green', alpha=0.5)

    # sequences were removed
    else:
        non_nan_sequences_for_data_array = non_nan_sequences[0]  # only first entry

        for start, end in non_nan_sequences_for_data_array:
            for ax in axes:
                ax.axvline(x=start - 0.5, color='red', linestyle='-', alpha=0.7)
                ax.axvline(x=end + 0.5, color='red', linestyle='-', alpha=0.7)

                #  also vizualize the segments removed due to insufficient duration
                if end - start < min_required_length:
                    ax.axvspan(start - 0.5, end + 0.5, color='blue', alpha=0.5)
                else:
                    ax.axvspan(start - 0.5, end + 0.5, color='green', alpha=0.5)

    # fig.suptitle(f"Record: {record_id} - Step: {step_id} Visualization of Long NaN Removal", fontsize=16)
    # fig.suptitle(f"Step: {step_id} Visualization of Long NaN Removal", fontsize=16)
    with _visualization_lock:
        fig.savefig(f"{output_path}/visualization_record_{record_id}_step_{step_id}_long_nan_removal.png")
    plt.close(fig)


def visualize_windowing_for_record( record_id: str, step_id: int, windows, processed_data_array, observation_window,
                prediction_horizon, prediction_window, step, expected_resolution, output_path="outputs/img/"):

    data_array = processed_data_array[0]  # only first entry
    step_size = expected_resolution * step
    observation_window_size = expected_resolution * observation_window
    prediction_horizon_size = expected_resolution * prediction_horizon
    prediction_window_size = expected_resolution * prediction_window

    channels = list(data_array.keys())
    number_of_windows_to_visualize = min(10, len(windows[channels[0]]))

    total_size_max_ten_windows = math.ceil((observation_window_size + prediction_horizon_size + prediction_window_size) * number_of_windows_to_visualize)
    zoom_snippet_full_view = min(len(data_array[channels[0]]), total_size_max_ten_windows)
    zoom_snippet_zoom = min(len(data_array[channels[0]]), _zoom_snippet)
    zoom_levels = [zoom_snippet_full_view, zoom_snippet_zoom]

    for zoom_idx in range(len(zoom_levels)):

        snipped_data_array = dict()
        for channel in channels:
            snipped_data_array[channel] = data_array[channel][:zoom_levels[zoom_idx]]
        fig, axes = _visualize_all_channels_for_record(snipped_data_array, number_of_additional_subplots=1, image_width=image_width / 2, image_height=image_height)

        # only show for a limited number of windows - e.g. 10
        curr_number_of_windows_to_visualize = min(number_of_windows_to_visualize, int((zoom_levels[zoom_idx] // (observation_window_size + prediction_horizon_size + prediction_window_size))))
        for i in range(curr_number_of_windows_to_visualize):
            axes[-1].hlines(y=i, xmin=step_size * i, xmax= step_size * i + observation_window_size, color='blue', alpha=0.7)  # h line for observation window
            axes[-1].hlines(y=i, xmin=step_size * i + observation_window_size + prediction_horizon_size, xmax= step_size * i + observation_window_size + prediction_horizon_size + prediction_window_size, color='red', alpha=0.7)  # h line for prediction window
            axes[-1].vlines(x=step_size * i, ymin=i, ymax = curr_number_of_windows_to_visualize - 0.5, color='blue', linestyle='-', alpha=0.7)
            axes[-1].vlines(x=step_size * i + observation_window_size, ymin=i, ymax = curr_number_of_windows_to_visualize - 0.5, color='blue', linestyle='-', alpha=0.7)
            axes[-1].vlines(x=step_size * i + observation_window_size + prediction_horizon_size, ymin=i, ymax = curr_number_of_windows_to_visualize - 0.5, color='red', linestyle='-', alpha=0.7)
            axes[-1].vlines(x=step_size * i + observation_window_size + prediction_horizon_size + prediction_window_size, ymin=i, ymax = curr_number_of_windows_to_visualize - 0.5, color='red', linestyle='-', alpha=0.7)  

            for ax in axes[:-1]:
                ax.axvline(x=step_size * i, color='blue', linestyle='-', alpha=0.7)
                ax.axvline(x=step_size * i + observation_window_size, color='blue', linestyle='-', alpha=0.7)
                ax.axvline(x=step_size * i + observation_window_size + prediction_horizon_size, color='red', linestyle='-', alpha=0.7)
                ax.axvline(x=step_size * i + observation_window_size + prediction_horizon_size + prediction_window_size, color='red', linestyle='-', alpha=0.7)

        # add partial window
        if zoom_idx == 1 and number_of_windows_to_visualize > curr_number_of_windows_to_visualize and curr_number_of_windows_to_visualize * (observation_window_size + prediction_horizon_size + prediction_window_size) < zoom_levels[zoom_idx]:
            axes[-1].hlines(y=curr_number_of_windows_to_visualize - 1, xmin=step_size * curr_number_of_windows_to_visualize, xmax= zoom_levels[zoom_idx], color='blue', alpha=0.7)  # h line for observation window
            axes[-1].vlines(x=step_size * curr_number_of_windows_to_visualize, ymin=curr_number_of_windows_to_visualize, ymax = curr_number_of_windows_to_visualize - 0.5, color='blue', linestyle='-', alpha=0.7)
            for ax in axes[:-1]:
                ax.axvline(x=step_size * curr_number_of_windows_to_visualize, color='blue', linestyle='-', alpha=0.7)

            if zoom_levels[zoom_idx] > step_size * curr_number_of_windows_to_visualize + observation_window_size + prediction_horizon_size:
                axes[-1].hlines(y=curr_number_of_windows_to_visualize - 1, xmin=step_size * curr_number_of_windows_to_visualize + observation_window_size + prediction_horizon_size, xmax= zoom_levels[zoom_idx], color='red', alpha=0.7)  # h line for prediction window
                axes[-1].vlines(x=step_size * curr_number_of_windows_to_visualize + observation_window_size, ymin=curr_number_of_windows_to_visualize, ymax = curr_number_of_windows_to_visualize - 0.5, color='blue', linestyle='-', alpha=0.7)
                for ax in axes[:-1]:
                    ax.axvline(x=step_size * curr_number_of_windows_to_visualize + observation_window_size, color='blue', linestyle='-', alpha=0.7)
                    ax.axvline(x=step_size * curr_number_of_windows_to_visualize + observation_window_size + prediction_horizon_size, color='red', linestyle='-', alpha=0.7)

            if zoom_levels[zoom_idx] > step_size * curr_number_of_windows_to_visualize + observation_window_size + prediction_horizon_size + prediction_window_size:
                axes[-1].vlines(x=step_size * curr_number_of_windows_to_visualize + observation_window_size + prediction_horizon_size, ymin=curr_number_of_windows_to_visualize, ymax = zoom_levels[zoom_idx] - 0.5, color='red', linestyle='-', alpha=0.7)
                axes[-1].vlines(x=step_size * curr_number_of_windows_to_visualize + observation_window_size + prediction_horizon_size + prediction_window_size, ymin=curr_number_of_windows_to_visualize, ymax = zoom_levels[zoom_idx] - 0.5, color='red', linestyle='-', alpha=0.7)  
                for ax in axes[:-1]:
                    ax.axvline(x=step_size * i + observation_window_size + prediction_horizon_size + prediction_window_size, color='red', linestyle='-', alpha=0.7)

            #window_subplot = plt.figure(2,1)
            #gs_subplots = window_subplot.add_gridspec(2, vspace=0, figure=window_subplot)
            #axes_subplots = gs_subplots.subplots(sharex=False, sharey=True)
            #axes_subplots[0].plot(windows[i]['observation_window'], label='Observation Window', alpha=0.7, color='blue')
            #axes_subplots[1].plot(windows[i]['prediction_window'], label='Prediction Window', alpha=0.7, color='red')
            
        if zoom_idx == 0:
            # fig.suptitle(f"Record: {record_id} - Step: {step_id} - windowing - {observation_window} - {prediction_horizon} - {prediction_window}", fontsize=16)
            # fig.suptitle(f"Step: {step_id} - windowing - {observation_window} - {prediction_horizon} - {prediction_window}", fontsize=16)
            with _visualization_lock:
                fig.savefig(f"{output_path}/visualization_record_{record_id}_step_{step_id}_windowing_full_length.png")
        else:
            # fig.suptitle(f"Record: {record_id} - Step: {step_id} - windowing - {observation_window} - {prediction_horizon} - {prediction_window}", fontsize=16)
            # fig.suptitle(f"Step: {step_id} - windowing - {observation_window} - {prediction_horizon} - {prediction_window}", fontsize=16)
            with _visualization_lock:
                fig.savefig(f"{output_path}/visualization_record_{record_id}_step_{step_id}_windowing_zoomed.png")
        plt.close(fig)


def visualize_step_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after, signal_processing_config, output_path="outputs/img/"):

        if record_id and processed_data_array_before and processed_data_array_after:

            # only visualize first entry of processed_data_array_before and processed_data_array_after
            data_before = processed_data_array_before[0]
            data_after = processed_data_array_after[0]

            channels_before = list(data_before.keys())
            channels_after = list(data_after.keys())

            fig = plt.figure(figsize=(image_width / 2, image_height))
            fig_zoom = plt.figure(figsize=(image_width / 2, image_height))
            gs = fig.add_gridspec(2 * len(channels_before), hspace=0, figure=fig)
            axes = gs.subplots(sharex=True, sharey=False)
            gs_zoom = fig_zoom.add_gridspec(2 * len(channels_before), hspace=0, figure=fig_zoom)
            axes_zoom = gs_zoom.subplots(sharex=True, sharey=False)
            
            # Ensure axes are always iterable
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            if not isinstance(axes_zoom, np.ndarray):
                axes_zoom = np.array([axes_zoom])
            
            channel_idx = 0
            for channel in channels_before:
                if channel not in channels_after:
                    # TODO do something
                    print(f"Channel {channel} not in after data, skipping visualization for this channel.")
                    continue
                # get step type from config

                # can be different for each channel
                # so we need to crete a dict of step types for each channel
                # TODO create a class for reading from config
                channel_config = next((item for item in signal_processing_config if item.get('channel') == channel), None)
                channel_steps = channel_config.get('steps', [])
                current_step = next((step for step in channel_steps if step.get('step') == step_id), None)

                # overwriting is on purpose
                if current_step.get("downsampling", {}) != {}:
                    step_type = 'downsampling'
                if current_step.get("data_cleaning", {}) != {}:
                    step_type = 'data_cleaning'
                if current_step.get("imputation", {}) != {}:
                    step_type = 'imputation'
                

                match step_type:
                    case 'downsampling':
                        fig, fig_zoom, axes, axes_zoom = _visualize_downsampling_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id, step_id, channel_idx, data_before[channel], data_after[channel])
                    case 'data_cleaning':
                        fig, fig_zoom, axes, axes_zoom = _visualize_data_cleaning_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id, step_id, channel_idx, data_before[channel], data_after[channel])
                    case 'imputation':
                        fig, fig_zoom, axes, axes_zoom = _visualize_imputing_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id, step_id, channel_idx, data_before[channel], data_after[channel])
                    case _:
                        pass
                     
                axes[channel_idx * 2].set_ylabel(f'{channel} \n Before')
                axes[channel_idx * 2 + 1].set_ylabel(f'{channel} \n After')

                axes_zoom[channel_idx * 2].set_ylabel(f'{channel} \n Before')
                axes_zoom[channel_idx * 2 + 1].set_ylabel(f'{channel} \n After')

                channel_idx += 1


            # 
            # fig.suptitle(f"Record: {record_id} - Step: {step_id} - {step_type}", fontsize=16)
            # fig.suptitle(f"Step: {step_id} - {step_type}", fontsize=16)
            max_len = 0
            for channel in channels_before:
                if max(len(data_before[channel]), len(data_after[channel])) > max_len:
                    max_len = max(len(data_before[channel]), len(data_after[channel]))
            for ax in axes:
                ax.label_outer()
                if max_len > 1:
                    ax.set_xlim(0, max_len)
                else:
                    ax.set_xlim(-1, 1)
            fig.savefig(f"{output_path}/visualization_record_{record_id}_step_{step_id}.png")
            plt.close(fig)

            max_len = 0
            for channel in channels_before:
                if min(len(data_before[channel]), _zoom_snippet) > max_len:
                    max_len = min(len(data_before[channel]), _zoom_snippet)
            for ax in axes_zoom:
                ax.label_outer()
                if max_len > 1:
                    ax.set_xlim(0, max_len)
                else:
                    ax.set_xlim(-1, 1)

            if (not "observation" in record_id) and (not "prediction" in record_id):
                # fig_zoom.suptitle(f"Record: {record_id} - Step: {step_id} - {step_type}", fontsize=16)
                # fig_zoom.suptitle(f"Step: {step_id} - {step_type}", fontsize=16)
                with _visualization_lock:
                    fig.savefig(f"{output_path}/visualization_record_{record_id}_step_{step_id}.png")
                    fig_zoom.savefig(f"{output_path}/visualization_record_{record_id}_step_{step_id}_zoomed.png")
            plt.close(fig_zoom)



def merge_step_visualizations_for_record(record_id, start_offset_seconds, end_offset_seconds, signal_processing_config, output_path="outputs/img/"):

    if (not "observation" in record_id) and (not "prediction" in record_id):

        # read all file-names from a path
        
        files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
        record_files = [f for f in files if f"visualization_record_{record_id}_" in f]

        # sort by step id
        record_files.sort(key=lambda x: int(x.split(".png")[0].split("_step_")[1].split("_")[0]))

        visualized_steps = [x.split(".png")[0].split("_step_")[1].split("_")[0] for x in record_files]
        visualized_steps = list(set(visualized_steps))
        visualized_steps.sort(key=lambda x: int(x))

        final_image_composition = []
        for step in visualized_steps:
            step_specific_files = [f for f in record_files if f"_step_{step}" in f]
            step_specific_files_first = [f for f in step_specific_files if "long_nan_removal" not in f and "windowing" not in f]

            # steps that are not long nan removal or windowing
            # observation - prediction
            if any("observation" in s for s in step_specific_files_first) and any("prediction" in s for s in step_specific_files_first):
                obs_file = [s for s in step_specific_files_first if "observation" in s]
                pred_file = [s for s in step_specific_files_first if "prediction" in s]
                if obs_file and pred_file:
                    final_image_composition.append([obs_file[0], pred_file[0]])
            else:
                normal_file = [s for s in step_specific_files_first if "zoomed" not in s]
                zoomed_file = [s for s in step_specific_files_first if "zoomed" in s]
                if normal_file and zoomed_file:
                    final_image_composition.append([normal_file[0], zoomed_file[0]])
                if normal_file and not zoomed_file:
                    final_image_composition.append([normal_file[0]])
            
            # long nan removal
            long_nan_file = [f for f in step_specific_files if "long_nan_removal" in f]
            if long_nan_file:
                final_image_composition.append([long_nan_file[0]])

            # windowing
            windowing_files = [f for f in step_specific_files if "windowing" in f]
            if windowing_files:
                # zoomed and not zoomed
                zoomed_file = [s for s in windowing_files if "zoomed" in s]
                normal_file = [s for s in windowing_files if "zoomed" not in s]
                if normal_file and zoomed_file:
                    final_image_composition.append([normal_file[0], zoomed_file[0]])
                if normal_file and not zoomed_file:
                    final_image_composition.append([normal_file[0]])

        images_to_append_vertically = []
        for image_file_group in final_image_composition:
            images = []
            for image_file in image_file_group:
                img_path = os.path.join(output_path, image_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)

            # concatenate images horizontally
            if images:
                merged_img = cv2.hconcat(images)

                if len(image_file_group) > 0:
                    first_file = image_file_group[0]
                    current_step_id = int(first_file.split("_step_")[1].split("_")[0].split(".")[0])
                    
                    # Determine Title Text
                    row_title = f"Step {current_step_id}"
                    if "windowing" in first_file:
                        row_title += " -> Windowing"
                    elif "long_nan_removal" in first_file:
                        row_title += " -> Long NaN Removal"
                    else:
                        # Infer type from config
                        found_type = "Processing"
                        for ch_cfg in signal_processing_config:
                            steps = ch_cfg.get("steps", [])
                            s_cfg = next((s for s in steps if s.get("step") == current_step_id), None)
                            if s_cfg:
                                if s_cfg.get("downsampling"):
                                    found_type = "Downsampling"
                                elif s_cfg.get("data_cleaning"):
                                    found_type = "Data Cleaning"
                                elif s_cfg.get("imputation"):
                                    found_type = "Imputation"
                                break
                        row_title += f" -> {found_type}"
                    
                    # Create and add header
                    merged_img = _add_header(merged_img, row_title, 60, 1.0)
                    
                images_to_append_vertically.append(merged_img)

        merged_img = None
        if images_to_append_vertically:
            # Find the maximum width
            max_width = max(img.shape[1] for img in images_to_append_vertically)
            
            # Resize all images to the same width, maintaining aspect ratio
            resized_images = []
            for img in images_to_append_vertically:
                if img.shape[1] != max_width:
                    new_height = int(img.shape[0] * max_width / img.shape[1])
                    resized_img = cv2.resize(img, (max_width, new_height))
                    resized_images.append(resized_img)
                else:
                    resized_images.append(img)
            
            merged_img = cv2.vconcat(resized_images)

        if merged_img is not None:
            super_title = f"Record: {record_id} | {start_offset_seconds}s -> {end_offset_seconds}s"
            merged_img = _add_header(merged_img, super_title, 100, 1.2)

        if merged_img is not None:
            with _visualization_lock:
                cv2.imwrite(f"{output_path}/merged_visualization_record_{record_id}.png", merged_img)

        # delete all files from record_files
        with _visualization_lock:
            for file in record_files:
                try:
                    os.remove(os.path.join(output_path, file))
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
