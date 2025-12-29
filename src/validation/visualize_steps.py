from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np
import random

import matplotlib.pyplot as plt

_zoom_snippet = 60
max_snippet_length = 1000
    
def initialize_visualization(list_of_all_records, config):
        # choose a random of max 10 records to visualize
        if len(list_of_all_records) > 10:
            records = random.sample(list_of_all_records, 10)
        else:
            records = list_of_all_records

        seed = config.get('random_seed', 42)
        random.seed(seed)

        records_including_after_split = []
        for record in records:
            records_including_after_split.append(record)
            records_including_after_split.append(record + "_observation")
            records_including_after_split.append(record + "_prediction")

        return records_including_after_split
        
    
def beautify_axes(axes, channel_idx, data_before, data_after):
    data_before_clean = data_before[~np.isnan(data_before)]
    data_after = np.array(data_after)
    data_after_clean = data_after[~np.isnan(data_after)]

    # TODO this doesn't seem to work 
    axes[channel_idx * 2].set_ylim(min(np.min(data_before_clean), np.min(data_after_clean)) - 1, max(np.max(data_before_clean), np.max(data_after_clean)) + 1)
    axes[channel_idx * 2 + 1].set_ylim(min(np.min(data_before_clean), np.min(data_after_clean)) - 1, max(np.max(data_before_clean), np.max(data_after_clean)) + 1)

    return axes

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
        channel_after = np.array(data_after)

        # fill up data_after to match length of data_before for visualization
        data_after_expanded = []
        data_after_expanded_with_nans = []
        for data_point in data_after:
            data_after_expanded.extend([data_point] * downsampling_factor)
            data_after_expanded_with_nans.extend([data_point] + [np.nan] * (downsampling_factor - 1))

        # plot two lines one for data_before and one for data_after
        axes[channel_id * 2].plot(data_before, label='Before Downsampling', alpha=0.7, color='blue')
        axes[channel_id * 2].scatter(range(len(data_before)), data_before, label='Before Downsampling', alpha=0.7, color='blue', s=10, marker='x')
        axes[channel_id * 2 + 1].plot(data_after_expanded, label='After Downsampling', alpha=0.7, color='red')
        axes[channel_id * 2 + 1].scatter(range(len(data_after_expanded_with_nans)), data_after_expanded_with_nans, label='After Downsampling', alpha=0.7, color='red', s=10, marker='x')
        # add a vline every downsampling_factor
        for i in range(0, len(data_before), downsampling_factor):
            axes[channel_id * 2].axvline(x=i, color='gray', linestyle='-', alpha=0.3)
            axes[channel_id * 2 + 1].axvline(x=i, color='gray', linestyle='-', alpha=0.3)

        #axes = beautify_axes(axes, channel_id, data_before, data_after_expanded)
        
        # zoomed in version - use original data for zoom since it's small
        # zoomed in version - use original data for zoom since it's small
        if len(data_before) < _zoom_snippet:
            data_before_zoom = data_before
            data_after_zoom = data_after
        else:
            data_before_zoom = data_before[:_zoom_snippet]
            data_after_zoom = data_after_expanded[:_zoom_snippet]
            data_after_zoom_with_nans = data_after_expanded_with_nans[:_zoom_snippet]

        # plot two lines one for data_before and one for data_after
        axes_zoom[channel_id * 2].plot(data_before_zoom, label='Before Downsampling', alpha=0.7, color='blue')
        axes_zoom[channel_id * 2].scatter(range(len(data_before_zoom)), data_before_zoom, label='Before Downsampling', alpha=0.7, color='blue', s=10, marker='x')
        axes_zoom[channel_id * 2 + 1].plot(data_after_zoom, label='After Downsampling', alpha=0.7, color='red')
        axes_zoom[channel_id * 2 + 1].scatter(range(len(data_after_zoom_with_nans)), data_after_zoom_with_nans, label='After Downsampling', alpha=0.7, color='red', s=10, marker='x')
        # add a vline every downsampling_factor
        for i in range(0, len(data_before), downsampling_factor):
            axes_zoom[channel_id * 2].axvline(x=i, color='gray', linestyle='-', alpha=0.3)
            axes_zoom[channel_id * 2 + 1].axvline(x=i, color='gray', linestyle='-', alpha=0.3)

        #axes_zoom = beautify_axes(axes_zoom, channel_id, data_before_zoom, data_after_zoom)

        return fig, fig_zoom, axes, axes_zoom

def visualize_long_nan_removal_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after, long_nan_config):
    pass


def visualize_windowing_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after):
    pass

def visualize_step_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after, signal_processing_config):

        if record_id and processed_data_array_before and processed_data_array_after:

            # only visualize first entry of processed_data_array_before and processed_data_array_after
            data_before = processed_data_array_before[0]
            data_after = processed_data_array_after[0]

            channels_before = list(data_before.keys())
            channels_after = list(data_after.keys())

            fig = plt.figure(figsize=(10, 5))
            fig_zoom = plt.figure(figsize=(10, 5))
            gs = fig.add_gridspec(2 * len(channels_before), hspace=0, figure=fig)
            axes = gs.subplots(sharex=True, sharey=False)
            gs_zoom = fig_zoom.add_gridspec(2 * len(channels_before), hspace=0, figure=fig_zoom)
            axes_zoom = gs_zoom.subplots(sharex=True, sharey=False)
            channel_idx = 0
            for channel in channels_before:
                if channel not in channels_after:
                    # TODO do soemthing
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
            fig.suptitle(f"Record: {record_id} - Step: {step_id} Visualization", fontsize=16)
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
            fig.savefig(f"outputs/img/visualization_record_{record_id}_step_{step_id}.png")
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
            fig_zoom.suptitle(f"Record: {record_id} - Step: {step_id} Visualization (Zoomed In)", fontsize=16)
            fig_zoom.savefig(f"outputs/img/visualization_record_{record_id}_step_{step_id}_zoomed.png")
            plt.close(fig_zoom)
            # match step_type:
            #    case 'downsampling':
            #        self.visualize_downsampling_for_record(record_id, step_id, data_before, data_after)
            #    case 'data_cleaning':
            #        pass
            #    case 'imputing':
            #        pass 
            #    case _:
            #        pass
        #else:
        #    pass 
        

    # for step -> array of cut records
    # long nan sequence removal

    # for array of cut records
    # windowing