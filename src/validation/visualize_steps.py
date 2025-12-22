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

        print(f"Records selected for visualization: {records}")

        seed = config.get('random_seed', 42)
        random.seed(seed)

        return records
        
    
# for step -> for channel -> array of cut records
# downsampling (input_channel, output_channel) -> less data
# data cleaning (input_channel, output_channel) -> less data (more np values)
# imputing (input_channel, output_channel) -> nan replaced by imputed values

def _visualize_downsampling_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id: str, step_id: int, channel_id, data_before, data_after):

        # calculate downsampling factor based on original data
        downsampling_factor = len(data_before) // len(data_after)

        # fill up data_after to match length of data_before for visualization
        data_after_expanded = []
        for data_point in data_after:
            data_after_expanded.extend([data_point] * downsampling_factor)

        print(channel_id)
        
        # plot two lines one for data_before and one for data_after
        axes[channel_id * 2].plot(data_before, label='Before Downsampling', alpha=0.7, color='blue')
        axes[channel_id * 2 + 1].plot(data_after_expanded, label='After Downsampling', alpha=0.7, color='red')
        # add a vline every downsampling_factor
        for i in range(0, len(data_before), downsampling_factor):
            axes[channel_id * 2].axvline(x=i, color='gray', linestyle='--', alpha=0.3)
            axes[channel_id * 2 + 1].axvline(x=i, color='gray', linestyle='--', alpha=0.3)

        axes[channel_id * 2].set_ylim(min(np.min(data_before), np.min(data_after_expanded)) - 1, max(np.max(data_before), np.max(data_after_expanded)) + 1)
        axes[channel_id * 2 + 1].set_ylim(min(np.min(data_before), np.min(data_after_expanded)) - 1, max(np.max(data_before), np.max(data_after_expanded)) + 1)
        
        for ax in axes:
             ax.label_outer()

        # zoomed in version - use original data for zoom since it's small
        data_before_zoom = data_before[:_zoom_snippet]
        data_after_zoom = data_after_expanded[:_zoom_snippet]

        # plot two lines one for data_before and one for data_after
        axes_zoom[channel_id * 2].plot(data_before_zoom, label='Before Downsampling', alpha=0.7, color='blue')
        axes_zoom[channel_id * 2 + 1].plot(data_after_zoom, label='After Downsampling', alpha=0.7, color='red')
        # add a vline every downsampling_factor
        for i in range(0, len(data_before), downsampling_factor):
            axes_zoom[channel_id * 2].axvline(x=i, color='gray', linestyle='--', alpha=0.3)
            axes_zoom[channel_id * 2 + 1].axvline(x=i, color='gray', linestyle='--', alpha=0.3)

        axes_zoom[channel_id * 2].set_ylim(min(np.min(data_before_zoom), np.min(data_after_zoom)) - 1, max(np.max(data_before_zoom), np.max(data_after_zoom)) + 1)
        axes_zoom[channel_id * 2 + 1].set_ylim(min(np.min(data_before_zoom), np.min(data_after_zoom)) - 1, max(np.max(data_before_zoom), np.max(data_after_zoom)) + 1)

        for ax in axes_zoom:
             ax.label_outer()
        
        return fig, fig_zoom, axes, axes_zoom


def visualize_step_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after):
        print(f"Visualizing step for record {record_id}, step {step_id}")
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
                
                print(f"Vizualize downsampling for record {record_id}, step {step_id}, channel {channel}")
                print(f"Data before length: {len(data_before[channel])}, Data after length: {len(data_after[channel])}")

                fig, fig_zoom, axes, axes_zoom = _visualize_downsampling_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id, step_id, channel_idx, data_before[channel], data_after[channel])
                axes[channel_idx * 2].set_ylabel(f'{channel} \n Before')
                axes[channel_idx * 2 + 1].set_ylabel(f'{channel} \n After')

                axes_zoom[channel_idx * 2].set_ylabel(f'{channel} \n Before')
                axes_zoom[channel_idx * 2 + 1].set_ylabel(f'{channel} \n After')

                channel_idx += 1


            # 
            fig.suptitle(f"Record: {record_id} - Step: {step_id} Visualization", fontsize=16)
            fig.savefig(f"outputs/img/visualization_record_{record_id}_step_{step_id}.png")
            plt.close(fig)

            # limit x axis to zoom snippet length
            for ax in axes_zoom:
                ax.set_xlim(0, _zoom_snippet)
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