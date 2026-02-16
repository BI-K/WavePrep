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
image_width = 30
image_height = 15
number_of_records_to_visualize = 20

_window_obs_color = "#A35ACD"
_window_pred_color = "#FF8C00"

_imputation_fill_color = "#00A388"

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

def _add_text(img, text, header_h, font_scale, position="top"):
    """Helper to add (header/footer) to image."""
    banner_img = np.full((header_h, img.shape[1], 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = max(10, (img.shape[1] - text_w) // 2)
    text_y = (header_h + text_h) // 2
    cv2.putText(banner_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    if position == "bottom":
        return cv2.vconcat([img, banner_img])
    else:
        return cv2.vconcat([banner_img, img])
    
def _highlight_nan_background(ax, data, color='grey', alpha=0.5):
    data = np.asarray(data)
    if data.size == 0:
        return

    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return

    nan_idx = np.where(nan_mask)[0]
    start = nan_idx[0]
    prev = nan_idx[0]

    for idx in nan_idx[1:]:
        if idx != prev + 1:
            ax.axvspan(start - 0.5, prev + 0.5, color=color, alpha=alpha, linewidth=0)
            start = idx
        prev = idx

    ax.axvspan(start - 0.5, prev + 0.5, color=color, alpha=alpha, linewidth=0)
    
def _hex_to_bgr_inline(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)
    
def _add_legend(img, legend_h=260):
    """
    Append a legend banner at the bottom of a merged visualization image.
    """
    w = img.shape[1]

    # scale legend visuals for very wide merged images
    # (keeps text readable when w is large)
    scale = max(1.0, min(1.8, w / 1600.0))

    legend = np.full((legend_h, w, 3), 255, dtype=np.uint8)

    # subtle top separator
    cv2.line(legend, (0, 0), (w, 0), (220, 220, 220), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, int(round(2 * scale)))

    pad_x = int(round(30 * scale))
    pad_y = int(round(18 * scale))

    title_scale = 0.95 * scale
    label_scale = 0.80 * scale
    expl_scale = 0.75 * scale

    # Title
    cv2.putText(
        legend,
        "Legend",
        (pad_x, pad_y + int(round(22 * scale))),
        font,
        title_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    # BGR colors for OpenCV drawing
    blue = (255, 0, 0)
    red = (0, 0, 255)
    gray = (155, 155, 155)

    window_obs_bgr = _hex_to_bgr_inline(_window_obs_color)
    window_pred_bgr = _hex_to_bgr_inline(_window_pred_color)
    imputed_bgr = _hex_to_bgr_inline(_imputation_fill_color)

    # layout (scaled)
    icon_w = int(round(118 * scale))
    icon_h = int(round(46 * scale))
    gap = int(round(14 * scale))
    item_gap = int(round(28 * scale))
    row_gap = int(round(18 * scale))

    x = pad_x
    y = pad_y + int(round(44 * scale))

    items = [
        ("before", "Before step"),
        ("after", "After step"),
        ("downsampling", "Downsampling bins"),
        ("nan", "NaN values"),
        ("imputed", "Imputed values"),

        ("row_break", ""),

        ("window_obs", "Observation window"),
        ("window_pred", "Prediction window"),

        ("row_break", ""),

        ("long_nan_keep", "Included"),
        ("long_nan_excl_long", "Excluded (long NaN)"),
        ("long_nan_excl_short", "Excluded (too short)"),
    ]

    for kind, label in items:
        if kind == "row_break":
            x = pad_x
            y += icon_h + row_gap
            continue

        (text_w, _), _ = cv2.getTextSize(label, font, label_scale, thickness)
        needed_w = icon_w + gap + text_w + item_gap

        # wrap row if needed
        if x + needed_w > w - pad_x:
            x = pad_x
            y += icon_h + row_gap

        # extend banner if needed (robust for narrow images)
        if y + icon_h + int(round(14 * scale)) > legend.shape[0]:
            extra_h = (y + icon_h + int(round(14 * scale))) - legend.shape[0]
            legend = cv2.copyMakeBorder(
                legend, 0, extra_h, 0, 0,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )

        # icon frame
        cv2.rectangle(legend, (x, y), (x + icon_w, y + icon_h), (245, 245, 245), -1)
        cv2.rectangle(legend, (x, y), (x + icon_w, y + icon_h), (225, 225, 225), 1)

        cy = y + icon_h // 2

        # draw icon content
        if kind == "before":
            x1 = x + int(round(0.12 * icon_w))
            x2 = x + int(round(0.88 * icon_w))
            cv2.line(legend, (x1, cy), (x2, cy), blue, max(2, int(round(3 * scale))))
            for fx in (0.25, 0.50, 0.75):
                px = x + int(round(fx * icon_w))
                d = int(round(0.10 * icon_h))
                cv2.line(legend, (px - d, cy - d), (px + d, cy + d), blue, max(1, int(round(2 * scale))))
                cv2.line(legend, (px - d, cy + d), (px + d, cy - d), blue, max(1, int(round(2 * scale))))

        elif kind == "after":
            x1 = x + int(round(0.12 * icon_w))
            x2 = x + int(round(0.88 * icon_w))
            cv2.line(legend, (x1, cy), (x2, cy), red, max(2, int(round(3 * scale))))
            # hint that markers may exist
            px = x + icon_w // 2
            d = int(round(0.09 * icon_h))
            cv2.line(legend, (px - d, cy - d), (px + d, cy + d), red, 1)
            cv2.line(legend, (px - d, cy + d), (px + d, cy - d), red, 1)

        elif kind == "downsampling":
            for fx in (0.30, 0.50, 0.70):
                px = x + int(round(fx * icon_w))
                cv2.line(
                    legend,
                    (px, y + int(round(0.15 * icon_h))),
                    (px, y + icon_h - int(round(0.15 * icon_h))),
                    gray,
                    max(1, int(round(2 * scale)))
                )

        elif kind == "nan":
            rx1 = x + int(round(0.23 * icon_w))
            rx2 = x + int(round(0.77 * icon_w))
            ry1 = y + int(round(0.18 * icon_h))
            ry2 = y + icon_h - int(round(0.18 * icon_h))
            cv2.rectangle(legend, (rx1, ry1), (rx2, ry2), (190, 190, 190), -1)
            cv2.rectangle(legend, (rx1, ry1), (rx2, ry2), (160, 160, 160), 1)

        elif kind == "imputed":
            # thin band like axvspan (teal)
            rx1 = x + int(round(0.23 * icon_w))
            rx2 = x + int(round(0.77 * icon_w))
            ry1 = y + int(round(0.18 * icon_h))
            ry2 = y + icon_h - int(round(0.18 * icon_h))
            cv2.rectangle(legend, (rx1, ry1), (rx2, ry2), imputed_bgr, -1)
            cv2.rectangle(legend, (rx1, ry1), (rx2, ry2), (120, 120, 120), 1)

        elif kind == "window_obs":
            # single band (observation)
            rx1 = x + int(round(0.14 * icon_w))
            rx2 = x + int(round(0.86 * icon_w))
            ry1 = y + int(round(0.38 * icon_h))
            ry2 = y + int(round(0.62 * icon_h))
            cv2.rectangle(legend, (rx1, ry1), (rx2, ry2), window_obs_bgr, -1)
            # end caps for clarity
            cv2.line(legend, (rx1, ry1), (rx1, ry2), (120, 120, 120), 1)
            cv2.line(legend, (rx2, ry1), (rx2, ry2), (120, 120, 120), 1)

        elif kind == "window_pred":
            # single band (prediction)
            rx1 = x + int(round(0.14 * icon_w))
            rx2 = x + int(round(0.86 * icon_w))
            ry1 = y + int(round(0.38 * icon_h))
            ry2 = y + int(round(0.62 * icon_h))
            cv2.rectangle(legend, (rx1, ry1), (rx2, ry2), window_pred_bgr, -1)
            cv2.line(legend, (rx1, ry1), (rx1, ry2), (120, 120, 120), 1)
            cv2.line(legend, (rx2, ry1), (rx2, ry2), (120, 120, 120), 1)

        elif kind == "long_nan_keep":
            # red bounds + green fill
            sx1 = x + int(round(0.20 * icon_w))
            sx2 = x + int(round(0.80 * icon_w))
            sy1 = y + int(round(0.30 * icon_h))
            sy2 = y + int(round(0.70 * icon_h))
            cv2.rectangle(legend, (sx1, sy1), (sx2, sy2), (120, 230, 120), -1)
            cv2.line(legend, (sx1, sy1 - 1), (sx1, sy2 + 1), red, max(1, int(round(2 * scale))))
            cv2.line(legend, (sx2, sy1 - 1), (sx2, sy2 + 1), red, max(1, int(round(2 * scale))))

        elif kind == "long_nan_excl_long":
            # red bounds + white fill (i.e., no fill), but add a subtle inner outline so it's visible
            sx1 = x + int(round(0.20 * icon_w))
            sx2 = x + int(round(0.80 * icon_w))
            sy1 = y + int(round(0.30 * icon_h))
            sy2 = y + int(round(0.70 * icon_h))
            cv2.rectangle(legend, (sx1, sy1), (sx2, sy2), (255, 255, 255), -1)
            cv2.rectangle(legend, (sx1, sy1), (sx2, sy2), (210, 210, 210), 1)
            cv2.line(legend, (sx1, sy1 - 1), (sx1, sy2 + 1), red, max(1, int(round(2 * scale))))
            cv2.line(legend, (sx2, sy1 - 1), (sx2, sy2 + 1), red, max(1, int(round(2 * scale))))

        elif kind == "long_nan_excl_short":
            # red bounds + light-blue fill (too short)
            sx1 = x + int(round(0.20 * icon_w))
            sx2 = x + int(round(0.80 * icon_w))
            sy1 = y + int(round(0.30 * icon_h))
            sy2 = y + int(round(0.70 * icon_h))
            cv2.rectangle(legend, (sx1, sy1), (sx2, sy2), (255, 0, 0), -1)  # light blue (BGR)
            cv2.line(legend, (sx1, sy1 - 1), (sx1, sy2 + 1), red, max(1, int(round(2 * scale))))
            cv2.line(legend, (sx2, sy1 - 1), (sx2, sy2 + 1), red, max(1, int(round(2 * scale))))

        # label
        cv2.putText(
            legend,
            label,
            (x + icon_w + gap, y + icon_h - int(round(12 * scale))),
            font,
            label_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA
        )

        x += needed_w

    return cv2.vconcat([img, legend])

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
            axes[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color=_imputation_fill_color, alpha=0.5)

        _highlight_nan_background(axes[channel_id * 2], data_before)
        _highlight_nan_background(axes[channel_id * 2 + 1], data_after)

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
                axes_zoom[channel_id * 2 + 1].axvspan(nan_index - 0.5, nan_index + 0.5, color=_imputation_fill_color, alpha=0.5)

            _highlight_nan_background(axes_zoom[channel_id * 2], data_before_zoom)
            _highlight_nan_background(axes_zoom[channel_id * 2 + 1], data_after_zoom)

            axes_zoom = beautify_axes(axes_zoom, channel_id, data_before_zoom, data_after_zoom)

        return fig, fig_zoom, axes, axes_zoom



def _visualize_data_cleaning_for_channel_record(fig, axes, fig_zoom, axes_zoom, record_id: str, step_id: int, channel_id, data_before, data_after):
        
        # plot two lines one for data_before and one for data_after
        axes[channel_id * 2].plot(data_before, label='Before Data Cleaning', alpha=0.7, color='blue')
        axes[channel_id * 2].scatter(range(len(data_before)), data_before, label='Before Data Cleaning', alpha=0.7, color='blue', s=10, marker='x')
        axes[channel_id * 2 + 1].plot(data_after, label='After Data Cleaning', alpha=0.7, color='red')
        axes[channel_id * 2 + 1].scatter(range(len(data_after)), data_after, label='After Data Cleaning', alpha=0.7, color='red', s=10, marker='x')

        _highlight_nan_background(axes[channel_id * 2], data_before)
        _highlight_nan_background(axes[channel_id * 2 + 1], data_after)

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

            _highlight_nan_background(axes_zoom[channel_id * 2], data_before_zoom)
            _highlight_nan_background(axes_zoom[channel_id * 2 + 1], data_after_zoom)

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
        
        _highlight_nan_background(axes[channel_id * 2], data_before)
        _highlight_nan_background(axes[channel_id * 2 + 1], data_after_expanded)

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

            _highlight_nan_background(axes_zoom[channel_id * 2], data_before_zoom)
            _highlight_nan_background(axes_zoom[channel_id * 2 + 1], data_after_zoom)

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
        _highlight_nan_background(axes[channel_idx], data_array[channel])
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

    full_window_size = observation_window_size + prediction_horizon_size + prediction_window_size
    total_size_max_ten_windows = math.ceil((number_of_windows_to_visualize - 1) * step_size + full_window_size)
    zoom_snippet_full_view = min(len(data_array[channels[0]]), total_size_max_ten_windows)
    zoom_snippet_zoom = min(len(data_array[channels[0]]), _zoom_snippet)
    zoom_levels = [zoom_snippet_full_view, zoom_snippet_zoom]

    for zoom_idx in range(len(zoom_levels)):

        snipped_data_array = dict()
        for channel in channels:
            snipped_data_array[channel] = data_array[channel][:zoom_levels[zoom_idx]]
        fig, axes = _visualize_all_channels_for_record(snipped_data_array, number_of_additional_subplots=1, image_width=image_width / 2, image_height=image_height)

        # only show for a limited number of windows - e.g. 10
        if step_size <= 0:
            curr_number_of_windows_to_visualize = 0
        else:
            curr_number_of_windows_to_visualize = min(number_of_windows_to_visualize, int(math.floor((zoom_levels[zoom_idx] - 1) / step_size) + 1))

        for i in range(curr_number_of_windows_to_visualize):

            obs_start = step_size * i
            obs_end = obs_start + observation_window_size
            pred_start = obs_end + prediction_horizon_size
            pred_end = pred_start + prediction_window_size

            # clip to zoom length (fixes partial windows properly for overlap)
            obs_end_clip = min(obs_end, zoom_levels[zoom_idx])
            pred_end_clip = min(pred_end, zoom_levels[zoom_idx])

            # observation window - ONLY horizontal bars, NO vertical lines on axes[-1]
            if obs_start < zoom_levels[zoom_idx]:
                axes[-1].hlines(y=i, xmin=obs_start, xmax=obs_end_clip, color=_window_obs_color, linewidth=3, alpha=0.7)

            # prediction window - ONLY horizontal bars, NO vertical lines on axes[-1]
            if pred_start < zoom_levels[zoom_idx]:
                axes[-1].hlines(y=i, xmin=pred_start, xmax=pred_end_clip, color=_window_pred_color, linewidth=3, alpha=0.7)

            # vertical lines ONLY on channel subplots (axes[:-1])
            for ax in axes[:-1]:
                if obs_start < zoom_levels[zoom_idx]:
                    ax.axvline(x=obs_start, color=_window_obs_color, linestyle='-', alpha=0.7, linewidth=0.8)
                if obs_end_clip <= zoom_levels[zoom_idx]:
                    ax.axvline(x=obs_end, color=_window_obs_color, linestyle='-', alpha=0.7, linewidth=0.8)
                if pred_start < zoom_levels[zoom_idx]:
                    ax.axvline(x=pred_start, color=_window_pred_color, linestyle='-', alpha=0.7, linewidth=0.8)
                if pred_end_clip <= zoom_levels[zoom_idx]:
                    ax.axvline(x=pred_end, color=_window_pred_color, linestyle='-', alpha=0.7, linewidth=0.8)

            y_top = -0.5
            if obs_start < zoom_levels[zoom_idx]:
                axes[-1].vlines(obs_start, y_top, i, color=_window_obs_color, alpha=0.7, linewidth=0.8)
            if obs_end_clip <= zoom_levels[zoom_idx]:
                axes[-1].vlines(obs_end_clip, y_top, i, color=_window_obs_color, alpha=0.7, linewidth=0.8)
            if pred_start < zoom_levels[zoom_idx]:
                axes[-1].vlines(pred_start, y_top, i, color=_window_pred_color, alpha=0.7, linewidth=0.8)
            if pred_end_clip <= zoom_levels[zoom_idx]:
                axes[-1].vlines(pred_end_clip, y_top, i, color=_window_pred_color, alpha=0.7, linewidth=0.8)

        # configure the window visualization subplot
        if curr_number_of_windows_to_visualize > 0:
            axes[-1].set_ylim(-0.5, curr_number_of_windows_to_visualize - 0.5)
            axes[-1].set_ylabel('Window')
            axes[-1].invert_yaxis()  # window 0 at top

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


def visualize_step_for_record( record_id: str, step_id: int, processed_data_array_before, processed_data_array_after, signal_processing_config, output_path="outputs/img/", windowing_config=None):

        if record_id and processed_data_array_before and processed_data_array_after:

            # only visualize first entry of processed_data_array_before and processed_data_array_after
            data_before = processed_data_array_before[0]
            data_after = processed_data_array_after[0]

            channels_before = list(data_before.keys())
            channels_after = list(data_after.keys())

            # Calculate figure width: use sqrt-based ratio for observation/prediction records
            fig_width = image_width / 2  # default: half width (two figures side by side = full width)
            if windowing_config and ("observation" in record_id or "prediction" in record_id):
                obs_w = windowing_config.get('observation_window', 3600)
                pred_w = windowing_config.get('prediction_window', 300)
                sqrt_obs = math.sqrt(max(obs_w, 1))
                sqrt_pred = math.sqrt(max(pred_w, 1))
                if "observation" in record_id:
                    fig_width = image_width * sqrt_obs / (sqrt_obs + sqrt_pred)
                else:
                    fig_width = image_width * sqrt_pred / (sqrt_obs + sqrt_pred)

            fig = plt.figure(figsize=(fig_width, image_height))
            fig_zoom = plt.figure(figsize=(fig_width, image_height))
            gs = fig.add_gridspec(2 * len(channels_before), hspace=0, figure=fig)
            axes = gs.subplots(sharex=True, sharey=False)
            gs_zoom = fig_zoom.add_gridspec(2 * len(channels_before), hspace=0, figure=fig_zoom)
            axes_zoom = gs_zoom.subplots(sharex=True, sharey=False)

            # For obs/pred records, enforce fixed absolute-pixel margins matching
            # normal figures (prevents content center shift in merged reports).
            # Matplotlib uses proportional margins (left=12.5%, right=10%) which
            # create different absolute pixel margins for different figure widths,
            # shifting the combined obs+pred content envelope to the right.
            if windowing_config and ("observation" in record_id or "prediction" in record_id):
                ref_half = image_width / 2  # normal figure width reference (15 inches)
                left_frac = 0.125 * ref_half / fig_width
                right_frac = 1 - 0.1 * ref_half / fig_width
                fig.subplots_adjust(left=left_frac, right=right_frac)
                fig_zoom.subplots_adjust(left=left_frac, right=right_frac)
            
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



def merge_step_visualizations_for_record(record_id, start_offset_seconds, end_offset_seconds, signal_processing_config, output_path="outputs/img/", windowing_config=None):

    if (not "observation" in record_id) and (not "prediction" in record_id):

        # read all file-names from a path
        
        files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
        record_files = [f for f in files if f"visualization_record_{record_id}_" in f]

        # sort by step id
        record_files.sort(key=lambda x: int(x.split(".png")[0].split("_step_")[1].split("_")[0]))

        visualized_steps = [x.split(".png")[0].split("_step_")[1].split("_")[0] for x in record_files]
        visualized_steps = list(set(visualized_steps))
        visualized_steps.sort(key=lambda x: int(x))

        expected_last_step = -1
        for ch_cfg in signal_processing_config:
            for s in ch_cfg.get("steps", []):
                if isinstance(s.get("step", None), int):
                    expected_last_step = max(expected_last_step, s["step"])

        observed_steps_int = [int(s) for s in visualized_steps] if len(visualized_steps) > 0 else []
        record_excluded = (expected_last_step >= 0 and expected_last_step not in observed_steps_int)

        has_windowing = any("windowing" in f for f in record_files)
        if not has_windowing:
            record_excluded = True

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
                # For observation + prediction pairs, match heights before concat
                # (they may differ slightly due to sqrt-based figsize)
                is_obs_pred = (len(images) == 2 and len(image_file_group) == 2 and
                               any("observation" in f for f in image_file_group) and
                               any("prediction" in f for f in image_file_group))
                if is_obs_pred and len(images) == 2:
                    target_h = max(images[0].shape[0], images[1].shape[0])
                    for k in range(2):
                        if images[k].shape[0] != target_h:
                            # Scale width proportionally to match target height
                            scale = target_h / images[k].shape[0]
                            new_w = int(images[k].shape[1] * scale)
                            images[k] = cv2.resize(images[k], (new_w, target_h))

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
                    merged_img = _add_text(merged_img, row_title, 80, 1.5)
                    
                images_to_append_vertically.append(merged_img)

        merged_img = None
        if images_to_append_vertically:
            # Find the maximum width
            max_width = max(img.shape[1] for img in images_to_append_vertically)
            
            # Center-pad narrower images to max_width with white background
            # (avoids stretching/distorting the plot content)
            resized_images = []
            for img in images_to_append_vertically:
                if img.shape[1] != max_width:
                    pad_total = max_width - img.shape[1]
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    padded_img = cv2.copyMakeBorder(
                        img, 0, 0, pad_left, pad_right,
                        cv2.BORDER_CONSTANT, value=(255, 255, 255)
                    )
                    resized_images.append(padded_img)
                else:
                    resized_images.append(img)
            
            merged_img = cv2.vconcat(resized_images)

        if merged_img is not None:
            super_title = f"Record: {record_id} | {start_offset_seconds}s -> {end_offset_seconds}s"
            merged_img = _add_text(merged_img, super_title, 160, 1.8)

            if record_excluded:
                merged_img = _add_text(
                    merged_img,
                    "No data remaining for further processing.",
                    header_h=160,
                    font_scale=1.8,
                    position="bottom"
                )

            merged_img = _add_legend(merged_img)

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
