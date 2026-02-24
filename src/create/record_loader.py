"""Strategy pattern for loading waveform records from different sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import wfdb
import logging
from pathlib import Path

from common.pipeline_config import PipelineConfig
from common.processing_context import ProcessingContext
from validation.validation import analyze_nan_values


@dataclass
class LoadedRecord:
    """Standardized container returned by all record loaders."""
    signal_data: np.ndarray
    channel_names: List[str]
    metadata: ProcessingContext


class RecordLoader(ABC):
    """Strategy interface for loading waveform records."""

    @abstractmethod
    def load(self) -> Tuple[List[LoadedRecord], str]:
        """Load record data.

        Returns:
            Tuple of (loaded records, error string). An empty error string
            means success.
        """


class WfdbRecordLoader(RecordLoader):
    """Loads records from the WFDB PhysioNet database."""

    def __init__(
        self,
        record_path: str,
        offset_start: int,
        offset_end: int,
        config: PipelineConfig,
        logger: logging.Logger,
        metadata_base: ProcessingContext,
    ):
        self.record_path = record_path
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.config = config
        self.logger = logger
        self.metadata_base = metadata_base

    def load(self) -> Tuple[List[LoadedRecord], str]:
        record_id = Path(self.record_path).parts[-1]
        if "n" in record_id:
            return self._load_numeric()
        return self._load_non_numeric()

    def _load_singular(
        self,
        record_name: str,
        directory: str,
        offset_start: int,
        offset_end: int,
        required_channels: List[str],
        metadata: ProcessingContext,
    ) -> Tuple[Dict, str]:
        record = wfdb.rdrecord(record_name, pn_dir=directory)
        available_channels = record.sig_name
        channel_indices = []
        found_channels = []

        for ch in required_channels:
            if ch in available_channels:
                idx = available_channels.index(ch)
                if idx not in channel_indices:
                    channel_indices.append(idx)
                    found_channels.append(ch)

        if not channel_indices:
            return {}, (
                f"No required channels found. Required: {required_channels}, "
                f"Available: {available_channels}"
            )

        signal_data = record.p_signal[:, channel_indices]

        if offset_start is not None and offset_end is not None:
            start_sample = int(offset_start * record.fs)
            end_sample = int(offset_end * record.fs)
            signal_data = signal_data[start_sample:end_sample]

        header = wfdb.rdheader(record_name, pn_dir=directory)
        if signal_data is None:
            return {}, "No signal data available in record"

        if self.config.validation.strict_nan_check:
            nan_diagnostic = analyze_nan_values(signal_data, found_channels, header)
            return {}, f"NaN values detected in required channels: {nan_diagnostic}"

        metadata.sampling_rate = header.fs
        metadata.n_channels = header.n_sig
        metadata.available_channels = found_channels

        return {
            "signal_data": signal_data,
            "found_channels": found_channels,
            "start_offset_seconds": 0,
            "metadata": metadata,
        }, ""

    def _load_numeric(self) -> Tuple[List[LoadedRecord], str]:
        try:
            cfg = self.config
            required_channels = cfg.required_channels
            path_parts = Path(self.record_path).parts
            directory = f"{cfg.database_name}/{'/'.join(path_parts[:-1])}"
            record_name = path_parts[-1]

            raw, error = self._load_singular(
                record_name, directory,
                self.offset_start, self.offset_end,
                required_channels, self.metadata_base,
            )
            if error:
                return [], error

            return [LoadedRecord(
                signal_data=raw["signal_data"],
                channel_names=raw["found_channels"],
                metadata=raw["metadata"],
            )], ""
        except Exception as e:
            return [], f"Load error: {str(e)}"

    def _load_non_numeric(self) -> Tuple[List[LoadedRecord], str]:
        try:
            cfg = self.config
            required_channels = cfg.required_channels
            min_length = cfg.validation.min_record_duration

            path_parts = Path(self.record_path).parts
            directory = f"{cfg.database_name}/{'/'.join(path_parts[:-1])}"
            record_name = path_parts[-1]

            header = wfdb.rdheader(record_name, pn_dir=directory)
            segments = [s for s in header.seg_name if s != "~"]

            current_offset = 0
            records: List[LoadedRecord] = []
            errors = ""

            for segment in segments:
                seg_hdr = wfdb.rdheader(record_name=segment, pn_dir=directory)
                seg_duration = seg_hdr.sig_len / seg_hdr.fs

                if current_offset < self.offset_start or current_offset > self.offset_end:
                    current_offset += seg_duration
                    continue

                if (
                    all(ch in seg_hdr.sig_name for ch in required_channels)
                    and seg_duration >= min_length
                ):
                    end_off = min(
                        seg_duration,
                        self.offset_end - current_offset,
                    )
                    raw, err = self._load_singular(
                        segment, directory, 0, end_off,
                        required_channels, self.metadata_base,
                    )
                    if raw:
                        records.append(LoadedRecord(
                            signal_data=raw["signal_data"],
                            channel_names=raw["found_channels"],
                            metadata=raw["metadata"],
                        ))
                    errors += err

                current_offset += seg_duration

            return records, errors
        except Exception as e:
            self.logger.error(f"Load error in non-numeric record {self.record_path}: {e}")
            return [], f"Load error: {str(e)}"


class CsvRecordLoader(RecordLoader):
    """Loads a single waveform record from a CSV file."""

    def __init__(self, file_path: str, metadata: ProcessingContext):
        self.file_path = file_path
        self.metadata = metadata

    def load(self) -> Tuple[List[LoadedRecord], str]:
        try:
            df = pd.read_csv(self.file_path)
            return [LoadedRecord(
                signal_data=df.to_numpy(),
                channel_names=list(df.columns),
                metadata=self.metadata,
            )], ""
        except Exception as e:
            return [], f"Load error: {str(e)}"
