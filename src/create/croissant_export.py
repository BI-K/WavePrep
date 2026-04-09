"""Croissant export for WavePrep datasets.

This module converts the split WavePrep dataset into a JSONL-backed Croissant
definition that can be parsed by `mlcroissant` and consumed by TFDS'
`CroissantBuilder`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd

from common.pipeline_config import PipelineConfig

FORMAT_SUFFIXES: Dict[str, str] = {
    'csv': '.csv',
    'edf': '.edf',
    'matlab': '.mat',
    'wav': '.wav',
    'wfdb': '.dat',
    'mlcroissant': '.csv',
}

LICENSE_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
PLACEHOLDER_DATASET_NAME = "PLACEHOLDER_DATASET_NAME"
PLACEHOLDER_DATASET_URL = "https://example.com/placeholder-dataset-url"
PLACEHOLDER_CREATOR = "PLACEHOLDER_CREATOR"
PLACEHOLDER_DATE_PUBLISHED = "1970-01-01"
MANIFEST_SUBDIR = Path("data") / "mlcroissant"
MANIFEST_FILENAME = "samples.jsonl"
METADATA_FILENAME = "mlcroissant_metadata.jsonld"
VALIDATION_REPORT_FILENAME = "mlcroissant_validation.json"


def export_croissant_dataset(
    config: PipelineConfig,
    output_manager,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Export the processed dataset as a Croissant JSON-LD package."""
    if not config.output.is_croissant_export:
        return {}

    _ensure_croissant_dependencies()

    base_dir = Path(config.output.base_dir)
    reports_dir = output_manager.get_reports_directory()
    manifest_dir = base_dir / MANIFEST_SUBDIR
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / MANIFEST_FILENAME
    metadata_path = reports_dir / METADATA_FILENAME
    validation_report_path = reports_dir / VALIDATION_REPORT_FILENAME

    logger.info("Building MLCroissant manifest from split outputs")
    stats = _write_manifest(config, base_dir, manifest_path)

    hf_repo_id = config.output.hf_repo_id
    if hf_repo_id:
        logger.info(f"Uploading Croissant data to HuggingFace Hub: {hf_repo_id}")
        _upload_to_hf_hub(hf_repo_id, manifest_dir, logger)

    logger.info("Writing MLCroissant dataset metadata")
    metadata = _build_metadata(
        config=config,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        stats=stats,
    )
    metadata_path.write_text(
        json.dumps(metadata.to_json(), indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("Validating MLCroissant metadata and TFDS compatibility")
    validation_results = validate_croissant_dataset(metadata_path, logger)
    validation_report_path.write_text(
        json.dumps(validation_results, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(f"MLCroissant export complete: {metadata_path}")
    return {
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "validation_report_path": str(validation_report_path),
        "sample_count": stats["sample_count"],
        "split_counts": dict(stats["split_counts"]),
    }


def validate_croissant_dataset(
    metadata_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Validate the generated Croissant JSON-LD.

    Validation happens in three layers:
    1. `mlcroissant` parsing.
    2. Materializing records from the primary record set.
    3. TFDS `CroissantBuilder` smoke validation, with `download_and_prepare`
       executed only when `apache_beam` is installed.
    """
    import mlcroissant as mlc
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core import file_adapters

    results: Dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "mlcroissant": {},
        "tfds": {},
    }

    dataset = mlc.Dataset(jsonld=metadata_path)
    record_set_ids = [record_set.id for record_set in dataset.metadata.record_sets]
    results["mlcroissant"] = {
        "status": "passed",
        "record_set_ids": record_set_ids,
        "record_count": sum(1 for _ in dataset.records("samples")),
    }

    builder = tfds.core.dataset_builders.CroissantBuilder(
        jsonld=metadata_path,
        file_format=file_adapters.FileFormat.ARRAY_RECORD,
        data_dir=metadata_path.parent / ".tfds_validation",
    )
    tfds_result: Dict[str, Any] = {
        "status": "passed",
        "builder_name": builder.name,
        "config_name": builder.builder_config.name,
        "feature_keys": list(builder.info.features.keys()),
    }

    try:
        import apache_beam  # noqa: F401
    except ModuleNotFoundError:
        tfds_result["download_and_prepare"] = "skipped_missing_apache_beam"
        logger.warning(
            "Skipping TFDS download_and_prepare for "
            f"{metadata_path} because apache_beam is not installed"
        )
    else:
        validation_data_dir = Path(builder.data_dir)
        builder.download_and_prepare()
        split_names = list(builder.info.splits.keys())
        tfds_result["download_and_prepare"] = "passed"
        tfds_result["split_names"] = split_names
        if split_names:
            data_source = builder.as_data_source(split=split_names[0])
            first_example = data_source[0]
            tfds_result["example_keys"] = sorted(first_example.keys())
        shutil.rmtree(validation_data_dir, ignore_errors=True)

    results["tfds"] = tfds_result
    return results


def _upload_to_hf_hub(
    repo_id: str,
    local_dir: Path,
    logger: logging.Logger,
) -> None:
    """Upload Croissant data directory to a HuggingFace Hub dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "HuggingFace Hub upload requires the 'huggingface_hub' package. "
            "Install it with: pip install huggingface_hub"
        )

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=MANIFEST_SUBDIR.as_posix(),
    )
    logger.info(f"Uploaded Croissant data to https://huggingface.co/datasets/{repo_id}")


def _resolve_content_url(
    hf_repo_id: str | None,
    manifest_relative_path: Path,
) -> str:
    """Return a resolvable contentUrl for the Croissant manifest.

    When *hf_repo_id* is set, returns a stable HuggingFace Hub URL.
    Otherwise falls back to the local relative path.
    """
    if hf_repo_id:
        remote_path = (MANIFEST_SUBDIR / MANIFEST_FILENAME).as_posix()
        return (
            f"https://huggingface.co/datasets/{hf_repo_id}"
            f"/resolve/main/{remote_path}"
        )
    return manifest_relative_path.as_posix()


def _ensure_croissant_dependencies() -> None:
    """Raise a clear error when Croissant export dependencies are missing."""
    missing = []
    try:
        import mlcroissant  # noqa: F401
    except ModuleNotFoundError:
        missing.append("mlcroissant")
    try:
        import tensorflow_datasets  # noqa: F401
    except ModuleNotFoundError:
        missing.append("tensorflow-datasets")

    if missing:
        missing_joined = ", ".join(missing)
        raise ModuleNotFoundError(
            "MLCroissant export requires the following packages: "
            f"{missing_joined}"
        )


def _write_manifest(
    config: PipelineConfig,
    base_dir: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "sample_count": 0,
        "split_counts": Counter(),
        "sample_counts_by_subject": Counter(),
        "subject_ids": set(),
        "record_ids": set(),
        "channel_names": [],
        "total_duration_seconds": 0.0,
        "observation_rows": None,
        "prediction_rows": None,
        "sampling_rate_hz": None,
    }

    sample_entries = _iter_split_sample_entries(config=config, base_dir=base_dir)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for entry in sample_entries:
            manifest_file.write(json.dumps(entry) + "\n")
            stats["sample_count"] += 1
            stats["split_counts"][entry["split"]] += 1
            stats["sample_counts_by_subject"][entry["subject_id"]] += 1
            stats["subject_ids"].add(entry["subject_id"])
            stats["record_ids"].add(entry["record_id"])
            stats["total_duration_seconds"] += entry["duration_seconds"]

            if not stats["channel_names"]:
                stats["channel_names"] = entry["channel_names"]
            if stats["observation_rows"] is None:
                stats["observation_rows"] = len(entry["observation"])
            if stats["prediction_rows"] is None:
                stats["prediction_rows"] = len(entry["prediction"])
            if stats["sampling_rate_hz"] is None:
                stats["sampling_rate_hz"] = entry["sampling_rate_hz"]

    if stats["sample_count"] == 0:
        raise ValueError(
            "No split sample files were found to export as MLCroissant."
        )

    return stats


def _read_sample_data(
    path: Path,
    save_format: str,
) -> tuple[list[list[float]], list[str]]:
    """Read a sample file and return (rows_as_nested_lists, channel_names).

    Each supported save format has a reader that normalises the data into a
    uniform shape: a list of rows where each row is a list of channel values,
    plus an ordered list of channel name strings.
    """
    if save_format in ('csv', 'mlcroissant'):
        df = pd.read_csv(path)
        return df.values.tolist(), list(df.columns)

    if save_format == 'edf':
        import pyedflib

        reader = pyedflib.EdfReader(str(path))
        try:
            n = reader.signals_in_file
            signals = [reader.readSignal(i) for i in range(n)]
            channels = [reader.getLabel(i).strip() for i in range(n)]
            return np.column_stack(signals).tolist(), channels
        finally:
            reader.close()

    if save_format == 'matlab':
        from scipy.io import loadmat

        contents = loadmat(str(path))
        arrays = {
            k: v for k, v in contents.items()
            if not k.startswith('__') and isinstance(v, np.ndarray)
        }
        _, value = max(arrays.items(), key=lambda item: item[1].size)
        arr = np.asarray(value)
        if arr.ndim == 2:
            arr = arr.T  # MATLAB stores (channels, samples)
        channels = [f"ch_{i}" for i in range(arr.shape[-1] if arr.ndim > 1 else 1)]
        return arr.tolist(), channels

    if save_format == 'wav':
        import soundfile as sf

        data, _ = sf.read(str(path), always_2d=True)
        sidecar = path.with_suffix('.wav.json')
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding='utf-8'))
            channels = meta.get('channel_names', [])
        else:
            channels = [f"ch_{i}" for i in range(data.shape[1])]
        return data.tolist(), channels

    if save_format == 'wfdb':
        import wfdb

        record = wfdb.rdrecord(str(path.with_suffix('')))
        data = record.p_signal if record.p_signal is not None else record.d_signal
        return np.asarray(data).tolist(), list(record.sig_name)

    raise ValueError(f"Unsupported format for Croissant manifest: {save_format}")


def _iter_split_sample_entries(
    config: PipelineConfig,
    base_dir: Path,
) -> Iterator[Dict[str, Any]]:
    data_dir = base_dir / "data"
    save_format = config.output.effective_save_format
    suffix = FORMAT_SUFFIXES.get(save_format)
    if suffix is None:
        raise ValueError(
            f"No known file suffix for save_format '{save_format}'. "
            f"Supported formats: {list(FORMAT_SUFFIXES.keys())}"
        )

    sampling_rate_hz = (
        1.0 / config.windowing.expected_resolution
        if config.windowing.expected_resolution
        else 1.0
    )

    for split_name in ("train", "validation", "test"):
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue

        for subject_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            observation_dir = subject_dir / "observation"
            prediction_dir = subject_dir / "prediction"
            if not observation_dir.exists():
                continue

            observation_files = sorted(
                path for path in observation_dir.iterdir() if path.suffix == suffix
            )
            for observation_file in observation_files:
                prediction_file = prediction_dir / observation_file.name
                if not prediction_file.exists():
                    raise FileNotFoundError(
                        f"Missing prediction file for {observation_file}"
                    )

                observation, obs_channels = _read_sample_data(observation_file, save_format)
                prediction, pred_channels = _read_sample_data(prediction_file, save_format)
                if obs_channels != pred_channels:
                    raise ValueError(
                        "Observation/prediction channels differ for "
                        f"{observation_file.name}"
                    )

                record_id, sample_index = _parse_sample_filename(observation_file.stem)
                duration_seconds = (
                    (len(observation) + len(prediction)) / sampling_rate_hz
                    if sampling_rate_hz
                    else 0.0
                )

                yield {
                    "sample_id": f"{split_name}/{subject_dir.name}/{observation_file.stem}",
                    "sample_index": sample_index,
                    "subject_id": subject_dir.name,
                    "record_id": record_id,
                    "split": split_name,
                    "channel_names": obs_channels,
                    "observation": observation,
                    "prediction": prediction,
                    "observation_rows": len(observation),
                    "prediction_rows": len(prediction),
                    "sampling_rate_hz": sampling_rate_hz,
                    "duration_seconds": duration_seconds,
                }


def _parse_sample_filename(stem: str) -> tuple[str, int]:
    marker = "_sample_"
    if marker not in stem:
        return stem, 0
    record_id, sample_suffix = stem.rsplit(marker, 1)
    try:
        return record_id, int(sample_suffix)
    except ValueError:
        return record_id, 0


def _build_metadata(
    config: PipelineConfig,
    metadata_path: Path,
    manifest_path: Path,
    stats: Dict[str, Any],
):
    import mlcroissant as mlc

    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    manifest_relative_path = Path(os.path.relpath(manifest_path, start=metadata_path.parent))
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    channel_count = len(stats["channel_names"])
    observation_shape = (
        f"-1,{channel_count}" if channel_count else "-1"
    )
    prediction_shape = f"-1,{channel_count}" if channel_count else "-1"

    description = _build_description(config, stats)
    keywords = _build_keywords(config, stats["channel_names"])
    content_url = _resolve_content_url(config.output.hf_repo_id, manifest_relative_path)

    return mlc.Metadata(
        name=PLACEHOLDER_DATASET_NAME,
        description=description,
        url=PLACEHOLDER_DATASET_URL,
        creators=[mlc.Organization(name=PLACEHOLDER_CREATOR)],
        date_published=PLACEHOLDER_DATE_PUBLISHED,
        date_created=timestamp,
        date_modified=timestamp,
        version="1.0.0",
        license=LICENSE_URL,
        keywords=keywords,
        cite_as=(
            "Generated by WavePrep from "
            f"{config.database_name}. Replace placeholder dataset identity fields "
            "before publication."
        ),
        conforms_to=["http://mlcommons.org/croissant/1.1"],
        distribution=[
            mlc.FileObject(
                id="samples_jsonl",
                name=MANIFEST_FILENAME,
                description="WavePrep sample manifest with observation/prediction arrays.",
                content_url=content_url,
                encoding_formats=["application/jsonlines"],
                sha256=manifest_sha256,
            )
        ],
        record_sets=[
            mlc.RecordSet(
                id="samples",
                name="samples",
                description="Windowed WavePrep samples grouped by split and subject.",
                fields=[
                    mlc.Field(
                        id="samples/sample_id",
                        name="sample_id",
                        description="Unique sample identifier.",
                        data_types=mlc.DataType.TEXT,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="sample_id"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/sample_index",
                        name="sample_index",
                        description="Zero-based sample index within the source record.",
                        data_types=mlc.DataType.INTEGER,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="sample_index"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/subject_id",
                        name="subject_id",
                        description="MIMIC subject identifier.",
                        data_types=mlc.DataType.TEXT,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="subject_id"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/record_id",
                        name="record_id",
                        description="Source waveform record identifier.",
                        data_types=mlc.DataType.TEXT,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="record_id"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/split",
                        name="split",
                        description="Dataset split assignment.",
                        data_types=mlc.DataType.TEXT,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="split"),
                        ),
                        references=mlc.Source(field="splits/name"),
                    ),
                    mlc.Field(
                        id="samples/channel_names",
                        name="channel_names",
                        description="Ordered signal channel names.",
                        data_types=mlc.DataType.TEXT,
                        is_array=True,
                        array_shape="-1",
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="channel_names"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/observation",
                        name="observation",
                        description="Observation window signal values.",
                        data_types=mlc.DataType.FLOAT,
                        is_array=True,
                        array_shape=observation_shape,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="observation"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/prediction",
                        name="prediction",
                        description="Prediction window signal values.",
                        data_types=mlc.DataType.FLOAT,
                        is_array=True,
                        array_shape=prediction_shape,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="prediction"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/observation_rows",
                        name="observation_rows",
                        description="Number of rows in the observation window.",
                        data_types=mlc.DataType.INTEGER,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="observation_rows"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/prediction_rows",
                        name="prediction_rows",
                        description="Number of rows in the prediction window.",
                        data_types=mlc.DataType.INTEGER,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="prediction_rows"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/sampling_rate_hz",
                        name="sampling_rate_hz",
                        description="Sampling rate of the exported windows in hertz.",
                        data_types=mlc.DataType.FLOAT,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="sampling_rate_hz"),
                        ),
                    ),
                    mlc.Field(
                        id="samples/duration_seconds",
                        name="duration_seconds",
                        description="Combined observation and prediction duration in seconds.",
                        data_types=mlc.DataType.FLOAT,
                        source=mlc.Source(
                            file_object="samples_jsonl",
                            extract=mlc.Extract(column="duration_seconds"),
                        ),
                    ),
                ],
            ),
            mlc.RecordSet(
                id="splits",
                name="splits",
                description="Allowed dataset split names.",
                key=["splits/name"],
                data_types=[mlc.DataType.SPLIT],
                fields=[
                    mlc.Field(
                        id="splits/name",
                        name="name",
                        description="Split name.",
                        data_types=mlc.DataType.TEXT,
                    )
                ],
                data=[
                    {"splits/name": split_name}
                    for split_name, count in sorted(stats["split_counts"].items())
                    if count > 0
                ],
            ),
        ],
    )


def _build_description(config: PipelineConfig, stats: Dict[str, Any]) -> str:
    split_summary = ", ".join(
        f"{split_name}={count}"
        for split_name, count in sorted(stats["split_counts"].items())
    )
    processing_summary = _summarize_processing(config)
    windowing_summary = (
        f"observation_window={config.windowing.observation_window}s, "
        f"prediction_horizon={config.windowing.prediction_horizon}s, "
        f"prediction_window={config.windowing.prediction_window}s, "
        f"step={config.windowing.step}s, "
        f"expected_resolution={config.windowing.expected_resolution}s"
    )
    channels_summary = ", ".join(stats["channel_names"])

    return (
        "WavePrep-generated dataset based on "
        f"{config.database_name}. Generated samples={stats['sample_count']}, "
        f"source waveform records={len(stats['record_ids'])}, "
        f"patients={len(stats['subject_ids'])}, "
        f"total duration={stats['total_duration_seconds']:.2f}s. "
        f"Splits: {split_summary}. "
        f"Splitting method: {config.splitting.splitting_method}. "
        f"Windowing: {windowing_summary}. "
        f"Channels included: {channels_summary}. "
        f"Processing summary: {processing_summary}."
    )


def _build_keywords(config: PipelineConfig, channel_names: Iterable[str]) -> List[str]:
    keywords = ["mimiciii", "waveform", "biosignal"]
    for channel_name in channel_names:
        if channel_name not in keywords:
            keywords.append(channel_name)
    for channel_name in config.input_channels + config.output_channels:
        if channel_name not in keywords:
            keywords.append(channel_name)
    return keywords


def _summarize_processing(config: PipelineConfig) -> str:
    step_summaries = []
    for channel_cfg in config.signal_processing:
        parts = []
        for step_cfg in channel_cfg.steps:
            step_parts = [f"step {step_cfg.step}"]
            if step_cfg.downsampling:
                step_parts.append(
                    "downsampling "
                    f"{step_cfg.downsampling.strategy} to "
                    f"{step_cfg.downsampling.desired_resolution}s"
                )
            if step_cfg.data_cleaning:
                cleaning = step_cfg.data_cleaning
                if cleaning.lower_threshold is not None or cleaning.upper_threshold is not None:
                    step_parts.append(
                        "cleaning thresholds "
                        f"[{cleaning.lower_threshold}, {cleaning.upper_threshold}]"
                    )
            if step_cfg.imputation:
                step_parts.append(f"imputation={step_cfg.imputation.method}")
            parts.append("; ".join(step_parts))

        channel_summary = f"{channel_cfg.channel}: " + " | ".join(parts)
        step_summaries.append(channel_summary)

    if config.long_nan_seq_removal:
        nan_rules = ", ".join(
            "after_step="
            f"{rule.get('after_step')} max_consecutive_nans="
            f"{rule.get('max_consecutive_nans')}"
            for rule in config.long_nan_seq_removal
        )
        step_summaries.append(f"long_nan_seq_removal: {nan_rules}")

    return " ".join(step_summaries)
