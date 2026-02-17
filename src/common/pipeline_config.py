"""Typed configuration for the dataset processing pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class WindowingConfig:
    observation_window: int = 3600
    prediction_horizon: int = 300
    prediction_window: int = 1800
    step: int = 300
    expected_resolution: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WindowingConfig':
        return cls(
            observation_window=d.get('observation_window', 3600),
            prediction_horizon=d.get('prediction_horizon', 300),
            prediction_window=d.get('prediction_window', 1800),
            step=d.get('step', 300),
            expected_resolution=d.get('expected_resolution', 1.0),
        )


@dataclass
class ValidationConfig:
    min_record_duration: int = 7200
    validate_channels: bool = True
    strict_nan_check: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ValidationConfig':
        return cls(
            min_record_duration=d.get('min_record_duration', 7200),
            validate_channels=d.get('validate_channels', True),
            strict_nan_check=d.get('strict_nan_check', True),
        )


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    directory_structure: List[str] = field(
        default_factory=lambda: ['logs', 'reports', 'data', 'splits']
    )
    include_metadata: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OutputConfig':
        return cls(
            base_dir=d.get('base_dir', 'outputs'),
            directory_structure=d.get(
                'directory_structure', ['logs', 'reports', 'data', 'splits']
            ),
            include_metadata=d.get('include_metadata', False),
        )


@dataclass
class SplittingConfig:
    train_ratio: float = 0.7
    validation_ratio: float = 0.1
    test_ratio: float = 0.2
    random_seed: int = 42
    min_samples_per_subject: int = 1
    exclude_subjects: List[str] = field(default_factory=list)
    include_only_subjects: List[str] = field(default_factory=list)
    splitting_method: str = "group_shuffle"
    dry_run: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SplittingConfig':
        return cls(
            train_ratio=d.get('train_ratio', 0.7),
            validation_ratio=d.get('validation_ratio', 0.1),
            test_ratio=d.get('test_ratio', 0.2),
            random_seed=d.get('random_seed', 42),
            min_samples_per_subject=d.get('min_samples_per_subject', 1),
            exclude_subjects=d.get('exclude_subjects', []),
            include_only_subjects=d.get('include_only_subjects', []),
            splitting_method=d.get('splitting_method', 'group_shuffle'),
            dry_run=d.get('dry_run', False),
        )


@dataclass
class PipelineConfig:
    """Central configuration for the dataset processing pipeline.

    Created once from the merged JSON config dict and passed through the
    entire pipeline. Nested dataclasses provide typed access to every
    setting, replacing scattered ``config.get()`` calls with consistent
    defaults.

    ``signal_processing`` is kept as raw dicts because its schema varies
    per channel and step — wrapping each layer in a dataclass would add
    boilerplate without meaningful type safety.
    """

    database_name: str = 'mimic3wdb-matched/1.0'
    input_channels: List[str] = field(default_factory=list)
    output_channels: List[str] = field(default_factory=list)
    record_list_file: str = 'inputs/record_list.txt'
    signal_processing: List[Dict[str, Any]] = field(default_factory=list)
    long_nan_seq_removal: Optional[List[Dict[str, Any]]] = None
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)

    @property
    def required_channels(self) -> List[str]:
        return list(set(self.input_channels + self.output_channels))

    @property
    def channel_names(self) -> List[str]:
        return [ch_config["channel"] for ch_config in self.signal_processing]

    @property
    def has_windowing(self) -> bool:
        return bool(self.signal_processing)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        return cls(
            database_name=config_dict.get('database_name', 'mimic3wdb-matched/1.0'),
            input_channels=config_dict.get('input_channels', []),
            output_channels=config_dict.get('output_channels', []),
            record_list_file=config_dict.get('record_list_file', 'inputs/record_list.txt'),
            signal_processing=config_dict.get('signal_processing', []),
            long_nan_seq_removal=config_dict.get('long_nan_seq_removal', None),
            windowing=WindowingConfig.from_dict(config_dict.get('windowing', {})),
            validation=ValidationConfig.from_dict(config_dict.get('validation', {})),
            output=OutputConfig.from_dict(config_dict.get('output', {})),
            splitting=SplittingConfig.from_dict(config_dict),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
