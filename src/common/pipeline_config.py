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


SUPPORTED_SAVE_FORMATS = ('csv', 'edf', 'matlab', 'wav', 'wfdb')


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    directory_structure: List[str] = field(
        default_factory=lambda: ['logs', 'reports', 'data', 'splits']
    )
    include_metadata: bool = False
    save_format: str = 'csv'
    generate_croissant: bool = False
    hf_repo_id: Optional[str] = None

    @property
    def is_croissant_export(self) -> bool:
        return self.generate_croissant

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OutputConfig':
        save_format = d.get('save_format', d.get('output_format', 'csv')).lower()
        if save_format not in SUPPORTED_SAVE_FORMATS:
            raise ValueError(
                f"Unsupported save_format '{save_format}'. "
                f"Supported formats: {SUPPORTED_SAVE_FORMATS}"
            )
        return cls(
            base_dir=d.get('base_dir', 'outputs'),
            directory_structure=d.get(
                'directory_structure', ['logs', 'reports', 'data', 'splits']
            ),
            include_metadata=d.get('include_metadata', False),
            save_format=save_format,
            generate_croissant=d.get('generate_croissant', False),
            hf_repo_id=d.get('hf_repo_id'),
        )


@dataclass
class PublicationConfig:
    """Metadata fields that flow into the Croissant JSON-LD and HuggingFace card.

    Defaults are placeholders — override in config before publishing.
    """
    dataset_name: str = 'PLACEHOLDER_DATASET_NAME'
    dataset_url: str = 'https://example.com/placeholder-dataset-url'
    creator: str = 'PLACEHOLDER_CREATOR'
    date_published: str = '1970-01-01'
    license: str = 'https://opendatacommons.org/licenses/odbl/1-0/'
    cite_as: Optional[str] = None
    version: str = '1.0.0'

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PublicationConfig':
        if not d:
            return cls()
        return cls(
            dataset_name=d.get('dataset_name', 'PLACEHOLDER_DATASET_NAME'),
            dataset_url=d.get('dataset_url', 'https://example.com/placeholder-dataset-url'),
            creator=d.get('creator', 'PLACEHOLDER_CREATOR'),
            date_published=d.get('date_published', '1970-01-01'),
            license=d.get('license', 'https://opendatacommons.org/licenses/odbl/1-0/'),
            cite_as=d.get('cite_as'),
            version=d.get('version', '1.0.0'),
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
class DownsamplingStepConfig:
    desired_resolution: float = 1.0
    strategy: str = 'decimate'

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional['DownsamplingStepConfig']:
        if not d:
            return None
        return cls(
            desired_resolution=d.get('desired_resolution', 1.0),
            strategy=d.get('downsampling_strategy', 'decimate'),
        )


@dataclass
class DataCleaningStepConfig:
    lower_threshold: Optional[float] = None
    upper_threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional['DataCleaningStepConfig']:
        if not d:
            return None
        return cls(
            lower_threshold=d.get('lower_threshold'),
            upper_threshold=d.get('upper_threshold'),
        )


@dataclass
class ImputationStepConfig:
    method: str = 'mean'

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional['ImputationStepConfig']:
        if not d:
            return None
        return cls(method=d.get('method', 'mean'))


@dataclass
class ProcessingStepConfig:
    step: int = 0
    downsampling: Optional[DownsamplingStepConfig] = None
    data_cleaning: Optional[DataCleaningStepConfig] = None
    imputation: Optional[ImputationStepConfig] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProcessingStepConfig':
        return cls(
            step=d.get('step', 0),
            downsampling=DownsamplingStepConfig.from_dict(d.get('downsampling', {})),
            data_cleaning=DataCleaningStepConfig.from_dict(d.get('data_cleaning', {})),
            imputation=ImputationStepConfig.from_dict(d.get('imputation', {})),
        )


@dataclass
class ChannelProcessingConfig:
    channel: str = ''
    steps: List[ProcessingStepConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ChannelProcessingConfig':
        return cls(
            channel=d.get('channel', ''),
            steps=[ProcessingStepConfig.from_dict(s) for s in d.get('steps', [])],
        )


@dataclass
class PipelineConfig:
    """Central configuration for the dataset processing pipeline.

    Created once from the merged JSON config dict and passed through the
    entire pipeline.  Nested dataclasses provide typed access to every
    setting, replacing scattered ``config.get()`` calls with consistent
    defaults.
    """

    database_name: str = 'mimic3wdb-matched/1.0'
    input_channels: List[str] = field(default_factory=list)
    output_channels: List[str] = field(default_factory=list)
    record_list_file: str = 'inputs/record_list.txt'
    signal_processing: List[ChannelProcessingConfig] = field(default_factory=list)
    long_nan_seq_removal: Optional[List[Dict[str, Any]]] = None
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    publication: PublicationConfig = field(default_factory=PublicationConfig)

    @property
    def required_channels(self) -> List[str]:
        return list(set(self.input_channels + self.output_channels))

    @property
    def channel_names(self) -> List[str]:
        return [ch.channel for ch in self.signal_processing]

    @property
    def has_windowing(self) -> bool:
        return bool(self.signal_processing)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        # Allow save_format/output_format at top-level or inside output section
        output_dict = dict(config_dict.get('output', {}))
        if 'save_format' in config_dict:
            output_dict.setdefault('save_format', config_dict['save_format'])
        if 'output_format' in config_dict:
            output_dict.setdefault('save_format', config_dict['output_format'])
        if 'output_format' in output_dict and 'save_format' not in output_dict:
            output_dict['save_format'] = output_dict['output_format']

        return cls(
            database_name=config_dict.get('database_name', 'mimic3wdb-matched/1.0'),
            input_channels=config_dict.get('input_channels', []),
            output_channels=config_dict.get('output_channels', []),
            record_list_file=config_dict.get('record_list_file', 'inputs/record_list.txt'),
            signal_processing=[
                ChannelProcessingConfig.from_dict(d)
                for d in config_dict.get('signal_processing', [])
            ],
            long_nan_seq_removal=config_dict.get('long_nan_seq_removal', None),
            windowing=WindowingConfig.from_dict(config_dict.get('windowing', {})),
            validation=ValidationConfig.from_dict(config_dict.get('validation', {})),
            output=OutputConfig.from_dict(output_dict),
            splitting=SplittingConfig.from_dict(config_dict),
            publication=PublicationConfig.from_dict(config_dict.get('publication', {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
