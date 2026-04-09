#!/usr/bin/env python3
"""
Common Utilities for MIMIC III Matched Waveform Dataset
"""

from .logging_utils import (
    Logger,
    OutputManager,
    create_logger,
    load_configuration,
    merge_configurations
)

from .config import (
    ConfigManager,
    ConfigError,
    ConfigPaths,
    create_config_manager,
    validate_config_files,
    get_default_config_paths
)

from .signal_data import (
    SignalData,
    WindowedData,
    Window
)

from .processing_context import ProcessingContext

from .pipeline_config import (
    PipelineConfig,
    WindowingConfig,
    ValidationConfig,
    OutputConfig,
    SplittingConfig,
    PublicationConfig,
)

from .script_utils import (
    create_standard_parser,
    load_script_configuration,
    setup_script_environment,
    validate_required_config,
    run_script_with_error_handling
)

from .signal_io import (
    get_signal_writer,
    get_file_extension,
    read_signal_data,
    FORMAT_EXTENSIONS,
)

__all__ = [
    # Logging utilities
    'Logger',
    'OutputManager', 
    'create_logger',
    'load_configuration',
    'merge_configurations',
    
    # Signal data classes
    'SignalData',
    'WindowedData',
    'Window',

    # Processing context
    'ProcessingContext',

    # Pipeline configuration
    'PipelineConfig',
    'WindowingConfig',
    'ValidationConfig',
    'OutputConfig',
    'SplittingConfig',
    'PublicationConfig',

    # Configuration utilities
    'ConfigManager',
    'ConfigError',
    'ConfigPaths',
    'create_config_manager',
    'validate_config_files',
    'get_default_config_paths',
    
    # Script utilities
    'create_standard_parser',
    'load_script_configuration',
    'setup_script_environment',
    'validate_required_config',
    'run_script_with_error_handling',

    # Signal I/O
    'get_signal_writer',
    'get_file_extension',
    'read_signal_data',
    'FORMAT_EXTENSIONS',
]