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

from .script_utils import (
    create_standard_parser,
    load_script_configuration,
    setup_script_environment,
    validate_required_config,
    run_script_with_error_handling
)

__all__ = [
    # Logging utilities
    'Logger',
    'OutputManager', 
    'create_logger',
    'load_configuration',
    'merge_configurations',
    
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
    'run_script_with_error_handling'
]