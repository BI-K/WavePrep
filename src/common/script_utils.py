#!/usr/bin/env python3
"""
Script Utilities for MIMIC III Research Project

Shared utilities for all scripts including argument parsing,
configuration loading, and error handling.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

from .config import create_config_manager, ConfigError
from .logging_utils import create_logger


def create_standard_parser(script_description: str) -> argparse.ArgumentParser:
    """
    Create a standard argument parser for research scripts.
    
    Args:
        script_description: Description of the script
        
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=script_description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to task configuration file'
    )
    
    return parser


def load_script_configuration(config_path: str, 
                            base_config_path: str = "configs/base_config.json") -> Dict[str, Any]:
    """
    Load and merge configurations for a script.
    
    Args:
        config_path: Path to task-specific configuration
        base_config_path: Path to base configuration
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    try:
        config_manager = create_config_manager(base_config_path)
        return config_manager.load_configuration(config_path)
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {str(e)}") from e


def setup_script_environment(task_name: str, 
                           config: Dict[str, Any],
                           config_path: str) -> tuple:
    """
    Set up the script environment including logging and directories.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (logger, run_manager)
    """
    # Create research logger and run manager
    logger, run_manager = create_logger(task_name, config)
    
    # Save the configuration used for this run
    run_manager.save_run_config(config)
    
    # Log script start
    logger.log_script_start(task_name, config_path, config)
    
    return logger, run_manager


def validate_required_config(config: Dict[str, Any], 
                           required_keys: list,
                           logger) -> None:
    """
    Validate that required configuration keys are present.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        logger: Logger instance
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    for key in required_keys:
        if '.' in key:
            # Handle nested keys like 'signals.heart_rate'
            parts = key.split('.')
            current = config
            try:
                for part in parts:
                    current = current[part]
            except (KeyError, TypeError):
                missing_keys.append(key)
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        error_msg = f"Missing required configuration keys: {missing_keys}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def run_script_with_error_handling(main_func, 
                                 task_name: str,
                                 script_description: str) -> int:
    """
    Run a script with standardized error handling and logging.
    
    Args:
        main_func: Main function to execute
        task_name: Name of the task
        script_description: Description for argument parser
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = None
    
    try:
        # Parse arguments
        parser = create_standard_parser(script_description)
        args = parser.parse_args()
        
        # Load configuration
        config = load_script_configuration(args.config)
        
        # Set up environment
        logger, run_manager = setup_script_environment(task_name, config, args.config)
        
        # Run main function
        result = main_func(config, logger, run_manager)
        
        # Log successful completion
        logger.log_script_end(success=True)
        logger.info(f"Script completed successfully")
        logger.info(f"Results saved to: {run_manager.get_run_directory()}")
        
        return result if result is not None else 0
        
    except ConfigError as e:
        error_msg = f"Configuration error: {str(e)}"
        if logger:
            logger.error(error_msg)
            logger.log_script_end(success=False)
        else:
            print(f"[   ERROR] {error_msg}", file=sys.stderr)
        return 1
        
    except Exception as e:
        error_msg = f"Script failed: {str(e)}"
        if logger:
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            logger.log_script_end(success=False)
        else:
            print(f"[   ERROR] {error_msg}", file=sys.stderr)
        return 1
