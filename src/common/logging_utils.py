#!/usr/bin/env python3
"""
Logging Utilities for MIMIC III Matched Waveform Dataset
"""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import os
from contextlib import contextmanager


class OutputManager:
    """
    Manages creation and organization of run-specific output directories.
    """
    
    def __init__(self, base_config: Dict[str, Any], task_name: str):
        """
        Initialize the run directory manager.
        
        Args:
            base_config: Base configuration dictionary
            task_name: Name of the current task/experiment
        """
        self.base_config = base_config
        self.task_name = task_name
        self.timestamp = datetime.now()
        self.run_id = self._generate_run_id()
        self.run_dir = self._create_run_directory()
        
    def _generate_run_id(self) -> str:
        """
        Generate a unique run identifier based on task name and timestamp.
        
        Returns:
            Formatted run identifier string
        """
        timestamp_format = self.base_config.get("output", {}).get(
            "run_naming", {}
        ).get("timestamp_format", "%Y%m%d_%H%M%S")
        
        timestamp_str = self.timestamp.strftime(timestamp_format)
        run_format = self.base_config.get("output", {}).get(
            "run_naming", {}
        ).get("format", "{task_name}_{timestamp}")
        
        return run_format.format(
            task_name=self.task_name,
            timestamp=timestamp_str
        )
    
    def _create_run_directory(self) -> Path:
        """
        Create the run-specific directory structure.
        
        Returns:
            Path to the created run directory
        """
        base_dir = self.base_config.get("output", {}).get("base_dir", "outputs")
        run_dir = Path(base_dir)
        
        # Create main run directory
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories from configuration
        subdirs = self.base_config.get("output", {}).get("directory_structure", [])
        for subdir in subdirs:
            (run_dir / subdir).mkdir(exist_ok=True)
            
        return run_dir
    
    def get_run_directory(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
    
    def get_logs_directory(self) -> Path:
        """Get the logs subdirectory path."""
        return self.run_dir / "logs"
    
    def get_analysis_directory(self) -> Path:
        """Get the analysis subdirectory path."""
        return self.run_dir / "analysis"
    
    def get_reports_directory(self) -> Path:
        """Get the reports subdirectory path."""
        return self.run_dir / "reports"
    
    def save_run_config(self, config: Dict[str, Any]) -> None:
        """
        Save the configuration used for this run.
        
        Args:
            config: Configuration dictionary to save
        """
        config_path = self.run_dir / "used_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class Logger:
    """
    Professional logging system for experimental workflows.
    """
    
    def __init__(self, 
                 name: str, 
                 run_manager: OutputManager,
                 config: Dict[str, Any]):
        """
        Initialize the research logger.
        
        Args:
            name: Logger name (typically module or task name)
            run_manager: Run directory manager instance
            config: Logging configuration
        """
        self.name = name
        self.run_manager = run_manager
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger with appropriate handlers and formatters.
        
        Returns:
            Configured logger instance
        """
        # Create a copy of the logging config to modify
        logging_config = self.config.get("logging", {}).copy()
        
        # Update file paths to use run-specific directories
        logs_dir = self.run_manager.get_logs_directory()
        
        if "handlers" in logging_config:
            for handler_name, handler_config in logging_config["handlers"].items():
                if handler_config.get("class") == "logging.FileHandler":
                    if "filename" in handler_config:
                        filename = Path(handler_config["filename"]).name
                        handler_config["filename"] = str(logs_dir / filename)
        
        # Configure logging
        logging.config.dictConfig(logging_config)
        
        # Get the logger
        logger = logging.getLogger(self.name)
        
        return logger
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self.logger.critical(message, **kwargs)
    
    @contextmanager
    def log_execution_time(self, operation_name: str):
        """
        Context manager to log execution time of operations.
        
        Args:
            operation_name: Name of the operation being timed
        """
        start_time = datetime.now()
        self.info(f"Starting {operation_name}...")
        
        try:
            yield
            execution_time = (datetime.now() - start_time).total_seconds()
            self.info(f"Completed {operation_name} in {execution_time:.2f} seconds")
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.error(f"Failed {operation_name} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    def log_experiment_start(self, 
                           task_name: str, 
                           config_path: str,
                           parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the start of an experimental run with key parameters.
        
        Args:
            task_name: Name of the experimental task
            config_path: Path to the configuration file
            parameters: Additional parameters to log
        """
        self.info(f"MIMIC III Research Project - {task_name.upper()}")
        self.info(f"Run ID: {self.run_manager.run_id}")
        self.info(f"Configuration: {config_path}")
        
        if parameters:
            for key, value in parameters.items():
                self.info(f"  {key}: {value}")
    
    def log_script_start(self, 
                        task_name: str, 
                        config_path: str,
                        config: Dict[str, Any]) -> None:
        """
        Log the start of a script run.
        
        Args:
            task_name: Name of the task
            config_path: Path to the configuration file
            config: Configuration dictionary
        """
        self.info(f"Starting {task_name}")
        self.debug(f"Run ID: {self.run_manager.run_id}")
        self.debug(f"Start time: {self.run_manager.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        self.debug(f"Output directory: {self.run_manager.get_run_directory()}")
        self.debug(f"Configuration: {config_path}")
    
    def log_experiment_end(self, success: bool = True, summary: Optional[str] = None) -> None:
        """
        Log the end of an experimental run.
        
        Args:
            success: Whether the experiment completed successfully
            summary: Optional summary of results
        """
        end_time = datetime.now()
        duration = (end_time - self.run_manager.timestamp).total_seconds()
        
        if success:
            self.info(f"Experiment completed successfully")
        else:
            self.error(f"Experiment failed")
            
        self.debug(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.debug(f"Total duration: {duration:.2f} seconds")
        
        if summary:
            self.info(f"Summary: {summary}")
    
    def log_script_end(self, success: bool = True) -> None:
        """
        Log the end of a script run.
        
        Args:
            success: Whether the script completed successfully
        """
        end_time = datetime.now()
        duration = (end_time - self.run_manager.timestamp).total_seconds()
        
        if success:
            self.debug(f"Script completed successfully in {duration:.2f} seconds")
        else:
            self.error(f"Script failed after {duration:.2f} seconds")


def create_logger(task_name: str, 
                         config: Dict[str, Any],
                         logger_name: Optional[str] = None) -> tuple[Logger, OutputManager]:
    """
    Factory function to create a logger and output manager.
    
    Args:
        task_name: Name of the task/experiment
        config: Configuration dictionary
        logger_name: Optional custom logger name
        
    Returns:
        Tuple of (Logger, OutputManager)
    """
    # Create a modified config for the run directory manager
    modified_config = config.copy()
    
    # Handle task-specific output directory if specified
    task_output_config = config.get('output', {})
    if 'base_dir' in task_output_config:
        # Use the task-specific base_dir directly
        modified_config['output'] = modified_config.get('output', {}).copy()
        # Don't duplicate the task name in the path
        pass
    else:
        # Use the base output config and append task name
        base_output = modified_config.get('output', {})
        base_dir = base_output.get('base_dir', 'outputs')
        modified_config['output'] = modified_config.get('output', {}).copy()
        modified_config['output']['base_dir'] = f"{base_dir}/{task_name}"
    
    run_manager = OutputManager(modified_config, task_name)
    
    if logger_name is None:
        logger_name = f"mimic_iii.{task_name}"
    
    logger = Logger(logger_name, run_manager, modified_config)
    
    return logger, run_manager


def load_configuration(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file with error handling.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        json.JSONDecodeError: If configuration file is invalid JSON
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file {config_path}: {str(e)}")


def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries with deep merging.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    if not configs:
        return {}
    
    merged = configs[0].copy()
    for config in configs[1:]:
        merged = deep_merge(merged, config)
    
    return merged

