#!/usr/bin/env python3
"""
Configuration Utilities for MIMIC III Matched Waveform Dataset
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging


@dataclass
class ConfigPaths:
    """Container for configuration file paths."""
    base_config: Path
    task_config: Path
    resolved_config: Optional[Path] = None


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class ConfigManager:
    """
    Manages hierarchical configuration loading and validation.
    
    Supports loading base configuration and task-specific configurations,
    merging them appropriately, and providing type-safe access to values.
    """
    
    def __init__(self, base_config_path: Union[str, Path]):
        """
        Initialize the configuration manager.
        
        Args:
            base_config_path: Path to the base configuration file
        """
        self.base_config_path = Path(base_config_path)
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self._validate_base_config()
    
    def _validate_base_config(self) -> None:
        """Validate that the base configuration exists and is readable."""
        if not self.base_config_path.exists():
            raise ConfigError(f"Base configuration not found: {self.base_config_path}")
        
        if not self.base_config_path.is_file():
            raise ConfigError(f"Base configuration is not a file: {self.base_config_path}")
    
    def load_configuration(self, 
                         task_config_path: Union[str, Path],
                         validate: bool = True) -> Dict[str, Any]:
        """
        Load and merge base and task-specific configurations.
        
        Args:
            task_config_path: Path to the task-specific configuration
            validate: Whether to validate the configuration
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        task_config_path = Path(task_config_path)
        cache_key = f"{self.base_config_path}:{task_config_path}"
        
        # Check cache first
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        try:
            # Load base configuration
            base_config = self._load_json_file(self.base_config_path)
            
            # Load task-specific configuration
            task_config = self._load_json_file(task_config_path)
            
            # Merge configurations
            merged_config = self._merge_configurations(base_config, task_config)
            
            # Substitute environment variables
            merged_config = self._substitute_environment_variables(merged_config)
            
            # Validate if requested
            if validate:
                self._validate_configuration(merged_config)
            
            # Cache the result
            self.config_cache[cache_key] = merged_config
            
            return merged_config
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}") from e
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a JSON configuration file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {file_path}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Error reading {file_path}: {str(e)}")
    
    def _merge_configurations(self, 
                            base_config: Dict[str, Any], 
                            task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base and task configurations with deep merging.
        
        Args:
            base_config: Base configuration dictionary
            task_config: Task-specific configuration dictionary
            
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
        
        return deep_merge(base_config, task_config)
    
    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variables substituted
        """
        def substitute_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {key: substitute_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                return os.path.expandvars(obj)
            else:
                return obj
        
        return substitute_recursive(config)
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        required_sections = ['output', 'logging']
        
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Missing required configuration section: {section}")
        
        # Validate output section
        output_config = config.get('output', {})
        if 'base_dir' not in output_config:
            raise ConfigError("Missing 'base_dir' in output configuration")
        
        # Validate logging section
        logging_config = config.get('logging', {})
        if 'handlers' not in logging_config:
            raise ConfigError("Missing 'handlers' in logging configuration")
    
    def get_config_value(self, 
                        config: Dict[str, Any], 
                        key_path: str, 
                        default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the value (e.g., 'output.base_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def create_config_paths(self, 
                          task_config_path: Union[str, Path]) -> ConfigPaths:
        """
        Create a ConfigPaths object for the given task configuration.
        
        Args:
            task_config_path: Path to the task configuration file
            
        Returns:
            ConfigPaths object
        """
        return ConfigPaths(
            base_config=self.base_config_path,
            task_config=Path(task_config_path)
        )


def create_config_manager(base_config_path: Union[str, Path] = "configs/base_config.json") -> ConfigManager:
    """
    Factory function to create a configuration manager.
    
    Args:
        base_config_path: Path to the base configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(base_config_path)


def validate_config_files(*config_paths: Union[str, Path]) -> List[Path]:
    """
    Validate that all configuration files exist and are readable.
    
    Args:
        *config_paths: Paths to configuration files
        
    Returns:
        List of validated Path objects
        
    Raises:
        ConfigError: If any configuration file is invalid
    """
    validated_paths = []
    
    for config_path in config_paths:
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        
        if not path.is_file():
            raise ConfigError(f"Configuration path is not a file: {path}")
        
        # Try to load and validate JSON
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {path}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Error reading {path}: {str(e)}")
        
        validated_paths.append(path)
    
    return validated_paths


def get_default_config_paths() -> Dict[str, Path]:
    """
    Get default configuration paths for the project.
    
    Returns:
        Dictionary mapping configuration names to their paths
    """
    base_dir = Path("configs")
    
    return {
        "base": base_dir / "base_config.json",
        "explore": base_dir / "explore" / "default.json",
        "create": base_dir / "create" / "default.json",
        "split": base_dir / "split" / "default.json"
    }

