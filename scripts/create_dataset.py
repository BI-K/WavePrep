#!/usr/bin/env python3
"""
MIMIC III Matched Waveform Dataset Creation Script
"""

import sys
from pathlib import Path
from typing import Dict, Any
import warnings
import urllib3
warnings.filterwarnings("ignore")
import logging

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from create.procress_dataset import run_dataset_creation

from common import (
    create_standard_parser,
    load_script_configuration,
    setup_script_environment,
    validate_required_config,
    ConfigError
)

def configure_root_logger(level=logging.INFO, log_file=None):
    """Configure root logger with proper settings."""
    # Suppress unnecessary logging
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('wfdb').setLevel(logging.ERROR)
    urllib3.disable_warnings()

    for logger_name in ['requests', 'urllib', 'urllib3', 'requests.packages.urllib3']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger(logger_name).propagate = False
        
    if not logging.getLogger().handlers:
        handlers = [logging.StreamHandler()]
        
        # Add file handler if log file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=handlers
        )

def main() -> int:
    """Main entry point for the dataset creation script."""
    logger = None
    
    try:
        # Parse arguments
        parser = create_standard_parser("Create structured dataset from MIMIC III waveform data")
        args = parser.parse_args()
        
        # Configure logging first (without file handler initially)
        configure_root_logger(logging.INFO)
        
        # Load configuration
        config = load_script_configuration(args.config)
        
        # Set up environment
        logger, output_manager = setup_script_environment("create", config, args.config)
        
        # Get log file path from output manager and reconfigure logging with file handler
        log_file = output_manager.get_logs_directory() / "log.txt"
        configure_root_logger(logging.INFO, log_file)
        
        # Log script start
        logger.info("Starting dataset creation")
        logger.debug(f"Configuration loaded from: {args.config}")
        
        # Validate required configuration
        required_keys = [
            "database_name",
            "input_channels",
            "output_channels",
            "signal_processing",
            "windowing"
        ]
        validate_required_config(config, required_keys, logger)
        
        # Run dataset creation with log file path
        run_dataset_creation(config, output_manager, logger, log_file)
        
        return 0
        
    except ConfigError as e:
        if logger:
            logger.error(f"Configuration error: {str(e)}")
        else:
            print(f"Configuration error: {str(e)}", file=sys.stderr)
        return 1
        
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {str(e)}")
        else:
            print(f"Unexpected error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
