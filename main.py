#!/usr/bin/env python3
"""
Bangladesh Macroeconomic Models Simulation Framework

A comprehensive framework for simulating various macroeconomic models
specifically calibrated for Bangladesh's economy.

Author: Bangladesh Economic Research Team
Version: 1.0.0
License: MIT
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import configuration
from src.config.config_manager import ConfigManager
from src.utils.logging_config import setup_logging

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Bangladesh Macroeconomic Models Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full economic analysis
  python main.py --config custom.yaml  # Use custom configuration
  python main.py --verbose          # Enable verbose logging
        """
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Bangladesh Macroeconomic Models Framework v1.0.0"
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point for the Bangladesh Macroeconomic Models Framework.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel("DEBUG")
    
    logger.info("Starting Bangladesh Macroeconomic Models Framework")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        logger.info(f"Loaded configuration for {config.get('general', {}).get('country', 'Bangladesh')}")
        logger.info(f"Base year: {config.get('general', {}).get('base_year', 2020)}")
        logger.info(f"Simulation periods: {config.get('general', {}).get('simulation_periods', 40)}")
        
        # Import and run the main application
        from src.bangladesh_economic_analysis import main as run_economic_analysis
        
        # Run the economic analysis
        logger.info("Starting economic analysis...")
        run_economic_analysis()
        
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise
    
    logger.info("Bangladesh Macroeconomic Models Framework completed")

if __name__ == "__main__":
    main()