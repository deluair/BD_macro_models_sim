#!/usr/bin/env python3
"""
Configuration Management System

This module provides centralized configuration management for the Bangladesh
Macroeconomic Models project with support for environment-specific overrides,
validation, and type safety.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from copy import deepcopy

try:
    from pydantic import BaseModel, validator, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


@dataclass
class ProjectConfig:
    """Project metadata configuration."""
    name: str = "Bangladesh Macroeconomic Models"
    version: str = "1.0.0"
    description: str = "Comprehensive macroeconomic modeling suite for Bangladesh"
    author: str = "Bangladesh Macro Models Team"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/application.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    console_output: bool = True


@dataclass
class DataConfig:
    """Data management configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    external_data_dir: str = "data/external"
    validation: Dict[str, Any] = field(default_factory=lambda: {
        "enable_checks": True,
        "fail_on_error": False,
        "max_missing_percentage": 10
    })
    # Allow additional fields for data sources
    def __post_init__(self):
        pass


@dataclass
class ModelConfig:
    """Model configuration settings."""
    global_settings: Dict[str, Any] = field(default_factory=lambda: {
        "random_seed": 42,
        "simulation_periods": 100,
        "monte_carlo_runs": 1000,
        "convergence_tolerance": 1e-6,
        "max_iterations": 1000
    })
    dsge: Dict[str, Any] = field(default_factory=dict)
    cge: Dict[str, Any] = field(default_factory=dict)
    abm: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    caching: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "backend": "disk",
        "cache_dir": "cache",
        "ttl": 3600
    })
    parallel: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_workers": 4,
        "chunk_size": 100
    })


class ConfigManager:
    """
    Centralized configuration manager for the Bangladesh Macroeconomic Models project.
    
    This class handles loading, merging, and validating configuration from multiple sources:
    - Default configuration file
    - Environment-specific configuration files
    - Environment variables
    - Runtime overrides
    """
    
    def __init__(self, 
                 config_dir: Optional[Union[str, Path]] = None,
                 environment: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, test, prod)
        """
        self.config_dir = Path(config_dir) if config_dir else self._get_default_config_dir()
        self.environment = environment or os.getenv('ENVIRONMENT', 'default')
        self._config: Dict[str, Any] = {}
        self._loaded = False
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config_dir(self) -> Path:
        """Get the default configuration directory."""
        # Try to find config directory relative to project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/config to project root
        config_dir = project_root / "config"
        
        if not config_dir.exists():
            # Fallback to current directory
            config_dir = Path.cwd() / "config"
        
        return config_dir
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from all sources.
        
        Args:
            force_reload: Force reloading even if already loaded
            
        Returns:
            Complete configuration dictionary
        """
        if self._loaded and not force_reload:
            return self._config
        
        try:
            # Start with default configuration
            self._config = self._load_default_config()
            
            # Override with environment-specific configuration
            env_config = self._load_environment_config()
            if env_config:
                self._config = self._merge_configs(self._config, env_config)
            
            # Override with environment variables
            env_vars = self._load_environment_variables()
            if env_vars:
                self._config = self._merge_configs(self._config, env_vars)
            
            # Validate configuration
            self._validate_config()
            
            # Create typed configuration objects
            self._create_typed_configs()
            
            self._loaded = True
            logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
        
        return self._config
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load the default configuration file."""
        default_file = self.config_dir / "default.yaml"
        
        if not default_file.exists():
            logger.warning(f"Default config file not found: {default_file}")
            return {}
        
        return self._load_yaml_file(default_file)
    
    def _load_environment_config(self) -> Optional[Dict[str, Any]]:
        """Load environment-specific configuration."""
        if self.environment == 'default':
            return None
        
        env_file = self.config_dir / f"{self.environment}.yaml"
        
        if not env_file.exists():
            logger.info(f"Environment config file not found: {env_file}")
            return None
        
        return self._load_yaml_file(env_file)
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded config from: {file_path}")
                return config or {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            raise ConfigurationError(f"Failed to load {file_path}: {e}")
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'LOG_LEVEL': ['logging', 'level'],
            'DEBUG': ['development', 'debug'],
            'CACHE_ENABLED': ['performance', 'caching', 'enabled'],
            'MAX_WORKERS': ['performance', 'parallel', 'max_workers'],
            'WORLD_BANK_API_KEY': ['security', 'api_keys', 'world_bank'],
            'QUANDL_API_KEY': ['security', 'api_keys', 'quandl'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                value = self._convert_env_value(value)
                self._set_nested_value(env_config, config_path, value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """Set a nested configuration value."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        # Basic validation
        required_sections = ['project', 'logging', 'data', 'models']
        
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing required configuration section: {section}")
        
        # Validate logging level
        if 'logging' in self._config:
            level = self._config['logging'].get('level', 'INFO')
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                raise ConfigurationError(f"Invalid logging level: {level}")
        
        # Validate data directories
        if 'data' in self._config:
            for dir_key in ['raw_data_dir', 'processed_data_dir', 'external_data_dir']:
                if dir_key in self._config['data']:
                    dir_path = Path(self._config['data'][dir_key])
                    dir_path.mkdir(parents=True, exist_ok=True)
    
    def _create_typed_configs(self):
        """Create typed configuration objects."""
        # Create typed config objects for better IDE support and validation
        if 'project' in self._config:
            self.project = ProjectConfig(**self._config['project'])
        
        if 'logging' in self._config:
            self.logging = LoggingConfig(**self._config['logging'])
        
        if 'data' in self._config:
            # Filter data config to only include known DataConfig fields
            data_config_fields = {'raw_data_dir', 'processed_data_dir', 'external_data_dir', 'validation'}
            filtered_data_config = {k: v for k, v in self._config['data'].items() if k in data_config_fields}
            self.data = DataConfig(**filtered_data_config)
            # Store data sources separately
            self.data_sources = self._config.get('data_sources', {})
        
        if 'models' in self._config:
            self.models = ModelConfig(
                global_settings=self._config['models'].get('global', {}),
                dsge=self._config['models'].get('dsge', {}),
                cge=self._config['models'].get('cge', {}),
                abm=self._config['models'].get('abm', {})
            )
        
        if 'performance' in self._config:
            # Filter performance config to only include known PerformanceConfig fields
            performance_config_fields = {'caching', 'parallel'}
            filtered_performance_config = {k: v for k, v in self._config['performance'].items() if k in performance_config_fields}
            self.performance = PerformanceConfig(**filtered_performance_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        if not self._loaded:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation."""
        if not self._loaded:
            self.load_config()
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None):
        """Save current configuration to a file."""
        if not file_path:
            file_path = self.config_dir / f"{self.environment}_generated.yaml"
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {file_path}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if not self._loaded:
            self.load_config()
        
        model_config = self._config.get('models', {}).get(model_name, {})
        global_config = self._config.get('models', {}).get('global', {})
        
        # Merge global settings with model-specific settings
        return self._merge_configs(global_config, model_config)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.set(key, value)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_config()
    return _config_manager


def init_config(config_dir: Optional[Union[str, Path]] = None,
                environment: Optional[str] = None) -> ConfigManager:
    """Initialize the global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_dir, environment)
    _config_manager.load_config()
    return _config_manager