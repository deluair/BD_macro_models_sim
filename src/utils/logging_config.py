"""Centralized logging configuration for Bangladesh Macroeconomic Models.

This module provides a comprehensive logging setup with:
- Multiple log levels and handlers
- Structured logging with JSON format
- Performance monitoring
- Error tracking and alerting
- Log rotation and archival
"""

import json
import logging
import logging.config
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to log records."""
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = round(process.memory_info().rss / 1024 / 1024, 2)
            record.cpu_percent = process.cpu_percent()
        except ImportError:
            pass
            
        return True


class ModelContextFilter(logging.Filter):
    """Filter to add model-specific context to log records."""
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Add model context to log records."""
        if self.model_name:
            record.model_name = self.model_name
        return True


class LoggingManager:
    """Centralized logging manager for the project."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize logging manager.
        
        Args:
            config_path: Path to logging configuration file
        """
        self.config_path = config_path
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.default_config = self._get_default_config()
        
    def setup_logging(
        self,
        level: str = "INFO",
        model_name: Optional[str] = None,
        enable_json: bool = True,
        enable_performance: bool = True
    ) -> logging.Logger:
        """Setup logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            model_name: Name of the model for context
            enable_json: Whether to use JSON formatting
            enable_performance: Whether to add performance metrics
            
        Returns:
            Configured logger instance
        """
        # Load configuration
        if self.config_path and self.config_path.exists():
            config = self._load_config()
        else:
            config = self.default_config.copy()
            
        # Update log level
        config['loggers']['']['level'] = level
        config['handlers']['console']['level'] = level
        config['handlers']['file']['level'] = level
        
        # Configure formatters
        if enable_json:
            config['handlers']['file']['formatter'] = 'json'
            
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Get root logger
        logger = logging.getLogger()
        
        # Add filters
        if enable_performance:
            logger.addFilter(PerformanceFilter())
            
        if model_name:
            logger.addFilter(ModelContextFilter(model_name))
            
        return logger
        
    def get_model_logger(self, model_name: str) -> logging.Logger:
        """Get a logger specific to a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model-specific logger
        """
        logger = logging.getLogger(f"models.{model_name}")
        logger.addFilter(ModelContextFilter(model_name))
        return logger
        
    def get_analysis_logger(self, analysis_type: str) -> logging.Logger:
        """Get a logger specific to an analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Analysis-specific logger
        """
        return logging.getLogger(f"analysis.{analysis_type}")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'json': {
                    '()': 'src.utils.logging_config.JSONFormatter'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'filename': str(self.log_dir / 'bd_macro_models.log'),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf8'
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'json',
                    'filename': str(self.log_dir / 'errors.log'),
                    'maxBytes': 5242880,  # 5MB
                    'backupCount': 3,
                    'encoding': 'utf8'
                },
                'performance_file': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(self.log_dir / 'performance.log'),
                    'when': 'midnight',
                    'interval': 1,
                    'backupCount': 7,
                    'encoding': 'utf8'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'level': 'INFO',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'models': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'analysis': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file'],
                    'propagate': False
                },
                'performance': {
                    'level': 'INFO',
                    'handlers': ['performance_file'],
                    'propagate': False
                },
                'scripts': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            }
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration from file."""
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)


# Performance monitoring decorator
def log_performance(logger: Optional[logging.Logger] = None):
    """Decorator to log function performance.
    
    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            log = logger or logging.getLogger('performance')
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log.info(
                    f"Function executed successfully",
                    extra={
                        'function_name': func.__name__,
                        'execution_time_seconds': round(execution_time, 4),
                        'args_count': len(args),
                        'kwargs_count': len(kwargs),
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                log.error(
                    f"Function execution failed",
                    extra={
                        'function_name': func.__name__,
                        'execution_time_seconds': round(execution_time, 4),
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'status': 'error'
                    },
                    exc_info=True
                )
                
                raise
                
        return wrapper
    return decorator


# Context manager for logging model operations
class ModelOperationLogger:
    """Context manager for logging model operations."""
    
    def __init__(self, model_name: str, operation: str, logger: Optional[logging.Logger] = None):
        """Initialize model operation logger.
        
        Args:
            model_name: Name of the model
            operation: Name of the operation
            logger: Logger instance to use
        """
        self.model_name = model_name
        self.operation = operation
        self.logger = logger or logging.getLogger(f'models.{model_name}')
        self.start_time = None
        
    def __enter__(self):
        """Start logging operation."""
        import time
        self.start_time = time.time()
        
        self.logger.info(
            f"Starting {self.operation}",
            extra={
                'model_name': self.model_name,
                'operation': self.operation,
                'status': 'started'
            }
        )
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End logging operation."""
        import time
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    'model_name': self.model_name,
                    'operation': self.operation,
                    'execution_time_seconds': round(execution_time, 4),
                    'status': 'completed'
                }
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                extra={
                    'model_name': self.model_name,
                    'operation': self.operation,
                    'execution_time_seconds': round(execution_time, 4),
                    'error_type': exc_type.__name__ if exc_type else None,
                    'error_message': str(exc_val) if exc_val else None,
                    'status': 'failed'
                },
                exc_info=True
            )


# Global logging manager instance
logging_manager = LoggingManager()


# Convenience functions
def setup_logging(
    level: str = "INFO",
    model_name: Optional[str] = None,
    enable_json: bool = True,
    enable_performance: bool = True
) -> logging.Logger:
    """Setup logging with default configuration."""
    return logging_manager.setup_logging(
        level=level,
        model_name=model_name,
        enable_json=enable_json,
        enable_performance=enable_performance
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def get_model_logger(model_name: str) -> logging.Logger:
    """Get a model-specific logger."""
    return logging_manager.get_model_logger(model_name)


def get_analysis_logger(analysis_type: str) -> logging.Logger:
    """Get an analysis-specific logger."""
    return logging_manager.get_analysis_logger(analysis_type)