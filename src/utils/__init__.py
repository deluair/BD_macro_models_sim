"""Utilities package for Bangladesh Macroeconomic Models.

This package provides common utilities for:
- Error handling and validation
- Performance monitoring
- Logging configuration
- Data processing helpers
"""

from .error_handling import (
    BaseModelError,
    DataError,
    ModelConfigurationError,
    ModelConvergenceError,
    ValidationError,
    NetworkError,
    SystemResourceError,
    UserInputError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    handle_errors,
    Validator,
    safe_divide,
    safe_log
)

from .performance_monitor import (
    PerformanceMetrics,
    ResourceThresholds,
    PerformanceMonitor,
    monitor_performance,
    performance_context
)

from .logging_config import (
    get_logger,
    setup_logging,
    get_model_logger,
    get_analysis_logger,
    log_performance,
    ModelOperationLogger
)

__all__ = [
    # Error handling
    'BaseModelError',
    'DataError',
    'ModelConfigurationError',
    'ModelConvergenceError',
    'ValidationError',
    'NetworkError',
    'SystemResourceError',
    'UserInputError',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'ErrorHandler',
    'handle_errors',
    'Validator',
    'safe_divide',
    'safe_log',
    
    # Performance monitoring
    'PerformanceMetrics',
    'ResourceThresholds',
    'PerformanceMonitor',
    'monitor_performance',
    'performance_context',
    
    # Logging
    'get_logger',
    'setup_logging',
    'get_model_logger',
    'get_analysis_logger',
    'log_performance',
    'ModelOperationLogger',
]