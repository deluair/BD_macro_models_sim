"""Comprehensive error handling utilities for Bangladesh Macroeconomic Models.

This module provides:
- Custom exception classes for different error types
- Error recovery mechanisms
- Validation utilities
- Error reporting and logging
- Graceful degradation strategies
"""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA = "data"
    MODEL = "model"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SYSTEM = "system"
    USER_INPUT = "user_input"


@dataclass
class ErrorContext:
    """Context information for errors."""
    model_name: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    data_shape: Optional[tuple] = None
    timestamp: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class BaseModelError(Exception):
    """Base exception class for all model-related errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.MODEL,
        context: Optional[ErrorContext] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = True
    ):
        """Initialize base model error.
        
        Args:
            message: Error message
            severity: Error severity level
            category: Error category
            context: Additional context information
            suggestions: List of suggested solutions
            recoverable: Whether the error is recoverable
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.__dict__,
            'suggestions': self.suggestions,
            'recoverable': self.recoverable,
            'traceback': traceback.format_exc()
        }


class DataError(BaseModelError):
    """Errors related to data issues."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)


class ModelConfigurationError(BaseModelError):
    """Errors related to model configuration."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class ModelConvergenceError(BaseModelError):
    """Errors related to model convergence issues."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ValidationError(BaseModelError):
    """Errors related to validation failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class NetworkError(BaseModelError):
    """Errors related to network operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class SystemResourceError(BaseModelError):
    """Errors related to system resources."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class UserInputError(BaseModelError):
    """Errors related to user input."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.USER_INPUT,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler.
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        strategy: Callable
    ) -> None:
        """Register a recovery strategy for a specific error type.
        
        Args:
            error_type: Exception type
            strategy: Recovery function
        """
        self.recovery_strategies[error_type] = strategy
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle an error with logging and optional recovery.
        
        Args:
            error: Exception to handle
            context: Additional context information
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error
        if isinstance(error, BaseModelError):
            self._log_model_error(error)
        else:
            self._log_generic_error(error, context)
            
        # Attempt recovery if enabled and strategy exists
        if attempt_recovery and type(error) in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[type(error)](error)
                self.logger.info(f"Successfully recovered from {error_type}")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery failed for {error_type}: {recovery_error}",
                    exc_info=True
                )
                
        return None
        
    def _log_model_error(self, error: BaseModelError) -> None:
        """Log a model-specific error."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"{error.category.value.upper()} ERROR: {error.message}",
            extra={
                'error_type': type(error).__name__,
                'severity': error.severity.value,
                'category': error.category.value,
                'context': error.context.__dict__,
                'suggestions': error.suggestions,
                'recoverable': error.recoverable,
                'error_count': self.error_counts.get(type(error).__name__, 0)
            },
            exc_info=True
        )
        
    def _log_generic_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> None:
        """Log a generic error."""
        self.logger.error(
            f"Unhandled error: {error}",
            extra={
                'error_type': type(error).__name__,
                'context': context.__dict__ if context else {},
                'error_count': self.error_counts.get(type(error).__name__, 0)
            },
            exc_info=True
        )
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts.copy(),
            'registered_strategies': list(self.recovery_strategies.keys())
        }


# Global error handler instance
error_handler = ErrorHandler()


# Decorator for automatic error handling
def handle_errors(
    recovery_value: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    context: Optional[ErrorContext] = None
):
    """Decorator for automatic error handling.
    
    Args:
        recovery_value: Value to return on error
        log_errors: Whether to log errors
        reraise: Whether to reraise the exception
        context: Additional context information
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    error_handler.handle_error(e, context)
                    
                if reraise:
                    raise
                    
                return recovery_value
                
        return wrapper
    return decorator


# Validation utilities
class Validator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_numeric(
        value: Any,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_none: bool = False
    ) -> None:
        """Validate numeric value.
        
        Args:
            value: Value to validate
            name: Parameter name
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_none: Whether None is allowed
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None and allow_none:
            return
            
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"{name} must be numeric, got {type(value).__name__}",
                suggestions=[f"Convert {name} to float or int"]
            )
            
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{name} must be >= {min_value}, got {value}",
                suggestions=[f"Increase {name} to at least {min_value}"]
            )
            
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{name} must be <= {max_value}, got {value}",
                suggestions=[f"Decrease {name} to at most {max_value}"]
            )
    
    @staticmethod
    def validate_array_shape(
        array: Any,
        name: str,
        expected_shape: Optional[tuple] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None
    ) -> None:
        """Validate array shape.
        
        Args:
            array: Array to validate
            name: Parameter name
            expected_shape: Expected shape
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            import numpy as np
            if not isinstance(array, np.ndarray):
                array = np.array(array)
        except ImportError:
            pass
            
        if not hasattr(array, 'shape'):
            raise ValidationError(
                f"{name} must be array-like with shape attribute",
                suggestions=[f"Convert {name} to numpy array"]
            )
            
        shape = array.shape
        
        if expected_shape is not None and shape != expected_shape:
            raise ValidationError(
                f"{name} shape {shape} does not match expected {expected_shape}",
                suggestions=[f"Reshape {name} to {expected_shape}"]
            )
            
        if min_dims is not None and len(shape) < min_dims:
            raise ValidationError(
                f"{name} must have at least {min_dims} dimensions, got {len(shape)}",
                suggestions=[f"Add dimensions to {name}"]
            )
            
        if max_dims is not None and len(shape) > max_dims:
            raise ValidationError(
                f"{name} must have at most {max_dims} dimensions, got {len(shape)}",
                suggestions=[f"Reduce dimensions of {name}"]
            )
    
    @staticmethod
    def validate_dict_keys(
        data: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> None:
        """Validate dictionary keys.
        
        Args:
            data: Dictionary to validate
            required_keys: Required keys
            optional_keys: Optional keys
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(
                f"Expected dictionary, got {type(data).__name__}",
                suggestions=["Provide data as dictionary"]
            )
            
        missing_keys = set(required_keys) - set(data.keys())
        if missing_keys:
            raise ValidationError(
                f"Missing required keys: {missing_keys}",
                suggestions=[f"Add missing keys: {list(missing_keys)}"]
            )
            
        if optional_keys is not None:
            allowed_keys = set(required_keys) | set(optional_keys)
            extra_keys = set(data.keys()) - allowed_keys
            if extra_keys:
                raise ValidationError(
                    f"Unexpected keys: {extra_keys}",
                    suggestions=[f"Remove unexpected keys: {list(extra_keys)}"]
                )


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in a block of code."""
    
    def __init__(
        self,
        operation: str,
        model_name: Optional[str] = None,
        reraise: bool = True,
        recovery_value: Any = None
    ):
        """Initialize error context.
        
        Args:
            operation: Name of the operation
            model_name: Name of the model
            reraise: Whether to reraise exceptions
            recovery_value: Value to return on error
        """
        self.operation = operation
        self.model_name = model_name
        self.reraise = reraise
        self.recovery_value = recovery_value
        self.error = None
        
    def __enter__(self):
        """Enter error context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit error context."""
        if exc_type is not None:
            context = ErrorContext(
                model_name=self.model_name,
                operation=self.operation
            )
            
            self.error = exc_val
            error_handler.handle_error(exc_val, context)
            
            if not self.reraise:
                return True  # Suppress exception
                
        return False


# Recovery strategies
def default_convergence_recovery(error: ModelConvergenceError) -> Dict[str, Any]:
    """Default recovery strategy for convergence errors."""
    return {
        'status': 'partial_convergence',
        'message': 'Model did not fully converge, returning partial results',
        'suggestions': [
            'Increase maximum iterations',
            'Adjust convergence tolerance',
            'Check model specification',
            'Try different initial values'
        ]
    }


def default_data_recovery(error: DataError) -> Dict[str, Any]:
    """Default recovery strategy for data errors."""
    return {
        'status': 'data_issue',
        'message': 'Data issue encountered, using fallback data',
        'suggestions': [
            'Check data quality',
            'Verify data sources',
            'Update data preprocessing',
            'Use alternative data sources'
        ]
    }


# Register default recovery strategies
error_handler.register_recovery_strategy(ModelConvergenceError, default_convergence_recovery)
error_handler.register_recovery_strategy(DataError, default_data_recovery)


# Convenience functions
def validate_model_parameters(parameters: Dict[str, Any], model_name: str) -> None:
    """Validate model parameters.
    
    Args:
        parameters: Model parameters to validate
        model_name: Name of the model
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(parameters, dict):
        raise ValidationError(
            f"Model parameters for {model_name} must be a dictionary",
            context=ErrorContext(model_name=model_name),
            suggestions=["Provide parameters as dictionary"]
        )
        
    # Add model-specific validation logic here
    # This can be extended based on specific model requirements


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def safe_log(value: float, base: Optional[float] = None, default: float = float('-inf')) -> float:
    """Safely compute logarithm.
    
    Args:
        value: Value to compute log of
        base: Logarithm base (natural log if None)
        default: Default value for invalid inputs
        
    Returns:
        Logarithm result or default value
    """
    try:
        import math
        if value <= 0:
            return default
        if base is None:
            return math.log(value)
        else:
            return math.log(value, base)
    except (TypeError, ValueError, ZeroDivisionError):
        return default