"""Performance monitoring and optimization utilities for Bangladesh Macroeconomic Models.

This module provides:
- Execution time tracking
- Memory usage monitoring
- Performance profiling
- Bottleneck identification
- Optimization suggestions
- Resource usage alerts
"""

import functools
import logging
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    function_name: Optional[str] = None
    model_name: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_percent': self.cpu_percent,
            'peak_memory_mb': self.peak_memory_mb,
            'function_name': self.function_name,
            'model_name': self.model_name,
            'operation': self.operation,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters
        }


@dataclass
class ResourceThresholds:
    """Resource usage thresholds for alerts."""
    max_execution_time: float = 300.0  # 5 minutes
    max_memory_mb: float = 2048.0  # 2 GB
    max_cpu_percent: float = 90.0
    max_peak_memory_mb: float = 4096.0  # 4 GB


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        thresholds: Optional[ResourceThresholds] = None,
        history_size: int = 1000
    ):
        """Initialize performance monitor.
        
        Args:
            logger: Logger for performance data
            thresholds: Resource usage thresholds
            history_size: Maximum number of metrics to keep in history
        """
        self.logger = logger or logging.getLogger('performance')
        self.thresholds = thresholds or ResourceThresholds()
        self.history_size = history_size
        
        # Performance data storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.function_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.model_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._system_metrics: deque = deque(maxlen=100)
        
        # Alerts
        self.alert_callbacks: List[Callable] = []
        
    def start_system_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, system monitoring disabled")
            return
            
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_system_monitoring(self) -> None:
        """Stop continuous system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
        
    def _system_monitor_loop(self, interval: float) -> None:
        """System monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._get_system_metrics()
                self._system_metrics.append(metrics)
                
                # Check for threshold violations
                self._check_system_thresholds(metrics)
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)
                
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not PSUTIL_AVAILABLE:
            return {}
            
        try:
            process = psutil.Process()
            return {
                'timestamp': datetime.now(),
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'system_cpu_percent': psutil.cpu_percent(),
                'system_memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else None
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
            
    def _check_system_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check if system metrics exceed thresholds."""
        alerts = []
        
        if metrics.get('memory_mb', 0) > self.thresholds.max_memory_mb:
            alerts.append(f"High memory usage: {metrics['memory_mb']:.1f} MB")
            
        if metrics.get('cpu_percent', 0) > self.thresholds.max_cpu_percent:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
            
        for alert in alerts:
            self.logger.warning(alert)
            self._trigger_alerts(alert, metrics)
            
    def _trigger_alerts(self, message: str, metrics: Dict[str, Any]) -> None:
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(message, metrics)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
                
    def register_alert_callback(self, callback: Callable) -> None:
        """Register an alert callback function.
        
        Args:
            callback: Function to call when alerts are triggered
        """
        self.alert_callbacks.append(callback)
        
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics.
        
        Args:
            metrics: Performance metrics to record
        """
        # Add to history
        self.metrics_history.append(metrics)
        
        # Add to function stats
        if metrics.function_name:
            self.function_stats[metrics.function_name].append(metrics)
            
        # Add to model stats
        if metrics.model_name:
            self.model_stats[metrics.model_name].append(metrics)
            
        # Check thresholds
        self._check_performance_thresholds(metrics)
        
        # Log metrics
        self.logger.info(
            f"Performance: {metrics.function_name or 'unknown'} - "
            f"{metrics.execution_time:.3f}s",
            extra=metrics.to_dict()
        )
        
    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if performance metrics exceed thresholds."""
        alerts = []
        
        if metrics.execution_time > self.thresholds.max_execution_time:
            alerts.append(
                f"Long execution time: {metrics.execution_time:.1f}s "
                f"for {metrics.function_name or 'unknown'}"
            )
            
        if metrics.peak_memory_mb and metrics.peak_memory_mb > self.thresholds.max_peak_memory_mb:
            alerts.append(
                f"High peak memory: {metrics.peak_memory_mb:.1f} MB "
                f"for {metrics.function_name or 'unknown'}"
            )
            
        for alert in alerts:
            self.logger.warning(alert)
            self._trigger_alerts(alert, metrics.to_dict())
            
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Dictionary with function statistics
        """
        metrics_list = self.function_stats.get(function_name, [])
        if not metrics_list:
            return {}
            
        execution_times = [m.execution_time for m in metrics_list]
        memory_usages = [m.memory_usage_mb for m in metrics_list if m.memory_usage_mb]
        
        stats = {
            'call_count': len(metrics_list),
            'total_time': sum(execution_times),
            'avg_time': sum(execution_times) / len(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'last_called': max(m.timestamp for m in metrics_list).isoformat()
        }
        
        if memory_usages:
            stats.update({
                'avg_memory_mb': sum(memory_usages) / len(memory_usages),
                'min_memory_mb': min(memory_usages),
                'max_memory_mb': max(memory_usages)
            })
            
        return stats
        
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model statistics
        """
        metrics_list = self.model_stats.get(model_name, [])
        if not metrics_list:
            return {}
            
        # Group by operation
        operations = defaultdict(list)
        for metric in metrics_list:
            operations[metric.operation or 'unknown'].append(metric)
            
        stats = {
            'total_operations': len(metrics_list),
            'operations': {}
        }
        
        for operation, op_metrics in operations.items():
            execution_times = [m.execution_time for m in op_metrics]
            stats['operations'][operation] = {
                'count': len(op_metrics),
                'total_time': sum(execution_times),
                'avg_time': sum(execution_times) / len(execution_times),
                'min_time': min(execution_times),
                'max_time': max(execution_times)
            }
            
        return stats
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.metrics_history:
            return {}
            
        total_metrics = len(self.metrics_history)
        execution_times = [m.execution_time for m in self.metrics_history]
        
        # Top slowest functions
        function_times = defaultdict(list)
        for metric in self.metrics_history:
            if metric.function_name:
                function_times[metric.function_name].append(metric.execution_time)
                
        slowest_functions = sorted(
            [(name, max(times)) for name, times in function_times.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Recent performance trend
        recent_metrics = list(self.metrics_history)[-50:] if len(self.metrics_history) >= 50 else list(self.metrics_history)
        recent_avg_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            'total_operations': total_metrics,
            'total_execution_time': sum(execution_times),
            'avg_execution_time': sum(execution_times) / total_metrics,
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'recent_avg_time': recent_avg_time,
            'slowest_functions': slowest_functions,
            'unique_functions': len(function_times),
            'unique_models': len(self.model_stats)
        }
        
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on performance data.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        if not self.metrics_history:
            return suggestions
            
        # Analyze execution times
        execution_times = [m.execution_time for m in self.metrics_history]
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        if max_time > 60:  # More than 1 minute
            suggestions.append(
                "Consider breaking down long-running operations into smaller chunks"
            )
            
        if avg_time > 10:  # More than 10 seconds average
            suggestions.append(
                "Average execution time is high - consider caching or optimization"
            )
            
        # Analyze memory usage
        memory_usages = [m.memory_usage_mb for m in self.metrics_history if m.memory_usage_mb]
        if memory_usages:
            avg_memory = sum(memory_usages) / len(memory_usages)
            max_memory = max(memory_usages)
            
            if max_memory > 1024:  # More than 1 GB
                suggestions.append(
                    "High memory usage detected - consider data chunking or streaming"
                )
                
            if avg_memory > 512:  # More than 512 MB average
                suggestions.append(
                    "Consider optimizing data structures or using more efficient algorithms"
                )
                
        # Analyze function call patterns
        function_calls = defaultdict(int)
        for metric in self.metrics_history:
            if metric.function_name:
                function_calls[metric.function_name] += 1
                
        most_called = max(function_calls.items(), key=lambda x: x[1]) if function_calls else None
        if most_called and most_called[1] > 100:
            suggestions.append(
                f"Function '{most_called[0]}' called {most_called[1]} times - "
                "consider caching results"
            )
            
        return suggestions
        
    def clear_history(self) -> None:
        """Clear performance history."""
        self.metrics_history.clear()
        self.function_stats.clear()
        self.model_stats.clear()
        self._system_metrics.clear()
        self.logger.info("Performance history cleared")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorators for performance monitoring
def monitor_performance(
    model_name: Optional[str] = None,
    operation: Optional[str] = None,
    track_memory: bool = True
):
    """Decorator to monitor function performance.
    
    Args:
        model_name: Name of the model
        operation: Name of the operation
        track_memory: Whether to track memory usage
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = None
            peak_memory = None
            
            # Get initial memory usage
            if track_memory and PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024
                except Exception:
                    pass
                    
            # Track peak memory if memory_profiler is available
            if track_memory and MEMORY_PROFILER_AVAILABLE:
                try:
                    peak_memory = memory_profiler.memory_usage((func, args, kwargs), max_usage=True)
                    result = func(*args, **kwargs)
                except Exception:
                    # Fallback to normal execution
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Calculate metrics
            execution_time = time.time() - start_time
            end_memory = None
            cpu_percent = None
            
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                except Exception:
                    pass
                    
            # Create metrics object
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=end_memory,
                cpu_percent=cpu_percent,
                peak_memory_mb=peak_memory,
                function_name=func.__name__,
                model_name=model_name,
                operation=operation,
                parameters={
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            )
            
            # Record metrics
            performance_monitor.record_metrics(metrics)
            
            return result
            
        return wrapper
    return decorator


@contextmanager
def performance_context(
    operation: str,
    model_name: Optional[str] = None,
    track_memory: bool = True
):
    """Context manager for performance monitoring.
    
    Args:
        operation: Name of the operation
        model_name: Name of the model
        track_memory: Whether to track memory usage
    """
    start_time = time.time()
    start_memory = None
    
    if track_memory and PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024
        except Exception:
            pass
            
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        end_memory = None
        cpu_percent = None
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
            except Exception:
                pass
                
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=end_memory,
            cpu_percent=cpu_percent,
            function_name=operation,
            model_name=model_name,
            operation=operation
        )
        
        performance_monitor.record_metrics(metrics)


# Utility functions
def benchmark_function(func: Callable, *args, iterations: int = 10, **kwargs) -> Dict[str, Any]:
    """Benchmark a function over multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start_time)
        
    return {
        'iterations': iterations,
        'total_time': sum(times),
        'avg_time': sum(times) / iterations,
        'min_time': min(times),
        'max_time': max(times),
        'std_time': (sum((t - sum(times)/iterations)**2 for t in times) / iterations)**0.5
    }


def profile_memory_usage(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Profile memory usage of a function.
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, peak_memory_mb)
    """
    if not MEMORY_PROFILER_AVAILABLE:
        result = func(*args, **kwargs)
        return result, 0.0
        
    try:
        peak_memory = memory_profiler.memory_usage((func, args, kwargs), max_usage=True)
        result = func(*args, **kwargs)
        return result, peak_memory
    except Exception:
        result = func(*args, **kwargs)
        return result, 0.0


# Performance report generation
def generate_performance_report() -> str:
    """Generate a comprehensive performance report.
    
    Returns:
        Performance report as string
    """
    summary = performance_monitor.get_performance_summary()
    suggestions = performance_monitor.get_optimization_suggestions()
    
    report = []
    report.append("=" * 60)
    report.append("PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    if summary:
        report.append("SUMMARY:")
        report.append(f"  Total Operations: {summary['total_operations']}")
        report.append(f"  Total Execution Time: {summary['total_execution_time']:.2f}s")
        report.append(f"  Average Execution Time: {summary['avg_execution_time']:.3f}s")
        report.append(f"  Min/Max Execution Time: {summary['min_execution_time']:.3f}s / {summary['max_execution_time']:.3f}s")
        report.append(f"  Recent Average Time: {summary['recent_avg_time']:.3f}s")
        report.append(f"  Unique Functions: {summary['unique_functions']}")
        report.append(f"  Unique Models: {summary['unique_models']}")
        report.append("")
        
        if summary['slowest_functions']:
            report.append("SLOWEST FUNCTIONS:")
            for func_name, max_time in summary['slowest_functions']:
                report.append(f"  {func_name}: {max_time:.3f}s")
            report.append("")
    
    if suggestions:
        report.append("OPTIMIZATION SUGGESTIONS:")
        for i, suggestion in enumerate(suggestions, 1):
            report.append(f"  {i}. {suggestion}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)