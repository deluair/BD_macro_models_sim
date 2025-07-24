#!/usr/bin/env python3
"""Performance benchmarking script for Bangladesh Macroeconomic Models.

This script provides comprehensive performance benchmarking for all models,
including execution time, memory usage, and scalability analysis.

Usage:
    python scripts/benchmark_models.py --models dsge,cge --output benchmarks.json
    python scripts/benchmark_models.py --all --detailed --save-plots
"""

import argparse
import json
import time
import sys
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

try:
    from memory_profiler import memory_usage
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print("Warning: memory_profiler not available. Detailed memory profiling disabled.")

# Import models (with fallbacks)
try:
    from models.dsge_model import DSGEModel, DSGEParameters
    from models.cge_model import CGEModel, CGEParameters
    from models.abm_model import ABMModel, ABMParameters
    from models.svar_model import SVARModel, SVARParameters
except ImportError:
    print("Warning: Some models not available. Creating mock implementations.")
    
    class MockModel:
        def __init__(self, parameters):
            self.parameters = parameters
            self.name = self.__class__.__name__
            
        def calibrate(self):
            time.sleep(0.1)  # Simulate work
            return {'status': 'success'}
            
        def simulate(self, periods=100):
            time.sleep(0.05 * periods / 100)  # Scale with periods
            return {'status': 'success', 'periods': periods}
            
        def forecast(self, periods=20):
            time.sleep(0.02 * periods / 20)
            return {'status': 'success', 'periods': periods}
    
    DSGEModel = CGEModel = ABMModel = SVARModel = MockModel
    DSGEParameters = CGEParameters = ABMParameters = SVARParameters = dict


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    operation: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    parameters: Dict[str, Any]
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelBenchmarker:
    """Comprehensive model benchmarking utility."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path('outputs/benchmarks')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
        # Model configurations
        self.model_configs = {
            'dsge': {
                'class': DSGEModel,
                'params': DSGEParameters(
                    beta=0.99,
                    alpha=0.33,
                    delta=0.025,
                    rho=0.95,
                    sigma=0.007
                ) if hasattr(DSGEParameters, '__call__') else {
                    'beta': 0.99, 'alpha': 0.33, 'delta': 0.025
                },
                'operations': ['calibrate', 'simulate', 'forecast']
            },
            'cge': {
                'class': CGEModel,
                'params': CGEParameters(
                    sectors=['agriculture', 'manufacturing', 'services'],
                    factors=['labor', 'capital'],
                    households=['rural', 'urban']
                ) if hasattr(CGEParameters, '__call__') else {
                    'sectors': ['agriculture', 'manufacturing', 'services']
                },
                'operations': ['calibrate', 'simulate']
            },
            'abm': {
                'class': ABMModel,
                'params': ABMParameters(
                    n_agents=1000,
                    n_periods=50,
                    learning_rate=0.1
                ) if hasattr(ABMParameters, '__call__') else {
                    'n_agents': 1000, 'n_periods': 50
                },
                'operations': ['calibrate', 'simulate']
            },
            'svar': {
                'class': SVARModel,
                'params': SVARParameters(
                    variables=['gdp', 'inflation', 'unemployment'],
                    lags=4,
                    identification='cholesky'
                ) if hasattr(SVARParameters, '__call__') else {
                    'variables': ['gdp', 'inflation', 'unemployment']
                },
                'operations': ['estimate', 'forecast', 'impulse_response']
            }
        }
        
    def benchmark_operation(
        self,
        model_name: str,
        operation: str,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a single model operation."""
        config = self.model_configs[model_name]
        
        # Initialize model
        model = config['class'](config['params'])
        
        # Prepare monitoring
        if HAS_PSUTIL:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        else:
            initial_memory = 0
            
        # Force garbage collection
        gc.collect()
        
        start_time = time.time()
        success = True
        error_message = None
        peak_memory = initial_memory
        
        try:
            if HAS_MEMORY_PROFILER:
                # Use memory_profiler for detailed monitoring
                def run_operation():
                    if operation == 'calibrate':
                        return model.calibrate()
                    elif operation == 'simulate':
                        periods = kwargs.get('periods', 100)
                        return model.simulate(periods=periods)
                    elif operation == 'forecast':
                        periods = kwargs.get('forecast_periods', 20)
                        return model.forecast(periods=periods)
                    elif operation == 'estimate':
                        # Generate sample data for estimation
                        data = self._generate_sample_data(model_name)
                        return model.estimate(data)
                    elif operation == 'impulse_response':
                        return model.impulse_response(periods=20)
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                
                memory_usage_list = memory_usage(run_operation, interval=0.1)
                peak_memory = max(memory_usage_list)
                result = run_operation()  # Run again to get result
                
            else:
                # Fallback without detailed memory monitoring
                if operation == 'calibrate':
                    result = model.calibrate()
                elif operation == 'simulate':
                    periods = kwargs.get('periods', 100)
                    result = model.simulate(periods=periods)
                elif operation == 'forecast':
                    periods = kwargs.get('forecast_periods', 20)
                    result = model.forecast(periods=periods)
                elif operation == 'estimate':
                    data = self._generate_sample_data(model_name)
                    result = model.estimate(data)
                elif operation == 'impulse_response':
                    result = model.impulse_response(periods=20)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                    
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get final memory usage
        if HAS_PSUTIL:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            cpu_percent = process.cpu_percent()
        else:
            memory_usage = 0
            cpu_percent = 0
            
        return BenchmarkResult(
            model_name=model_name,
            operation=operation,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            cpu_percent=cpu_percent,
            parameters=kwargs,
            timestamp=datetime.now().isoformat(),
            success=success,
            error_message=error_message
        )
        
    def _generate_sample_data(self, model_name: str) -> pd.DataFrame:
        """Generate sample data for model estimation."""
        np.random.seed(42)
        periods = 100
        
        if model_name == 'svar':
            # Time series data for SVAR
            dates = pd.date_range('2000-01-01', periods=periods, freq='Q')
            data = pd.DataFrame({
                'date': dates,
                'gdp': np.random.normal(0.02, 0.01, periods).cumsum(),
                'inflation': np.random.normal(0.05, 0.02, periods),
                'unemployment': np.random.normal(0.06, 0.01, periods)
            })
        else:
            # General macro data
            dates = pd.date_range('2000-01-01', periods=periods, freq='Q')
            gdp = 1000 * (1 + np.random.normal(0.02, 0.01, periods)).cumprod()
            data = pd.DataFrame({
                'date': dates,
                'gdp': gdp,
                'consumption': gdp * 0.6,
                'investment': gdp * 0.2,
                'government': gdp * 0.15,
                'exports': gdp * 0.25,
                'imports': gdp * 0.2
            })
            
        return data
        
    def benchmark_model(self, model_name: str, detailed: bool = False) -> List[BenchmarkResult]:
        """Benchmark all operations for a specific model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
            
        config = self.model_configs[model_name]
        results = []
        
        print(f"\nBenchmarking {model_name.upper()} model...")
        
        for operation in config['operations']:
            print(f"  Testing {operation}...", end=' ')
            
            if detailed:
                # Test with different parameter sets
                if operation == 'simulate':
                    for periods in [50, 100, 200, 500]:
                        result = self.benchmark_operation(
                            model_name, operation, periods=periods
                        )
                        results.append(result)
                        
                elif operation == 'forecast':
                    for periods in [10, 20, 50, 100]:
                        result = self.benchmark_operation(
                            model_name, operation, forecast_periods=periods
                        )
                        results.append(result)
                else:
                    result = self.benchmark_operation(model_name, operation)
                    results.append(result)
            else:
                # Single test with default parameters
                result = self.benchmark_operation(model_name, operation)
                results.append(result)
                
            if results[-1].success:
                print(f"✓ ({results[-1].execution_time:.3f}s)")
            else:
                print(f"✗ ({results[-1].error_message})")
                
        return results
        
    def benchmark_all_models(self, models: List[str] = None, detailed: bool = False) -> List[BenchmarkResult]:
        """Benchmark all specified models."""
        if models is None:
            models = list(self.model_configs.keys())
            
        all_results = []
        
        print("Starting comprehensive model benchmarking...")
        print(f"Models to test: {', '.join(models)}")
        print(f"Detailed mode: {detailed}")
        
        for model_name in models:
            try:
                results = self.benchmark_model(model_name, detailed=detailed)
                all_results.extend(results)
                self.results.extend(results)
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                
        return all_results
        
    def scalability_test(self, model_name: str, operation: str, scale_param: str, scale_values: List[int]) -> List[BenchmarkResult]:
        """Test model scalability with different parameter values."""
        print(f"\nScalability test: {model_name}.{operation} scaling {scale_param}")
        
        results = []
        for value in scale_values:
            print(f"  Testing {scale_param}={value}...", end=' ')
            
            kwargs = {scale_param: value}
            result = self.benchmark_operation(model_name, operation, **kwargs)
            results.append(result)
            
            if result.success:
                print(f"✓ ({result.execution_time:.3f}s, {result.memory_usage_mb:.1f}MB)")
            else:
                print(f"✗ ({result.error_message})")
                
        return results
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}
            
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        # Summary statistics
        summary = {
            'total_tests': len(self.results),
            'successful_tests': df['success'].sum(),
            'failed_tests': (~df['success']).sum(),
            'total_execution_time': df['execution_time'].sum(),
            'average_execution_time': df['execution_time'].mean(),
            'total_memory_usage': df['memory_usage_mb'].sum(),
            'average_memory_usage': df['memory_usage_mb'].mean()
        }
        
        # Per-model statistics
        model_stats = {}
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            model_stats[model] = {
                'tests': len(model_df),
                'success_rate': model_df['success'].mean(),
                'avg_execution_time': model_df['execution_time'].mean(),
                'avg_memory_usage': model_df['memory_usage_mb'].mean(),
                'operations': model_df['operation'].unique().tolist()
            }
            
        # Per-operation statistics
        operation_stats = {}
        for operation in df['operation'].unique():
            op_df = df[df['operation'] == operation]
            operation_stats[operation] = {
                'tests': len(op_df),
                'success_rate': op_df['success'].mean(),
                'avg_execution_time': op_df['execution_time'].mean(),
                'avg_memory_usage': op_df['memory_usage_mb'].mean(),
                'models': op_df['model_name'].unique().tolist()
            }
            
        # Performance rankings
        successful_df = df[df['success']]
        if len(successful_df) > 0:
            fastest_operations = successful_df.nsmallest(5, 'execution_time')[['model_name', 'operation', 'execution_time']].to_dict('records')
            slowest_operations = successful_df.nlargest(5, 'execution_time')[['model_name', 'operation', 'execution_time']].to_dict('records')
            memory_efficient = successful_df.nsmallest(5, 'memory_usage_mb')[['model_name', 'operation', 'memory_usage_mb']].to_dict('records')
            memory_intensive = successful_df.nlargest(5, 'memory_usage_mb')[['model_name', 'operation', 'memory_usage_mb']].to_dict('records')
        else:
            fastest_operations = slowest_operations = memory_efficient = memory_intensive = []
            
        return {
            'summary': summary,
            'model_statistics': model_stats,
            'operation_statistics': operation_stats,
            'performance_rankings': {
                'fastest_operations': fastest_operations,
                'slowest_operations': slowest_operations,
                'memory_efficient': memory_efficient,
                'memory_intensive': memory_intensive
            },
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info()
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        if HAS_PSUTIL:
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
            })
            
        return info
        
    def save_results(self, filename: str = None) -> Path:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'benchmark_results_{timestamp}.json'
            
        filepath = self.output_dir / filename
        
        report = self.generate_report()
        report['raw_results'] = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\nResults saved to: {filepath}")
        return filepath
        
    def create_visualizations(self, save_plots: bool = True) -> None:
        """Create benchmark visualization plots."""
        if not self.results:
            print("No results to visualize")
            return
            
        df = pd.DataFrame([result.to_dict() for result in self.results])
        successful_df = df[df['success']]
        
        if len(successful_df) == 0:
            print("No successful results to visualize")
            return
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Benchmarks', fontsize=16, fontweight='bold')
        
        # 1. Execution time by model and operation
        pivot_time = successful_df.pivot_table(
            values='execution_time', 
            index='model_name', 
            columns='operation', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_time, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Average Execution Time (seconds)')
        axes[0,0].set_xlabel('Operation')
        axes[0,0].set_ylabel('Model')
        
        # 2. Memory usage by model and operation
        pivot_memory = successful_df.pivot_table(
            values='memory_usage_mb', 
            index='model_name', 
            columns='operation', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_memory, annot=True, fmt='.1f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Average Memory Usage (MB)')
        axes[0,1].set_xlabel('Operation')
        axes[0,1].set_ylabel('Model')
        
        # 3. Execution time distribution
        sns.boxplot(data=successful_df, x='model_name', y='execution_time', ax=axes[1,0])
        axes[1,0].set_title('Execution Time Distribution by Model')
        axes[1,0].set_xlabel('Model')
        axes[1,0].set_ylabel('Execution Time (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Memory vs Time scatter
        for model in successful_df['model_name'].unique():
            model_data = successful_df[successful_df['model_name'] == model]
            axes[1,1].scatter(
                model_data['execution_time'], 
                model_data['memory_usage_mb'], 
                label=model, 
                alpha=0.7,
                s=60
            )
        axes[1,1].set_xlabel('Execution Time (seconds)')
        axes[1,1].set_ylabel('Memory Usage (MB)')
        axes[1,1].set_title('Memory Usage vs Execution Time')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = self.output_dir / f'benchmark_plots_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
            
        plt.show()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Benchmark Bangladesh Macroeconomic Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_models.py --models dsge,cge
  python scripts/benchmark_models.py --all --detailed
  python scripts/benchmark_models.py --scalability dsge simulate periods 50,100,200,500
        """
    )
    
    parser.add_argument(
        '--models', 
        type=str, 
        help='Comma-separated list of models to benchmark (dsge,cge,abm,svar)'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Benchmark all available models'
    )
    parser.add_argument(
        '--detailed', 
        action='store_true', 
        help='Run detailed benchmarks with multiple parameter sets'
    )
    parser.add_argument(
        '--scalability', 
        nargs=4, 
        metavar=('MODEL', 'OPERATION', 'PARAMETER', 'VALUES'),
        help='Run scalability test: MODEL OPERATION PARAMETER VALUES (comma-separated)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help='Output filename for results (default: auto-generated)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs/benchmarks',
        help='Output directory for results (default: outputs/benchmarks)'
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true', 
        help='Save visualization plots'
    )
    parser.add_argument(
        '--no-plots', 
        action='store_true', 
        help='Skip creating plots'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(output_dir=Path(args.output_dir))
    
    try:
        if args.scalability:
            # Run scalability test
            model, operation, parameter, values_str = args.scalability
            values = [int(v.strip()) for v in values_str.split(',')]
            
            results = benchmarker.scalability_test(model, operation, parameter, values)
            benchmarker.results.extend(results)
            
        elif args.all:
            # Benchmark all models
            benchmarker.benchmark_all_models(detailed=args.detailed)
            
        elif args.models:
            # Benchmark specified models
            models = [m.strip() for m in args.models.split(',')]
            benchmarker.benchmark_all_models(models=models, detailed=args.detailed)
            
        else:
            # Default: benchmark all models
            benchmarker.benchmark_all_models()
            
        # Generate and display report
        report = benchmarker.generate_report()
        
        print("\n" + "="*60)
        print("BENCHMARK REPORT")
        print("="*60)
        
        summary = report['summary']
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Total execution time: {summary['total_execution_time']:.3f}s")
        print(f"Average execution time: {summary['average_execution_time']:.3f}s")
        print(f"Average memory usage: {summary['average_memory_usage']:.1f}MB")
        
        # Save results
        benchmarker.save_results(args.output)
        
        # Create visualizations
        if not args.no_plots:
            benchmarker.create_visualizations(save_plots=args.save_plots)
            
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()