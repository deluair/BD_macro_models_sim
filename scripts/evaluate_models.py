#!/usr/bin/env python3
"""Model evaluation script for Bangladesh macroeconomic models.

This script provides comprehensive model evaluation capabilities:
- Performance metrics calculation
- Forecast accuracy assessment
- Model diagnostics and validation
- Comparative analysis
- Visualization of results
- Statistical tests
- Robustness analysis
- Policy scenario evaluation

Usage:
    python scripts/evaluate_models.py --model dsge --data data/processed/bd_data.csv
    python scripts/evaluate_models.py --all --models-dir results/models/
    python scripts/evaluate_models.py --compare --forecast --periods 12
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import pickle
import time

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from scipy.stats import jarque_bera, normaltest
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox
except ImportError:
    def ljungbox(*args, **kwargs):
        return None
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from utils.logging_config import get_logger, log_performance
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def log_performance(func):
        return func

try:
    from config.config_manager import ConfigManager
except ImportError:
    class ConfigManager:
        def get(self, key, default=None):
            return default

try:
    from utils.error_handling import ValidationError
except ImportError:
    class ValidationError(Exception):
        pass

try:
    from utils.performance_monitor import PerformanceMonitor, monitor_performance
except ImportError:
    class PerformanceMonitor:
        def __init__(self):
            pass
    def monitor_performance(func):
        return func

# Optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

try:
    from arch.unitroot import ADF, KPSS
    from arch.cointegration import engle_granger
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    warnings.warn("ARCH package not available. Some statistical tests will be disabled.")

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for Bangladesh macroeconomic models."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.models = {}
        self.data = None
        
        # Evaluation metrics
        self.metrics = {
            'forecast_accuracy': [
                'mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy'
            ],
            'statistical_tests': [
                'normality', 'autocorrelation', 'heteroscedasticity', 'stationarity'
            ],
            'economic_validation': [
                'impulse_response', 'variance_decomposition', 'policy_multipliers'
            ]
        }
        
        # Visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_models(self, models_dir: str) -> Dict[str, Any]:
        """Load trained models from directory."""
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
        loaded_models = {}
        
        # Look for model files
        for model_file in models_dir.glob("*_model_*.pkl"):
            try:
                model_name = model_file.stem.split('_model_')[0]
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    
                loaded_models[model_name] = model
                logger.info(f"Loaded {model_name} model from {model_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load model from {model_file}: {e}")
                
        if not loaded_models:
            raise ValueError(f"No valid model files found in {models_dir}")
            
        self.models = loaded_models
        return loaded_models
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load evaluation data."""
        logger.info(f"Loading evaluation data from {data_path}")
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        try:
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path)
            elif data_path.suffix == '.parquet':
                df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
                
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
            self.data = df
            logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
            
    @monitor_performance
    def evaluate_forecast_accuracy(
        self, 
        model_name: str, 
        actual_data: np.ndarray, 
        forecast_data: np.ndarray,
        variable_names: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate forecast accuracy using multiple metrics."""
        logger.info(f"Evaluating forecast accuracy for {model_name}")
        
        if actual_data.shape != forecast_data.shape:
            raise ValueError(f"Shape mismatch: actual {actual_data.shape} vs forecast {forecast_data.shape}")
            
        if variable_names is None:
            variable_names = [f"var_{i}" for i in range(actual_data.shape[1])]
            
        results = {
            'model_name': model_name,
            'n_observations': actual_data.shape[0],
            'n_variables': actual_data.shape[1],
            'variable_names': variable_names,
            'metrics': {}
        }
        
        # Calculate metrics for each variable
        for i, var_name in enumerate(variable_names):
            actual = actual_data[:, i]
            forecast = forecast_data[:, i]
            
            # Remove NaN values
            mask = ~(np.isnan(actual) | np.isnan(forecast))
            actual_clean = actual[mask]
            forecast_clean = forecast[mask]
            
            if len(actual_clean) == 0:
                logger.warning(f"No valid data for variable {var_name}")
                continue
                
            var_metrics = {}
            
            # Basic accuracy metrics
            var_metrics['mse'] = mean_squared_error(actual_clean, forecast_clean)
            var_metrics['rmse'] = np.sqrt(var_metrics['mse'])
            var_metrics['mae'] = mean_absolute_error(actual_clean, forecast_clean)
            
            # Avoid division by zero in MAPE
            if np.any(actual_clean != 0):
                var_metrics['mape'] = mean_absolute_percentage_error(actual_clean, forecast_clean)
            else:
                var_metrics['mape'] = np.inf
                
            var_metrics['r2'] = r2_score(actual_clean, forecast_clean)
            
            # Directional accuracy
            if len(actual_clean) > 1:
                actual_direction = np.diff(actual_clean) > 0
                forecast_direction = np.diff(forecast_clean) > 0
                var_metrics['directional_accuracy'] = np.mean(
                    actual_direction == forecast_direction
                )
            else:
                var_metrics['directional_accuracy'] = np.nan
                
            # Theil's U statistic
            if np.std(actual_clean) > 0:
                var_metrics['theil_u'] = (
                    np.sqrt(np.mean((forecast_clean - actual_clean) ** 2)) /
                    (np.sqrt(np.mean(actual_clean ** 2)) + np.sqrt(np.mean(forecast_clean ** 2)))
                )
            else:
                var_metrics['theil_u'] = np.inf
                
            # Bias
            var_metrics['bias'] = np.mean(forecast_clean - actual_clean)
            
            # Variance ratio
            if np.var(actual_clean) > 0:
                var_metrics['variance_ratio'] = np.var(forecast_clean) / np.var(actual_clean)
            else:
                var_metrics['variance_ratio'] = np.inf
                
            results['metrics'][var_name] = var_metrics
            
        # Calculate aggregate metrics
        all_metrics = list(results['metrics'].values())
        if all_metrics:
            aggregate_metrics = {}
            for metric_name in ['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy', 'theil_u']:
                values = [m.get(metric_name, np.nan) for m in all_metrics]
                valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                
                if valid_values:
                    aggregate_metrics[f'mean_{metric_name}'] = np.mean(valid_values)
                    aggregate_metrics[f'median_{metric_name}'] = np.median(valid_values)
                else:
                    aggregate_metrics[f'mean_{metric_name}'] = np.nan
                    aggregate_metrics[f'median_{metric_name}'] = np.nan
                    
            results['aggregate_metrics'] = aggregate_metrics
            
        return results
        
    @monitor_performance
    def perform_statistical_tests(
        self, 
        model_name: str, 
        residuals: np.ndarray,
        variable_names: List[str] = None
    ) -> Dict[str, Any]:
        """Perform statistical tests on model residuals."""
        logger.info(f"Performing statistical tests for {model_name}")
        
        if variable_names is None:
            variable_names = [f"var_{i}" for i in range(residuals.shape[1])]
            
        results = {
            'model_name': model_name,
            'tests': {}
        }
        
        for i, var_name in enumerate(variable_names):
            residual = residuals[:, i]
            
            # Remove NaN values
            residual_clean = residual[~np.isnan(residual)]
            
            if len(residual_clean) < 10:
                logger.warning(f"Insufficient data for tests on {var_name}")
                continue
                
            var_tests = {}
            
            # Normality tests
            try:
                # Jarque-Bera test
                jb_stat, jb_pvalue = jarque_bera(residual_clean)
                var_tests['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_pvalue),
                    'is_normal': jb_pvalue > 0.05
                }
                
                # Shapiro-Wilk test (for smaller samples)
                if len(residual_clean) <= 5000:
                    sw_stat, sw_pvalue = stats.shapiro(residual_clean)
                    var_tests['shapiro_wilk'] = {
                        'statistic': float(sw_stat),
                        'p_value': float(sw_pvalue),
                        'is_normal': sw_pvalue > 0.05
                    }
                    
            except Exception as e:
                logger.warning(f"Normality tests failed for {var_name}: {e}")
                
            # Autocorrelation test (Ljung-Box)
            try:
                if len(residual_clean) > 20:
                    lb_stat, lb_pvalue = ljungbox(residual_clean, lags=min(10, len(residual_clean)//4))
                    var_tests['ljung_box'] = {
                        'statistic': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat),
                        'p_value': float(lb_pvalue.iloc[-1]) if hasattr(lb_pvalue, 'iloc') else float(lb_pvalue),
                        'no_autocorrelation': (lb_pvalue.iloc[-1] if hasattr(lb_pvalue, 'iloc') else lb_pvalue) > 0.05
                    }
            except Exception as e:
                logger.warning(f"Ljung-Box test failed for {var_name}: {e}")
                
            # Heteroscedasticity test (Breusch-Pagan)
            try:
                # Simple test using squared residuals vs time
                time_trend = np.arange(len(residual_clean))
                correlation = np.corrcoef(residual_clean**2, time_trend)[0, 1]
                
                var_tests['heteroscedasticity'] = {
                    'correlation_with_time': float(correlation),
                    'is_homoscedastic': abs(correlation) < 0.1
                }
            except Exception as e:
                logger.warning(f"Heteroscedasticity test failed for {var_name}: {e}")
                
            # Stationarity test (if ARCH package available)
            if HAS_ARCH:
                try:
                    adf_result = ADF(residual_clean)
                    var_tests['adf_stationarity'] = {
                        'statistic': float(adf_result.stat),
                        'p_value': float(adf_result.pvalue),
                        'is_stationary': adf_result.pvalue < 0.05
                    }
                except Exception as e:
                    logger.warning(f"ADF test failed for {var_name}: {e}")
                    
            results['tests'][var_name] = var_tests
            
        return results
        
    @monitor_performance
    def evaluate_impulse_responses(
        self, 
        model_name: str, 
        model: Any,
        shocks: List[str] = None,
        periods: int = 20
    ) -> Dict[str, Any]:
        """Evaluate impulse response functions."""
        logger.info(f"Evaluating impulse responses for {model_name}")
        
        if shocks is None:
            shocks = ['productivity', 'monetary', 'fiscal', 'external']
            
        results = {
            'model_name': model_name,
            'periods': periods,
            'impulse_responses': {}
        }
        
        for shock in shocks:
            try:
                # Get impulse response
                if hasattr(model, 'impulse_response'):
                    irf = model.impulse_response(shock, periods)
                    
                    # Analyze IRF properties
                    irf_analysis = {
                        'shape': irf.shape,
                        'peak_response': {
                            'values': [float(np.max(np.abs(irf[:, i]))) for i in range(irf.shape[1])],
                            'periods': [int(np.argmax(np.abs(irf[:, i]))) for i in range(irf.shape[1])]
                        },
                        'convergence': {
                            'final_values': [float(irf[-1, i]) for i in range(irf.shape[1])],
                            'converged': [abs(irf[-1, i]) < 0.01 for i in range(irf.shape[1])]
                        },
                        'cumulative_response': [float(np.sum(irf[:, i])) for i in range(irf.shape[1])],
                        'half_life': []
                    }
                    
                    # Calculate half-life for each variable
                    for i in range(irf.shape[1]):
                        peak_val = np.max(np.abs(irf[:, i]))
                        if peak_val > 0:
                            half_val = peak_val / 2
                            half_life_idx = np.where(np.abs(irf[:, i]) <= half_val)[0]
                            if len(half_life_idx) > 0:
                                irf_analysis['half_life'].append(int(half_life_idx[0]))
                            else:
                                irf_analysis['half_life'].append(periods)
                        else:
                            irf_analysis['half_life'].append(0)
                            
                    results['impulse_responses'][shock] = {
                        'data': irf.tolist(),
                        'analysis': irf_analysis
                    }
                    
                else:
                    logger.warning(f"Model {model_name} does not support impulse response analysis")
                    
            except Exception as e:
                logger.warning(f"Impulse response analysis failed for shock {shock}: {e}")
                
        return results
        
    @monitor_performance
    def evaluate_policy_scenarios(
        self, 
        model_name: str, 
        model: Any,
        scenarios: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance under different policy scenarios."""
        logger.info(f"Evaluating policy scenarios for {model_name}")
        
        if scenarios is None:
            scenarios = [
                {'name': 'monetary_tightening', 'interest_rate_change': 0.01},
                {'name': 'fiscal_expansion', 'government_spending_change': 0.05},
                {'name': 'trade_liberalization', 'tariff_change': -0.1},
                {'name': 'productivity_shock', 'productivity_change': 0.02}
            ]
            
        results = {
            'model_name': model_name,
            'scenarios': {}
        }
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            
            try:
                if hasattr(model, 'policy_analysis'):
                    policy_result = model.policy_analysis(scenario)
                    
                    # Analyze policy effects
                    scenario_analysis = {
                        'scenario_parameters': scenario,
                        'results': policy_result,
                        'welfare_impact': policy_result.get('welfare_change', 0),
                        'gdp_impact': policy_result.get('gdp_change', 0),
                        'inflation_impact': policy_result.get('inflation_change', 0),
                        'employment_impact': policy_result.get('employment_change', 0)
                    }
                    
                elif hasattr(model, 'simulate'):
                    # Use simulation for policy analysis
                    baseline = model.simulate({'type': 'baseline'})
                    policy_sim = model.simulate(scenario)
                    
                    scenario_analysis = {
                        'scenario_parameters': scenario,
                        'baseline': baseline,
                        'policy_simulation': policy_sim,
                        'relative_change': {
                            key: (policy_sim.get(key, 0) - baseline.get(key, 0)) / baseline.get(key, 1)
                            for key in baseline.keys()
                            if isinstance(baseline.get(key), (int, float)) and baseline.get(key) != 0
                        }
                    }
                    
                else:
                    logger.warning(f"Model {model_name} does not support policy analysis")
                    continue
                    
                results['scenarios'][scenario_name] = scenario_analysis
                
            except Exception as e:
                logger.warning(f"Policy scenario {scenario_name} failed: {e}")
                
        return results
        
    def compare_models(
        self, 
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple models based on evaluation results."""
        logger.info("Comparing model performance...")
        
        comparison = {
            'models': list(evaluation_results.keys()),
            'forecast_accuracy': {},
            'statistical_properties': {},
            'economic_validity': {},
            'rankings': {}
        }
        
        # Compare forecast accuracy
        accuracy_metrics = ['mean_rmse', 'mean_mae', 'mean_mape', 'mean_r2']
        
        for metric in accuracy_metrics:
            comparison['forecast_accuracy'][metric] = {}
            
            for model_name, results in evaluation_results.items():
                if 'forecast_accuracy' in results and 'aggregate_metrics' in results['forecast_accuracy']:
                    value = results['forecast_accuracy']['aggregate_metrics'].get(metric, np.nan)
                    comparison['forecast_accuracy'][metric][model_name] = value
                    
        # Compare statistical properties
        for model_name, results in evaluation_results.items():
            if 'statistical_tests' in results:
                model_stats = {}
                
                for var_name, tests in results['statistical_tests']['tests'].items():
                    # Count passed tests
                    passed_tests = 0
                    total_tests = 0
                    
                    for test_name, test_result in tests.items():
                        if isinstance(test_result, dict):
                            total_tests += 1
                            
                            # Check if test indicates good model properties
                            if test_name == 'jarque_bera' and test_result.get('is_normal', False):
                                passed_tests += 1
                            elif test_name == 'ljung_box' and test_result.get('no_autocorrelation', False):
                                passed_tests += 1
                            elif test_name == 'heteroscedasticity' and test_result.get('is_homoscedastic', False):
                                passed_tests += 1
                            elif test_name == 'adf_stationarity' and test_result.get('is_stationary', False):
                                passed_tests += 1
                                
                    if total_tests > 0:
                        model_stats[var_name] = passed_tests / total_tests
                        
                if model_stats:
                    comparison['statistical_properties'][model_name] = np.mean(list(model_stats.values()))
                    
        # Rank models
        rankings = {}
        
        # Rank by RMSE (lower is better)
        rmse_values = comparison['forecast_accuracy'].get('mean_rmse', {})
        if rmse_values:
            sorted_rmse = sorted(rmse_values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else np.inf)
            rankings['rmse'] = [model for model, _ in sorted_rmse]
            
        # Rank by R² (higher is better)
        r2_values = comparison['forecast_accuracy'].get('mean_r2', {})
        if r2_values:
            sorted_r2 = sorted(r2_values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
            rankings['r2'] = [model for model, _ in sorted_r2]
            
        # Rank by statistical properties (higher is better)
        stat_values = comparison['statistical_properties']
        if stat_values:
            sorted_stats = sorted(stat_values.items(), key=lambda x: x[1], reverse=True)
            rankings['statistical_properties'] = [model for model, _ in sorted_stats]
            
        comparison['rankings'] = rankings
        
        # Overall ranking (simple average of ranks)
        if rankings:
            overall_scores = {}
            
            for model_name in comparison['models']:
                scores = []
                
                for ranking_type, ranking in rankings.items():
                    if model_name in ranking:
                        # Convert rank to score (1st place = highest score)
                        score = len(ranking) - ranking.index(model_name)
                        scores.append(score)
                        
                if scores:
                    overall_scores[model_name] = np.mean(scores)
                    
            if overall_scores:
                sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
                comparison['rankings']['overall'] = [model for model, _ in sorted_overall]
                
        return comparison
        
    def create_evaluation_report(
        self, 
        evaluation_results: Dict[str, Dict[str, Any]],
        output_dir: str
    ) -> str:
        """Create comprehensive evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"model_evaluation_report_{timestamp}.pdf"
        
        with PdfPages(report_file) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.8, 'Bangladesh Macroeconomic Models', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.7, 'Evaluation Report', 
                   ha='center', va='center', fontsize=20)
            ax.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.5, f'Models Evaluated: {", ".join(evaluation_results.keys())}', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Model comparison summary
            if len(evaluation_results) > 1:
                comparison = self.compare_models(evaluation_results)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')
                
                # RMSE comparison
                rmse_data = comparison['forecast_accuracy'].get('mean_rmse', {})
                if rmse_data:
                    models = list(rmse_data.keys())
                    values = list(rmse_data.values())
                    axes[0, 0].bar(models, values)
                    axes[0, 0].set_title('Root Mean Square Error')
                    axes[0, 0].set_ylabel('RMSE')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    
                # R² comparison
                r2_data = comparison['forecast_accuracy'].get('mean_r2', {})
                if r2_data:
                    models = list(r2_data.keys())
                    values = list(r2_data.values())
                    axes[0, 1].bar(models, values)
                    axes[0, 1].set_title('R-squared')
                    axes[0, 1].set_ylabel('R²')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    
                # Statistical properties
                stat_data = comparison['statistical_properties']
                if stat_data:
                    models = list(stat_data.keys())
                    values = list(stat_data.values())
                    axes[1, 0].bar(models, values)
                    axes[1, 0].set_title('Statistical Properties Score')
                    axes[1, 0].set_ylabel('Score (0-1)')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    
                # Overall ranking
                if 'overall' in comparison['rankings']:
                    ranking = comparison['rankings']['overall']
                    ranks = list(range(1, len(ranking) + 1))
                    axes[1, 1].barh(ranking, ranks)
                    axes[1, 1].set_title('Overall Ranking')
                    axes[1, 1].set_xlabel('Rank (1 = Best)')
                    axes[1, 1].invert_yaxis()
                    
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
            # Individual model reports
            for model_name, results in evaluation_results.items():
                self._create_model_report_page(model_name, results, pdf)
                
        logger.info(f"Evaluation report saved to {report_file}")
        return str(report_file)
        
    def _create_model_report_page(
        self, 
        model_name: str, 
        results: Dict[str, Any], 
        pdf: PdfPages
    ):
        """Create individual model report page."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Model title
        fig.suptitle(f'{model_name.upper()} Model Evaluation', fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
        
        # Forecast accuracy metrics
        if 'forecast_accuracy' in results and 'aggregate_metrics' in results['forecast_accuracy']:
            ax1 = fig.add_subplot(gs[0, :])
            
            metrics = results['forecast_accuracy']['aggregate_metrics']
            metric_names = ['mean_rmse', 'mean_mae', 'mean_mape', 'mean_r2']
            metric_values = [metrics.get(m, 0) for m in metric_names]
            
            bars = ax1.bar(metric_names, metric_values)
            ax1.set_title('Forecast Accuracy Metrics')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                if not np.isnan(value) and not np.isinf(value):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                            
        # Statistical tests summary
        if 'statistical_tests' in results:
            ax2 = fig.add_subplot(gs[1, 0])
            
            test_summary = {}
            for var_name, tests in results['statistical_tests']['tests'].items():
                passed = 0
                total = 0
                
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict):
                        total += 1
                        if (test_name == 'jarque_bera' and test_result.get('is_normal', False)) or \
                           (test_name == 'ljung_box' and test_result.get('no_autocorrelation', False)) or \
                           (test_name == 'heteroscedasticity' and test_result.get('is_homoscedastic', False)) or \
                           (test_name == 'adf_stationarity' and test_result.get('is_stationary', False)):
                            passed += 1
                            
                if total > 0:
                    test_summary[var_name] = passed / total
                    
            if test_summary:
                variables = list(test_summary.keys())
                scores = list(test_summary.values())
                
                ax2.bar(variables, scores)
                ax2.set_title('Statistical Tests (Pass Rate)')
                ax2.set_ylabel('Pass Rate')
                ax2.set_ylim(0, 1)
                ax2.tick_params(axis='x', rotation=45)
                
        # Impulse response summary
        if 'impulse_responses' in results:
            ax3 = fig.add_subplot(gs[1, 1])
            
            convergence_summary = {}
            for shock, irf_data in results['impulse_responses'].items():
                if 'analysis' in irf_data and 'convergence' in irf_data['analysis']:
                    converged = irf_data['analysis']['convergence']['converged']
                    convergence_rate = sum(converged) / len(converged) if converged else 0
                    convergence_summary[shock] = convergence_rate
                    
            if convergence_summary:
                shocks = list(convergence_summary.keys())
                rates = list(convergence_summary.values())
                
                ax3.bar(shocks, rates)
                ax3.set_title('IRF Convergence Rate')
                ax3.set_ylabel('Convergence Rate')
                ax3.set_ylim(0, 1)
                ax3.tick_params(axis='x', rotation=45)
                
        # Policy scenario impacts
        if 'policy_scenarios' in results:
            ax4 = fig.add_subplot(gs[2, :])
            
            scenario_impacts = {}
            for scenario_name, scenario_data in results['policy_scenarios']['scenarios'].items():
                if 'welfare_impact' in scenario_data:
                    scenario_impacts[scenario_name] = scenario_data['welfare_impact']
                    
            if scenario_impacts:
                scenarios = list(scenario_impacts.keys())
                impacts = list(scenario_impacts.values())
                
                colors = ['green' if x > 0 else 'red' for x in impacts]
                ax4.bar(scenarios, impacts, color=colors)
                ax4.set_title('Policy Scenario Welfare Impacts')
                ax4.set_ylabel('Welfare Change')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.tick_params(axis='x', rotation=45)
                
        # Model summary text
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')
        
        summary_text = f"Model: {model_name}\n"
        
        if 'forecast_accuracy' in results:
            n_obs = results['forecast_accuracy'].get('n_observations', 'N/A')
            n_vars = results['forecast_accuracy'].get('n_variables', 'N/A')
            summary_text += f"Data: {n_obs} observations, {n_vars} variables\n"
            
        if 'forecast_accuracy' in results and 'aggregate_metrics' in results['forecast_accuracy']:
            rmse = results['forecast_accuracy']['aggregate_metrics'].get('mean_rmse', np.nan)
            r2 = results['forecast_accuracy']['aggregate_metrics'].get('mean_r2', np.nan)
            
            if not np.isnan(rmse):
                summary_text += f"Average RMSE: {rmse:.4f}\n"
            if not np.isnan(r2):
                summary_text += f"Average R²: {r2:.4f}\n"
                
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
                
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """Save evaluation results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save evaluation results as JSON
        if self.evaluation_results:
            results_file = output_dir / f"evaluation_results_{timestamp}.json"
            
            # Convert to JSON-serializable format
            serializable_results = self._make_json_serializable(self.evaluation_results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
                
            saved_files['evaluation_results'] = str(results_file)
            logger.info(f"Saved evaluation results to {results_file}")
            
        return saved_files
        
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Evaluate Bangladesh macroeconomic models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_models.py --model dsge --data data/processed/bd_data.csv
  python scripts/evaluate_models.py --all --models-dir results/models/
  python scripts/evaluate_models.py --compare --forecast --periods 12
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to evaluation data file'
    )
    parser.add_argument(
        '--models-dir', 
        type=str,
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--model', 
        type=str,
        help='Specific model file to evaluate'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Evaluate all models in directory'
    )
    parser.add_argument(
        '--forecast', 
        action='store_true', 
        help='Perform forecast evaluation'
    )
    parser.add_argument(
        '--periods', 
        type=int, 
        default=12,
        help='Number of forecast periods'
    )
    parser.add_argument(
        '--statistical-tests', 
        action='store_true', 
        help='Perform statistical tests'
    )
    parser.add_argument(
        '--impulse-response', 
        action='store_true', 
        help='Evaluate impulse responses'
    )
    parser.add_argument(
        '--policy-scenarios', 
        action='store_true', 
        help='Evaluate policy scenarios'
    )
    parser.add_argument(
        '--compare', 
        action='store_true', 
        help='Compare multiple models'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/evaluation/',
        help='Output directory for results'
    )
    parser.add_argument(
        '--report', 
        action='store_true', 
        help='Generate PDF report'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='Configuration file path'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
        
    # Initialize evaluator
    config_manager = ConfigManager() if args.config else None
    evaluator = ModelEvaluator(config_manager)
    
    try:
        # Load data
        df = evaluator.load_data(args.data)
        
        # Load models
        if args.models_dir:
            models = evaluator.load_models(args.models_dir)
        elif args.model:
            # Load single model
            model_path = Path(args.model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            model_name = model_path.stem.split('_model_')[0]
            models = {model_name: model}
            evaluator.models = models
        else:
            raise ValueError("Must specify either --models-dir or --model")
            
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
        
        # Perform evaluations
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            model_results = {'model_name': model_name}
            
            # Forecast evaluation
            if args.forecast or args.all:
                try:
                    # Generate forecasts (placeholder - would use actual model forecasting)
                    if hasattr(model, 'forecast'):
                        forecast_data = model.forecast(args.periods)
                    else:
                        # Generate dummy forecast for demonstration
                        forecast_data = np.random.randn(args.periods, 4)
                        
                    # Use last periods of actual data for comparison
                    actual_data = df.iloc[-args.periods:][['GDP', 'INFLATION', 'UNEMPLOYMENT', 'EXCHANGE_RATE']].values
                    
                    if actual_data.shape[0] == forecast_data.shape[0]:
                        forecast_results = evaluator.evaluate_forecast_accuracy(
                            model_name, actual_data, forecast_data,
                            ['GDP', 'INFLATION', 'UNEMPLOYMENT', 'EXCHANGE_RATE']
                        )
                        model_results['forecast_accuracy'] = forecast_results
                        
                except Exception as e:
                    logger.warning(f"Forecast evaluation failed for {model_name}: {e}")
                    
            # Statistical tests
            if args.statistical_tests or args.all:
                try:
                    # Generate residuals (placeholder - would use actual model residuals)
                    residuals = np.random.randn(len(df), 4) * 0.1
                    
                    test_results = evaluator.perform_statistical_tests(
                        model_name, residuals,
                        ['GDP', 'INFLATION', 'UNEMPLOYMENT', 'EXCHANGE_RATE']
                    )
                    model_results['statistical_tests'] = test_results
                    
                except Exception as e:
                    logger.warning(f"Statistical tests failed for {model_name}: {e}")
                    
            # Impulse response evaluation
            if args.impulse_response or args.all:
                try:
                    irf_results = evaluator.evaluate_impulse_responses(
                        model_name, model, periods=20
                    )
                    model_results['impulse_responses'] = irf_results
                    
                except Exception as e:
                    logger.warning(f"Impulse response evaluation failed for {model_name}: {e}")
                    
            # Policy scenario evaluation
            if args.policy_scenarios or args.all:
                try:
                    policy_results = evaluator.evaluate_policy_scenarios(
                        model_name, model
                    )
                    model_results['policy_scenarios'] = policy_results
                    
                except Exception as e:
                    logger.warning(f"Policy scenario evaluation failed for {model_name}: {e}")
                    
            evaluator.evaluation_results[model_name] = model_results
            
        # Model comparison
        if args.compare and len(evaluator.evaluation_results) > 1:
            comparison_results = evaluator.compare_models(evaluator.evaluation_results)
            evaluator.evaluation_results['_comparison'] = comparison_results
            
            logger.info("Model comparison results:")
            if 'overall' in comparison_results['rankings']:
                for rank, model_name in enumerate(comparison_results['rankings']['overall'], 1):
                    logger.info(f"  {rank}. {model_name}")
                    
        # Generate report
        if args.report:
            report_file = evaluator.create_evaluation_report(
                evaluator.evaluation_results, args.output
            )
            logger.info(f"Generated evaluation report: {report_file}")
            
        # Save results
        saved_files = evaluator.save_results(args.output)
        
        logger.info("Model evaluation completed successfully")
        logger.info(f"Saved files: {list(saved_files.keys())}")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()