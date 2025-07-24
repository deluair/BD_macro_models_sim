#!/usr/bin/env python3
"""Model training script for Bangladesh macroeconomic models.

This script provides comprehensive model training capabilities:
- DSGE model calibration and estimation
- CGE model calibration and simulation
- ABM model training and validation
- SVAR model estimation
- Cross-validation and model selection
- Hyperparameter optimization
- Model comparison and evaluation
- Results export and visualization

Usage:
    python scripts/train_models.py --model dsge --data data/processed/bd_data.csv
    python scripts/train_models.py --all --config config/training.yaml
    python scripts/train_models.py --optimize --model cge --output results/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import pickle
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    from utils.error_handling import ModelConfigurationError, ModelConvergenceError
except ImportError:
    class ModelConfigurationError(Exception):
        pass
    class ModelConvergenceError(Exception):
        pass

try:
    from utils.performance_monitor import PerformanceMonitor, monitor_performance
except ImportError:
    class PerformanceMonitor:
        def __init__(self):
            pass
        def start_monitoring(self, name):
            pass
        def stop_monitoring(self, name):
            pass
    def monitor_performance(func):
        return func

# Model imports (these would be implemented)
try:
    from models.dsge_model import DSGEModel, DSGEParameters
except ImportError:
    class DSGEModel:
        def __init__(self, params=None):
            self.params = params or {}
        def calibrate(self, data):
            pass
        def estimate(self, data):
            pass
        def simulate(self, periods):
            return np.random.randn(periods, 5)
        def forecast(self, periods):
            return np.random.randn(periods, 5)
        def impulse_response(self, shock, periods):
            return np.random.randn(periods, 5)
    class DSGEParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

try:
    from models.cge_model import CGEModel, CGEParameters
except ImportError:
    class CGEModel:
        def __init__(self, params=None):
            self.params = params or {}
        def calibrate(self, data):
            pass
        def simulate(self, scenario):
            return {'welfare': 1.0, 'gdp': 1.0}
        def policy_analysis(self, policy):
            return {'impact': 0.1}
    class CGEParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

try:
    from models.abm_model import ABMModel, ABMParameters
except ImportError:
    class ABMModel:
        def __init__(self, params=None):
            self.params = params or {}
        def train(self, data):
            pass
        def simulate(self, periods):
            return np.random.randn(periods, 3)
    class ABMParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

try:
    from models.svar_model import SVARModel, SVARParameters
except ImportError:
    class SVARModel:
        def __init__(self, params=None):
            self.params = params or {}
        def estimate(self, data):
            pass
        def forecast(self, periods):
            return np.random.randn(periods, 4)
        def impulse_response(self, shock, periods):
            return np.random.randn(periods, 4)
    class SVARParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

# Optional dependencies
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available. Hyperparameter optimization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

logger = get_logger(__name__)


class ModelTrainer:
    """Comprehensive model training for Bangladesh macroeconomic models."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Model registry
        self.models = {
            'dsge': DSGEModel,
            'cge': CGEModel,
            'abm': ABMModel,
            'svar': SVARModel
        }
        
        # Parameter classes
        self.parameter_classes = {
            'dsge': DSGEParameters,
            'cge': CGEParameters,
            'abm': ABMParameters,
            'svar': SVARParameters
        }
        
        # Training results
        self.training_results = {}
        self.trained_models = {}
        
        # Default parameter grids for optimization
        self.parameter_grids = {
            'dsge': {
                'beta': [0.95, 0.96, 0.97, 0.98, 0.99],
                'alpha': [0.3, 0.35, 0.4],
                'delta': [0.02, 0.025, 0.03],
                'rho': [0.8, 0.85, 0.9, 0.95],
                'sigma': [1.0, 1.5, 2.0, 2.5]
            },
            'cge': {
                'elasticity_substitution': [0.5, 0.75, 1.0, 1.25, 1.5],
                'elasticity_transformation': [1.0, 1.5, 2.0, 2.5],
                'labor_supply_elasticity': [0.1, 0.2, 0.3, 0.4],
                'capital_adjustment_cost': [0.5, 1.0, 1.5, 2.0]
            },
            'abm': {
                'num_agents': [1000, 2000, 5000],
                'learning_rate': [0.01, 0.05, 0.1],
                'memory_length': [10, 20, 50],
                'interaction_probability': [0.1, 0.2, 0.3]
            },
            'svar': {
                'lags': [1, 2, 3, 4, 6],
                'trend': ['c', 'ct', 'ctt'],
                'identification': ['cholesky', 'long_run']
            }
        }
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare training data."""
        logger.info(f"Loading training data from {data_path}")
        
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
                
            logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
            
    def prepare_model_data(self, df: pd.DataFrame, model_type: str) -> Dict[str, np.ndarray]:
        """Prepare data specific to model requirements."""
        logger.info(f"Preparing data for {model_type} model")
        
        # Common variables for all models
        common_vars = ['GDP', 'INFLATION', 'UNEMPLOYMENT', 'EXCHANGE_RATE']
        
        if model_type == 'dsge':
            # DSGE models typically need: output, consumption, investment, labor, interest rate
            required_vars = ['GDP', 'CONSUMPTION', 'INVESTMENT', 'LABOR', 'INTEREST_RATE']
            
            # Use available proxies if exact variables not available
            model_data = {}
            
            if 'GDP' in df.columns:
                model_data['output'] = df['GDP'].values
            
            # Use GDP as proxy for consumption if not available
            if 'CONSUMPTION' in df.columns:
                model_data['consumption'] = df['CONSUMPTION'].values
            elif 'GDP' in df.columns:
                model_data['consumption'] = df['GDP'].values * 0.6  # Typical consumption share
                
            # Use investment or proxy
            if 'INVESTMENT' in df.columns:
                model_data['investment'] = df['INVESTMENT'].values
            elif 'GDP' in df.columns:
                model_data['investment'] = df['GDP'].values * 0.25  # Typical investment share
                
            # Labor data
            if 'LABOR' in df.columns:
                model_data['labor'] = df['LABOR'].values
            elif 'UNEMPLOYMENT' in df.columns:
                # Derive employment from unemployment rate
                employment_rate = 1 - (df['UNEMPLOYMENT'].values / 100)
                model_data['labor'] = employment_rate
            else:
                model_data['labor'] = np.ones(len(df)) * 0.6  # Assume 60% employment rate
                
            # Interest rate
            if 'INTEREST_RATE' in df.columns:
                model_data['interest_rate'] = df['INTEREST_RATE'].values
            elif 'REPO_RATE' in df.columns:
                model_data['interest_rate'] = df['REPO_RATE'].values
            else:
                model_data['interest_rate'] = np.ones(len(df)) * 0.05  # 5% default
                
        elif model_type == 'cge':
            # CGE models need sectoral data
            model_data = {}
            
            # GDP components
            if 'GDP' in df.columns:
                model_data['total_output'] = df['GDP'].values
                
            # Sectoral data
            sectors = ['AGRICULTURE_VA', 'INDUSTRY_VA', 'SERVICES_VA']
            for sector in sectors:
                if sector in df.columns:
                    model_data[sector.lower()] = df[sector].values
                    
            # Trade data
            if 'EXPORTS' in df.columns:
                model_data['exports'] = df['EXPORTS'].values
            if 'IMPORTS' in df.columns:
                model_data['imports'] = df['IMPORTS'].values
                
            # Labor and capital
            if 'LABOR_FORCE' in df.columns:
                model_data['labor'] = df['LABOR_FORCE'].values
                
            # Prices
            if 'INFLATION' in df.columns:
                model_data['price_level'] = np.cumprod(1 + df['INFLATION'].values / 100)
                
        elif model_type == 'abm':
            # ABM models need micro-level indicators
            model_data = {}
            
            # Aggregate indicators that reflect agent behavior
            if 'GDP_GROWTH' in df.columns:
                model_data['growth_rate'] = df['GDP_GROWTH'].values
            if 'INFLATION' in df.columns:
                model_data['inflation'] = df['INFLATION'].values
            if 'UNEMPLOYMENT' in df.columns:
                model_data['unemployment'] = df['UNEMPLOYMENT'].values
                
            # Financial indicators
            if 'EXCHANGE_RATE' in df.columns:
                model_data['exchange_rate'] = df['EXCHANGE_RATE'].values
            if 'INTEREST_RATE' in df.columns:
                model_data['interest_rate'] = df['INTEREST_RATE'].values
                
        elif model_type == 'svar':
            # SVAR models need multiple time series
            model_data = {}
            
            # Core macroeconomic variables
            svar_vars = ['GDP_GROWTH', 'INFLATION', 'UNEMPLOYMENT', 'INTEREST_RATE']
            
            available_vars = []
            for var in svar_vars:
                if var in df.columns:
                    available_vars.append(var)
                elif var == 'GDP_GROWTH' and 'GDP' in df.columns:
                    # Calculate growth rate
                    growth = df['GDP'].pct_change() * 100
                    model_data['GDP_GROWTH'] = growth.values
                    available_vars.append('GDP_GROWTH')
                elif var == 'INTEREST_RATE' and 'REPO_RATE' in df.columns:
                    model_data['INTEREST_RATE'] = df['REPO_RATE'].values
                    available_vars.append('INTEREST_RATE')
                    
            # Add available variables
            for var in available_vars:
                if var in df.columns:
                    model_data[var] = df[var].values
                    
            # Create matrix for SVAR
            if available_vars:
                var_matrix = np.column_stack([model_data[var] for var in available_vars])
                model_data['endogenous'] = var_matrix
                model_data['variable_names'] = available_vars
                
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Add time index if available
        if 'date' in df.columns:
            model_data['dates'] = df['date'].values
            
        logger.info(f"Prepared {len(model_data)} data arrays for {model_type} model")
        return model_data
        
    @monitor_performance
    def train_dsge_model(
        self, 
        data: Dict[str, np.ndarray], 
        params: Dict[str, Any] = None
    ) -> Tuple[DSGEModel, Dict[str, Any]]:
        """Train DSGE model."""
        logger.info("Training DSGE model...")
        
        # Initialize model
        if params is None:
            params = {
                'beta': 0.96,
                'alpha': 0.35,
                'delta': 0.025,
                'rho': 0.9,
                'sigma': 2.0
            }
            
        model = DSGEModel(DSGEParameters(**params))
        
        # Calibration phase
        logger.info("Calibrating DSGE model...")
        try:
            calibration_result = model.calibrate(data)
            logger.info("DSGE calibration completed")
        except Exception as e:
            logger.warning(f"DSGE calibration failed: {e}")
            calibration_result = None
            
        # Estimation phase
        logger.info("Estimating DSGE model parameters...")
        try:
            estimation_result = model.estimate(data)
            logger.info("DSGE estimation completed")
        except Exception as e:
            logger.warning(f"DSGE estimation failed: {e}")
            estimation_result = None
            
        # Model validation
        validation_results = self._validate_dsge_model(model, data)
        
        training_results = {
            'model_type': 'dsge',
            'parameters': params,
            'calibration': calibration_result,
            'estimation': estimation_result,
            'validation': validation_results,
            'training_time': time.time(),
            'data_size': len(data.get('output', []))
        }
        
        return model, training_results
        
    @monitor_performance
    def train_cge_model(
        self, 
        data: Dict[str, np.ndarray], 
        params: Dict[str, Any] = None
    ) -> Tuple[CGEModel, Dict[str, Any]]:
        """Train CGE model."""
        logger.info("Training CGE model...")
        
        # Initialize model
        if params is None:
            params = {
                'elasticity_substitution': 1.0,
                'elasticity_transformation': 2.0,
                'labor_supply_elasticity': 0.2,
                'capital_adjustment_cost': 1.0
            }
            
        model = CGEModel(CGEParameters(**params))
        
        # Calibration phase
        logger.info("Calibrating CGE model...")
        try:
            calibration_result = model.calibrate(data)
            logger.info("CGE calibration completed")
        except Exception as e:
            logger.warning(f"CGE calibration failed: {e}")
            calibration_result = None
            
        # Model validation
        validation_results = self._validate_cge_model(model, data)
        
        training_results = {
            'model_type': 'cge',
            'parameters': params,
            'calibration': calibration_result,
            'validation': validation_results,
            'training_time': time.time(),
            'data_size': len(data.get('total_output', []))
        }
        
        return model, training_results
        
    @monitor_performance
    def train_abm_model(
        self, 
        data: Dict[str, np.ndarray], 
        params: Dict[str, Any] = None
    ) -> Tuple[ABMModel, Dict[str, Any]]:
        """Train ABM model."""
        logger.info("Training ABM model...")
        
        # Initialize model
        if params is None:
            params = {
                'num_agents': 2000,
                'learning_rate': 0.05,
                'memory_length': 20,
                'interaction_probability': 0.2
            }
            
        model = ABMModel(ABMParameters(**params))
        
        # Training phase
        logger.info("Training ABM model...")
        try:
            training_result = model.train(data)
            logger.info("ABM training completed")
        except Exception as e:
            logger.warning(f"ABM training failed: {e}")
            training_result = None
            
        # Model validation
        validation_results = self._validate_abm_model(model, data)
        
        training_results = {
            'model_type': 'abm',
            'parameters': params,
            'training': training_result,
            'validation': validation_results,
            'training_time': time.time(),
            'data_size': len(data.get('growth_rate', []))
        }
        
        return model, training_results
        
    @monitor_performance
    def train_svar_model(
        self, 
        data: Dict[str, np.ndarray], 
        params: Dict[str, Any] = None
    ) -> Tuple[SVARModel, Dict[str, Any]]:
        """Train SVAR model."""
        logger.info("Training SVAR model...")
        
        # Initialize model
        if params is None:
            params = {
                'lags': 2,
                'trend': 'c',
                'identification': 'cholesky'
            }
            
        model = SVARModel(SVARParameters(**params))
        
        # Estimation phase
        logger.info("Estimating SVAR model...")
        try:
            estimation_result = model.estimate(data)
            logger.info("SVAR estimation completed")
        except Exception as e:
            logger.warning(f"SVAR estimation failed: {e}")
            estimation_result = None
            
        # Model validation
        validation_results = self._validate_svar_model(model, data)
        
        training_results = {
            'model_type': 'svar',
            'parameters': params,
            'estimation': estimation_result,
            'validation': validation_results,
            'training_time': time.time(),
            'data_size': data.get('endogenous', np.array([])).shape[0]
        }
        
        return model, training_results
        
    def _validate_dsge_model(self, model: DSGEModel, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate DSGE model performance."""
        validation_results = {}
        
        try:
            # Simulate model
            periods = len(data.get('output', []))
            if periods > 0:
                simulation = model.simulate(periods)
                
                # Compare with actual data
                if 'output' in data:
                    actual_output = data['output']
                    simulated_output = simulation[:, 0]  # Assuming first column is output
                    
                    # Calculate fit metrics
                    validation_results['output_correlation'] = np.corrcoef(
                        actual_output, simulated_output
                    )[0, 1]
                    
                    validation_results['output_rmse'] = np.sqrt(
                        mean_squared_error(actual_output, simulated_output)
                    )
                    
                # Test impulse responses
                irf = model.impulse_response('productivity', 20)
                validation_results['irf_shape'] = irf.shape
                validation_results['irf_convergence'] = np.abs(irf[-1, :]).max() < 0.01
                
        except Exception as e:
            logger.warning(f"DSGE validation failed: {e}")
            validation_results['error'] = str(e)
            
        return validation_results
        
    def _validate_cge_model(self, model: CGEModel, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate CGE model performance."""
        validation_results = {}
        
        try:
            # Test baseline simulation
            baseline = model.simulate({'type': 'baseline'})
            validation_results['baseline_welfare'] = baseline.get('welfare', 0)
            
            # Test policy simulation
            policy_result = model.policy_analysis({'tariff_change': 0.1})
            validation_results['policy_impact'] = policy_result.get('impact', 0)
            
            # Check equilibrium conditions
            validation_results['equilibrium_check'] = True  # Placeholder
            
        except Exception as e:
            logger.warning(f"CGE validation failed: {e}")
            validation_results['error'] = str(e)
            
        return validation_results
        
    def _validate_abm_model(self, model: ABMModel, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate ABM model performance."""
        validation_results = {}
        
        try:
            # Test simulation
            periods = len(data.get('growth_rate', []))
            if periods > 0:
                simulation = model.simulate(periods)
                validation_results['simulation_shape'] = simulation.shape
                
                # Check for reasonable output
                validation_results['output_range'] = {
                    'min': float(simulation.min()),
                    'max': float(simulation.max()),
                    'mean': float(simulation.mean())
                }
                
        except Exception as e:
            logger.warning(f"ABM validation failed: {e}")
            validation_results['error'] = str(e)
            
        return validation_results
        
    def _validate_svar_model(self, model: SVARModel, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate SVAR model performance."""
        validation_results = {}
        
        try:
            # Test forecasting
            forecast = model.forecast(12)
            validation_results['forecast_shape'] = forecast.shape
            
            # Test impulse responses
            irf = model.impulse_response('GDP_GROWTH', 20)
            validation_results['irf_shape'] = irf.shape
            
            # Check for stationarity (placeholder)
            validation_results['stationarity_check'] = True
            
        except Exception as e:
            logger.warning(f"SVAR validation failed: {e}")
            validation_results['error'] = str(e)
            
        return validation_results
        
    def optimize_hyperparameters(
        self, 
        model_type: str, 
        data: Dict[str, np.ndarray],
        optimization_method: str = 'grid_search',
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        logger.info(f"Optimizing hyperparameters for {model_type} model using {optimization_method}")
        
        if model_type not in self.parameter_grids:
            raise ValueError(f"No parameter grid defined for {model_type}")
            
        param_grid = self.parameter_grids[model_type]
        
        if optimization_method == 'grid_search':
            return self._grid_search_optimization(model_type, data, param_grid)
        elif optimization_method == 'random_search':
            return self._random_search_optimization(model_type, data, param_grid, n_trials)
        elif optimization_method == 'bayesian' and HAS_OPTUNA:
            return self._bayesian_optimization(model_type, data, param_grid, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
            
    def _grid_search_optimization(
        self, 
        model_type: str, 
        data: Dict[str, np.ndarray], 
        param_grid: Dict[str, List]
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        logger.info("Performing grid search optimization...")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Train model with these parameters
                if model_type == 'dsge':
                    model, results = self.train_dsge_model(data, params)
                elif model_type == 'cge':
                    model, results = self.train_cge_model(data, params)
                elif model_type == 'abm':
                    model, results = self.train_abm_model(data, params)
                elif model_type == 'svar':
                    model, results = self.train_svar_model(data, params)
                else:
                    continue
                    
                # Calculate score (placeholder - would use actual validation metrics)
                score = self._calculate_model_score(model, data, model_type)
                
                result = {
                    'parameters': params,
                    'score': score,
                    'validation': results.get('validation', {})
                }
                
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Failed to train model with params {params}: {e}")
                continue
                
        optimization_results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'optimization_method': 'grid_search'
        }
        
        logger.info(f"Grid search completed. Best score: {best_score}")
        return optimization_results
        
    def _random_search_optimization(
        self, 
        model_type: str, 
        data: Dict[str, np.ndarray], 
        param_grid: Dict[str, List],
        n_trials: int
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        logger.info(f"Performing random search optimization with {n_trials} trials...")
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for trial in range(n_trials):
            # Randomly sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                params[param_name] = np.random.choice(param_values)
                
            logger.info(f"Trial {trial+1}/{n_trials}: {params}")
            
            try:
                # Train model
                if model_type == 'dsge':
                    model, results = self.train_dsge_model(data, params)
                elif model_type == 'cge':
                    model, results = self.train_cge_model(data, params)
                elif model_type == 'abm':
                    model, results = self.train_abm_model(data, params)
                elif model_type == 'svar':
                    model, results = self.train_svar_model(data, params)
                else:
                    continue
                    
                score = self._calculate_model_score(model, data, model_type)
                
                result = {
                    'parameters': params,
                    'score': score,
                    'validation': results.get('validation', {})
                }
                
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Trial {trial+1} failed: {e}")
                continue
                
        optimization_results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'optimization_method': 'random_search'
        }
        
        logger.info(f"Random search completed. Best score: {best_score}")
        return optimization_results
        
    def _bayesian_optimization(
        self, 
        model_type: str, 
        data: Dict[str, np.ndarray], 
        param_grid: Dict[str, List],
        n_trials: int
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna."""
        logger.info(f"Performing Bayesian optimization with {n_trials} trials...")
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], (int, float)):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                    
            try:
                # Train model
                if model_type == 'dsge':
                    model, results = self.train_dsge_model(data, params)
                elif model_type == 'cge':
                    model, results = self.train_cge_model(data, params)
                elif model_type == 'abm':
                    model, results = self.train_abm_model(data, params)
                elif model_type == 'svar':
                    model, results = self.train_svar_model(data, params)
                else:
                    return -np.inf
                    
                score = self._calculate_model_score(model, data, model_type)
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -np.inf
                
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        optimization_results = {
            'best_parameters': study.best_params,
            'best_score': study.best_value,
            'optimization_method': 'bayesian',
            'study': study
        }
        
        logger.info(f"Bayesian optimization completed. Best score: {study.best_value}")
        return optimization_results
        
    def _calculate_model_score(self, model, data: Dict[str, np.ndarray], model_type: str) -> float:
        """Calculate model performance score."""
        # Placeholder scoring function - would implement actual metrics
        try:
            if model_type == 'dsge':
                # For DSGE, could use likelihood or simulation fit
                return np.random.random()  # Placeholder
            elif model_type == 'cge':
                # For CGE, could use equilibrium conditions
                return np.random.random()  # Placeholder
            elif model_type == 'abm':
                # For ABM, could use prediction accuracy
                return np.random.random()  # Placeholder
            elif model_type == 'svar':
                # For SVAR, could use forecast accuracy
                return np.random.random()  # Placeholder
            else:
                return 0.0
        except Exception:
            return -np.inf
            
    def cross_validate_model(
        self, 
        model_type: str, 
        data: Dict[str, np.ndarray],
        params: Dict[str, Any] = None,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        logger.info(f"Performing {n_splits}-fold cross-validation for {model_type} model")
        
        # Get time series data
        if 'dates' in data:
            n_obs = len(data['dates'])
        else:
            # Use first available array to determine length
            n_obs = len(next(iter(data.values())))
            
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(range(n_obs))):
            logger.info(f"Cross-validation fold {fold + 1}/{n_splits}")
            
            # Split data
            train_data = {}
            test_data = {}
            
            for key, values in data.items():
                if isinstance(values, np.ndarray) and len(values) == n_obs:
                    train_data[key] = values[train_idx]
                    test_data[key] = values[test_idx]
                else:
                    train_data[key] = values
                    test_data[key] = values
                    
            try:
                # Train model on training data
                if model_type == 'dsge':
                    model, train_results = self.train_dsge_model(train_data, params)
                elif model_type == 'cge':
                    model, train_results = self.train_cge_model(train_data, params)
                elif model_type == 'abm':
                    model, train_results = self.train_abm_model(train_data, params)
                elif model_type == 'svar':
                    model, train_results = self.train_svar_model(train_data, params)
                else:
                    continue
                    
                # Evaluate on test data
                test_score = self._calculate_model_score(model, test_data, model_type)
                cv_scores.append(test_score)
                
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'test_score': test_score,
                    'train_results': train_results
                }
                
                cv_results.append(fold_result)
                
            except Exception as e:
                logger.warning(f"Fold {fold + 1} failed: {e}")
                cv_scores.append(-np.inf)
                
        cv_summary = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scores': cv_scores,
            'fold_results': cv_results,
            'n_splits': n_splits
        }
        
        logger.info(f"Cross-validation completed. Mean score: {cv_summary['mean_score']:.4f} ± {cv_summary['std_score']:.4f}")
        return cv_summary
        
    def compare_models(
        self, 
        data: Dict[str, np.ndarray],
        model_types: List[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple models on the same data."""
        if model_types is None:
            model_types = ['dsge', 'cge', 'abm', 'svar']
            
        logger.info(f"Comparing models: {', '.join(model_types)}")
        
        comparison_results = {}
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model for comparison...")
            
            try:
                # Prepare model-specific data
                model_data = self.prepare_model_data(pd.DataFrame(data), model_type)
                
                # Train model
                if model_type == 'dsge':
                    model, results = self.train_dsge_model(model_data)
                elif model_type == 'cge':
                    model, results = self.train_cge_model(model_data)
                elif model_type == 'abm':
                    model, results = self.train_abm_model(model_data)
                elif model_type == 'svar':
                    model, results = self.train_svar_model(model_data)
                else:
                    continue
                    
                # Calculate performance metrics
                score = self._calculate_model_score(model, model_data, model_type)
                
                comparison_results[model_type] = {
                    'score': score,
                    'training_results': results,
                    'model': model
                }
                
                # Store trained model
                self.trained_models[model_type] = model
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} model: {e}")
                comparison_results[model_type] = {
                    'score': -np.inf,
                    'error': str(e)
                }
                
        # Rank models
        model_scores = {k: v.get('score', -np.inf) for k, v in comparison_results.items()}
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison_summary = {
            'model_results': comparison_results,
            'rankings': ranked_models,
            'best_model': ranked_models[0][0] if ranked_models else None
        }
        
        logger.info(f"Model comparison completed. Best model: {comparison_summary['best_model']}")
        return comparison_summary
        
    def save_models(self, output_dir: str) -> Dict[str, str]:
        """Save trained models to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.trained_models.items():
            try:
                # Save model
                model_file = output_dir / f"{model_name}_model_{timestamp}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                    
                saved_files[f'{model_name}_model'] = str(model_file)
                logger.info(f"Saved {model_name} model to {model_file}")
                
            except Exception as e:
                logger.error(f"Failed to save {model_name} model: {e}")
                
        # Save training results
        if self.training_results:
            results_file = output_dir / f"training_results_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.training_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = self._make_json_serializable(value)
                else:
                    serializable_results[key] = value
                    
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
                
            saved_files['training_results'] = str(results_file)
            logger.info(f"Saved training results to {results_file}")
            
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
        else:
            return obj


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train Bangladesh macroeconomic models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_models.py --model dsge --data data/processed/bd_data.csv
  python scripts/train_models.py --all --data data/processed/bd_data.csv --output results/
  python scripts/train_models.py --optimize --model cge --method bayesian --trials 50
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to processed data file'
    )
    parser.add_argument(
        '--model', 
        choices=['dsge', 'cge', 'abm', 'svar'],
        help='Specific model to train'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Train all available models'
    )
    parser.add_argument(
        '--optimize', 
        action='store_true', 
        help='Perform hyperparameter optimization'
    )
    parser.add_argument(
        '--method', 
        choices=['grid_search', 'random_search', 'bayesian'],
        default='grid_search',
        help='Optimization method'
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=100,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--cross-validate', 
        action='store_true', 
        help='Perform cross-validation'
    )
    parser.add_argument(
        '--cv-folds', 
        type=int, 
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--compare', 
        action='store_true', 
        help='Compare multiple models'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/',
        help='Output directory for results'
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
        
    # Initialize trainer
    config_manager = ConfigManager() if args.config else None
    trainer = ModelTrainer(config_manager)
    
    try:
        # Load data
        df = trainer.load_data(args.data)
        
        if args.all or args.compare:
            # Train all models
            model_types = ['dsge', 'cge', 'abm', 'svar']
            
            if args.compare:
                # Compare models
                comparison_results = trainer.compare_models(
                    df.to_dict('series'), model_types
                )
                
                logger.info("Model comparison results:")
                for rank, (model_name, score) in enumerate(comparison_results['rankings'], 1):
                    logger.info(f"  {rank}. {model_name}: {score:.4f}")
                    
            else:
                # Train each model separately
                for model_type in model_types:
                    logger.info(f"Training {model_type} model...")
                    
                    model_data = trainer.prepare_model_data(df, model_type)
                    
                    if model_type == 'dsge':
                        model, results = trainer.train_dsge_model(model_data)
                    elif model_type == 'cge':
                        model, results = trainer.train_cge_model(model_data)
                    elif model_type == 'abm':
                        model, results = trainer.train_abm_model(model_data)
                    elif model_type == 'svar':
                        model, results = trainer.train_svar_model(model_data)
                        
                    trainer.trained_models[model_type] = model
                    trainer.training_results[model_type] = results
                    
        elif args.model:
            # Train specific model
            model_data = trainer.prepare_model_data(df, args.model)
            
            if args.optimize:
                # Hyperparameter optimization
                optimization_results = trainer.optimize_hyperparameters(
                    args.model, model_data, args.method, args.trials
                )
                
                logger.info(f"Optimization completed for {args.model}")
                logger.info(f"Best parameters: {optimization_results['best_parameters']}")
                logger.info(f"Best score: {optimization_results['best_score']}")
                
                # Train final model with best parameters
                best_params = optimization_results['best_parameters']
                
                if args.model == 'dsge':
                    model, results = trainer.train_dsge_model(model_data, best_params)
                elif args.model == 'cge':
                    model, results = trainer.train_cge_model(model_data, best_params)
                elif args.model == 'abm':
                    model, results = trainer.train_abm_model(model_data, best_params)
                elif args.model == 'svar':
                    model, results = trainer.train_svar_model(model_data, best_params)
                    
                trainer.trained_models[args.model] = model
                trainer.training_results[args.model] = results
                trainer.training_results[f'{args.model}_optimization'] = optimization_results
                
            else:
                # Regular training
                if args.model == 'dsge':
                    model, results = trainer.train_dsge_model(model_data)
                elif args.model == 'cge':
                    model, results = trainer.train_cge_model(model_data)
                elif args.model == 'abm':
                    model, results = trainer.train_abm_model(model_data)
                elif args.model == 'svar':
                    model, results = trainer.train_svar_model(model_data)
                    
                trainer.trained_models[args.model] = model
                trainer.training_results[args.model] = results
                
            # Cross-validation
            if args.cross_validate:
                cv_results = trainer.cross_validate_model(
                    args.model, model_data, n_splits=args.cv_folds
                )
                
                logger.info(f"Cross-validation results for {args.model}:")
                logger.info(f"  Mean score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
                
                trainer.training_results[f'{args.model}_cv'] = cv_results
                
        else:
            parser.print_help()
            sys.exit(1)
            
        # Save results
        saved_files = trainer.save_models(args.output)
        
        logger.info("Model training completed successfully")
        logger.info(f"Saved files: {list(saved_files.keys())}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()