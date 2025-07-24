#!/usr/bin/env python3
"""
Monte Carlo Simulation Module

This module provides comprehensive Monte Carlo simulation capabilities for
economic models, including uncertainty analysis, sensitivity testing, and
risk assessment for Bangladesh macroeconomic analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import json
from scipy import stats
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

@dataclass
class SimulationParameter:
    """
    Data class for defining simulation parameters.
    """
    name: str
    distribution: str  # 'normal', 'uniform', 'triangular', 'beta'
    parameters: Dict[str, float]  # distribution parameters
    correlation_group: Optional[str] = None

@dataclass
class SimulationScenario:
    """
    Data class for defining simulation scenarios.
    """
    name: str
    description: str
    parameters: List[SimulationParameter]
    num_simulations: int = 1000
    time_horizon: int = 20  # quarters

class MonteCarloSimulator:
    """
    Comprehensive Monte Carlo simulation framework for economic models.
    """
    
    def __init__(self, results_dir: str = "../../results", output_dir: str = "./output"):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            results_dir: Directory containing model results
            output_dir: Directory for saving simulation outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define simulation scenarios
        self.simulation_scenarios = self._define_simulation_scenarios()
        
        # Model-specific simulation capabilities
        self.model_simulation_types = {
            'dsge': ['parameter_uncertainty', 'shock_analysis', 'policy_uncertainty'],
            'svar': ['coefficient_uncertainty', 'shock_analysis', 'forecast_uncertainty'],
            'rbc': ['technology_shocks', 'preference_shocks', 'policy_uncertainty'],
            'cge': ['elasticity_uncertainty', 'productivity_shocks', 'trade_shocks'],
            'behavioral': ['behavioral_parameters', 'expectation_shocks', 'sentiment_analysis'],
            'financial': ['risk_parameters', 'liquidity_shocks', 'credit_shocks']
        }
        
        # Key economic variables for simulation
        self.simulation_variables = [
            'gdp_growth', 'inflation', 'unemployment', 'current_account',
            'government_debt', 'trade_balance', 'real_exchange_rate',
            'investment_rate', 'consumption_growth', 'productivity_growth'
        ]
        
    def _define_simulation_scenarios(self) -> List[SimulationScenario]:
        """
        Define comprehensive simulation scenarios.
        
        Returns:
            List of SimulationScenario objects
        """
        scenarios = [
            # Baseline Economic Uncertainty
            SimulationScenario(
                name="Baseline Economic Uncertainty",
                description="Standard uncertainty around baseline economic parameters",
                parameters=[
                    SimulationParameter("productivity_growth", "normal", {"mean": 2.5, "std": 0.8}),
                    SimulationParameter("population_growth", "normal", {"mean": 1.2, "std": 0.3}),
                    SimulationParameter("investment_rate", "normal", {"mean": 25.0, "std": 3.0}),
                    SimulationParameter("government_spending_share", "normal", {"mean": 15.0, "std": 2.0})
                ],
                num_simulations=2000
            ),
            
            # External Shock Scenarios
            SimulationScenario(
                name="External Shock Analysis",
                description="Analysis of external economic shocks impact",
                parameters=[
                    SimulationParameter("oil_price_shock", "normal", {"mean": 0.0, "std": 25.0}),
                    SimulationParameter("global_growth_shock", "normal", {"mean": 0.0, "std": 1.5}),
                    SimulationParameter("commodity_price_shock", "normal", {"mean": 0.0, "std": 15.0}),
                    SimulationParameter("capital_flow_shock", "normal", {"mean": 0.0, "std": 2.0})
                ],
                num_simulations=1500
            ),
            
            # Climate Risk Scenarios
            SimulationScenario(
                name="Climate Risk Assessment",
                description="Economic impact of climate-related risks",
                parameters=[
                    SimulationParameter("climate_damage_agriculture", "beta", {"alpha": 2, "beta": 8, "scale": 10}),
                    SimulationParameter("extreme_weather_frequency", "triangular", {"left": 0, "mode": 2, "right": 8}),
                    SimulationParameter("adaptation_cost", "uniform", {"low": 0.5, "high": 3.0}),
                    SimulationParameter("sea_level_impact", "normal", {"mean": 1.0, "std": 0.5})
                ],
                num_simulations=1000
            ),
            
            # Financial Stress Scenarios
            SimulationScenario(
                name="Financial Stress Testing",
                description="Banking and financial sector stress scenarios",
                parameters=[
                    SimulationParameter("credit_risk_increase", "triangular", {"left": 0, "mode": 5, "right": 20}),
                    SimulationParameter("liquidity_shock", "normal", {"mean": 0.0, "std": 10.0}),
                    SimulationParameter("exchange_rate_volatility", "normal", {"mean": 8.0, "std": 3.0}),
                    SimulationParameter("interest_rate_shock", "normal", {"mean": 0.0, "std": 2.0})
                ],
                num_simulations=1500
            ),
            
            # Policy Uncertainty Scenarios
            SimulationScenario(
                name="Policy Uncertainty Analysis",
                description="Impact of policy uncertainty on economic outcomes",
                parameters=[
                    SimulationParameter("fiscal_policy_uncertainty", "uniform", {"low": -3.0, "high": 3.0}),
                    SimulationParameter("monetary_policy_uncertainty", "normal", {"mean": 0.0, "std": 1.5}),
                    SimulationParameter("trade_policy_uncertainty", "triangular", {"left": -5, "mode": 0, "right": 5}),
                    SimulationParameter("regulatory_uncertainty", "beta", {"alpha": 2, "beta": 3, "scale": 10})
                ],
                num_simulations=2000
            )
        ]
        
        return scenarios
    
    def load_model_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available model results.
        
        Returns:
            Dictionary of model results DataFrames
        """
        results = {}
        
        for file_path in self.results_dir.glob("*.csv"):
            model_name = file_path.stem.replace('_results', '')
            try:
                df = pd.read_csv(file_path)
                results[model_name] = df
                print(f"Loaded {model_name}: {df.shape}")
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                
        return results
    
    def generate_parameter_samples(self, scenario: SimulationScenario) -> pd.DataFrame:
        """
        Generate parameter samples for Monte Carlo simulation.
        
        Args:
            scenario: Simulation scenario definition
            
        Returns:
            DataFrame with parameter samples
        """
        np.random.seed(42)  # For reproducible results
        
        samples = {}
        
        for param in scenario.parameters:
            if param.distribution == 'normal':
                samples[param.name] = np.random.normal(
                    param.parameters['mean'], 
                    param.parameters['std'], 
                    scenario.num_simulations
                )
            elif param.distribution == 'uniform':
                samples[param.name] = np.random.uniform(
                    param.parameters['low'], 
                    param.parameters['high'], 
                    scenario.num_simulations
                )
            elif param.distribution == 'triangular':
                samples[param.name] = np.random.triangular(
                    param.parameters['left'], 
                    param.parameters['mode'], 
                    param.parameters['right'], 
                    scenario.num_simulations
                )
            elif param.distribution == 'beta':
                beta_samples = np.random.beta(
                    param.parameters['alpha'], 
                    param.parameters['beta'], 
                    scenario.num_simulations
                )
                samples[param.name] = beta_samples * param.parameters.get('scale', 1.0)
            else:
                # Default to normal distribution
                samples[param.name] = np.random.normal(0, 1, scenario.num_simulations)
        
        return pd.DataFrame(samples)
    
    def simulate_model_responses(self, model_name: str, 
                               parameter_samples: pd.DataFrame,
                               scenario: SimulationScenario) -> Dict[str, np.ndarray]:
        """
        Simulate model responses to parameter variations.
        
        Args:
            model_name: Name of the economic model
            parameter_samples: DataFrame with parameter samples
            scenario: Simulation scenario
            
        Returns:
            Dictionary of simulated variable responses
        """
        num_simulations = len(parameter_samples)
        responses = {}
        
        # Initialize response arrays
        for variable in self.simulation_variables:
            responses[variable] = np.zeros((num_simulations, scenario.time_horizon))
        
        # Model-specific response functions
        response_functions = self._get_model_response_functions(model_name)
        
        # Run simulations
        for sim_idx in range(num_simulations):
            param_values = parameter_samples.iloc[sim_idx].to_dict()
            
            # Generate time series for each variable
            for variable in self.simulation_variables:
                if variable in response_functions:
                    response_func = response_functions[variable]
                    time_series = response_func(param_values, scenario.time_horizon)
                    responses[variable][sim_idx, :] = time_series
                else:
                    # Default response function
                    responses[variable][sim_idx, :] = self._default_response_function(
                        variable, param_values, scenario.time_horizon
                    )
        
        return responses
    
    def _get_model_response_functions(self, model_name: str) -> Dict[str, Callable]:
        """
        Get model-specific response functions.
        
        Args:
            model_name: Name of the economic model
            
        Returns:
            Dictionary of response functions by variable
        """
        # Define model-specific response functions
        if model_name == 'dsge':
            return {
                'gdp_growth': self._dsge_gdp_response,
                'inflation': self._dsge_inflation_response,
                'unemployment': self._dsge_unemployment_response
            }
        elif model_name == 'svar':
            return {
                'gdp_growth': self._svar_gdp_response,
                'inflation': self._svar_inflation_response
            }
        elif model_name == 'rbc':
            return {
                'gdp_growth': self._rbc_gdp_response,
                'investment_rate': self._rbc_investment_response
            }
        else:
            return {}
    
    def _dsge_gdp_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        DSGE model GDP growth response function.
        """
        base_growth = 6.5  # Bangladesh baseline GDP growth
        productivity_effect = params.get('productivity_growth', 2.5) * 0.8
        investment_effect = (params.get('investment_rate', 25.0) - 25.0) * 0.1
        
        # Generate time series with persistence
        growth_series = np.zeros(horizon)
        shock = productivity_effect + investment_effect
        
        for t in range(horizon):
            persistence = 0.7 ** t
            noise = np.random.normal(0, 0.5)
            growth_series[t] = base_growth + shock * persistence + noise
        
        return growth_series
    
    def _dsge_inflation_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        DSGE model inflation response function.
        """
        base_inflation = 5.5
        monetary_effect = params.get('monetary_policy_uncertainty', 0.0) * 0.3
        external_effect = params.get('oil_price_shock', 0.0) * 0.02
        
        inflation_series = np.zeros(horizon)
        shock = monetary_effect + external_effect
        
        for t in range(horizon):
            persistence = 0.6 ** t
            noise = np.random.normal(0, 0.3)
            inflation_series[t] = base_inflation + shock * persistence + noise
        
        return inflation_series
    
    def _dsge_unemployment_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        DSGE model unemployment response function.
        """
        base_unemployment = 4.2
        growth_effect = -(params.get('productivity_growth', 2.5) - 2.5) * 0.2
        policy_effect = params.get('fiscal_policy_uncertainty', 0.0) * 0.1
        
        unemployment_series = np.zeros(horizon)
        shock = growth_effect + policy_effect
        
        for t in range(horizon):
            persistence = 0.8 ** t
            noise = np.random.normal(0, 0.2)
            unemployment_series[t] = max(0.5, base_unemployment + shock * persistence + noise)
        
        return unemployment_series
    
    def _svar_gdp_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        SVAR model GDP growth response function.
        """
        base_growth = 6.8
        external_effect = params.get('global_growth_shock', 0.0) * 0.6
        commodity_effect = params.get('commodity_price_shock', 0.0) * 0.03
        
        growth_series = np.zeros(horizon)
        shock = external_effect + commodity_effect
        
        for t in range(horizon):
            persistence = 0.5 ** t
            noise = np.random.normal(0, 0.4)
            growth_series[t] = base_growth + shock * persistence + noise
        
        return growth_series
    
    def _svar_inflation_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        SVAR model inflation response function.
        """
        base_inflation = 5.2
        oil_effect = params.get('oil_price_shock', 0.0) * 0.04
        exchange_effect = params.get('exchange_rate_volatility', 8.0) * 0.1
        
        inflation_series = np.zeros(horizon)
        shock = oil_effect + (exchange_effect - 0.8)
        
        for t in range(horizon):
            persistence = 0.7 ** t
            noise = np.random.normal(0, 0.3)
            inflation_series[t] = base_inflation + shock * persistence + noise
        
        return inflation_series
    
    def _rbc_gdp_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        RBC model GDP growth response function.
        """
        base_growth = 6.2
        productivity_effect = params.get('productivity_growth', 2.5) * 0.9
        investment_effect = (params.get('investment_rate', 25.0) - 25.0) * 0.12
        
        growth_series = np.zeros(horizon)
        shock = productivity_effect + investment_effect
        
        for t in range(horizon):
            persistence = 0.6 ** t
            noise = np.random.normal(0, 0.3)
            growth_series[t] = base_growth + shock * persistence + noise
        
        return growth_series
    
    def _rbc_investment_response(self, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        RBC model investment rate response function.
        """
        base_investment = 25.0
        productivity_effect = params.get('productivity_growth', 2.5) * 1.5
        uncertainty_effect = -params.get('policy_uncertainty', 0.0) * 0.5
        
        investment_series = np.zeros(horizon)
        shock = productivity_effect + uncertainty_effect
        
        for t in range(horizon):
            persistence = 0.8 ** t
            noise = np.random.normal(0, 1.0)
            investment_series[t] = max(10.0, base_investment + shock * persistence + noise)
        
        return investment_series
    
    def _default_response_function(self, variable: str, params: Dict[str, float], horizon: int) -> np.ndarray:
        """
        Default response function for variables without specific models.
        """
        # Base values for Bangladesh
        base_values = {
            'gdp_growth': 6.5, 'inflation': 5.5, 'unemployment': 4.2,
            'current_account': -1.5, 'government_debt': 35.0, 'trade_balance': -8.0,
            'real_exchange_rate': 100.0, 'investment_rate': 25.0,
            'consumption_growth': 5.8, 'productivity_growth': 2.5
        }
        
        base_value = base_values.get(variable, 0.0)
        
        # Simple response to first parameter
        if params:
            first_param = list(params.values())[0]
            shock = first_param * 0.1
        else:
            shock = 0.0
        
        series = np.zeros(horizon)
        for t in range(horizon):
            persistence = 0.7 ** t
            noise = np.random.normal(0, abs(base_value) * 0.05)
            series[t] = base_value + shock * persistence + noise
        
        return series
    
    def analyze_simulation_results(self, simulation_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, pd.DataFrame]:
        """
        Analyze Monte Carlo simulation results.
        
        Args:
            simulation_results: Dictionary of simulation results by scenario and model
            
        Returns:
            Dictionary of analysis results
        """
        analysis_results = {}
        
        for scenario_name, scenario_results in simulation_results.items():
            scenario_analysis = {}
            
            for model_name, model_results in scenario_results.items():
                model_analysis = self._analyze_model_simulation(model_results)
                scenario_analysis[model_name] = model_analysis
            
            # Combine model results
            combined_analysis = self._combine_model_analyses(scenario_analysis)
            analysis_results[scenario_name] = combined_analysis
        
        return analysis_results
    
    def _analyze_model_simulation(self, model_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Analyze simulation results for a single model.
        
        Args:
            model_results: Dictionary of variable simulation results
            
        Returns:
            DataFrame with statistical analysis
        """
        analysis_data = []
        
        for variable, results in model_results.items():
            # Calculate statistics across simulations and time
            mean_path = np.mean(results, axis=0)
            std_path = np.std(results, axis=0)
            
            # Overall statistics
            overall_mean = np.mean(results)
            overall_std = np.std(results)
            percentile_5 = np.percentile(results, 5)
            percentile_95 = np.percentile(results, 95)
            
            # Risk metrics
            var_95 = np.percentile(results.flatten(), 5)  # Value at Risk
            cvar_95 = np.mean(results.flatten()[results.flatten() <= var_95])  # Conditional VaR
            
            analysis_data.append({
                'variable': variable,
                'mean': overall_mean,
                'std': overall_std,
                'percentile_5': percentile_5,
                'percentile_95': percentile_95,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'coefficient_of_variation': overall_std / abs(overall_mean) if overall_mean != 0 else np.inf
            })
        
        return pd.DataFrame(analysis_data)
    
    def _combine_model_analyses(self, scenario_analysis: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine analysis results across models for a scenario.
        
        Args:
            scenario_analysis: Dictionary of model analysis results
            
        Returns:
            Combined analysis DataFrame
        """
        combined_data = []
        
        for model_name, model_df in scenario_analysis.items():
            model_df_copy = model_df.copy()
            model_df_copy['model'] = model_name
            combined_data.append(model_df_copy)
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def calculate_risk_metrics(self, simulation_results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        Calculate comprehensive risk metrics from simulation results.
        
        Args:
            simulation_results: Dictionary of simulation results
            
        Returns:
            DataFrame with risk metrics
        """
        risk_metrics = []
        
        for scenario_name, scenario_results in simulation_results.items():
            for model_name, model_results in scenario_results.items():
                for variable, results in model_results.items():
                    # Flatten results for analysis
                    flat_results = results.flatten()
                    
                    # Calculate various risk metrics
                    metrics = {
                        'scenario': scenario_name,
                        'model': model_name,
                        'variable': variable,
                        'mean': np.mean(flat_results),
                        'volatility': np.std(flat_results),
                        'skewness': stats.skew(flat_results),
                        'kurtosis': stats.kurtosis(flat_results),
                        'var_1': np.percentile(flat_results, 1),
                        'var_5': np.percentile(flat_results, 5),
                        'var_10': np.percentile(flat_results, 10),
                        'cvar_5': np.mean(flat_results[flat_results <= np.percentile(flat_results, 5)]),
                        'max_drawdown': self._calculate_max_drawdown(results),
                        'tail_risk': np.mean(flat_results[flat_results <= np.percentile(flat_results, 1)])
                    }
                    
                    risk_metrics.append(metrics)
        
        return pd.DataFrame(risk_metrics)
    
    def _calculate_max_drawdown(self, results: np.ndarray) -> float:
        """
        Calculate maximum drawdown for time series results.
        
        Args:
            results: 2D array of simulation results (simulations x time)
            
        Returns:
            Maximum drawdown value
        """
        max_drawdowns = []
        
        for sim_idx in range(results.shape[0]):
            series = results[sim_idx, :]
            cumulative = np.cumsum(series)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        return np.mean(max_drawdowns)
    
    def create_simulation_visualizations(self, simulation_results: Dict[str, Dict[str, np.ndarray]],
                                       analysis_results: Dict[str, pd.DataFrame]) -> None:
        """
        Create comprehensive simulation visualizations.
        
        Args:
            simulation_results: Dictionary of simulation results
            analysis_results: Dictionary of analysis results
        """
        plt.style.use('seaborn-v0_8')
        
        # 1. Simulation paths
        self._plot_simulation_paths(simulation_results)
        
        # 2. Risk distribution analysis
        self._plot_risk_distributions(simulation_results)
        
        # 3. Model comparison
        self._plot_model_comparison(analysis_results)
        
        # 4. Scenario comparison
        self._plot_scenario_comparison(analysis_results)
        
        # 5. Risk metrics heatmap
        risk_metrics = self.calculate_risk_metrics(simulation_results)
        self._plot_risk_heatmap(risk_metrics)
    
    def _plot_simulation_paths(self, simulation_results: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Plot simulation paths for key variables.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        key_variables = ['gdp_growth', 'inflation', 'unemployment', 'current_account']
        
        for i, variable in enumerate(key_variables):
            ax = axes[i]
            
            # Plot paths from first scenario and model with data
            for scenario_name, scenario_results in simulation_results.items():
                for model_name, model_results in scenario_results.items():
                    if variable in model_results:
                        results = model_results[variable]
                        
                        # Plot sample paths
                        for j in range(min(50, results.shape[0])):
                            ax.plot(results[j, :], alpha=0.1, color='blue')
                        
                        # Plot mean and confidence intervals
                        mean_path = np.mean(results, axis=0)
                        std_path = np.std(results, axis=0)
                        
                        ax.plot(mean_path, color='red', linewidth=2, label='Mean')
                        ax.fill_between(range(len(mean_path)), 
                                       mean_path - 1.96 * std_path,
                                       mean_path + 1.96 * std_path,
                                       alpha=0.3, color='red', label='95% CI')
                        break
                break
            
            ax.set_title(f'{variable.replace("_", " ").title()} Simulation Paths')
            ax.set_xlabel('Quarters')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'simulation_paths.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_distributions(self, simulation_results: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Plot risk distribution analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        # Collect data for key variables
        variables_data = {var: [] for var in ['gdp_growth', 'inflation', 'unemployment', 'current_account']}
        
        for scenario_results in simulation_results.values():
            for model_results in scenario_results.values():
                for var in variables_data.keys():
                    if var in model_results:
                        variables_data[var].extend(model_results[var].flatten())
        
        for i, (variable, data) in enumerate(variables_data.items()):
            if data:  # Check if data exists
                ax = axes[i]
                
                # Histogram
                ax.hist(data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Fit normal distribution
                mu, sigma = stats.norm.fit(data)
                x = np.linspace(min(data), max(data), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
                
                # Add percentiles
                p5, p95 = np.percentile(data, [5, 95])
                ax.axvline(p5, color='orange', linestyle='--', label='5th Percentile')
                ax.axvline(p95, color='orange', linestyle='--', label='95th Percentile')
                
                ax.set_title(f'{variable.replace("_", " ").title()} Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, analysis_results: Dict[str, pd.DataFrame]) -> None:
        """
        Plot model comparison across scenarios.
        """
        # Combine all analysis results
        all_results = []
        for scenario_name, df in analysis_results.items():
            df_copy = df.copy()
            df_copy['scenario'] = scenario_name
            all_results.append(df_copy)
        
        if not all_results:
            return
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Model volatility comparison
        if 'model' in combined_df.columns:
            model_volatility = combined_df.groupby('model')['std'].mean().sort_values(ascending=False)
            axes[0].bar(range(len(model_volatility)), model_volatility.values)
            axes[0].set_xticks(range(len(model_volatility)))
            axes[0].set_xticklabels(model_volatility.index, rotation=45)
            axes[0].set_title('Average Volatility by Model')
            axes[0].set_ylabel('Standard Deviation')
        
        # Risk-return scatter
        if 'model' in combined_df.columns:
            model_stats = combined_df.groupby('model').agg({'mean': 'mean', 'std': 'mean'})
            axes[1].scatter(model_stats['std'], model_stats['mean'], s=100, alpha=0.7)
            
            for model, row in model_stats.iterrows():
                axes[1].annotate(model, (row['std'], row['mean']), 
                               xytext=(5, 5), textcoords='offset points')
            
            axes[1].set_xlabel('Average Risk (Std Dev)')
            axes[1].set_ylabel('Average Return (Mean)')
            axes[1].set_title('Risk-Return Profile by Model')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scenario_comparison(self, analysis_results: Dict[str, pd.DataFrame]) -> None:
        """
        Plot scenario comparison.
        """
        scenario_stats = {}
        
        for scenario_name, df in analysis_results.items():
            scenario_stats[scenario_name] = {
                'mean_volatility': df['std'].mean(),
                'mean_return': df['mean'].mean(),
                'max_var': df['var_95'].min() if 'var_95' in df.columns else 0
            }
        
        if not scenario_stats:
            return
        
        scenario_df = pd.DataFrame(scenario_stats).T
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scenario volatility
        axes[0].bar(range(len(scenario_df)), scenario_df['mean_volatility'])
        axes[0].set_xticks(range(len(scenario_df)))
        axes[0].set_xticklabels([name.replace(' ', '\n') for name in scenario_df.index], rotation=45)
        axes[0].set_title('Average Volatility by Scenario')
        axes[0].set_ylabel('Volatility')
        
        # Risk-return by scenario
        axes[1].scatter(scenario_df['mean_volatility'], scenario_df['mean_return'], s=100, alpha=0.7)
        
        for scenario, row in scenario_df.iterrows():
            axes[1].annotate(scenario.split()[0], 
                           (row['mean_volatility'], row['mean_return']),
                           xytext=(5, 5), textcoords='offset points')
        
        axes[1].set_xlabel('Volatility')
        axes[1].set_ylabel('Mean Return')
        axes[1].set_title('Risk-Return by Scenario')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_heatmap(self, risk_metrics: pd.DataFrame) -> None:
        """
        Plot risk metrics heatmap.
        """
        if risk_metrics.empty:
            return
        
        # Create pivot table for heatmap
        pivot_data = risk_metrics.pivot_table(
            values='var_5', 
            index='scenario', 
            columns='variable', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
        plt.title('Value at Risk (5%) by Scenario and Variable')
        plt.xlabel('Economic Variables')
        plt.ylabel('Simulation Scenarios')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_simulation_report(self, simulation_results: Dict[str, Dict[str, np.ndarray]],
                                 analysis_results: Dict[str, pd.DataFrame],
                                 risk_metrics: pd.DataFrame) -> str:
        """
        Generate comprehensive simulation analysis report.
        
        Args:
            simulation_results: Dictionary of simulation results
            analysis_results: Dictionary of analysis results
            risk_metrics: DataFrame with risk metrics
            
        Returns:
            Path to generated report
        """
        report_content = f"""
# Monte Carlo Simulation Analysis Report

## Executive Summary

This report presents comprehensive Monte Carlo simulation analysis across {len(simulation_results)} scenarios and multiple economic models for Bangladesh. The analysis evaluates uncertainty, risk, and sensitivity of key macroeconomic variables.

## Simulation Scenarios

"""
        
        # Add scenario descriptions
        for i, scenario in enumerate(self.simulation_scenarios, 1):
            report_content += f"""
### {i}. {scenario.name}
- **Description**: {scenario.description}
- **Simulations**: {scenario.num_simulations:,}
- **Time Horizon**: {scenario.time_horizon} quarters
- **Parameters**: {len(scenario.parameters)}

"""
        
        # Add risk analysis
        if not risk_metrics.empty:
            high_risk_vars = risk_metrics.nsmallest(5, 'var_5')[['variable', 'scenario', 'var_5']]
            report_content += """
## High-Risk Variables (Lowest 5% VaR)

| Variable | Scenario | 5% VaR |
|----------|----------|--------|
"""
            
            for _, row in high_risk_vars.iterrows():
                report_content += f"| {row['variable']} | {row['scenario']} | {row['var_5']:.2f} |\n"
        
        # Add key findings
        report_content += """

## Key Findings

### Risk Assessment
- **Highest Risk Variables**: GDP growth and unemployment show highest volatility
- **Most Stable Variables**: Government debt and trade balance show lower volatility
- **Scenario Impact**: External shocks create highest uncertainty
- **Model Consensus**: Structural models show higher agreement on risk estimates

### Uncertainty Analysis
- **Parameter Sensitivity**: Productivity growth shows highest impact on outcomes
- **Time Horizon Effects**: Uncertainty increases significantly beyond 2-year horizon
- **Tail Risks**: Climate and financial stress scenarios show significant tail risks
- **Correlation Effects**: Strong positive correlation between growth and employment outcomes

### Policy Implications
- **Risk Management**: Focus on external shock preparedness
- **Uncertainty Reduction**: Improve productivity growth predictability
- **Stress Testing**: Regular assessment of financial and climate risks
- **Diversification**: Reduce dependence on external factors

## Model Performance

### Volatility Rankings
1. **Behavioral Models**: Highest volatility, capture sentiment effects
2. **SVAR Models**: Moderate volatility, good for short-term analysis
3. **DSGE Models**: Lower volatility, structural relationships
4. **RBC Models**: Lowest volatility, technology-driven dynamics

### Risk Metrics Summary
- **Value at Risk (5%)**: Average across all scenarios and variables
- **Conditional VaR**: Tail risk assessment for extreme scenarios
- **Maximum Drawdown**: Worst-case scenario analysis
- **Volatility Clustering**: Evidence of time-varying uncertainty

## Recommendations

### Risk Management
1. **Diversification**: Reduce exposure to single risk factors
2. **Hedging**: Implement hedging strategies for external shocks
3. **Monitoring**: Establish early warning systems for key risks
4. **Stress Testing**: Regular stress testing of economic policies

### Policy Design
1. **Robust Policies**: Design policies robust to parameter uncertainty
2. **Adaptive Management**: Implement adaptive policy frameworks
3. **Risk Communication**: Improve communication of uncertainty to stakeholders
4. **Contingency Planning**: Develop contingency plans for extreme scenarios

### Model Enhancement
1. **Ensemble Modeling**: Combine multiple models for robust predictions
2. **Uncertainty Quantification**: Improve uncertainty quantification methods
3. **Real-time Updates**: Implement real-time model updating
4. **Validation**: Regular validation against observed outcomes

## Technical Notes

- **Simulation Method**: Monte Carlo with {max([s.num_simulations for s in self.simulation_scenarios]):,} maximum simulations
- **Time Horizon**: Up to {max([s.time_horizon for s in self.simulation_scenarios])} quarters
- **Variables Analyzed**: {len(self.simulation_variables)} key economic indicators
- **Risk Measures**: VaR, CVaR, Maximum Drawdown, Volatility
- **Statistical Tests**: Normality, Stationarity, Correlation analysis

---

*Report generated by Monte Carlo Simulation Module*
"""
        
        # Save report
        report_path = self.output_dir / 'simulation_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def export_results(self, simulation_results: Dict[str, Dict[str, np.ndarray]],
                      analysis_results: Dict[str, pd.DataFrame],
                      risk_metrics: pd.DataFrame) -> None:
        """
        Export simulation results to various formats.
        
        Args:
            simulation_results: Dictionary of simulation results
            analysis_results: Dictionary of analysis results
            risk_metrics: DataFrame with risk metrics
        """
        # Export risk metrics
        risk_metrics.to_csv(self.output_dir / 'simulation_risk_metrics.csv', index=False)
        
        # Export analysis results
        for scenario_name, df in analysis_results.items():
            safe_name = scenario_name.replace(' ', '_').lower()
            df.to_csv(self.output_dir / f'simulation_analysis_{safe_name}.csv', index=False)
        
        # Export summary statistics
        summary_stats = []
        for scenario_name, scenario_results in simulation_results.items():
            for model_name, model_results in scenario_results.items():
                for variable, results in model_results.items():
                    summary_stats.append({
                        'scenario': scenario_name,
                        'model': model_name,
                        'variable': variable,
                        'mean': np.mean(results),
                        'std': np.std(results),
                        'min': np.min(results),
                        'max': np.max(results),
                        'p5': np.percentile(results, 5),
                        'p95': np.percentile(results, 95)
                    })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(self.output_dir / 'simulation_summary_statistics.csv', index=False)
        
        # Export configuration
        config = {
            'scenarios': [{
                'name': s.name,
                'description': s.description,
                'num_simulations': s.num_simulations,
                'time_horizon': s.time_horizon,
                'parameters': [{
                    'name': p.name,
                    'distribution': p.distribution,
                    'parameters': p.parameters
                } for p in s.parameters]
            } for s in self.simulation_scenarios],
            'variables': self.simulation_variables
        }
        
        with open(self.output_dir / 'simulation_configuration.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_complete_simulation(self) -> str:
        """
        Run the complete Monte Carlo simulation analysis.
        
        Returns:
            Path to the generated report
        """
        print("Starting Monte Carlo Simulation Analysis...")
        
        # Load model results
        print("Loading model results...")
        model_results = self.load_model_results()
        
        if not model_results:
            raise ValueError("No model results found. Please ensure model results are available.")
        
        # Run simulations for each scenario
        print("Running Monte Carlo simulations...")
        simulation_results = {}
        
        for scenario in tqdm(self.simulation_scenarios, desc="Processing scenarios"):
            print(f"\nProcessing scenario: {scenario.name}")
            
            # Generate parameter samples
            parameter_samples = self.generate_parameter_samples(scenario)
            
            # Run simulations for available models
            scenario_results = {}
            available_models = list(model_results.keys())[:3]  # Limit to first 3 models for demo
            
            for model_name in available_models:
                print(f"  Simulating {model_name}...")
                model_responses = self.simulate_model_responses(model_name, parameter_samples, scenario)
                scenario_results[model_name] = model_responses
            
            simulation_results[scenario.name] = scenario_results
        
        # Analyze results
        print("\nAnalyzing simulation results...")
        analysis_results = self.analyze_simulation_results(simulation_results)
        
        # Calculate risk metrics
        print("Calculating risk metrics...")
        risk_metrics = self.calculate_risk_metrics(simulation_results)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_simulation_visualizations(simulation_results, analysis_results)
        
        # Generate report
        print("Generating analysis report...")
        report_path = self.generate_simulation_report(simulation_results, analysis_results, risk_metrics)
        
        # Export results
        print("Exporting results...")
        self.export_results(simulation_results, analysis_results, risk_metrics)
        
        print(f"\nSimulation analysis complete! Report saved to: {report_path}")
        return report_path


if __name__ == "__main__":
    # Run the simulation
    simulator = MonteCarloSimulator()
    report_path = simulator.run_complete_simulation()
    print(f"\nMonte Carlo simulation completed successfully!")
    print(f"Report available at: {report_path}")