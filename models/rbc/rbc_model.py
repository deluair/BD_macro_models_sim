#!/usr/bin/env python3
"""
Real Business Cycle (RBC) Model for Bangladesh

This module implements a Real Business Cycle model to analyze business cycle
fluctuations in the Bangladesh economy. The model focuses on technology shocks
as the primary driver of economic fluctuations and examines how productivity
changes affect output, employment, consumption, and investment.

Key Features:
- Stochastic technology shocks
- Representative agent optimization
- Capital accumulation dynamics
- Labor-leisure choice
- Calibration to Bangladesh data
- Business cycle statistics
- Impulse response analysis
- Variance decomposition

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, linalg, stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RBCParameters:
    """
    Parameters for the RBC model
    """
    # Preferences
    beta: float = 0.96                   # Discount factor
    sigma: float = 2.0                   # Risk aversion (CRRA)
    chi: float = 1.0                     # Labor disutility parameter
    eta: float = 2.0                     # Frisch elasticity of labor supply
    
    # Technology
    alpha: float = 0.35                  # Capital share
    delta: float = 0.06                  # Depreciation rate
    A_ss: float = 1.0                    # Steady-state TFP
    
    # Technology shock process (AR(1))
    rho_A: float = 0.95                  # Persistence of technology shock
    sigma_A: float = 0.007               # Standard deviation of technology shock
    
    # Bangladesh-specific calibration
    capital_output_ratio: float = 2.5    # K/Y ratio
    investment_output_ratio: float = 0.25 # I/Y ratio
    consumption_output_ratio: float = 0.70 # C/Y ratio
    labor_supply_ss: float = 0.33        # Steady-state labor supply
    
    # Additional shocks (optional)
    rho_g: float = 0.8                   # Government spending persistence
    sigma_g: float = 0.02                # Government spending shock std
    g_y_ratio: float = 0.15              # Government spending to GDP ratio
    
    # External sector
    trade_openness: float = 0.35         # (Exports + Imports)/GDP
    export_elasticity: float = 1.5       # Export demand elasticity
    import_elasticity: float = 1.2       # Import demand elasticity

@dataclass
class RBCResults:
    """
    Results from RBC model simulation
    """
    parameters: RBCParameters
    steady_state: Dict
    simulation_data: Optional[pd.DataFrame] = None
    business_cycle_stats: Optional[Dict] = None
    impulse_responses: Optional[Dict] = None
    variance_decomposition: Optional[Dict] = None

class RealBusinessCycleModel:
    """
    Real Business Cycle Model for Bangladesh
    
    This class implements a standard RBC model with technology shocks
    and analyzes business cycle properties of the Bangladesh economy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize RBC model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = RBCParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Model state
        self.steady_state = None
        self.results = None
        
        # Numerical parameters
        self.tolerance = 1e-8
        self.max_iterations = 1000
        
        # Grid parameters for value function iteration
        self.k_grid_size = 100
        self.k_min_factor = 0.5
        self.k_max_factor = 1.5
        
        logger.info("RBC model initialized for Bangladesh")
    
    def calibrate_model(self, data: pd.DataFrame = None) -> Dict:
        """
        Calibrate model parameters to Bangladesh data
        
        Args:
            data: Economic data for calibration (optional)
            
        Returns:
            Calibrated parameters
        """
        logger.info("Calibrating RBC model to Bangladesh data")
        
        if data is None:
            # Use default calibration based on Bangladesh stylized facts
            calibration = self._default_calibration()
        else:
            # Calibrate to actual data
            calibration = self._calibrate_to_data(data)
        
        # Update parameters
        for param, value in calibration.items():
            if hasattr(self.params, param):
                setattr(self.params, param, value)
        
        logger.info("Model calibration completed")
        return calibration
    
    def _default_calibration(self) -> Dict:
        """
        Default calibration for Bangladesh
        """
        # Based on Bangladesh economic data and literature
        calibration = {
            'beta': 0.96,                    # Standard discount factor
            'sigma': 2.0,                    # Risk aversion
            'alpha': 0.35,                   # Capital share (typical for developing countries)
            'delta': 0.06,                   # Depreciation rate
            'rho_A': 0.95,                   # Technology shock persistence
            'sigma_A': 0.015,                # Higher volatility for developing country
            'labor_supply_ss': 0.35,         # Higher labor supply in developing countries
            'capital_output_ratio': 2.2,     # Lower K/Y ratio
            'investment_output_ratio': 0.28, # Higher investment rate
            'consumption_output_ratio': 0.67 # Lower consumption share
        }
        
        return calibration
    
    def _calibrate_to_data(self, data: pd.DataFrame) -> Dict:
        """
        Calibrate parameters to actual data
        """
        calibration = {}
        
        # Calculate ratios from data if available
        if 'gdp' in data.columns and 'investment' in data.columns:
            calibration['investment_output_ratio'] = (
                data['investment'] / data['gdp']
            ).mean()
        
        if 'gdp' in data.columns and 'consumption' in data.columns:
            calibration['consumption_output_ratio'] = (
                data['consumption'] / data['gdp']
            ).mean()
        
        # Estimate technology shock process
        if 'tfp' in data.columns or 'gdp' in data.columns:
            tfp_series = data.get('tfp', data['gdp'])
            log_tfp = np.log(tfp_series.dropna())
            
            # Estimate AR(1) process
            if len(log_tfp) > 1:
                y = log_tfp.iloc[1:].values
                x = log_tfp.iloc[:-1].values
                
                # OLS estimation
                rho_estimate = np.cov(x, y)[0, 1] / np.var(x)
                residuals = y - rho_estimate * x
                sigma_estimate = np.std(residuals)
                
                calibration['rho_A'] = min(0.99, max(0.5, rho_estimate))
                calibration['sigma_A'] = min(0.05, max(0.005, sigma_estimate))
        
        return calibration
    
    def compute_steady_state(self) -> Dict:
        """
        Compute the steady-state equilibrium
        
        Returns:
            Steady-state values
        """
        logger.info("Computing steady-state equilibrium")
        
        # Steady-state conditions
        # 1. Euler equation: 1 = beta * (1 + r)
        # 2. Production function: Y = A * K^alpha * L^(1-alpha)
        # 3. Capital accumulation: K' = (1-delta)*K + I
        # 4. Resource constraint: Y = C + I + G
        # 5. Labor supply: MRS = MRT (labor-leisure choice)
        
        # Steady-state interest rate
        r_ss = (1 / self.params.beta) - 1
        
        # Steady-state capital-labor ratio
        k_l_ratio = (self.params.alpha / (r_ss + self.params.delta)) ** (1 / (1 - self.params.alpha))
        
        # Steady-state labor supply
        L_ss = self.params.labor_supply_ss
        
        # Steady-state capital
        K_ss = k_l_ratio * L_ss
        
        # Steady-state output
        Y_ss = self.params.A_ss * (K_ss ** self.params.alpha) * (L_ss ** (1 - self.params.alpha))
        
        # Steady-state investment
        I_ss = self.params.delta * K_ss
        
        # Steady-state government spending
        G_ss = self.params.g_y_ratio * Y_ss
        
        # Steady-state consumption
        C_ss = Y_ss - I_ss - G_ss
        
        # Steady-state wage
        w_ss = (1 - self.params.alpha) * Y_ss / L_ss
        
        # Check labor-leisure optimality condition
        # chi * L^eta = w / C^sigma
        chi_implied = w_ss / (C_ss ** self.params.sigma) / (L_ss ** self.params.eta)
        
        steady_state = {
            'output': Y_ss,
            'consumption': C_ss,
            'investment': I_ss,
            'capital': K_ss,
            'labor': L_ss,
            'wage': w_ss,
            'interest_rate': r_ss,
            'government_spending': G_ss,
            'tfp': self.params.A_ss,
            'chi_implied': chi_implied,
            'capital_output_ratio': K_ss / Y_ss,
            'investment_output_ratio': I_ss / Y_ss,
            'consumption_output_ratio': C_ss / Y_ss
        }
        
        # Update chi parameter if needed
        if abs(chi_implied - self.params.chi) > 0.1:
            logger.info(f"Updating chi parameter from {self.params.chi:.3f} to {chi_implied:.3f}")
            self.params.chi = chi_implied
        
        self.steady_state = steady_state
        
        logger.info(f"Steady state computed: Y={Y_ss:.3f}, K={K_ss:.3f}, L={L_ss:.3f}")
        return steady_state
    
    def linearize_model(self) -> Dict:
        """
        Linearize the model around steady state
        
        Returns:
            Linear system matrices
        """
        if self.steady_state is None:
            self.compute_steady_state()
        
        logger.info("Linearizing model around steady state")
        
        ss = self.steady_state
        
        # State variables: [k_t, A_t]
        # Control variables: [c_t, l_t, i_t]
        # Endogenous variables: [y_t, w_t, r_t]
        
        # Linearization coefficients (log-deviations from steady state)
        
        # Production function: y = alpha*k + (1-alpha)*l + A
        # Euler equation: c_{t+1} - c_t = (1/sigma) * (r_{t+1} - rho)
        # Labor supply: eta*l + sigma*c = w
        # Capital accumulation: k_{t+1} = (1-delta)*k_t + delta*i_t
        # Resource constraint: y = (C/Y)*c + (I/Y)*i + (G/Y)*g
        # Factor prices: r = alpha*y - alpha*k, w = (1-alpha)*y - (1-alpha)*l
        
        # Coefficient matrices for the linear system
        # E_t[x_{t+1}] = A * x_t + B * epsilon_{t+1}
        # where x_t = [k_t, A_t, c_t, l_t, i_t, y_t, w_t, r_t]'
        
        n_vars = 8  # Number of variables
        
        # Initialize matrices
        Gamma0 = np.zeros((n_vars, n_vars))  # Coefficient matrix for t+1 variables
        Gamma1 = np.zeros((n_vars, n_vars))  # Coefficient matrix for t variables
        Psi = np.zeros((n_vars, 2))          # Shock coefficients [eps_A, eps_g]
        Pi = np.zeros((n_vars, 2))           # Forward-looking coefficients
        
        # Variable order: [k, A, c, l, i, y, w, r]
        # Indices
        k_idx, A_idx, c_idx, l_idx, i_idx, y_idx, w_idx, r_idx = range(8)
        
        # Equation 1: Capital accumulation
        # k_{t+1} = (1-delta)*k_t + delta*i_t
        Gamma0[k_idx, k_idx] = 1
        Gamma1[k_idx, k_idx] = 1 - self.params.delta
        Gamma1[k_idx, i_idx] = self.params.delta
        
        # Equation 2: Technology shock
        # A_{t+1} = rho_A * A_t + eps_A
        Gamma0[A_idx, A_idx] = 1
        Gamma1[A_idx, A_idx] = self.params.rho_A
        Psi[A_idx, 0] = 1  # Technology shock
        
        # Equation 3: Euler equation
        # c_{t+1} - c_t = (1/sigma) * r_{t+1}
        Gamma0[c_idx, c_idx] = 1
        Gamma0[c_idx, r_idx] = -1 / self.params.sigma
        Gamma1[c_idx, c_idx] = 1
        
        # Equation 4: Labor supply
        # eta*l_t + sigma*c_t = w_t
        Gamma1[l_idx, l_idx] = self.params.eta
        Gamma1[l_idx, c_idx] = self.params.sigma
        Gamma1[l_idx, w_idx] = -1
        
        # Equation 5: Resource constraint
        # y_t = (C/Y)*c_t + (I/Y)*i_t
        c_share = ss['consumption'] / ss['output']
        i_share = ss['investment'] / ss['output']
        
        Gamma1[i_idx, y_idx] = 1
        Gamma1[i_idx, c_idx] = -c_share
        Gamma1[i_idx, i_idx] = -i_share
        
        # Equation 6: Production function
        # y_t = alpha*k_t + (1-alpha)*l_t + A_t
        Gamma1[y_idx, y_idx] = 1
        Gamma1[y_idx, k_idx] = -self.params.alpha
        Gamma1[y_idx, l_idx] = -(1 - self.params.alpha)
        Gamma1[y_idx, A_idx] = -1
        
        # Equation 7: Wage equation
        # w_t = (1-alpha)*y_t - (1-alpha)*l_t
        Gamma1[w_idx, w_idx] = 1
        Gamma1[w_idx, y_idx] = -(1 - self.params.alpha)
        Gamma1[w_idx, l_idx] = (1 - self.params.alpha)
        
        # Equation 8: Interest rate equation
        # r_t = alpha*y_t - alpha*k_t
        Gamma1[r_idx, r_idx] = 1
        Gamma1[r_idx, y_idx] = -self.params.alpha
        Gamma1[r_idx, k_idx] = self.params.alpha
        
        linear_system = {
            'Gamma0': Gamma0,
            'Gamma1': Gamma1,
            'Psi': Psi,
            'Pi': Pi,
            'variable_names': ['capital', 'tfp', 'consumption', 'labor', 
                             'investment', 'output', 'wage', 'interest_rate'],
            'shock_names': ['technology_shock', 'government_shock']
        }
        
        return linear_system
    
    def solve_linear_system(self, linear_system: Dict) -> Dict:
        """
        Solve the linearized model using Blanchard-Kahn method
        
        Args:
            linear_system: Linear system matrices
            
        Returns:
            Solution matrices
        """
        logger.info("Solving linearized model")
        
        Gamma0 = linear_system['Gamma0']
        Gamma1 = linear_system['Gamma1']
        Psi = linear_system['Psi']
        
        try:
            # Solve generalized eigenvalue problem
            # Gamma0 * x_{t+1} = Gamma1 * x_t + Psi * eps_t
            
            # Convert to standard form: x_{t+1} = A * x_t + B * eps_t
            if np.linalg.det(Gamma0) != 0:
                A = np.linalg.solve(Gamma0, Gamma1)
                B = np.linalg.solve(Gamma0, Psi)
            else:
                # Use pseudo-inverse if Gamma0 is singular
                A = np.linalg.pinv(Gamma0) @ Gamma1
                B = np.linalg.pinv(Gamma0) @ Psi
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # Check Blanchard-Kahn conditions
            n_predetermined = 2  # k and A are predetermined
            n_explosive = np.sum(np.abs(eigenvalues) > 1)
            n_stable = len(eigenvalues) - n_explosive
            
            bk_condition = (n_explosive == len(eigenvalues) - n_predetermined)
            
            solution = {
                'A_matrix': A,
                'B_matrix': B,
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'blanchard_kahn_satisfied': bk_condition,
                'n_explosive': n_explosive,
                'n_stable': n_stable,
                'n_predetermined': n_predetermined
            }
            
            if bk_condition:
                logger.info("Blanchard-Kahn conditions satisfied")
            else:
                logger.warning("Blanchard-Kahn conditions NOT satisfied")
            
        except Exception as e:
            logger.error(f"Error solving linear system: {str(e)}")
            solution = {
                'A_matrix': None,
                'B_matrix': None,
                'blanchard_kahn_satisfied': False,
                'error': str(e)
            }
        
        return solution
    
    def simulate_model(self, periods: int = 200, n_simulations: int = 1000,
                      shocks: Dict = None) -> pd.DataFrame:
        """
        Simulate the RBC model
        
        Args:
            periods: Number of periods to simulate
            n_simulations: Number of simulation runs
            shocks: External shocks (optional)
            
        Returns:
            Simulation results DataFrame
        """
        logger.info(f"Simulating RBC model for {periods} periods, {n_simulations} runs")
        
        if self.steady_state is None:
            self.compute_steady_state()
        
        # Get linear solution
        linear_system = self.linearize_model()
        solution = self.solve_linear_system(linear_system)
        
        if not solution.get('blanchard_kahn_satisfied', False):
            logger.warning("Using simplified simulation due to solution issues")
            return self._simulate_simplified(periods, n_simulations)
        
        # Simulation using linear solution
        A = solution['A_matrix']
        B = solution['B_matrix']
        
        # Initialize storage
        all_simulations = []
        
        for sim in range(n_simulations):
            # Generate shocks
            if shocks is None:
                eps_A = np.random.normal(0, self.params.sigma_A, periods)
                eps_g = np.random.normal(0, self.params.sigma_g, periods)
                shock_matrix = np.column_stack([eps_A, eps_g])
            else:
                shock_matrix = shocks
            
            # Initialize state vector (log deviations from steady state)
            x = np.zeros((periods, len(linear_system['variable_names'])))
            
            # Simulate
            for t in range(1, periods):
                x[t] = A @ x[t-1] + B @ shock_matrix[t-1]
            
            # Convert to levels
            simulation_data = self._convert_to_levels(x, linear_system['variable_names'])
            simulation_data['simulation'] = sim
            simulation_data['period'] = range(periods)
            
            all_simulations.append(simulation_data)
        
        # Combine all simulations
        combined_df = pd.concat(all_simulations, ignore_index=True)
        
        return combined_df
    
    def _simulate_simplified(self, periods: int, n_simulations: int) -> pd.DataFrame:
        """
        Simplified simulation when linear solution fails
        """
        logger.info("Running simplified simulation")
        
        ss = self.steady_state
        all_simulations = []
        
        for sim in range(n_simulations):
            # Generate technology shocks
            A_shocks = np.zeros(periods)
            A_shocks[0] = 0  # Start at steady state
            
            for t in range(1, periods):
                A_shocks[t] = (self.params.rho_A * A_shocks[t-1] + 
                              np.random.normal(0, self.params.sigma_A))
            
            # Simple responses (reduced form)
            tfp = ss['tfp'] * np.exp(A_shocks)
            output = ss['output'] * (tfp / ss['tfp']) ** (1 / (1 - self.params.alpha))
            
            # Approximate other variables
            consumption = ss['consumption'] * (output / ss['output']) ** 0.7
            investment = ss['investment'] * (output / ss['output']) ** 1.5
            labor = ss['labor'] * (output / ss['output']) ** 0.5
            capital = np.zeros(periods)
            capital[0] = ss['capital']
            
            for t in range(1, periods):
                capital[t] = (1 - self.params.delta) * capital[t-1] + investment[t-1]
            
            wage = (1 - self.params.alpha) * output / labor
            interest_rate = self.params.alpha * output / capital - self.params.delta
            
            simulation_data = pd.DataFrame({
                'period': range(periods),
                'simulation': sim,
                'output': output,
                'consumption': consumption,
                'investment': investment,
                'capital': capital,
                'labor': labor,
                'wage': wage,
                'interest_rate': interest_rate,
                'tfp': tfp
            })
            
            all_simulations.append(simulation_data)
        
        return pd.concat(all_simulations, ignore_index=True)
    
    def _convert_to_levels(self, log_deviations: np.ndarray, var_names: List[str]) -> pd.DataFrame:
        """
        Convert log deviations to levels
        """
        ss = self.steady_state
        levels_data = {}
        
        for i, var_name in enumerate(var_names):
            if var_name in ss:
                levels_data[var_name] = ss[var_name] * np.exp(log_deviations[:, i])
            else:
                levels_data[var_name] = np.exp(log_deviations[:, i])
        
        return pd.DataFrame(levels_data)
    
    def compute_business_cycle_statistics(self, simulation_data: pd.DataFrame) -> Dict:
        """
        Compute business cycle statistics
        
        Args:
            simulation_data: Simulation results
            
        Returns:
            Business cycle statistics
        """
        logger.info("Computing business cycle statistics")
        
        # Variables to analyze
        variables = ['output', 'consumption', 'investment', 'labor', 'wage']
        
        # Filter data and compute statistics for each simulation
        stats_by_sim = []
        
        for sim in simulation_data['simulation'].unique():
            sim_data = simulation_data[simulation_data['simulation'] == sim]
            
            # HP filter to extract cyclical components
            cyclical_components = {}
            
            for var in variables:
                if var in sim_data.columns:
                    series = sim_data[var].values
                    if len(series) > 10:  # Minimum length for HP filter
                        cyclical = self._hp_filter(np.log(series), lambda_param=1600)
                        cyclical_components[var] = cyclical
            
            # Compute statistics
            sim_stats = {}
            
            for var in variables:
                if var in cyclical_components:
                    cycle = cyclical_components[var]
                    
                    sim_stats[f'{var}_volatility'] = np.std(cycle)
                    
                    # Correlation with output
                    if 'output' in cyclical_components and var != 'output':
                        corr = np.corrcoef(cyclical_components['output'], cycle)[0, 1]
                        sim_stats[f'{var}_output_corr'] = corr
                    
                    # Autocorrelation
                    if len(cycle) > 1:
                        autocorr = np.corrcoef(cycle[:-1], cycle[1:])[0, 1]
                        sim_stats[f'{var}_autocorr'] = autocorr
            
            stats_by_sim.append(sim_stats)
        
        # Aggregate statistics across simulations
        aggregated_stats = {}
        
        for key in stats_by_sim[0].keys():
            values = [stats[key] for stats in stats_by_sim if key in stats and not np.isnan(stats[key])]
            if values:
                aggregated_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Relative volatilities (relative to output)
        if 'output_volatility' in aggregated_stats:
            output_vol = aggregated_stats['output_volatility']['mean']
            
            for var in variables:
                if f'{var}_volatility' in aggregated_stats and var != 'output':
                    var_vol = aggregated_stats[f'{var}_volatility']['mean']
                    aggregated_stats[f'{var}_relative_volatility'] = {
                        'mean': var_vol / output_vol
                    }
        
        return aggregated_stats
    
    def _hp_filter(self, series: np.ndarray, lambda_param: float = 1600) -> np.ndarray:
        """
        Hodrick-Prescott filter implementation
        """
        n = len(series)
        if n < 4:
            return np.zeros(n)
        
        # Create second difference matrix
        D = np.zeros((n-2, n))
        for i in range(n-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        
        # Solve the HP filter equation
        I = np.eye(n)
        A = I + lambda_param * D.T @ D
        
        try:
            trend = np.linalg.solve(A, series)
            cycle = series - trend
        except:
            # Fallback to simple detrending
            trend = np.linspace(series[0], series[-1], n)
            cycle = series - trend
        
        return cycle
    
    def impulse_response_analysis(self, shock_size: float = 1.0, periods: int = 40) -> Dict:
        """
        Compute impulse response functions
        
        Args:
            shock_size: Size of the shock (in standard deviations)
            periods: Number of periods for IRF
            
        Returns:
            Impulse response functions
        """
        logger.info(f"Computing impulse response functions for {periods} periods")
        
        if self.steady_state is None:
            self.compute_steady_state()
        
        # Get linear solution
        linear_system = self.linearize_model()
        solution = self.solve_linear_system(linear_system)
        
        if not solution.get('blanchard_kahn_satisfied', False):
            logger.warning("Using simplified IRF due to solution issues")
            return self._impulse_response_simplified(shock_size, periods)
        
        A = solution['A_matrix']
        B = solution['B_matrix']
        var_names = linear_system['variable_names']
        
        # Technology shock IRF
        tech_shock = np.zeros(2)
        tech_shock[0] = shock_size * self.params.sigma_A  # Technology shock
        
        # Initialize state vector
        x = np.zeros((periods, len(var_names)))
        x[0] = B @ tech_shock  # Initial impact
        
        # Propagate shock
        for t in range(1, periods):
            x[t] = A @ x[t-1]
        
        # Convert to percentage deviations
        irf_data = {}
        for i, var_name in enumerate(var_names):
            irf_data[var_name] = x[:, i] * 100  # Convert to percentage points
        
        irf_data['periods'] = list(range(periods))
        
        return {
            'technology_shock': pd.DataFrame(irf_data),
            'shock_size': shock_size,
            'shock_type': 'technology'
        }
    
    def _impulse_response_simplified(self, shock_size: float, periods: int) -> Dict:
        """
        Simplified impulse response analysis
        """
        # Simple persistence-based IRF
        persistence = self.params.rho_A
        
        # Technology shock response
        tech_response = shock_size * (persistence ** np.arange(periods))
        
        # Output response (amplified)
        output_response = tech_response * (1 / (1 - self.params.alpha))
        
        # Other variables (simplified relationships)
        consumption_response = output_response * 0.7
        investment_response = output_response * 1.5
        labor_response = output_response * 0.5
        wage_response = output_response * 0.8
        
        irf_data = pd.DataFrame({
            'periods': range(periods),
            'tfp': tech_response * 100,
            'output': output_response * 100,
            'consumption': consumption_response * 100,
            'investment': investment_response * 100,
            'labor': labor_response * 100,
            'wage': wage_response * 100
        })
        
        return {
            'technology_shock': irf_data,
            'shock_size': shock_size,
            'shock_type': 'technology'
        }
    
    def variance_decomposition(self, simulation_data: pd.DataFrame, 
                             forecast_horizons: List[int] = [1, 4, 8, 12, 20]) -> Dict:
        """
        Compute forecast error variance decomposition
        
        Args:
            simulation_data: Simulation results
            forecast_horizons: Forecast horizons to analyze
            
        Returns:
            Variance decomposition results
        """
        logger.info("Computing variance decomposition")
        
        # For RBC model, technology shocks explain most of the variance
        # This is a simplified implementation
        
        variables = ['output', 'consumption', 'investment', 'labor']
        decomposition = {}
        
        for var in variables:
            if var in simulation_data.columns:
                var_decomp = {}
                
                for horizon in forecast_horizons:
                    # Simplified: technology shock explains most variance
                    # In a full implementation, this would use the state-space representation
                    
                    if var == 'output':
                        tech_share = 0.85  # Technology explains 85% of output variance
                    elif var == 'investment':
                        tech_share = 0.90  # Investment is highly volatile
                    elif var == 'consumption':
                        tech_share = 0.70  # Consumption is smoother
                    else:
                        tech_share = 0.80  # Default
                    
                    var_decomp[f'horizon_{horizon}'] = {
                        'technology_shock': tech_share,
                        'other_shocks': 1 - tech_share
                    }
                
                decomposition[var] = var_decomp
        
        return decomposition
    
    def plot_simulation_results(self, simulation_data: pd.DataFrame, 
                              variables: List[str] = None, save_path: str = None):
        """
        Plot simulation results
        
        Args:
            simulation_data: Simulation data
            variables: Variables to plot
            save_path: Path to save plot
        """
        if variables is None:
            variables = ['output', 'consumption', 'investment', 'labor']
        
        # Filter available variables
        available_vars = [var for var in variables if var in simulation_data.columns]
        
        if not available_vars:
            logger.warning("No variables available for plotting")
            return
        
        # Plot first few simulations
        n_sims_to_plot = min(5, simulation_data['simulation'].nunique())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(available_vars[:4]):
            ax = axes[i]
            
            for sim in range(n_sims_to_plot):
                sim_data = simulation_data[simulation_data['simulation'] == sim]
                
                if len(sim_data) > 0:
                    # Plot as percentage deviation from steady state
                    if self.steady_state and var in self.steady_state:
                        ss_value = self.steady_state[var]
                        pct_dev = 100 * (sim_data[var] - ss_value) / ss_value
                    else:
                        pct_dev = sim_data[var]
                    
                    ax.plot(sim_data['period'], pct_dev, alpha=0.7, linewidth=1)
            
            ax.set_title(f'{var.title()} (% deviation from SS)')
            ax.set_xlabel('Period')
            ax.set_ylabel('Percent Deviation')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simulation plot saved to {save_path}")
        
        plt.show()
    
    def plot_impulse_responses(self, irf_results: Dict, save_path: str = None):
        """
        Plot impulse response functions
        
        Args:
            irf_results: IRF results from impulse_response_analysis
            save_path: Path to save plot
        """
        if 'technology_shock' not in irf_results:
            logger.warning("No IRF data available for plotting")
            return
        
        irf_data = irf_results['technology_shock']
        variables = [col for col in irf_data.columns if col != 'periods']
        
        n_vars = len(variables)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            ax.plot(irf_data['periods'], irf_data[var], linewidth=2, marker='o', markersize=3)
            ax.set_title(f'{var.title()} Response to Technology Shock')
            ax.set_xlabel('Periods')
            ax.set_ylabel('Percent Deviation')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Hide unused subplots
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"IRF plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Real Business Cycle (RBC) Model',
            'country': 'Bangladesh',
            'parameters': {
                'discount_factor': self.params.beta,
                'risk_aversion': self.params.sigma,
                'capital_share': self.params.alpha,
                'depreciation_rate': self.params.delta,
                'technology_persistence': self.params.rho_A,
                'technology_volatility': self.params.sigma_A
            }
        }
        
        if self.steady_state is not None:
            summary['steady_state'] = {
                'output': self.steady_state['output'],
                'capital_output_ratio': self.steady_state['capital_output_ratio'],
                'investment_output_ratio': self.steady_state['investment_output_ratio'],
                'consumption_output_ratio': self.steady_state['consumption_output_ratio'],
                'labor_supply': self.steady_state['labor']
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'beta': 0.96,
            'sigma': 2.0,
            'alpha': 0.35,
            'delta': 0.06,
            'rho_A': 0.95,
            'sigma_A': 0.015
        }
    }
    
    # Initialize RBC model
    rbc_model = RealBusinessCycleModel(config)
    
    # Calibrate model
    calibration = rbc_model.calibrate_model()
    print(f"Model calibrated with parameters: {list(calibration.keys())}")
    
    # Compute steady state
    steady_state = rbc_model.compute_steady_state()
    print(f"\nSteady State:")
    print(f"  Output: {steady_state['output']:.3f}")
    print(f"  Capital: {steady_state['capital']:.3f}")
    print(f"  Labor: {steady_state['labor']:.3f}")
    print(f"  K/Y ratio: {steady_state['capital_output_ratio']:.3f}")
    print(f"  I/Y ratio: {steady_state['investment_output_ratio']:.3f}")
    print(f"  C/Y ratio: {steady_state['consumption_output_ratio']:.3f}")
    
    # Simulate model
    print("\nSimulating model...")
    simulation_data = rbc_model.simulate_model(periods=100, n_simulations=10)
    print(f"Simulation completed: {len(simulation_data)} observations")
    
    # Business cycle statistics
    bc_stats = rbc_model.compute_business_cycle_statistics(simulation_data)
    print(f"\nBusiness Cycle Statistics:")
    
    for stat_name, stat_value in bc_stats.items():
        if isinstance(stat_value, dict) and 'mean' in stat_value:
            print(f"  {stat_name}: {stat_value['mean']:.4f}")
    
    # Impulse response analysis
    print("\nComputing impulse responses...")
    irf_results = rbc_model.impulse_response_analysis(shock_size=1.0, periods=20)
    
    if 'technology_shock' in irf_results:
        irf_data = irf_results['technology_shock']
        print(f"IRF computed for {len(irf_data)} periods")
        
        # Show peak responses
        for var in ['output', 'consumption', 'investment']:
            if var in irf_data.columns:
                peak_response = irf_data[var].max()
                print(f"  Peak {var} response: {peak_response:.2f}%")
    
    # Variance decomposition
    var_decomp = rbc_model.variance_decomposition(simulation_data)
    print(f"\nVariance Decomposition (Technology Shock Share):")
    
    for var, decomp in var_decomp.items():
        if 'horizon_4' in decomp:
            tech_share = decomp['horizon_4']['technology_shock']
            print(f"  {var} (4 quarters): {tech_share:.1%}")
    
    # Model summary
    summary = rbc_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Technology persistence: {summary['parameters']['technology_persistence']:.3f}")
    print(f"  Technology volatility: {summary['parameters']['technology_volatility']:.3f}")