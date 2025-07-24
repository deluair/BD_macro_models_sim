#!/usr/bin/env python3
"""
Dynamic Stochastic General Equilibrium (DSGE) Model for Bangladesh

This module implements a New Keynesian DSGE model calibrated for the Bangladesh economy.
The model includes:
- Households with utility maximization
- Firms with price stickiness (Calvo pricing)
- Central bank with Taylor rule
- Government with fiscal policy
- External sector (small open economy features)

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Symbolic computation for model derivation
try:
    import sympy as sp
except ImportError:
    sp = None

# Bayesian estimation
try:
    import pymc as pm
    import arviz as az
except ImportError:
    pm = None
    az = None

logger = logging.getLogger(__name__)

@dataclass
class DSGEParameters:
    """
    DSGE Model Parameters for Bangladesh
    """
    # Household parameters
    beta: float = 0.99      # Discount factor
    sigma: float = 2.0      # Risk aversion (inverse of intertemporal elasticity)
    phi: float = 1.0        # Inverse Frisch elasticity of labor supply
    chi: float = 1.0        # Labor disutility parameter
    
    # Firm parameters
    alpha: float = 0.33     # Capital share in production
    theta: float = 0.75     # Calvo parameter (price stickiness)
    epsilon: float = 6.0    # Elasticity of substitution between goods
    delta: float = 0.025    # Depreciation rate
    
    # Monetary policy (Taylor rule)
    phi_pi: float = 1.5     # Inflation response
    phi_y: float = 0.5      # Output gap response
    rho_r: float = 0.8      # Interest rate smoothing
    
    # Fiscal policy
    phi_g: float = 0.2      # Government spending to GDP ratio
    phi_tau: float = 0.15   # Tax rate
    rho_g: float = 0.8      # Government spending persistence
    
    # External sector (small open economy)
    phi_f: float = 0.3      # Import share
    eta: float = 1.5        # Trade elasticity
    psi: float = 0.01       # Risk premium elasticity
    
    # Shock persistence parameters
    rho_a: float = 0.9      # Technology shock persistence
    rho_m: float = 0.5      # Monetary shock persistence
    rho_g_shock: float = 0.8  # Government spending shock persistence
    rho_f: float = 0.85     # Foreign shock persistence
    
    # Shock standard deviations
    sigma_a: float = 0.01   # Technology shock std
    sigma_m: float = 0.0025 # Monetary shock std
    sigma_g: float = 0.01   # Government spending shock std
    sigma_f: float = 0.01   # Foreign shock std
    
    # Bangladesh-specific parameters
    remittance_share: float = 0.06  # Remittances to GDP ratio
    export_share: float = 0.15      # Exports to GDP ratio
    import_share: float = 0.18      # Imports to GDP ratio
    
class DSGEModel:
    """
    New Keynesian DSGE Model for Bangladesh Economy
    """
    
    def __init__(self, config: Dict, data: Optional[pd.DataFrame] = None):
        """
        Initialize DSGE model
        
        Args:
            config: Model configuration
            data: Economic data for estimation
        """
        self.config = config
        self.data = data
        
        # Initialize parameters
        self.params = DSGEParameters()
        self._update_params_from_config()
        
        # Model variables
        self.variables = [
            'y',      # Output
            'c',      # Consumption
            'i',      # Investment
            'k',      # Capital
            'l',      # Labor
            'w',      # Real wage
            'r',      # Real interest rate
            'pi',     # Inflation
            'mc',     # Marginal cost
            'g',      # Government spending
            'tau',    # Tax rate
            'nx',     # Net exports
            'rer',    # Real exchange rate
            'y_star', # Foreign output
            'r_star', # Foreign interest rate
            'a',      # Technology shock
            'eps_m',  # Monetary shock
            'eps_g',  # Government spending shock
            'eps_f'   # Foreign shock
        ]
        
        # State space representation
        self.state_vars = ['k', 'a', 'eps_m', 'eps_g', 'eps_f']
        self.control_vars = ['y', 'c', 'i', 'l', 'w', 'r', 'pi', 'mc']
        
        # Steady state values
        self.steady_state = {}
        
        # Solution matrices
        self.solution = {}
        
        logger.info("DSGE Model for Bangladesh initialized")
    
    def _update_params_from_config(self):
        """
        Update parameters from configuration
        """
        if 'calibration' in self.config:
            calib = self.config['calibration']
            for param_name, param_value in calib.items():
                if hasattr(self.params, param_name):
                    setattr(self.params, param_name, param_value)
    
    def derive_equilibrium_conditions(self) -> Dict[str, str]:
        """
        Derive the equilibrium conditions of the DSGE model
        
        Returns:
            Dictionary of equilibrium equations
        """
        logger.info("Deriving equilibrium conditions...")
        
        equations = {
            # 1. Euler equation (consumption)
            'euler': "c[t]^(-sigma) = beta * E[t](c[t+1]^(-sigma) * (1 + r[t+1]))",
            
            # 2. Labor supply
            'labor_supply': "chi * l[t]^phi = w[t] * c[t]^(-sigma)",
            
            # 3. Capital accumulation
            'capital': "k[t+1] = (1 - delta) * k[t] + i[t]",
            
            # 4. Production function
            'production': "y[t] = a[t] * k[t]^alpha * l[t]^(1-alpha)",
            
            # 5. Capital rental rate
            'capital_rental': "r_k[t] = alpha * y[t] / k[t]",
            
            # 6. Wage equation
            'wage': "w[t] = (1 - alpha) * y[t] / l[t]",
            
            # 7. Marginal cost
            'marginal_cost': "mc[t] = (w[t]^(1-alpha) * r_k[t]^alpha) / (a[t] * alpha^alpha * (1-alpha)^(1-alpha))",
            
            # 8. Phillips curve (New Keynesian)
            'phillips': "pi[t] = beta * E[t](pi[t+1]) + ((1-theta)*(1-beta*theta)/theta) * mc[t]",
            
            # 9. Taylor rule
            'taylor': "r[t] = rho_r * r[t-1] + (1-rho_r) * (phi_pi * pi[t] + phi_y * (y[t] - y_ss)) + eps_m[t]",
            
            # 10. Fisher equation
            'fisher': "1 + r[t] = (1 + i[t]) / E[t](1 + pi[t+1])",
            
            # 11. Resource constraint
            'resource': "y[t] = c[t] + i[t] + g[t] + nx[t]",
            
            # 12. Government spending
            'government': "g[t] = rho_g * g[t-1] + eps_g[t]",
            
            # 13. Net exports (small open economy)
            'net_exports': "nx[t] = export_share * rer[t]^eta * y_star[t] - import_share * rer[t]^(-eta) * y[t]",
            
            # 14. Real exchange rate
            'rer_equation': "rer[t] = rer[t-1] * (1 + pi[t] - pi_star[t] - delta_e[t])",
            
            # 15. Technology shock
            'technology': "a[t] = rho_a * a[t-1] + eps_a[t]",
            
            # 16. Monetary shock
            'monetary_shock': "eps_m[t] = rho_m * eps_m[t-1] + u_m[t]",
            
            # 17. Government spending shock
            'fiscal_shock': "eps_g[t] = rho_g_shock * eps_g[t-1] + u_g[t]",
            
            # 18. Foreign shock
            'foreign_shock': "eps_f[t] = rho_f * eps_f[t-1] + u_f[t]"
        }
        
        return equations
    
    def compute_steady_state(self) -> Dict[str, float]:
        """
        Compute the steady state of the model
        
        Returns:
            Dictionary of steady state values
        """
        logger.info("Computing steady state...")
        
        # Parameters
        beta = self.params.beta
        alpha = self.params.alpha
        delta = self.params.delta
        sigma = self.params.sigma
        phi = self.params.phi
        chi = self.params.chi
        phi_g = self.params.phi_g
        
        # Steady state calculations
        r_ss = (1 / beta) - 1  # Real interest rate
        r_k_ss = r_ss + delta  # Capital rental rate
        
        # Capital-labor ratio
        k_l_ratio = (alpha / r_k_ss) ** (1 / (1 - alpha))
        
        # Wage
        w_ss = (1 - alpha) * k_l_ratio ** alpha
        
        # From labor supply condition
        # chi * l^phi = w * c^(-sigma)
        # From resource constraint and other conditions
        
        # Normalize labor to 1 in steady state
        l_ss = 1.0
        k_ss = k_l_ratio * l_ss
        
        # Output
        y_ss = k_ss ** alpha * l_ss ** (1 - alpha)
        
        # Investment
        i_ss = delta * k_ss
        
        # Government spending
        g_ss = phi_g * y_ss
        
        # Net exports (assume balanced trade in steady state)
        nx_ss = 0.0
        
        # Consumption
        c_ss = y_ss - i_ss - g_ss - nx_ss
        
        # Check labor supply condition
        chi_implied = w_ss * c_ss ** (-sigma) / (l_ss ** phi)
        
        # Marginal cost (should equal (epsilon-1)/epsilon in steady state)
        mc_ss = (self.params.epsilon - 1) / self.params.epsilon
        
        # Inflation (zero in steady state)
        pi_ss = 0.0
        
        # Real exchange rate (normalized to 1)
        rer_ss = 1.0
        
        # Technology (normalized to 1)
        a_ss = 1.0
        
        self.steady_state = {
            'y': y_ss,
            'c': c_ss,
            'i': i_ss,
            'k': k_ss,
            'l': l_ss,
            'w': w_ss,
            'r': r_ss,
            'r_k': r_k_ss,
            'pi': pi_ss,
            'mc': mc_ss,
            'g': g_ss,
            'nx': nx_ss,
            'rer': rer_ss,
            'a': a_ss,
            'eps_m': 0.0,
            'eps_g': 0.0,
            'eps_f': 0.0
        }
        
        logger.info(f"Steady state computed. Output: {y_ss:.4f}, Consumption: {c_ss:.4f}")
        return self.steady_state
    
    def linearize_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the model around steady state
        
        Returns:
            Tuple of (A, B, C, D) matrices for state space representation
            x[t+1] = A * x[t] + B * u[t]
            y[t] = C * x[t] + D * u[t]
        """
        logger.info("Linearizing model around steady state...")
        
        if not self.steady_state:
            self.compute_steady_state()
        
        # Number of variables
        n_vars = len(self.variables)
        n_states = len(self.state_vars)
        n_controls = len(self.control_vars)
        n_shocks = 4  # Technology, monetary, fiscal, foreign
        
        # Initialize matrices (simplified linearization)
        # In practice, this would involve symbolic differentiation
        
        # State transition matrix (simplified)
        A = np.zeros((n_states, n_states))
        
        # Capital accumulation: k[t+1] = (1-delta)*k[t] + i[t]
        A[0, 0] = 1 - self.params.delta  # k[t] coefficient
        
        # Technology shock: a[t+1] = rho_a * a[t]
        A[1, 1] = self.params.rho_a
        
        # Monetary shock: eps_m[t+1] = rho_m * eps_m[t]
        A[2, 2] = self.params.rho_m
        
        # Fiscal shock: eps_g[t+1] = rho_g * eps_g[t]
        A[3, 3] = self.params.rho_g_shock
        
        # Foreign shock: eps_f[t+1] = rho_f * eps_f[t]
        A[4, 4] = self.params.rho_f
        
        # Control matrix B (impact of shocks)
        B = np.zeros((n_states, n_shocks))
        B[1, 0] = 1  # Technology shock
        B[2, 1] = 1  # Monetary shock
        B[3, 2] = 1  # Fiscal shock
        B[4, 3] = 1  # Foreign shock
        
        # Observation matrix C (how states map to observables)
        C = np.eye(n_states, n_vars)
        
        # Direct transmission matrix D
        D = np.zeros((n_vars, n_shocks))
        
        self.solution = {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'state_vars': self.state_vars,
            'control_vars': self.control_vars
        }
        
        logger.info("Model linearization completed")
        return A, B, C, D
    
    def solve_model(self, method: str = 'klein') -> Dict:
        """
        Solve the linearized DSGE model
        
        Args:
            method: Solution method ('klein', 'schur', 'qz')
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving DSGE model using {method} method...")
        
        if not self.solution:
            self.linearize_model()
        
        # For demonstration, we'll use a simplified solution
        # In practice, you would use specialized DSGE solution methods
        
        A, B, C, D = self.solution['A'], self.solution['B'], self.solution['C'], self.solution['D']
        
        # Check stability (eigenvalues should be inside unit circle)
        eigenvalues = np.linalg.eigvals(A)
        stable = np.all(np.abs(eigenvalues) < 1)
        
        if not stable:
            logger.warning("Model solution may be unstable")
        
        # Policy functions (simplified)
        # x[t+1] = A * x[t] + B * shocks[t]
        
        self.solution.update({
            'eigenvalues': eigenvalues,
            'stable': stable,
            'method': method
        })
        
        logger.info(f"Model solved. Stability: {stable}")
        return self.solution
    
    def simulate(self, periods: int = 200, shocks: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Simulate the DSGE model
        
        Args:
            periods: Number of periods to simulate
            shocks: Shock matrix (periods x n_shocks)
            
        Returns:
            DataFrame with simulated time series
        """
        logger.info(f"Simulating DSGE model for {periods} periods...")
        
        if not self.solution:
            self.solve_model()
        
        n_states = len(self.state_vars)
        n_shocks = 4
        
        # Generate shocks if not provided
        if shocks is None:
            shocks = np.random.multivariate_normal(
                mean=np.zeros(n_shocks),
                cov=np.diag([
                    self.params.sigma_a**2,
                    self.params.sigma_m**2,
                    self.params.sigma_g**2,
                    self.params.sigma_f**2
                ]),
                size=periods
            )
        
        # Initialize state variables
        states = np.zeros((periods + 1, n_states))
        
        # Simulation
        A, B = self.solution['A'], self.solution['B']
        
        for t in range(periods):
            states[t + 1] = A @ states[t] + B @ shocks[t]
        
        # Convert to DataFrame
        simulation_data = pd.DataFrame(
            states[1:],  # Exclude initial period
            columns=self.state_vars,
            index=pd.date_range(start='2000-01-01', periods=periods, freq='Q')
        )
        
        # Add shocks
        shock_names = ['shock_technology', 'shock_monetary', 'shock_fiscal', 'shock_foreign']
        for i, shock_name in enumerate(shock_names):
            simulation_data[shock_name] = shocks[:, i]
        
        # Compute control variables (simplified)
        # In practice, these would be computed from policy functions
        ss = self.steady_state
        
        simulation_data['y'] = ss['y'] + 0.8 * simulation_data['a'] + 0.3 * simulation_data['k']
        simulation_data['c'] = ss['c'] + 0.6 * (simulation_data['y'] - ss['y'])
        simulation_data['i'] = ss['i'] + 0.4 * (simulation_data['k'] - 0)
        simulation_data['pi'] = 0.5 * simulation_data['eps_m'] + 0.3 * (simulation_data['y'] - ss['y'])
        simulation_data['r'] = ss['r'] + 1.5 * simulation_data['pi'] + 0.5 * (simulation_data['y'] - ss['y'])
        
        logger.info("Simulation completed")
        return simulation_data
    
    def impulse_response(self, shock_type: str, shock_size: float = 1.0, periods: int = 40) -> pd.DataFrame:
        """
        Compute impulse response functions
        
        Args:
            shock_type: Type of shock ('technology', 'monetary', 'fiscal', 'foreign')
            shock_size: Size of shock in standard deviations
            periods: Number of periods for IRF
            
        Returns:
            DataFrame with impulse responses
        """
        logger.info(f"Computing impulse response to {shock_type} shock...")
        
        if not self.solution:
            self.solve_model()
        
        shock_mapping = {
            'technology': 0,
            'monetary': 1,
            'fiscal': 2,
            'foreign': 3
        }
        
        if shock_type not in shock_mapping:
            raise ValueError(f"Unknown shock type: {shock_type}")
        
        shock_index = shock_mapping[shock_type]
        
        # Create shock vector
        shocks = np.zeros((periods, 4))
        shocks[0, shock_index] = shock_size
        
        # Simulate with shock
        irf_data = self.simulate(periods=periods, shocks=shocks)
        
        # Subtract steady state to get deviations
        for var in self.steady_state:
            if var in irf_data.columns:
                irf_data[var] -= self.steady_state[var]
        
        logger.info(f"Impulse response computed for {shock_type} shock")
        return irf_data
    
    def estimate_parameters(self, method: str = 'mle') -> Dict:
        """
        Estimate model parameters using data
        
        Args:
            method: Estimation method ('mle', 'bayesian', 'gmm')
            
        Returns:
            Estimation results
        """
        logger.info(f"Estimating DSGE parameters using {method} method...")
        
        if self.data is None:
            raise ValueError("No data provided for estimation")
        
        if method == 'bayesian' and pm is None:
            logger.warning("PyMC not available. Using MLE instead.")
            method = 'mle'
        
        if method == 'bayesian':
            return self._bayesian_estimation()
        elif method == 'mle':
            return self._mle_estimation()
        elif method == 'gmm':
            return self._gmm_estimation()
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    
    def _mle_estimation(self) -> Dict:
        """
        Maximum likelihood estimation
        """
        logger.info("Running MLE estimation...")
        
        # Prepare data
        obs_vars = ['gdp_growth', 'inflation', 'interest_rate']
        data_matrix = self._prepare_data_for_estimation(obs_vars)
        
        # Define likelihood function
        def log_likelihood(params_vector):
            # Update parameters
            self._update_parameters(params_vector)
            
            # Solve model
            try:
                self.solve_model()
                
                # Compute likelihood using Kalman filter
                # (Simplified implementation)
                ll = self._kalman_filter_likelihood(data_matrix)
                return -ll  # Negative for minimization
            except:
                return 1e10  # Large penalty for invalid parameters
        
        # Initial parameter guess
        initial_params = self._get_parameter_vector()
        
        # Bounds for parameters
        bounds = self._get_parameter_bounds()
        
        # Optimize
        result = opt.minimize(
            log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update parameters with estimated values
        self._update_parameters(result.x)
        
        estimation_results = {
            'method': 'mle',
            'success': result.success,
            'log_likelihood': -result.fun,
            'parameters': self._get_parameter_dict(result.x),
            'optimization_result': result
        }
        
        logger.info(f"MLE estimation completed. Success: {result.success}")
        return estimation_results
    
    def _bayesian_estimation(self) -> Dict:
        """
        Bayesian estimation using PyMC
        """
        logger.info("Running Bayesian estimation...")
        
        # Prepare data
        obs_vars = ['gdp_growth', 'inflation', 'interest_rate']
        data_matrix = self._prepare_data_for_estimation(obs_vars)
        
        with pm.Model() as model:
            # Prior distributions for parameters
            beta = pm.Beta('beta', alpha=99, beta=1)  # Discount factor
            sigma = pm.Gamma('sigma', alpha=2, beta=1)  # Risk aversion
            phi = pm.Gamma('phi', alpha=1, beta=1)     # Frisch elasticity
            theta = pm.Beta('theta', alpha=3, beta=1)   # Calvo parameter
            phi_pi = pm.Gamma('phi_pi', alpha=1.5, beta=1)  # Taylor rule
            
            # Model likelihood (simplified)
            # In practice, this would involve the full Kalman filter
            
            # Sample from posterior
            trace = pm.sample(
                draws=2000,
                tune=1000,
                chains=4,
                return_inferencedata=True
            )
        
        # Extract results
        estimation_results = {
            'method': 'bayesian',
            'trace': trace,
            'summary': az.summary(trace) if az else None
        }
        
        logger.info("Bayesian estimation completed")
        return estimation_results
    
    def _prepare_data_for_estimation(self, obs_vars: List[str]) -> np.ndarray:
        """
        Prepare data for estimation
        """
        # Extract relevant variables from data
        # This is a simplified implementation
        
        if 'combined' in self.data:
            data = self.data['combined']
        else:
            data = list(self.data.values())[0]
        
        # Map data columns to model variables
        data_mapping = {
            'gdp_growth': ['gdp_growth', 'bbs_gdp_growth', 'world_bank_gdp_growth'],
            'inflation': ['inflation', 'cpi_inflation', 'bbs_cpi_inflation'],
            'interest_rate': ['policy_rate', 'bangladesh_bank_policy_rate']
        }
        
        data_matrix = []
        for var in obs_vars:
            for col_name in data_mapping.get(var, [var]):
                if col_name in data.columns:
                    series = data[col_name].dropna()
                    if len(series) > 0:
                        data_matrix.append(series.values)
                        break
        
        return np.column_stack(data_matrix) if data_matrix else np.array([])
    
    def _kalman_filter_likelihood(self, data: np.ndarray) -> float:
        """
        Compute likelihood using Kalman filter (simplified)
        """
        # This is a placeholder for the full Kalman filter implementation
        # In practice, this would involve the full state space representation
        
        if data.size == 0:
            return -1e10
        
        # Simple likelihood based on data fit
        T, n_obs = data.shape
        
        # Simulate model
        sim_data = self.simulate(periods=T)
        
        # Match simulated to observed variables
        obs_vars = ['y', 'pi', 'r']
        
        likelihood = 0
        for i, var in enumerate(obs_vars[:n_obs]):
            if var in sim_data.columns:
                residuals = data[:, i] - sim_data[var].values[:T]
                likelihood += -0.5 * np.sum(residuals**2)
        
        return likelihood
    
    def _get_parameter_vector(self) -> np.ndarray:
        """
        Get parameter vector for estimation
        """
        return np.array([
            self.params.beta,
            self.params.sigma,
            self.params.phi,
            self.params.theta,
            self.params.phi_pi,
            self.params.phi_y,
            self.params.rho_r,
            self.params.rho_a,
            self.params.rho_m
        ])
    
    def _update_parameters(self, params_vector: np.ndarray):
        """
        Update model parameters from vector
        """
        param_names = [
            'beta', 'sigma', 'phi', 'theta', 'phi_pi',
            'phi_y', 'rho_r', 'rho_a', 'rho_m'
        ]
        
        for i, param_name in enumerate(param_names):
            setattr(self.params, param_name, params_vector[i])
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """
        Get parameter bounds for estimation
        """
        return [
            (0.9, 0.999),   # beta
            (0.5, 5.0),     # sigma
            (0.1, 3.0),     # phi
            (0.5, 0.9),     # theta
            (1.01, 3.0),    # phi_pi
            (0.0, 2.0),     # phi_y
            (0.0, 0.95),    # rho_r
            (0.0, 0.99),    # rho_a
            (0.0, 0.95)     # rho_m
        ]
    
    def _get_parameter_dict(self, params_vector: np.ndarray) -> Dict[str, float]:
        """
        Convert parameter vector to dictionary
        """
        param_names = [
            'beta', 'sigma', 'phi', 'theta', 'phi_pi',
            'phi_y', 'rho_r', 'rho_a', 'rho_m'
        ]
        
        return {name: params_vector[i] for i, name in enumerate(param_names)}
    
    def forecast(self, horizon: int = 8, scenario: str = 'baseline') -> pd.DataFrame:
        """
        Generate forecasts using the DSGE model
        
        Args:
            horizon: Forecast horizon in quarters
            scenario: Forecast scenario
            
        Returns:
            DataFrame with forecasts
        """
        logger.info(f"Generating {horizon}-period forecast using DSGE model...")
        
        if not self.solution:
            self.solve_model()
        
        # Define scenario-specific shocks
        scenario_shocks = {
            'baseline': np.zeros((horizon, 4)),
            'optimistic': np.array([
                [0.5, 0, 0, 0.3] if t == 0 else [0, 0, 0, 0] 
                for t in range(horizon)
            ]),
            'pessimistic': np.array([
                [-0.5, 0.5, 0, -0.3] if t == 0 else [0, 0, 0, 0] 
                for t in range(horizon)
            ])
        }
        
        shocks = scenario_shocks.get(scenario, scenario_shocks['baseline'])
        
        # Generate forecast
        forecast_data = self.simulate(periods=horizon, shocks=shocks)
        
        # Add confidence intervals (simplified)
        for var in ['y', 'pi', 'r', 'c']:
            if var in forecast_data.columns:
                std_dev = forecast_data[var].std()
                forecast_data[f"{var}_lower"] = forecast_data[var] - 1.96 * std_dev
                forecast_data[f"{var}_upper"] = forecast_data[var] + 1.96 * std_dev
        
        logger.info(f"Forecast generated for scenario: {scenario}")
        return forecast_data
    
    def policy_analysis(self, policy_change: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        """
        Analyze policy changes
        
        Args:
            policy_change: Dictionary of parameter changes
            
        Returns:
            Dictionary with policy analysis results
        """
        logger.info(f"Analyzing policy change: {policy_change}")
        
        # Store original parameters
        original_params = {}
        for param_name, new_value in policy_change.items():
            if hasattr(self.params, param_name):
                original_params[param_name] = getattr(self.params, param_name)
                setattr(self.params, param_name, new_value)
        
        # Solve model with new parameters
        self.solve_model()
        
        # Generate baseline simulation
        baseline = self.simulate(periods=40)
        
        # Generate impulse responses
        irfs = {}
        for shock_type in ['technology', 'monetary', 'fiscal', 'foreign']:
            irfs[shock_type] = self.impulse_response(shock_type)
        
        # Restore original parameters
        for param_name, original_value in original_params.items():
            setattr(self.params, param_name, original_value)
        
        results = {
            'baseline_simulation': baseline,
            'impulse_responses': irfs,
            'policy_change': policy_change
        }
        
        logger.info("Policy analysis completed")
        return results
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of model specification and parameters
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'New Keynesian DSGE',
            'country': 'Bangladesh',
            'frequency': 'Quarterly',
            'variables': self.variables,
            'state_variables': self.state_vars,
            'control_variables': self.control_vars,
            'parameters': {
                'beta': self.params.beta,
                'sigma': self.params.sigma,
                'phi': self.params.phi,
                'alpha': self.params.alpha,
                'theta': self.params.theta,
                'phi_pi': self.params.phi_pi,
                'phi_y': self.params.phi_y,
                'rho_r': self.params.rho_r
            },
            'steady_state': self.steady_state,
            'solution_available': bool(self.solution)
        }
        
        return summary

# Convenience functions
def create_bangladesh_dsge(config: Dict, data: Optional[pd.DataFrame] = None) -> DSGEModel:
    """
    Create a DSGE model for Bangladesh
    
    Args:
        config: Model configuration
        data: Economic data
        
    Returns:
        Configured DSGE model
    """
    return DSGEModel(config, data)

def run_dsge_analysis(config: Dict, data: pd.DataFrame) -> Dict:
    """
    Run complete DSGE analysis
    
    Args:
        config: Model configuration
        data: Economic data
        
    Returns:
        Analysis results
    """
    # Create model
    model = DSGEModel(config, data)
    
    # Solve model
    model.solve_model()
    
    # Run simulations
    simulation = model.simulate(periods=100)
    
    # Generate impulse responses
    irfs = {}
    for shock in ['technology', 'monetary', 'fiscal', 'foreign']:
        irfs[shock] = model.impulse_response(shock)
    
    # Generate forecasts
    forecasts = {}
    for scenario in ['baseline', 'optimistic', 'pessimistic']:
        forecasts[scenario] = model.forecast(horizon=8, scenario=scenario)
    
    return {
        'model': model,
        'simulation': simulation,
        'impulse_responses': irfs,
        'forecasts': forecasts
    }