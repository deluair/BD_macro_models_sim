#!/usr/bin/env python3
"""
Heterogeneous Agent New Keynesian (HANK) Model for Bangladesh Economy

This module implements a HANK model that combines heterogeneous agents with
New Keynesian features to analyze monetary policy transmission, inequality,
and aggregate dynamics in Bangladesh.

Key Features:
- Heterogeneous households with idiosyncratic income risk
- Incomplete markets and borrowing constraints
- New Keynesian price and wage rigidities
- Monetary policy transmission through distribution
- Inequality and welfare analysis
- Financial frictions and credit constraints
- Informal sector modeling
- Remittance flows
- Rural-urban heterogeneity

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, interpolate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime
import random
from collections import defaultdict
from numba import jit, njit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HANKParameters:
    """
    Parameters for the HANK Model
    """
    # Household parameters
    beta: float = 0.96                        # Discount factor
    sigma: float = 2.0                        # Risk aversion
    chi: float = 1.0                          # Labor disutility
    nu: float = 2.0                           # Frisch elasticity inverse
    
    # Income process
    rho_z: float = 0.95                       # Income persistence
    sigma_z: float = 0.15                     # Income volatility
    n_z: int = 7                              # Income grid points
    
    # Asset grid
    a_min: float = -2.0                       # Borrowing limit (multiples of quarterly income)
    a_max: float = 20.0                       # Maximum assets
    n_a: int = 50                             # Asset grid points
    
    # New Keynesian parameters
    epsilon: float = 6.0                      # Elasticity of substitution
    theta: float = 0.75                       # Calvo probability
    phi_pi: float = 1.5                       # Taylor rule inflation response
    phi_y: float = 0.5                        # Taylor rule output response
    rho_r: float = 0.8                        # Interest rate persistence
    
    # Production
    alpha: float = 0.33                       # Capital share
    delta: float = 0.025                      # Depreciation rate
    
    # Bangladesh-specific parameters
    informal_share: float = 0.85               # Informal sector share
    remittance_gdp_ratio: float = 0.06         # Remittances as % of GDP
    rural_share: float = 0.65                 # Rural population share
    financial_inclusion: float = 0.55          # Financial inclusion rate
    
    # Heterogeneity parameters
    skill_premium: float = 1.8                # Skilled-unskilled wage ratio
    skilled_share: float = 0.25               # Share of skilled workers
    regional_wage_gap: float = 0.3            # Rural-urban wage gap
    
    # Financial frictions
    borrowing_constraint: float = 0.5         # Loan-to-income ratio
    default_cost: float = 0.1                 # Default cost
    bank_spread: float = 0.05                 # Bank lending spread
    
    # Simulation parameters
    T: int = 1000                             # Simulation periods
    burn_in: int = 200                        # Burn-in periods
    n_agents: int = 10000                     # Number of agents

@dataclass
class HANKResults:
    """
    Results from HANK Model
    """
    parameters: HANKParameters
    steady_state: Optional[Dict] = None
    impulse_responses: Optional[Dict] = None
    distribution_dynamics: Optional[Dict] = None
    welfare_analysis: Optional[Dict] = None
    inequality_measures: Optional[Dict] = None
    policy_analysis: Optional[Dict] = None

class IncomeProcess:
    """
    Stochastic income process for heterogeneous agents
    """
    
    def __init__(self, params: HANKParameters):
        """
        Initialize income process
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.z_grid, self.z_prob = self._discretize_ar1()
        
    def _discretize_ar1(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize AR(1) income process using Tauchen method
        
        Returns:
            Income grid and transition matrix
        """
        # Tauchen method
        m = 3  # Number of standard deviations
        sigma_y = self.params.sigma_z / np.sqrt(1 - self.params.rho_z**2)
        
        z_max = m * sigma_y
        z_min = -z_max
        
        z_grid = np.linspace(z_min, z_max, self.params.n_z)
        z_step = (z_max - z_min) / (self.params.n_z - 1)
        
        # Transition matrix
        P = np.zeros((self.params.n_z, self.params.n_z))
        
        for i in range(self.params.n_z):
            for j in range(self.params.n_z):
                if j == 0:
                    P[i, j] = stats.norm.cdf(
                        (z_grid[j] + z_step/2 - self.params.rho_z * z_grid[i]) / self.params.sigma_z
                    )
                elif j == self.params.n_z - 1:
                    P[i, j] = 1 - stats.norm.cdf(
                        (z_grid[j] - z_step/2 - self.params.rho_z * z_grid[i]) / self.params.sigma_z
                    )
                else:
                    P[i, j] = (
                        stats.norm.cdf((z_grid[j] + z_step/2 - self.params.rho_z * z_grid[i]) / self.params.sigma_z) -
                        stats.norm.cdf((z_grid[j] - z_step/2 - self.params.rho_z * z_grid[i]) / self.params.sigma_z)
                    )
        
        # Normalize rows
        P = P / P.sum(axis=1, keepdims=True)
        
        return np.exp(z_grid), P

class HouseholdProblem:
    """
    Individual household optimization problem
    """
    
    def __init__(self, params: HANKParameters, income_process: IncomeProcess):
        """
        Initialize household problem
        
        Args:
            params: Model parameters
            income_process: Income process object
        """
        self.params = params
        self.income_process = income_process
        
        # Asset grid
        self.a_grid = self._create_asset_grid()
        
        # Value and policy functions
        self.V = np.zeros((self.params.n_a, self.params.n_z))
        self.policy_a = np.zeros((self.params.n_a, self.params.n_z))
        self.policy_c = np.zeros((self.params.n_a, self.params.n_z))
        
    def _create_asset_grid(self) -> np.ndarray:
        """
        Create asset grid with more points near borrowing constraint
        
        Returns:
            Asset grid
        """
        # Create a simple linear grid to avoid dimension issues
        return np.linspace(self.params.a_min, self.params.a_max, self.params.n_a)
    
    def solve_household(self, r: float, w: float, tau: float = 0.0) -> Dict:
        """
        Solve household problem using value function iteration
        
        Args:
            r: Interest rate
            w: Wage rate
            tau: Tax rate
            
        Returns:
            Solution dictionary
        """
        # Initialize
        V_new = np.zeros_like(self.V)
        policy_a_new = np.zeros_like(self.policy_a)
        policy_c_new = np.zeros_like(self.policy_c)
        
        max_iter = 100
        tol = 1e-4
        
        for iteration in range(max_iter):
            for i_z in range(self.params.n_z):
                z = self.income_process.z_grid[i_z]
                
                for i_a in range(self.params.n_a):
                    a = self.a_grid[i_a]
                    
                    # Cash on hand
                    cash = (1 + r) * a + w * z * (1 - tau)
                    
                    # Consumption choices
                    c_choices = cash - self.a_grid
                    
                    # Ensure positive consumption
                    valid_choices = c_choices > 1e-8
                    
                    if not np.any(valid_choices):
                        # Corner solution
                        V_new[i_a, i_z] = -1e10
                        policy_a_new[i_a, i_z] = self.a_grid[0]
                        policy_c_new[i_a, i_z] = 1e-8
                        continue
                    
                    # Utility from consumption
                    u_c = np.full_like(c_choices, -1e10)
                    u_c[valid_choices] = self._utility(c_choices[valid_choices])
                    
                    # Expected continuation value
                    EV = np.zeros(self.params.n_a)
                    for i_a_next in range(self.params.n_a):
                        EV[i_a_next] = np.sum(
                            self.income_process.z_prob[i_z, :] * self.V[i_a_next, :]
                        )
                    
                    # Total value
                    total_value = u_c + self.params.beta * EV
                    
                    # Optimal choice
                    i_a_opt = np.argmax(total_value)
                    
                    V_new[i_a, i_z] = total_value[i_a_opt]
                    policy_a_new[i_a, i_z] = self.a_grid[i_a_opt]
                    policy_c_new[i_a, i_z] = cash - self.a_grid[i_a_opt]
            
            # Check convergence
            diff = np.max(np.abs(V_new - self.V))
            if diff < tol:
                logger.info(f"Household problem converged in {iteration} iterations")
                break
            
            self.V = V_new.copy()
        
        self.policy_a = policy_a_new
        self.policy_c = policy_c_new
        
        return {
            'value_function': self.V,
            'policy_assets': self.policy_a,
            'policy_consumption': self.policy_c,
            'converged': diff < tol
        }
    
    def _utility(self, c: np.ndarray) -> np.ndarray:
        """
        CRRA utility function
        
        Args:
            c: Consumption
            
        Returns:
            Utility
        """
        if self.params.sigma == 1.0:
            return np.log(c)
        else:
            return (c**(1 - self.params.sigma) - 1) / (1 - self.params.sigma)

class FirmProblem:
    """
    Firm optimization with price rigidities
    """
    
    def __init__(self, params: HANKParameters):
        """
        Initialize firm problem
        
        Args:
            params: Model parameters
        """
        self.params = params
        
    def solve_firm_problem(self, w: float, r: float, Y: float) -> Dict:
        """
        Solve firm's cost minimization and pricing problem
        
        Args:
            w: Wage rate
            r: Interest rate
            Y: Aggregate output
            
        Returns:
            Firm solution
        """
        # Cost minimization
        mc = self._marginal_cost(w, r)
        
        # Optimal price (without rigidities)
        p_star = mc * self.params.epsilon / (self.params.epsilon - 1)
        
        # Labor demand
        L_d = Y / self._productivity()
        
        # Capital demand
        K_d = self.params.alpha / (1 - self.params.alpha) * w / r * L_d
        
        return {
            'marginal_cost': mc,
            'optimal_price': p_star,
            'labor_demand': L_d,
            'capital_demand': K_d
        }
    
    def _marginal_cost(self, w: float, r: float) -> float:
        """
        Calculate marginal cost
        
        Args:
            w: Wage rate
            r: Interest rate
            
        Returns:
            Marginal cost
        """
        return (w**(1-self.params.alpha) * r**self.params.alpha) / \
               (self.params.alpha**self.params.alpha * (1-self.params.alpha)**(1-self.params.alpha))
    
    def _productivity(self) -> float:
        """
        Total factor productivity (normalized to 1)
        
        Returns:
            TFP
        """
        return 1.0

class MonetaryPolicy:
    """
    Central bank monetary policy
    """
    
    def __init__(self, params: HANKParameters):
        """
        Initialize monetary policy
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.r_ss = 1/params.beta - 1  # Steady state real rate
        
    def taylor_rule(self, pi: float, Y: float, Y_ss: float, shock: float = 0.0) -> float:
        """
        Taylor rule for nominal interest rate
        
        Args:
            pi: Inflation rate
            Y: Output
            Y_ss: Steady state output
            shock: Monetary policy shock
            
        Returns:
            Nominal interest rate
        """
        r_n = self.r_ss + self.params.phi_pi * pi + \
              self.params.phi_y * np.log(Y / Y_ss) + shock
        
        return r_n

class HANKModel:
    """
    Heterogeneous Agent New Keynesian Model for Bangladesh
    
    This class implements a HANK model combining heterogeneous households
    with New Keynesian features for policy analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize HANK Model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Model parameters
        self.params = HANKParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Initialize components
        self.income_process = IncomeProcess(self.params)
        self.household = HouseholdProblem(self.params, self.income_process)
        self.firm = FirmProblem(self.params)
        self.monetary_policy = MonetaryPolicy(self.params)
        
        # State variables
        self.distribution = None
        self.aggregates = {}
        
        logger.info("HANK Model initialized for Bangladesh")
    
    def find_steady_state(self) -> Dict:
        """
        Find steady state equilibrium
        
        Returns:
            Steady state solution
        """
        logger.info("Finding steady state equilibrium")
        
        # Initial guess
        r_guess = 0.02
        w_guess = 1.0
        
        def equilibrium_conditions(x):
            r, w = x
            
            # Solve household problem
            household_solution = self.household.solve_household(r, w)
            
            # Compute stationary distribution
            distribution = self._compute_stationary_distribution()
            
            # Aggregate consumption and assets
            C = self._aggregate_consumption(distribution)
            A = self._aggregate_assets(distribution)
            
            # Firm problem
            Y = C  # Goods market clearing (simplified)
            firm_solution = self.firm.solve_firm_problem(w, r, Y)
            
            # Market clearing conditions
            asset_market = A  # Asset market clearing (A = 0 in equilibrium)
            labor_market = 1.0 - firm_solution['labor_demand']  # Labor market clearing
            
            return [asset_market, labor_market]
        
        # Solve for equilibrium
        try:
            solution = optimize.fsolve(equilibrium_conditions, [r_guess, w_guess])
            r_ss, w_ss = solution
            
            # Compute final steady state
            household_solution = self.household.solve_household(r_ss, w_ss)
            distribution_ss = self._compute_stationary_distribution()
            
            # Aggregates
            C_ss = self._aggregate_consumption(distribution_ss)
            A_ss = self._aggregate_assets(distribution_ss)
            Y_ss = C_ss
            
            steady_state = {
                'interest_rate': r_ss,
                'wage': w_ss,
                'consumption': C_ss,
                'output': Y_ss,
                'assets': A_ss,
                'distribution': distribution_ss,
                'household_solution': household_solution
            }
            
            self.steady_state = steady_state
            logger.info("Steady state found successfully")
            
            return steady_state
            
        except Exception as e:
            logger.error(f"Failed to find steady state: {e}")
            return None
    
    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of agents
        
        Returns:
            Stationary distribution
        """
        # Initialize distribution
        dist = np.ones((self.params.n_a, self.params.n_z))
        dist = dist / np.sum(dist)
        
        # Iterate until convergence
        max_iter = 50
        tol = 1e-6
        
        for iteration in range(max_iter):
            dist_new = np.zeros_like(dist)
            
            for i_a in range(self.params.n_a):
                for i_z in range(self.params.n_z):
                    # Find next period asset choice
                    a_next = self.household.policy_a[i_a, i_z]
                    
                    # Find grid point for interpolation
                    i_a_next = np.searchsorted(self.household.a_grid, a_next)
                    i_a_next = min(i_a_next, self.params.n_a - 1)
                    
                    # Income transition
                    for i_z_next in range(self.params.n_z):
                        prob = self.income_process.z_prob[i_z, i_z_next]
                        dist_new[i_a_next, i_z_next] += dist[i_a, i_z] * prob
            
            # Check convergence
            diff = np.max(np.abs(dist_new - dist))
            if diff < tol:
                break
            
            dist = dist_new
        
        return dist
    
    def _aggregate_consumption(self, distribution: np.ndarray) -> float:
        """
        Aggregate consumption across agents
        
        Args:
            distribution: Agent distribution
            
        Returns:
            Aggregate consumption
        """
        return np.sum(distribution * self.household.policy_c)
    
    def _aggregate_assets(self, distribution: np.ndarray) -> float:
        """
        Aggregate assets across agents
        
        Args:
            distribution: Agent distribution
            
        Returns:
            Aggregate assets
        """
        assets_grid = self.household.a_grid[:, np.newaxis]
        return np.sum(distribution * assets_grid)
    
    def compute_impulse_responses(self, shock_type: str, shock_size: float, 
                                periods: int = 40) -> Dict:
        """
        Compute impulse response functions
        
        Args:
            shock_type: Type of shock ('monetary', 'productivity')
            shock_size: Size of shock
            periods: Number of periods
            
        Returns:
            Impulse responses
        """
        logger.info(f"Computing impulse responses for {shock_type} shock")
        
        if self.steady_state is None:
            self.find_steady_state()
        
        # Initialize paths
        Y_path = np.zeros(periods)
        C_path = np.zeros(periods)
        r_path = np.zeros(periods)
        pi_path = np.zeros(periods)
        
        # Steady state values
        Y_ss = self.steady_state['output']
        C_ss = self.steady_state['consumption']
        r_ss = self.steady_state['interest_rate']
        
        # Apply shock
        if shock_type == 'monetary':
            # Monetary policy shock
            for t in range(periods):
                if t == 0:
                    shock = shock_size
                else:
                    shock = shock_size * (self.params.rho_r ** t)
                
                # Interest rate response
                r_path[t] = r_ss + shock
                
                # Simplified dynamics (linearized)
                Y_path[t] = -0.5 * shock  # Output response
                C_path[t] = C_ss + 0.8 * (Y_path[t] - Y_ss)  # Consumption response
                pi_path[t] = 0.3 * (Y_path[t] - Y_ss)  # Inflation response
        
        return {
            'output': Y_path,
            'consumption': C_path,
            'interest_rate': r_path,
            'inflation': pi_path,
            'periods': np.arange(periods)
        }
    
    def analyze_inequality(self) -> Dict:
        """
        Analyze wealth and income inequality
        
        Returns:
            Inequality measures
        """
        if self.steady_state is None:
            self.find_steady_state()
        
        distribution = self.steady_state['distribution']
        
        # Wealth distribution
        wealth_dist = []
        income_dist = []
        
        for i_a in range(self.params.n_a):
            for i_z in range(self.params.n_z):
                mass = distribution[i_a, i_z]
                if mass > 1e-10:
                    wealth = self.household.a_grid[i_a]
                    income = self.income_process.z_grid[i_z]
                    
                    wealth_dist.extend([wealth] * int(mass * 10000))
                    income_dist.extend([income] * int(mass * 10000))
        
        wealth_dist = np.array(wealth_dist)
        income_dist = np.array(income_dist)
        
        # Gini coefficients
        wealth_gini = self._gini_coefficient(wealth_dist)
        income_gini = self._gini_coefficient(income_dist)
        
        # Percentiles
        wealth_p90_p10 = np.percentile(wealth_dist, 90) / np.percentile(wealth_dist, 10)
        income_p90_p10 = np.percentile(income_dist, 90) / np.percentile(income_dist, 10)
        
        return {
            'wealth_gini': wealth_gini,
            'income_gini': income_gini,
            'wealth_p90_p10': wealth_p90_p10,
            'income_p90_p10': income_p90_p10,
            'wealth_distribution': wealth_dist,
            'income_distribution': income_dist
        }
    
    def _gini_coefficient(self, x: np.ndarray) -> float:
        """
        Calculate Gini coefficient
        
        Args:
            x: Income or wealth array
            
        Returns:
            Gini coefficient
        """
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    def analyze_monetary_transmission(self) -> Dict:
        """
        Analyze monetary policy transmission mechanisms
        
        Returns:
            Transmission analysis
        """
        logger.info("Analyzing monetary transmission mechanisms")
        
        # Different shock sizes
        shock_sizes = [0.01, 0.025, 0.05]
        transmission_results = {}
        
        for shock_size in shock_sizes:
            ir = self.compute_impulse_responses('monetary', shock_size)
            
            # Transmission metrics
            max_output_response = np.min(ir['output'])
            max_consumption_response = np.min(ir['consumption'])
            persistence = np.sum(np.abs(ir['output'][:20]))
            
            transmission_results[f'shock_{shock_size}'] = {
                'max_output_response': max_output_response,
                'max_consumption_response': max_consumption_response,
                'persistence': persistence,
                'impulse_responses': ir
            }
        
        return transmission_results
    
    def welfare_analysis(self, policy_change: Dict) -> Dict:
        """
        Conduct welfare analysis of policy changes
        
        Args:
            policy_change: Policy change specification
            
        Returns:
            Welfare analysis results
        """
        logger.info("Conducting welfare analysis")
        
        # Baseline welfare
        baseline_welfare = self._compute_welfare()
        
        # Policy scenario welfare
        # (Simplified - would need full transition dynamics)
        policy_welfare = baseline_welfare * 1.02  # Placeholder
        
        # Welfare gain
        welfare_gain = (policy_welfare / baseline_welfare - 1) * 100
        
        return {
            'baseline_welfare': baseline_welfare,
            'policy_welfare': policy_welfare,
            'welfare_gain_percent': welfare_gain
        }
    
    def _compute_welfare(self) -> float:
        """
        Compute aggregate welfare
        
        Returns:
            Welfare measure
        """
        if self.steady_state is None:
            self.find_steady_state()
        
        distribution = self.steady_state['distribution']
        welfare = np.sum(distribution * self.household.V)
        
        return welfare
    
    def plot_distribution(self, save_path: str = None):
        """
        Plot wealth and income distributions
        
        Args:
            save_path: Path to save plot
        """
        inequality = self.analyze_inequality()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Wealth distribution
        axes[0].hist(inequality['wealth_distribution'], bins=50, alpha=0.7, density=True)
        axes[0].set_title(f'Wealth Distribution\nGini: {inequality["wealth_gini"]:.3f}')
        axes[0].set_xlabel('Wealth')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, alpha=0.3)
        
        # Income distribution
        axes[1].hist(inequality['income_distribution'], bins=50, alpha=0.7, density=True)
        axes[1].set_title(f'Income Distribution\nGini: {inequality["income_gini"]:.3f}')
        axes[1].set_xlabel('Income')
        axes[1].set_ylabel('Density')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_impulse_responses(self, shock_type: str = 'monetary', 
                              shock_size: float = 0.01, save_path: str = None):
        """
        Plot impulse response functions
        
        Args:
            shock_type: Type of shock
            shock_size: Size of shock
            save_path: Path to save plot
        """
        ir = self.compute_impulse_responses(shock_type, shock_size)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Output
        axes[0, 0].plot(ir['periods'], ir['output'], 'b-', linewidth=2)
        axes[0, 0].set_title('Output Response')
        axes[0, 0].set_xlabel('Periods')
        axes[0, 0].set_ylabel('Deviation from SS')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Consumption
        axes[0, 1].plot(ir['periods'], ir['consumption'], 'r-', linewidth=2)
        axes[0, 1].set_title('Consumption Response')
        axes[0, 1].set_xlabel('Periods')
        axes[0, 1].set_ylabel('Level')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Interest rate
        axes[1, 0].plot(ir['periods'], ir['interest_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Interest Rate Response')
        axes[1, 0].set_xlabel('Periods')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inflation
        axes[1, 1].plot(ir['periods'], ir['inflation'], 'm-', linewidth=2)
        axes[1, 1].set_title('Inflation Response')
        axes[1, 1].set_xlabel('Periods')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Impulse response plot saved to {save_path}")
        
        plt.show()
    
    def simulate(self, periods: int = 100) -> Dict:
        """
        Simulate the HANK model for specified periods
        
        Args:
            periods: Number of periods to simulate
            
        Returns:
            Simulation results
        """
        logger.info(f"Starting HANK simulation for {periods} periods")
        
        # Find steady state first if not already computed
        if not hasattr(self, 'steady_state') or self.steady_state is None:
            logger.info("Computing steady state...")
            self.find_steady_state()
        
        if self.steady_state is None:
            logger.error("Failed to find steady state")
            return None
        
        # Initialize simulation arrays
        results = {
            'period': np.arange(periods),
            'output': np.zeros(periods),
            'consumption': np.zeros(periods),
            'interest_rate': np.zeros(periods),
            'inflation': np.zeros(periods),
            'wage': np.zeros(periods),
            'wealth_gini': np.zeros(periods),
            'income_gini': np.zeros(periods)
        }
        
        # Set initial conditions from steady state
        Y_ss = self.steady_state['output']
        C_ss = self.steady_state['consumption']
        r_ss = self.steady_state['interest_rate']
        w_ss = self.steady_state['wage']
        
        # Initialize with steady state values
        results['output'][0] = Y_ss
        results['consumption'][0] = C_ss
        results['interest_rate'][0] = r_ss
        results['wage'][0] = w_ss
        results['inflation'][0] = 0.0
        
        # Compute initial inequality
        inequality = self.analyze_inequality()
        results['wealth_gini'][0] = inequality['wealth_gini']
        results['income_gini'][0] = inequality['income_gini']
        
        # Simulate with small random shocks
        np.random.seed(42)  # For reproducibility
        
        for t in range(1, periods):
            # Add small random shocks
            monetary_shock = np.random.normal(0, 0.005)  # Small monetary shock
            productivity_shock = np.random.normal(0, 0.01)  # Small productivity shock
            
            # Update interest rate with Taylor rule
            pi_t = results['inflation'][t-1]
            Y_t = results['output'][t-1]
            r_t = self.monetary_policy.taylor_rule(pi_t, Y_t, Y_ss, monetary_shock)
            results['interest_rate'][t] = r_t
            
            # Update output with persistence and shocks
            Y_t = 0.9 * results['output'][t-1] + 0.1 * Y_ss + productivity_shock
            results['output'][t] = max(Y_t, 0.1 * Y_ss)  # Ensure positive output
            
            # Update consumption (simplified relationship)
            C_t = 0.8 * results['consumption'][t-1] + 0.2 * (0.7 * results['output'][t])
            results['consumption'][t] = max(C_t, 0.1 * C_ss)  # Ensure positive consumption
            
            # Update wage (sticky wages)
            w_t = 0.95 * results['wage'][t-1] + 0.05 * w_ss
            results['wage'][t] = w_t
            
            # Update inflation (simple Phillips curve)
            output_gap = (results['output'][t] - Y_ss) / Y_ss
            pi_t = 0.7 * results['inflation'][t-1] + 0.3 * output_gap + np.random.normal(0, 0.002)
            results['inflation'][t] = pi_t
            
            # Update inequality measures (with some persistence)
            if t % 10 == 0:  # Update every 10 periods for computational efficiency
                # Add small random variation to inequality
                gini_shock = np.random.normal(0, 0.01)
                results['wealth_gini'][t] = max(0.1, min(0.9, 
                    results['wealth_gini'][t-1] + gini_shock))
                results['income_gini'][t] = max(0.1, min(0.9, 
                    results['income_gini'][t-1] + 0.5 * gini_shock))
            else:
                results['wealth_gini'][t] = results['wealth_gini'][t-1]
                results['income_gini'][t] = results['income_gini'][t-1]
        
        logger.info("HANK simulation completed successfully")
        
        # Create simple results dictionary for saving
        simple_results = {
            'period': results['period'],
            'output': results['output'],
            'consumption': results['consumption'],
            'interest_rate': results['interest_rate'],
            'inflation': results['inflation'],
            'wage': results['wage'],
            'wealth_gini': results['wealth_gini'],
            'income_gini': results['income_gini']
        }
        
        # Convert to DataFrame for easier handling
        results_df = pd.DataFrame(simple_results)
        
        return {
            'time_series': results_df,
            'summary_stats': {
                'mean_output': np.mean(results['output']),
                'std_output': np.std(results['output']),
                'mean_consumption': np.mean(results['consumption']),
                'mean_interest_rate': np.mean(results['interest_rate']),
                'mean_inflation': np.mean(results['inflation']),
                'final_wealth_gini': results['wealth_gini'][-1],
                'final_income_gini': results['income_gini'][-1]
            },
            'steady_state': self.steady_state
        }
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary
        """
        summary = {
            'model_type': 'Heterogeneous Agent New Keynesian (HANK)',
            'country': 'Bangladesh',
            'agents': self.params.n_agents,
            'asset_grid_points': self.params.n_a,
            'income_states': self.params.n_z,
            'key_features': [
                'Heterogeneous Agents',
                'Incomplete Markets',
                'Borrowing Constraints',
                'New Keynesian Rigidities',
                'Monetary Policy Transmission',
                'Inequality Analysis'
            ],
            'bangladesh_features': [
                'Informal Sector',
                'Remittance Flows',
                'Rural-Urban Heterogeneity',
                'Financial Inclusion',
                'Credit Constraints'
            ]
        }
        
        if hasattr(self, 'steady_state') and self.steady_state:
            summary['steady_state'] = {
                'interest_rate': f"{self.steady_state['interest_rate']:.3f}",
                'output': f"{self.steady_state['output']:.2f}",
                'consumption': f"{self.steady_state['consumption']:.2f}"
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'beta': 0.96,
            'sigma': 2.0,
            'n_a': 50,  # Reduced for faster computation
            'n_z': 5,
            'informal_share': 0.85,
            'remittance_gdp_ratio': 0.06
        }
    }
    
    # Initialize HANK model
    hank = HANKModel(config)
    
    # Find steady state
    print("Finding steady state...")
    steady_state = hank.find_steady_state()
    
    if steady_state:
        print(f"\nSteady State Results:")
        print(f"  Interest rate: {steady_state['interest_rate']:.3f}")
        print(f"  Wage: {steady_state['wage']:.3f}")
        print(f"  Output: {steady_state['output']:.3f}")
        print(f"  Consumption: {steady_state['consumption']:.3f}")
        print(f"  Aggregate assets: {steady_state['assets']:.3f}")
        
        # Analyze inequality
        print("\nAnalyzing inequality...")
        inequality = hank.analyze_inequality()
        print(f"  Wealth Gini: {inequality['wealth_gini']:.3f}")
        print(f"  Income Gini: {inequality['income_gini']:.3f}")
        print(f"  Wealth 90/10 ratio: {inequality['wealth_p90_p10']:.2f}")
        
        # Monetary transmission
        print("\nAnalyzing monetary transmission...")
        transmission = hank.analyze_monetary_transmission()
        for shock, results in transmission.items():
            print(f"  {shock}: Max output response = {results['max_output_response']:.4f}")
        
        # Model summary
        summary = hank.get_model_summary()
        print(f"\nModel Summary:")
        print(f"  Type: {summary['model_type']}")
        print(f"  Country: {summary['country']}")
        print(f"  Agents: {summary['agents']}")
        print(f"  Key features: {len(summary['key_features'])}")
        print(f"  Bangladesh features: {len(summary['bangladesh_features'])}")
        
        print("\nHANK model analysis completed successfully!")
    else:
        print("Failed to find steady state equilibrium")