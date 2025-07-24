#!/usr/bin/env python3
"""
Search and Matching Model for Bangladesh Labor Market

This module implements a search and matching model to analyze labor market
dynamics, unemployment, job creation and destruction, and wage determination
in the Bangladesh economy. The model incorporates both formal and informal
sectors, which is particularly relevant for developing countries.

Key Features:
- Job search and matching frictions
- Endogenous job creation and destruction
- Wage bargaining
- Formal and informal sector dynamics
- Unemployment benefits and policies
- Sectoral mobility
- Skills heterogeneity
- Business cycle effects on labor markets

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, stats
from scipy.integrate import odeint
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
class SearchMatchingParameters:
    """
    Parameters for the Search and Matching model
    """
    # Matching function parameters
    matching_efficiency: float = 0.5        # Matching function efficiency
    matching_elasticity: float = 0.5        # Elasticity of matching w.r.t. unemployment
    
    # Job creation and destruction
    job_creation_cost: float = 0.1          # Cost of creating a job
    job_destruction_rate: float = 0.05      # Exogenous job destruction rate
    productivity_threshold: float = 0.3     # Endogenous destruction threshold
    
    # Wage bargaining
    worker_bargaining_power: float = 0.5    # Worker's bargaining power
    unemployment_benefit: float = 0.3       # Unemployment benefit (replacement ratio)
    
    # Discount factors
    discount_rate: float = 0.05             # Discount rate
    
    # Productivity parameters
    productivity_mean: float = 1.0          # Mean productivity
    productivity_std: float = 0.2           # Productivity standard deviation
    productivity_persistence: float = 0.9   # AR(1) persistence of productivity
    
    # Bangladesh-specific parameters
    formal_sector_share: float = 0.15       # Share of formal sector employment
    informal_sector_productivity: float = 0.6  # Relative productivity of informal sector
    
    # Sectoral parameters
    formal_matching_efficiency: float = 0.6  # Formal sector matching efficiency
    informal_matching_efficiency: float = 0.8  # Informal sector matching efficiency
    
    # Policy parameters
    minimum_wage: float = 0.4               # Minimum wage (relative to mean productivity)
    employment_protection: float = 0.1      # Employment protection strength
    active_labor_policy: float = 0.0        # Active labor market policy intensity
    
    # Demographics and skills
    skill_levels: int = 3                   # Number of skill levels
    skill_distribution: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])  # Low, medium, high skill shares
    skill_productivity: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.4])  # Relative productivity by skill
    
    # External shocks
    business_cycle_volatility: float = 0.02  # Business cycle shock volatility
    trade_shock_probability: float = 0.1    # Probability of trade shock
    trade_shock_magnitude: float = 0.15     # Magnitude of trade shock

@dataclass
class SearchMatchingResults:
    """
    Results from Search and Matching model
    """
    parameters: SearchMatchingParameters
    steady_state: Optional[Dict] = None
    simulation_data: Optional[pd.DataFrame] = None
    labor_market_indicators: Optional[Dict] = None
    policy_analysis: Optional[Dict] = None
    sectoral_dynamics: Optional[Dict] = None

class SearchMatchingModel:
    """
    Search and Matching Model for Bangladesh Labor Market
    
    This class implements a comprehensive search and matching model
    to analyze labor market dynamics and policy interventions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Search and Matching model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = SearchMatchingParameters()
        
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
        
        logger.info("Search and Matching model initialized for Bangladesh")
    
    def calibrate_model(self, data: pd.DataFrame = None) -> Dict:
        """
        Calibrate model parameters to Bangladesh labor market data
        
        Args:
            data: Labor market data for calibration (optional)
            
        Returns:
            Calibrated parameters
        """
        logger.info("Calibrating Search and Matching model to Bangladesh data")
        
        if data is None:
            # Use default calibration based on Bangladesh labor market facts
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
        Default calibration for Bangladesh labor market
        """
        # Based on Bangladesh labor market statistics and literature
        calibration = {
            'matching_efficiency': 0.4,         # Lower efficiency in developing countries
            'matching_elasticity': 0.6,         # Higher elasticity due to informal sector
            'job_destruction_rate': 0.08,       # Higher turnover in developing countries
            'worker_bargaining_power': 0.3,     # Lower bargaining power
            'unemployment_benefit': 0.1,        # Limited social protection
            'formal_sector_share': 0.13,        # Based on Bangladesh statistics
            'informal_sector_productivity': 0.55, # Lower informal productivity
            'minimum_wage': 0.35,               # Relatively low minimum wage
            'employment_protection': 0.15       # Moderate employment protection
        }
        
        return calibration
    
    def _calibrate_to_data(self, data: pd.DataFrame) -> Dict:
        """
        Calibrate parameters to actual labor market data
        """
        calibration = {}
        
        # Unemployment rate
        if 'unemployment_rate' in data.columns:
            avg_unemployment = data['unemployment_rate'].mean()
            # Adjust matching efficiency to target unemployment rate
            calibration['matching_efficiency'] = 0.5 * (0.05 / max(avg_unemployment, 0.01))
        
        # Job finding rate
        if 'job_finding_rate' in data.columns:
            avg_finding_rate = data['job_finding_rate'].mean()
            calibration['matching_efficiency'] = avg_finding_rate / 0.5
        
        # Formal sector employment
        if 'formal_employment_share' in data.columns:
            formal_share = data['formal_employment_share'].mean()
            calibration['formal_sector_share'] = formal_share
        
        # Wage volatility
        if 'real_wage' in data.columns:
            wage_volatility = data['real_wage'].std() / data['real_wage'].mean()
            calibration['productivity_std'] = wage_volatility * 0.5
        
        return calibration
    
    def compute_steady_state(self) -> Dict:
        """
        Compute the steady-state equilibrium
        
        Returns:
            Steady-state values
        """
        logger.info("Computing steady-state equilibrium")
        
        # Steady-state conditions:
        # 1. Job creation condition: Expected value of job = Job creation cost
        # 2. Job destruction condition: Productivity threshold
        # 3. Matching function: Job finding rate
        # 4. Flow equilibrium: Inflows = Outflows
        # 5. Wage bargaining: Nash bargaining solution
        
        def steady_state_equations(vars):
            u_formal, u_informal, theta_formal, theta_informal, w_formal, w_informal = vars
            
            # Ensure non-negative values
            u_formal = max(u_formal, 0.001)
            u_informal = max(u_informal, 0.001)
            theta_formal = max(theta_formal, 0.001)
            theta_informal = max(theta_informal, 0.001)
            
            # Employment rates
            e_formal = 1 - u_formal
            e_informal = 1 - u_informal
            
            # Job finding rates
            f_formal = self._job_finding_rate(theta_formal, 'formal')
            f_informal = self._job_finding_rate(theta_informal, 'informal')
            
            # Job filling rates
            q_formal = self._job_filling_rate(theta_formal, 'formal')
            q_informal = self._job_filling_rate(theta_informal, 'informal')
            
            # Productivity thresholds
            R_formal = self._reservation_productivity(w_formal, 'formal')
            R_informal = self._reservation_productivity(w_informal, 'informal')
            
            # Expected productivity (above threshold)
            E_p_formal = self._expected_productivity(R_formal)
            E_p_informal = self._expected_productivity(R_informal)
            
            # Job destruction rates
            s_formal = self._job_destruction_rate(R_formal)
            s_informal = self._job_destruction_rate(R_informal)
            
            # Equations
            eq1 = u_formal - s_formal / (s_formal + f_formal)  # Formal unemployment
            eq2 = u_informal - s_informal / (s_informal + f_informal)  # Informal unemployment
            
            # Job creation conditions
            J_formal = (E_p_formal - w_formal) / (self.params.discount_rate + s_formal)
            eq3 = J_formal - self.params.job_creation_cost  # Formal job creation
            
            J_informal = (E_p_informal * self.params.informal_sector_productivity - w_informal) / (self.params.discount_rate + s_informal)
            eq4 = J_informal - self.params.job_creation_cost * 0.5  # Informal job creation (lower cost)
            
            # Wage bargaining
            U_formal = self.params.unemployment_benefit / self.params.discount_rate
            W_formal = w_formal / self.params.discount_rate
            
            eq5 = w_formal - (self.params.worker_bargaining_power * E_p_formal + 
                             (1 - self.params.worker_bargaining_power) * self.params.unemployment_benefit)
            
            eq6 = w_informal - (self.params.worker_bargaining_power * E_p_informal * self.params.informal_sector_productivity + 
                               (1 - self.params.worker_bargaining_power) * self.params.unemployment_benefit * 0.5)
            
            return [eq1, eq2, eq3, eq4, eq5, eq6]
        
        # Initial guess
        initial_guess = [0.05, 0.15, 1.0, 2.0, 0.8, 0.5]
        
        try:
            # Solve the system
            solution = optimize.fsolve(steady_state_equations, initial_guess, xtol=self.tolerance)
            u_formal, u_informal, theta_formal, theta_informal, w_formal, w_informal = solution
            
            # Ensure valid solution
            u_formal = max(min(u_formal, 0.5), 0.001)
            u_informal = max(min(u_informal, 0.8), 0.001)
            theta_formal = max(theta_formal, 0.001)
            theta_informal = max(theta_informal, 0.001)
            
            # Compute additional steady-state variables
            e_formal = 1 - u_formal
            e_informal = 1 - u_informal
            
            f_formal = self._job_finding_rate(theta_formal, 'formal')
            f_informal = self._job_finding_rate(theta_informal, 'informal')
            
            q_formal = self._job_filling_rate(theta_formal, 'formal')
            q_informal = self._job_filling_rate(theta_informal, 'informal')
            
            # Overall labor market indicators
            total_employment = (self.params.formal_sector_share * e_formal + 
                              (1 - self.params.formal_sector_share) * e_informal)
            total_unemployment = 1 - total_employment
            
            # Average wage
            avg_wage = (self.params.formal_sector_share * w_formal + 
                       (1 - self.params.formal_sector_share) * w_informal)
            
            steady_state = {
                'unemployment_formal': u_formal,
                'unemployment_informal': u_informal,
                'employment_formal': e_formal,
                'employment_informal': e_informal,
                'total_unemployment': total_unemployment,
                'total_employment': total_employment,
                'market_tightness_formal': theta_formal,
                'market_tightness_informal': theta_informal,
                'job_finding_rate_formal': f_formal,
                'job_finding_rate_informal': f_informal,
                'job_filling_rate_formal': q_formal,
                'job_filling_rate_informal': q_informal,
                'wage_formal': w_formal,
                'wage_informal': w_informal,
                'average_wage': avg_wage,
                'wage_gap': w_formal - w_informal
            }
            
            self.steady_state = steady_state
            
            logger.info(f"Steady state computed: Total unemployment = {total_unemployment:.3f}")
            logger.info(f"Formal sector: u = {u_formal:.3f}, w = {w_formal:.3f}")
            logger.info(f"Informal sector: u = {u_informal:.3f}, w = {w_informal:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to compute steady state: {str(e)}")
            # Fallback to approximate values
            steady_state = self._approximate_steady_state()
            self.steady_state = steady_state
        
        return steady_state
    
    def _approximate_steady_state(self) -> Dict:
        """
        Compute approximate steady state when exact solution fails
        """
        logger.info("Computing approximate steady state")
        
        # Simple approximations based on typical values
        u_formal = 0.04  # 4% unemployment in formal sector
        u_informal = 0.20  # 20% unemployment in informal sector
        
        theta_formal = 0.8
        theta_informal = 1.5
        
        w_formal = 0.9
        w_informal = 0.6
        
        e_formal = 1 - u_formal
        e_informal = 1 - u_informal
        
        total_employment = (self.params.formal_sector_share * e_formal + 
                          (1 - self.params.formal_sector_share) * e_informal)
        total_unemployment = 1 - total_employment
        
        return {
            'unemployment_formal': u_formal,
            'unemployment_informal': u_informal,
            'employment_formal': e_formal,
            'employment_informal': e_informal,
            'total_unemployment': total_unemployment,
            'total_employment': total_employment,
            'market_tightness_formal': theta_formal,
            'market_tightness_informal': theta_informal,
            'job_finding_rate_formal': 0.3,
            'job_finding_rate_informal': 0.5,
            'job_filling_rate_formal': 0.4,
            'job_filling_rate_informal': 0.3,
            'wage_formal': w_formal,
            'wage_informal': w_informal,
            'average_wage': (self.params.formal_sector_share * w_formal + 
                           (1 - self.params.formal_sector_share) * w_informal),
            'wage_gap': w_formal - w_informal
        }
    
    def _job_finding_rate(self, theta: float, sector: str) -> float:
        """
        Compute job finding rate from market tightness
        
        Args:
            theta: Market tightness (vacancies/unemployment)
            sector: 'formal' or 'informal'
            
        Returns:
            Job finding rate
        """
        if sector == 'formal':
            efficiency = self.params.formal_matching_efficiency
        else:
            efficiency = self.params.informal_matching_efficiency
        
        # Matching function: m = A * u^alpha * v^(1-alpha)
        # Job finding rate: f = m/u = A * theta^(1-alpha)
        alpha = self.params.matching_elasticity
        
        return efficiency * (theta ** (1 - alpha))
    
    def _job_filling_rate(self, theta: float, sector: str) -> float:
        """
        Compute job filling rate from market tightness
        
        Args:
            theta: Market tightness
            sector: 'formal' or 'informal'
            
        Returns:
            Job filling rate
        """
        if sector == 'formal':
            efficiency = self.params.formal_matching_efficiency
        else:
            efficiency = self.params.informal_matching_efficiency
        
        # Job filling rate: q = m/v = A * theta^(-alpha)
        alpha = self.params.matching_elasticity
        
        return efficiency * (theta ** (-alpha))
    
    def _reservation_productivity(self, wage: float, sector: str) -> float:
        """
        Compute reservation productivity for job destruction
        
        Args:
            wage: Wage in the sector
            sector: 'formal' or 'informal'
            
        Returns:
            Reservation productivity
        """
        # Simplified: reservation productivity equals wage plus adjustment
        if sector == 'formal':
            return wage - self.params.employment_protection
        else:
            return wage * 0.8  # Lower threshold in informal sector
    
    def _expected_productivity(self, threshold: float) -> float:
        """
        Compute expected productivity above threshold
        
        Args:
            threshold: Productivity threshold
            
        Returns:
            Expected productivity
        """
        # Assume normal distribution of productivity
        mean = self.params.productivity_mean
        std = self.params.productivity_std
        
        # Truncated normal expectation
        from scipy.stats import norm
        
        # Probability of productivity above threshold
        prob_above = 1 - norm.cdf(threshold, mean, std)
        
        if prob_above > 0.001:
            # Expected value conditional on being above threshold
            expected = mean + std * norm.pdf(threshold, mean, std) / prob_above
        else:
            expected = mean
        
        return max(expected, threshold)
    
    def _job_destruction_rate(self, threshold: float) -> float:
        """
        Compute job destruction rate based on productivity threshold
        
        Args:
            threshold: Productivity threshold
            
        Returns:
            Job destruction rate
        """
        # Exogenous destruction
        exogenous = self.params.job_destruction_rate
        
        # Endogenous destruction (productivity below threshold)
        from scipy.stats import norm
        mean = self.params.productivity_mean
        std = self.params.productivity_std
        
        endogenous = norm.cdf(threshold, mean, std) * 0.1  # Scale factor
        
        return exogenous + endogenous
    
    def simulate_model(self, periods: int = 200, n_simulations: int = 100,
                      shocks: Dict = None) -> pd.DataFrame:
        """
        Simulate the search and matching model
        
        Args:
            periods: Number of periods to simulate
            n_simulations: Number of simulation runs
            shocks: External shocks (optional)
            
        Returns:
            Simulation results DataFrame
        """
        logger.info(f"Simulating Search and Matching model for {periods} periods, {n_simulations} runs")
        
        if self.steady_state is None:
            self.compute_steady_state()
        
        all_simulations = []
        
        for sim in range(n_simulations):
            # Initialize state variables
            u_formal = self.steady_state['unemployment_formal']
            u_informal = self.steady_state['unemployment_informal']
            
            # Storage for this simulation
            sim_data = []
            
            # Generate shocks
            if shocks is None:
                productivity_shocks = self._generate_productivity_shocks(periods)
                demand_shocks = np.random.normal(0, self.params.business_cycle_volatility, periods)
            else:
                productivity_shocks = shocks.get('productivity', np.zeros(periods))
                demand_shocks = shocks.get('demand', np.zeros(periods))
            
            for t in range(periods):
                # Apply shocks
                current_productivity = self.params.productivity_mean * np.exp(productivity_shocks[t])
                demand_factor = 1 + demand_shocks[t]
                
                # Compute current period variables
                theta_formal = self._compute_market_tightness(u_formal, current_productivity, demand_factor, 'formal')
                theta_informal = self._compute_market_tightness(u_informal, current_productivity, demand_factor, 'informal')
                
                f_formal = self._job_finding_rate(theta_formal, 'formal')
                f_informal = self._job_finding_rate(theta_informal, 'informal')
                
                s_formal = self.params.job_destruction_rate * (1 + 0.5 * demand_shocks[t])
                s_informal = self.params.job_destruction_rate * 1.5 * (1 + 0.7 * demand_shocks[t])
                
                # Wage determination
                w_formal = self._compute_wage(current_productivity, theta_formal, 'formal')
                w_informal = self._compute_wage(current_productivity, theta_informal, 'informal')
                
                # Update unemployment
                u_formal_new = (s_formal * (1 - u_formal) + u_formal) / (1 + f_formal)
                u_informal_new = (s_informal * (1 - u_informal) + u_informal) / (1 + f_informal)
                
                # Ensure bounds
                u_formal = np.clip(u_formal_new, 0.001, 0.5)
                u_informal = np.clip(u_informal_new, 0.001, 0.8)
                
                # Store results
                sim_data.append({
                    'period': t,
                    'simulation': sim,
                    'unemployment_formal': u_formal,
                    'unemployment_informal': u_informal,
                    'employment_formal': 1 - u_formal,
                    'employment_informal': 1 - u_informal,
                    'total_unemployment': (self.params.formal_sector_share * u_formal + 
                                         (1 - self.params.formal_sector_share) * u_informal),
                    'market_tightness_formal': theta_formal,
                    'market_tightness_informal': theta_informal,
                    'job_finding_rate_formal': f_formal,
                    'job_finding_rate_informal': f_informal,
                    'job_destruction_rate_formal': s_formal,
                    'job_destruction_rate_informal': s_informal,
                    'wage_formal': w_formal,
                    'wage_informal': w_informal,
                    'productivity_shock': productivity_shocks[t],
                    'demand_shock': demand_shocks[t]
                })
            
            all_simulations.extend(sim_data)
        
        return pd.DataFrame(all_simulations)
    
    def _generate_productivity_shocks(self, periods: int) -> np.ndarray:
        """
        Generate AR(1) productivity shocks
        
        Args:
            periods: Number of periods
            
        Returns:
            Productivity shock series
        """
        shocks = np.zeros(periods)
        
        for t in range(1, periods):
            shocks[t] = (self.params.productivity_persistence * shocks[t-1] + 
                        np.random.normal(0, self.params.productivity_std))
        
        return shocks
    
    def _compute_market_tightness(self, unemployment: float, productivity: float, 
                                 demand_factor: float, sector: str) -> float:
        """
        Compute market tightness based on job creation condition
        
        Args:
            unemployment: Unemployment rate
            productivity: Current productivity
            demand_factor: Demand shock factor
            sector: 'formal' or 'informal'
            
        Returns:
            Market tightness
        """
        # Simplified job creation condition
        base_tightness = productivity * demand_factor / self.params.job_creation_cost
        
        if sector == 'formal':
            return base_tightness * 0.8  # Lower tightness in formal sector
        else:
            return base_tightness * 1.2  # Higher tightness in informal sector
    
    def _compute_wage(self, productivity: float, theta: float, sector: str) -> float:
        """
        Compute wage based on Nash bargaining
        
        Args:
            productivity: Current productivity
            theta: Market tightness
            sector: 'formal' or 'informal'
            
        Returns:
            Wage
        """
        beta = self.params.worker_bargaining_power
        b = self.params.unemployment_benefit
        
        if sector == 'formal':
            # Formal sector wage
            outside_option = b + beta * theta * self.params.job_creation_cost
            wage = beta * productivity + (1 - beta) * outside_option
            
            # Apply minimum wage constraint
            min_wage = self.params.minimum_wage * self.params.productivity_mean
            wage = max(wage, min_wage)
            
        else:
            # Informal sector wage (lower bargaining power, no minimum wage)
            outside_option = b * 0.5  # Lower benefits in informal sector
            wage = beta * 0.7 * productivity * self.params.informal_sector_productivity + (1 - beta) * outside_option
        
        return wage
    
    def analyze_policy_intervention(self, policy_changes: Dict) -> Dict:
        """
        Analyze the impact of labor market policy interventions
        
        Args:
            policy_changes: Dictionary of policy parameter changes
            
        Returns:
            Policy analysis results
        """
        logger.info("Analyzing labor market policy interventions")
        
        # Baseline equilibrium
        baseline = self.compute_steady_state()
        
        # Store original parameters
        original_params = {}
        for param, value in policy_changes.items():
            if hasattr(self.params, param):
                original_params[param] = getattr(self.params, param)
                setattr(self.params, param, value)
        
        # Compute new equilibrium
        new_equilibrium = self.compute_steady_state()
        
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self.params, param, value)
        
        # Analyze changes
        policy_impact = {
            'baseline': baseline,
            'new_equilibrium': new_equilibrium,
            'changes': {},
            'policy_changes': policy_changes
        }
        
        # Calculate impacts
        key_indicators = ['total_unemployment', 'average_wage', 'wage_gap', 
                         'job_finding_rate_formal', 'job_finding_rate_informal']
        
        for indicator in key_indicators:
            if indicator in baseline and indicator in new_equilibrium:
                baseline_val = baseline[indicator]
                new_val = new_equilibrium[indicator]
                
                absolute_change = new_val - baseline_val
                relative_change = (absolute_change / baseline_val) * 100 if baseline_val != 0 else 0
                
                policy_impact['changes'][indicator] = {
                    'absolute': absolute_change,
                    'relative': relative_change,
                    'baseline': baseline_val,
                    'new': new_val
                }
        
        return policy_impact
    
    def compute_labor_market_indicators(self, simulation_data: pd.DataFrame) -> Dict:
        """
        Compute comprehensive labor market indicators
        
        Args:
            simulation_data: Simulation results
            
        Returns:
            Labor market indicators
        """
        logger.info("Computing labor market indicators")
        
        indicators = {}
        
        # Aggregate statistics
        for var in ['total_unemployment', 'wage_formal', 'wage_informal', 
                   'job_finding_rate_formal', 'job_finding_rate_informal']:
            if var in simulation_data.columns:
                indicators[f'{var}_mean'] = simulation_data[var].mean()
                indicators[f'{var}_std'] = simulation_data[var].std()
                indicators[f'{var}_cv'] = simulation_data[var].std() / simulation_data[var].mean()
        
        # Unemployment duration
        if 'job_finding_rate_formal' in simulation_data.columns:
            avg_finding_rate = simulation_data['job_finding_rate_formal'].mean()
            indicators['unemployment_duration_formal'] = 1 / avg_finding_rate if avg_finding_rate > 0 else np.inf
        
        if 'job_finding_rate_informal' in simulation_data.columns:
            avg_finding_rate = simulation_data['job_finding_rate_informal'].mean()
            indicators['unemployment_duration_informal'] = 1 / avg_finding_rate if avg_finding_rate > 0 else np.inf
        
        # Labor market flows
        if 'job_finding_rate_formal' in simulation_data.columns and 'unemployment_formal' in simulation_data.columns:
            job_finding_flow = (simulation_data['job_finding_rate_formal'] * 
                               simulation_data['unemployment_formal']).mean()
            indicators['job_finding_flow_formal'] = job_finding_flow
        
        # Wage inequality
        if 'wage_formal' in simulation_data.columns and 'wage_informal' in simulation_data.columns:
            wage_ratio = simulation_data['wage_formal'] / simulation_data['wage_informal']
            indicators['wage_ratio_formal_informal'] = wage_ratio.mean()
            indicators['wage_ratio_std'] = wage_ratio.std()
        
        # Business cycle correlations
        if 'demand_shock' in simulation_data.columns:
            for var in ['total_unemployment', 'wage_formal', 'job_finding_rate_formal']:
                if var in simulation_data.columns:
                    corr = simulation_data[var].corr(simulation_data['demand_shock'])
                    indicators[f'{var}_demand_correlation'] = corr
        
        return indicators
    
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
            variables = ['total_unemployment', 'wage_formal', 'wage_informal', 'job_finding_rate_formal']
        
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
                    ax.plot(sim_data['period'], sim_data[var], alpha=0.7, linewidth=1)
            
            # Add mean across simulations
            mean_data = simulation_data.groupby('period')[var].mean()
            ax.plot(mean_data.index, mean_data.values, color='red', linewidth=2, label='Mean')
            
            ax.set_title(f'{var.replace("_", " ").title()}')
            ax.set_xlabel('Period')
            ax.set_ylabel(var.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simulation plot saved to {save_path}")
        
        plt.show()
    
    def plot_beveridge_curve(self, simulation_data: pd.DataFrame, save_path: str = None):
        """
        Plot Beveridge curve (unemployment vs. vacancies relationship)
        
        Args:
            simulation_data: Simulation data
            save_path: Path to save plot
        """
        if 'total_unemployment' not in simulation_data.columns or 'market_tightness_formal' not in simulation_data.columns:
            logger.warning("Required data not available for Beveridge curve")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Formal sector Beveridge curve
        if 'unemployment_formal' in simulation_data.columns:
            ax1.scatter(simulation_data['unemployment_formal'], 
                       simulation_data['market_tightness_formal'],
                       alpha=0.6, s=20)
            ax1.set_xlabel('Unemployment Rate (Formal)')
            ax1.set_ylabel('Market Tightness (Formal)')
            ax1.set_title('Beveridge Curve - Formal Sector')
            ax1.grid(True, alpha=0.3)
        
        # Informal sector Beveridge curve
        if 'unemployment_informal' in simulation_data.columns and 'market_tightness_informal' in simulation_data.columns:
            ax2.scatter(simulation_data['unemployment_informal'], 
                       simulation_data['market_tightness_informal'],
                       alpha=0.6, s=20, color='orange')
            ax2.set_xlabel('Unemployment Rate (Informal)')
            ax2.set_ylabel('Market Tightness (Informal)')
            ax2.set_title('Beveridge Curve - Informal Sector')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Beveridge curve plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Search and Matching Model',
            'country': 'Bangladesh',
            'sectors': ['Formal', 'Informal'],
            'parameters': {
                'matching_efficiency': self.params.matching_efficiency,
                'matching_elasticity': self.params.matching_elasticity,
                'worker_bargaining_power': self.params.worker_bargaining_power,
                'job_destruction_rate': self.params.job_destruction_rate,
                'formal_sector_share': self.params.formal_sector_share,
                'unemployment_benefit': self.params.unemployment_benefit
            }
        }
        
        if self.steady_state is not None:
            summary['steady_state'] = {
                'total_unemployment_rate': self.steady_state['total_unemployment'],
                'formal_unemployment_rate': self.steady_state['unemployment_formal'],
                'informal_unemployment_rate': self.steady_state['unemployment_informal'],
                'wage_gap': self.steady_state['wage_gap'],
                'average_wage': self.steady_state['average_wage']
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'matching_efficiency': 0.4,
            'matching_elasticity': 0.6,
            'worker_bargaining_power': 0.3,
            'job_destruction_rate': 0.08,
            'formal_sector_share': 0.13,
            'unemployment_benefit': 0.1
        }
    }
    
    # Initialize Search and Matching model
    sm_model = SearchMatchingModel(config)
    
    # Calibrate model
    calibration = sm_model.calibrate_model()
    print(f"Model calibrated with parameters: {list(calibration.keys())}")
    
    # Compute steady state
    steady_state = sm_model.compute_steady_state()
    print(f"\nSteady State Results:")
    print(f"  Total unemployment rate: {steady_state['total_unemployment']:.3f}")
    print(f"  Formal sector unemployment: {steady_state['unemployment_formal']:.3f}")
    print(f"  Informal sector unemployment: {steady_state['unemployment_informal']:.3f}")
    print(f"  Wage gap (formal-informal): {steady_state['wage_gap']:.3f}")
    print(f"  Average wage: {steady_state['average_wage']:.3f}")
    
    # Simulate model
    print("\nSimulating model...")
    simulation_data = sm_model.simulate_model(periods=100, n_simulations=10)
    print(f"Simulation completed: {len(simulation_data)} observations")
    
    # Labor market indicators
    indicators = sm_model.compute_labor_market_indicators(simulation_data)
    print(f"\nLabor Market Indicators:")
    
    key_indicators = ['total_unemployment_mean', 'wage_formal_mean', 'wage_informal_mean', 
                     'unemployment_duration_formal', 'wage_ratio_formal_informal']
    
    for indicator in key_indicators:
        if indicator in indicators:
            print(f"  {indicator}: {indicators[indicator]:.3f}")
    
    # Policy analysis
    print("\nAnalyzing policy interventions...")
    
    # Increase unemployment benefits
    policy_changes = {
        'unemployment_benefit': 0.2,  # Increase from 0.1 to 0.2
        'active_labor_policy': 0.1    # Introduce active labor market policies
    }
    
    policy_impact = sm_model.analyze_policy_intervention(policy_changes)
    
    print("Policy Impact:")
    for indicator, change in policy_impact['changes'].items():
        print(f"  {indicator}: {change['baseline']:.3f} → {change['new']:.3f} ({change['relative']:+.1f}%)")
    
    # Model summary
    summary = sm_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Sectors: {summary['sectors']}")
    print(f"  Formal sector share: {summary['parameters']['formal_sector_share']:.1%}")
    print(f"  Steady-state unemployment: {summary['steady_state']['total_unemployment_rate']:.1%}")