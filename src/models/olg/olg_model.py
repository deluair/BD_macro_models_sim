#!/usr/bin/env python3
"""
Overlapping Generations (OLG) Life Cycle Model for Bangladesh

This module implements an OLG model to analyze demographic transitions,
intergenerational dynamics, and long-term economic development in Bangladesh.
The model captures life cycle behavior, savings decisions, human capital
accumulation, and the effects of population aging.

Key Features:
- Multi-period life cycle optimization
- Demographic transition dynamics
- Human capital accumulation
- Pension and social security systems
- Intergenerational transfers
- Labor market dynamics across age groups
- Healthcare and education investments

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, interpolate
from scipy.linalg import solve, inv
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
class OLGParameters:
    """
    Parameters for the OLG model
    """
    # Demographics
    max_age: int = 80                    # Maximum age
    retirement_age: int = 60             # Retirement age
    working_age_start: int = 20          # Start of working life
    
    # Preferences
    beta: float = 0.96                   # Discount factor
    sigma: float = 2.0                   # Risk aversion (CRRA)
    gamma: float = 0.3                   # Leisure preference
    
    # Technology
    alpha: float = 0.35                  # Capital share
    delta: float = 0.06                  # Depreciation rate
    A: float = 1.0                       # Total factor productivity
    
    # Human capital
    phi: float = 0.8                     # Human capital persistence
    eta_education: float = 0.15          # Education productivity
    eta_experience: float = 0.05         # Experience premium
    
    # Demographics parameters
    birth_rate: float = 0.018            # Birth rate
    death_rate: float = 0.006            # Death rate
    migration_rate: float = 0.002        # Net migration rate
    
    # Social security
    pension_replacement: float = 0.4     # Pension replacement rate
    social_security_tax: float = 0.08    # Social security tax rate
    
    # Government
    gov_spending_gdp: float = 0.15       # Government spending (% GDP)
    tax_rate_labor: float = 0.15         # Labor income tax rate
    tax_rate_capital: float = 0.25       # Capital income tax rate
    
    # Bangladesh-specific
    remittances_gdp: float = 0.06        # Remittances (% GDP)
    informal_sector_share: float = 0.85  # Informal sector employment
    rural_population_share: float = 0.62 # Rural population share

@dataclass
class OLGResults:
    """
    Results from OLG model simulation
    """
    parameters: OLGParameters
    steady_state: Dict
    transition_dynamics: Optional[pd.DataFrame] = None
    policy_experiments: Optional[Dict] = None
    welfare_analysis: Optional[Dict] = None
    demographic_projections: Optional[pd.DataFrame] = None

class OverlappingGenerationsModel:
    """
    Overlapping Generations Model for Bangladesh
    
    This class implements a comprehensive OLG model that captures
    demographic transitions, life cycle behavior, and intergenerational
    dynamics in the Bangladesh economy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize OLG model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = OLGParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Model state
        self.steady_state = None
        self.results = None
        
        # Age structure
        self.ages = np.arange(self.params.working_age_start, self.params.max_age + 1)
        self.n_ages = len(self.ages)
        self.working_ages = np.arange(self.params.working_age_start, self.params.retirement_age)
        self.retired_ages = np.arange(self.params.retirement_age, self.params.max_age + 1)
        
        # Demographic data
        self.population_data = None
        self.survival_probabilities = None
        
        # Economic variables
        self.wage_profile = None
        self.productivity_profile = None
        
        logger.info("OLG model initialized for Bangladesh")
    
    def load_demographic_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load demographic data for Bangladesh
        
        Args:
            data: Demographic data (optional, will create synthetic if None)
            
        Returns:
            Processed demographic data
        """
        logger.info("Loading demographic data")
        
        if data is None:
            # Create synthetic demographic data for Bangladesh
            data = self._create_synthetic_demographics()
        
        # Process demographic data
        self.population_data = self._process_demographic_data(data)
        
        # Compute survival probabilities
        self.survival_probabilities = self._compute_survival_probabilities()
        
        logger.info(f"Demographic data loaded for {len(self.ages)} age groups")
        return self.population_data
    
    def _create_synthetic_demographics(self) -> pd.DataFrame:
        """
        Create synthetic demographic data for Bangladesh
        """
        # Bangladesh demographic profile (approximate)
        age_groups = np.arange(0, 81, 5)
        
        # Population distribution (millions)
        population_2020 = np.array([
            16.5, 15.8, 14.2, 12.8, 11.5, 10.2, 8.9, 7.6, 6.3, 5.1,
            4.0, 3.1, 2.4, 1.8, 1.3, 0.9, 0.6
        ])
        
        # Mortality rates (per 1000)
        mortality_rates = np.array([
            35, 2, 1, 1, 1, 2, 3, 4, 6, 9,
            14, 22, 35, 55, 85, 130, 200
        ]) / 1000
        
        # Birth rates by age (for women)
        fertility_rates = np.array([
            0, 0, 0.05, 0.15, 0.18, 0.12, 0.08, 0.03, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ])
        
        # Create DataFrame
        demo_data = pd.DataFrame({
            'age_group': age_groups,
            'population_2020': population_2020,
            'mortality_rate': mortality_rates,
            'fertility_rate': fertility_rates
        })
        
        # Add projections for 2025, 2030, etc.
        years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
        
        # Simple demographic projection
        for i, year in enumerate(years[1:], 1):
            # Aging and mortality effects
            pop_projection = population_2020 * (1 - mortality_rates) ** (5 * i)
            
            # Declining fertility
            fertility_decline = 0.98 ** i
            
            demo_data[f'population_{year}'] = pop_projection
            demo_data[f'fertility_{year}'] = fertility_rates * fertility_decline
        
        return demo_data
    
    def _process_demographic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process demographic data for model use
        """
        # Interpolate to single-year age groups
        processed_data = pd.DataFrame()
        processed_data['age'] = self.ages
        
        # Interpolate population data
        for col in data.columns:
            if col.startswith('population_'):
                interp_func = interpolate.interp1d(
                    data['age_group'], data[col], 
                    kind='linear', fill_value='extrapolate'
                )
                processed_data[col] = interp_func(self.ages)
        
        # Interpolate mortality rates
        if 'mortality_rate' in data.columns:
            interp_func = interpolate.interp1d(
                data['age_group'], data['mortality_rate'],
                kind='linear', fill_value='extrapolate'
            )
            processed_data['mortality_rate'] = interp_func(self.ages)
        
        return processed_data
    
    def _compute_survival_probabilities(self) -> np.ndarray:
        """
        Compute survival probabilities by age
        """
        if 'mortality_rate' in self.population_data.columns:
            mortality = self.population_data['mortality_rate'].values
            survival = 1 - mortality
        else:
            # Default survival probabilities
            survival = np.ones(self.n_ages)
            # Declining survival with age
            for i, age in enumerate(self.ages):
                if age < 30:
                    survival[i] = 0.999
                elif age < 50:
                    survival[i] = 0.998
                elif age < 65:
                    survival[i] = 0.995
                elif age < 75:
                    survival[i] = 0.985
                else:
                    survival[i] = 0.95
        
        return survival
    
    def setup_economy(self):
        """
        Set up the economic environment
        """
        logger.info("Setting up economic environment")
        
        # Create wage and productivity profiles
        self._create_wage_profile()
        self._create_productivity_profile()
        
        # Initialize population distribution
        self._initialize_population()
        
        logger.info("Economic environment setup completed")
    
    def _create_wage_profile(self):
        """
        Create age-wage profile for Bangladesh
        """
        # Typical wage profile: low at start, peak in 40s-50s, decline after
        wage_profile = np.zeros(self.n_ages)
        
        for i, age in enumerate(self.ages):
            if age < self.params.retirement_age:
                # Experience premium
                experience = age - self.params.working_age_start
                experience_premium = self.params.eta_experience * experience
                
                # Quadratic age profile
                age_effect = 0.05 * (age - 20) - 0.0005 * (age - 20) ** 2
                
                # Base wage with experience and age effects
                wage_profile[i] = 1.0 + experience_premium + age_effect
            else:
                # Retirement: pension income
                wage_profile[i] = self.params.pension_replacement
        
        # Ensure positive wages
        wage_profile = np.maximum(wage_profile, 0.1)
        
        self.wage_profile = wage_profile
    
    def _create_productivity_profile(self):
        """
        Create age-productivity profile
        """
        # Productivity profile: increases with experience, peaks, then declines
        productivity_profile = np.zeros(self.n_ages)
        
        for i, age in enumerate(self.ages):
            if age < self.params.retirement_age:
                # Experience effect
                experience = age - self.params.working_age_start
                
                # Inverted U-shape productivity
                productivity_profile[i] = (
                    1.0 + 0.03 * experience - 0.0003 * experience ** 2
                )
            else:
                productivity_profile[i] = 0  # No productivity in retirement
        
        # Ensure non-negative productivity
        productivity_profile = np.maximum(productivity_profile, 0)
        
        self.productivity_profile = productivity_profile
    
    def _initialize_population(self):
        """
        Initialize population distribution
        """
        if self.population_data is not None and 'population_2020' in self.population_data.columns:
            # Use actual data
            self.population_dist = self.population_data['population_2020'].values
        else:
            # Create synthetic population distribution
            # Declining population with age (typical for developing countries)
            self.population_dist = np.exp(-0.02 * (self.ages - self.params.working_age_start))
        
        # Normalize to sum to 1
        self.population_dist = self.population_dist / np.sum(self.population_dist)
    
    def solve_household_problem(self, prices: Dict) -> Dict:
        """
        Solve the household optimization problem
        
        Args:
            prices: Dictionary with factor prices (wage, interest rate)
            
        Returns:
            Household decision rules
        """
        logger.info("Solving household optimization problem")
        
        wage = prices['wage']
        interest_rate = prices['interest_rate']
        
        # Create asset grid
        n_assets = 50
        asset_max = 20.0
        asset_grid = np.linspace(0, asset_max, n_assets)
        
        # Initialize value functions and policy functions
        value_function = np.zeros((self.n_ages, n_assets))
        consumption_policy = np.zeros((self.n_ages, n_assets))
        savings_policy = np.zeros((self.n_ages, n_assets))
        labor_supply_policy = np.zeros((self.n_ages, n_assets))
        
        # Solve backwards from retirement
        for i in reversed(range(self.n_ages)):
            age = self.ages[i]
            
            for j, assets in enumerate(asset_grid):
                if age >= self.params.retirement_age:
                    # Retirement period
                    result = self._solve_retirement_period(i, j, assets, value_function, 
                                                         asset_grid, wage, interest_rate)
                else:
                    # Working period
                    result = self._solve_working_period(i, j, assets, value_function, 
                                                      asset_grid, wage, interest_rate)
                
                value_function[i, j] = result['value']
                consumption_policy[i, j] = result['consumption']
                savings_policy[i, j] = result['savings']
                labor_supply_policy[i, j] = result['labor_supply']
        
        # Aggregate policies using population distribution
        # For simplicity, assume uniform distribution over asset grid
        agg_consumption = np.mean(consumption_policy, axis=1)
        agg_savings = np.mean(savings_policy, axis=1)
        agg_labor_supply = np.mean(labor_supply_policy, axis=1)
        
        household_solution = {
            'value_function': value_function,
            'consumption_policy': agg_consumption,
            'savings_policy': agg_savings,
            'labor_supply_policy': agg_labor_supply,
            'asset_grid': asset_grid
        }
        
        return household_solution
    
    def _solve_retirement_period(self, age_index: int, asset_index: int, assets_current: float,
                               value_function: np.ndarray, asset_grid: np.ndarray,
                               wage: float, interest_rate: float) -> Dict:
        """
        Solve optimization for retirement period
        """
        age = self.ages[age_index]
        
        def objective(assets_next):
            # Pension income
            pension_income = self.params.pension_replacement * wage * self.wage_profile[age_index]
            
            # Budget constraint
            consumption = assets_current * (1 + interest_rate) + pension_income - assets_next
            
            if consumption <= 0:
                return -1e10  # Penalty for negative consumption
            
            # Utility from consumption
            utility = self._utility(consumption, 0)  # No labor in retirement
            
            # Continuation value
            if age_index < self.n_ages - 1:
                survival_prob = self.survival_probabilities[age_index]
                # Interpolate value function for next period
                continuation_value = np.interp(assets_next, asset_grid, 
                                             value_function[age_index + 1, :])
                continuation = self.params.beta * survival_prob * continuation_value
            else:
                continuation = 0
            
            return -(utility + continuation)  # Negative for minimization
        
        # Optimize over next period assets
        result = optimize.minimize_scalar(objective, bounds=(0, asset_grid[-1]), method='bounded')
        
        optimal_assets_next = result.x
        optimal_value = -result.fun
        
        # Compute optimal consumption
        pension_income = self.params.pension_replacement * wage * self.wage_profile[age_index]
        optimal_consumption = assets_current * (1 + interest_rate) + pension_income - optimal_assets_next
        
        return {
            'value': optimal_value,
            'consumption': max(optimal_consumption, 0.01),  # Ensure positive consumption
            'savings': optimal_assets_next,
            'labor_supply': 0  # No labor in retirement
        }
    
    def _solve_working_period(self, age_index: int, asset_index: int, assets_current: float,
                            value_function: np.ndarray, asset_grid: np.ndarray,
                            wage: float, interest_rate: float) -> Dict:
        """
        Solve optimization for working period
        """
        age = self.ages[age_index]
        
        def objective(choices):
            assets_next, labor_supply = choices
            
            # Labor income
            effective_wage = wage * self.wage_profile[age_index] * self.productivity_profile[age_index]
            labor_income = effective_wage * labor_supply
            
            # Taxes
            labor_tax = self.params.tax_rate_labor * labor_income
            ss_tax = self.params.social_security_tax * labor_income
            
            # Budget constraint
            total_income = assets_current * (1 + interest_rate) + labor_income - labor_tax - ss_tax
            consumption = total_income - assets_next
            
            if consumption <= 0 or labor_supply < 0 or labor_supply > 1:
                return 1e10  # Penalty
            
            # Utility
            utility = self._utility(consumption, labor_supply)
            
            # Continuation value
            if age_index < self.n_ages - 1:
                survival_prob = self.survival_probabilities[age_index]
                # Interpolate value function for next period
                continuation_value = np.interp(assets_next, asset_grid, 
                                             value_function[age_index + 1, :])
                continuation = self.params.beta * survival_prob * continuation_value
            else:
                continuation = 0
            
            return -(utility + continuation)
        
        # Optimize over assets and labor supply
        bounds = [(0, asset_grid[-1]), (0, 1)]  # Assets and labor supply bounds
        initial_guess = [min(assets_current * 1.1, asset_grid[-1] * 0.5), 0.4]
        
        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimal_assets_next, optimal_labor = result.x
                optimal_value = -result.fun
            else:
                # Fallback to simple choices if optimization fails
                optimal_assets_next = min(assets_current * 0.9, asset_grid[-1] * 0.3)
                optimal_labor = 0.4
                optimal_value = objective([optimal_assets_next, optimal_labor])
                optimal_value = -optimal_value
        except:
            # Fallback for any optimization errors
            optimal_assets_next = min(assets_current * 0.9, asset_grid[-1] * 0.3)
            optimal_labor = 0.4
            optimal_value = -1e6
        
        # Compute optimal consumption
        effective_wage = wage * self.wage_profile[age_index] * self.productivity_profile[age_index]
        labor_income = effective_wage * optimal_labor
        labor_tax = self.params.tax_rate_labor * labor_income
        ss_tax = self.params.social_security_tax * labor_income
        
        total_income = assets_current * (1 + interest_rate) + labor_income - labor_tax - ss_tax
        optimal_consumption = total_income - optimal_assets_next
        
        return {
            'value': optimal_value,
            'consumption': max(optimal_consumption, 0.01),  # Ensure positive consumption
            'savings': optimal_assets_next,
            'labor_supply': max(min(optimal_labor, 1.0), 0.0)  # Ensure valid labor supply
        }
    
    def _utility(self, consumption: float, labor_supply: float) -> float:
        """
        Utility function (CRRA with leisure)
        
        U(c, l) = (c^(1-σ) - 1)/(1-σ) + γ * ((1-l)^(1-σ) - 1)/(1-σ)
        """
        if consumption <= 0:
            return -1e10
        
        leisure = 1 - labor_supply
        
        if self.params.sigma == 1:
            # Log utility
            utility_c = np.log(consumption)
            utility_l = self.params.gamma * np.log(leisure) if leisure > 0 else -1e10
        else:
            # CRRA utility
            utility_c = (consumption ** (1 - self.params.sigma) - 1) / (1 - self.params.sigma)
            utility_l = self.params.gamma * (leisure ** (1 - self.params.sigma) - 1) / (1 - self.params.sigma) if leisure > 0 else -1e10
        
        return utility_c + utility_l
    
    def solve_firm_problem(self, prices: Dict) -> Dict:
        """
        Solve the firm's profit maximization problem
        
        Args:
            prices: Dictionary with factor prices
            
        Returns:
            Firm's optimal choices
        """
        wage = prices['wage']
        interest_rate = prices['interest_rate']
        
        # Aggregate labor (efficiency units)
        total_labor = np.sum(self.population_dist * self.productivity_profile)
        
        # Capital demand from FOC: MPK = r + δ
        capital_demand = ((self.params.alpha * self.params.A) / (interest_rate + self.params.delta)) ** (1 / (1 - self.params.alpha)) * total_labor
        
        # Labor demand from FOC: MPL = w
        # This is satisfied by equilibrium wage
        
        # Output
        output = self.params.A * (capital_demand ** self.params.alpha) * (total_labor ** (1 - self.params.alpha))
        
        return {
            'capital_demand': capital_demand,
            'labor_demand': total_labor,
            'output': output
        }
    
    def solve_government_budget(self, household_solution: Dict, firm_solution: Dict, prices: Dict) -> Dict:
        """
        Solve government budget constraint
        
        Args:
            household_solution: Household optimization results
            firm_solution: Firm optimization results
            prices: Factor prices
            
        Returns:
            Government budget analysis
        """
        wage = prices['wage']
        interest_rate = prices['interest_rate']
        
        # Tax revenues
        labor_income = np.sum(
            self.population_dist * 
            household_solution['labor_supply_policy'] * 
            wage * self.wage_profile * self.productivity_profile
        )
        
        labor_tax_revenue = self.params.tax_rate_labor * labor_income
        ss_tax_revenue = self.params.social_security_tax * labor_income
        
        # Capital tax revenue (simplified)
        capital_tax_revenue = self.params.tax_rate_capital * interest_rate * firm_solution['capital_demand']
        
        total_tax_revenue = labor_tax_revenue + ss_tax_revenue + capital_tax_revenue
        
        # Government spending
        government_spending = self.params.gov_spending_gdp * firm_solution['output']
        
        # Pension payments
        retired_population = np.sum(self.population_dist[self.ages >= self.params.retirement_age])
        pension_payments = retired_population * self.params.pension_replacement * wage
        
        # Budget balance
        budget_balance = total_tax_revenue - government_spending - pension_payments
        
        return {
            'tax_revenue': total_tax_revenue,
            'labor_tax_revenue': labor_tax_revenue,
            'ss_tax_revenue': ss_tax_revenue,
            'capital_tax_revenue': capital_tax_revenue,
            'government_spending': government_spending,
            'pension_payments': pension_payments,
            'budget_balance': budget_balance
        }
    
    def find_equilibrium(self, initial_guess: Dict = None) -> Dict:
        """
        Find general equilibrium
        
        Args:
            initial_guess: Initial guess for prices
            
        Returns:
            Equilibrium prices and allocations
        """
        logger.info("Finding general equilibrium")
        
        if initial_guess is None:
            initial_guess = {'wage': 1.0, 'interest_rate': 0.04}
        
        def equilibrium_conditions(prices_vec):
            wage, interest_rate = prices_vec
            prices = {'wage': wage, 'interest_rate': interest_rate}
            
            # Solve household problem
            household_solution = self.solve_household_problem(prices)
            
            # Solve firm problem
            firm_solution = self.solve_firm_problem(prices)
            
            # Market clearing conditions
            
            # Labor market clearing
            labor_supply = np.sum(
                self.population_dist * household_solution['labor_supply_policy'] * self.productivity_profile
            )
            labor_demand = firm_solution['labor_demand']
            labor_excess = labor_supply - labor_demand
            
            # Capital market clearing
            capital_supply = np.sum(self.population_dist * household_solution['savings_policy'])
            capital_demand = firm_solution['capital_demand']
            capital_excess = capital_supply - capital_demand
            
            return [labor_excess, capital_excess]
        
        # Solve for equilibrium prices
        initial_prices = [initial_guess['wage'], initial_guess['interest_rate']]
        
        try:
            solution = optimize.root(equilibrium_conditions, initial_prices, method='hybr')
            
            if solution.success:
                equilibrium_wage, equilibrium_interest_rate = solution.x
                
                equilibrium_prices = {
                    'wage': equilibrium_wage,
                    'interest_rate': equilibrium_interest_rate
                }
                
                # Compute equilibrium allocations
                household_solution = self.solve_household_problem(equilibrium_prices)
                firm_solution = self.solve_firm_problem(equilibrium_prices)
                government_solution = self.solve_government_budget(household_solution, firm_solution, equilibrium_prices)
                
                equilibrium = {
                    'prices': equilibrium_prices,
                    'household': household_solution,
                    'firm': firm_solution,
                    'government': government_solution,
                    'convergence': True
                }
                
                logger.info(f"Equilibrium found: wage={equilibrium_wage:.4f}, r={equilibrium_interest_rate:.4f}")
                
            else:
                logger.warning("Equilibrium not found, using initial guess")
                equilibrium = {
                    'prices': initial_guess,
                    'convergence': False
                }
        
        except Exception as e:
            logger.error(f"Error finding equilibrium: {str(e)}")
            equilibrium = {
                'prices': initial_guess,
                'convergence': False,
                'error': str(e)
            }
        
        self.steady_state = equilibrium
        return equilibrium
    
    def simulate_transition(self, initial_state: Dict, final_state: Dict, 
                          periods: int = 50) -> pd.DataFrame:
        """
        Simulate transition dynamics between steady states
        
        Args:
            initial_state: Initial steady state
            final_state: Final steady state
            periods: Number of transition periods
            
        Returns:
            Transition path DataFrame
        """
        logger.info(f"Simulating {periods}-period transition")
        
        # Initialize transition path
        transition_data = []
        
        for t in range(periods):
            # Linear interpolation between initial and final states
            weight = t / (periods - 1)
            
            # Interpolate key variables
            current_state = {}
            for key in initial_state['prices']:
                current_state[key] = (
                    (1 - weight) * initial_state['prices'][key] + 
                    weight * final_state['prices'][key]
                )
            
            # Solve for current period
            household_solution = self.solve_household_problem(current_state)
            firm_solution = self.solve_firm_problem(current_state)
            
            # Store results
            period_data = {
                'period': t,
                'wage': current_state['wage'],
                'interest_rate': current_state['interest_rate'],
                'output': firm_solution['output'],
                'capital': firm_solution['capital_demand'],
                'labor': firm_solution['labor_demand'],
                'consumption': np.sum(self.population_dist * household_solution['consumption_policy']),
                'savings': np.sum(self.population_dist * household_solution['savings_policy'])
            }
            
            transition_data.append(period_data)
        
        transition_df = pd.DataFrame(transition_data)
        return transition_df
    
    def demographic_transition_analysis(self, scenarios: Dict) -> Dict:
        """
        Analyze effects of demographic transition
        
        Args:
            scenarios: Different demographic scenarios
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing demographic transition effects")
        
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Analyzing scenario: {scenario_name}")
            
            # Update demographic parameters
            original_params = {
                'birth_rate': self.params.birth_rate,
                'death_rate': self.params.death_rate,
                'retirement_age': self.params.retirement_age
            }
            
            # Apply scenario changes
            for param, value in scenario_params.items():
                if hasattr(self.params, param):
                    setattr(self.params, param, value)
            
            # Recompute population distribution and survival probabilities
            self._initialize_population()
            self.survival_probabilities = self._compute_survival_probabilities()
            
            # Find new equilibrium
            scenario_equilibrium = self.find_equilibrium()
            
            # Store results
            results[scenario_name] = {
                'equilibrium': scenario_equilibrium,
                'demographic_params': scenario_params.copy()
            }
            
            # Restore original parameters
            for param, value in original_params.items():
                setattr(self.params, param, value)
        
        return results
    
    def policy_experiments(self, policies: Dict) -> Dict:
        """
        Conduct policy experiments
        
        Args:
            policies: Dictionary of policy scenarios
            
        Returns:
            Policy experiment results
        """
        logger.info("Conducting policy experiments")
        
        # Baseline equilibrium
        baseline = self.find_equilibrium()
        
        policy_results = {'baseline': baseline}
        
        for policy_name, policy_changes in policies.items():
            logger.info(f"Analyzing policy: {policy_name}")
            
            # Store original parameters
            original_params = {}
            for param in policy_changes:
                if hasattr(self.params, param):
                    original_params[param] = getattr(self.params, param)
            
            # Apply policy changes
            for param, value in policy_changes.items():
                if hasattr(self.params, param):
                    setattr(self.params, param, value)
            
            # Find new equilibrium
            policy_equilibrium = self.find_equilibrium()
            
            # Compute welfare effects
            welfare_change = self._compute_welfare_change(baseline, policy_equilibrium)
            
            policy_results[policy_name] = {
                'equilibrium': policy_equilibrium,
                'welfare_change': welfare_change,
                'policy_params': policy_changes.copy()
            }
            
            # Restore original parameters
            for param, value in original_params.items():
                setattr(self.params, param, value)
        
        return policy_results
    
    def _compute_welfare_change(self, baseline: Dict, policy: Dict) -> Dict:
        """
        Compute welfare change from policy
        """
        if not baseline.get('convergence', False) or not policy.get('convergence', False):
            return {'welfare_change': None, 'note': 'Equilibrium not found'}
        
        # Simplified welfare calculation
        baseline_welfare = np.sum(
            self.population_dist * baseline['household']['value_function']
        )
        
        policy_welfare = np.sum(
            self.population_dist * policy['household']['value_function']
        )
        
        welfare_change = (policy_welfare - baseline_welfare) / abs(baseline_welfare)
        
        return {
            'welfare_change_percent': welfare_change * 100,
            'baseline_welfare': baseline_welfare,
            'policy_welfare': policy_welfare
        }
    
    def plot_life_cycle_profiles(self, equilibrium: Dict = None, save_path: str = None):
        """
        Plot life cycle profiles
        
        Args:
            equilibrium: Equilibrium solution (uses steady state if None)
            save_path: Path to save plot
        """
        if equilibrium is None:
            equilibrium = self.steady_state
        
        if equilibrium is None or not equilibrium.get('convergence', False):
            logger.warning("No equilibrium solution available for plotting")
            return
        
        household_solution = equilibrium['household']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Consumption profile
        axes[0, 0].plot(self.ages, household_solution['consumption_policy'])
        axes[0, 0].set_title('Life Cycle Consumption')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Consumption')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Savings profile
        axes[0, 1].plot(self.ages, household_solution['savings_policy'])
        axes[0, 1].set_title('Life Cycle Savings')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Savings')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Labor supply profile
        axes[1, 0].plot(self.ages, household_solution['labor_supply_policy'])
        axes[1, 0].axvline(x=self.params.retirement_age, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Life Cycle Labor Supply')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Labor Supply')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Wage profile
        axes[1, 1].plot(self.ages, self.wage_profile * equilibrium['prices']['wage'])
        axes[1, 1].set_title('Age-Wage Profile')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Wage')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Life cycle profiles saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Overlapping Generations (OLG) Model',
            'country': 'Bangladesh',
            'age_range': f"{self.params.working_age_start}-{self.params.max_age}",
            'retirement_age': self.params.retirement_age,
            'number_of_age_groups': self.n_ages
        }
        
        if self.steady_state is not None:
            summary['steady_state'] = {
                'convergence': self.steady_state.get('convergence', False),
                'wage': self.steady_state.get('prices', {}).get('wage', None),
                'interest_rate': self.steady_state.get('prices', {}).get('interest_rate', None)
            }
            
            if 'firm' in self.steady_state:
                summary['steady_state']['output'] = self.steady_state['firm'].get('output', None)
                summary['steady_state']['capital'] = self.steady_state['firm'].get('capital_demand', None)
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'max_age': 80,
            'retirement_age': 60,
            'beta': 0.96,
            'sigma': 2.0,
            'alpha': 0.35
        }
    }
    
    # Initialize OLG model
    olg_model = OverlappingGenerationsModel(config)
    
    # Load demographic data
    demo_data = olg_model.load_demographic_data()
    print(f"Demographic data loaded: {demo_data.shape}")
    
    # Setup economy
    olg_model.setup_economy()
    print("Economic environment setup completed")
    
    # Find equilibrium
    equilibrium = olg_model.find_equilibrium()
    print(f"Equilibrium convergence: {equilibrium.get('convergence', False)}")
    
    if equilibrium.get('convergence', False):
        print(f"Equilibrium wage: {equilibrium['prices']['wage']:.4f}")
        print(f"Equilibrium interest rate: {equilibrium['prices']['interest_rate']:.4f}")
        print(f"Output: {equilibrium['firm']['output']:.4f}")
    
    # Policy experiments
    policies = {
        'pension_reform': {'pension_replacement': 0.5, 'retirement_age': 65},
        'tax_reform': {'tax_rate_labor': 0.10, 'tax_rate_capital': 0.20}
    }
    
    policy_results = olg_model.policy_experiments(policies)
    print(f"\nPolicy experiments completed for {len(policy_results)-1} policies")
    
    for policy_name, result in policy_results.items():
        if policy_name != 'baseline' and 'welfare_change' in result:
            welfare_change = result['welfare_change'].get('welfare_change_percent', 'N/A')
            print(f"  {policy_name}: Welfare change = {welfare_change}%")
    
    # Model summary
    summary = olg_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Age range: {summary['age_range']}")
    print(f"  Retirement age: {summary['retirement_age']}")
    print(f"  Convergence: {summary.get('steady_state', {}).get('convergence', 'N/A')}")