#!/usr/bin/env python3
"""
Integrated Assessment Model (IAM) for Bangladesh Economy

This module implements a comprehensive integrated assessment model that combines
economic, environmental, and climate components to analyze the interactions
between economic development and environmental sustainability in Bangladesh.

Key Features:
- Economic-climate interactions
- Environmental damage functions
- Adaptation and mitigation policies
- Sea level rise impacts
- Agricultural productivity effects
- Energy transition pathways
- Carbon pricing mechanisms
- Sustainable development scenarios
- Climate vulnerability assessment
- Green growth strategies

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.integrate import odeint
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import random
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IAMParameters:
    """
    Parameters for the Integrated Assessment Model
    """
    # Time parameters
    start_year: int = 2025                     # Start year
    end_year: int = 2100                       # End year
    time_step: float = 1.0                     # Time step in years
    
    # Economic parameters
    initial_gdp: float = 460e9                 # Initial GDP (USD, 2023)
    population_growth_rate: float = 0.01       # Annual population growth
    productivity_growth_rate: float = 0.025    # Annual productivity growth
    capital_depreciation: float = 0.05         # Capital depreciation rate
    savings_rate: float = 0.25                # National savings rate
    
    # Climate parameters
    climate_sensitivity: float = 3.0           # Climate sensitivity (°C per CO2 doubling)
    initial_temperature: float = 1.1           # Initial temperature increase (°C)
    initial_co2: float = 420                   # Initial CO2 concentration (ppm)
    pre_industrial_co2: float = 280            # Pre-industrial CO2 (ppm)
    
    # Damage function parameters
    damage_coefficient: float = 0.0023         # Quadratic damage coefficient
    damage_exponent: float = 2.0               # Damage function exponent
    sea_level_damage_coeff: float = 0.001      # Sea level rise damage coefficient
    
    # Bangladesh-specific parameters
    coastal_area_share: float = 0.32           # Share of coastal area
    agricultural_gdp_share: float = 0.13       # Agriculture share of GDP
    flood_vulnerability: float = 0.26          # Flood vulnerability index
    cyclone_frequency: float = 1.4             # Annual cyclone frequency
    
    # Adaptation parameters
    adaptation_cost_share: float = 0.02        # Adaptation cost as share of GDP
    adaptation_effectiveness: float = 0.6      # Adaptation effectiveness (0-1)
    
    # Mitigation parameters
    carbon_price_initial: float = 10           # Initial carbon price (USD/tCO2)
    carbon_price_growth: float = 0.05          # Annual carbon price growth
    mitigation_cost_coeff: float = 0.05        # Mitigation cost coefficient
    
    # Energy parameters
    energy_intensity_decline: float = 0.02     # Annual energy intensity decline
    renewable_share_target: float = 0.4        # Renewable energy target by 2041
    coal_share_initial: float = 0.35           # Initial coal share
    
    # Social parameters
    discount_rate: float = 0.03                # Social discount rate
    inequality_climate_factor: float = 1.5     # Climate impact inequality multiplier
    
    # Uncertainty parameters
    temperature_uncertainty: float = 0.5       # Temperature projection uncertainty
    damage_uncertainty: float = 0.3            # Damage function uncertainty
    economic_uncertainty: float = 0.1          # Economic growth uncertainty

@dataclass
class IAMResults:
    """
    Results from Integrated Assessment Model
    """
    parameters: IAMParameters
    time_series_data: Optional[pd.DataFrame] = None
    scenario_results: Optional[Dict] = None
    policy_analysis: Optional[Dict] = None
    uncertainty_analysis: Optional[Dict] = None
    welfare_analysis: Optional[Dict] = None

class ClimateModule:
    """
    Climate system component of the IAM
    """
    
    def __init__(self, params: IAMParameters):
        """
        Initialize climate module
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.temperature = params.initial_temperature
        self.co2_concentration = params.initial_co2
        self.sea_level_rise = 0.0
        
        # Climate system parameters
        self.thermal_inertia = 0.02  # Ocean thermal inertia
        self.carbon_cycle_decay = 0.3  # Carbon cycle decay rate
        
    def update_climate(self, emissions: float, year: int) -> Dict[str, float]:
        """
        Update climate state based on emissions
        
        Args:
            emissions: CO2 emissions (GtCO2)
            year: Current year
            
        Returns:
            Climate state variables
        """
        # Update CO2 concentration (simplified carbon cycle)
        co2_increase = emissions * 0.47  # Airborne fraction
        co2_decay = (self.co2_concentration - self.params.pre_industrial_co2) * self.carbon_cycle_decay * 0.01
        self.co2_concentration += co2_increase - co2_decay
        
        # Update temperature (simplified climate response)
        co2_ratio = self.co2_concentration / self.params.pre_industrial_co2
        equilibrium_temp = self.params.climate_sensitivity * np.log2(co2_ratio)
        
        # Temperature adjustment with thermal inertia
        temp_adjustment = self.thermal_inertia * (equilibrium_temp - self.temperature)
        self.temperature += temp_adjustment
        
        # Update sea level rise (simplified)
        thermal_expansion = self.temperature * 0.2  # 20cm per degree
        ice_melt = max(0, self.temperature - 1.5) * 0.5  # Accelerated after 1.5°C
        self.sea_level_rise = thermal_expansion + ice_melt
        
        return {
            'temperature': self.temperature,
            'co2_concentration': self.co2_concentration,
            'sea_level_rise': self.sea_level_rise,
            'equilibrium_temp': equilibrium_temp
        }
    
    def get_extreme_weather_intensity(self) -> float:
        """
        Calculate extreme weather intensity based on temperature
        
        Returns:
            Extreme weather intensity multiplier
        """
        # Exponential increase in extreme weather with temperature
        return 1 + 0.5 * (np.exp(self.temperature * 0.5) - 1)

class EconomicModule:
    """
    Economic system component of the IAM
    """
    
    def __init__(self, params: IAMParameters):
        """
        Initialize economic module
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.gdp = params.initial_gdp
        self.capital = params.initial_gdp * 3  # Initial capital stock
        self.population = 170e6  # Bangladesh population (2023)
        self.productivity = 1.0
        
        # Sectoral structure
        self.agriculture_share = params.agricultural_gdp_share
        self.industry_share = 0.35
        self.services_share = 0.52
        
        # Energy system
        self.energy_intensity = 0.8  # Energy per unit GDP
        self.renewable_share = 0.03  # Current renewable share
        self.coal_share = params.coal_share_initial
        
    def update_economy(self, climate_damages: float, adaptation_costs: float, 
                      mitigation_costs: float, year: int) -> Dict[str, float]:
        """
        Update economic state
        
        Args:
            climate_damages: Climate damage as share of GDP
            adaptation_costs: Adaptation costs as share of GDP
            mitigation_costs: Mitigation costs as share of GDP
            year: Current year
            
        Returns:
            Economic state variables
        """
        # Population growth
        years_elapsed = year - self.params.start_year
        self.population *= (1 + self.params.population_growth_rate)
        
        # Productivity growth
        self.productivity *= (1 + self.params.productivity_growth_rate)
        
        # Capital accumulation
        investment = self.gdp * self.params.savings_rate
        depreciation = self.capital * self.params.capital_depreciation
        self.capital += investment - depreciation
        
        # GDP calculation (Cobb-Douglas production function)
        alpha = 0.35  # Capital share
        potential_gdp = self.productivity * (self.capital ** alpha) * (self.population ** (1 - alpha))
        
        # Apply climate damages and costs
        total_costs = climate_damages + adaptation_costs + mitigation_costs
        self.gdp = potential_gdp * (1 - total_costs)
        
        # Update energy system
        self.energy_intensity *= (1 - self.params.energy_intensity_decline)
        
        # Renewable energy transition
        target_year = 2041  # Bangladesh renewable target year
        if year <= target_year:
            progress = (year - self.params.start_year) / (target_year - self.params.start_year)
            self.renewable_share = 0.03 + progress * (self.params.renewable_share_target - 0.03)
        
        return {
            'gdp': self.gdp,
            'gdp_per_capita': self.gdp / self.population,
            'capital': self.capital,
            'population': self.population,
            'productivity': self.productivity,
            'energy_intensity': self.energy_intensity,
            'renewable_share': self.renewable_share
        }
    
    def calculate_emissions(self) -> float:
        """
        Calculate CO2 emissions based on economic activity
        
        Returns:
            CO2 emissions (GtCO2)
        """
        # Energy demand
        energy_demand = self.gdp * self.energy_intensity
        
        # Emissions from fossil fuels
        fossil_share = 1 - self.renewable_share
        coal_emissions = energy_demand * self.coal_share * 0.9  # High carbon intensity
        gas_emissions = energy_demand * (fossil_share - self.coal_share) * 0.5  # Lower carbon intensity
        
        total_emissions = (coal_emissions + gas_emissions) / 1e9  # Convert to GtCO2
        
        return total_emissions

class DamageModule:
    """
    Climate damage assessment component
    """
    
    def __init__(self, params: IAMParameters):
        """
        Initialize damage module
        
        Args:
            params: Model parameters
        """
        self.params = params
        
    def calculate_damages(self, temperature: float, sea_level_rise: float, 
                         extreme_weather: float, gdp: float) -> Dict[str, float]:
        """
        Calculate climate damages
        
        Args:
            temperature: Global temperature increase (°C)
            sea_level_rise: Sea level rise (meters)
            extreme_weather: Extreme weather intensity
            gdp: Current GDP
            
        Returns:
            Damage components
        """
        # Temperature-based damages (Nordhaus-style)
        temp_damage = self.params.damage_coefficient * (temperature ** self.params.damage_exponent)
        
        # Sea level rise damages (Bangladesh-specific)
        slr_damage = self.params.sea_level_damage_coeff * sea_level_rise * self.params.coastal_area_share
        
        # Agricultural damages
        ag_temp_damage = self._calculate_agricultural_damage(temperature)
        ag_damage = ag_temp_damage * self.params.agricultural_gdp_share
        
        # Extreme weather damages
        cyclone_damage = self._calculate_cyclone_damage(extreme_weather)
        flood_damage = self._calculate_flood_damage(temperature, extreme_weather)
        
        # Total damages
        total_damage_share = temp_damage + slr_damage + ag_damage + cyclone_damage + flood_damage
        total_damage_value = total_damage_share * gdp
        
        return {
            'total_damage_share': min(total_damage_share, 0.5),  # Cap at 50% of GDP
            'total_damage_value': total_damage_value,
            'temperature_damage': temp_damage,
            'sea_level_damage': slr_damage,
            'agricultural_damage': ag_damage,
            'cyclone_damage': cyclone_damage,
            'flood_damage': flood_damage
        }
    
    def _calculate_agricultural_damage(self, temperature: float) -> float:
        """
        Calculate agricultural productivity damage
        
        Args:
            temperature: Temperature increase
            
        Returns:
            Agricultural damage share
        """
        # Bangladesh agriculture is sensitive to temperature and precipitation
        # Optimal temperature around 1°C increase, then declining
        if temperature <= 1.0:
            return -0.02 * temperature  # Small benefit initially
        else:
            return 0.05 * (temperature - 1.0) ** 1.5  # Accelerating damage
    
    def _calculate_cyclone_damage(self, extreme_weather: float) -> float:
        """
        Calculate cyclone damage
        
        Args:
            extreme_weather: Extreme weather intensity
            
        Returns:
            Cyclone damage share
        """
        # Bangladesh faces frequent cyclones
        base_cyclone_cost = 0.005  # 0.5% of GDP annually
        return base_cyclone_cost * extreme_weather * self.params.cyclone_frequency
    
    def _calculate_flood_damage(self, temperature: float, extreme_weather: float) -> float:
        """
        Calculate flood damage
        
        Args:
            temperature: Temperature increase
            extreme_weather: Extreme weather intensity
            
        Returns:
            Flood damage share
        """
        # Monsoon intensification and river flooding
        base_flood_cost = 0.01  # 1% of GDP annually
        temp_multiplier = 1 + 0.3 * temperature  # 30% increase per degree
        return base_flood_cost * temp_multiplier * extreme_weather * self.params.flood_vulnerability

class PolicyModule:
    """
    Policy intervention component
    """
    
    def __init__(self, params: IAMParameters):
        """
        Initialize policy module
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.carbon_price = params.carbon_price_initial
        self.adaptation_spending = 0.0
        self.mitigation_spending = 0.0
        
    def update_policies(self, year: int, emissions: float, damages: float) -> Dict[str, float]:
        """
        Update policy instruments
        
        Args:
            year: Current year
            emissions: Current emissions
            damages: Current climate damages
            
        Returns:
            Policy costs and effectiveness
        """
        # Carbon price trajectory
        years_elapsed = year - self.params.start_year
        self.carbon_price = self.params.carbon_price_initial * (1 + self.params.carbon_price_growth) ** years_elapsed
        
        # Mitigation costs
        mitigation_effort = min(0.5, self.carbon_price / 100)  # Effort increases with carbon price
        mitigation_cost_share = self.params.mitigation_cost_coeff * (mitigation_effort ** 2)
        
        # Adaptation costs
        adaptation_cost_share = self.params.adaptation_cost_share
        
        # Adaptation effectiveness
        adaptation_effectiveness = self.params.adaptation_effectiveness
        
        return {
            'carbon_price': self.carbon_price,
            'mitigation_cost_share': mitigation_cost_share,
            'adaptation_cost_share': adaptation_cost_share,
            'mitigation_effectiveness': mitigation_effort,
            'adaptation_effectiveness': adaptation_effectiveness,
            'emissions_reduction': mitigation_effort * 0.8  # 80% max reduction
        }

class IntegratedAssessmentModel:
    """
    Integrated Assessment Model for Bangladesh
    
    This class integrates economic, climate, and policy modules to analyze
    the interactions between economic development and climate change.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Integrated Assessment Model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = IAMParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Initialize modules
        self.climate = ClimateModule(self.params)
        self.economy = EconomicModule(self.params)
        self.damages = DamageModule(self.params)
        self.policy = PolicyModule(self.params)
        
        # Time vector
        self.years = np.arange(self.params.start_year, self.params.end_year + 1, self.params.time_step)
        
        # Results storage
        self.results = []
        
        logger.info("Integrated Assessment Model initialized for Bangladesh")
    
    def run_baseline_scenario(self) -> pd.DataFrame:
        """
        Run baseline scenario without additional climate policies
        
        Returns:
            Time series results
        """
        logger.info("Running baseline scenario")
        
        self.results = []
        
        for year in self.years:
            # Calculate emissions
            emissions = self.economy.calculate_emissions()
            
            # Update climate
            climate_state = self.climate.update_climate(emissions, year)
            
            # Calculate damages
            extreme_weather = self.climate.get_extreme_weather_intensity()
            damage_results = self.damages.calculate_damages(
                climate_state['temperature'],
                climate_state['sea_level_rise'],
                extreme_weather,
                self.economy.gdp
            )
            
            # Update policies (minimal in baseline)
            policy_results = self.policy.update_policies(year, emissions, damage_results['total_damage_share'])
            
            # Update economy
            economic_state = self.economy.update_economy(
                damage_results['total_damage_share'],
                0.0,  # No adaptation in baseline
                0.0,  # No mitigation in baseline
                year
            )
            
            # Store results
            result = {
                'year': year,
                'gdp': economic_state['gdp'],
                'gdp_per_capita': economic_state['gdp_per_capita'],
                'population': economic_state['population'],
                'emissions': emissions,
                'temperature': climate_state['temperature'],
                'co2_concentration': climate_state['co2_concentration'],
                'sea_level_rise': climate_state['sea_level_rise'],
                'total_damages': damage_results['total_damage_value'],
                'damage_share': damage_results['total_damage_share'],
                'extreme_weather': extreme_weather,
                'renewable_share': economic_state['renewable_share'],
                'carbon_price': policy_results['carbon_price']
            }
            
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def run_policy_scenario(self, scenario_name: str, policy_config: Dict) -> pd.DataFrame:
        """
        Run policy scenario with specified interventions
        
        Args:
            scenario_name: Name of the scenario
            policy_config: Policy configuration
            
        Returns:
            Time series results
        """
        logger.info(f"Running policy scenario: {scenario_name}")
        
        # Reset modules
        self.climate = ClimateModule(self.params)
        self.economy = EconomicModule(self.params)
        self.damages = DamageModule(self.params)
        self.policy = PolicyModule(self.params)
        
        # Update policy parameters
        for key, value in policy_config.items():
            if hasattr(self.policy.params, key):
                setattr(self.policy.params, key, value)
        
        results = []
        
        for year in self.years:
            # Calculate emissions
            base_emissions = self.economy.calculate_emissions()
            
            # Update policies
            policy_results = self.policy.update_policies(year, base_emissions, 0)
            
            # Apply mitigation
            emissions_reduction = policy_results['emissions_reduction']
            actual_emissions = base_emissions * (1 - emissions_reduction)
            
            # Update climate
            climate_state = self.climate.update_climate(actual_emissions, year)
            
            # Calculate damages
            extreme_weather = self.climate.get_extreme_weather_intensity()
            damage_results = self.damages.calculate_damages(
                climate_state['temperature'],
                climate_state['sea_level_rise'],
                extreme_weather,
                self.economy.gdp
            )
            
            # Apply adaptation effectiveness
            adapted_damages = damage_results['total_damage_share'] * (1 - policy_results['adaptation_effectiveness'])
            
            # Update economy
            economic_state = self.economy.update_economy(
                adapted_damages,
                policy_results['adaptation_cost_share'],
                policy_results['mitigation_cost_share'],
                year
            )
            
            # Store results
            result = {
                'year': year,
                'scenario': scenario_name,
                'gdp': economic_state['gdp'],
                'gdp_per_capita': economic_state['gdp_per_capita'],
                'population': economic_state['population'],
                'emissions': actual_emissions,
                'temperature': climate_state['temperature'],
                'co2_concentration': climate_state['co2_concentration'],
                'sea_level_rise': climate_state['sea_level_rise'],
                'total_damages': adapted_damages * economic_state['gdp'],
                'damage_share': adapted_damages,
                'adaptation_costs': policy_results['adaptation_cost_share'] * economic_state['gdp'],
                'mitigation_costs': policy_results['mitigation_cost_share'] * economic_state['gdp'],
                'extreme_weather': extreme_weather,
                'renewable_share': economic_state['renewable_share'],
                'carbon_price': policy_results['carbon_price']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def run_scenario_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Run multiple scenarios for comparison
        
        Returns:
            Dictionary of scenario results
        """
        logger.info("Running scenario analysis")
        
        scenarios = {}
        
        # Baseline scenario
        scenarios['baseline'] = self.run_baseline_scenario()
        
        # Moderate mitigation scenario
        moderate_policy = {
            'carbon_price_initial': 25,
            'carbon_price_growth': 0.08,
            'adaptation_cost_share': 0.015,
            'mitigation_cost_coeff': 0.03
        }
        scenarios['moderate_action'] = self.run_policy_scenario('moderate_action', moderate_policy)
        
        # Ambitious mitigation scenario
        ambitious_policy = {
            'carbon_price_initial': 50,
            'carbon_price_growth': 0.12,
            'adaptation_cost_share': 0.025,
            'mitigation_cost_coeff': 0.04,
            'renewable_share_target': 0.6
        }
        scenarios['ambitious_action'] = self.run_policy_scenario('ambitious_action', ambitious_policy)
        
        # Adaptation-focused scenario
        adaptation_policy = {
            'carbon_price_initial': 15,
            'carbon_price_growth': 0.05,
            'adaptation_cost_share': 0.04,
            'adaptation_effectiveness': 0.8,
            'mitigation_cost_coeff': 0.02
        }
        scenarios['adaptation_focus'] = self.run_policy_scenario('adaptation_focus', adaptation_policy)
        
        return scenarios
    
    def calculate_social_cost_carbon(self, discount_rate: float = None) -> float:
        """
        Calculate Social Cost of Carbon (SCC)
        
        Args:
            discount_rate: Discount rate for SCC calculation
            
        Returns:
            Social Cost of Carbon (USD/tCO2)
        """
        if discount_rate is None:
            discount_rate = self.params.discount_rate
        
        logger.info("Calculating Social Cost of Carbon")
        
        # Run baseline scenario
        baseline = self.run_baseline_scenario()
        
        # Run scenario with additional 1 GtCO2 emission in first year
        self.climate = ClimateModule(self.params)
        self.economy = EconomicModule(self.params)
        self.damages = DamageModule(self.params)
        
        marginal_results = []
        
        for i, year in enumerate(self.years):
            # Add marginal emission in first year only
            emissions = self.economy.calculate_emissions()
            if i == 0:
                emissions += 1.0  # Add 1 GtCO2
            
            # Update climate
            climate_state = self.climate.update_climate(emissions, year)
            
            # Calculate damages
            extreme_weather = self.climate.get_extreme_weather_intensity()
            damage_results = self.damages.calculate_damages(
                climate_state['temperature'],
                climate_state['sea_level_rise'],
                extreme_weather,
                self.economy.gdp
            )
            
            # Update economy
            economic_state = self.economy.update_economy(
                damage_results['total_damage_share'], 0.0, 0.0, year
            )
            
            marginal_results.append({
                'year': year,
                'total_damages': damage_results['total_damage_value']
            })
        
        marginal_df = pd.DataFrame(marginal_results)
        
        # Calculate present value of marginal damages
        years_from_start = marginal_df['year'] - self.params.start_year
        discount_factors = (1 + discount_rate) ** (-years_from_start)
        
        baseline_pv_damages = np.sum(baseline['total_damages'] * discount_factors)
        marginal_pv_damages = np.sum(marginal_df['total_damages'] * discount_factors)
        
        # SCC is the difference in present value damages per tCO2
        scc = (marginal_pv_damages - baseline_pv_damages) / 1e9  # Convert from GtCO2 to tCO2
        
        return scc
    
    def analyze_uncertainty(self, n_simulations: int = 100) -> Dict:
        """
        Perform uncertainty analysis using Monte Carlo simulation
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Uncertainty analysis results
        """
        logger.info(f"Running uncertainty analysis with {n_simulations} simulations")
        
        results = []
        
        for sim in range(n_simulations):
            # Sample uncertain parameters
            temp_sensitivity = np.random.normal(self.params.climate_sensitivity, 
                                              self.params.temperature_uncertainty)
            damage_coeff = np.random.normal(self.params.damage_coefficient, 
                                          self.params.damage_coefficient * self.params.damage_uncertainty)
            growth_rate = np.random.normal(self.params.productivity_growth_rate,
                                         self.params.productivity_growth_rate * self.params.economic_uncertainty)
            
            # Create temporary parameters
            temp_params = IAMParameters()
            temp_params.climate_sensitivity = max(1.5, temp_sensitivity)
            temp_params.damage_coefficient = max(0.001, damage_coeff)
            temp_params.productivity_growth_rate = max(0.01, growth_rate)
            
            # Run simulation with sampled parameters
            temp_model = IntegratedAssessmentModel({'parameters': temp_params.__dict__})
            baseline_result = temp_model.run_baseline_scenario()
            
            # Store key outcomes
            final_year = baseline_result.iloc[-1]
            results.append({
                'simulation': sim,
                'final_gdp': final_year['gdp'],
                'final_temperature': final_year['temperature'],
                'total_damages_2100': final_year['total_damages'],
                'cumulative_emissions': baseline_result['emissions'].sum()
            })
        
        uncertainty_df = pd.DataFrame(results)
        
        # Calculate statistics
        uncertainty_stats = {
            'gdp_2100': {
                'mean': uncertainty_df['final_gdp'].mean(),
                'std': uncertainty_df['final_gdp'].std(),
                'p5': uncertainty_df['final_gdp'].quantile(0.05),
                'p95': uncertainty_df['final_gdp'].quantile(0.95)
            },
            'temperature_2100': {
                'mean': uncertainty_df['final_temperature'].mean(),
                'std': uncertainty_df['final_temperature'].std(),
                'p5': uncertainty_df['final_temperature'].quantile(0.05),
                'p95': uncertainty_df['final_temperature'].quantile(0.95)
            },
            'damages_2100': {
                'mean': uncertainty_df['total_damages_2100'].mean(),
                'std': uncertainty_df['total_damages_2100'].std(),
                'p5': uncertainty_df['total_damages_2100'].quantile(0.05),
                'p95': uncertainty_df['total_damages_2100'].quantile(0.95)
            }
        }
        
        return {
            'simulation_results': uncertainty_df,
            'statistics': uncertainty_stats
        }
    
    def plot_scenario_comparison(self, scenarios: Dict[str, pd.DataFrame], save_path: str = None):
        """
        Plot comparison of different scenarios
        
        Args:
            scenarios: Dictionary of scenario results
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # GDP trajectories
        for name, data in scenarios.items():
            axes[0, 0].plot(data['year'], data['gdp'] / 1e12, label=name, linewidth=2)
        axes[0, 0].set_title('GDP Trajectories')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('GDP (Trillion USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temperature trajectories
        for name, data in scenarios.items():
            axes[0, 1].plot(data['year'], data['temperature'], label=name, linewidth=2)
        axes[0, 1].set_title('Global Temperature Increase')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Temperature Increase (°C)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Emissions trajectories
        for name, data in scenarios.items():
            axes[0, 2].plot(data['year'], data['emissions'], label=name, linewidth=2)
        axes[0, 2].set_title('CO2 Emissions')
        axes[0, 2].set_xlabel('Year')
        axes[0, 2].set_ylabel('Emissions (GtCO2/year)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Climate damages
        for name, data in scenarios.items():
            axes[1, 0].plot(data['year'], data['damage_share'] * 100, label=name, linewidth=2)
        axes[1, 0].set_title('Climate Damages')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Damages (% of GDP)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sea level rise
        for name, data in scenarios.items():
            axes[1, 1].plot(data['year'], data['sea_level_rise'] * 100, label=name, linewidth=2)
        axes[1, 1].set_title('Sea Level Rise')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Sea Level Rise (cm)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Renewable energy share
        for name, data in scenarios.items():
            axes[1, 2].plot(data['year'], data['renewable_share'] * 100, label=name, linewidth=2)
        axes[1, 2].set_title('Renewable Energy Share')
        axes[1, 2].set_xlabel('Year')
        axes[1, 2].set_ylabel('Renewable Share (%)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scenario comparison plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Integrated Assessment Model',
            'country': 'Bangladesh',
            'time_horizon': f"{self.params.start_year}-{self.params.end_year}",
            'components': [
                'Climate System',
                'Economic System', 
                'Damage Functions',
                'Policy Module'
            ],
            'key_features': [
                'Economic-Climate Interactions',
                'Sea Level Rise Impacts',
                'Extreme Weather Effects',
                'Adaptation Strategies',
                'Mitigation Policies',
                'Uncertainty Analysis',
                'Social Cost of Carbon'
            ],
            'bangladesh_specifics': [
                'Coastal Vulnerability',
                'Flood Risk Assessment',
                'Cyclone Impacts',
                'Agricultural Sensitivity',
                'Energy Transition Pathways'
            ]
        }
        
        return summary
    
    def simulate(self, periods: int = 100) -> Dict:
        """
        Run simulation for compatibility with individual model runner
        
        Args:
            periods: Number of periods to simulate
            
        Returns:
            Dictionary containing simulation results
        """
        logger.info(f"Running IAM simulation for {periods} periods")
        
        # Run baseline scenario
        baseline_results = self.run_baseline_scenario()
        
        # Convert to the expected format
        results = {
            'status': 'converged',
            'periods': len(baseline_results),
            'data': baseline_results.to_dict('records'),
            'summary': {
                'final_gdp': baseline_results.iloc[-1]['gdp'],
                'final_temperature': baseline_results.iloc[-1]['temperature'],
                'total_damages': baseline_results['total_damages'].sum(),
                'renewable_share': baseline_results.iloc[-1]['renewable_share']
            }
        }
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'start_year': 2025,
            'end_year': 2080,
            'climate_sensitivity': 3.0,
            'damage_coefficient': 0.0023,
            'adaptation_cost_share': 0.02,
            'carbon_price_initial': 15
        }
    }
    
    # Initialize Integrated Assessment Model
    iam = IntegratedAssessmentModel(config)
    
    # Run baseline scenario
    print("Running baseline scenario...")
    baseline = iam.run_baseline_scenario()
    print(f"Baseline scenario completed: {len(baseline)} years")
    
    # Display key results
    final_year = baseline.iloc[-1]
    print(f"\nBaseline Results for {final_year['year']:.0f}:")
    print(f"  GDP: ${final_year['gdp']/1e12:.1f} trillion")
    print(f"  GDP per capita: ${final_year['gdp_per_capita']:,.0f}")
    print(f"  Temperature increase: {final_year['temperature']:.1f}°C")
    print(f"  Sea level rise: {final_year['sea_level_rise']*100:.0f} cm")
    print(f"  Climate damages: {final_year['damage_share']:.1%} of GDP")
    print(f"  Renewable energy: {final_year['renewable_share']:.1%}")
    
    # Run scenario analysis
    print("\nRunning scenario analysis...")
    scenarios = iam.run_scenario_analysis()
    
    # Compare scenarios
    print("\nScenario Comparison (2080):")
    for name, data in scenarios.items():
        final = data.iloc[-1]
        print(f"  {name}:")
        print(f"    GDP: ${final['gdp']/1e12:.1f}T, Temp: {final['temperature']:.1f}°C, Damages: {final['damage_share']:.1%}")
    
    # Calculate Social Cost of Carbon
    print("\nCalculating Social Cost of Carbon...")
    scc = iam.calculate_social_cost_carbon()
    print(f"Social Cost of Carbon: ${scc:.0f}/tCO2")
    
    # Model summary
    summary = iam.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Time horizon: {summary['time_horizon']}")
    print(f"  Components: {len(summary['components'])}")
    print(f"  Bangladesh-specific features: {len(summary['bangladesh_specifics'])}")
    
    print("\nIntegrated Assessment Model analysis completed successfully!")