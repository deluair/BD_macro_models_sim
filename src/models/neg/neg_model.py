#!/usr/bin/env python3
"""
New Economic Geography (NEG) Model for Bangladesh

This module implements a New Economic Geography model to analyze spatial
economic patterns, regional development, and agglomeration effects in Bangladesh.
The model examines how economic activity concentrates in space due to
transportation costs, economies of scale, and factor mobility.

Key Features:
- Core-periphery model structure
- Transportation costs and trade
- Industrial agglomeration
- Regional wage differentials
- Migration dynamics
- Urban-rural development patterns
- Infrastructure effects
- Policy analysis for regional development

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, spatial
from scipy.integrate import odeint
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NEGParameters:
    """
    Parameters for the New Economic Geography model
    """
    # Core model parameters
    sigma: float = 5.0                   # Elasticity of substitution
    mu: float = 0.4                      # Share of manufacturing
    rho: float = 0.5                     # Discount rate for migration
    
    # Transportation and trade costs
    tau: float = 1.5                     # Iceberg transportation cost
    tau_min: float = 1.1                 # Minimum transportation cost
    tau_max: float = 3.0                 # Maximum transportation cost
    
    # Regional characteristics
    n_regions: int = 8                   # Number of regions (Bangladesh divisions)
    total_labor: float = 1.0             # Total labor force
    total_land: float = 1.0              # Total agricultural land
    
    # Bangladesh-specific parameters
    dhaka_share: float = 0.25            # Dhaka's initial manufacturing share
    chittagong_share: float = 0.15       # Chittagong's initial share
    rural_agri_share: float = 0.6        # Rural agricultural employment share
    
    # Infrastructure and connectivity
    road_density: Dict[str, float] = field(default_factory=lambda: {
        'Dhaka': 1.0, 'Chittagong': 0.8, 'Sylhet': 0.4, 'Rajshahi': 0.5,
        'Khulna': 0.6, 'Barisal': 0.3, 'Rangpur': 0.4, 'Mymensingh': 0.4
    })
    
    # Economic development indicators
    initial_wages: Dict[str, float] = field(default_factory=lambda: {
        'Dhaka': 1.2, 'Chittagong': 1.0, 'Sylhet': 0.7, 'Rajshahi': 0.8,
        'Khulna': 0.9, 'Barisal': 0.6, 'Rangpur': 0.6, 'Mymensingh': 0.7
    })
    
    # Migration parameters
    migration_cost: float = 0.1          # Cost of migration
    migration_elasticity: float = 2.0    # Elasticity of migration to wage differentials
    
    # Agglomeration parameters
    agglomeration_strength: float = 0.3  # Strength of agglomeration effects
    congestion_parameter: float = 0.1    # Congestion costs parameter
    
    # Policy parameters
    infrastructure_elasticity: float = 0.5  # Infrastructure impact on transport costs
    regional_subsidy: Dict[str, float] = field(default_factory=dict)  # Regional subsidies

@dataclass
class NEGResults:
    """
    Results from NEG model simulation
    """
    parameters: NEGParameters
    equilibrium_data: Optional[pd.DataFrame] = None
    regional_indicators: Optional[Dict] = None
    migration_flows: Optional[pd.DataFrame] = None
    agglomeration_index: Optional[Dict] = None
    policy_analysis: Optional[Dict] = None
    spatial_distribution: Optional[Dict] = None

class NewEconomicGeographyModel:
    """
    New Economic Geography Model for Bangladesh
    
    This class implements a NEG model to analyze spatial patterns of
    economic development, regional disparities, and agglomeration effects.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize NEG model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = NEGParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Regional data
        self.regions = list(self.params.initial_wages.keys())
        self.n_regions = len(self.regions)
        
        # Initialize regional characteristics
        self.regional_data = self._initialize_regional_data()
        
        # Transportation cost matrix
        self.transport_costs = self._compute_transport_costs()
        
        # Model state
        self.equilibrium = None
        self.results = None
        
        logger.info(f"NEG model initialized for Bangladesh with {self.n_regions} regions")
    
    def _initialize_regional_data(self) -> pd.DataFrame:
        """
        Initialize regional characteristics data
        
        Returns:
            DataFrame with regional data
        """
        # Bangladesh divisions with approximate coordinates
        regional_coords = {
            'Dhaka': (23.8103, 90.4125),
            'Chittagong': (22.3569, 91.7832),
            'Sylhet': (24.8949, 91.8687),
            'Rajshahi': (24.3745, 88.6042),
            'Khulna': (22.8456, 89.5403),
            'Barisal': (22.7010, 90.3535),
            'Rangpur': (25.7439, 89.2752),
            'Mymensingh': (24.7471, 90.4203)
        }
        
        # Population shares (approximate)
        population_shares = {
            'Dhaka': 0.22, 'Chittagong': 0.18, 'Sylhet': 0.08, 'Rajshahi': 0.12,
            'Khulna': 0.10, 'Barisal': 0.06, 'Rangpur': 0.10, 'Mymensingh': 0.14
        }
        
        # Industrial development index
        industrial_index = {
            'Dhaka': 1.0, 'Chittagong': 0.8, 'Sylhet': 0.3, 'Rajshahi': 0.4,
            'Khulna': 0.5, 'Barisal': 0.2, 'Rangpur': 0.3, 'Mymensingh': 0.3
        }
        
        regional_data = []
        
        for region in self.regions:
            data = {
                'region': region,
                'latitude': regional_coords[region][0],
                'longitude': regional_coords[region][1],
                'population_share': population_shares.get(region, 0.1),
                'initial_wage': self.params.initial_wages.get(region, 0.8),
                'road_density': self.params.road_density.get(region, 0.5),
                'industrial_index': industrial_index.get(region, 0.4),
                'agricultural_land': 1.0 - industrial_index.get(region, 0.4),
                'is_major_city': region in ['Dhaka', 'Chittagong']
            }
            regional_data.append(data)
        
        return pd.DataFrame(regional_data)
    
    def _compute_transport_costs(self) -> np.ndarray:
        """
        Compute transportation cost matrix between regions
        
        Returns:
            Transportation cost matrix
        """
        n = len(self.regions)
        transport_matrix = np.ones((n, n))
        
        # Compute distances
        coords = self.regional_data[['latitude', 'longitude']].values
        distances = spatial.distance_matrix(coords, coords)
        
        # Normalize distances
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        
        # Transportation costs based on distance and infrastructure
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Base cost from distance
                    distance_cost = 1 + normalized_distances[i, j]
                    
                    # Infrastructure adjustment
                    region_i = self.regions[i]
                    region_j = self.regions[j]
                    
                    infra_i = self.params.road_density.get(region_i, 0.5)
                    infra_j = self.params.road_density.get(region_j, 0.5)
                    
                    # Better infrastructure reduces transport costs
                    infra_factor = 1 / (1 + 0.5 * (infra_i + infra_j))
                    
                    transport_matrix[i, j] = distance_cost * infra_factor
                    
                    # Ensure within bounds
                    transport_matrix[i, j] = np.clip(
                        transport_matrix[i, j],
                        self.params.tau_min,
                        self.params.tau_max
                    )
        
        return transport_matrix
    
    def compute_equilibrium(self, max_iterations: int = 1000, tolerance: float = 1e-6) -> Dict:
        """
        Compute spatial equilibrium
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Equilibrium solution
        """
        logger.info("Computing spatial equilibrium")
        
        n = self.n_regions
        
        # Initial conditions
        # Manufacturing labor shares
        lambda_m = np.zeros(n)
        lambda_m[0] = self.params.dhaka_share  # Dhaka
        lambda_m[1] = self.params.chittagong_share  # Chittagong
        
        # Distribute remaining manufacturing employment
        remaining = 1 - lambda_m.sum()
        for i in range(2, n):
            lambda_m[i] = remaining / (n - 2)
        
        # Agricultural labor shares (uniform initially)
        lambda_a = np.ones(n) / n
        
        # Iterative solution
        for iteration in range(max_iterations):
            lambda_m_old = lambda_m.copy()
            
            # Compute price indices
            price_indices = self._compute_price_indices(lambda_m)
            
            # Compute wages
            wages_m, wages_a = self._compute_wages(lambda_m, lambda_a, price_indices)
            
            # Compute real wages
            real_wages_m = wages_m / price_indices
            real_wages_a = wages_a  # Agricultural good is numeraire
            
            # Update manufacturing labor allocation based on real wage differentials
            lambda_m = self._update_labor_allocation(lambda_m, real_wages_m, real_wages_a)
            
            # Check convergence
            if np.max(np.abs(lambda_m - lambda_m_old)) < tolerance:
                logger.info(f"Equilibrium converged after {iteration + 1} iterations")
                break
        else:
            logger.warning(f"Equilibrium did not converge after {max_iterations} iterations")
        
        # Final calculations
        price_indices = self._compute_price_indices(lambda_m)
        wages_m, wages_a = self._compute_wages(lambda_m, lambda_a, price_indices)
        real_wages_m = wages_m / price_indices
        real_wages_a = wages_a
        
        # Compute additional indicators
        agglomeration_index = self._compute_agglomeration_index(lambda_m)
        regional_output = self._compute_regional_output(lambda_m, lambda_a)
        
        equilibrium = {
            'manufacturing_labor_share': lambda_m,
            'agricultural_labor_share': lambda_a,
            'manufacturing_wages': wages_m,
            'agricultural_wages': wages_a,
            'real_wages_manufacturing': real_wages_m,
            'real_wages_agriculture': real_wages_a,
            'price_indices': price_indices,
            'agglomeration_index': agglomeration_index,
            'regional_output': regional_output,
            'transport_costs': self.transport_costs
        }
        
        self.equilibrium = equilibrium
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'region': self.regions,
            'manufacturing_labor_share': lambda_m,
            'agricultural_labor_share': lambda_a,
            'manufacturing_wage': wages_m,
            'agricultural_wage': wages_a,
            'real_wage_manufacturing': real_wages_m,
            'real_wage_agriculture': real_wages_a,
            'price_index': price_indices,
            'total_output': regional_output
        })
        
        # Add regional characteristics
        results_df = results_df.merge(self.regional_data, on='region')
        
        logger.info("Spatial equilibrium computed successfully")
        return equilibrium
    
    def _compute_price_indices(self, lambda_m: np.ndarray) -> np.ndarray:
        """
        Compute price indices for each region
        
        Args:
            lambda_m: Manufacturing labor shares
            
        Returns:
            Price indices
        """
        n = len(lambda_m)
        sigma = self.params.sigma
        
        price_indices = np.zeros(n)
        
        for i in range(n):
            price_sum = 0
            for j in range(n):
                if lambda_m[j] > 0:
                    # Price of variety from region j in region i
                    tau_ij = self.transport_costs[i, j]
                    price_sum += lambda_m[j] * (tau_ij ** (1 - sigma))
            
            price_indices[i] = price_sum ** (1 / (1 - sigma))
        
        return price_indices
    
    def _compute_wages(self, lambda_m: np.ndarray, lambda_a: np.ndarray, 
                      price_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute wages in manufacturing and agriculture
        
        Args:
            lambda_m: Manufacturing labor shares
            lambda_a: Agricultural labor shares
            price_indices: Price indices
            
        Returns:
            Manufacturing and agricultural wages
        """
        n = len(lambda_m)
        sigma = self.params.sigma
        mu = self.params.mu
        
        # Manufacturing wages
        wages_m = np.zeros(n)
        
        for i in range(n):
            if lambda_m[i] > 0:
                # Market access term
                market_access = 0
                for j in range(n):
                    tau_ij = self.transport_costs[i, j]
                    market_access += (tau_ij ** (1 - sigma)) * (price_indices[j] ** (sigma - 1))
                
                wages_m[i] = market_access ** (1 / (sigma - 1))
        
        # Agricultural wages (normalized to 1)
        wages_a = np.ones(n)
        
        # Apply agglomeration effects
        wages_m = self._apply_agglomeration_effects(wages_m, lambda_m)
        
        return wages_m, wages_a
    
    def _apply_agglomeration_effects(self, wages: np.ndarray, lambda_m: np.ndarray) -> np.ndarray:
        """
        Apply agglomeration and congestion effects to wages
        
        Args:
            wages: Base wages
            lambda_m: Manufacturing labor shares
            
        Returns:
            Adjusted wages
        """
        agglomeration_strength = self.params.agglomeration_strength
        congestion_parameter = self.params.congestion_parameter
        
        # Agglomeration benefits (increasing returns)
        agglomeration_effect = (lambda_m ** agglomeration_strength)
        
        # Congestion costs (decreasing returns)
        congestion_effect = (1 + congestion_parameter * lambda_m)
        
        adjusted_wages = wages * agglomeration_effect / congestion_effect
        
        return adjusted_wages
    
    def _update_labor_allocation(self, lambda_m: np.ndarray, real_wages_m: np.ndarray, 
                                real_wages_a: np.ndarray) -> np.ndarray:
        """
        Update manufacturing labor allocation based on real wage differentials
        
        Args:
            lambda_m: Current manufacturing labor shares
            real_wages_m: Real wages in manufacturing
            real_wages_a: Real wages in agriculture
            
        Returns:
            Updated manufacturing labor shares
        """
        migration_elasticity = self.params.migration_elasticity
        migration_cost = self.params.migration_cost
        
        # Compute utility differentials
        utility_m = np.log(real_wages_m + 1e-10)  # Add small constant to avoid log(0)
        utility_a = np.log(real_wages_a + 1e-10)
        
        # Migration flows based on utility differentials
        # Simplified logit-type migration
        exp_utility = np.exp(migration_elasticity * (utility_m - np.mean(utility_a)))
        
        # Normalize to ensure sum equals total manufacturing employment
        new_lambda_m = exp_utility / np.sum(exp_utility)
        
        # Apply migration costs (reduce mobility)
        adjustment_speed = 1 / (1 + migration_cost)
        lambda_m_updated = (1 - adjustment_speed) * lambda_m + adjustment_speed * new_lambda_m
        
        # Ensure non-negative and sum to 1
        lambda_m_updated = np.maximum(lambda_m_updated, 0.001)  # Minimum employment
        lambda_m_updated = lambda_m_updated / np.sum(lambda_m_updated)
        
        return lambda_m_updated
    
    def _compute_agglomeration_index(self, lambda_m: np.ndarray) -> Dict:
        """
        Compute agglomeration indices
        
        Args:
            lambda_m: Manufacturing labor shares
            
        Returns:
            Agglomeration indices
        """
        n = len(lambda_m)
        
        # Herfindahl index (concentration)
        herfindahl = np.sum(lambda_m ** 2)
        
        # Gini coefficient
        sorted_shares = np.sort(lambda_m)
        n_regions = len(sorted_shares)
        index = np.arange(1, n_regions + 1)
        gini = (2 * np.sum(index * sorted_shares)) / (n_regions * np.sum(sorted_shares)) - (n_regions + 1) / n_regions
        
        # Theil index
        theil = np.sum(lambda_m * np.log(lambda_m * n + 1e-10)) / n
        
        # Core-periphery index (share of top 2 regions)
        top_2_share = np.sum(np.sort(lambda_m)[-2:])
        
        return {
            'herfindahl_index': herfindahl,
            'gini_coefficient': gini,
            'theil_index': theil,
            'core_periphery_index': top_2_share,
            'max_regional_share': np.max(lambda_m),
            'min_regional_share': np.min(lambda_m)
        }
    
    def _compute_regional_output(self, lambda_m: np.ndarray, lambda_a: np.ndarray) -> np.ndarray:
        """
        Compute total output by region
        
        Args:
            lambda_m: Manufacturing labor shares
            lambda_a: Agricultural labor shares
            
        Returns:
            Regional output levels
        """
        # Simplified output calculation
        # Manufacturing output proportional to labor share
        manufacturing_output = lambda_m * self.params.mu
        
        # Agricultural output proportional to labor and land
        agricultural_output = lambda_a * (1 - self.params.mu)
        
        total_output = manufacturing_output + agricultural_output
        
        return total_output
    
    def simulate_migration_dynamics(self, periods: int = 50, 
                                  shock_region: str = None, shock_size: float = 0.1) -> pd.DataFrame:
        """
        Simulate migration dynamics over time
        
        Args:
            periods: Number of periods to simulate
            shock_region: Region to apply shock (optional)
            shock_size: Size of productivity shock
            
        Returns:
            Migration dynamics DataFrame
        """
        logger.info(f"Simulating migration dynamics for {periods} periods")
        
        if self.equilibrium is None:
            self.compute_equilibrium()
        
        # Initial conditions from equilibrium
        lambda_m = self.equilibrium['manufacturing_labor_share'].copy()
        
        # Storage for results
        dynamics_data = []
        
        for t in range(periods):
            # Apply shock if specified
            if shock_region and t == 10:  # Apply shock at period 10
                shock_idx = self.regions.index(shock_region)
                # Temporary productivity increase
                temp_wages = self.equilibrium['manufacturing_wages'].copy()
                temp_wages[shock_idx] *= (1 + shock_size)
            else:
                temp_wages = self.equilibrium['manufacturing_wages']
            
            # Compute current state
            price_indices = self._compute_price_indices(lambda_m)
            real_wages_m = temp_wages / price_indices
            real_wages_a = self.equilibrium['agricultural_wages']
            
            # Update labor allocation
            lambda_m = self._update_labor_allocation(lambda_m, real_wages_m, real_wages_a)
            
            # Store results
            for i, region in enumerate(self.regions):
                dynamics_data.append({
                    'period': t,
                    'region': region,
                    'manufacturing_labor_share': lambda_m[i],
                    'real_wage_manufacturing': real_wages_m[i],
                    'price_index': price_indices[i]
                })
        
        return pd.DataFrame(dynamics_data)
    
    def analyze_infrastructure_policy(self, infrastructure_improvements: Dict[str, float]) -> Dict:
        """
        Analyze the impact of infrastructure improvements
        
        Args:
            infrastructure_improvements: Dict of region -> improvement factor
            
        Returns:
            Policy analysis results
        """
        logger.info("Analyzing infrastructure policy impacts")
        
        # Baseline equilibrium
        baseline = self.compute_equilibrium()
        
        # Store original transport costs
        original_costs = self.transport_costs.copy()
        
        # Apply infrastructure improvements
        for region, improvement in infrastructure_improvements.items():
            if region in self.regions:
                region_idx = self.regions.index(region)
                
                # Reduce transport costs from/to this region
                reduction_factor = 1 / (1 + self.params.infrastructure_elasticity * improvement)
                
                self.transport_costs[region_idx, :] *= reduction_factor
                self.transport_costs[:, region_idx] *= reduction_factor
                
                # Diagonal elements remain 1
                self.transport_costs[region_idx, region_idx] = 1.0
        
        # Compute new equilibrium
        improved = self.compute_equilibrium()
        
        # Restore original transport costs
        self.transport_costs = original_costs
        
        # Compare results
        policy_impact = {
            'baseline': baseline,
            'improved': improved,
            'changes': {}
        }
        
        # Calculate changes
        for key in ['manufacturing_labor_share', 'real_wages_manufacturing', 'regional_output']:
            baseline_values = baseline[key]
            improved_values = improved[key]
            
            absolute_change = improved_values - baseline_values
            relative_change = (improved_values - baseline_values) / (baseline_values + 1e-10) * 100
            
            policy_impact['changes'][key] = {
                'absolute': absolute_change,
                'relative': relative_change
            }
        
        # Welfare analysis
        baseline_welfare = np.sum(baseline['real_wages_manufacturing'] * baseline['manufacturing_labor_share'])
        improved_welfare = np.sum(improved['real_wages_manufacturing'] * improved['manufacturing_labor_share'])
        
        policy_impact['welfare_change'] = {
            'absolute': improved_welfare - baseline_welfare,
            'relative': (improved_welfare - baseline_welfare) / baseline_welfare * 100
        }
        
        return policy_impact
    
    def analyze_regional_subsidies(self, subsidies: Dict[str, float]) -> Dict:
        """
        Analyze the impact of regional development subsidies
        
        Args:
            subsidies: Dict of region -> subsidy rate
            
        Returns:
            Subsidy analysis results
        """
        logger.info("Analyzing regional subsidy impacts")
        
        # This would modify the wage equations to include subsidies
        # Simplified implementation: subsidies increase effective wages
        
        baseline = self.compute_equilibrium()
        
        # Apply subsidies by modifying wages
        modified_wages = baseline['manufacturing_wages'].copy()
        
        for region, subsidy_rate in subsidies.items():
            if region in self.regions:
                region_idx = self.regions.index(region)
                modified_wages[region_idx] *= (1 + subsidy_rate)
        
        # Recompute equilibrium with modified wages
        # (This is a simplified approach; full implementation would modify the equilibrium conditions)
        
        subsidy_analysis = {
            'baseline_wages': baseline['manufacturing_wages'],
            'subsidized_wages': modified_wages,
            'wage_increase': modified_wages - baseline['manufacturing_wages'],
            'subsidy_cost': sum(subsidies.values()) * 0.1,  # Simplified cost calculation
            'targeted_regions': list(subsidies.keys())
        }
        
        return subsidy_analysis
    
    def compute_trade_flows(self) -> pd.DataFrame:
        """
        Compute inter-regional trade flows
        
        Returns:
            Trade flows DataFrame
        """
        if self.equilibrium is None:
            self.compute_equilibrium()
        
        lambda_m = self.equilibrium['manufacturing_labor_share']
        price_indices = self.equilibrium['price_indices']
        
        trade_flows = []
        
        for i, origin in enumerate(self.regions):
            for j, destination in enumerate(self.regions):
                if i != j and lambda_m[i] > 0:
                    # Trade flow from i to j
                    tau_ij = self.transport_costs[i, j]
                    
                    # Simplified trade flow calculation
                    flow = (lambda_m[i] * (tau_ij ** (1 - self.params.sigma)) * 
                           (price_indices[j] ** (self.params.sigma - 1)))
                    
                    trade_flows.append({
                        'origin': origin,
                        'destination': destination,
                        'flow': flow,
                        'transport_cost': tau_ij,
                        'distance_km': self._get_distance(i, j)
                    })
        
        return pd.DataFrame(trade_flows)
    
    def _get_distance(self, i: int, j: int) -> float:
        """
        Get distance between regions i and j in kilometers
        """
        coords_i = self.regional_data.iloc[i][['latitude', 'longitude']].values
        coords_j = self.regional_data.iloc[j][['latitude', 'longitude']].values
        
        # Haversine distance formula (approximate)
        lat1, lon1 = np.radians(coords_i)
        lat2, lon2 = np.radians(coords_j)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in km
        R = 6371
        distance = R * c
        
        return distance
    
    def plot_spatial_distribution(self, save_path: str = None):
        """
        Plot spatial distribution of economic activity
        
        Args:
            save_path: Path to save plot
        """
        if self.equilibrium is None:
            self.compute_equilibrium()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Manufacturing labor share
        ax1 = axes[0, 0]
        manufacturing_shares = self.equilibrium['manufacturing_labor_share']
        bars1 = ax1.bar(self.regions, manufacturing_shares, color='steelblue', alpha=0.7)
        ax1.set_title('Manufacturing Labor Share by Region')
        ax1.set_ylabel('Share')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, manufacturing_shares):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Real wages
        ax2 = axes[0, 1]
        real_wages = self.equilibrium['real_wages_manufacturing']
        bars2 = ax2.bar(self.regions, real_wages, color='orange', alpha=0.7)
        ax2.set_title('Real Wages in Manufacturing')
        ax2.set_ylabel('Real Wage')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, real_wages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Regional output
        ax3 = axes[1, 0]
        regional_output = self.equilibrium['regional_output']
        bars3 = ax3.bar(self.regions, regional_output, color='green', alpha=0.7)
        ax3.set_title('Total Regional Output')
        ax3.set_ylabel('Output')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, regional_output):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Transport costs heatmap
        ax4 = axes[1, 1]
        im = ax4.imshow(self.transport_costs, cmap='YlOrRd', aspect='auto')
        ax4.set_title('Transportation Costs Matrix')
        ax4.set_xticks(range(len(self.regions)))
        ax4.set_yticks(range(len(self.regions)))
        ax4.set_xticklabels(self.regions, rotation=45)
        ax4.set_yticklabels(self.regions)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        # Add values to heatmap
        for i in range(len(self.regions)):
            for j in range(len(self.regions)):
                text = ax4.text(j, i, f'{self.transport_costs[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spatial distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_migration_dynamics(self, dynamics_data: pd.DataFrame, save_path: str = None):
        """
        Plot migration dynamics over time
        
        Args:
            dynamics_data: Migration dynamics data
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Manufacturing labor share evolution
        ax1 = axes[0]
        for region in self.regions:
            region_data = dynamics_data[dynamics_data['region'] == region]
            ax1.plot(region_data['period'], region_data['manufacturing_labor_share'], 
                    label=region, marker='o', markersize=3)
        
        ax1.set_title('Evolution of Manufacturing Labor Shares')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Manufacturing Labor Share')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Real wage evolution
        ax2 = axes[1]
        for region in self.regions:
            region_data = dynamics_data[dynamics_data['region'] == region]
            ax2.plot(region_data['period'], region_data['real_wage_manufacturing'], 
                    label=region, marker='s', markersize=3)
        
        ax2.set_title('Evolution of Real Wages in Manufacturing')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Real Wage')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Migration dynamics plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'New Economic Geography (NEG) Model',
            'country': 'Bangladesh',
            'regions': self.regions,
            'n_regions': self.n_regions,
            'parameters': {
                'elasticity_substitution': self.params.sigma,
                'manufacturing_share': self.params.mu,
                'transport_cost_range': [self.params.tau_min, self.params.tau_max],
                'agglomeration_strength': self.params.agglomeration_strength,
                'migration_elasticity': self.params.migration_elasticity
            }
        }
        
        if self.equilibrium is not None:
            agglomeration_idx = self._compute_agglomeration_index(
                self.equilibrium['manufacturing_labor_share']
            )
            
            summary['equilibrium_indicators'] = {
                'herfindahl_index': agglomeration_idx['herfindahl_index'],
                'core_periphery_index': agglomeration_idx['core_periphery_index'],
                'max_regional_share': agglomeration_idx['max_regional_share'],
                'dominant_region': self.regions[np.argmax(self.equilibrium['manufacturing_labor_share'])]
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'sigma': 5.0,
            'mu': 0.4,
            'tau': 1.5,
            'agglomeration_strength': 0.3,
            'migration_elasticity': 2.0
        }
    }
    
    # Initialize NEG model
    neg_model = NewEconomicGeographyModel(config)
    
    print(f"NEG model initialized for {neg_model.n_regions} regions:")
    print(f"Regions: {neg_model.regions}")
    
    # Compute equilibrium
    print("\nComputing spatial equilibrium...")
    equilibrium = neg_model.compute_equilibrium()
    
    print("\nEquilibrium Results:")
    print("Manufacturing Labor Shares:")
    for i, region in enumerate(neg_model.regions):
        share = equilibrium['manufacturing_labor_share'][i]
        wage = equilibrium['real_wages_manufacturing'][i]
        print(f"  {region}: {share:.3f} (Real wage: {wage:.3f})")
    
    # Agglomeration analysis
    agglomeration_idx = neg_model._compute_agglomeration_index(
        equilibrium['manufacturing_labor_share']
    )
    print(f"\nAgglomeration Indices:")
    print(f"  Herfindahl Index: {agglomeration_idx['herfindahl_index']:.3f}")
    print(f"  Core-Periphery Index: {agglomeration_idx['core_periphery_index']:.3f}")
    print(f"  Gini Coefficient: {agglomeration_idx['gini_coefficient']:.3f}")
    
    # Migration dynamics
    print("\nSimulating migration dynamics...")
    dynamics = neg_model.simulate_migration_dynamics(
        periods=30, shock_region='Sylhet', shock_size=0.2
    )
    print(f"Migration simulation completed: {len(dynamics)} observations")
    
    # Infrastructure policy analysis
    print("\nAnalyzing infrastructure policy...")
    infrastructure_improvements = {
        'Rangpur': 0.5,  # 50% improvement in infrastructure
        'Barisal': 0.3   # 30% improvement
    }
    
    policy_impact = neg_model.analyze_infrastructure_policy(infrastructure_improvements)
    
    print("Policy Impact on Manufacturing Labor Shares:")
    for i, region in enumerate(neg_model.regions):
        baseline = policy_impact['baseline']['manufacturing_labor_share'][i]
        improved = policy_impact['improved']['manufacturing_labor_share'][i]
        change = policy_impact['changes']['manufacturing_labor_share']['relative'][i]
        print(f"  {region}: {baseline:.3f} → {improved:.3f} ({change:+.1f}%)")
    
    # Trade flows
    print("\nComputing trade flows...")
    trade_flows = neg_model.compute_trade_flows()
    
    # Show top trade flows
    top_flows = trade_flows.nlargest(5, 'flow')
    print("Top 5 Trade Flows:")
    for _, flow in top_flows.iterrows():
        print(f"  {flow['origin']} → {flow['destination']}: {flow['flow']:.4f}")
    
    # Model summary
    summary = neg_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Regions: {summary['n_regions']}")
    print(f"  Dominant region: {summary['equilibrium_indicators']['dominant_region']}")
    print(f"  Concentration (Herfindahl): {summary['equilibrium_indicators']['herfindahl_index']:.3f}")