#!/usr/bin/env python3
"""
Agent-Based Model (ABM) for Bangladesh Economy

This module implements a comprehensive agent-based model to simulate complex
interactions between heterogeneous economic agents and analyze emergent
macroeconomic phenomena in Bangladesh. The model captures micro-level
behaviors that aggregate to macro-level outcomes.

Key Features:
- Heterogeneous households, firms, banks, and government
- Spatial economic geography
- Network effects and social interactions
- Adaptive learning and evolution
- Market formation and price discovery
- Financial intermediation and credit markets
- Labor market dynamics
- Innovation and technology diffusion
- Policy transmission mechanisms
- Emergent macroeconomic patterns

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, spatial
from scipy.optimize import minimize
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import random
from collections import defaultdict, deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ABMParameters:
    """
    Parameters for the Agent-Based Model
    """
    # Population parameters
    n_households: int = 5000                    # Number of household agents
    n_firms: int = 500                          # Number of firm agents
    n_banks: int = 20                           # Number of bank agents
    n_regions: int = 8                          # Number of regions (divisions)
    
    # Spatial parameters
    spatial_decay: float = 0.1                 # Spatial interaction decay
    migration_threshold: float = 0.2           # Threshold for migration decisions
    transport_cost_factor: float = 0.05        # Transportation cost factor
    
    # Network parameters
    household_network_degree: int = 8          # Average household network connections
    firm_network_degree: int = 12              # Average firm network connections
    network_rewiring_prob: float = 0.1         # Network rewiring probability
    
    # Learning parameters
    learning_rate: float = 0.1                 # Agent learning rate
    memory_length: int = 20                    # Length of agent memory
    imitation_probability: float = 0.3         # Probability of imitating successful agents
    
    # Market parameters
    price_adjustment_speed: float = 0.2        # Speed of price adjustment
    market_power_factor: float = 0.1           # Market power influence on pricing
    search_cost: float = 0.01                  # Cost of searching for trading partners
    
    # Financial parameters
    bank_capital_ratio: float = 0.12           # Bank capital adequacy ratio
    credit_risk_premium: float = 0.03          # Credit risk premium
    interbank_rate: float = 0.06               # Interbank lending rate
    
    # Innovation parameters
    innovation_probability: float = 0.05       # Probability of innovation
    technology_diffusion_rate: float = 0.15    # Rate of technology diffusion
    rd_investment_share: float = 0.02          # R&D investment share of revenue
    
    # Government parameters
    tax_collection_efficiency: float = 0.7     # Tax collection efficiency
    public_investment_share: float = 0.06      # Public investment as share of GDP
    social_transfer_rate: float = 0.03         # Social transfers as share of GDP
    
    # Bangladesh-specific parameters
    informal_sector_share: float = 0.85        # Share of informal sector
    rural_urban_ratio: float = 1.86            # Rural to urban population ratio
    remittance_share: float = 0.06             # Remittances as share of GDP
    export_orientation: float = 0.15           # Export orientation of economy

@dataclass
class ABMResults:
    """
    Results from Agent-Based Model
    """
    parameters: ABMParameters
    time_series_data: Optional[pd.DataFrame] = None
    agent_data: Optional[pd.DataFrame] = None
    network_data: Optional[Dict] = None
    spatial_data: Optional[Dict] = None
    emergent_patterns: Optional[Dict] = None

class BaseAgent(ABC):
    """
    Abstract base class for all agents
    """
    
    def __init__(self, agent_id: int, agent_type: str, location: Tuple[float, float],
                 region: int, params: ABMParameters):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (household, firm, bank, government)
            location: Spatial coordinates (x, y)
            region: Region identifier
            params: Model parameters
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.location = location
        self.region = region
        self.params = params
        
        # Common agent attributes
        self.wealth = 0.0
        self.income = 0.0
        self.memory = deque(maxlen=params.memory_length)
        self.network = set()
        self.active = True
        
        # Learning and adaptation
        self.strategy = 'default'
        self.performance_history = deque(maxlen=params.memory_length)
        self.last_performance = 0.0
    
    @abstractmethod
    def step(self, model_state: Dict) -> Dict:
        """
        Execute one time step of agent behavior
        
        Args:
            model_state: Current state of the model
            
        Returns:
            Agent actions and decisions
        """
        pass
    
    def update_memory(self, observation: Dict):
        """
        Update agent memory with new observation
        
        Args:
            observation: New observation to store
        """
        self.memory.append(observation)
    
    def learn_from_neighbors(self, neighbors: List['BaseAgent']):
        """
        Learn from neighboring agents
        
        Args:
            neighbors: List of neighboring agents
        """
        if not neighbors or random.random() > self.params.imitation_probability:
            return
        
        # Find best performing neighbor
        best_neighbor = max(neighbors, key=lambda x: x.last_performance, default=None)
        
        if best_neighbor and best_neighbor.last_performance > self.last_performance:
            # Imitate successful neighbor's strategy
            self.strategy = best_neighbor.strategy
    
    def get_spatial_distance(self, other_agent: 'BaseAgent') -> float:
        """
        Calculate spatial distance to another agent
        
        Args:
            other_agent: Target agent
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((self.location[0] - other_agent.location[0])**2 + 
                      (self.location[1] - other_agent.location[1])**2)

class Household(BaseAgent):
    """
    Household agent
    """
    
    def __init__(self, agent_id: int, location: Tuple[float, float], 
                 region: int, params: ABMParameters):
        super().__init__(agent_id, 'household', location, region, params)
        
        # Household characteristics
        self.size = max(1, int(np.random.normal(4.5, 1.5)))  # Bangladesh average
        self.education_level = np.random.beta(2, 5)  # Education distribution
        self.age = max(18, int(np.random.normal(35, 12)))
        self.sector = 'informal' if np.random.random() < params.informal_sector_share else 'formal'
        
        # Economic variables
        self.consumption = 0.0
        self.savings = np.random.exponential(1000)  # Initial savings
        self.debt = 0.0
        self.labor_supply = 1.0
        self.wage = 0.0
        self.employed = True
        
        # Behavioral parameters
        self.consumption_propensity = np.random.normal(0.8, 0.1)
        self.risk_aversion = np.random.gamma(2, 1)
        self.social_influence = np.random.beta(3, 2)
        
        # Location-specific factors
        self.is_urban = region in [0, 1, 2]  # Dhaka, Chittagong, Sylhet as urban
        self.migration_propensity = np.random.beta(1, 4) if not self.is_urban else np.random.beta(1, 8)
    
    def step(self, model_state: Dict) -> Dict:
        """
        Household decision making for one period
        
        Args:
            model_state: Current model state
            
        Returns:
            Household decisions
        """
        decisions = {
            'consumption': 0,
            'saving': 0,
            'labor_supply': 0,
            'migration': False,
            'credit_demand': 0
        }
        
        # Labor supply decision
        unemployment_rate = model_state.get('unemployment_rate', 0.04)
        if self.employed or np.random.random() > unemployment_rate:
            self.employed = True
            decisions['labor_supply'] = self.labor_supply
            
            # Wage determination (simplified)
            regional_wage = model_state.get(f'wage_region_{self.region}', 15000)
            skill_premium = 1 + self.education_level * 0.5
            self.wage = regional_wage * skill_premium
            self.income = self.wage * self.labor_supply
        else:
            self.employed = False
            self.income = model_state.get('unemployment_benefit', 0)
        
        # Consumption decision
        total_resources = self.income + self.savings * 0.1  # Can access 10% of savings
        
        # Social influence on consumption
        network_consumption = self._get_network_average('consumption')
        if network_consumption > 0:
            social_adjustment = self.social_influence * (network_consumption - self.consumption) * 0.1
        else:
            social_adjustment = 0
        
        target_consumption = (self.consumption_propensity * total_resources + social_adjustment)
        self.consumption = max(0, min(target_consumption, total_resources))
        decisions['consumption'] = self.consumption
        
        # Saving decision
        saving = self.income - self.consumption
        self.savings += saving
        decisions['saving'] = saving
        
        # Credit demand
        if self.consumption > self.income and self.savings < 1000:
            decisions['credit_demand'] = min(5000, self.consumption - self.income)
        
        # Migration decision
        if self._should_migrate(model_state):
            decisions['migration'] = True
        
        # Update performance
        utility = np.log(max(1, self.consumption)) - 0.5 * self.risk_aversion * (saving / self.income)**2 if self.income > 0 else 0
        self.last_performance = utility
        self.performance_history.append(utility)
        
        return decisions
    
    def _get_network_average(self, variable: str) -> float:
        """
        Get average value of variable from network
        
        Args:
            variable: Variable name
            
        Returns:
            Network average
        """
        if not self.network:
            return 0
        
        values = [getattr(agent, variable, 0) for agent in self.network 
                 if hasattr(agent, variable)]
        return np.mean(values) if values else 0
    
    def _should_migrate(self, model_state: Dict) -> bool:
        """
        Decide whether to migrate to another region
        
        Args:
            model_state: Current model state
            
        Returns:
            Migration decision
        """
        if np.random.random() > self.migration_propensity:
            return False
        
        current_wage = model_state.get(f'wage_region_{self.region}', 15000)
        
        # Find best alternative region
        best_wage = current_wage
        for region in range(self.params.n_regions):
            if region != self.region:
                region_wage = model_state.get(f'wage_region_{region}', 15000)
                if region_wage > best_wage:
                    best_wage = region_wage
        
        # Migrate if wage difference exceeds threshold
        wage_gain = (best_wage - current_wage) / current_wage
        return wage_gain > self.params.migration_threshold

class Firm(BaseAgent):
    """
    Firm agent
    """
    
    def __init__(self, agent_id: int, location: Tuple[float, float], 
                 region: int, params: ABMParameters):
        super().__init__(agent_id, 'firm', location, region, params)
        
        # Firm characteristics
        self.sector = np.random.choice(['agriculture', 'manufacturing', 'services'], 
                                     p=[0.4, 0.2, 0.4])  # Bangladesh sectoral distribution
        self.size = np.random.choice(['micro', 'small', 'medium', 'large'], 
                                   p=[0.7, 0.2, 0.08, 0.02])
        
        # Production parameters
        self.productivity = np.random.lognormal(0, 0.3)
        self.capital = np.random.exponential(50000)
        self.technology_level = np.random.beta(2, 3)
        
        # Economic variables
        self.output = 0.0
        self.revenue = 0.0
        self.costs = 0.0
        self.profit = 0.0
        self.price = np.random.normal(100, 20)
        self.labor_demand = 0
        self.wage_offered = 15000  # Starting wage
        
        # Financial variables
        self.cash = np.random.exponential(10000)
        self.debt = 0.0
        self.credit_demand = 0.0
        
        # Innovation and learning
        self.rd_spending = 0.0
        self.innovation_success = False
        
        # Market variables
        self.market_share = 0.0
        self.customers = set()
        self.suppliers = set()
    
    def step(self, model_state: Dict) -> Dict:
        """
        Firm decision making for one period
        
        Args:
            model_state: Current model state
            
        Returns:
            Firm decisions
        """
        decisions = {
            'production': 0,
            'labor_demand': 0,
            'price': 0,
            'investment': 0,
            'rd_spending': 0,
            'credit_demand': 0
        }
        
        # Production decision
        labor_available = model_state.get('labor_supply', 1000)
        capital_constraint = self.capital
        
        # Cobb-Douglas production function
        alpha = 0.3  # Capital share
        optimal_labor = min(labor_available, 
                          (alpha * self.productivity * capital_constraint**(alpha) / self.wage_offered)**(1/(1-alpha)))
        
        self.labor_demand = max(1, int(optimal_labor))
        decisions['labor_demand'] = self.labor_demand
        
        # Production
        self.output = self.productivity * (capital_constraint**alpha) * (self.labor_demand**(1-alpha))
        decisions['production'] = self.output
        
        # Pricing decision
        demand_factor = model_state.get('aggregate_demand', 1.0)
        competition_factor = model_state.get(f'competition_{self.sector}', 1.0)
        
        # Markup pricing with demand and competition adjustments
        marginal_cost = self.wage_offered / (self.productivity * (1-alpha))
        markup = 1.2 / competition_factor  # Lower markup with more competition
        
        target_price = marginal_cost * markup * demand_factor
        price_adjustment = self.params.price_adjustment_speed * (target_price - self.price)
        self.price = max(0.1, self.price + price_adjustment)
        decisions['price'] = self.price
        
        # Revenue and costs
        self.revenue = self.output * self.price
        labor_cost = self.labor_demand * self.wage_offered
        capital_cost = self.capital * 0.1  # Depreciation and maintenance
        self.costs = labor_cost + capital_cost
        
        # R&D spending
        if self.revenue > 0:
            self.rd_spending = self.revenue * self.params.rd_investment_share
            decisions['rd_spending'] = self.rd_spending
            self.costs += self.rd_spending
        
        # Profit
        self.profit = self.revenue - self.costs
        
        # Investment decision
        if self.profit > 0:
            investment_rate = 0.2 if self.profit > self.revenue * 0.1 else 0.1
            investment = self.profit * investment_rate
            self.capital += investment
            decisions['investment'] = investment
        
        # Credit demand
        if self.cash < self.costs and self.profit < 0:
            decisions['credit_demand'] = min(self.costs - self.cash, self.capital * 0.5)
        
        # Innovation
        self._attempt_innovation()
        
        # Update performance
        self.last_performance = self.profit / max(1, self.revenue)  # Profit margin
        self.performance_history.append(self.last_performance)
        
        return decisions
    
    def _attempt_innovation(self):
        """
        Attempt innovation based on R&D spending
        """
        innovation_prob = min(0.5, self.rd_spending / 10000 * self.params.innovation_probability)
        
        if np.random.random() < innovation_prob:
            self.innovation_success = True
            # Productivity improvement
            productivity_gain = np.random.normal(0.1, 0.05)
            self.productivity *= (1 + productivity_gain)
            self.technology_level = min(1.0, self.technology_level + 0.1)
        else:
            self.innovation_success = False
    
    def adopt_technology(self, technology_level: float):
        """
        Adopt technology from other firms
        
        Args:
            technology_level: Technology level to potentially adopt
        """
        if technology_level > self.technology_level:
            adoption_prob = self.params.technology_diffusion_rate
            if np.random.random() < adoption_prob:
                # Partial technology adoption
                improvement = (technology_level - self.technology_level) * 0.5
                self.technology_level += improvement
                self.productivity *= (1 + improvement * 0.1)

class Bank(BaseAgent):
    """
    Bank agent
    """
    
    def __init__(self, agent_id: int, location: Tuple[float, float], 
                 region: int, params: ABMParameters):
        super().__init__(agent_id, 'bank', location, region, params)
        
        # Bank characteristics
        self.bank_type = np.random.choice(['commercial', 'islamic', 'specialized'], 
                                        p=[0.7, 0.2, 0.1])
        
        # Balance sheet
        self.deposits = np.random.exponential(100000000)  # 100M average
        self.loans = self.deposits * 0.7  # Initial loan-to-deposit ratio
        self.capital = self.deposits * self.params.bank_capital_ratio
        self.reserves = self.deposits * 0.1
        
        # Interest rates
        self.deposit_rate = 0.05
        self.lending_rate = 0.12
        
        # Risk management
        self.loan_loss_provision = 0.02
        self.risk_appetite = np.random.beta(3, 2)
        
        # Performance metrics
        self.net_interest_income = 0.0
        self.loan_defaults = 0.0
    
    def step(self, model_state: Dict) -> Dict:
        """
        Bank decision making for one period
        
        Args:
            model_state: Current model state
            
        Returns:
            Bank decisions
        """
        decisions = {
            'lending_rate': 0,
            'deposit_rate': 0,
            'credit_supply': 0,
            'loan_approvals': []
        }
        
        # Interest rate setting
        central_bank_rate = model_state.get('policy_rate', 0.06)
        credit_risk = model_state.get('credit_risk', 0.02)
        
        self.deposit_rate = central_bank_rate + 0.01
        self.lending_rate = central_bank_rate + self.params.credit_risk_premium + credit_risk
        
        decisions['lending_rate'] = self.lending_rate
        decisions['deposit_rate'] = self.deposit_rate
        
        # Credit supply decision
        capital_constraint = self.capital / self.params.bank_capital_ratio
        liquidity_constraint = self.deposits - self.reserves
        
        max_lending = min(capital_constraint, liquidity_constraint)
        current_lending_capacity = max_lending - self.loans
        
        decisions['credit_supply'] = max(0, current_lending_capacity)
        
        # Process loan applications (simplified)
        loan_applications = model_state.get('loan_applications', [])
        approved_loans = []
        
        for application in loan_applications:
            if self._evaluate_loan_application(application):
                approved_loans.append(application)
        
        decisions['loan_approvals'] = approved_loans
        
        # Update balance sheet
        self.net_interest_income = self.loans * self.lending_rate - self.deposits * self.deposit_rate
        
        # Handle loan defaults (simplified)
        default_rate = model_state.get('default_rate', 0.02)
        self.loan_defaults = self.loans * default_rate
        self.loans -= self.loan_defaults
        
        # Update performance
        self.last_performance = self.net_interest_income / max(1, self.capital)
        self.performance_history.append(self.last_performance)
        
        return decisions
    
    def _evaluate_loan_application(self, application: Dict) -> bool:
        """
        Evaluate loan application
        
        Args:
            application: Loan application details
            
        Returns:
            Approval decision
        """
        # Simplified credit scoring
        applicant_income = application.get('income', 0)
        loan_amount = application.get('amount', 0)
        collateral_value = application.get('collateral', 0)
        
        # Debt-to-income ratio
        debt_to_income = loan_amount / max(1, applicant_income)
        
        # Loan-to-value ratio
        loan_to_value = loan_amount / max(1, collateral_value)
        
        # Risk assessment
        risk_score = debt_to_income * 0.4 + loan_to_value * 0.3 + np.random.normal(0, 0.1)
        
        # Approval threshold based on bank's risk appetite
        approval_threshold = 1.0 - self.risk_appetite * 0.5
        
        return risk_score < approval_threshold

class Government(BaseAgent):
    """
    Government agent
    """
    
    def __init__(self, params: ABMParameters):
        super().__init__(0, 'government', (0.5, 0.5), 0, params)
        
        # Fiscal variables
        self.tax_revenue = 0.0
        self.government_spending = 0.0
        self.public_debt = 0.0
        self.budget_deficit = 0.0
        
        # Policy instruments
        self.tax_rate = 0.15
        self.public_investment_rate = params.public_investment_share
        self.social_transfer_rate = params.social_transfer_rate
        
        # Policy targets
        self.inflation_target = 0.055
        self.unemployment_target = 0.04
        self.growth_target = 0.07
    
    def step(self, model_state: Dict) -> Dict:
        """
        Government policy decisions
        
        Args:
            model_state: Current model state
            
        Returns:
            Policy decisions
        """
        decisions = {
            'tax_rate': self.tax_rate,
            'government_spending': 0,
            'social_transfers': 0,
            'public_investment': 0,
            'policy_rate': 0.06
        }
        
        # Fiscal policy
        gdp = model_state.get('gdp', 100000000)
        
        # Tax collection
        tax_base = gdp * 0.8  # Not all GDP is taxable
        self.tax_revenue = tax_base * self.tax_rate * self.params.tax_collection_efficiency
        
        # Government spending components
        public_investment = gdp * self.public_investment_rate
        social_transfers = gdp * self.social_transfer_rate
        other_spending = self.tax_revenue * 0.6  # Administrative and other expenses
        
        self.government_spending = public_investment + social_transfers + other_spending
        decisions['government_spending'] = self.government_spending
        decisions['social_transfers'] = social_transfers
        decisions['public_investment'] = public_investment
        
        # Budget balance
        self.budget_deficit = self.government_spending - self.tax_revenue
        self.public_debt += self.budget_deficit
        
        # Monetary policy (simplified)
        inflation_rate = model_state.get('inflation_rate', 0.05)
        unemployment_rate = model_state.get('unemployment_rate', 0.04)
        
        # Taylor rule
        neutral_rate = 0.06
        inflation_gap = inflation_rate - self.inflation_target
        unemployment_gap = unemployment_rate - self.unemployment_target
        
        policy_rate = neutral_rate + 1.5 * inflation_gap - 0.5 * unemployment_gap
        policy_rate = max(0.01, min(0.15, policy_rate))  # Bounds
        
        decisions['policy_rate'] = policy_rate
        
        return decisions

class AgentBasedModel:
    """
    Agent-Based Model for Bangladesh Economy
    
    This class coordinates all agents and simulates the macroeconomy
    through agent interactions and emergent phenomena.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Agent-Based Model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = ABMParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Agents
        self.households = []
        self.firms = []
        self.banks = []
        self.government = None
        
        # Networks
        self.household_network = None
        self.firm_network = None
        self.spatial_network = None
        
        # Model state
        self.current_period = 0
        self.model_state = {}
        self.time_series_data = []
        
        # Spatial structure
        self.regions = self._create_spatial_structure()
        
        logger.info("Agent-Based Model initialized for Bangladesh")
    
    def _create_spatial_structure(self) -> Dict:
        """
        Create spatial structure representing Bangladesh regions
        
        Returns:
            Regional structure
        """
        # Bangladesh divisions as regions
        region_names = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 
                       'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh']
        
        regions = {}
        for i, name in enumerate(region_names):
            regions[i] = {
                'name': name,
                'center': (np.random.uniform(0, 1), np.random.uniform(0, 1)),
                'population': 0,
                'gdp': 0,
                'wage_level': 15000 + np.random.normal(0, 3000),
                'development_level': np.random.beta(2, 3)
            }
        
        return regions
    
    def initialize_agents(self):
        """
        Initialize all agents in the model
        """
        logger.info("Initializing agents")
        
        # Initialize households
        for i in range(self.params.n_households):
            location = (np.random.uniform(0, 1), np.random.uniform(0, 1))
            region = np.random.choice(self.params.n_regions, 
                                    p=self._get_regional_population_weights())
            
            household = Household(i, location, region, self.params)
            self.households.append(household)
            self.regions[region]['population'] += 1
        
        # Initialize firms
        for i in range(self.params.n_firms):
            location = (np.random.uniform(0, 1), np.random.uniform(0, 1))
            region = np.random.choice(self.params.n_regions,
                                    p=self._get_regional_economic_weights())
            
            firm = Firm(i, location, region, self.params)
            self.firms.append(firm)
        
        # Initialize banks
        for i in range(self.params.n_banks):
            location = (np.random.uniform(0, 1), np.random.uniform(0, 1))
            region = np.random.choice(self.params.n_regions,
                                    p=self._get_regional_economic_weights())
            
            bank = Bank(i, location, region, self.params)
            self.banks.append(bank)
        
        # Initialize government
        self.government = Government(self.params)
        
        # Create networks
        self._create_networks()
        
        logger.info(f"Initialized {len(self.households)} households, {len(self.firms)} firms, {len(self.banks)} banks")
    
    def _get_regional_population_weights(self) -> np.ndarray:
        """
        Get population weights for regional distribution
        
        Returns:
            Population weights array
        """
        # Based on Bangladesh division populations (approximate)
        weights = np.array([0.25, 0.18, 0.12, 0.10, 0.08, 0.08, 0.10, 0.09])
        return weights / weights.sum()
    
    def _get_regional_economic_weights(self) -> np.ndarray:
        """
        Get economic activity weights for regional distribution
        
        Returns:
            Economic weights array
        """
        # Economic activity more concentrated in Dhaka and Chittagong
        weights = np.array([0.35, 0.25, 0.10, 0.08, 0.05, 0.07, 0.06, 0.04])
        return weights / weights.sum()
    
    def _create_networks(self):
        """
        Create social and economic networks among agents
        """
        logger.info("Creating agent networks")
        
        # Household social network
        self.household_network = self._create_small_world_network(
            self.households, self.params.household_network_degree)
        
        # Firm business network
        self.firm_network = self._create_small_world_network(
            self.firms, self.params.firm_network_degree)
        
        # Spatial network (based on geographic proximity)
        self.spatial_network = self._create_spatial_network()
    
    def _create_small_world_network(self, agents: List[BaseAgent], 
                                   avg_degree: int) -> nx.Graph:
        """
        Create small-world network among agents
        
        Args:
            agents: List of agents
            avg_degree: Average network degree
            
        Returns:
            NetworkX graph
        """
        n_agents = len(agents)
        G = nx.watts_strogatz_graph(n_agents, avg_degree, 
                                   self.params.network_rewiring_prob)
        
        # Assign network connections to agents
        for i, agent in enumerate(agents):
            neighbors = list(G.neighbors(i))
            agent.network = {agents[j] for j in neighbors if j < len(agents)}
        
        return G
    
    def _create_spatial_network(self) -> nx.Graph:
        """
        Create spatial network based on geographic proximity
        
        Returns:
            Spatial network graph
        """
        all_agents = self.households + self.firms + self.banks
        n_agents = len(all_agents)
        
        G = nx.Graph()
        G.add_nodes_from(range(n_agents))
        
        # Add edges based on spatial proximity
        for i, agent1 in enumerate(all_agents):
            for j, agent2 in enumerate(all_agents[i+1:], i+1):
                distance = agent1.get_spatial_distance(agent2)
                
                # Connection probability decreases with distance
                connection_prob = np.exp(-distance / self.params.spatial_decay)
                
                if np.random.random() < connection_prob:
                    G.add_edge(i, j, weight=1/distance)
        
        return G
    
    def simulate(self, periods: int = 100) -> pd.DataFrame:
        """
        Run the agent-based simulation
        
        Args:
            periods: Number of periods to simulate
            
        Returns:
            Time series data
        """
        if not self.households:
            self.initialize_agents()
        
        logger.info(f"Starting ABM simulation for {periods} periods")
        
        for period in range(periods):
            self.current_period = period
            
            # Update model state
            self._update_model_state()
            
            # Agent decisions
            household_decisions = self._execute_household_decisions()
            firm_decisions = self._execute_firm_decisions()
            bank_decisions = self._execute_bank_decisions()
            government_decisions = self._execute_government_decisions()
            
            # Market clearing and interactions
            self._clear_markets(household_decisions, firm_decisions, bank_decisions)
            
            # Technology diffusion
            self._diffuse_technology()
            
            # Network evolution
            if period % 10 == 0:  # Update networks every 10 periods
                self._evolve_networks()
            
            # Collect aggregate data
            aggregate_data = self._collect_aggregate_data(period)
            self.time_series_data.append(aggregate_data)
            
            # Learning and adaptation
            self._agent_learning()
            
            if period % 20 == 0:
                logger.info(f"Completed period {period}, GDP: {aggregate_data['gdp']:,.0f}")
        
        return pd.DataFrame(self.time_series_data)
    
    def _update_model_state(self):
        """
        Update global model state variables
        """
        # Labor market
        total_labor_supply = sum(h.labor_supply for h in self.households if h.employed)
        total_labor_demand = sum(f.labor_demand for f in self.firms)
        
        unemployment_rate = max(0, 1 - total_labor_demand / max(1, total_labor_supply))
        
        # Regional wages
        for region in range(self.params.n_regions):
            regional_firms = [f for f in self.firms if f.region == region]
            if regional_firms:
                avg_wage = np.mean([f.wage_offered for f in regional_firms])
                self.model_state[f'wage_region_{region}'] = avg_wage
        
        # Aggregate demand
        total_consumption = sum(h.consumption for h in self.households)
        total_investment = sum(f.capital for f in self.firms) * 0.1  # Investment rate
        government_spending = getattr(self.government, 'government_spending', 0)
        
        aggregate_demand = total_consumption + total_investment + government_spending
        
        # Update model state
        self.model_state.update({
            'unemployment_rate': unemployment_rate,
            'aggregate_demand': aggregate_demand,
            'total_labor_supply': total_labor_supply,
            'total_labor_demand': total_labor_demand,
            'period': self.current_period
        })
    
    def _execute_household_decisions(self) -> List[Dict]:
        """
        Execute household decisions for current period
        
        Returns:
            List of household decisions
        """
        decisions = []
        for household in self.households:
            decision = household.step(self.model_state)
            decisions.append(decision)
        
        return decisions
    
    def _execute_firm_decisions(self) -> List[Dict]:
        """
        Execute firm decisions for current period
        
        Returns:
            List of firm decisions
        """
        decisions = []
        for firm in self.firms:
            decision = firm.step(self.model_state)
            decisions.append(decision)
        
        return decisions
    
    def _execute_bank_decisions(self) -> List[Dict]:
        """
        Execute bank decisions for current period
        
        Returns:
            List of bank decisions
        """
        decisions = []
        for bank in self.banks:
            decision = bank.step(self.model_state)
            decisions.append(decision)
        
        return decisions
    
    def _execute_government_decisions(self) -> Dict:
        """
        Execute government decisions for current period
        
        Returns:
            Government decisions
        """
        return self.government.step(self.model_state)
    
    def _clear_markets(self, household_decisions: List[Dict], 
                      firm_decisions: List[Dict], bank_decisions: List[Dict]):
        """
        Clear markets and determine equilibrium prices and quantities
        
        Args:
            household_decisions: Household decisions
            firm_decisions: Firm decisions
            bank_decisions: Bank decisions
        """
        # Labor market clearing (simplified)
        total_labor_demand = sum(d['labor_demand'] for d in firm_decisions)
        total_labor_supply = sum(d['labor_supply'] for d in household_decisions)
        
        if total_labor_demand < total_labor_supply:
            # Unemployment - some households don't find jobs
            employment_rate = total_labor_demand / total_labor_supply
            for i, household in enumerate(self.households):
                if np.random.random() > employment_rate:
                    household.employed = False
        
        # Credit market clearing
        total_credit_demand = (sum(d.get('credit_demand', 0) for d in household_decisions) +
                              sum(d.get('credit_demand', 0) for d in firm_decisions))
        total_credit_supply = sum(d.get('credit_supply', 0) for d in bank_decisions)
        
        if total_credit_demand > total_credit_supply:
            # Credit rationing
            rationing_factor = total_credit_supply / max(1, total_credit_demand)
            # Apply rationing (simplified)
            pass
    
    def _diffuse_technology(self):
        """
        Implement technology diffusion among firms
        """
        # Find firms with highest technology levels
        tech_leaders = sorted(self.firms, key=lambda f: f.technology_level, reverse=True)[:10]
        
        for firm in self.firms:
            if firm not in tech_leaders:
                # Find nearby tech leaders
                nearby_leaders = [leader for leader in tech_leaders 
                                if firm.get_spatial_distance(leader) < 0.3]
                
                if nearby_leaders:
                    best_tech = max(leader.technology_level for leader in nearby_leaders)
                    firm.adopt_technology(best_tech)
    
    def _evolve_networks(self):
        """
        Evolve agent networks over time
        """
        # Rewire some network connections
        for household in self.households:
            if np.random.random() < self.params.network_rewiring_prob:
                # Remove a random connection
                if household.network:
                    old_connection = np.random.choice(list(household.network))
                    household.network.remove(old_connection)
                    old_connection.network.discard(household)
                
                # Add a new connection (preferential attachment)
                potential_connections = [h for h in self.households 
                                       if h != household and h not in household.network]
                if potential_connections:
                    # Prefer connections to successful agents
                    weights = [h.last_performance + 1 for h in potential_connections]
                    new_connection = np.random.choice(potential_connections, 
                                                    p=np.array(weights)/sum(weights))
                    household.network.add(new_connection)
                    new_connection.network.add(household)
    
    def _agent_learning(self):
        """
        Implement agent learning and adaptation
        """
        # Households learn from neighbors
        for household in self.households:
            neighbors = list(household.network)
            household.learn_from_neighbors(neighbors)
        
        # Firms learn from competitors
        for firm in self.firms:
            # Find firms in same sector and region
            competitors = [f for f in self.firms 
                          if f.sector == firm.sector and f.region == firm.region and f != firm]
            firm.learn_from_neighbors(competitors[:5])  # Learn from top 5 competitors
    
    def _collect_aggregate_data(self, period: int) -> Dict:
        """
        Collect aggregate economic data for current period
        
        Args:
            period: Current period
            
        Returns:
            Aggregate data dictionary
        """
        # Household aggregates
        total_consumption = sum(h.consumption for h in self.households)
        total_savings = sum(h.savings for h in self.households)
        total_household_income = sum(h.income for h in self.households)
        
        # Firm aggregates
        total_output = sum(f.output for f in self.firms)
        total_investment = sum(f.capital for f in self.firms) * 0.1
        total_firm_revenue = sum(f.revenue for f in self.firms)
        total_profits = sum(f.profit for f in self.firms)
        
        # Bank aggregates
        total_loans = sum(b.loans for b in self.banks)
        total_deposits = sum(b.deposits for b in self.banks)
        avg_lending_rate = np.mean([b.lending_rate for b in self.banks])
        
        # Government
        government_spending = self.government.government_spending
        tax_revenue = self.government.tax_revenue
        public_debt = self.government.public_debt
        
        # Macroeconomic indicators
        gdp = total_consumption + total_investment + government_spending
        unemployment_rate = self.model_state.get('unemployment_rate', 0)
        
        # Inflation (simplified)
        if period > 0 and self.time_series_data:
            prev_gdp = self.time_series_data[-1]['gdp']
            inflation_rate = (gdp - prev_gdp) / prev_gdp if prev_gdp > 0 else 0
        else:
            inflation_rate = 0.05  # Initial inflation
        
        return {
            'period': period,
            'gdp': gdp,
            'consumption': total_consumption,
            'investment': total_investment,
            'government_spending': government_spending,
            'savings': total_savings,
            'unemployment_rate': unemployment_rate,
            'inflation_rate': inflation_rate,
            'total_loans': total_loans,
            'total_deposits': total_deposits,
            'avg_lending_rate': avg_lending_rate,
            'tax_revenue': tax_revenue,
            'public_debt': public_debt,
            'total_profits': total_profits,
            'avg_productivity': np.mean([f.productivity for f in self.firms]),
            'avg_technology': np.mean([f.technology_level for f in self.firms])
        }
    
    def analyze_emergent_patterns(self) -> Dict:
        """
        Analyze emergent patterns from the simulation
        
        Returns:
            Analysis of emergent patterns
        """
        if not self.time_series_data:
            logger.warning("No simulation data available for analysis")
            return {}
        
        logger.info("Analyzing emergent patterns")
        
        df = pd.DataFrame(self.time_series_data)
        
        patterns = {}
        
        # Business cycle patterns
        if len(df) > 20:
            gdp_growth = df['gdp'].pct_change().dropna()
            patterns['business_cycle'] = {
                'avg_growth_rate': gdp_growth.mean(),
                'growth_volatility': gdp_growth.std(),
                'recession_periods': len(gdp_growth[gdp_growth < -0.02]),
                'expansion_periods': len(gdp_growth[gdp_growth > 0.02])
            }
        
        # Inequality patterns
        household_incomes = [h.income for h in self.households]
        if household_incomes:
            gini_coefficient = self._calculate_gini(household_incomes)
            patterns['inequality'] = {
                'gini_coefficient': gini_coefficient,
                'income_ratio_90_10': np.percentile(household_incomes, 90) / max(1, np.percentile(household_incomes, 10)),
                'poverty_rate': sum(1 for income in household_incomes if income < np.median(household_incomes) * 0.6) / len(household_incomes)
            }
        
        # Regional patterns
        regional_data = {}
        for region in range(self.params.n_regions):
            regional_households = [h for h in self.households if h.region == region]
            regional_firms = [f for f in self.firms if f.region == region]
            
            if regional_households and regional_firms:
                regional_data[region] = {
                    'population': len(regional_households),
                    'avg_income': np.mean([h.income for h in regional_households]),
                    'avg_productivity': np.mean([f.productivity for f in regional_firms]),
                    'unemployment_rate': sum(1 for h in regional_households if not h.employed) / len(regional_households)
                }
        
        patterns['regional_patterns'] = regional_data
        
        # Network effects
        if self.household_network:
            patterns['network_effects'] = {
                'avg_clustering': nx.average_clustering(self.household_network),
                'avg_path_length': nx.average_shortest_path_length(self.household_network) if nx.is_connected(self.household_network) else float('inf'),
                'network_density': nx.density(self.household_network)
            }
        
        # Innovation patterns
        innovation_data = {
            'avg_rd_spending': np.mean([f.rd_spending for f in self.firms]),
            'innovation_rate': sum(1 for f in self.firms if f.innovation_success) / len(self.firms),
            'technology_dispersion': np.std([f.technology_level for f in self.firms])
        }
        patterns['innovation_patterns'] = innovation_data
        
        return patterns
    
    def _calculate_gini(self, incomes: List[float]) -> float:
        """
        Calculate Gini coefficient
        
        Args:
            incomes: List of income values
            
        Returns:
            Gini coefficient
        """
        if not incomes or len(incomes) < 2:
            return 0
        
        sorted_incomes = sorted(incomes)
        n = len(sorted_incomes)
        cumsum = np.cumsum(sorted_incomes)
        
        return (n + 1 - 2 * sum((n + 1 - i) * income for i, income in enumerate(sorted_incomes, 1))) / (n * sum(sorted_incomes))
    
    def plot_simulation_results(self, save_path: str = None):
        """
        Plot simulation results
        
        Args:
            save_path: Path to save plot
        """
        if not self.time_series_data:
            logger.warning("No simulation data to plot")
            return
        
        df = pd.DataFrame(self.time_series_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # GDP growth
        axes[0, 0].plot(df['period'], df['gdp'])
        axes[0, 0].set_title('GDP Over Time')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('GDP')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Unemployment rate
        axes[0, 1].plot(df['period'], df['unemployment_rate'] * 100)
        axes[0, 1].set_title('Unemployment Rate')
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Unemployment Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Inflation rate
        axes[0, 2].plot(df['period'], df['inflation_rate'] * 100)
        axes[0, 2].set_title('Inflation Rate')
        axes[0, 2].set_xlabel('Period')
        axes[0, 2].set_ylabel('Inflation Rate (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Consumption and investment
        axes[1, 0].plot(df['period'], df['consumption'], label='Consumption')
        axes[1, 0].plot(df['period'], df['investment'], label='Investment')
        axes[1, 0].set_title('Consumption and Investment')
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].set_ylabel('Amount')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Technology and productivity
        axes[1, 1].plot(df['period'], df['avg_productivity'], label='Avg Productivity')
        axes[1, 1].plot(df['period'], df['avg_technology'], label='Avg Technology')
        axes[1, 1].set_title('Technology and Productivity')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('Level')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Public finances
        axes[1, 2].plot(df['period'], df['tax_revenue'], label='Tax Revenue')
        axes[1, 2].plot(df['period'], df['government_spending'], label='Gov Spending')
        axes[1, 2].plot(df['period'], df['public_debt'], label='Public Debt')
        axes[1, 2].set_title('Public Finances')
        axes[1, 2].set_xlabel('Period')
        axes[1, 2].set_ylabel('Amount')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simulation plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Agent-Based Model',
            'country': 'Bangladesh',
            'agent_counts': {
                'households': len(self.households),
                'firms': len(self.firms),
                'banks': len(self.banks),
                'government': 1 if self.government else 0
            },
            'spatial_structure': {
                'regions': self.params.n_regions,
                'spatial_interactions': True
            },
            'network_structure': {
                'household_network_degree': self.params.household_network_degree,
                'firm_network_degree': self.params.firm_network_degree,
                'network_evolution': True
            },
            'key_features': [
                'Heterogeneous Agents',
                'Spatial Geography',
                'Network Effects',
                'Adaptive Learning',
                'Technology Diffusion',
                'Market Formation',
                'Emergent Patterns'
            ]
        }
        
        if self.time_series_data:
            latest_data = self.time_series_data[-1]
            summary['latest_indicators'] = {
                'gdp': latest_data['gdp'],
                'unemployment_rate': latest_data['unemployment_rate'],
                'inflation_rate': latest_data['inflation_rate'],
                'avg_productivity': latest_data['avg_productivity']
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'n_households': 1000,
            'n_firms': 100,
            'n_banks': 10,
            'n_regions': 8,
            'learning_rate': 0.1,
            'innovation_probability': 0.05,
            'informal_sector_share': 0.85
        }
    }
    
    # Initialize Agent-Based Model
    abm = AgentBasedModel(config)
    
    # Initialize agents
    abm.initialize_agents()
    print(f"Initialized ABM with {len(abm.households)} households, {len(abm.firms)} firms, {len(abm.banks)} banks")
    
    # Run simulation
    print("\nRunning ABM simulation...")
    simulation_data = abm.simulate(periods=50)
    print(f"Simulation completed: {len(simulation_data)} periods")
    
    # Display key results
    latest = simulation_data.iloc[-1]
    print(f"\nFinal Period Results:")
    print(f"  GDP: {latest['gdp']:,.0f}")
    print(f"  Unemployment rate: {latest['unemployment_rate']:.1%}")
    print(f"  Inflation rate: {latest['inflation_rate']:.1%}")
    print(f"  Average productivity: {latest['avg_productivity']:.3f}")
    print(f"  Average technology level: {latest['avg_technology']:.3f}")
    
    # Analyze emergent patterns
    print("\nAnalyzing emergent patterns...")
    patterns = abm.analyze_emergent_patterns()
    
    if 'business_cycle' in patterns:
        bc = patterns['business_cycle']
        print(f"Business Cycle:")
        print(f"  Average growth rate: {bc['avg_growth_rate']:.1%}")
        print(f"  Growth volatility: {bc['growth_volatility']:.3f}")
        print(f"  Recession periods: {bc['recession_periods']}")
    
    if 'inequality' in patterns:
        ineq = patterns['inequality']
        print(f"Inequality:")
        print(f"  Gini coefficient: {ineq['gini_coefficient']:.3f}")
        print(f"  90/10 income ratio: {ineq['income_ratio_90_10']:.1f}")
        print(f"  Poverty rate: {ineq['poverty_rate']:.1%}")
    
    if 'innovation_patterns' in patterns:
        innov = patterns['innovation_patterns']
        print(f"Innovation:")
        print(f"  Average R&D spending: {innov['avg_rd_spending']:,.0f}")
        print(f"  Innovation rate: {innov['innovation_rate']:.1%}")
        print(f"  Technology dispersion: {innov['technology_dispersion']:.3f}")
    
    # Model summary
    summary = abm.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Total agents: {sum(summary['agent_counts'].values())}")
    print(f"  Regions: {summary['spatial_structure']['regions']}")
    print(f"  Key features: {len(summary['key_features'])}")