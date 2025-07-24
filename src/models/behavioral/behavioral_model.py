#!/usr/bin/env python3
"""
Behavioral Economics Model for Bangladesh

This module implements a behavioral economics model that incorporates psychological
factors, bounded rationality, and behavioral biases into macroeconomic analysis
for Bangladesh. The model examines how cognitive limitations and behavioral
patterns affect economic outcomes and policy effectiveness.

Key Features:
- Bounded rationality and limited attention
- Loss aversion and reference dependence
- Present bias and hyperbolic discounting
- Social preferences and fairness concerns
- Overconfidence and optimism bias
- Mental accounting and framing effects
- Herding behavior and social learning
- Behavioral responses to policy interventions
- Cultural and social factors specific to Bangladesh

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, stats
from scipy.special import expit  # sigmoid function
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BehavioralParameters:
    """
    Parameters for the Behavioral Economics model
    """
    # Bounded rationality parameters
    attention_span: float = 0.7                 # Fraction of information processed
    cognitive_capacity: float = 0.8             # Cognitive processing capacity
    information_processing_cost: float = 0.01   # Cost of processing information
    
    # Prospect theory parameters
    loss_aversion: float = 2.25                 # Loss aversion coefficient (Kahneman-Tversky)
    reference_point_adaptation: float = 0.5     # Speed of reference point adaptation
    probability_weighting_alpha: float = 0.69   # Probability weighting parameter
    probability_weighting_beta: float = 0.61    # Probability weighting parameter
    
    # Time preferences
    present_bias: float = 0.7                   # Present bias parameter (beta)
    long_term_discount: float = 0.95            # Long-term discount factor (delta)
    hyperbolic_curvature: float = 1.0           # Hyperbolic discounting curvature
    
    # Social preferences
    fairness_concern: float = 0.3               # Weight on fairness in utility
    reciprocity_strength: float = 0.5           # Strength of reciprocal behavior
    social_comparison_weight: float = 0.2       # Weight on relative consumption
    
    # Overconfidence and optimism
    overconfidence_level: float = 0.15          # Degree of overconfidence
    optimism_bias: float = 0.1                  # Optimism bias in expectations
    self_attribution_bias: float = 0.3          # Self-attribution bias strength
    
    # Mental accounting
    mental_account_categories: int = 4          # Number of mental accounts
    fungibility_parameter: float = 0.6         # Degree of money fungibility
    
    # Herding and social learning
    herding_tendency: float = 0.4               # Tendency to follow others
    social_network_influence: float = 0.3       # Influence of social network
    conformity_pressure: float = 0.25           # Pressure to conform
    
    # Bangladesh-specific cultural factors
    family_influence: float = 0.6               # Influence of family decisions
    religious_influence: float = 0.3            # Influence of religious considerations
    community_trust: float = 0.7                # Level of community trust
    informal_network_strength: float = 0.8      # Strength of informal networks
    
    # Learning and adaptation
    learning_rate: float = 0.1                  # Rate of belief updating
    memory_decay: float = 0.05                  # Rate of memory decay
    experience_weight: float = 0.7              # Weight on personal vs. social experience
    
    # Policy response parameters
    policy_attention_threshold: float = 0.1     # Threshold for policy attention
    policy_credibility_factor: float = 0.6      # Credibility of policy announcements
    implementation_uncertainty: float = 0.2     # Uncertainty about policy implementation

@dataclass
class BehavioralResults:
    """
    Results from Behavioral Economics model
    """
    parameters: BehavioralParameters
    agent_data: Optional[pd.DataFrame] = None
    aggregate_outcomes: Optional[Dict] = None
    behavioral_patterns: Optional[Dict] = None
    policy_effectiveness: Optional[Dict] = None
    welfare_analysis: Optional[Dict] = None

class BehavioralAgent:
    """
    Individual behavioral agent with psychological characteristics
    """
    
    def __init__(self, agent_id: int, params: BehavioralParameters, 
                 characteristics: Dict = None):
        """
        Initialize behavioral agent
        
        Args:
            agent_id: Unique agent identifier
            params: Behavioral parameters
            characteristics: Agent-specific characteristics
        """
        self.agent_id = agent_id
        self.params = params
        
        # Agent characteristics
        if characteristics is None:
            characteristics = self._generate_random_characteristics()
        
        self.income = characteristics.get('income', 1.0)
        self.education = characteristics.get('education', 0.5)
        self.age = characteristics.get('age', 35)
        self.risk_aversion = characteristics.get('risk_aversion', 2.0)
        self.social_class = characteristics.get('social_class', 'middle')
        
        # Behavioral state
        self.reference_point = self.income
        self.beliefs = {'inflation': 0.05, 'growth': 0.06, 'unemployment': 0.04}
        self.mental_accounts = self._initialize_mental_accounts()
        self.social_network = []
        self.experience_history = []
        
        # Decision history
        self.consumption_history = []
        self.saving_history = []
        self.investment_history = []
    
    def _generate_random_characteristics(self) -> Dict:
        """
        Generate random agent characteristics
        """
        return {
            'income': np.random.lognormal(0, 0.5),
            'education': np.random.beta(2, 3),
            'age': np.random.normal(35, 10),
            'risk_aversion': np.random.gamma(2, 1),
            'social_class': np.random.choice(['low', 'middle', 'high'], p=[0.6, 0.3, 0.1])
        }
    
    def _initialize_mental_accounts(self) -> Dict:
        """
        Initialize mental accounting categories
        """
        total_income = self.income
        
        # Typical mental accounts in Bangladesh context
        accounts = {
            'necessities': total_income * 0.6,      # Food, housing, basic needs
            'family_obligations': total_income * 0.15,  # Family support, ceremonies
            'savings': total_income * 0.15,         # Formal and informal savings
            'discretionary': total_income * 0.1     # Entertainment, luxury goods
        }
        
        return accounts
    
    def update_beliefs(self, new_information: Dict, social_information: Dict = None):
        """
        Update beliefs using bounded rationality and social learning
        
        Args:
            new_information: New economic information
            social_information: Information from social network
        """
        # Attention allocation
        attention_weights = self._allocate_attention(new_information)
        
        # Process information with cognitive limitations
        processed_info = self._process_information(new_information, attention_weights)
        
        # Update beliefs with learning rate
        for variable, new_value in processed_info.items():
            if variable in self.beliefs:
                # Bayesian-like updating with bounded rationality
                old_belief = self.beliefs[variable]
                
                # Incorporate overconfidence
                confidence_adjustment = 1 + self.params.overconfidence_level
                learning_rate = self.params.learning_rate / confidence_adjustment
                
                # Social learning component
                if social_information and variable in social_information:
                    social_weight = self.params.social_network_influence
                    social_belief = social_information[variable]
                    
                    # Weighted average of individual and social learning
                    individual_update = old_belief + learning_rate * (new_value - old_belief)
                    social_update = old_belief + social_weight * (social_belief - old_belief)
                    
                    self.beliefs[variable] = (1 - social_weight) * individual_update + social_weight * social_update
                else:
                    self.beliefs[variable] = old_belief + learning_rate * (new_value - old_belief)
                
                # Add optimism bias
                if variable in ['growth', 'income_growth']:
                    self.beliefs[variable] += self.params.optimism_bias
    
    def _allocate_attention(self, information: Dict) -> Dict:
        """
        Allocate limited attention across information sources
        
        Args:
            information: Available information
            
        Returns:
            Attention weights for each piece of information
        """
        # Salience-based attention allocation
        salience_scores = {}
        
        for variable, value in information.items():
            # Higher salience for larger changes and losses
            if variable in self.beliefs:
                change = abs(value - self.beliefs[variable])
                
                # Loss aversion: losses are more salient
                if value < self.beliefs[variable]:
                    change *= self.params.loss_aversion
                
                salience_scores[variable] = change
            else:
                salience_scores[variable] = 1.0
        
        # Normalize to sum to attention span
        total_salience = sum(salience_scores.values())
        if total_salience > 0:
            attention_weights = {k: (v / total_salience) * self.params.attention_span 
                               for k, v in salience_scores.items()}
        else:
            # Equal attention if no salience differences
            n_items = len(information)
            attention_weights = {k: self.params.attention_span / n_items 
                               for k in information.keys()}
        
        return attention_weights
    
    def _process_information(self, information: Dict, attention_weights: Dict) -> Dict:
        """
        Process information with cognitive limitations
        
        Args:
            information: Raw information
            attention_weights: Attention allocation
            
        Returns:
            Processed information
        """
        processed = {}
        
        for variable, value in information.items():
            attention = attention_weights.get(variable, 0)
            
            # Information processing with noise
            processing_noise = np.random.normal(0, 1 - self.params.cognitive_capacity)
            
            # Only process information that receives sufficient attention
            if attention > self.params.policy_attention_threshold:
                processed[variable] = value + processing_noise * attention
            else:
                # Use previous belief if insufficient attention
                processed[variable] = self.beliefs.get(variable, value)
        
        return processed
    
    def make_consumption_decision(self, current_income: float, 
                                 economic_conditions: Dict) -> Dict:
        """
        Make consumption decision using behavioral factors
        
        Args:
            current_income: Current period income
            economic_conditions: Current economic conditions
            
        Returns:
            Consumption decision
        """
        # Update reference point
        self._update_reference_point(current_income)
        
        # Compute utility from different consumption levels
        consumption_options = np.linspace(0.3 * current_income, 1.2 * current_income, 50)
        utilities = []
        
        for consumption in consumption_options:
            utility = self._compute_utility(consumption, current_income, economic_conditions)
            utilities.append(utility)
        
        # Choose consumption with highest utility (with some noise for bounded rationality)
        best_idx = np.argmax(utilities)
        noise = np.random.normal(0, 1 - self.params.cognitive_capacity)
        chosen_idx = max(0, min(len(consumption_options) - 1, 
                               int(best_idx + noise * 5)))
        
        chosen_consumption = consumption_options[chosen_idx]
        
        # Mental accounting allocation
        account_allocation = self._allocate_to_mental_accounts(chosen_consumption)
        
        # Store decision
        decision = {
            'consumption': chosen_consumption,
            'saving': current_income - chosen_consumption,
            'account_allocation': account_allocation,
            'utility': utilities[chosen_idx]
        }
        
        self.consumption_history.append(chosen_consumption)
        self.saving_history.append(current_income - chosen_consumption)
        
        return decision
    
    def _update_reference_point(self, current_income: float):
        """
        Update reference point with adaptation
        
        Args:
            current_income: Current income
        """
        adaptation_rate = self.params.reference_point_adaptation
        self.reference_point = (1 - adaptation_rate) * self.reference_point + adaptation_rate * current_income
    
    def _compute_utility(self, consumption: float, income: float, 
                        economic_conditions: Dict) -> float:
        """
        Compute utility using prospect theory and behavioral factors
        
        Args:
            consumption: Consumption level
            income: Current income
            economic_conditions: Economic conditions
            
        Returns:
            Utility value
        """
        # Basic utility from consumption
        if consumption > 0:
            base_utility = np.log(consumption)
        else:
            base_utility = -np.inf
        
        # Reference dependence and loss aversion
        consumption_relative_to_reference = consumption - self.reference_point
        
        if consumption_relative_to_reference >= 0:
            # Gains
            reference_utility = consumption_relative_to_reference ** 0.88
        else:
            # Losses (with loss aversion)
            reference_utility = -self.params.loss_aversion * ((-consumption_relative_to_reference) ** 0.88)
        
        # Present bias in saving utility
        saving = income - consumption
        if saving > 0:
            # Future utility discounted with present bias
            future_utility = self.params.present_bias * self.params.long_term_discount * np.log(1 + saving)
        else:
            future_utility = 0
        
        # Social comparison
        if hasattr(self, 'social_network') and self.social_network:
            avg_network_consumption = np.mean([agent.consumption_history[-1] 
                                             for agent in self.social_network 
                                             if agent.consumption_history])
            if avg_network_consumption > 0:
                relative_consumption = consumption / avg_network_consumption
                social_utility = self.params.social_comparison_weight * np.log(relative_consumption)
            else:
                social_utility = 0
        else:
            social_utility = 0
        
        # Fairness considerations (simplified)
        fairness_utility = 0
        if 'inequality' in economic_conditions:
            inequality = economic_conditions['inequality']
            # Disutility from high inequality
            fairness_utility = -self.params.fairness_concern * inequality
        
        # Total utility
        total_utility = (base_utility + 0.5 * reference_utility + future_utility + 
                        social_utility + fairness_utility)
        
        return total_utility
    
    def _allocate_to_mental_accounts(self, total_consumption: float) -> Dict:
        """
        Allocate consumption across mental accounts
        
        Args:
            total_consumption: Total consumption amount
            
        Returns:
            Allocation across mental accounts
        """
        # Start with proportional allocation
        total_account_budget = sum(self.mental_accounts.values())
        
        allocation = {}
        remaining = total_consumption
        
        # Prioritize necessities
        necessities_budget = self.mental_accounts['necessities']
        allocation['necessities'] = min(remaining, necessities_budget)
        remaining -= allocation['necessities']
        
        # Family obligations (cultural importance in Bangladesh)
        if remaining > 0:
            family_budget = self.mental_accounts['family_obligations']
            allocation['family_obligations'] = min(remaining, family_budget)
            remaining -= allocation['family_obligations']
        else:
            allocation['family_obligations'] = 0
        
        # Savings (if any remaining)
        if remaining > 0:
            savings_budget = self.mental_accounts['savings']
            allocation['savings'] = min(remaining, savings_budget)
            remaining -= allocation['savings']
        else:
            allocation['savings'] = 0
        
        # Discretionary spending
        allocation['discretionary'] = max(0, remaining)
        
        return allocation
    
    def respond_to_policy(self, policy: Dict) -> Dict:
        """
        Respond to policy intervention with behavioral considerations
        
        Args:
            policy: Policy intervention details
            
        Returns:
            Behavioral response
        """
        response = {
            'attention_given': 0,
            'credibility_assessment': 0,
            'behavioral_change': 0,
            'compliance_probability': 0
        }
        
        # Attention to policy
        policy_salience = policy.get('salience', 0.5)
        response['attention_given'] = min(1.0, policy_salience * self.params.attention_span)
        
        # Credibility assessment
        base_credibility = self.params.policy_credibility_factor
        
        # Adjust for policy type and agent characteristics
        if policy.get('type') == 'monetary':
            # Monetary policy credibility depends on education
            credibility_adjustment = self.education * 0.5
        elif policy.get('type') == 'fiscal':
            # Fiscal policy credibility depends on trust in government
            credibility_adjustment = self.params.community_trust * 0.3
        else:
            credibility_adjustment = 0
        
        response['credibility_assessment'] = min(1.0, base_credibility + credibility_adjustment)
        
        # Behavioral change magnitude
        if response['attention_given'] > self.params.policy_attention_threshold:
            # Change depends on attention, credibility, and policy strength
            policy_strength = policy.get('strength', 0.5)
            
            behavioral_change = (response['attention_given'] * 
                               response['credibility_assessment'] * 
                               policy_strength)
            
            # Add implementation uncertainty
            uncertainty_discount = 1 - self.params.implementation_uncertainty
            response['behavioral_change'] = behavioral_change * uncertainty_discount
        
        # Compliance probability (for regulatory policies)
        if policy.get('type') == 'regulatory':
            # Compliance depends on social norms and enforcement expectations
            social_compliance = self.params.conformity_pressure
            enforcement_expectation = policy.get('enforcement_probability', 0.5)
            
            response['compliance_probability'] = (social_compliance * 0.7 + 
                                                enforcement_expectation * 0.3)
        
        return response

class BehavioralEconomicsModel:
    """
    Behavioral Economics Model for Bangladesh
    
    This class implements a comprehensive behavioral economics model
    that incorporates psychological factors into macroeconomic analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Behavioral Economics model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model parameters
        self.params = BehavioralParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Model components
        self.agents = []
        self.social_network = None
        self.results = None
        
        # Simulation parameters
        self.n_agents = config.get('n_agents', 1000)
        self.n_periods = config.get('n_periods', 100)
        
        logger.info("Behavioral Economics model initialized for Bangladesh")
    
    def initialize_agents(self, n_agents: int = None) -> List[BehavioralAgent]:
        """
        Initialize population of behavioral agents
        
        Args:
            n_agents: Number of agents to create
            
        Returns:
            List of behavioral agents
        """
        if n_agents is None:
            n_agents = self.n_agents
        
        logger.info(f"Initializing {n_agents} behavioral agents")
        
        agents = []
        
        for i in range(n_agents):
            # Generate agent characteristics based on Bangladesh demographics
            characteristics = self._generate_agent_characteristics(i)
            agent = BehavioralAgent(i, self.params, characteristics)
            agents.append(agent)
        
        self.agents = agents
        
        # Create social network
        self._create_social_network()
        
        return agents
    
    def _generate_agent_characteristics(self, agent_id: int) -> Dict:
        """
        Generate agent characteristics based on Bangladesh demographics
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent characteristics
        """
        # Income distribution (log-normal with Bangladesh parameters)
        income = np.random.lognormal(np.log(50000), 0.8)  # In Taka
        
        # Education levels (based on Bangladesh statistics)
        education_probs = [0.4, 0.35, 0.2, 0.05]  # No formal, primary, secondary, tertiary
        education_level = np.random.choice([0, 0.3, 0.7, 1.0], p=education_probs)
        
        # Age distribution
        age = np.random.normal(32, 12)  # Bangladesh median age
        age = max(18, min(80, age))
        
        # Risk aversion (higher in developing countries)
        risk_aversion = np.random.gamma(3, 0.8) + 1
        
        # Social class
        class_probs = [0.65, 0.25, 0.1]  # Low, middle, high
        social_class = np.random.choice(['low', 'middle', 'high'], p=class_probs)
        
        # Location (urban vs rural)
        location = np.random.choice(['urban', 'rural'], p=[0.35, 0.65])
        
        return {
            'income': income,
            'education': education_level,
            'age': age,
            'risk_aversion': risk_aversion,
            'social_class': social_class,
            'location': location
        }
    
    def _create_social_network(self):
        """
        Create social network among agents
        """
        logger.info("Creating social network")
        
        # Create network graph
        G = nx.watts_strogatz_graph(len(self.agents), 6, 0.3)  # Small-world network
        
        # Assign network connections to agents
        for agent in self.agents:
            neighbors = list(G.neighbors(agent.agent_id))
            agent.social_network = [self.agents[i] for i in neighbors if i < len(self.agents)]
        
        self.social_network = G
    
    def simulate_economy(self, periods: int = None, 
                        economic_shocks: Dict = None) -> pd.DataFrame:
        """
        Simulate the behavioral economy
        
        Args:
            periods: Number of periods to simulate
            economic_shocks: External economic shocks
            
        Returns:
            Simulation results DataFrame
        """
        if periods is None:
            periods = self.n_periods
        
        if not self.agents:
            self.initialize_agents()
        
        logger.info(f"Simulating behavioral economy for {periods} periods")
        
        # Initialize economic conditions
        economic_conditions = {
            'inflation': 0.05,
            'growth': 0.06,
            'unemployment': 0.04,
            'inequality': 0.35
        }
        
        simulation_data = []
        
        for period in range(periods):
            # Update economic conditions
            if economic_shocks:
                for variable, shock_series in economic_shocks.items():
                    if period < len(shock_series):
                        economic_conditions[variable] += shock_series[period]
            
            # Add random fluctuations
            economic_conditions['inflation'] += np.random.normal(0, 0.01)
            economic_conditions['growth'] += np.random.normal(0, 0.02)
            economic_conditions['unemployment'] += np.random.normal(0, 0.005)
            
            # Ensure reasonable bounds
            economic_conditions['inflation'] = max(0, economic_conditions['inflation'])
            economic_conditions['unemployment'] = max(0, min(0.3, economic_conditions['unemployment']))
            
            # Agent decisions
            period_decisions = []
            
            for agent in self.agents:
                # Update agent beliefs
                social_info = self._get_social_information(agent)
                agent.update_beliefs(economic_conditions, social_info)
                
                # Make consumption decision
                current_income = agent.income * (1 + economic_conditions['growth'])
                decision = agent.make_consumption_decision(current_income, economic_conditions)
                
                period_decisions.append({
                    'agent_id': agent.agent_id,
                    'period': period,
                    'income': current_income,
                    'consumption': decision['consumption'],
                    'saving': decision['saving'],
                    'utility': decision['utility'],
                    'social_class': agent.social_class,
                    'education': agent.education,
                    'age': agent.age
                })
            
            # Aggregate outcomes
            period_df = pd.DataFrame(period_decisions)
            
            aggregate_data = {
                'period': period,
                'total_consumption': period_df['consumption'].sum(),
                'total_saving': period_df['saving'].sum(),
                'average_consumption': period_df['consumption'].mean(),
                'consumption_inequality': period_df['consumption'].std() / period_df['consumption'].mean(),
                'saving_rate': period_df['saving'].sum() / period_df['income'].sum(),
                'inflation': economic_conditions['inflation'],
                'growth': economic_conditions['growth'],
                'unemployment': economic_conditions['unemployment']
            }
            
            simulation_data.append(aggregate_data)
        
        return pd.DataFrame(simulation_data)
    
    def _get_social_information(self, agent: BehavioralAgent) -> Dict:
        """
        Get information from agent's social network
        
        Args:
            agent: Target agent
            
        Returns:
            Social information
        """
        if not agent.social_network:
            return {}
        
        # Aggregate beliefs from social network
        network_beliefs = {}
        
        for neighbor in agent.social_network:
            for variable, belief in neighbor.beliefs.items():
                if variable not in network_beliefs:
                    network_beliefs[variable] = []
                network_beliefs[variable].append(belief)
        
        # Average network beliefs
        social_info = {}
        for variable, beliefs in network_beliefs.items():
            if beliefs:
                social_info[variable] = np.mean(beliefs)
        
        return social_info
    
    def analyze_policy_intervention(self, policy: Dict, 
                                  baseline_periods: int = 50,
                                  intervention_periods: int = 50) -> Dict:
        """
        Analyze behavioral responses to policy interventions
        
        Args:
            policy: Policy intervention details
            baseline_periods: Periods before intervention
            intervention_periods: Periods with intervention
            
        Returns:
            Policy analysis results
        """
        logger.info(f"Analyzing policy intervention: {policy.get('name', 'Unknown')}")
        
        # Simulate baseline
        baseline_data = self.simulate_economy(baseline_periods)
        
        # Reset agents for intervention simulation
        self.initialize_agents()
        
        # Simulate with policy intervention
        intervention_data = []
        
        for period in range(intervention_periods):
            # Policy becomes active
            if period == 0:
                # Agents respond to policy announcement
                for agent in self.agents:
                    response = agent.respond_to_policy(policy)
                    # Modify agent behavior based on response
                    self._apply_policy_response(agent, policy, response)
            
            # Continue simulation with modified behavior
            # (This is a simplified implementation)
            period_data = self._simulate_single_period(period, policy)
            intervention_data.append(period_data)
        
        intervention_df = pd.DataFrame(intervention_data)
        
        # Compare baseline and intervention outcomes
        analysis = {
            'baseline_data': baseline_data,
            'intervention_data': intervention_df,
            'policy': policy,
            'effectiveness_metrics': self._compute_policy_effectiveness(baseline_data, intervention_df)
        }
        
        return analysis
    
    def _apply_policy_response(self, agent: BehavioralAgent, policy: Dict, response: Dict):
        """
        Apply policy response to agent behavior
        
        Args:
            agent: Target agent
            policy: Policy details
            response: Agent's response to policy
        """
        # Modify agent parameters based on policy response
        behavioral_change = response['behavioral_change']
        
        if policy.get('type') == 'monetary':
            # Monetary policy affects saving behavior
            if 'interest_rate_change' in policy:
                rate_change = policy['interest_rate_change']
                # Modify saving propensity
                agent.params.long_term_discount *= (1 + rate_change * behavioral_change)
        
        elif policy.get('type') == 'fiscal':
            # Fiscal policy affects consumption
            if 'tax_change' in policy:
                tax_change = policy['tax_change']
                # Modify consumption through income effect
                agent.income *= (1 - tax_change * behavioral_change)
        
        elif policy.get('type') == 'behavioral_nudge':
            # Behavioral interventions
            if 'saving_nudge' in policy:
                # Increase present bias (reduce present bias parameter)
                agent.params.present_bias *= (1 + 0.1 * behavioral_change)
    
    def _simulate_single_period(self, period: int, policy: Dict = None) -> Dict:
        """
        Simulate a single period with optional policy
        
        Args:
            period: Current period
            policy: Active policy (optional)
            
        Returns:
            Period simulation results
        """
        # Simplified single period simulation
        economic_conditions = {
            'inflation': 0.05 + np.random.normal(0, 0.01),
            'growth': 0.06 + np.random.normal(0, 0.02),
            'unemployment': 0.04 + np.random.normal(0, 0.005),
            'inequality': 0.35
        }
        
        total_consumption = 0
        total_saving = 0
        total_income = 0
        
        for agent in self.agents:
            current_income = agent.income * (1 + economic_conditions['growth'])
            decision = agent.make_consumption_decision(current_income, economic_conditions)
            
            total_consumption += decision['consumption']
            total_saving += decision['saving']
            total_income += current_income
        
        return {
            'period': period,
            'total_consumption': total_consumption,
            'total_saving': total_saving,
            'saving_rate': total_saving / total_income if total_income > 0 else 0,
            'average_consumption': total_consumption / len(self.agents)
        }
    
    def _compute_policy_effectiveness(self, baseline: pd.DataFrame, 
                                    intervention: pd.DataFrame) -> Dict:
        """
        Compute policy effectiveness metrics
        
        Args:
            baseline: Baseline simulation data
            intervention: Intervention simulation data
            
        Returns:
            Effectiveness metrics
        """
        metrics = {}
        
        # Compare key variables
        variables = ['average_consumption', 'saving_rate', 'consumption_inequality']
        
        for var in variables:
            if var in baseline.columns and var in intervention.columns:
                baseline_mean = baseline[var].mean()
                intervention_mean = intervention[var].mean()
                
                absolute_change = intervention_mean - baseline_mean
                relative_change = (absolute_change / baseline_mean) * 100 if baseline_mean != 0 else 0
                
                metrics[f'{var}_change'] = {
                    'absolute': absolute_change,
                    'relative': relative_change,
                    'baseline': baseline_mean,
                    'intervention': intervention_mean
                }
        
        return metrics
    
    def analyze_behavioral_patterns(self) -> Dict:
        """
        Analyze behavioral patterns in the agent population
        
        Returns:
            Behavioral pattern analysis
        """
        if not self.agents:
            self.initialize_agents()
        
        logger.info("Analyzing behavioral patterns")
        
        patterns = {}
        
        # Collect agent characteristics
        agent_data = []
        for agent in self.agents:
            agent_data.append({
                'agent_id': agent.agent_id,
                'income': agent.income,
                'education': agent.education,
                'age': agent.age,
                'risk_aversion': agent.risk_aversion,
                'social_class': agent.social_class,
                'loss_aversion': agent.params.loss_aversion,
                'present_bias': agent.params.present_bias
            })
        
        df = pd.DataFrame(agent_data)
        
        # Behavioral clustering
        behavioral_features = ['risk_aversion', 'loss_aversion', 'present_bias', 'education']
        X = df[behavioral_features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        df['behavioral_cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster in range(n_clusters):
            cluster_data = df[df['behavioral_cluster'] == cluster]
            
            cluster_analysis[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'proportion': len(cluster_data) / len(df),
                'characteristics': {
                    'avg_income': cluster_data['income'].mean(),
                    'avg_education': cluster_data['education'].mean(),
                    'avg_risk_aversion': cluster_data['risk_aversion'].mean(),
                    'avg_loss_aversion': cluster_data['loss_aversion'].mean(),
                    'avg_present_bias': cluster_data['present_bias'].mean(),
                    'dominant_social_class': cluster_data['social_class'].mode().iloc[0] if not cluster_data['social_class'].mode().empty else 'unknown'
                }
            }
        
        patterns['cluster_analysis'] = cluster_analysis
        patterns['agent_data'] = df
        
        # Correlation analysis
        correlation_matrix = df[behavioral_features + ['income']].corr()
        patterns['correlations'] = correlation_matrix.to_dict()
        
        return patterns
    
    def plot_behavioral_analysis(self, patterns: Dict, save_path: str = None):
        """
        Plot behavioral analysis results
        
        Args:
            patterns: Behavioral patterns from analyze_behavioral_patterns
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        df = patterns['agent_data']
        
        # 1. Income vs Education scatter plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['education'], df['income'], 
                            c=df['behavioral_cluster'], 
                            alpha=0.6, cmap='viridis')
        ax1.set_xlabel('Education Level')
        ax1.set_ylabel('Income')
        ax1.set_title('Income vs Education by Behavioral Cluster')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Risk aversion distribution
        ax2 = axes[0, 1]
        ax2.hist(df['risk_aversion'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Risk Aversion')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Risk Aversion')
        
        # 3. Behavioral cluster sizes
        ax3 = axes[1, 0]
        cluster_sizes = df['behavioral_cluster'].value_counts().sort_index()
        ax3.bar(cluster_sizes.index, cluster_sizes.values)
        ax3.set_xlabel('Behavioral Cluster')
        ax3.set_ylabel('Number of Agents')
        ax3.set_title('Behavioral Cluster Sizes')
        
        # 4. Correlation heatmap
        ax4 = axes[1, 1]
        corr_matrix = df[['risk_aversion', 'loss_aversion', 'present_bias', 'education', 'income']].corr()
        im = ax4.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_yticks(range(len(corr_matrix.columns)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45)
        ax4.set_yticklabels(corr_matrix.columns)
        ax4.set_title('Behavioral Characteristics Correlation')
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Behavioral analysis plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Behavioral Economics Model',
            'country': 'Bangladesh',
            'n_agents': len(self.agents) if self.agents else self.n_agents,
            'behavioral_factors': [
                'Bounded Rationality',
                'Loss Aversion',
                'Present Bias',
                'Social Preferences',
                'Mental Accounting',
                'Herding Behavior'
            ],
            'parameters': {
                'loss_aversion': self.params.loss_aversion,
                'present_bias': self.params.present_bias,
                'attention_span': self.params.attention_span,
                'herding_tendency': self.params.herding_tendency,
                'family_influence': self.params.family_influence,
                'community_trust': self.params.community_trust
            }
        }
        
        if self.agents:
            # Agent statistics
            agent_stats = {
                'avg_income': np.mean([agent.income for agent in self.agents]),
                'avg_education': np.mean([agent.education for agent in self.agents]),
                'avg_age': np.mean([agent.age for agent in self.agents]),
                'social_class_distribution': {}
            }
            
            # Social class distribution
            class_counts = {}
            for agent in self.agents:
                class_counts[agent.social_class] = class_counts.get(agent.social_class, 0) + 1
            
            total_agents = len(self.agents)
            agent_stats['social_class_distribution'] = {
                class_name: count / total_agents 
                for class_name, count in class_counts.items()
            }
            
            summary['agent_statistics'] = agent_stats
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'loss_aversion': 2.25,
            'present_bias': 0.7,
            'attention_span': 0.7,
            'herding_tendency': 0.4,
            'family_influence': 0.6,
            'community_trust': 0.7
        },
        'n_agents': 500,
        'n_periods': 50
    }
    
    # Initialize Behavioral Economics model
    behavioral_model = BehavioralEconomicsModel(config)
    
    # Initialize agents
    agents = behavioral_model.initialize_agents()
    print(f"Initialized {len(agents)} behavioral agents")
    
    # Analyze behavioral patterns
    patterns = behavioral_model.analyze_behavioral_patterns()
    print(f"\nBehavioral Cluster Analysis:")
    
    for cluster_name, cluster_info in patterns['cluster_analysis'].items():
        print(f"  {cluster_name}: {cluster_info['size']} agents ({cluster_info['proportion']:.1%})")
        print(f"    Avg income: {cluster_info['characteristics']['avg_income']:,.0f}")
        print(f"    Avg education: {cluster_info['characteristics']['avg_education']:.2f}")
        print(f"    Dominant class: {cluster_info['characteristics']['dominant_social_class']}")
    
    # Simulate economy
    print("\nSimulating behavioral economy...")
    simulation_data = behavioral_model.simulate_economy(periods=30)
    print(f"Simulation completed: {len(simulation_data)} periods")
    
    # Display key results
    print(f"\nKey Economic Indicators:")
    print(f"  Average consumption: {simulation_data['average_consumption'].mean():,.0f}")
    print(f"  Average saving rate: {simulation_data['saving_rate'].mean():.1%}")
    print(f"  Consumption inequality (CV): {simulation_data['consumption_inequality'].mean():.3f}")
    
    # Policy analysis
    print("\nAnalyzing behavioral nudge policy...")
    
    nudge_policy = {
        'name': 'Saving Nudge',
        'type': 'behavioral_nudge',
        'saving_nudge': True,
        'strength': 0.3,
        'salience': 0.7
    }
    
    policy_analysis = behavioral_model.analyze_policy_intervention(nudge_policy, 
                                                                  baseline_periods=20, 
                                                                  intervention_periods=20)
    
    effectiveness = policy_analysis['effectiveness_metrics']
    print("Policy Effectiveness:")
    
    for metric, change in effectiveness.items():
        if isinstance(change, dict) and 'relative' in change:
            print(f"  {metric}: {change['relative']:+.1f}% change")
    
    # Model summary
    summary = behavioral_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Number of agents: {summary['n_agents']}")
    print(f"  Behavioral factors: {len(summary['behavioral_factors'])}")
    print(f"  Loss aversion parameter: {summary['parameters']['loss_aversion']}")
    print(f"  Present bias parameter: {summary['parameters']['present_bias']}")