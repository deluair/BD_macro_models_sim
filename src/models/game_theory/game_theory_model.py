#!/usr/bin/env python3
"""
Game Theoretic Models for Bangladesh Macroeconomic Analysis

This module implements various game theoretic models to analyze strategic
interactions in the Bangladesh economy, including:
- International trade negotiations
- Monetary and fiscal policy coordination
- Market competition and industrial organization
- Regional economic cooperation (SAARC, BIMSTEC)
- Climate change cooperation
- Labor market bargaining

Key Features:
- Nash equilibrium computation
- Cooperative and non-cooperative games
- Dynamic games and repeated interactions
- Mechanism design for policy coordination
- Auction theory for resource allocation
- Bargaining models

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, linalg
from scipy.special import comb
from itertools import product, combinations
import networkx as nx
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameParameters:
    """
    Parameters for game theoretic models
    """
    # General game parameters
    num_players: int = 2
    discount_factor: float = 0.95
    cooperation_threshold: float = 0.5
    
    # Economic parameters
    trade_elasticity: float = 1.5
    policy_effectiveness: float = 0.8
    spillover_coefficient: float = 0.3
    
    # Bangladesh-specific parameters
    export_dependence: float = 0.15      # Exports as % of GDP
    import_dependence: float = 0.18      # Imports as % of GDP
    remittance_share: float = 0.06       # Remittances as % of GDP
    aid_dependence: float = 0.02         # Foreign aid as % of GDP
    
    # Regional cooperation parameters
    saarc_trade_share: float = 0.05      # Trade with SAARC countries
    china_trade_share: float = 0.15      # Trade with China
    india_trade_share: float = 0.08      # Trade with India
    eu_trade_share: float = 0.20         # Trade with EU
    
    # Policy coordination parameters
    monetary_independence: float = 0.7    # Degree of monetary independence
    fiscal_coordination: float = 0.3      # Fiscal policy coordination
    
    # Climate cooperation
    climate_vulnerability: float = 0.8    # Climate vulnerability index
    adaptation_cost_share: float = 0.02   # Adaptation costs as % of GDP

@dataclass
class GameResults:
    """
    Results from game theoretic analysis
    """
    parameters: GameParameters
    equilibria: Dict
    payoff_matrices: Optional[Dict] = None
    cooperation_analysis: Optional[Dict] = None
    mechanism_design: Optional[Dict] = None
    dynamic_analysis: Optional[Dict] = None

class Player:
    """
    Represents a player in the game
    """
    
    def __init__(self, name: str, player_type: str, characteristics: Dict):
        self.name = name
        self.player_type = player_type  # 'country', 'firm', 'government', 'central_bank'
        self.characteristics = characteristics
        self.strategies = []
        self.payoffs = {}
        self.beliefs = {}
    
    def add_strategy(self, strategy: str, description: str = ""):
        """Add a strategy to the player's strategy set"""
        self.strategies.append({
            'name': strategy,
            'description': description
        })
    
    def set_payoff(self, strategy_profile: Tuple, payoff: float):
        """Set payoff for a given strategy profile"""
        self.payoffs[strategy_profile] = payoff
    
    def get_payoff(self, strategy_profile: Tuple) -> float:
        """Get payoff for a strategy profile"""
        return self.payoffs.get(strategy_profile, 0.0)

class GameTheoreticModel:
    """
    Main class for game theoretic analysis of Bangladesh economy
    """
    
    def __init__(self, config: Dict):
        """
        Initialize game theoretic model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.params = GameParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Game components
        self.players = {}
        self.games = {}
        self.results = None
        
        logger.info("Game theoretic model initialized for Bangladesh")
    
    def add_player(self, name: str, player_type: str, characteristics: Dict) -> Player:
        """
        Add a player to the game
        
        Args:
            name: Player name
            player_type: Type of player
            characteristics: Player characteristics
            
        Returns:
            Player object
        """
        player = Player(name, player_type, characteristics)
        self.players[name] = player
        return player
    
    def setup_trade_negotiation_game(self) -> Dict:
        """
        Set up international trade negotiation game
        
        Returns:
            Trade game specification
        """
        logger.info("Setting up trade negotiation game")
        
        # Players: Bangladesh, Trading Partners
        bangladesh = self.add_player(
            'Bangladesh', 'country',
            {
                'gdp': 400,  # Billion USD
                'export_capacity': 50,
                'import_demand': 60,
                'bargaining_power': 0.3
            }
        )
        
        china = self.add_player(
            'China', 'country',
            {
                'gdp': 17000,
                'export_capacity': 2500,
                'import_demand': 2000,
                'bargaining_power': 0.9
            }
        )
        
        india = self.add_player(
            'India', 'country',
            {
                'gdp': 3500,
                'export_capacity': 350,
                'import_demand': 400,
                'bargaining_power': 0.7
            }
        )
        
        eu = self.add_player(
            'EU', 'country',
            {
                'gdp': 18000,
                'export_capacity': 2200,
                'import_demand': 2100,
                'bargaining_power': 0.8
            }
        )
        
        # Strategies
        for player in [bangladesh, china, india, eu]:
            player.add_strategy('cooperate', 'Offer favorable trade terms')
            player.add_strategy('defect', 'Impose trade barriers')
            player.add_strategy('conditional', 'Cooperate if others cooperate')
        
        # Payoff matrix for bilateral trade (Bangladesh vs others)
        trade_game = self._compute_trade_payoffs()
        
        self.games['trade_negotiation'] = {
            'players': ['Bangladesh', 'China', 'India', 'EU'],
            'strategies': ['cooperate', 'defect', 'conditional'],
            'payoffs': trade_game,
            'game_type': 'simultaneous',
            'repeated': True
        }
        
        return self.games['trade_negotiation']
    
    def _compute_trade_payoffs(self) -> Dict:
        """
        Compute payoffs for trade negotiation game
        """
        # Simplified bilateral trade payoffs
        # Payoffs represent welfare gains from trade agreements
        
        payoffs = {}
        
        # Bangladesh vs China
        payoffs[('Bangladesh', 'China')] = {
            ('cooperate', 'cooperate'): (8, 12),
            ('cooperate', 'defect'): (-2, 15),
            ('defect', 'cooperate'): (3, -5),
            ('defect', 'defect'): (0, 0)
        }
        
        # Bangladesh vs India
        payoffs[('Bangladesh', 'India')] = {
            ('cooperate', 'cooperate'): (6, 8),
            ('cooperate', 'defect'): (-1, 10),
            ('defect', 'cooperate'): (2, -3),
            ('defect', 'defect'): (0, 0)
        }
        
        # Bangladesh vs EU
        payoffs[('Bangladesh', 'EU')] = {
            ('cooperate', 'cooperate'): (10, 15),
            ('cooperate', 'defect'): (-3, 18),
            ('defect', 'cooperate'): (4, -8),
            ('defect', 'defect'): (0, 0)
        }
        
        return payoffs
    
    def setup_policy_coordination_game(self) -> Dict:
        """
        Set up monetary and fiscal policy coordination game
        
        Returns:
            Policy coordination game specification
        """
        logger.info("Setting up policy coordination game")
        
        # Players: Central Bank, Government
        central_bank = self.add_player(
            'Bangladesh_Bank', 'central_bank',
            {
                'independence': 0.7,
                'inflation_target': 5.5,
                'exchange_rate_concern': 0.8
            }
        )
        
        government = self.add_player(
            'Government', 'government',
            {
                'fiscal_space': 0.6,
                'growth_priority': 0.9,
                'debt_sustainability': 0.7
            }
        )
        
        # Strategies
        central_bank.add_strategy('tight_monetary', 'Contractionary monetary policy')
        central_bank.add_strategy('loose_monetary', 'Expansionary monetary policy')
        central_bank.add_strategy('neutral_monetary', 'Neutral monetary stance')
        
        government.add_strategy('tight_fiscal', 'Contractionary fiscal policy')
        government.add_strategy('loose_fiscal', 'Expansionary fiscal policy')
        government.add_strategy('neutral_fiscal', 'Neutral fiscal stance')
        
        # Compute policy coordination payoffs
        policy_payoffs = self._compute_policy_payoffs()
        
        self.games['policy_coordination'] = {
            'players': ['Bangladesh_Bank', 'Government'],
            'strategies': {
                'Bangladesh_Bank': ['tight_monetary', 'loose_monetary', 'neutral_monetary'],
                'Government': ['tight_fiscal', 'loose_fiscal', 'neutral_fiscal']
            },
            'payoffs': policy_payoffs,
            'game_type': 'simultaneous',
            'repeated': True
        }
        
        return self.games['policy_coordination']
    
    def _compute_policy_payoffs(self) -> Dict:
        """
        Compute payoffs for policy coordination game
        """
        # Payoffs represent macroeconomic outcomes (growth, inflation, stability)
        # Format: (Central Bank utility, Government utility)
        
        payoffs = {
            ('tight_monetary', 'tight_fiscal'): (-2, -3),    # Recession risk
            ('tight_monetary', 'loose_fiscal'): (1, 2),      # Balanced approach
            ('tight_monetary', 'neutral_fiscal'): (0, -1),   # Moderate contraction
            ('loose_monetary', 'tight_fiscal'): (2, 1),      # Balanced approach
            ('loose_monetary', 'loose_fiscal'): (-1, 3),     # Overheating risk
            ('loose_monetary', 'neutral_fiscal'): (1, 2),    # Moderate expansion
            ('neutral_monetary', 'tight_fiscal'): (-1, 0),   # Moderate contraction
            ('neutral_monetary', 'loose_fiscal'): (2, 1),    # Moderate expansion
            ('neutral_monetary', 'neutral_fiscal'): (0, 0)   # Status quo
        }
        
        return payoffs
    
    def setup_climate_cooperation_game(self) -> Dict:
        """
        Set up climate change cooperation game
        
        Returns:
            Climate cooperation game specification
        """
        logger.info("Setting up climate cooperation game")
        
        # Players: Bangladesh, India, China, Developed Countries
        players_climate = [
            ('Bangladesh', 'developing', {'vulnerability': 0.9, 'emissions': 0.3, 'capacity': 0.2}),
            ('India', 'developing', {'vulnerability': 0.6, 'emissions': 7.0, 'capacity': 0.5}),
            ('China', 'developing', {'vulnerability': 0.4, 'emissions': 28.0, 'capacity': 0.8}),
            ('USA', 'developed', {'vulnerability': 0.3, 'emissions': 15.0, 'capacity': 0.9}),
            ('EU', 'developed', {'vulnerability': 0.2, 'emissions': 9.0, 'capacity': 0.9})
        ]
        
        for name, category, chars in players_climate:
            if name not in self.players:
                self.add_player(name, 'country', chars)
        
        # Strategies
        strategies = ['high_mitigation', 'low_mitigation', 'adaptation_focus', 'free_ride']
        
        for player_name, _, _ in players_climate:
            player = self.players[player_name]
            for strategy in strategies:
                player.add_strategy(strategy, f"{strategy.replace('_', ' ').title()} strategy")
        
        # Compute climate payoffs
        climate_payoffs = self._compute_climate_payoffs(players_climate)
        
        self.games['climate_cooperation'] = {
            'players': [name for name, _, _ in players_climate],
            'strategies': strategies,
            'payoffs': climate_payoffs,
            'game_type': 'simultaneous',
            'repeated': True,
            'public_good': True
        }
        
        return self.games['climate_cooperation']
    
    def _compute_climate_payoffs(self, players_info: List) -> Dict:
        """
        Compute payoffs for climate cooperation game
        """
        payoffs = {}
        
        # Simplified climate payoff calculation
        # Benefits from global mitigation minus costs of own actions
        
        base_damage = {
            'Bangladesh': -20,  # High vulnerability
            'India': -15,
            'China': -10,
            'USA': -8,
            'EU': -6
        }
        
        mitigation_cost = {
            'high_mitigation': -8,
            'low_mitigation': -3,
            'adaptation_focus': -5,
            'free_ride': 0
        }
        
        global_benefit = {
            'high_mitigation': 4,
            'low_mitigation': 2,
            'adaptation_focus': 1,
            'free_ride': 0
        }
        
        # Compute payoffs for each player and strategy combination
        for name, category, chars in players_info:
            player_payoffs = {}
            
            for own_strategy in ['high_mitigation', 'low_mitigation', 'adaptation_focus', 'free_ride']:
                # Own cost
                own_cost = mitigation_cost[own_strategy]
                
                # Benefit depends on global cooperation level
                # Simplified: assume others play mixed strategies
                global_cooperation = 0.3  # Average cooperation level
                benefit = global_benefit[own_strategy] * global_cooperation
                
                # Climate damage (reduced by global cooperation)
                damage = base_damage[name] * (1 - global_cooperation * 0.5)
                
                # Total payoff
                total_payoff = own_cost + benefit + damage
                
                player_payoffs[own_strategy] = total_payoff
            
            payoffs[name] = player_payoffs
        
        return payoffs
    
    def setup_regional_cooperation_game(self) -> Dict:
        """
        Set up SAARC/BIMSTEC regional cooperation game
        
        Returns:
            Regional cooperation game specification
        """
        logger.info("Setting up regional cooperation game")
        
        # SAARC countries
        saarc_countries = [
            ('Bangladesh', {'gdp': 400, 'trade_openness': 0.3, 'political_stability': 0.6}),
            ('India', {'gdp': 3500, 'trade_openness': 0.4, 'political_stability': 0.7}),
            ('Pakistan', {'gdp': 350, 'trade_openness': 0.25, 'political_stability': 0.4}),
            ('Sri_Lanka', {'gdp': 85, 'trade_openness': 0.5, 'political_stability': 0.5}),
            ('Nepal', {'gdp': 35, 'trade_openness': 0.6, 'political_stability': 0.6})
        ]
        
        for name, chars in saarc_countries:
            if name not in self.players:
                self.add_player(name, 'country', chars)
        
        # Cooperation strategies
        coop_strategies = [
            'full_cooperation',    # Reduce all trade barriers
            'selective_cooperation', # Cooperate in some sectors
            'minimal_cooperation',  # Status quo
            'non_cooperation'      # Increase barriers
        ]
        
        for name, _ in saarc_countries:
            player = self.players[name]
            for strategy in coop_strategies:
                player.add_strategy(strategy, f"{strategy.replace('_', ' ').title()}")
        
        # Compute regional cooperation payoffs
        regional_payoffs = self._compute_regional_payoffs(saarc_countries)
        
        self.games['regional_cooperation'] = {
            'players': [name for name, _ in saarc_countries],
            'strategies': coop_strategies,
            'payoffs': regional_payoffs,
            'game_type': 'simultaneous',
            'repeated': True,
            'network_effects': True
        }
        
        return self.games['regional_cooperation']
    
    def _compute_regional_payoffs(self, countries_info: List) -> Dict:
        """
        Compute payoffs for regional cooperation game
        """
        payoffs = {}
        
        cooperation_benefits = {
            'full_cooperation': 8,
            'selective_cooperation': 4,
            'minimal_cooperation': 1,
            'non_cooperation': 0
        }
        
        cooperation_costs = {
            'full_cooperation': -3,
            'selective_cooperation': -1,
            'minimal_cooperation': 0,
            'non_cooperation': 0
        }
        
        # Network effects: benefits increase with number of cooperating countries
        for name, chars in countries_info:
            player_payoffs = {}
            
            for strategy in cooperation_benefits.keys():
                # Base benefit and cost
                benefit = cooperation_benefits[strategy]
                cost = cooperation_costs[strategy]
                
                # Adjust for country size (GDP)
                size_factor = chars['gdp'] / 1000  # Normalize
                adjusted_benefit = benefit * (1 + size_factor * 0.1)
                
                # Political stability affects cooperation costs
                stability_factor = chars['political_stability']
                adjusted_cost = cost * (2 - stability_factor)
                
                total_payoff = adjusted_benefit + adjusted_cost
                player_payoffs[strategy] = total_payoff
            
            payoffs[name] = player_payoffs
        
        return payoffs
    
    def find_nash_equilibrium(self, game_name: str) -> Dict:
        """
        Find Nash equilibrium for a given game
        
        Args:
            game_name: Name of the game
            
        Returns:
            Nash equilibrium analysis
        """
        if game_name not in self.games:
            raise ValueError(f"Game '{game_name}' not found")
        
        game = self.games[game_name]
        logger.info(f"Finding Nash equilibrium for {game_name}")
        
        if game['game_type'] == 'simultaneous':
            if len(game['players']) == 2:
                return self._find_nash_2player(game)
            else:
                return self._find_nash_multiplayer(game)
        else:
            return self._find_sequential_equilibrium(game)
    
    def _find_nash_2player(self, game: Dict) -> Dict:
        """
        Find Nash equilibrium for 2-player game
        """
        players = game['players']
        payoffs = game['payoffs']
        
        if isinstance(game['strategies'], list):
            strategies = {player: game['strategies'] for player in players}
        else:
            strategies = game['strategies']
        
        equilibria = []
        
        # Check all strategy combinations
        for s1 in strategies[players[0]]:
            for s2 in strategies[players[1]]:
                strategy_profile = (s1, s2)
                
                # Check if this is a Nash equilibrium
                is_nash = True
                
                # Check player 1's best response
                current_payoff_1 = self._get_payoff(players[0], strategy_profile, payoffs)
                for alt_s1 in strategies[players[0]]:
                    alt_profile = (alt_s1, s2)
                    alt_payoff_1 = self._get_payoff(players[0], alt_profile, payoffs)
                    if alt_payoff_1 > current_payoff_1:
                        is_nash = False
                        break
                
                if is_nash:
                    # Check player 2's best response
                    current_payoff_2 = self._get_payoff(players[1], strategy_profile, payoffs)
                    for alt_s2 in strategies[players[1]]:
                        alt_profile = (s1, alt_s2)
                        alt_payoff_2 = self._get_payoff(players[1], alt_profile, payoffs)
                        if alt_payoff_2 > current_payoff_2:
                            is_nash = False
                            break
                
                if is_nash:
                    payoff_1 = self._get_payoff(players[0], strategy_profile, payoffs)
                    payoff_2 = self._get_payoff(players[1], strategy_profile, payoffs)
                    
                    equilibria.append({
                        'strategy_profile': strategy_profile,
                        'payoffs': {players[0]: payoff_1, players[1]: payoff_2},
                        'type': 'pure_strategy'
                    })
        
        return {
            'equilibria': equilibria,
            'num_equilibria': len(equilibria),
            'game_type': '2-player simultaneous'
        }
    
    def _find_nash_multiplayer(self, game: Dict) -> Dict:
        """
        Find Nash equilibrium for multiplayer game
        """
        players = game['players']
        strategies = game['strategies']
        payoffs = game['payoffs']
        
        equilibria = []
        
        # Generate all strategy profiles
        if isinstance(strategies, list):
            strategy_combinations = list(product(strategies, repeat=len(players)))
        else:
            strategy_lists = [strategies[player] for player in players]
            strategy_combinations = list(product(*strategy_lists))
        
        for strategy_profile in strategy_combinations:
            is_nash = True
            
            # Check each player's best response
            for i, player in enumerate(players):
                current_payoff = self._get_payoff(player, strategy_profile, payoffs)
                
                # Check all alternative strategies for this player
                player_strategies = strategies if isinstance(strategies, list) else strategies[player]
                
                for alt_strategy in player_strategies:
                    # Create alternative strategy profile
                    alt_profile = list(strategy_profile)
                    alt_profile[i] = alt_strategy
                    alt_profile = tuple(alt_profile)
                    
                    alt_payoff = self._get_payoff(player, alt_profile, payoffs)
                    
                    if alt_payoff > current_payoff:
                        is_nash = False
                        break
                
                if not is_nash:
                    break
            
            if is_nash:
                player_payoffs = {}
                for player in players:
                    player_payoffs[player] = self._get_payoff(player, strategy_profile, payoffs)
                
                equilibria.append({
                    'strategy_profile': strategy_profile,
                    'payoffs': player_payoffs,
                    'type': 'pure_strategy'
                })
        
        return {
            'equilibria': equilibria,
            'num_equilibria': len(equilibria),
            'game_type': 'multiplayer simultaneous'
        }
    
    def _get_payoff(self, player: str, strategy_profile: Tuple, payoffs: Dict) -> float:
        """
        Get payoff for a player given strategy profile
        """
        # Handle different payoff structures
        if player in payoffs:
            # Player-specific payoffs
            if isinstance(payoffs[player], dict):
                if len(strategy_profile) == 1:
                    return payoffs[player].get(strategy_profile[0], 0)
                else:
                    return payoffs[player].get(strategy_profile, 0)
            else:
                return payoffs[player]
        
        # Bilateral payoffs
        for key, value in payoffs.items():
            if isinstance(key, tuple) and player in key:
                if strategy_profile in value:
                    player_index = key.index(player)
                    return value[strategy_profile][player_index]
        
        return 0.0
    
    def _find_sequential_equilibrium(self, game: Dict) -> Dict:
        """
        Find equilibrium for sequential games (backward induction)
        """
        # Simplified sequential game analysis
        # This would need to be expanded for specific sequential games
        
        return {
            'equilibria': [],
            'num_equilibria': 0,
            'game_type': 'sequential',
            'note': 'Sequential equilibrium analysis not fully implemented'
        }
    
    def analyze_cooperation(self, game_name: str) -> Dict:
        """
        Analyze cooperation possibilities in repeated games
        
        Args:
            game_name: Name of the game
            
        Returns:
            Cooperation analysis
        """
        if game_name not in self.games:
            raise ValueError(f"Game '{game_name}' not found")
        
        game = self.games[game_name]
        logger.info(f"Analyzing cooperation in {game_name}")
        
        # Find Nash equilibrium
        nash_result = self.find_nash_equilibrium(game_name)
        
        # Analyze repeated game cooperation
        cooperation_analysis = self._analyze_repeated_game_cooperation(game, nash_result)
        
        return cooperation_analysis
    
    def _analyze_repeated_game_cooperation(self, game: Dict, nash_result: Dict) -> Dict:
        """
        Analyze cooperation in repeated games using folk theorem
        """
        players = game['players']
        discount_factor = self.params.discount_factor
        
        # Find cooperative and non-cooperative payoffs
        cooperative_payoffs = self._find_cooperative_solution(game)
        nash_payoffs = nash_result['equilibria'][0]['payoffs'] if nash_result['equilibria'] else {}
        
        # Check if cooperation is sustainable
        cooperation_sustainable = {}
        
        for player in players:
            if player in cooperative_payoffs and player in nash_payoffs:
                coop_payoff = cooperative_payoffs[player]
                nash_payoff = nash_payoffs[player]
                
                # Simplified sustainability condition
                # Cooperation sustainable if discounted future gains > short-term deviation gains
                sustainability_condition = (
                    discount_factor / (1 - discount_factor) * (coop_payoff - nash_payoff) > 
                    abs(coop_payoff - nash_payoff)
                )
                
                cooperation_sustainable[player] = sustainability_condition
        
        return {
            'cooperative_payoffs': cooperative_payoffs,
            'nash_payoffs': nash_payoffs,
            'cooperation_sustainable': cooperation_sustainable,
            'discount_factor': discount_factor,
            'overall_sustainability': all(cooperation_sustainable.values())
        }
    
    def _find_cooperative_solution(self, game: Dict) -> Dict:
        """
        Find cooperative (Pareto efficient) solution
        """
        players = game['players']
        payoffs = game['payoffs']
        strategies = game['strategies']
        
        # Find strategy profile that maximizes sum of payoffs
        best_total = float('-inf')
        best_profile = None
        best_payoffs = {}
        
        if isinstance(strategies, list):
            strategy_combinations = list(product(strategies, repeat=len(players)))
        else:
            strategy_lists = [strategies[player] for player in players]
            strategy_combinations = list(product(*strategy_lists))
        
        for strategy_profile in strategy_combinations:
            total_payoff = 0
            current_payoffs = {}
            
            for player in players:
                player_payoff = self._get_payoff(player, strategy_profile, payoffs)
                current_payoffs[player] = player_payoff
                total_payoff += player_payoff
            
            if total_payoff > best_total:
                best_total = total_payoff
                best_profile = strategy_profile
                best_payoffs = current_payoffs
        
        return best_payoffs
    
    def mechanism_design_analysis(self, game_name: str) -> Dict:
        """
        Analyze mechanism design for achieving cooperation
        
        Args:
            game_name: Name of the game
            
        Returns:
            Mechanism design analysis
        """
        logger.info(f"Analyzing mechanism design for {game_name}")
        
        if game_name not in self.games:
            raise ValueError(f"Game '{game_name}' not found")
        
        game = self.games[game_name]
        
        # Analyze different mechanisms
        mechanisms = {
            'side_payments': self._analyze_side_payments(game),
            'conditional_cooperation': self._analyze_conditional_cooperation(game),
            'punishment_mechanisms': self._analyze_punishment_mechanisms(game)
        }
        
        return mechanisms
    
    def _analyze_side_payments(self, game: Dict) -> Dict:
        """
        Analyze side payment mechanisms
        """
        # Find cooperative solution
        cooperative_payoffs = self._find_cooperative_solution(game)
        
        # Find Nash equilibrium
        nash_result = self.find_nash_equilibrium(game['players'][0])  # Simplified
        
        # Calculate required transfers
        transfers = {}
        if nash_result['equilibria']:
            nash_payoffs = nash_result['equilibria'][0]['payoffs']
            
            for player in game['players']:
                if player in cooperative_payoffs and player in nash_payoffs:
                    # Transfer needed to make cooperation individually rational
                    transfer = max(0, nash_payoffs[player] - cooperative_payoffs[player])
                    transfers[player] = transfer
        
        return {
            'feasible': sum(transfers.values()) <= sum(cooperative_payoffs.values()),
            'required_transfers': transfers,
            'total_transfer': sum(transfers.values())
        }
    
    def _analyze_conditional_cooperation(self, game: Dict) -> Dict:
        """
        Analyze conditional cooperation strategies
        """
        # Simplified analysis of tit-for-tat and other conditional strategies
        return {
            'tit_for_tat_viable': True,  # Simplified
            'trigger_strategies': ['grim_trigger', 'tit_for_tat'],
            'cooperation_threshold': self.params.cooperation_threshold
        }
    
    def _analyze_punishment_mechanisms(self, game: Dict) -> Dict:
        """
        Analyze punishment mechanisms for sustaining cooperation
        """
        return {
            'credible_punishments': True,  # Simplified
            'punishment_severity': 'moderate',
            'enforcement_mechanisms': ['reputation', 'institutional']
        }
    
    def simulate_dynamic_game(self, game_name: str, periods: int = 20, 
                            strategies: Dict = None) -> pd.DataFrame:
        """
        Simulate dynamic game over multiple periods
        
        Args:
            game_name: Name of the game
            periods: Number of periods to simulate
            strategies: Player strategies (if None, use Nash equilibrium)
            
        Returns:
            Simulation results DataFrame
        """
        logger.info(f"Simulating {periods}-period dynamic game: {game_name}")
        
        if game_name not in self.games:
            raise ValueError(f"Game '{game_name}' not found")
        
        game = self.games[game_name]
        players = game['players']
        
        # Initialize simulation
        simulation_data = []
        
        # Player strategies (simplified)
        if strategies is None:
            # Use mixed strategies or simple rules
            strategies = {player: 'cooperate' for player in players}
        
        for period in range(periods):
            # Determine actions for this period
            period_actions = {}
            period_payoffs = {}
            
            for player in players:
                # Simple strategy: cooperate with some probability
                if np.random.random() < 0.7:  # 70% cooperation probability
                    action = 'cooperate'
                else:
                    action = 'defect'
                
                period_actions[player] = action
            
            # Calculate payoffs
            strategy_profile = tuple(period_actions[player] for player in players)
            
            for player in players:
                payoff = self._get_payoff(player, strategy_profile, game['payoffs'])
                period_payoffs[player] = payoff
            
            # Store period data
            period_data = {
                'period': period,
                **{f'{player}_action': period_actions[player] for player in players},
                **{f'{player}_payoff': period_payoffs[player] for player in players}
            }
            
            simulation_data.append(period_data)
        
        return pd.DataFrame(simulation_data)
    
    def plot_game_analysis(self, game_name: str, analysis_type: str = 'payoff_matrix', 
                          save_path: str = None):
        """
        Plot game analysis results
        
        Args:
            game_name: Name of the game
            analysis_type: Type of analysis to plot
            save_path: Path to save plot
        """
        if game_name not in self.games:
            logger.warning(f"Game '{game_name}' not found")
            return
        
        game = self.games[game_name]
        
        if analysis_type == 'payoff_matrix' and len(game['players']) == 2:
            self._plot_payoff_matrix(game, save_path)
        elif analysis_type == 'cooperation_dynamics':
            self._plot_cooperation_dynamics(game_name, save_path)
        elif analysis_type == 'equilibrium_analysis':
            self._plot_equilibrium_analysis(game_name, save_path)
        else:
            logger.warning(f"Plot type '{analysis_type}' not supported for this game")
    
    def _plot_payoff_matrix(self, game: Dict, save_path: str = None):
        """
        Plot payoff matrix for 2-player game
        """
        players = game['players']
        strategies = game['strategies']
        payoffs = game['payoffs']
        
        if isinstance(strategies, list):
            p1_strategies = p2_strategies = strategies
        else:
            p1_strategies = strategies[players[0]]
            p2_strategies = strategies[players[1]]
        
        # Create payoff matrices
        p1_matrix = np.zeros((len(p1_strategies), len(p2_strategies)))
        p2_matrix = np.zeros((len(p1_strategies), len(p2_strategies)))
        
        for i, s1 in enumerate(p1_strategies):
            for j, s2 in enumerate(p2_strategies):
                strategy_profile = (s1, s2)
                p1_payoff = self._get_payoff(players[0], strategy_profile, payoffs)
                p2_payoff = self._get_payoff(players[1], strategy_profile, payoffs)
                
                p1_matrix[i, j] = p1_payoff
                p2_matrix[i, j] = p2_payoff
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Player 1 payoffs
        sns.heatmap(p1_matrix, annot=True, fmt='.1f', 
                   xticklabels=p2_strategies, yticklabels=p1_strategies,
                   ax=ax1, cmap='RdYlBu_r')
        ax1.set_title(f'{players[0]} Payoffs')
        ax1.set_xlabel(f'{players[1]} Strategies')
        ax1.set_ylabel(f'{players[0]} Strategies')
        
        # Player 2 payoffs
        sns.heatmap(p2_matrix, annot=True, fmt='.1f',
                   xticklabels=p2_strategies, yticklabels=p1_strategies,
                   ax=ax2, cmap='RdYlBu_r')
        ax2.set_title(f'{players[1]} Payoffs')
        ax2.set_xlabel(f'{players[1]} Strategies')
        ax2.set_ylabel(f'{players[0]} Strategies')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Payoff matrix plot saved to {save_path}")
        
        plt.show()
    
    def _plot_cooperation_dynamics(self, game_name: str, save_path: str = None):
        """
        Plot cooperation dynamics over time
        """
        # Simulate dynamic game
        simulation_df = self.simulate_dynamic_game(game_name, periods=50)
        
        # Calculate cooperation rates
        players = self.games[game_name]['players']
        cooperation_rates = {}
        
        for player in players:
            action_col = f'{player}_action'
            if action_col in simulation_df.columns:
                cooperation_rates[player] = (
                    simulation_df[action_col] == 'cooperate'
                ).rolling(window=5).mean()
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        for player, rates in cooperation_rates.items():
            plt.plot(simulation_df['period'], rates, label=player, linewidth=2)
        
        plt.xlabel('Period')
        plt.ylabel('Cooperation Rate (5-period moving average)')
        plt.title(f'Cooperation Dynamics: {game_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cooperation dynamics plot saved to {save_path}")
        
        plt.show()
    
    def _plot_equilibrium_analysis(self, game_name: str, save_path: str = None):
        """
        Plot equilibrium analysis
        """
        # Find equilibria
        nash_result = self.find_nash_equilibrium(game_name)
        cooperation_analysis = self.analyze_cooperation(game_name)
        
        if not nash_result['equilibria']:
            logger.warning("No equilibria found for plotting")
            return
        
        # Plot equilibrium payoffs vs cooperative payoffs
        players = self.games[game_name]['players']
        
        nash_payoffs = [nash_result['equilibria'][0]['payoffs'].get(player, 0) for player in players]
        coop_payoffs = [cooperation_analysis['cooperative_payoffs'].get(player, 0) for player in players]
        
        x = np.arange(len(players))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, nash_payoffs, width, label='Nash Equilibrium', alpha=0.8)
        bars2 = ax.bar(x + width/2, coop_payoffs, width, label='Cooperative Solution', alpha=0.8)
        
        ax.set_xlabel('Players')
        ax.set_ylabel('Payoffs')
        ax.set_title(f'Equilibrium vs Cooperative Payoffs: {game_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(players)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equilibrium analysis plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Game Theoretic Models',
            'country': 'Bangladesh',
            'games_implemented': list(self.games.keys()),
            'total_players': len(self.players),
            'player_types': list(set(player.player_type for player in self.players.values()))
        }
        
        # Add game-specific summaries
        for game_name, game in self.games.items():
            summary[f'{game_name}_summary'] = {
                'players': game['players'],
                'game_type': game['game_type'],
                'repeated': game.get('repeated', False)
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'discount_factor': 0.95,
            'cooperation_threshold': 0.6,
            'trade_elasticity': 1.5
        }
    }
    
    # Initialize game theoretic model
    game_model = GameTheoreticModel(config)
    
    # Set up games
    print("Setting up games...")
    
    # Trade negotiation game
    trade_game = game_model.setup_trade_negotiation_game()
    print(f"Trade game setup: {len(trade_game['players'])} players")
    
    # Policy coordination game
    policy_game = game_model.setup_policy_coordination_game()
    print(f"Policy game setup: {len(policy_game['players'])} players")
    
    # Climate cooperation game
    climate_game = game_model.setup_climate_cooperation_game()
    print(f"Climate game setup: {len(climate_game['players'])} players")
    
    # Regional cooperation game
    regional_game = game_model.setup_regional_cooperation_game()
    print(f"Regional game setup: {len(regional_game['players'])} players")
    
    # Analyze equilibria
    print("\nAnalyzing equilibria...")
    
    for game_name in ['policy_coordination']:
        try:
            nash_result = game_model.find_nash_equilibrium(game_name)
            print(f"\n{game_name}:")
            print(f"  Number of Nash equilibria: {nash_result['num_equilibria']}")
            
            if nash_result['equilibria']:
                eq = nash_result['equilibria'][0]
                print(f"  First equilibrium: {eq['strategy_profile']}")
                print(f"  Payoffs: {eq['payoffs']}")
            
            # Cooperation analysis
            coop_analysis = game_model.analyze_cooperation(game_name)
            print(f"  Cooperation sustainable: {coop_analysis.get('overall_sustainability', 'N/A')}")
            
        except Exception as e:
            print(f"  Error analyzing {game_name}: {str(e)}")
    
    # Simulate dynamic games
    print("\nSimulating dynamic games...")
    
    try:
        simulation_df = game_model.simulate_dynamic_game('policy_coordination', periods=20)
        print(f"Simulation completed: {len(simulation_df)} periods")
        
        # Calculate average cooperation rate
        action_cols = [col for col in simulation_df.columns if col.endswith('_action')]
        if action_cols:
            cooperation_rate = (simulation_df[action_cols[0]] == 'cooperate').mean()
            print(f"Average cooperation rate: {cooperation_rate:.2f}")
    
    except Exception as e:
        print(f"Simulation error: {str(e)}")
    
    # Model summary
    summary = game_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Games implemented: {summary['games_implemented']}")
    print(f"  Total players: {summary['total_players']}")
    print(f"  Player types: {summary['player_types']}")