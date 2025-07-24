#!/usr/bin/env python3
"""
Small Open Economy Model for Bangladesh

This module implements a comprehensive Small Open Economy (SOE) model that analyzes
external sector dynamics, exchange rate determination, international trade,
capital flows, and external vulnerabilities for Bangladesh.

Key Features:
- Exchange rate determination (floating, managed, fixed regimes)
- Current account and capital account dynamics
- International trade modeling (exports, imports, terms of trade)
- Capital flows (FDI, portfolio investment, remittances)
- External debt sustainability analysis
- Balance of payments modeling
- Real exchange rate and competitiveness
- External shocks and vulnerability assessment
- Monetary policy in open economy context
- Sudden stop and capital flight scenarios

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, linalg
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SOEParameters:
    """
    Parameters for the Small Open Economy Model
    """
    # Exchange rate regime
    exchange_rate_regime: str = 'managed_float'  # 'fixed', 'float', 'managed_float'
    
    # Trade parameters
    export_elasticity: float = 1.2              # Export price elasticity
    import_elasticity: float = 1.5              # Import price elasticity
    export_income_elasticity: float = 1.8       # Export income elasticity (world)
    import_income_elasticity: float = 1.4       # Import income elasticity (domestic)
    
    # Bangladesh trade structure
    textile_export_share: float = 0.84          # RMG exports share
    agricultural_export_share: float = 0.03     # Agricultural exports
    other_export_share: float = 0.13            # Other exports
    
    fuel_import_share: float = 0.15             # Fuel imports share
    capital_goods_import_share: float = 0.25    # Capital goods imports
    consumer_goods_import_share: float = 0.35   # Consumer goods imports
    raw_materials_import_share: float = 0.25    # Raw materials imports
    
    # Capital flows
    fdi_gdp_ratio: float = 0.008                # FDI to GDP ratio
    portfolio_volatility: float = 0.3           # Portfolio flow volatility
    remittance_gdp_ratio: float = 0.055         # Remittances to GDP
    
    # External debt
    external_debt_gdp: float = 0.20             # External debt to GDP
    debt_service_ratio: float = 0.05            # Debt service to exports
    
    # Monetary policy
    interest_rate_differential: float = 0.02    # Domestic-foreign rate differential
    sterilization_coefficient: float = 0.7     # Sterilization of capital flows
    
    # Real exchange rate
    reer_trend: float = 0.0                     # REER trend appreciation
    productivity_differential: float = 0.02     # Productivity growth differential
    
    # External vulnerabilities
    import_coverage_months: float = 3.5         # Import coverage (months)
    short_term_debt_reserves: float = 0.8       # ST debt to reserves ratio
    
    # Shock parameters
    terms_of_trade_volatility: float = 0.1      # Terms of trade volatility
    world_growth_volatility: float = 0.02       # World growth volatility
    capital_flow_volatility: float = 0.4        # Capital flow volatility
    
    # Policy parameters
    fiscal_response: float = 0.3                # Fiscal response to external shocks
    monetary_response: float = 1.5              # Monetary response (Taylor rule)
    
    # Simulation parameters
    time_periods: int = 100                     # Simulation periods
    monte_carlo_runs: int = 1000                # Monte Carlo simulations
    
    # Initial conditions
    initial_reer: float = 100.0                 # Initial REER index
    initial_ca_gdp: float = -0.015              # Initial current account/GDP
    initial_reserves_months: float = 3.5        # Initial reserves (months of imports)

@dataclass
class SOEResults:
    """
    Results from Small Open Economy Model
    """
    parameters: SOEParameters
    simulation_data: Optional[pd.DataFrame] = None
    bop_analysis: Optional[Dict] = None
    vulnerability_assessment: Optional[Dict] = None
    policy_analysis: Optional[Dict] = None
    shock_analysis: Optional[Dict] = None

class ExchangeRateModel:
    """
    Exchange rate determination model
    """
    
    def __init__(self, params: SOEParameters):
        """
        Initialize exchange rate model
        
        Args:
            params: Model parameters
        """
        self.params = params
        self.regime = params.exchange_rate_regime
    
    def determine_exchange_rate(self, fundamentals: Dict, shocks: Dict) -> Dict:
        """
        Determine exchange rate based on regime and fundamentals
        
        Args:
            fundamentals: Economic fundamentals
            shocks: External shocks
            
        Returns:
            Exchange rate outcomes
        """
        if self.regime == 'fixed':
            return self._fixed_rate(fundamentals, shocks)
        elif self.regime == 'float':
            return self._floating_rate(fundamentals, shocks)
        else:  # managed_float
            return self._managed_float(fundamentals, shocks)
    
    def _fixed_rate(self, fundamentals: Dict, shocks: Dict) -> Dict:
        """
        Fixed exchange rate regime
        
        Args:
            fundamentals: Economic fundamentals
            shocks: External shocks
            
        Returns:
            Exchange rate outcomes
        """
        # Nominal rate fixed, real rate adjusts through prices
        nominal_rate_change = 0.0
        
        # Pressure on reserves
        ca_balance = fundamentals.get('current_account', 0.0)
        capital_flows = fundamentals.get('capital_flows', 0.0)
        
        reserve_change = ca_balance + capital_flows
        
        # Real exchange rate through inflation differential
        inflation_diff = fundamentals.get('inflation_differential', 0.0)
        real_rate_change = -inflation_diff  # Appreciation with higher inflation
        
        return {
            'nominal_rate_change': nominal_rate_change,
            'real_rate_change': real_rate_change,
            'reserve_change': reserve_change,
            'intervention': -reserve_change  # Central bank intervention
        }
    
    def _floating_rate(self, fundamentals: Dict, shocks: Dict) -> Dict:
        """
        Floating exchange rate regime
        
        Args:
            fundamentals: Economic fundamentals
            shocks: External shocks
            
        Returns:
            Exchange rate outcomes
        """
        # Exchange rate adjusts to clear BOP
        ca_balance = fundamentals.get('current_account', 0.0)
        capital_flows = fundamentals.get('capital_flows', 0.0)
        
        # Exchange rate adjustment
        bop_imbalance = ca_balance + capital_flows
        nominal_rate_change = -bop_imbalance * 2.0  # Depreciation with deficit
        
        # Add market shocks
        market_shock = shocks.get('exchange_rate_shock', 0.0)
        nominal_rate_change += market_shock
        
        # Real exchange rate
        inflation_diff = fundamentals.get('inflation_differential', 0.0)
        real_rate_change = nominal_rate_change - inflation_diff
        
        return {
            'nominal_rate_change': nominal_rate_change,
            'real_rate_change': real_rate_change,
            'reserve_change': 0.0,  # No intervention
            'intervention': 0.0
        }
    
    def _managed_float(self, fundamentals: Dict, shocks: Dict) -> Dict:
        """
        Managed floating exchange rate regime
        
        Args:
            fundamentals: Economic fundamentals
            shocks: External shocks
            
        Returns:
            Exchange rate outcomes
        """
        # Partial adjustment with intervention
        ca_balance = fundamentals.get('current_account', 0.0)
        capital_flows = fundamentals.get('capital_flows', 0.0)
        
        bop_imbalance = ca_balance + capital_flows
        
        # Central bank allows partial adjustment
        intervention_share = 0.6  # 60% intervention, 40% adjustment
        
        nominal_rate_change = -bop_imbalance * (1 - intervention_share) * 2.0
        intervention = -bop_imbalance * intervention_share
        
        # Add market shocks (dampened by intervention)
        market_shock = shocks.get('exchange_rate_shock', 0.0)
        nominal_rate_change += market_shock * (1 - intervention_share)
        intervention -= market_shock * intervention_share
        
        # Real exchange rate
        inflation_diff = fundamentals.get('inflation_differential', 0.0)
        real_rate_change = nominal_rate_change - inflation_diff
        
        return {
            'nominal_rate_change': nominal_rate_change,
            'real_rate_change': real_rate_change,
            'reserve_change': intervention,
            'intervention': intervention
        }

class TradeModel:
    """
    International trade model
    """
    
    def __init__(self, params: SOEParameters):
        """
        Initialize trade model
        
        Args:
            params: Model parameters
        """
        self.params = params
    
    def calculate_trade_flows(self, economic_conditions: Dict, 
                            exchange_rates: Dict) -> Dict:
        """
        Calculate export and import flows
        
        Args:
            economic_conditions: Economic conditions
            exchange_rates: Exchange rate changes
            
        Returns:
            Trade flow results
        """
        # Export performance
        exports = self._calculate_exports(economic_conditions, exchange_rates)
        
        # Import demand
        imports = self._calculate_imports(economic_conditions, exchange_rates)
        
        # Trade balance
        trade_balance = exports['total'] - imports['total']
        
        # Terms of trade
        terms_of_trade = self._calculate_terms_of_trade(economic_conditions)
        
        return {
            'exports': exports,
            'imports': imports,
            'trade_balance': trade_balance,
            'terms_of_trade': terms_of_trade
        }
    
    def _calculate_exports(self, conditions: Dict, rates: Dict) -> Dict:
        """
        Calculate export performance by sector
        
        Args:
            conditions: Economic conditions
            rates: Exchange rate changes
            
        Returns:
            Export results
        """
        # World demand
        world_growth = conditions.get('world_growth', 0.03)
        
        # Competitiveness effect
        reer_change = rates.get('real_rate_change', 0.0)
        competitiveness_effect = -self.params.export_elasticity * reer_change
        
        # Textile exports (RMG)
        textile_growth = (
            self.params.export_income_elasticity * world_growth +
            competitiveness_effect +
            conditions.get('textile_demand_shock', 0.0)
        )
        
        # Agricultural exports
        agri_growth = (
            world_growth +
            0.5 * competitiveness_effect +  # Less price sensitive
            conditions.get('commodity_price_shock', 0.0)
        )
        
        # Other exports
        other_growth = (
            1.2 * world_growth +
            competitiveness_effect
        )
        
        # Calculate export values
        textile_exports = self.params.textile_export_share * (1 + textile_growth)
        agri_exports = self.params.agricultural_export_share * (1 + agri_growth)
        other_exports = self.params.other_export_share * (1 + other_growth)
        
        total_exports = textile_exports + agri_exports + other_exports
        
        return {
            'textile': textile_exports,
            'agricultural': agri_exports,
            'other': other_exports,
            'total': total_exports,
            'growth_rate': (total_exports - 1.0)
        }
    
    def _calculate_imports(self, conditions: Dict, rates: Dict) -> Dict:
        """
        Calculate import demand by category
        
        Args:
            conditions: Economic conditions
            rates: Exchange rate changes
            
        Returns:
            Import results
        """
        # Domestic demand
        domestic_growth = conditions.get('gdp_growth', 0.06)
        
        # Price effect
        reer_change = rates.get('real_rate_change', 0.0)
        price_effect = self.params.import_elasticity * reer_change
        
        # Fuel imports
        fuel_growth = (
            0.8 * domestic_growth +  # Less income elastic
            price_effect +
            conditions.get('oil_price_shock', 0.0)
        )
        
        # Capital goods imports
        capital_growth = (
            1.5 * domestic_growth +  # Investment driven
            0.8 * price_effect  # Less price sensitive
        )
        
        # Consumer goods imports
        consumer_growth = (
            self.params.import_income_elasticity * domestic_growth +
            price_effect
        )
        
        # Raw materials imports
        materials_growth = (
            1.2 * domestic_growth +  # Production driven
            0.6 * price_effect
        )
        
        # Calculate import values
        fuel_imports = self.params.fuel_import_share * (1 + fuel_growth)
        capital_imports = self.params.capital_goods_import_share * (1 + capital_growth)
        consumer_imports = self.params.consumer_goods_import_share * (1 + consumer_growth)
        materials_imports = self.params.raw_materials_import_share * (1 + materials_growth)
        
        total_imports = fuel_imports + capital_imports + consumer_imports + materials_imports
        
        return {
            'fuel': fuel_imports,
            'capital_goods': capital_imports,
            'consumer_goods': consumer_imports,
            'raw_materials': materials_imports,
            'total': total_imports,
            'growth_rate': (total_imports - 1.0)
        }
    
    def _calculate_terms_of_trade(self, conditions: Dict) -> float:
        """
        Calculate terms of trade changes
        
        Args:
            conditions: Economic conditions
            
        Returns:
            Terms of trade change
        """
        # Export price changes (mainly textile prices)
        export_price_change = conditions.get('textile_price_shock', 0.0)
        
        # Import price changes (mainly oil and commodity prices)
        oil_price_change = conditions.get('oil_price_shock', 0.0)
        commodity_price_change = conditions.get('commodity_price_shock', 0.0)
        
        # Weighted import price change
        import_price_change = (
            self.params.fuel_import_share * oil_price_change +
            (1 - self.params.fuel_import_share) * commodity_price_change
        )
        
        # Terms of trade = export prices / import prices
        terms_of_trade_change = export_price_change - import_price_change
        
        return terms_of_trade_change

class CapitalFlowModel:
    """
    Capital flows and financial account model
    """
    
    def __init__(self, params: SOEParameters):
        """
        Initialize capital flow model
        
        Args:
            params: Model parameters
        """
        self.params = params
    
    def calculate_capital_flows(self, economic_conditions: Dict, 
                              policy_rates: Dict) -> Dict:
        """
        Calculate capital flows
        
        Args:
            economic_conditions: Economic conditions
            policy_rates: Interest rates
            
        Returns:
            Capital flow results
        """
        # Foreign Direct Investment
        fdi = self._calculate_fdi(economic_conditions)
        
        # Portfolio investment
        portfolio = self._calculate_portfolio_flows(economic_conditions, policy_rates)
        
        # Remittances
        remittances = self._calculate_remittances(economic_conditions)
        
        # Other flows (trade credit, etc.)
        other_flows = self._calculate_other_flows(economic_conditions)
        
        # Total capital flows
        total_flows = fdi + portfolio + remittances + other_flows
        
        return {
            'fdi': fdi,
            'portfolio': portfolio,
            'remittances': remittances,
            'other_flows': other_flows,
            'total': total_flows
        }
    
    def _calculate_fdi(self, conditions: Dict) -> float:
        """
        Calculate FDI flows
        
        Args:
            conditions: Economic conditions
            
        Returns:
            FDI flows
        """
        # Base FDI
        base_fdi = self.params.fdi_gdp_ratio
        
        # Growth effect
        gdp_growth = conditions.get('gdp_growth', 0.06)
        growth_effect = 0.5 * (gdp_growth - 0.06)  # Positive correlation
        
        # Risk effect
        risk_premium = conditions.get('country_risk', 0.0)
        risk_effect = -0.3 * risk_premium
        
        # Policy effect
        policy_uncertainty = conditions.get('policy_uncertainty', 0.0)
        policy_effect = -0.2 * policy_uncertainty
        
        fdi = base_fdi * (1 + growth_effect + risk_effect + policy_effect)
        
        return max(0.0, fdi)
    
    def _calculate_portfolio_flows(self, conditions: Dict, rates: Dict) -> float:
        """
        Calculate portfolio investment flows
        
        Args:
            conditions: Economic conditions
            rates: Interest rates
            
        Returns:
            Portfolio flows
        """
        # Interest rate differential
        rate_differential = rates.get('interest_differential', self.params.interest_rate_differential)
        
        # Risk appetite
        global_risk_appetite = conditions.get('global_risk_appetite', 0.0)
        
        # Country risk
        country_risk = conditions.get('country_risk', 0.0)
        
        # Portfolio flows (can be negative)
        portfolio_flows = (
            0.5 * rate_differential +
            0.3 * global_risk_appetite -
            0.4 * country_risk
        )
        
        # Add volatility
        volatility_shock = np.random.normal(0, self.params.portfolio_volatility)
        portfolio_flows += volatility_shock
        
        return portfolio_flows * 0.02  # Scale to reasonable size
    
    def _calculate_remittances(self, conditions: Dict) -> float:
        """
        Calculate remittance flows
        
        Args:
            conditions: Economic conditions
            
        Returns:
            Remittance flows
        """
        # Base remittances
        base_remittances = self.params.remittance_gdp_ratio
        
        # Host country growth (mainly Middle East, US, Europe)
        host_growth = conditions.get('host_country_growth', 0.025)
        growth_effect = 0.8 * (host_growth - 0.025)
        
        # Oil price effect (Middle East remittances)
        oil_price_effect = 0.3 * conditions.get('oil_price_shock', 0.0)
        
        # Exchange rate effect (workers send more when taka depreciates)
        exchange_rate_effect = 0.2 * conditions.get('nominal_rate_change', 0.0)
        
        remittances = base_remittances * (1 + growth_effect + oil_price_effect + exchange_rate_effect)
        
        return max(0.0, remittances)
    
    def _calculate_other_flows(self, conditions: Dict) -> float:
        """
        Calculate other capital flows
        
        Args:
            conditions: Economic conditions
            
        Returns:
            Other flows
        """
        # Trade credit and other flows
        trade_growth = conditions.get('trade_growth', 0.05)
        other_flows = 0.01 * trade_growth  # Small positive correlation
        
        return other_flows

class ExternalVulnerabilityAssessment:
    """
    External vulnerability and sustainability analysis
    """
    
    def __init__(self, params: SOEParameters):
        """
        Initialize vulnerability assessment
        
        Args:
            params: Model parameters
        """
        self.params = params
    
    def assess_vulnerabilities(self, economic_data: Dict) -> Dict:
        """
        Comprehensive vulnerability assessment
        
        Args:
            economic_data: Economic indicators
            
        Returns:
            Vulnerability measures
        """
        # External debt sustainability
        debt_sustainability = self._assess_debt_sustainability(economic_data)
        
        # Liquidity indicators
        liquidity_indicators = self._assess_liquidity(economic_data)
        
        # Current account sustainability
        ca_sustainability = self._assess_current_account(economic_data)
        
        # Exchange rate assessment
        exchange_rate_assessment = self._assess_exchange_rate(economic_data)
        
        # Overall vulnerability score
        vulnerability_score = self._calculate_vulnerability_score({
            'debt': debt_sustainability,
            'liquidity': liquidity_indicators,
            'current_account': ca_sustainability,
            'exchange_rate': exchange_rate_assessment
        })
        
        return {
            'overall_score': vulnerability_score,
            'debt_sustainability': debt_sustainability,
            'liquidity_indicators': liquidity_indicators,
            'current_account': ca_sustainability,
            'exchange_rate_assessment': exchange_rate_assessment
        }
    
    def _assess_debt_sustainability(self, data: Dict) -> Dict:
        """
        Assess external debt sustainability
        
        Args:
            data: Economic data
            
        Returns:
            Debt sustainability measures
        """
        # Debt to GDP ratio
        debt_gdp = data.get('external_debt_gdp', self.params.external_debt_gdp)
        
        # Debt service ratio
        debt_service = data.get('debt_service_ratio', self.params.debt_service_ratio)
        
        # Debt sustainability thresholds
        debt_gdp_threshold = 0.40  # 40% of GDP
        debt_service_threshold = 0.15  # 15% of exports
        
        # Risk scores (0 = low risk, 1 = high risk)
        debt_gdp_risk = min(1.0, debt_gdp / debt_gdp_threshold)
        debt_service_risk = min(1.0, debt_service / debt_service_threshold)
        
        # Overall debt risk
        debt_risk = 0.6 * debt_gdp_risk + 0.4 * debt_service_risk
        
        return {
            'debt_gdp_ratio': debt_gdp,
            'debt_service_ratio': debt_service,
            'debt_gdp_risk': debt_gdp_risk,
            'debt_service_risk': debt_service_risk,
            'overall_debt_risk': debt_risk
        }
    
    def _assess_liquidity(self, data: Dict) -> Dict:
        """
        Assess liquidity indicators
        
        Args:
            data: Economic data
            
        Returns:
            Liquidity measures
        """
        # Import coverage
        import_coverage = data.get('import_coverage', self.params.import_coverage_months)
        
        # Short-term debt to reserves
        st_debt_reserves = data.get('st_debt_reserves', self.params.short_term_debt_reserves)
        
        # Liquidity thresholds
        import_coverage_threshold = 3.0  # 3 months
        st_debt_threshold = 1.0  # 100% coverage
        
        # Risk scores
        import_coverage_risk = max(0.0, 1.0 - import_coverage / import_coverage_threshold)
        st_debt_risk = max(0.0, st_debt_reserves / st_debt_threshold - 1.0)
        st_debt_risk = min(1.0, st_debt_risk)
        
        # Overall liquidity risk
        liquidity_risk = 0.7 * import_coverage_risk + 0.3 * st_debt_risk
        
        return {
            'import_coverage': import_coverage,
            'st_debt_reserves_ratio': st_debt_reserves,
            'import_coverage_risk': import_coverage_risk,
            'st_debt_risk': st_debt_risk,
            'overall_liquidity_risk': liquidity_risk
        }
    
    def _assess_current_account(self, data: Dict) -> Dict:
        """
        Assess current account sustainability
        
        Args:
            data: Economic data
            
        Returns:
            Current account measures
        """
        # Current account to GDP
        ca_gdp = data.get('current_account_gdp', self.params.initial_ca_gdp)
        
        # Current account sustainability threshold
        ca_threshold = -0.05  # -5% of GDP
        
        # Risk score
        ca_risk = max(0.0, -(ca_gdp - ca_threshold) / abs(ca_threshold))
        ca_risk = min(1.0, ca_risk)
        
        return {
            'current_account_gdp': ca_gdp,
            'ca_sustainability_risk': ca_risk
        }
    
    def _assess_exchange_rate(self, data: Dict) -> Dict:
        """
        Assess exchange rate sustainability
        
        Args:
            data: Economic data
            
        Returns:
            Exchange rate measures
        """
        # Real exchange rate level
        reer_level = data.get('reer_level', 100.0)
        
        # Equilibrium REER (simplified)
        equilibrium_reer = 100.0 + self.params.productivity_differential * 100
        
        # Misalignment
        misalignment = (reer_level - equilibrium_reer) / equilibrium_reer
        
        # Risk score (overvaluation is risky)
        exchange_rate_risk = max(0.0, misalignment / 0.20)  # 20% overvaluation threshold
        exchange_rate_risk = min(1.0, exchange_rate_risk)
        
        return {
            'reer_level': reer_level,
            'equilibrium_reer': equilibrium_reer,
            'misalignment': misalignment,
            'exchange_rate_risk': exchange_rate_risk
        }
    
    def _calculate_vulnerability_score(self, components: Dict) -> float:
        """
        Calculate overall vulnerability score
        
        Args:
            components: Vulnerability components
            
        Returns:
            Overall vulnerability score
        """
        # Weighted average of components
        weights = {
            'debt': 0.3,
            'liquidity': 0.3,
            'current_account': 0.2,
            'exchange_rate': 0.2
        }
        
        vulnerability_score = sum(
            weights[key] * components[key].get('overall_' + key.replace('_', '_') + '_risk', 
                                             components[key].get(key + '_risk', 0.0))
            for key in weights.keys()
        )
        
        return vulnerability_score

class SmallOpenEconomyModel:
    """
    Comprehensive Small Open Economy Model for Bangladesh
    
    This class integrates exchange rate determination, trade flows,
    capital flows, and external vulnerability analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Small Open Economy Model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Model parameters
        self.params = SOEParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Initialize components
        self.exchange_rate_model = ExchangeRateModel(self.params)
        self.trade_model = TradeModel(self.params)
        self.capital_flow_model = CapitalFlowModel(self.params)
        self.vulnerability_assessment = ExternalVulnerabilityAssessment(self.params)
        
        # State variables
        self.state = {
            'reer_level': self.params.initial_reer,
            'current_account_gdp': self.params.initial_ca_gdp,
            'reserves_months': self.params.initial_reserves_months,
            'external_debt_gdp': self.params.external_debt_gdp
        }
        
        # Results storage
        self.simulation_results = []
        
        logger.info("Small Open Economy Model initialized for Bangladesh")
    
    def simulate_economy(self, periods: int = None) -> pd.DataFrame:
        """
        Simulate small open economy dynamics
        
        Args:
            periods: Number of simulation periods
            
        Returns:
            Simulation results
        """
        if periods is None:
            periods = self.params.time_periods
        
        logger.info(f"Simulating small open economy for {periods} periods")
        
        results = []
        
        for t in range(periods):
            # Generate external conditions
            external_conditions = self._generate_external_conditions(t)
            
            # Generate domestic conditions
            domestic_conditions = self._generate_domestic_conditions(t)
            
            # Combine conditions
            economic_conditions = {**external_conditions, **domestic_conditions}
            
            # Generate shocks
            shocks = self._generate_shocks(t)
            
            # Calculate trade flows
            trade_results = self.trade_model.calculate_trade_flows(
                economic_conditions, {'real_rate_change': 0.0}  # Initial guess
            )
            
            # Calculate capital flows
            capital_results = self.capital_flow_model.calculate_capital_flows(
                economic_conditions, {'interest_differential': self.params.interest_rate_differential}
            )
            
            # Current account
            current_account = trade_results['trade_balance'] + capital_results['remittances']
            
            # Financial account
            financial_account = capital_results['fdi'] + capital_results['portfolio'] + capital_results['other_flows']
            
            # Exchange rate determination
            fundamentals = {
                'current_account': current_account,
                'capital_flows': financial_account,
                'inflation_differential': economic_conditions.get('inflation_differential', 0.0)
            }
            
            exchange_rate_results = self.exchange_rate_model.determine_exchange_rate(
                fundamentals, shocks
            )
            
            # Update state variables
            self._update_state(exchange_rate_results, current_account, financial_account)
            
            # Recalculate trade with updated exchange rates
            trade_results = self.trade_model.calculate_trade_flows(
                economic_conditions, exchange_rate_results
            )
            
            # Vulnerability assessment
            vulnerability = self.vulnerability_assessment.assess_vulnerabilities({
                'external_debt_gdp': self.state['external_debt_gdp'],
                'current_account_gdp': self.state['current_account_gdp'],
                'import_coverage': self.state['reserves_months'],
                'reer_level': self.state['reer_level']
            })
            
            # Store results
            result = {
                'period': t,
                'world_growth': external_conditions['world_growth'],
                'oil_price_shock': external_conditions.get('oil_price_shock', 0.0),
                'gdp_growth': domestic_conditions['gdp_growth'],
                'inflation': domestic_conditions['inflation'],
                'nominal_rate_change': exchange_rate_results['nominal_rate_change'],
                'real_rate_change': exchange_rate_results['real_rate_change'],
                'reer_level': self.state['reer_level'],
                'exports_growth': trade_results['exports']['growth_rate'],
                'imports_growth': trade_results['imports']['growth_rate'],
                'trade_balance': trade_results['trade_balance'],
                'current_account': current_account,
                'current_account_gdp': self.state['current_account_gdp'],
                'fdi': capital_results['fdi'],
                'portfolio_flows': capital_results['portfolio'],
                'remittances': capital_results['remittances'],
                'financial_account': financial_account,
                'reserves_change': exchange_rate_results['reserve_change'],
                'reserves_months': self.state['reserves_months'],
                'external_debt_gdp': self.state['external_debt_gdp'],
                'vulnerability_score': vulnerability['overall_score'],
                'terms_of_trade': trade_results['terms_of_trade']
            }
            
            results.append(result)
            self.simulation_results.append(result)
        
        return pd.DataFrame(results)
    
    def _generate_external_conditions(self, period: int) -> Dict:
        """
        Generate external economic conditions
        
        Args:
            period: Current period
            
        Returns:
            External conditions
        """
        # World growth with cycle
        trend_world_growth = 0.03
        cycle_component = 0.01 * np.sin(2 * np.pi * period / 15)  # 15-period cycle
        world_growth_shock = np.random.normal(0, self.params.world_growth_volatility)
        
        world_growth = trend_world_growth + cycle_component + world_growth_shock
        
        # Oil price shocks
        oil_price_shock = np.random.normal(0, 0.15)  # 15% volatility
        
        # Commodity price shocks
        commodity_price_shock = np.random.normal(0, 0.10)
        
        # Textile price shocks
        textile_price_shock = np.random.normal(0, 0.08)
        
        # Global risk appetite
        global_risk_appetite = np.random.normal(0, 0.2)
        
        return {
            'world_growth': world_growth,
            'oil_price_shock': oil_price_shock,
            'commodity_price_shock': commodity_price_shock,
            'textile_price_shock': textile_price_shock,
            'global_risk_appetite': global_risk_appetite,
            'host_country_growth': world_growth * 0.8  # Remittance source countries
        }
    
    def _generate_domestic_conditions(self, period: int) -> Dict:
        """
        Generate domestic economic conditions
        
        Args:
            period: Current period
            
        Returns:
            Domestic conditions
        """
        # GDP growth
        trend_gdp_growth = 0.06
        gdp_shock = np.random.normal(0, 0.015)
        gdp_growth = trend_gdp_growth + gdp_shock
        
        # Inflation
        target_inflation = 0.055
        inflation_shock = np.random.normal(0, 0.01)
        inflation = target_inflation + inflation_shock
        
        # Inflation differential (domestic - foreign)
        foreign_inflation = 0.02  # Developed country inflation
        inflation_differential = inflation - foreign_inflation
        
        return {
            'gdp_growth': gdp_growth,
            'inflation': inflation,
            'inflation_differential': inflation_differential,
            'country_risk': np.random.normal(0, 0.05),
            'policy_uncertainty': np.random.normal(0, 0.03)
        }
    
    def _generate_shocks(self, period: int) -> Dict:
        """
        Generate external shocks
        
        Args:
            period: Current period
            
        Returns:
            Shock realizations
        """
        # Exchange rate market shocks
        exchange_rate_shock = np.random.normal(0, 0.05)
        
        # Sudden stop scenario
        if period > 60 and period < 65:  # Crisis period
            exchange_rate_shock += 0.15  # Depreciation pressure
        
        return {
            'exchange_rate_shock': exchange_rate_shock
        }
    
    def _update_state(self, exchange_results: Dict, current_account: float, 
                     financial_account: float):
        """
        Update model state variables
        
        Args:
            exchange_results: Exchange rate results
            current_account: Current account balance
            financial_account: Financial account balance
        """
        # Update REER
        self.state['reer_level'] *= (1 + exchange_results['real_rate_change'])
        
        # Update current account to GDP ratio
        self.state['current_account_gdp'] = current_account
        
        # Update reserves
        reserve_change = exchange_results['reserve_change']
        # Convert to months of imports (simplified)
        self.state['reserves_months'] += reserve_change * 12  # Approximate conversion
        self.state['reserves_months'] = max(0.5, self.state['reserves_months'])  # Minimum reserves
        
        # Update external debt (simplified)
        debt_change = max(0, -current_account) * 0.5  # Half of deficit financed by debt
        self.state['external_debt_gdp'] += debt_change
    
    def analyze_policy_scenarios(self) -> Dict:
        """
        Analyze different policy scenarios
        
        Returns:
            Policy scenario results
        """
        logger.info("Analyzing policy scenarios")
        
        scenarios = {
            'baseline': {'exchange_rate_regime': 'managed_float'},
            'fixed_rate': {'exchange_rate_regime': 'fixed'},
            'floating_rate': {'exchange_rate_regime': 'float'},
            'capital_controls': {'portfolio_volatility': 0.1},  # Reduced volatility
            'export_promotion': {'export_elasticity': 1.5}  # Higher export response
        }
        
        scenario_results = {}
        
        for scenario_name, policy_changes in scenarios.items():
            # Save original parameters
            original_params = {}
            for key, value in policy_changes.items():
                if hasattr(self.params, key):
                    original_params[key] = getattr(self.params, key)
                    setattr(self.params, key, value)
                elif hasattr(self.exchange_rate_model, 'regime') and key == 'exchange_rate_regime':
                    original_params['regime'] = self.exchange_rate_model.regime
                    self.exchange_rate_model.regime = value
            
            # Reset state
            self.state = {
                'reer_level': self.params.initial_reer,
                'current_account_gdp': self.params.initial_ca_gdp,
                'reserves_months': self.params.initial_reserves_months,
                'external_debt_gdp': self.params.external_debt_gdp
            }
            
            # Run simulation
            results = self.simulate_economy(50)  # Shorter simulation
            
            # Calculate performance metrics
            performance = {
                'avg_gdp_growth': results['gdp_growth'].mean(),
                'gdp_volatility': results['gdp_growth'].std(),
                'avg_inflation': results['inflation'].mean(),
                'inflation_volatility': results['inflation'].std(),
                'exchange_rate_volatility': results['nominal_rate_change'].std(),
                'avg_current_account': results['current_account_gdp'].mean(),
                'avg_vulnerability': results['vulnerability_score'].mean(),
                'reserves_stability': results['reserves_months'].std()
            }
            
            scenario_results[scenario_name] = {
                'performance': performance,
                'final_state': {
                    'reer_level': self.state['reer_level'],
                    'current_account_gdp': self.state['current_account_gdp'],
                    'reserves_months': self.state['reserves_months'],
                    'vulnerability_score': results['vulnerability_score'].iloc[-1]
                }
            }
            
            # Restore original parameters
            for key, value in original_params.items():
                if key == 'regime':
                    self.exchange_rate_model.regime = value
                else:
                    setattr(self.params, key, value)
        
        return scenario_results
    
    def conduct_stress_tests(self) -> Dict:
        """
        Conduct external stress tests
        
        Returns:
            Stress test results
        """
        logger.info("Conducting external stress tests")
        
        stress_scenarios = {
            'global_recession': {
                'world_growth_shock': -0.03,
                'oil_price_shock': 0.5,
                'risk_appetite_shock': -0.5
            },
            'oil_price_spike': {
                'oil_price_shock': 0.8,
                'commodity_price_shock': 0.3
            },
            'sudden_stop': {
                'portfolio_flow_shock': -0.8,
                'exchange_rate_shock': 0.25
            },
            'textile_demand_collapse': {
                'textile_demand_shock': -0.3,
                'world_growth_shock': -0.01
            },
            'remittance_decline': {
                'host_country_growth_shock': -0.04,
                'oil_price_shock': -0.3
            }
        }
        
        stress_results = {}
        
        for scenario_name, shocks in stress_scenarios.items():
            # Reset state
            self.state = {
                'reer_level': self.params.initial_reer,
                'current_account_gdp': self.params.initial_ca_gdp,
                'reserves_months': self.params.initial_reserves_months,
                'external_debt_gdp': self.params.external_debt_gdp
            }
            
            # Apply stress for 10 periods
            stressed_results = []
            
            for t in range(10):
                # Generate stressed conditions
                external_conditions = self._generate_external_conditions(t)
                domestic_conditions = self._generate_domestic_conditions(t)
                
                # Apply shocks
                for shock_key, shock_value in shocks.items():
                    if 'world_growth' in shock_key:
                        external_conditions['world_growth'] += shock_value
                    elif 'oil_price' in shock_key:
                        external_conditions['oil_price_shock'] += shock_value
                    elif 'risk_appetite' in shock_key:
                        external_conditions['global_risk_appetite'] += shock_value
                    elif 'textile_demand' in shock_key:
                        external_conditions['textile_demand_shock'] = shock_value
                    elif 'host_country' in shock_key:
                        external_conditions['host_country_growth'] += shock_value
                
                # Simulate one period
                economic_conditions = {**external_conditions, **domestic_conditions}
                
                # Calculate impacts (simplified)
                trade_impact = -abs(shocks.get('world_growth_shock', 0)) * 2
                capital_impact = shocks.get('portfolio_flow_shock', 0) * 0.02
                exchange_impact = shocks.get('exchange_rate_shock', 0)
                
                # Update vulnerability
                vulnerability_change = abs(trade_impact) + abs(capital_impact) + abs(exchange_impact)
                
                stressed_results.append({
                    'period': t,
                    'trade_impact': trade_impact,
                    'capital_impact': capital_impact,
                    'exchange_impact': exchange_impact,
                    'vulnerability_increase': vulnerability_change
                })
            
            # Aggregate stress results
            stress_df = pd.DataFrame(stressed_results)
            
            stress_results[scenario_name] = {
                'max_trade_impact': stress_df['trade_impact'].min(),
                'max_capital_impact': stress_df['capital_impact'].min(),
                'max_exchange_impact': stress_df['exchange_impact'].max(),
                'avg_vulnerability_increase': stress_df['vulnerability_increase'].mean(),
                'cumulative_impact': stress_df['vulnerability_increase'].sum()
            }
        
        return stress_results
    
    def plot_simulation_results(self, results: pd.DataFrame, save_path: str = None):
        """
        Plot simulation results
        
        Args:
            results: Simulation results
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Exchange rates
        axes[0, 0].plot(results['period'], results['reer_level'], 'b-', linewidth=2)
        axes[0, 0].set_title('Real Effective Exchange Rate')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('REER Index')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trade balance
        axes[0, 1].plot(results['period'], results['trade_balance'] * 100, 'g-', linewidth=2)
        axes[0, 1].set_title('Trade Balance (% of GDP)')
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Trade Balance (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Current account
        axes[0, 2].plot(results['period'], results['current_account_gdp'] * 100, 'r-', linewidth=2)
        axes[0, 2].set_title('Current Account (% of GDP)')
        axes[0, 2].set_xlabel('Period')
        axes[0, 2].set_ylabel('Current Account (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Capital flows
        axes[1, 0].plot(results['period'], results['fdi'] * 100, 'c-', label='FDI', linewidth=2)
        axes[1, 0].plot(results['period'], results['portfolio_flows'] * 100, 'm-', label='Portfolio', linewidth=2)
        axes[1, 0].plot(results['period'], results['remittances'] * 100, 'y-', label='Remittances', linewidth=2)
        axes[1, 0].set_title('Capital Flows (% of GDP)')
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].set_ylabel('Flows (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reserves
        axes[1, 1].plot(results['period'], results['reserves_months'], 'orange', linewidth=2)
        axes[1, 1].axhline(y=3.0, color='r', linestyle='--', label='Minimum Threshold')
        axes[1, 1].set_title('Foreign Reserves (Months of Imports)')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('Months')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # External debt
        axes[1, 2].plot(results['period'], results['external_debt_gdp'] * 100, 'purple', linewidth=2)
        axes[1, 2].set_title('External Debt (% of GDP)')
        axes[1, 2].set_xlabel('Period')
        axes[1, 2].set_ylabel('Debt (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Export and import growth
        axes[2, 0].plot(results['period'], results['exports_growth'] * 100, 'g-', label='Exports', linewidth=2)
        axes[2, 0].plot(results['period'], results['imports_growth'] * 100, 'r-', label='Imports', linewidth=2)
        axes[2, 0].set_title('Trade Growth Rates')
        axes[2, 0].set_xlabel('Period')
        axes[2, 0].set_ylabel('Growth Rate (%)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Terms of trade
        axes[2, 1].plot(results['period'], results['terms_of_trade'] * 100, 'brown', linewidth=2)
        axes[2, 1].set_title('Terms of Trade Change')
        axes[2, 1].set_xlabel('Period')
        axes[2, 1].set_ylabel('Change (%)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Vulnerability score
        axes[2, 2].plot(results['period'], results['vulnerability_score'], 'darkred', linewidth=2)
        axes[2, 2].set_title('External Vulnerability Score')
        axes[2, 2].set_xlabel('Period')
        axes[2, 2].set_ylabel('Vulnerability Score')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simulation results plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary
        """
        summary = {
            'model_type': 'Small Open Economy Model',
            'country': 'Bangladesh',
            'exchange_rate_regime': self.params.exchange_rate_regime,
            'key_features': [
                'Exchange Rate Determination',
                'International Trade Modeling',
                'Capital Flow Analysis',
                'External Vulnerability Assessment',
                'Balance of Payments Dynamics',
                'Policy Scenario Analysis'
            ],
            'bangladesh_features': [
                'RMG Export Dominance',
                'Remittance Flows',
                'Import Dependency',
                'Managed Float Regime',
                'External Debt Sustainability'
            ],
            'trade_structure': {
                'textile_exports': f"{self.params.textile_export_share:.1%}",
                'fuel_imports': f"{self.params.fuel_import_share:.1%}",
                'remittance_gdp': f"{self.params.remittance_gdp_ratio:.1%}"
            },
            'current_state': {
                'reer_level': f"{self.state['reer_level']:.1f}",
                'current_account_gdp': f"{self.state['current_account_gdp']:.1%}",
                'reserves_months': f"{self.state['reserves_months']:.1f}",
                'external_debt_gdp': f"{self.state['external_debt_gdp']:.1%}"
            }
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'time_periods': 50,
            'exchange_rate_regime': 'managed_float',
            'textile_export_share': 0.84,
            'remittance_gdp_ratio': 0.055,
            'external_debt_gdp': 0.20
        }
    }
    
    # Initialize Small Open Economy Model
    soe_model = SmallOpenEconomyModel(config)
    
    # Run simulation
    print("Running small open economy simulation...")
    results = soe_model.simulate_economy()
    
    # Display key results
    final_period = results.iloc[-1]
    print(f"\nSimulation Results (Final Period):")
    print(f"  REER Level: {final_period['reer_level']:.1f}")
    print(f"  Current Account/GDP: {final_period['current_account_gdp']:.1%}")
    print(f"  Trade Balance/GDP: {final_period['trade_balance']:.1%}")
    print(f"  Reserves (months): {final_period['reserves_months']:.1f}")
    print(f"  External Debt/GDP: {final_period['external_debt_gdp']:.1%}")
    print(f"  Vulnerability Score: {final_period['vulnerability_score']:.3f}")
    
    # Analyze policy scenarios
    print("\nAnalyzing policy scenarios...")
    policy_results = soe_model.analyze_policy_scenarios()
    
    print("\nPolicy Scenario Results:")
    for scenario, results_dict in policy_results.items():
        perf = results_dict['performance']
        print(f"  {scenario}:")
        print(f"    Avg GDP Growth: {perf['avg_gdp_growth']:.1%}")
        print(f"    Exchange Rate Volatility: {perf['exchange_rate_volatility']:.3f}")
        print(f"    Avg Vulnerability: {perf['avg_vulnerability']:.3f}")
    
    # Conduct stress tests
    print("\nConducting stress tests...")
    stress_results = soe_model.conduct_stress_tests()
    
    print("\nStress Test Results:")
    for scenario, results_dict in stress_results.items():
        print(f"  {scenario}:")
        print(f"    Max Trade Impact: {results_dict['max_trade_impact']:.1%}")
        print(f"    Max Exchange Impact: {results_dict['max_exchange_impact']:.1%}")
        print(f"    Cumulative Impact: {results_dict['cumulative_impact']:.3f}")
    
    # Model summary
    summary = soe_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Exchange Rate Regime: {summary['exchange_rate_regime']}")
    print(f"  Key features: {len(summary['key_features'])}")
    print(f"  Bangladesh features: {len(summary['bangladesh_features'])}")
    
    print("\nSmall Open Economy Model analysis completed successfully!")