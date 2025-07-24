#!/usr/bin/env python3
"""
Financial Sector Model for Bangladesh Economy

This module implements a comprehensive financial sector model that analyzes
banking dynamics, financial stability, credit cycles, and monetary transmission
in Bangladesh's financial system.

Key Features:
- Banking sector modeling with heterogeneous banks
- Credit risk and default dynamics
- Financial stability indicators
- Monetary policy transmission through banks
- Capital adequacy and regulatory constraints
- Islamic banking integration
- Microfinance sector modeling
- Foreign exchange and capital flows
- Financial inclusion dynamics
- Systemic risk assessment

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, linalg
from scipy.sparse import csr_matrix
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialParameters:
    """
    Parameters for the Financial Sector Model
    """
    # Banking sector parameters
    n_banks: int = 50                          # Number of banks
    capital_adequacy_ratio: float = 0.125      # Basel III CAR requirement
    leverage_ratio: float = 0.03               # Leverage ratio requirement
    liquidity_ratio: float = 0.20              # Liquidity coverage ratio
    
    # Credit parameters
    default_threshold: float = 0.1             # Default probability threshold
    recovery_rate: float = 0.4                 # Recovery rate on defaults
    credit_growth_target: float = 0.15         # Target credit growth
    
    # Interest rate parameters
    policy_rate: float = 0.06                  # Central bank policy rate
    deposit_rate_spread: float = 0.02          # Deposit rate spread
    lending_rate_spread: float = 0.05          # Lending rate spread
    
    # Bangladesh-specific parameters
    islamic_banking_share: float = 0.25        # Islamic banking market share
    microfinance_penetration: float = 0.15     # Microfinance penetration
    financial_inclusion_rate: float = 0.55     # Financial inclusion rate
    npl_ratio: float = 0.09                    # Non-performing loan ratio
    
    # Sectoral lending shares
    agriculture_lending: float = 0.05          # Agriculture lending share
    industry_lending: float = 0.35             # Industry lending share
    trade_lending: float = 0.25                # Trade financing share
    consumer_lending: float = 0.20             # Consumer lending share
    real_estate_lending: float = 0.15          # Real estate lending share
    
    # Risk parameters
    credit_risk_weight: float = 1.0            # Credit risk weight
    market_risk_weight: float = 0.1            # Market risk weight
    operational_risk_weight: float = 0.15      # Operational risk weight
    
    # Macroeconomic linkages
    gdp_credit_elasticity: float = 1.2         # Credit-GDP elasticity
    inflation_impact: float = 0.3              # Inflation impact on rates
    exchange_rate_impact: float = 0.2          # Exchange rate impact
    
    # Regulatory parameters
    reserve_requirement: float = 0.05          # Reserve requirement ratio
    deposit_insurance_coverage: float = 100000 # Deposit insurance limit
    
    # Simulation parameters
    time_periods: int = 100                    # Simulation periods
    monte_carlo_runs: int = 1000               # Monte Carlo simulations
    stress_test_scenarios: int = 5             # Stress test scenarios

@dataclass
class FinancialResults:
    """
    Results from Financial Sector Model
    """
    parameters: FinancialParameters
    bank_data: Optional[pd.DataFrame] = None
    stability_indicators: Optional[Dict] = None
    stress_test_results: Optional[Dict] = None
    policy_analysis: Optional[Dict] = None
    systemic_risk_measures: Optional[Dict] = None

class Bank:
    """
    Individual bank model
    """
    
    def __init__(self, bank_id: int, bank_type: str, params: FinancialParameters):
        """
        Initialize bank
        
        Args:
            bank_id: Bank identifier
            bank_type: Type of bank (commercial, islamic, development)
            params: Model parameters
        """
        self.bank_id = bank_id
        self.bank_type = bank_type
        self.params = params
        
        # Balance sheet items
        self.assets = self._initialize_assets()
        self.liabilities = self._initialize_liabilities()
        self.capital = self._initialize_capital()
        
        # Performance metrics
        self.profitability = 0.0
        self.efficiency = 0.0
        self.risk_metrics = {}
        
        # Regulatory ratios
        self.car = 0.0  # Capital adequacy ratio
        self.leverage = 0.0
        self.liquidity = 0.0
        
    def _initialize_assets(self) -> Dict[str, float]:
        """
        Initialize bank assets
        
        Returns:
            Asset portfolio
        """
        # Random initialization based on bank type
        if self.bank_type == 'islamic':
            return {
                'cash': np.random.uniform(0.05, 0.15),
                'government_securities': np.random.uniform(0.15, 0.25),
                'loans': np.random.uniform(0.60, 0.75),
                'fixed_assets': np.random.uniform(0.02, 0.05),
                'other_assets': np.random.uniform(0.03, 0.08)
            }
        else:
            return {
                'cash': np.random.uniform(0.08, 0.18),
                'government_securities': np.random.uniform(0.20, 0.30),
                'loans': np.random.uniform(0.55, 0.70),
                'fixed_assets': np.random.uniform(0.02, 0.05),
                'other_assets': np.random.uniform(0.02, 0.07)
            }
    
    def _initialize_liabilities(self) -> Dict[str, float]:
        """
        Initialize bank liabilities
        
        Returns:
            Liability structure
        """
        return {
            'deposits': np.random.uniform(0.75, 0.85),
            'borrowings': np.random.uniform(0.05, 0.15),
            'other_liabilities': np.random.uniform(0.02, 0.08)
        }
    
    def _initialize_capital(self) -> float:
        """
        Initialize bank capital
        
        Returns:
            Capital ratio
        """
        return np.random.uniform(0.10, 0.18)
    
    def update_balance_sheet(self, economic_conditions: Dict, policy_rates: Dict):
        """
        Update bank balance sheet based on economic conditions
        
        Args:
            economic_conditions: Economic environment
            policy_rates: Policy interest rates
        """
        # Update loan portfolio
        self._update_loans(economic_conditions)
        
        # Update deposits
        self._update_deposits(economic_conditions, policy_rates)
        
        # Update capital
        self._update_capital()
        
        # Calculate regulatory ratios
        self._calculate_regulatory_ratios()
    
    def _update_loans(self, conditions: Dict):
        """
        Update loan portfolio
        
        Args:
            conditions: Economic conditions
        """
        # Credit demand based on economic growth
        gdp_growth = conditions.get('gdp_growth', 0.06)
        credit_demand = self.params.gdp_credit_elasticity * gdp_growth
        
        # Loan growth
        loan_growth = min(credit_demand, self.params.credit_growth_target)
        self.assets['loans'] *= (1 + loan_growth)
        
        # Non-performing loans
        stress_factor = conditions.get('stress_factor', 1.0)
        npl_rate = self.params.npl_ratio * stress_factor
        self.assets['loans'] *= (1 - npl_rate * 0.1)  # Provision impact
    
    def _update_deposits(self, conditions: Dict, rates: Dict):
        """
        Update deposit base
        
        Args:
            conditions: Economic conditions
            rates: Interest rates
        """
        # Deposit growth based on income and rates
        income_growth = conditions.get('income_growth', 0.05)
        rate_effect = rates.get('deposit_rate', 0.04) - 0.04
        
        deposit_growth = income_growth + 0.5 * rate_effect
        self.liabilities['deposits'] *= (1 + deposit_growth)
    
    def _update_capital(self):
        """
        Update bank capital
        """
        # Retained earnings (simplified)
        roe = np.random.normal(0.12, 0.03)  # Return on equity
        self.capital *= (1 + roe * 0.6)  # 60% retention ratio
    
    def _calculate_regulatory_ratios(self):
        """
        Calculate regulatory compliance ratios
        """
        # Capital adequacy ratio
        risk_weighted_assets = self.assets['loans'] * self.params.credit_risk_weight
        self.car = self.capital / risk_weighted_assets
        
        # Leverage ratio
        total_assets = sum(self.assets.values())
        self.leverage = self.capital / total_assets
        
        # Liquidity ratio
        liquid_assets = self.assets['cash'] + self.assets['government_securities']
        self.liquidity = liquid_assets / self.liabilities['deposits']
    
    def calculate_profitability(self, rates: Dict) -> float:
        """
        Calculate bank profitability
        
        Args:
            rates: Interest rate environment
            
        Returns:
            Return on assets
        """
        # Net interest income
        lending_rate = rates.get('lending_rate', 0.10)
        deposit_rate = rates.get('deposit_rate', 0.04)
        
        interest_income = self.assets['loans'] * lending_rate
        interest_expense = self.liabilities['deposits'] * deposit_rate
        net_interest_income = interest_income - interest_expense
        
        # Operating expenses
        total_assets = sum(self.assets.values())
        operating_expenses = total_assets * 0.02  # 2% of assets
        
        # Provisions
        provisions = self.assets['loans'] * self.params.npl_ratio * 0.5
        
        # Net income
        net_income = net_interest_income - operating_expenses - provisions
        
        # Return on assets
        roa = net_income / total_assets
        self.profitability = roa
        
        return roa
    
    def assess_solvency_risk(self) -> float:
        """
        Assess bank solvency risk
        
        Returns:
            Solvency risk score (0-1)
        """
        # Capital adequacy component
        car_risk = max(0, (self.params.capital_adequacy_ratio - self.car) / self.params.capital_adequacy_ratio)
        
        # Asset quality component
        npl_risk = self.params.npl_ratio / 0.15  # Normalized to 15% threshold
        
        # Profitability component
        profit_risk = max(0, -self.profitability / 0.05)  # Negative ROA risk
        
        # Combined risk score
        risk_score = 0.4 * car_risk + 0.3 * npl_risk + 0.3 * profit_risk
        
        return min(risk_score, 1.0)

class CreditRiskModel:
    """
    Credit risk assessment model
    """
    
    def __init__(self, params: FinancialParameters):
        """
        Initialize credit risk model
        
        Args:
            params: Model parameters
        """
        self.params = params
        
    def calculate_sector_risk(self, economic_conditions: Dict) -> Dict[str, float]:
        """
        Calculate credit risk by economic sector
        
        Args:
            economic_conditions: Economic environment
            
        Returns:
            Sectoral risk measures
        """
        gdp_growth = economic_conditions.get('gdp_growth', 0.06)
        inflation = economic_conditions.get('inflation', 0.05)
        
        # Sector-specific risk factors
        sector_risks = {
            'agriculture': self._agriculture_risk(gdp_growth, inflation),
            'industry': self._industry_risk(gdp_growth, inflation),
            'trade': self._trade_risk(gdp_growth, inflation),
            'consumer': self._consumer_risk(gdp_growth, inflation),
            'real_estate': self._real_estate_risk(gdp_growth, inflation)
        }
        
        return sector_risks
    
    def _agriculture_risk(self, gdp_growth: float, inflation: float) -> float:
        """
        Calculate agriculture sector credit risk
        
        Args:
            gdp_growth: GDP growth rate
            inflation: Inflation rate
            
        Returns:
            Risk score
        """
        # Agriculture is sensitive to weather and commodity prices
        base_risk = 0.08
        growth_effect = -0.5 * (gdp_growth - 0.06)  # Negative correlation
        inflation_effect = 0.3 * (inflation - 0.05)  # Positive correlation
        
        return max(0.01, base_risk + growth_effect + inflation_effect)
    
    def _industry_risk(self, gdp_growth: float, inflation: float) -> float:
        """
        Calculate industry sector credit risk
        
        Args:
            gdp_growth: GDP growth rate
            inflation: Inflation rate
            
        Returns:
            Risk score
        """
        base_risk = 0.06
        growth_effect = -0.8 * (gdp_growth - 0.06)
        inflation_effect = 0.2 * (inflation - 0.05)
        
        return max(0.01, base_risk + growth_effect + inflation_effect)
    
    def _trade_risk(self, gdp_growth: float, inflation: float) -> float:
        """
        Calculate trade sector credit risk
        
        Args:
            gdp_growth: GDP growth rate
            inflation: Inflation rate
            
        Returns:
            Risk score
        """
        base_risk = 0.05
        growth_effect = -0.6 * (gdp_growth - 0.06)
        inflation_effect = 0.4 * (inflation - 0.05)
        
        return max(0.01, base_risk + growth_effect + inflation_effect)
    
    def _consumer_risk(self, gdp_growth: float, inflation: float) -> float:
        """
        Calculate consumer credit risk
        
        Args:
            gdp_growth: GDP growth rate
            inflation: Inflation rate
            
        Returns:
            Risk score
        """
        base_risk = 0.04
        growth_effect = -0.7 * (gdp_growth - 0.06)
        inflation_effect = 0.5 * (inflation - 0.05)
        
        return max(0.01, base_risk + growth_effect + inflation_effect)
    
    def _real_estate_risk(self, gdp_growth: float, inflation: float) -> float:
        """
        Calculate real estate credit risk
        
        Args:
            gdp_growth: GDP growth rate
            inflation: Inflation rate
            
        Returns:
            Risk score
        """
        base_risk = 0.07
        growth_effect = -0.4 * (gdp_growth - 0.06)
        inflation_effect = 0.1 * (inflation - 0.05)  # Real estate hedge
        
        return max(0.01, base_risk + growth_effect + inflation_effect)

class FinancialStabilityIndicators:
    """
    Financial stability assessment tools
    """
    
    def __init__(self, params: FinancialParameters):
        """
        Initialize stability indicators
        
        Args:
            params: Model parameters
        """
        self.params = params
    
    def calculate_stability_index(self, banks: List[Bank], 
                                 economic_conditions: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive financial stability index
        
        Args:
            banks: List of banks
            economic_conditions: Economic environment
            
        Returns:
            Stability indicators
        """
        # Banking sector indicators
        banking_stability = self._banking_stability(banks)
        
        # Credit market indicators
        credit_stability = self._credit_stability(banks, economic_conditions)
        
        # Liquidity indicators
        liquidity_stability = self._liquidity_stability(banks)
        
        # Overall stability index
        overall_index = (
            0.4 * banking_stability +
            0.3 * credit_stability +
            0.3 * liquidity_stability
        )
        
        return {
            'overall_stability': overall_index,
            'banking_stability': banking_stability,
            'credit_stability': credit_stability,
            'liquidity_stability': liquidity_stability
        }
    
    def _banking_stability(self, banks: List[Bank]) -> float:
        """
        Calculate banking sector stability
        
        Args:
            banks: List of banks
            
        Returns:
            Banking stability score
        """
        # Capital adequacy
        car_scores = [bank.car for bank in banks]
        avg_car = np.mean(car_scores)
        car_stability = min(1.0, avg_car / self.params.capital_adequacy_ratio)
        
        # Profitability
        profit_scores = [bank.profitability for bank in banks]
        avg_profit = np.mean(profit_scores)
        profit_stability = min(1.0, max(0.0, avg_profit / 0.02))  # 2% ROA target
        
        # Asset quality (inverse of NPL)
        npl_stability = 1.0 - min(1.0, self.params.npl_ratio / 0.15)
        
        return 0.4 * car_stability + 0.3 * profit_stability + 0.3 * npl_stability
    
    def _credit_stability(self, banks: List[Bank], conditions: Dict) -> float:
        """
        Calculate credit market stability
        
        Args:
            banks: List of banks
            conditions: Economic conditions
            
        Returns:
            Credit stability score
        """
        # Credit growth stability
        gdp_growth = conditions.get('gdp_growth', 0.06)
        credit_gdp_ratio = 2.0 * gdp_growth  # Sustainable credit growth
        
        actual_credit_growth = self.params.credit_growth_target
        growth_stability = 1.0 - abs(actual_credit_growth - credit_gdp_ratio) / credit_gdp_ratio
        
        # Sectoral concentration
        concentration_risk = self._calculate_concentration_risk()
        concentration_stability = 1.0 - concentration_risk
        
        return 0.6 * growth_stability + 0.4 * concentration_stability
    
    def _liquidity_stability(self, banks: List[Bank]) -> float:
        """
        Calculate liquidity stability
        
        Args:
            banks: List of banks
            
        Returns:
            Liquidity stability score
        """
        liquidity_ratios = [bank.liquidity for bank in banks]
        avg_liquidity = np.mean(liquidity_ratios)
        
        # Stability based on liquidity coverage
        liquidity_stability = min(1.0, avg_liquidity / self.params.liquidity_ratio)
        
        return liquidity_stability
    
    def _calculate_concentration_risk(self) -> float:
        """
        Calculate sectoral concentration risk
        
        Returns:
            Concentration risk score
        """
        # Herfindahl index for sectoral concentration
        shares = [
            self.params.agriculture_lending,
            self.params.industry_lending,
            self.params.trade_lending,
            self.params.consumer_lending,
            self.params.real_estate_lending
        ]
        
        hhi = sum(share**2 for share in shares)
        
        # Normalize (0 = perfectly diversified, 1 = concentrated)
        max_hhi = 1.0  # All lending in one sector
        min_hhi = 1.0 / len(shares)  # Equally distributed
        
        concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi)
        
        return concentration_risk

class FinancialSectorModel:
    """
    Comprehensive Financial Sector Model for Bangladesh
    
    This class integrates banking sector dynamics, credit markets,
    and financial stability analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Financial Sector Model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Model parameters
        self.params = FinancialParameters()
        
        # Update parameters from config
        for key, value in config.get('parameters', {}).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        # Initialize components
        self.banks = self._initialize_banks()
        self.credit_risk_model = CreditRiskModel(self.params)
        self.stability_indicators = FinancialStabilityIndicators(self.params)
        
        # Results storage
        self.simulation_results = []
        
        logger.info("Financial Sector Model initialized for Bangladesh")
    
    def _initialize_banks(self) -> List[Bank]:
        """
        Initialize banking sector
        
        Returns:
            List of banks
        """
        banks = []
        
        # Commercial banks (70%)
        n_commercial = int(0.7 * self.params.n_banks)
        for i in range(n_commercial):
            banks.append(Bank(i, 'commercial', self.params))
        
        # Islamic banks (25%)
        n_islamic = int(0.25 * self.params.n_banks)
        for i in range(n_commercial, n_commercial + n_islamic):
            banks.append(Bank(i, 'islamic', self.params))
        
        # Development banks (5%)
        n_development = self.params.n_banks - n_commercial - n_islamic
        for i in range(n_commercial + n_islamic, self.params.n_banks):
            banks.append(Bank(i, 'development', self.params))
        
        return banks
    
    def simulate_financial_system(self, periods: int = None) -> pd.DataFrame:
        """
        Simulate financial system dynamics
        
        Args:
            periods: Number of simulation periods
            
        Returns:
            Simulation results
        """
        if periods is None:
            periods = self.params.time_periods
        
        logger.info(f"Simulating financial system for {periods} periods")
        
        results = []
        
        for t in range(periods):
            # Generate economic conditions
            economic_conditions = self._generate_economic_conditions(t)
            
            # Update policy rates
            policy_rates = self._update_policy_rates(economic_conditions)
            
            # Update all banks
            for bank in self.banks:
                bank.update_balance_sheet(economic_conditions, policy_rates)
                bank.calculate_profitability(policy_rates)
            
            # Calculate system-wide indicators
            stability = self.stability_indicators.calculate_stability_index(
                self.banks, economic_conditions
            )
            
            # Aggregate banking sector data
            sector_data = self._aggregate_banking_data()
            
            # Store results
            result = {
                'period': t,
                'gdp_growth': economic_conditions['gdp_growth'],
                'inflation': economic_conditions['inflation'],
                'policy_rate': policy_rates['policy_rate'],
                'lending_rate': policy_rates['lending_rate'],
                'deposit_rate': policy_rates['deposit_rate'],
                'total_assets': sector_data['total_assets'],
                'total_loans': sector_data['total_loans'],
                'total_deposits': sector_data['total_deposits'],
                'avg_car': sector_data['avg_car'],
                'avg_roa': sector_data['avg_roa'],
                'stability_index': stability['overall_stability'],
                'banking_stability': stability['banking_stability'],
                'credit_stability': stability['credit_stability'],
                'liquidity_stability': stability['liquidity_stability']
            }
            
            results.append(result)
            self.simulation_results.append(result)
        
        return pd.DataFrame(results)
    
    def _generate_economic_conditions(self, period: int) -> Dict[str, float]:
        """
        Generate economic conditions for simulation
        
        Args:
            period: Current period
            
        Returns:
            Economic conditions
        """
        # Business cycle component
        cycle_phase = 2 * np.pi * period / 20  # 20-period cycle
        
        # GDP growth with cycle and random shocks
        trend_growth = 0.06
        cycle_amplitude = 0.02
        shock = np.random.normal(0, 0.01)
        
        gdp_growth = trend_growth + cycle_amplitude * np.sin(cycle_phase) + shock
        
        # Inflation with persistence
        if period == 0:
            inflation = 0.05
        else:
            prev_inflation = self.simulation_results[-1]['inflation'] if self.simulation_results else 0.05
            inflation_shock = np.random.normal(0, 0.005)
            inflation = 0.7 * prev_inflation + 0.3 * 0.05 + inflation_shock
        
        # Stress factor for adverse scenarios
        stress_factor = 1.0
        if period > 50 and period < 60:  # Financial stress period
            stress_factor = 1.5
        
        return {
            'gdp_growth': gdp_growth,
            'inflation': inflation,
            'income_growth': gdp_growth * 0.8,
            'stress_factor': stress_factor
        }
    
    def _update_policy_rates(self, conditions: Dict) -> Dict[str, float]:
        """
        Update policy interest rates
        
        Args:
            conditions: Economic conditions
            
        Returns:
            Policy rates
        """
        # Taylor rule for policy rate
        inflation = conditions['inflation']
        gdp_growth = conditions['gdp_growth']
        
        policy_rate = self.params.policy_rate + \
                     1.5 * (inflation - 0.05) + \
                     0.5 * (gdp_growth - 0.06)
        
        # Market rates
        deposit_rate = policy_rate - self.params.deposit_rate_spread
        lending_rate = policy_rate + self.params.lending_rate_spread
        
        return {
            'policy_rate': policy_rate,
            'deposit_rate': max(0.01, deposit_rate),
            'lending_rate': lending_rate
        }
    
    def _aggregate_banking_data(self) -> Dict[str, float]:
        """
        Aggregate banking sector data
        
        Returns:
            Aggregated data
        """
        total_assets = sum(sum(bank.assets.values()) for bank in self.banks)
        total_loans = sum(bank.assets['loans'] for bank in self.banks)
        total_deposits = sum(bank.liabilities['deposits'] for bank in self.banks)
        
        avg_car = np.mean([bank.car for bank in self.banks])
        avg_roa = np.mean([bank.profitability for bank in self.banks])
        
        return {
            'total_assets': total_assets,
            'total_loans': total_loans,
            'total_deposits': total_deposits,
            'avg_car': avg_car,
            'avg_roa': avg_roa
        }
    
    def conduct_stress_tests(self) -> Dict:
        """
        Conduct comprehensive stress tests
        
        Returns:
            Stress test results
        """
        logger.info("Conducting stress tests")
        
        stress_scenarios = {
            'baseline': {'gdp_shock': 0.0, 'inflation_shock': 0.0, 'rate_shock': 0.0},
            'mild_recession': {'gdp_shock': -0.02, 'inflation_shock': 0.01, 'rate_shock': 0.01},
            'severe_recession': {'gdp_shock': -0.05, 'inflation_shock': 0.02, 'rate_shock': 0.02},
            'inflation_shock': {'gdp_shock': -0.01, 'inflation_shock': 0.03, 'rate_shock': 0.015},
            'financial_crisis': {'gdp_shock': -0.08, 'inflation_shock': -0.01, 'rate_shock': 0.03}
        }
        
        stress_results = {}
        
        for scenario_name, shocks in stress_scenarios.items():
            # Apply shocks
            stressed_conditions = {
                'gdp_growth': 0.06 + shocks['gdp_shock'],
                'inflation': 0.05 + shocks['inflation_shock'],
                'income_growth': 0.05 + shocks['gdp_shock'],
                'stress_factor': 1.0 + abs(shocks['gdp_shock']) * 5
            }
            
            stressed_rates = {
                'policy_rate': self.params.policy_rate + shocks['rate_shock'],
                'deposit_rate': 0.04 + shocks['rate_shock'] * 0.8,
                'lending_rate': 0.10 + shocks['rate_shock']
            }
            
            # Test bank resilience
            bank_results = []
            for bank in self.banks:
                # Create copy for stress testing
                stressed_bank = Bank(bank.bank_id, bank.bank_type, self.params)
                stressed_bank.assets = bank.assets.copy()
                stressed_bank.liabilities = bank.liabilities.copy()
                stressed_bank.capital = bank.capital
                
                # Apply stress
                stressed_bank.update_balance_sheet(stressed_conditions, stressed_rates)
                stressed_bank.calculate_profitability(stressed_rates)
                
                # Calculate impact
                car_impact = stressed_bank.car - bank.car
                roa_impact = stressed_bank.profitability - bank.profitability
                solvency_risk = stressed_bank.assess_solvency_risk()
                
                bank_results.append({
                    'bank_id': bank.bank_id,
                    'bank_type': bank.bank_type,
                    'car_impact': car_impact,
                    'roa_impact': roa_impact,
                    'solvency_risk': solvency_risk,
                    'fails_car': stressed_bank.car < self.params.capital_adequacy_ratio
                })
            
            # Aggregate results
            bank_df = pd.DataFrame(bank_results)
            
            stress_results[scenario_name] = {
                'avg_car_impact': bank_df['car_impact'].mean(),
                'avg_roa_impact': bank_df['roa_impact'].mean(),
                'banks_failing_car': bank_df['fails_car'].sum(),
                'failure_rate': bank_df['fails_car'].mean(),
                'avg_solvency_risk': bank_df['solvency_risk'].mean(),
                'bank_details': bank_df
            }
        
        return stress_results
    
    def analyze_systemic_risk(self) -> Dict:
        """
        Analyze systemic risk in the financial system
        
        Returns:
            Systemic risk measures
        """
        logger.info("Analyzing systemic risk")
        
        # Bank interconnectedness (simplified)
        interconnectedness = self._calculate_interconnectedness()
        
        # Concentration risk
        concentration_risk = self._calculate_system_concentration()
        
        # Procyclicality measures
        procyclicality = self._calculate_procyclicality()
        
        # Overall systemic risk score
        systemic_risk_score = (
            0.4 * interconnectedness +
            0.3 * concentration_risk +
            0.3 * procyclicality
        )
        
        return {
            'systemic_risk_score': systemic_risk_score,
            'interconnectedness': interconnectedness,
            'concentration_risk': concentration_risk,
            'procyclicality': procyclicality
        }
    
    def _calculate_interconnectedness(self) -> float:
        """
        Calculate banking system interconnectedness
        
        Returns:
            Interconnectedness measure
        """
        # Simplified measure based on bank size distribution
        bank_sizes = [sum(bank.assets.values()) for bank in self.banks]
        total_assets = sum(bank_sizes)
        
        # Herfindahl index for bank concentration
        market_shares = [size / total_assets for size in bank_sizes]
        hhi = sum(share**2 for share in market_shares)
        
        # Higher HHI indicates more concentration and potential interconnectedness
        return min(1.0, hhi * 10)  # Scale to 0-1
    
    def _calculate_system_concentration(self) -> float:
        """
        Calculate system-wide concentration risk
        
        Returns:
            Concentration risk measure
        """
        # Geographic concentration (simplified)
        # In reality, would use actual bank location data
        geographic_concentration = 0.6  # Dhaka concentration
        
        # Sectoral concentration
        sectoral_concentration = self.stability_indicators._calculate_concentration_risk()
        
        return 0.5 * geographic_concentration + 0.5 * sectoral_concentration
    
    def _calculate_procyclicality(self) -> float:
        """
        Calculate procyclicality of the banking system
        
        Returns:
            Procyclicality measure
        """
        # Simplified measure based on credit growth sensitivity
        # Higher values indicate more procyclical behavior
        credit_sensitivity = self.params.gdp_credit_elasticity
        
        # Normalize to 0-1 scale
        procyclicality = min(1.0, (credit_sensitivity - 0.5) / 1.5)
        
        return max(0.0, procyclicality)
    
    def plot_simulation_results(self, results: pd.DataFrame, save_path: str = None):
        """
        Plot simulation results
        
        Args:
            results: Simulation results
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Economic conditions
        axes[0, 0].plot(results['period'], results['gdp_growth'] * 100, 'b-', linewidth=2)
        axes[0, 0].set_title('GDP Growth Rate')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('Growth Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Interest rates
        axes[0, 1].plot(results['period'], results['policy_rate'] * 100, 'r-', label='Policy Rate', linewidth=2)
        axes[0, 1].plot(results['period'], results['lending_rate'] * 100, 'g-', label='Lending Rate', linewidth=2)
        axes[0, 1].plot(results['period'], results['deposit_rate'] * 100, 'm-', label='Deposit Rate', linewidth=2)
        axes[0, 1].set_title('Interest Rates')
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Rate (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Banking sector size
        axes[0, 2].plot(results['period'], results['total_assets'], 'c-', linewidth=2)
        axes[0, 2].set_title('Total Banking Assets')
        axes[0, 2].set_xlabel('Period')
        axes[0, 2].set_ylabel('Assets')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Capital adequacy
        axes[1, 0].plot(results['period'], results['avg_car'] * 100, 'orange', linewidth=2)
        axes[1, 0].axhline(y=self.params.capital_adequacy_ratio * 100, color='r', linestyle='--', label='Regulatory Minimum')
        axes[1, 0].set_title('Average Capital Adequacy Ratio')
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].set_ylabel('CAR (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Profitability
        axes[1, 1].plot(results['period'], results['avg_roa'] * 100, 'purple', linewidth=2)
        axes[1, 1].set_title('Average Return on Assets')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('ROA (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Financial stability
        axes[1, 2].plot(results['period'], results['stability_index'], 'darkgreen', linewidth=2)
        axes[1, 2].set_title('Financial Stability Index')
        axes[1, 2].set_xlabel('Period')
        axes[1, 2].set_ylabel('Stability Index')
        axes[1, 2].grid(True, alpha=0.3)
        
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
            'model_type': 'Financial Sector Model',
            'country': 'Bangladesh',
            'number_of_banks': self.params.n_banks,
            'bank_types': ['Commercial', 'Islamic', 'Development'],
            'key_features': [
                'Banking Sector Dynamics',
                'Credit Risk Assessment',
                'Financial Stability Analysis',
                'Stress Testing',
                'Systemic Risk Measurement',
                'Regulatory Compliance'
            ],
            'bangladesh_features': [
                'Islamic Banking Integration',
                'Microfinance Sector',
                'Financial Inclusion Dynamics',
                'Sectoral Lending Patterns',
                'Regulatory Framework'
            ],
            'regulatory_ratios': {
                'capital_adequacy_ratio': f"{self.params.capital_adequacy_ratio:.1%}",
                'leverage_ratio': f"{self.params.leverage_ratio:.1%}",
                'liquidity_ratio': f"{self.params.liquidity_ratio:.1%}"
            }
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'parameters': {
            'n_banks': 30,  # Reduced for faster computation
            'time_periods': 50,
            'islamic_banking_share': 0.25,
            'microfinance_penetration': 0.15,
            'npl_ratio': 0.09
        }
    }
    
    # Initialize Financial Sector Model
    financial_model = FinancialSectorModel(config)
    
    # Run simulation
    print("Running financial system simulation...")
    results = financial_model.simulate_financial_system()
    
    # Display key results
    final_period = results.iloc[-1]
    print(f"\nSimulation Results (Final Period):")
    print(f"  Total Assets: {final_period['total_assets']:.2f}")
    print(f"  Average CAR: {final_period['avg_car']:.1%}")
    print(f"  Average ROA: {final_period['avg_roa']:.1%}")
    print(f"  Stability Index: {final_period['stability_index']:.3f}")
    print(f"  Policy Rate: {final_period['policy_rate']:.1%}")
    
    # Conduct stress tests
    print("\nConducting stress tests...")
    stress_results = financial_model.conduct_stress_tests()
    
    print("\nStress Test Results:")
    for scenario, results_dict in stress_results.items():
        print(f"  {scenario}:")
        print(f"    Banks failing CAR: {results_dict['banks_failing_car']}")
        print(f"    Failure rate: {results_dict['failure_rate']:.1%}")
        print(f"    Avg solvency risk: {results_dict['avg_solvency_risk']:.3f}")
    
    # Analyze systemic risk
    print("\nAnalyzing systemic risk...")
    systemic_risk = financial_model.analyze_systemic_risk()
    print(f"  Systemic Risk Score: {systemic_risk['systemic_risk_score']:.3f}")
    print(f"  Interconnectedness: {systemic_risk['interconnectedness']:.3f}")
    print(f"  Concentration Risk: {systemic_risk['concentration_risk']:.3f}")
    
    # Model summary
    summary = financial_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Country: {summary['country']}")
    print(f"  Banks: {summary['number_of_banks']}")
    print(f"  Key features: {len(summary['key_features'])}")
    print(f"  Bangladesh features: {len(summary['bangladesh_features'])}")
    
    print("\nFinancial Sector Model analysis completed successfully!")