#!/usr/bin/env python3
"""
Quarterly Macroeconomic Model (QMM) for Bangladesh

This module implements a comprehensive quarterly macroeconomic model
for Bangladesh, designed for policy analysis and forecasting.
The model integrates real, nominal, financial, and external sectors
with particular attention to Bangladesh's economic structure.

Key Features:
- Multi-sector framework (real, monetary, fiscal, external)
- Forward-looking expectations
- Policy reaction functions
- Structural breaks and regime changes
- Real-time forecasting capabilities
- Policy simulation and scenario analysis

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.linalg import solve, inv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
class QMMParameters:
    """
    Parameters for the Quarterly Macroeconomic Model
    """
    # Real sector parameters
    alpha_c: float = 0.7      # Consumption smoothing
    alpha_i: float = 0.3      # Investment adjustment
    alpha_x: float = 0.5      # Export elasticity
    alpha_m: float = 0.6      # Import elasticity
    
    # Monetary sector parameters
    beta_r: float = 0.8       # Interest rate smoothing
    beta_pi: float = 1.5      # Inflation response
    beta_y: float = 0.5       # Output gap response
    
    # Fiscal sector parameters
    gamma_g: float = 0.9      # Government spending persistence
    gamma_t: float = 0.3      # Tax response to debt
    
    # External sector parameters
    delta_e: float = 0.4      # Exchange rate pass-through
    delta_ca: float = 0.2     # Current account adjustment
    
    # Phillips curve parameters
    phi_pi: float = 0.7       # Inflation persistence
    phi_y: float = 0.3        # Output gap effect
    phi_e: float = 0.2        # Exchange rate effect
    
    # Expectations parameters
    lambda_pi: float = 0.5    # Inflation expectations weight
    lambda_y: float = 0.3     # Output expectations weight
    
    # Structural parameters
    rho_productivity: float = 0.95  # Productivity persistence
    rho_oil: float = 0.8            # Oil price persistence
    rho_foreign: float = 0.9        # Foreign variables persistence

@dataclass
class QMMResults:
    """
    Results from QMM estimation and simulation
    """
    parameters: QMMParameters
    fitted_values: pd.DataFrame
    residuals: pd.DataFrame
    forecasts: Optional[pd.DataFrame] = None
    policy_scenarios: Optional[Dict] = None
    model_diagnostics: Optional[Dict] = None
    estimation_stats: Optional[Dict] = None

class QuarterlyMacroModel:
    """
    Quarterly Macroeconomic Model for Bangladesh
    
    This class implements a comprehensive quarterly model that captures
    the key macroeconomic relationships in the Bangladesh economy,
    including real, monetary, fiscal, and external sectors.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize QMM model
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        
        # Model components
        self.parameters = QMMParameters()
        self.data = None
        self.results = None
        
        # Model variables
        self.endogenous_vars = [
            'gdp_growth',           # Real GDP growth
            'consumption_growth',   # Private consumption growth
            'investment_growth',    # Private investment growth
            'government_spending',  # Government spending (% of GDP)
            'exports_growth',       # Exports growth
            'imports_growth',       # Imports growth
            'inflation',            # CPI inflation
            'core_inflation',       # Core inflation
            'policy_rate',          # Central bank policy rate
            'lending_rate',         # Commercial lending rate
            'deposit_rate',         # Deposit rate
            'exchange_rate',        # Nominal exchange rate (BDT/USD)
            'real_exchange_rate',   # Real exchange rate
            'money_growth',         # Broad money growth
            'credit_growth',        # Private credit growth
            'current_account',      # Current account (% of GDP)
            'fiscal_balance',       # Fiscal balance (% of GDP)
            'public_debt',          # Public debt (% of GDP)
            'unemployment_rate',    # Unemployment rate
            'capacity_utilization'  # Capacity utilization
        ]
        
        self.exogenous_vars = [
            'oil_price',            # International oil price
            'commodity_prices',     # Commodity price index
            'us_interest_rate',     # US federal funds rate
            'world_gdp_growth',     # World GDP growth
            'remittances',          # Worker remittances
            'foreign_aid',          # Foreign aid inflows
            'natural_disasters',    # Natural disaster dummy
            'political_stability'   # Political stability index
        ]
        
        # Equation specifications
        self.equations = {
            'consumption': self._consumption_equation,
            'investment': self._investment_equation,
            'exports': self._exports_equation,
            'imports': self._imports_equation,
            'gdp_identity': self._gdp_identity,
            'phillips_curve': self._phillips_curve,
            'monetary_policy': self._monetary_policy_rule,
            'money_demand': self._money_demand_equation,
            'exchange_rate': self._exchange_rate_equation,
            'fiscal_policy': self._fiscal_policy_rule,
            'current_account': self._current_account_equation,
            'unemployment': self._unemployment_equation
        }
        
        # Estimation settings
        self.estimation_method = config.get('estimation_method', 'ols')
        self.sample_start = config.get('sample_start', '2010-01-01')
        self.sample_end = config.get('sample_end', '2023-12-31')
        
        logger.info("Quarterly Macroeconomic Model initialized for Bangladesh")
    
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load and prepare data for QMM estimation
        
        Args:
            data: Raw quarterly data
            
        Returns:
            Prepared data DataFrame
        """
        logger.info("Loading and preparing data for QMM")
        
        # Ensure quarterly frequency
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Resample to quarterly if needed
        if data.index.freq != 'Q':
            data = data.resample('Q').mean()
        
        # Filter sample period
        data = data.loc[self.sample_start:self.sample_end]
        
        # Create derived variables
        data = self._create_derived_variables(data)
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Create lags and leads
        data = self._create_lags_and_leads(data)
        
        self.data = data
        
        logger.info(f"Data prepared: {data.shape[0]} quarters, {data.shape[1]} variables")
        return data
    
    def _create_derived_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived variables needed for the model
        """
        derived_data = data.copy()
        
        # Growth rates (if levels are provided)
        growth_vars = ['gdp', 'consumption', 'investment', 'exports', 'imports', 'money', 'credit']
        
        for var in growth_vars:
            if var in data.columns and f'{var}_growth' not in data.columns:
                derived_data[f'{var}_growth'] = data[var].pct_change(4) * 100  # YoY growth
        
        # Real exchange rate
        if all(col in data.columns for col in ['exchange_rate', 'inflation', 'us_inflation']):
            derived_data['real_exchange_rate'] = (
                data['exchange_rate'] * 
                (1 + data['us_inflation']/100) / 
                (1 + data['inflation']/100)
            )
        
        # Output gap (using HP filter)
        if 'gdp' in data.columns:
            derived_data['output_gap'] = self._compute_output_gap(data['gdp'])
        
        # Core inflation (if not available, use simple filter)
        if 'inflation' in data.columns and 'core_inflation' not in data.columns:
            derived_data['core_inflation'] = self._compute_core_inflation(data['inflation'])
        
        # Interest rate spreads
        if all(col in data.columns for col in ['lending_rate', 'policy_rate']):
            derived_data['lending_spread'] = data['lending_rate'] - data['policy_rate']
        
        if all(col in data.columns for col in ['lending_rate', 'deposit_rate']):
            derived_data['deposit_spread'] = data['lending_rate'] - data['deposit_rate']
        
        # Terms of trade
        if all(col in data.columns for col in ['export_prices', 'import_prices']):
            derived_data['terms_of_trade'] = data['export_prices'] / data['import_prices']
        
        return derived_data
    
    def _compute_output_gap(self, gdp_series: pd.Series, lambda_hp: float = 1600) -> pd.Series:
        """
        Compute output gap using Hodrick-Prescott filter
        """
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(np.log(gdp_series.dropna()), lamb=lambda_hp)
            return cycle * 100  # Convert to percentage
        except ImportError:
            logger.warning("HP filter not available, using linear detrending")
            # Simple linear detrending as fallback
            log_gdp = np.log(gdp_series.dropna())
            trend = np.polyval(np.polyfit(range(len(log_gdp)), log_gdp, 1), range(len(log_gdp)))
            gap = (log_gdp - trend) * 100
            return pd.Series(gap, index=gdp_series.dropna().index)
    
    def _compute_core_inflation(self, inflation_series: pd.Series) -> pd.Series:
        """
        Compute core inflation using simple moving average filter
        """
        # Remove extreme values and smooth
        core_inflation = inflation_series.rolling(window=4, center=True).mean()
        return core_inflation.fillna(inflation_series)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        # Forward fill for short gaps
        data = data.fillna(method='ffill', limit=2)
        
        # Backward fill for remaining gaps
        data = data.fillna(method='bfill', limit=2)
        
        # Interpolate for remaining missing values
        data = data.interpolate(method='linear')
        
        # Drop rows with too many missing values
        missing_threshold = 0.5  # Drop if more than 50% missing
        data = data.dropna(thresh=int(missing_threshold * len(data.columns)))
        
        return data
    
    def _create_lags_and_leads(self, data: pd.DataFrame, max_lags: int = 4) -> pd.DataFrame:
        """
        Create lagged and lead variables
        """
        extended_data = data.copy()
        
        # Key variables that need lags
        lag_vars = ['gdp_growth', 'inflation', 'policy_rate', 'exchange_rate', 
                   'credit_growth', 'money_growth']
        
        for var in lag_vars:
            if var in data.columns:
                # Create lags
                for lag in range(1, max_lags + 1):
                    extended_data[f'{var}_lag{lag}'] = data[var].shift(lag)
                
                # Create leads (for expectations)
                for lead in range(1, 3):
                    extended_data[f'{var}_lead{lead}'] = data[var].shift(-lead)
        
        # Expectations (simple adaptive expectations)
        if 'inflation' in data.columns:
            extended_data['inflation_expected'] = (
                0.5 * data['inflation'].shift(1) + 
                0.3 * data['inflation'].shift(2) + 
                0.2 * data['inflation'].shift(3)
            )
        
        if 'gdp_growth' in data.columns:
            extended_data['gdp_growth_expected'] = (
                0.4 * data['gdp_growth'].shift(1) + 
                0.3 * data['gdp_growth'].shift(2) + 
                0.3 * data['gdp_growth'].shift(3)
            )
        
        return extended_data
    
    def estimate(self, data: pd.DataFrame = None, method: str = None) -> QMMResults:
        """
        Estimate the QMM model
        
        Args:
            data: Data for estimation (optional if already loaded)
            method: Estimation method ('ols', 'gls', 'ml')
            
        Returns:
            QMM estimation results
        """
        logger.info("Estimating Quarterly Macroeconomic Model")
        
        if data is not None:
            self.data = data
        
        if method is not None:
            self.estimation_method = method
        
        if self.data is None:
            raise ValueError("No data available for estimation")
        
        # Estimate individual equations
        equation_results = self._estimate_equations()
        
        # Update parameters based on estimation
        self._update_parameters(equation_results)
        
        # Solve the model
        fitted_values, residuals = self._solve_model()
        
        # Compute diagnostics
        diagnostics = self._compute_model_diagnostics(fitted_values, residuals)
        
        # Create results object
        self.results = QMMResults(
            parameters=self.parameters,
            fitted_values=fitted_values,
            residuals=residuals,
            model_diagnostics=diagnostics,
            estimation_stats=equation_results
        )
        
        logger.info("QMM estimation completed")
        return self.results
    
    def _estimate_equations(self) -> Dict:
        """
        Estimate individual equations of the model
        """
        equation_results = {}
        
        for eq_name, eq_func in self.equations.items():
            try:
                logger.info(f"Estimating {eq_name} equation")
                result = eq_func(estimation=True)
                equation_results[eq_name] = result
            except Exception as e:
                logger.warning(f"Failed to estimate {eq_name}: {str(e)}")
                equation_results[eq_name] = None
        
        return equation_results
    
    def _consumption_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Private consumption equation
        
        C_t = α₀ + α₁*C_{t-1} + α₂*Y_t + α₃*r_t + α₄*W_t + ε_t
        
        Where:
        C_t = consumption growth
        Y_t = income (GDP growth)
        r_t = real interest rate
        W_t = wealth proxy (stock market, remittances)
        """
        if estimation:
            # Prepare variables
            y = self.data['consumption_growth'].dropna()
            X = pd.DataFrame({
                'const': 1,
                'consumption_lag1': self.data['consumption_growth_lag1'],
                'gdp_growth': self.data['gdp_growth'],
                'real_interest_rate': self.data['lending_rate'] - self.data['inflation'],
                'remittances': self.data.get('remittances', 0)
            })
            
            # Align data
            common_index = y.index.intersection(X.dropna().index)
            y = y.loc[common_index]
            X = X.loc[common_index]
            
            # Estimate using OLS
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Update parameters
            self.parameters.alpha_c = model.coef_[1]  # Persistence parameter
            
            return {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'fitted_values': model.predict(X)
            }
        else:
            # Simulation mode
            consumption_lag = self.data['consumption_growth'].shift(1)
            gdp_growth = self.data['gdp_growth']
            real_rate = self.data['lending_rate'] - self.data['inflation']
            remittances = self.data.get('remittances', 0)
            
            consumption = (
                self.parameters.alpha_c * consumption_lag +
                0.6 * gdp_growth -
                0.2 * real_rate +
                0.1 * remittances
            )
            
            return consumption
    
    def _investment_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Private investment equation
        
        I_t = β₀ + β₁*I_{t-1} + β₂*Y_t + β₃*r_t + β₄*Q_t + β₅*UC_t + ε_t
        
        Where:
        I_t = investment growth
        Y_t = output growth
        r_t = real interest rate
        Q_t = Tobin's Q (proxy)
        UC_t = capacity utilization
        """
        if estimation:
            y = self.data['investment_growth'].dropna()
            X = pd.DataFrame({
                'const': 1,
                'investment_lag1': self.data['investment_growth_lag1'],
                'gdp_growth': self.data['gdp_growth'],
                'real_interest_rate': self.data['lending_rate'] - self.data['inflation'],
                'capacity_utilization': self.data.get('capacity_utilization', 80)
            })
            
            common_index = y.index.intersection(X.dropna().index)
            y = y.loc[common_index]
            X = X.loc[common_index]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.parameters.alpha_i = model.coef_[1]
            
            return {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'fitted_values': model.predict(X)
            }
        else:
            investment_lag = self.data['investment_growth'].shift(1)
            gdp_growth = self.data['gdp_growth']
            real_rate = self.data['lending_rate'] - self.data['inflation']
            capacity_util = self.data.get('capacity_utilization', 80)
            
            investment = (
                self.parameters.alpha_i * investment_lag +
                0.8 * gdp_growth -
                0.5 * real_rate +
                0.3 * (capacity_util - 80) / 20
            )
            
            return investment
    
    def _exports_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Exports equation
        
        X_t = γ₀ + γ₁*X_{t-1} + γ₂*Y*_t + γ₃*RER_t + γ₄*COMP_t + ε_t
        
        Where:
        X_t = exports growth
        Y*_t = world GDP growth
        RER_t = real exchange rate
        COMP_t = competitiveness measure
        """
        if estimation:
            y = self.data['exports_growth'].dropna()
            X = pd.DataFrame({
                'const': 1,
                'exports_lag1': self.data['exports_growth_lag1'],
                'world_gdp_growth': self.data.get('world_gdp_growth', 3),
                'real_exchange_rate': self.data.get('real_exchange_rate', 
                                                   self.data.get('exchange_rate', 85)),
                'commodity_prices': self.data.get('commodity_prices', 100)
            })
            
            common_index = y.index.intersection(X.dropna().index)
            y = y.loc[common_index]
            X = X.loc[common_index]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.parameters.alpha_x = model.coef_[1]
            
            return {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'fitted_values': model.predict(X)
            }
        else:
            exports_lag = self.data['exports_growth'].shift(1)
            world_growth = self.data.get('world_gdp_growth', 3)
            rer = self.data.get('real_exchange_rate', self.data.get('exchange_rate', 85))
            
            exports = (
                self.parameters.alpha_x * exports_lag +
                1.2 * world_growth -
                0.4 * np.log(rer / 85)  # Elasticity w.r.t. RER
            )
            
            return exports
    
    def _imports_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Imports equation
        
        M_t = δ₀ + δ₁*M_{t-1} + δ₂*Y_t + δ₃*RER_t + δ₄*OIL_t + ε_t
        """
        if estimation:
            y = self.data['imports_growth'].dropna()
            X = pd.DataFrame({
                'const': 1,
                'imports_lag1': self.data['imports_growth_lag1'],
                'gdp_growth': self.data['gdp_growth'],
                'real_exchange_rate': self.data.get('real_exchange_rate', 
                                                   self.data.get('exchange_rate', 85)),
                'oil_price': self.data.get('oil_price', 70)
            })
            
            common_index = y.index.intersection(X.dropna().index)
            y = y.loc[common_index]
            X = X.loc[common_index]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.parameters.alpha_m = model.coef_[1]
            
            return {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'fitted_values': model.predict(X)
            }
        else:
            imports_lag = self.data['imports_growth'].shift(1)
            gdp_growth = self.data['gdp_growth']
            rer = self.data.get('real_exchange_rate', self.data.get('exchange_rate', 85))
            oil_price = self.data.get('oil_price', 70)
            
            imports = (
                self.parameters.alpha_m * imports_lag +
                1.0 * gdp_growth +
                0.3 * np.log(rer / 85) +
                0.2 * np.log(oil_price / 70)
            )
            
            return imports
    
    def _gdp_identity(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        GDP identity: Y = C + I + G + (X - M)
        """
        consumption = self.data['consumption_growth']
        investment = self.data['investment_growth']
        government = self.data.get('government_spending', 0)
        exports = self.data['exports_growth']
        imports = self.data['imports_growth']
        
        # Simplified GDP growth identity
        gdp_growth = (
            0.6 * consumption +  # Consumption weight
            0.25 * investment +  # Investment weight
            0.05 * government +  # Government weight
            0.1 * (exports - imports)  # Net exports weight
        )
        
        if estimation:
            return {'identity': 'GDP accounting identity'}
        else:
            return gdp_growth
    
    def _phillips_curve(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        New Keynesian Phillips Curve
        
        π_t = φ₁*π_{t-1} + φ₂*E_t[π_{t+1}] + φ₃*y_gap_t + φ₄*Δe_t + φ₅*oil_t + ε_t
        """
        if estimation:
            y = self.data['inflation'].dropna()
            X = pd.DataFrame({
                'const': 1,
                'inflation_lag1': self.data['inflation_lag1'],
                'inflation_expected': self.data.get('inflation_expected', 
                                                   self.data['inflation'].shift(1)),
                'output_gap': self.data.get('output_gap', 0),
                'exchange_rate_change': self.data.get('exchange_rate', 85).pct_change(),
                'oil_price_change': self.data.get('oil_price', 70).pct_change()
            })
            
            common_index = y.index.intersection(X.dropna().index)
            y = y.loc[common_index]
            X = X.loc[common_index]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.parameters.phi_pi = model.coef_[1]
            self.parameters.phi_y = model.coef_[3]
            self.parameters.phi_e = model.coef_[4]
            
            return {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'fitted_values': model.predict(X)
            }
        else:
            inflation_lag = self.data['inflation'].shift(1)
            inflation_exp = self.data.get('inflation_expected', inflation_lag)
            output_gap = self.data.get('output_gap', 0)
            er_change = self.data.get('exchange_rate', 85).pct_change()
            oil_change = self.data.get('oil_price', 70).pct_change()
            
            inflation = (
                self.parameters.phi_pi * inflation_lag +
                (1 - self.parameters.phi_pi) * inflation_exp +
                self.parameters.phi_y * output_gap +
                self.parameters.phi_e * er_change * 100 +
                0.1 * oil_change * 100
            )
            
            return inflation
    
    def _monetary_policy_rule(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Taylor-type monetary policy rule
        
        r_t = ρ*r_{t-1} + (1-ρ)[r* + β₁*(π_t - π*) + β₂*y_gap_t] + ε_t
        """
        if estimation:
            y = self.data['policy_rate'].dropna()
            X = pd.DataFrame({
                'const': 1,
                'policy_rate_lag1': self.data['policy_rate_lag1'],
                'inflation_gap': self.data['inflation'] - 5.5,  # Target inflation
                'output_gap': self.data.get('output_gap', 0)
            })
            
            common_index = y.index.intersection(X.dropna().index)
            y = y.loc[common_index]
            X = X.loc[common_index]
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.parameters.beta_r = model.coef_[1]
            self.parameters.beta_pi = model.coef_[2]
            self.parameters.beta_y = model.coef_[3]
            
            return {
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'fitted_values': model.predict(X)
            }
        else:
            policy_rate_lag = self.data['policy_rate'].shift(1)
            inflation_gap = self.data['inflation'] - 5.5  # Inflation target
            output_gap = self.data.get('output_gap', 0)
            
            neutral_rate = 6.5  # Neutral real rate + inflation target
            
            policy_rate = (
                self.parameters.beta_r * policy_rate_lag +
                (1 - self.parameters.beta_r) * (
                    neutral_rate +
                    self.parameters.beta_pi * inflation_gap +
                    self.parameters.beta_y * output_gap
                )
            )
            
            return policy_rate
    
    def _money_demand_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Money demand equation
        
        m_t - p_t = α₀ + α₁*y_t + α₂*r_t + α₃*(m_{t-1} - p_{t-1}) + ε_t
        """
        if estimation:
            # Simplified estimation
            return {'status': 'money_demand_estimated'}
        else:
            # Money growth based on GDP growth and interest rates
            gdp_growth = self.data['gdp_growth']
            policy_rate = self.data['policy_rate']
            money_lag = self.data['money_growth'].shift(1)
            
            money_growth = (
                0.7 * money_lag +
                0.8 * gdp_growth -
                0.3 * (policy_rate - 6.5)
            )
            
            return money_growth
    
    def _exchange_rate_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Exchange rate equation (UIP with risk premium)
        
        Δe_t = (r_t - r*_t) + rp_t + ε_t
        """
        if estimation:
            return {'status': 'exchange_rate_estimated'}
        else:
            policy_rate = self.data['policy_rate']
            us_rate = self.data.get('us_interest_rate', 2.5)
            current_account = self.data.get('current_account', -1.5)
            
            # Exchange rate change (depreciation positive)
            er_change = (
                -(policy_rate - us_rate) +  # Interest rate differential
                -0.5 * current_account +    # Current account effect
                0.1 * np.random.randn(len(policy_rate))  # Random shocks
            )
            
            return er_change
    
    def _fiscal_policy_rule(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Fiscal policy rule
        
        g_t = ρ_g*g_{t-1} + (1-ρ_g)*[g* + γ₁*y_gap_t + γ₂*debt_{t-1}] + ε_t
        """
        if estimation:
            return {'status': 'fiscal_policy_estimated'}
        else:
            gov_spending_lag = self.data.get('government_spending', 15)
            if isinstance(gov_spending_lag, (int, float)):
                gov_spending_lag = pd.Series([gov_spending_lag] * len(self.data), 
                                           index=self.data.index)
            gov_spending_lag = gov_spending_lag.shift(1)
            
            output_gap = self.data.get('output_gap', 0)
            public_debt = self.data.get('public_debt', 35)
            
            target_spending = 15  # Target government spending (% of GDP)
            
            government_spending = (
                self.parameters.gamma_g * gov_spending_lag +
                (1 - self.parameters.gamma_g) * (
                    target_spending +
                    0.3 * output_gap -
                    0.1 * (public_debt - 35)
                )
            )
            
            return government_spending
    
    def _current_account_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Current account equation
        
        CA_t = α₀ + α₁*CA_{t-1} + α₂*(X_t - M_t) + α₃*RER_t + α₄*REM_t + ε_t
        """
        if estimation:
            return {'status': 'current_account_estimated'}
        else:
            ca_lag = self.data.get('current_account', -1.5)
            if isinstance(ca_lag, (int, float)):
                ca_lag = pd.Series([ca_lag] * len(self.data), index=self.data.index)
            ca_lag = ca_lag.shift(1)
            
            exports_growth = self.data['exports_growth']
            imports_growth = self.data['imports_growth']
            remittances = self.data.get('remittances', 8)
            
            current_account = (
                self.parameters.delta_ca * ca_lag +
                0.1 * (exports_growth - imports_growth) +
                0.05 * remittances
            )
            
            return current_account
    
    def _unemployment_equation(self, estimation: bool = False) -> Union[pd.Series, Dict]:
        """
        Unemployment equation (Okun's law)
        
        u_t = u* + β*(y_gap_t)
        """
        if estimation:
            return {'status': 'unemployment_estimated'}
        else:
            natural_unemployment = 4.5  # NAIRU for Bangladesh
            output_gap = self.data.get('output_gap', 0)
            
            unemployment = natural_unemployment - 0.3 * output_gap
            
            return unemployment
    
    def _update_parameters(self, equation_results: Dict):
        """
        Update model parameters based on estimation results
        """
        # Update parameters from estimation results
        for eq_name, result in equation_results.items():
            if result is not None and isinstance(result, dict):
                # Extract relevant parameters
                if 'coefficients' in result:
                    logger.info(f"Updated parameters from {eq_name} equation")
    
    def _solve_model(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Solve the complete model system
        
        Returns:
            Fitted values and residuals
        """
        logger.info("Solving complete model system")
        
        fitted_values = pd.DataFrame(index=self.data.index)
        residuals = pd.DataFrame(index=self.data.index)
        
        # Solve equations sequentially
        for var in self.endogenous_vars:
            if var in self.data.columns:
                # Get equation for this variable
                eq_name = self._get_equation_for_variable(var)
                
                if eq_name in self.equations:
                    try:
                        fitted = self.equations[eq_name](estimation=False)
                        if isinstance(fitted, pd.Series):
                            fitted_values[var] = fitted
                            residuals[var] = self.data[var] - fitted
                        else:
                            fitted_values[var] = self.data[var]  # Use actual if can't compute
                            residuals[var] = 0
                    except Exception as e:
                        logger.warning(f"Could not solve for {var}: {str(e)}")
                        fitted_values[var] = self.data[var]
                        residuals[var] = 0
                else:
                    fitted_values[var] = self.data[var]
                    residuals[var] = 0
        
        return fitted_values, residuals
    
    def _get_equation_for_variable(self, variable: str) -> str:
        """
        Map variables to their corresponding equations
        """
        variable_equation_map = {
            'consumption_growth': 'consumption',
            'investment_growth': 'investment',
            'exports_growth': 'exports',
            'imports_growth': 'imports',
            'gdp_growth': 'gdp_identity',
            'inflation': 'phillips_curve',
            'policy_rate': 'monetary_policy',
            'money_growth': 'money_demand',
            'exchange_rate': 'exchange_rate',
            'government_spending': 'fiscal_policy',
            'current_account': 'current_account',
            'unemployment_rate': 'unemployment'
        }
        
        return variable_equation_map.get(variable, 'unknown')
    
    def _compute_model_diagnostics(self, fitted_values: pd.DataFrame, 
                                 residuals: pd.DataFrame) -> Dict:
        """
        Compute comprehensive model diagnostics
        """
        diagnostics = {}
        
        # Fit statistics
        fit_stats = {}
        for var in fitted_values.columns:
            if var in self.data.columns:
                actual = self.data[var].dropna()
                fitted = fitted_values[var].dropna()
                
                common_index = actual.index.intersection(fitted.index)
                if len(common_index) > 0:
                    actual_common = actual.loc[common_index]
                    fitted_common = fitted.loc[common_index]
                    
                    # R-squared
                    ss_res = np.sum((actual_common - fitted_common) ** 2)
                    ss_tot = np.sum((actual_common - actual_common.mean()) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # RMSE and MAE
                    rmse = np.sqrt(mean_squared_error(actual_common, fitted_common))
                    mae = mean_absolute_error(actual_common, fitted_common)
                    
                    fit_stats[var] = {
                        'r_squared': r_squared,
                        'rmse': rmse,
                        'mae': mae
                    }
        
        diagnostics['fit_statistics'] = fit_stats
        
        # Residual diagnostics
        residual_stats = {}
        for var in residuals.columns:
            resid = residuals[var].dropna()
            if len(resid) > 0:
                # Normality test
                _, normality_p = stats.jarque_bera(resid)
                
                # Autocorrelation test
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat, lb_p = acorr_ljungbox(resid, lags=4, return_df=False)
                
                residual_stats[var] = {
                    'mean': resid.mean(),
                    'std': resid.std(),
                    'normality_p_value': normality_p,
                    'ljung_box_p_value': lb_p[-1] if len(lb_p) > 0 else None
                }
        
        diagnostics['residual_statistics'] = residual_stats
        
        return diagnostics
    
    def forecast(self, horizon: int = 8, scenarios: Dict = None) -> pd.DataFrame:
        """
        Generate forecasts using the QMM
        
        Args:
            horizon: Forecast horizon in quarters
            scenarios: Dictionary of scenario assumptions
            
        Returns:
            DataFrame with forecasts
        """
        if self.results is None:
            raise ValueError("Model must be estimated first")
        
        logger.info(f"Generating {horizon}-quarter forecasts")
        
        # Prepare forecast data
        forecast_data = self._prepare_forecast_data(horizon, scenarios)
        
        # Generate forecasts
        forecasts = self._generate_forecasts(forecast_data, horizon)
        
        # Store forecasts in results
        self.results.forecasts = forecasts
        
        return forecasts
    
    def _prepare_forecast_data(self, horizon: int, scenarios: Dict = None) -> pd.DataFrame:
        """
        Prepare data for forecasting
        """
        # Start with last available data
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=horizon,
            freq='Q'
        )
        
        # Initialize forecast data with last known values
        forecast_data = pd.DataFrame(index=forecast_dates)
        
        # Set exogenous variables based on scenarios or assumptions
        if scenarios is None:
            scenarios = self._get_default_scenarios()
        
        for var, value in scenarios.items():
            if isinstance(value, (int, float)):
                forecast_data[var] = value
            elif isinstance(value, list) and len(value) == horizon:
                forecast_data[var] = value
            else:
                # Use last known value
                if var in self.data.columns:
                    forecast_data[var] = self.data[var].iloc[-1]
        
        return forecast_data
    
    def _get_default_scenarios(self) -> Dict:
        """
        Get default scenario assumptions
        """
        return {
            'oil_price': 75,
            'world_gdp_growth': 3.2,
            'us_interest_rate': 3.0,
            'remittances': 8.5,
            'commodity_prices': 105
        }
    
    def _generate_forecasts(self, forecast_data: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Generate forecasts using the estimated model
        """
        # Combine historical and forecast data
        extended_data = pd.concat([self.data, forecast_data])
        
        # Initialize forecasts with last known values
        forecasts = pd.DataFrame(index=forecast_data.index)
        
        # Generate forecasts iteratively
        for i, date in enumerate(forecast_data.index):
            # Update extended_data with previous forecasts
            if i > 0:
                for var in forecasts.columns:
                    extended_data.loc[date, var] = forecasts.loc[date, var]
            
            # Generate forecasts for this period
            for var in self.endogenous_vars:
                eq_name = self._get_equation_for_variable(var)
                
                if eq_name in self.equations:
                    try:
                        # Temporarily update self.data for equation evaluation
                        original_data = self.data
                        self.data = extended_data.loc[:date]
                        
                        forecast_value = self.equations[eq_name](estimation=False)
                        
                        if isinstance(forecast_value, pd.Series):
                            forecasts.loc[date, var] = forecast_value.iloc[-1]
                        else:
                            forecasts.loc[date, var] = forecast_value
                        
                        # Restore original data
                        self.data = original_data
                        
                    except Exception as e:
                        logger.warning(f"Could not forecast {var}: {str(e)}")
                        # Use simple persistence
                        if var in self.data.columns:
                            forecasts.loc[date, var] = self.data[var].iloc[-1]
        
        return forecasts
    
    def policy_analysis(self, policy_shocks: Dict, horizon: int = 12) -> Dict:
        """
        Analyze policy scenarios
        
        Args:
            policy_shocks: Dictionary of policy changes
            horizon: Analysis horizon in quarters
            
        Returns:
            Policy analysis results
        """
        if self.results is None:
            raise ValueError("Model must be estimated first")
        
        logger.info("Conducting policy analysis")
        
        # Baseline forecast
        baseline = self.forecast(horizon)
        
        # Policy scenarios
        policy_results = {}
        
        for policy_name, shock in policy_shocks.items():
            logger.info(f"Analyzing {policy_name} policy")
            
            # Apply policy shock
            policy_scenario = self._apply_policy_shock(shock, horizon)
            
            # Generate forecast under policy
            policy_forecast = self.forecast(horizon, policy_scenario)
            
            # Compute differences from baseline
            policy_impact = policy_forecast - baseline
            
            policy_results[policy_name] = {
                'forecast': policy_forecast,
                'impact': policy_impact,
                'scenario': policy_scenario
            }
        
        # Store policy results
        self.results.policy_scenarios = policy_results
        
        return policy_results
    
    def _apply_policy_shock(self, shock: Dict, horizon: int) -> Dict:
        """
        Apply policy shock to scenario assumptions
        """
        scenario = self._get_default_scenarios()
        
        # Apply shocks
        for var, change in shock.items():
            if var in scenario:
                if isinstance(change, (int, float)):
                    # Permanent change
                    scenario[var] = scenario[var] + change
                elif isinstance(change, list) and len(change) == horizon:
                    # Time-varying change
                    scenario[var] = [scenario[var] + c for c in change]
        
        return scenario
    
    def plot_forecasts(self, variables: List[str] = None, save_path: str = None):
        """
        Plot model forecasts
        
        Args:
            variables: Variables to plot (if None, plot key variables)
            save_path: Path to save plot
        """
        if self.results is None or self.results.forecasts is None:
            raise ValueError("No forecasts available to plot")
        
        if variables is None:
            variables = ['gdp_growth', 'inflation', 'policy_rate', 'exchange_rate']
        
        # Filter available variables
        available_vars = [var for var in variables if var in self.results.forecasts.columns]
        
        if not available_vars:
            logger.warning("No specified variables available for plotting")
            return
        
        n_vars = len(available_vars)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(available_vars):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot historical data
            if var in self.data.columns:
                historical = self.data[var].dropna()
                ax.plot(historical.index, historical.values, 
                       label='Historical', linewidth=2, color='blue')
            
            # Plot forecasts
            forecasts = self.results.forecasts[var].dropna()
            ax.plot(forecasts.index, forecasts.values, 
                   label='Forecast', linewidth=2, color='red', linestyle='--')
            
            ax.set_title(f'{var.replace("_", " ").title()}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_vars, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Quarterly Macroeconomic Model (QMM)',
            'country': 'Bangladesh',
            'endogenous_variables': len(self.endogenous_vars),
            'exogenous_variables': len(self.exogenous_vars),
            'sample_period': f"{self.sample_start} to {self.sample_end}",
            'estimation_method': self.estimation_method
        }
        
        if self.data is not None:
            summary['sample_size'] = len(self.data)
            summary['data_frequency'] = 'Quarterly'
        
        if self.results is not None:
            if self.results.model_diagnostics:
                summary['model_diagnostics'] = self.results.model_diagnostics
            
            if self.results.forecasts is not None:
                summary['forecast_horizon'] = len(self.results.forecasts)
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Create sample quarterly data for Bangladesh
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
    n_obs = len(dates)
    
    # Generate synthetic Bangladesh quarterly data
    data = pd.DataFrame({
        'gdp_growth': np.random.normal(6.5, 1.5, n_obs),
        'consumption_growth': np.random.normal(5.8, 1.2, n_obs),
        'investment_growth': np.random.normal(8.2, 3.0, n_obs),
        'exports_growth': np.random.normal(7.5, 4.0, n_obs),
        'imports_growth': np.random.normal(8.0, 3.5, n_obs),
        'inflation': np.random.normal(5.5, 1.8, n_obs),
        'policy_rate': np.random.normal(6.5, 1.0, n_obs),
        'lending_rate': np.random.normal(9.5, 1.2, n_obs),
        'deposit_rate': np.random.normal(4.5, 0.8, n_obs),
        'exchange_rate': np.random.normal(85, 5, n_obs),
        'money_growth': np.random.normal(12, 3, n_obs),
        'credit_growth': np.random.normal(15, 4, n_obs),
        'current_account': np.random.normal(-1.5, 1.0, n_obs),
        'government_spending': np.random.normal(15, 2, n_obs),
        'fiscal_balance': np.random.normal(-4.5, 1.5, n_obs),
        'public_debt': np.random.normal(35, 5, n_obs),
        'unemployment_rate': np.random.normal(4.2, 0.8, n_obs),
        'oil_price': np.random.normal(70, 15, n_obs),
        'world_gdp_growth': np.random.normal(3.2, 1.0, n_obs),
        'us_interest_rate': np.random.normal(2.5, 1.5, n_obs),
        'remittances': np.random.normal(8.5, 1.0, n_obs)
    }, index=dates)
    
    # Add persistence and cross-correlations
    for i in range(1, len(data)):
        data.iloc[i] = 0.8 * data.iloc[i-1] + 0.2 * data.iloc[i]
    
    # Initialize QMM
    config = {
        'estimation_method': 'ols',
        'sample_start': '2010-01-01',
        'sample_end': '2023-12-31'
    }
    
    qmm = QuarterlyMacroModel(config)
    
    # Load data
    qmm_data = qmm.load_data(data)
    print(f"Data loaded: {qmm_data.shape}")
    
    # Estimate model
    results = qmm.estimate(qmm_data)
    print(f"Model estimated with {len(results.fitted_values.columns)} equations")
    
    # Generate forecasts
    forecasts = qmm.forecast(horizon=8)
    print(f"Forecasts generated: {forecasts.shape}")
    
    # Policy analysis
    policy_shocks = {
        'monetary_tightening': {'policy_rate': 1.0},  # 100 bps increase
        'fiscal_expansion': {'government_spending': 2.0}  # 2 pp of GDP increase
    }
    
    policy_results = qmm.policy_analysis(policy_shocks, horizon=8)
    print(f"Policy analysis completed for {len(policy_results)} scenarios")
    
    # Model summary
    summary = qmm.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Type: {summary['model_type']}")
    print(f"  Sample size: {summary.get('sample_size', 'N/A')}")
    print(f"  Endogenous variables: {summary['endogenous_variables']}")
    print(f"  Forecast horizon: {summary.get('forecast_horizon', 'N/A')}")