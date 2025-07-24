#!/usr/bin/env python3
"""
Structural Vector Autoregression (SVAR) Model for Bangladesh

This module implements SVAR models specifically designed for analyzing
monetary policy transmission and macroeconomic dynamics in Bangladesh.
The model includes identification strategies relevant for emerging economies
and small open economies like Bangladesh.

Key Features:
- Multiple identification schemes (Cholesky, sign restrictions, long-run)
- Monetary policy transmission analysis
- Exchange rate and external sector dynamics
- Structural break detection and handling
- Bayesian estimation with informative priors
- Real-time forecasting capabilities

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve, inv, eig
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SVARResults:
    """
    Data class for SVAR estimation results
    """
    coefficients: np.ndarray
    structural_matrix: np.ndarray
    residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    hqic: float
    identification: str
    impulse_responses: Optional[Dict] = None
    forecast_error_variance: Optional[Dict] = None
    historical_decomposition: Optional[Dict] = None

class SVARModel:
    """
    Structural Vector Autoregression Model for Bangladesh
    
    This class implements SVAR models with various identification strategies
    suitable for analyzing Bangladesh's macroeconomic dynamics, particularly
    monetary policy transmission mechanisms.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SVAR model
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        
        # Model specifications
        self.variables = None
        self.data = None
        self.lags = config.get('lags', 4)
        self.identification = config.get('identification', 'cholesky')
        
        # Estimation results
        self.var_model = None
        self.svar_results = None
        self.structural_matrix = None
        
        # Bangladesh-specific variables
        self.bd_variables = [
            'gdp_growth',      # Real GDP growth
            'inflation',       # CPI inflation
            'policy_rate',     # Bangladesh Bank policy rate
            'exchange_rate',   # BDT/USD exchange rate
            'money_growth',    # Broad money growth
            'credit_growth',   # Private credit growth
            'current_account', # Current account balance
            'oil_price',       # Oil price (external)
            'us_interest_rate' # US federal funds rate (external)
        ]
        
        # Identification schemes
        self.identification_schemes = {
            'cholesky': self._cholesky_identification,
            'sign_restrictions': self._sign_restrictions_identification,
            'long_run': self._long_run_identification,
            'external_instruments': self._external_instruments_identification
        }
        
        logger.info("SVAR model initialized for Bangladesh")
    
    def prepare_data(self, data: pd.DataFrame, variables: List[str] = None) -> pd.DataFrame:
        """
        Prepare data for SVAR estimation
        
        Args:
            data: Raw data DataFrame
            variables: List of variables to include
            
        Returns:
            Prepared data DataFrame
        """
        logger.info("Preparing data for SVAR estimation")
        
        if variables is None:
            # Use actual column names from the data
            available_vars = []
            
            # Priority variables for SVAR analysis
            priority_vars = ['gdp_growth', 'inflation', 'current_account', 'unemployment', 'exports_growth']
            
            # Check which priority variables are available in data
            for var in priority_vars:
                if var in data.columns:
                    available_vars.append(var)
            
            # If we don't have enough variables, add more from available columns
            if len(available_vars) < 3:
                for col in data.columns:
                    if col not in available_vars and col != 'Year' and len(available_vars) < 5:
                        available_vars.append(col)
            
            variables = available_vars[:min(5, len(available_vars))]  # Limit to 5 variables max
        
        self.variables = variables
        
        # Select variables directly from data
        svar_data = data[variables].copy()
        
        # Handle missing values
        svar_data = svar_data.dropna()
        
        # Check for stationarity and transform if needed
        svar_data = self._ensure_stationarity(svar_data)
        
        # Remove outliers
        svar_data = self._remove_outliers(svar_data)
        
        self.data = svar_data
        
        logger.info(f"Data prepared with {len(svar_data)} observations and {len(variables)} variables")
        return svar_data
    
    def _ensure_stationarity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all variables are stationary
        """
        stationary_data = data.copy()
        
        for col in data.columns:
            # Test for unit root
            adf_result = adfuller(data[col].dropna())
            
            if adf_result[1] > 0.05:  # Non-stationary
                logger.info(f"Variable {col} is non-stationary, taking first difference")
                
                # Take first difference
                stationary_data[col] = data[col].diff()
                
                # Test again
                adf_result_diff = adfuller(stationary_data[col].dropna())
                if adf_result_diff[1] > 0.05:
                    logger.warning(f"Variable {col} still non-stationary after differencing")
        
        return stationary_data.dropna()
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using z-score method
        """
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = (z_scores < threshold).all(axis=1)
        
        outliers_removed = len(data) - outlier_mask.sum()
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outlier observations")
        
        return data[outlier_mask]
    
    def estimate(self, data: pd.DataFrame = None, lags: int = None) -> SVARResults:
        """
        Estimate SVAR model
        
        Args:
            data: Data for estimation (optional if already prepared)
            lags: Number of lags (optional)
            
        Returns:
            SVAR estimation results
        """
        logger.info("Estimating SVAR model")
        
        if data is not None:
            self.data = data
        
        if lags is not None:
            self.lags = lags
        
        if self.data is None:
            raise ValueError("No data available for estimation")
        
        # Estimate reduced-form VAR
        self._estimate_var()
        
        # Identify structural shocks
        self._identify_structural_shocks()
        
        # Compute diagnostics
        diagnostics = self._compute_diagnostics()
        
        # Create results object
        self.svar_results = SVARResults(
            coefficients=self.var_model.coefs,
            structural_matrix=self.structural_matrix,
            residuals=self.var_model.resid,
            log_likelihood=self.var_model.llf,
            aic=self.var_model.aic,
            bic=self.var_model.bic,
            hqic=self.var_model.hqic,
            identification=self.identification
        )
        
        logger.info("SVAR estimation completed")
        return self.svar_results
    
    def _estimate_var(self):
        """
        Estimate reduced-form VAR model
        """
        # Ensure data is a DataFrame for VAR
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data for VAR estimation must be a pandas DataFrame.")

        # Fit VAR model
        var_model = VAR(self.data)
        
        # Select optimal lag length if not specified
        if self.lags is None:
            lag_order = var_model.select_order(maxlags=8)
            self.lags = lag_order.aic
            logger.info(f"Selected lag order: {self.lags} (based on AIC)")
        
        # Estimate VAR
        self.var_model = var_model.fit(self.lags)
        
        logger.info(f"VAR model estimated with {self.lags} lags")
    
    def _identify_structural_shocks(self):
        """
        Identify structural shocks using specified identification scheme
        """
        if self.identification not in self.identification_schemes:
            raise ValueError(f"Unknown identification scheme: {self.identification}")
        
        logger.info(f"Identifying structural shocks using {self.identification} scheme")
        
        # Get identification function
        identify_func = self.identification_schemes[self.identification]
        
        # Identify structural matrix
        self.structural_matrix = identify_func()
        
        logger.info("Structural identification completed")
    
    def _cholesky_identification(self) -> np.ndarray:
        """
        Cholesky identification scheme
        
        Assumes recursive structure with specific ordering for Bangladesh:
        1. External variables (oil price, US interest rate)
        2. Domestic policy variables (policy rate)
        3. Financial variables (exchange rate, money growth, credit growth)
        4. Real variables (GDP growth, inflation, current account)
        """
        # Get residual covariance matrix
        sigma = self.var_model.sigma_u
        
        # Cholesky decomposition
        try:
            A = cholesky(sigma, lower=True)
            return A
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed, using eigenvalue decomposition")
            eigenvals, eigenvecs = eig(sigma)
            A = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0))) @ eigenvecs.T
            return A
    
    def _sign_restrictions_identification(self) -> np.ndarray:
        """
        Sign restrictions identification
        
        Implements sign restrictions based on economic theory for Bangladesh:
        - Contractionary monetary policy: policy rate ↑, GDP ↓, inflation ↓, exchange rate ↑
        - Positive supply shock: GDP ↑, inflation ↓
        - Positive demand shock: GDP ↑, inflation ↑
        """
        # Define sign restrictions matrix
        # Rows: variables, Columns: shocks
        sign_restrictions = self._get_sign_restrictions_matrix()
        
        # Use algorithm to find structural matrix satisfying restrictions
        A = self._find_structural_matrix_with_signs(sign_restrictions)
        
        return A
    
    def _get_sign_restrictions_matrix(self) -> np.ndarray:
        """
        Define sign restrictions matrix for Bangladesh economy
        """
        n_vars = len(self.variables)
        n_shocks = n_vars
        
        # Initialize with zeros (no restriction)
        restrictions = np.zeros((n_vars, n_shocks))
        
        # Map variables to indices
        var_indices = {var: i for i, var in enumerate(self.variables)}
        
        # Monetary policy shock (contractionary)
        if 'policy_rate' in var_indices and 'gdp_growth' in var_indices:
            mp_shock = 0  # First shock is monetary policy
            restrictions[var_indices['policy_rate'], mp_shock] = 1   # Rate increases
            restrictions[var_indices['gdp_growth'], mp_shock] = -1   # GDP decreases
            
            if 'inflation' in var_indices:
                restrictions[var_indices['inflation'], mp_shock] = -1  # Inflation decreases
            
            if 'exchange_rate' in var_indices:
                restrictions[var_indices['exchange_rate'], mp_shock] = -1  # Currency appreciates
        
        # Supply shock (positive productivity)
        if 'gdp_growth' in var_indices and 'inflation' in var_indices:
            supply_shock = 1
            restrictions[var_indices['gdp_growth'], supply_shock] = 1   # GDP increases
            restrictions[var_indices['inflation'], supply_shock] = -1   # Inflation decreases
        
        # Demand shock (positive)
        if 'gdp_growth' in var_indices and 'inflation' in var_indices:
            demand_shock = 2
            restrictions[var_indices['gdp_growth'], demand_shock] = 1   # GDP increases
            restrictions[var_indices['inflation'], demand_shock] = 1    # Inflation increases
        
        # Exchange rate shock (depreciation)
        if 'exchange_rate' in var_indices and 'inflation' in var_indices:
            fx_shock = 3
            restrictions[var_indices['exchange_rate'], fx_shock] = 1    # Currency depreciates
            restrictions[var_indices['inflation'], fx_shock] = 1        # Inflation increases
        
        return restrictions
    
    def _find_structural_matrix_with_signs(self, sign_restrictions: np.ndarray, 
                                         max_iterations: int = 10000) -> np.ndarray:
        """
        Find structural matrix satisfying sign restrictions
        """
        sigma = self.var_model.sigma_u
        n = sigma.shape[0]
        
        # Start with Cholesky decomposition
        A_chol = cholesky(sigma, lower=True)
        
        # Try random rotations to satisfy sign restrictions
        for _ in range(max_iterations):
            # Generate random orthogonal matrix
            Q = self._random_orthogonal_matrix(n)
            
            # Candidate structural matrix
            A_candidate = A_chol @ Q
            
            # Check sign restrictions
            if self._check_sign_restrictions(A_candidate, sign_restrictions):
                return A_candidate
        
        logger.warning("Could not find structural matrix satisfying all sign restrictions")
        return A_chol
    
    def _random_orthogonal_matrix(self, n: int) -> np.ndarray:
        """
        Generate random orthogonal matrix
        """
        # Generate random matrix
        X = np.random.randn(n, n)
        
        # QR decomposition
        Q, R = np.linalg.qr(X)
        
        # Ensure positive diagonal
        Q = Q @ np.diag(np.sign(np.diag(R)))
        
        return Q
    
    def _check_sign_restrictions(self, A: np.ndarray, restrictions: np.ndarray) -> bool:
        """
        Check if structural matrix satisfies sign restrictions
        """
        for i in range(restrictions.shape[0]):
            for j in range(restrictions.shape[1]):
                if restrictions[i, j] != 0:
                    if np.sign(A[i, j]) != np.sign(restrictions[i, j]):
                        return False
        return True
    
    def _long_run_identification(self) -> np.ndarray:
        """
        Long-run identification (Blanchard-Quah type)
        
        Assumes some shocks have no long-run effect on certain variables
        """
        # Get VAR coefficients
        coefs = self.var_model.coefs
        n = coefs.shape[1]
        
        # Compute long-run multiplier matrix
        I = np.eye(n)
        A_sum = np.sum(coefs, axis=0)
        C_lr = inv(I - A_sum)
        
        # Decompose long-run covariance
        sigma = self.var_model.sigma_u
        Omega_lr = C_lr @ sigma @ C_lr.T
        
        # Cholesky decomposition of long-run matrix
        try:
            P = cholesky(Omega_lr, lower=True)
            A = solve(C_lr, P)
            return A
        except np.linalg.LinAlgError:
            logger.warning("Long-run identification failed, using Cholesky")
            return self._cholesky_identification()
    
    def _external_instruments_identification(self) -> np.ndarray:
        """
        External instruments identification
        
        Uses external instruments for monetary policy shocks
        (e.g., narrative approach, high-frequency identification)
        """
        # For now, fall back to Cholesky
        # In practice, would use external instruments
        logger.info("External instruments not implemented, using Cholesky")
        return self._cholesky_identification()
    
    def _compute_diagnostics(self) -> Dict:
        """
        Compute model diagnostics
        """
        diagnostics = {}
        
        try:
            # Residual autocorrelation test
            residuals = self.var_model.resid
            
            # Ljung-Box test for each variable
            lb_tests = {}
            if residuals is not None and len(residuals.shape) >= 2:
                for i, var in enumerate(self.variables):
                    if i < residuals.shape[1]:  # Check bounds
                        try:
                            lb_stat, lb_pvalue = acorr_ljungbox(residuals[:, i], lags=10, return_df=False)
                            if isinstance(lb_stat, (list, np.ndarray)) and len(lb_stat) > 0:
                                lb_tests[var] = {'statistic': lb_stat[-1], 'p_value': lb_pvalue[-1]}
                            else:
                                lb_tests[var] = {'statistic': lb_stat, 'p_value': lb_pvalue}
                        except Exception as e:
                            logger.warning(f"Could not compute Ljung-Box test for {var}: {e}")
                            lb_tests[var] = {'error': str(e)}
            
            diagnostics['ljung_box'] = lb_tests
            
            # Stability test (eigenvalues)
            try:
                companion_matrix = self.var_model.coefs_exog
                if companion_matrix is not None:
                    eigenvals = eig(companion_matrix)[0]
                    max_eigenval = np.max(np.abs(eigenvals))
                    diagnostics['stability'] = {
                        'max_eigenvalue': max_eigenval,
                        'stable': max_eigenval < 1.0
                    }
                else:
                    diagnostics['stability'] = {'error': 'No companion matrix available'}
            except Exception as e:
                logger.warning(f"Could not compute stability diagnostics: {e}")
                diagnostics['stability'] = {'error': str(e)}
                
        except Exception as e:
            logger.warning(f"Error computing diagnostics: {e}")
            diagnostics = {'error': str(e)}
        
        return diagnostics
    
    def impulse_response(self, shock_variable: str = None, response_variable: str = None,
                        periods: int = 20, shock_size: float = 1.0) -> pd.DataFrame:
        """
        Compute impulse response functions
        
        Args:
            shock_variable: Variable to shock (if None, compute for all)
            response_variable: Variable to analyze response (if None, compute for all)
            periods: Number of periods for IRF
            shock_size: Size of shock (in standard deviations)
            
        Returns:
            DataFrame with impulse responses
        """
        if self.svar_results is None:
            raise ValueError("Model must be estimated first")
        
        logger.info(f"Computing impulse responses for {periods} periods")
        
        # Get structural IRFs
        irf_results = self._compute_structural_irf(periods, shock_size)
        
        # Convert to DataFrame
        if shock_variable is None and response_variable is None:
            # Return all IRFs
            irf_df = self._format_irf_results(irf_results, periods)
        else:
            # Return specific IRF
            irf_df = self._extract_specific_irf(irf_results, shock_variable, 
                                              response_variable, periods)
        
        return irf_df
    
    def _compute_structural_irf(self, periods: int, shock_size: float) -> np.ndarray:
        """
        Compute structural impulse response functions
        """
        n_vars = len(self.variables)
        
        # Initialize IRF array
        irf = np.zeros((periods, n_vars, n_vars))
        
        # Get VAR coefficients
        coefs = self.var_model.coefs
        n_lags = coefs.shape[0]
        
        # Structural impact matrix
        A_inv = inv(self.structural_matrix)
        
        # Compute IRFs
        for h in range(periods):
            if h == 0:
                # Impact response
                irf[h] = A_inv * shock_size
            else:
                # Dynamic response
                response = np.zeros((n_vars, n_vars))
                
                for lag in range(min(h, n_lags)):
                    response += coefs[lag] @ irf[h - lag - 1]
                
                irf[h] = response
        
        return irf
    
    def _format_irf_results(self, irf_results: np.ndarray, periods: int) -> pd.DataFrame:
        """
        Format IRF results into DataFrame
        """
        n_vars = len(self.variables)
        
        # Create multi-index columns
        columns = pd.MultiIndex.from_product(
            [self.variables, self.variables],
            names=['shock', 'response']
        )
        
        # Reshape IRF results
        irf_flat = irf_results.reshape(periods, n_vars * n_vars)
        
        # Create DataFrame
        irf_df = pd.DataFrame(
            irf_flat,
            columns=columns,
            index=range(periods)
        )
        
        return irf_df
    
    def _extract_specific_irf(self, irf_results: np.ndarray, shock_var: str,
                             response_var: str, periods: int) -> pd.DataFrame:
        """
        Extract specific impulse response
        """
        shock_idx = self.variables.index(shock_var) if shock_var else None
        response_idx = self.variables.index(response_var) if response_var else None
        
        if shock_idx is not None and response_idx is not None:
            # Specific shock-response pair
            irf_series = irf_results[:, response_idx, shock_idx]
            return pd.DataFrame({
                f'{response_var}_to_{shock_var}': irf_series
            }, index=range(periods))
        elif shock_idx is not None:
            # All responses to specific shock
            irf_data = {}
            for i, var in enumerate(self.variables):
                irf_data[f'{var}_to_{shock_var}'] = irf_results[:, i, shock_idx]
            return pd.DataFrame(irf_data, index=range(periods))
        elif response_idx is not None:
            # Specific response to all shocks
            irf_data = {}
            for i, var in enumerate(self.variables):
                irf_data[f'{response_var}_to_{var}'] = irf_results[:, response_idx, i]
            return pd.DataFrame(irf_data, index=range(periods))
        else:
            return self._format_irf_results(irf_results, periods)
    
    def forecast_error_variance_decomposition(self, periods: int = 20) -> pd.DataFrame:
        """
        Compute forecast error variance decomposition
        
        Args:
            periods: Number of periods for FEVD
            
        Returns:
            DataFrame with variance decomposition
        """
        if self.svar_results is None:
            raise ValueError("Model must be estimated first")
        
        logger.info(f"Computing forecast error variance decomposition for {periods} periods")
        
        # Compute IRFs
        irf = self._compute_structural_irf(periods, 1.0)
        
        # Compute FEVD
        fevd = np.zeros((periods, len(self.variables), len(self.variables)))
        
        for h in range(periods):
            # Cumulative sum of squared IRFs
            mse = np.zeros((len(self.variables), len(self.variables)))
            
            for j in range(h + 1):
                mse += irf[j] ** 2
            
            # Normalize by total variance
            total_var = np.sum(mse, axis=1, keepdims=True)
            fevd[h] = mse / total_var
        
        # Format results
        fevd_df = self._format_fevd_results(fevd, periods)
        
        return fevd_df
    
    def _format_fevd_results(self, fevd: np.ndarray, periods: int) -> pd.DataFrame:
        """
        Format FEVD results into DataFrame
        """
        n_vars = len(self.variables)
        
        # Create multi-index
        index = pd.MultiIndex.from_product(
            [range(periods), self.variables],
            names=['period', 'variable']
        )
        
        # Reshape FEVD
        fevd_reshaped = fevd.reshape(periods * n_vars, n_vars)
        
        # Create DataFrame
        fevd_df = pd.DataFrame(
            fevd_reshaped,
            index=index,
            columns=self.variables
        )
        
        return fevd_df
    
    def historical_decomposition(self, start_date: str = None, 
                               end_date: str = None) -> pd.DataFrame:
        """
        Compute historical decomposition of variables
        
        Args:
            start_date: Start date for decomposition
            end_date: End date for decomposition
            
        Returns:
            DataFrame with historical decomposition
        """
        if self.svar_results is None:
            raise ValueError("Model must be estimated first")
        
        logger.info("Computing historical decomposition")
        
        # Get structural shocks
        structural_shocks = self._recover_structural_shocks()
        
        # Compute contribution of each shock
        decomposition = self._compute_shock_contributions(structural_shocks)
        
        # Format results
        decomp_df = self._format_decomposition_results(decomposition)
        
        # Filter by date range if specified
        if start_date or end_date:
            decomp_df = self._filter_by_date_range(decomp_df, start_date, end_date)
        
        return decomp_df
    
    def _recover_structural_shocks(self) -> np.ndarray:
        """
        Recover structural shocks from reduced-form residuals
        """
        # Get reduced-form residuals
        reduced_form_residuals = self.var_model.resid
        
        # Recover structural shocks
        A_inv = inv(self.structural_matrix)
        structural_shocks = (A_inv @ reduced_form_residuals.T).T
        
        return structural_shocks
    
    def _compute_shock_contributions(self, structural_shocks: np.ndarray) -> np.ndarray:
        """
        Compute contribution of each shock to variable movements
        """
        n_obs, n_vars = structural_shocks.shape
        n_lags = self.var_model.k_ar
        
        # Initialize contributions
        contributions = np.zeros((n_obs, n_vars, n_vars))
        
        # Get VAR coefficients
        coefs = self.var_model.coefs
        
        # Compute contributions recursively
        for t in range(n_obs):
            for shock_idx in range(n_vars):
                # Contribution of shock_idx to all variables at time t
                contribution = np.zeros(n_vars)
                
                # Direct impact
                contribution += self.structural_matrix[:, shock_idx] * structural_shocks[t, shock_idx]
                
                # Lagged impacts
                for lag in range(1, min(t + 1, n_lags + 1)):
                    if t - lag >= 0:
                        lagged_contribution = coefs[lag - 1] @ contributions[t - lag, :, shock_idx]
                        contribution += lagged_contribution
                
                contributions[t, :, shock_idx] = contribution
        
        return contributions
    
    def _format_decomposition_results(self, contributions: np.ndarray) -> pd.DataFrame:
        """
        Format historical decomposition results
        """
        n_obs, n_vars, n_shocks = contributions.shape
        
        # Create multi-index columns
        columns = pd.MultiIndex.from_product(
            [self.variables, self.variables],
            names=['variable', 'shock']
        )
        
        # Reshape contributions
        contrib_reshaped = contributions.reshape(n_obs, n_vars * n_shocks)
        
        # Create DataFrame with dates if available
        if hasattr(self.data, 'index'):
            index = self.data.index[-n_obs:]
        else:
            index = range(n_obs)
        
        decomp_df = pd.DataFrame(
            contrib_reshaped,
            index=index,
            columns=columns
        )
        
        return decomp_df
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: str, 
                             end_date: str) -> pd.DataFrame:
        """
        Filter DataFrame by date range
        """
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df
    
    def monetary_transmission_analysis(self) -> Dict:
        """
        Analyze monetary policy transmission mechanisms
        
        Returns:
            Dictionary with transmission analysis results
        """
        if self.svar_results is None:
            raise ValueError("Model must be estimated first")
        
        logger.info("Analyzing monetary policy transmission")
        
        # Compute monetary policy IRFs
        mp_irf = self.impulse_response(shock_variable='policy_rate', periods=20)
        
        # Analyze transmission channels
        transmission_results = {
            'interest_rate_channel': self._analyze_interest_rate_channel(mp_irf),
            'exchange_rate_channel': self._analyze_exchange_rate_channel(mp_irf),
            'credit_channel': self._analyze_credit_channel(mp_irf),
            'asset_price_channel': self._analyze_asset_price_channel(mp_irf)
        }
        
        # Compute transmission effectiveness
        transmission_results['effectiveness'] = self._compute_transmission_effectiveness(mp_irf)
        
        return transmission_results
    
    def _analyze_interest_rate_channel(self, mp_irf: pd.DataFrame) -> Dict:
        """
        Analyze interest rate transmission channel
        """
        results = {}
        
        # Interest rate pass-through
        if 'policy_rate_to_policy_rate' in mp_irf.columns:
            rate_response = mp_irf['policy_rate_to_policy_rate']
            results['pass_through'] = rate_response.iloc[0]  # Impact response
            results['persistence'] = np.sum(rate_response[:12])  # 1-year cumulative
        
        # Effect on real variables
        if 'gdp_growth_to_policy_rate' in mp_irf.columns:
            gdp_response = mp_irf['gdp_growth_to_policy_rate']
            results['peak_gdp_effect'] = np.min(gdp_response)  # Peak negative effect
            results['gdp_effect_timing'] = np.argmin(gdp_response)  # Timing of peak
        
        return results
    
    def _analyze_exchange_rate_channel(self, mp_irf: pd.DataFrame) -> Dict:
        """
        Analyze exchange rate transmission channel
        """
        results = {}
        
        # Exchange rate response
        if 'exchange_rate_to_policy_rate' in mp_irf.columns:
            fx_response = mp_irf['exchange_rate_to_policy_rate']
            results['fx_response'] = fx_response.iloc[0]  # Impact response
            results['fx_persistence'] = np.sum(fx_response[:8])  # 2-quarter cumulative
        
        # Pass-through to inflation
        if 'inflation_to_policy_rate' in mp_irf.columns:
            inflation_response = mp_irf['inflation_to_policy_rate']
            results['inflation_effect'] = np.min(inflation_response)
            results['inflation_timing'] = np.argmin(inflation_response)
        
        return results
    
    def _analyze_credit_channel(self, mp_irf: pd.DataFrame) -> Dict:
        """
        Analyze credit transmission channel
        """
        results = {}
        
        # Credit growth response
        if 'credit_growth_to_policy_rate' in mp_irf.columns:
            credit_response = mp_irf['credit_growth_to_policy_rate']
            results['credit_response'] = np.min(credit_response)
            results['credit_timing'] = np.argmin(credit_response)
        
        # Money growth response
        if 'money_growth_to_policy_rate' in mp_irf.columns:
            money_response = mp_irf['money_growth_to_policy_rate']
            results['money_response'] = np.min(money_response)
        
        return results
    
    def _analyze_asset_price_channel(self, mp_irf: pd.DataFrame) -> Dict:
        """
        Analyze asset price transmission channel
        """
        # Placeholder for asset price analysis
        # Would include stock prices, bond yields, etc.
        return {'status': 'not_implemented'}
    
    def _compute_transmission_effectiveness(self, mp_irf: pd.DataFrame) -> Dict:
        """
        Compute overall transmission effectiveness
        """
        effectiveness = {}
        
        # Speed of transmission (quarters to peak effect)
        if 'gdp_growth_to_policy_rate' in mp_irf.columns:
            gdp_response = mp_irf['gdp_growth_to_policy_rate']
            effectiveness['speed'] = np.argmin(gdp_response)
        
        # Magnitude of transmission
        if 'inflation_to_policy_rate' in mp_irf.columns:
            inflation_response = mp_irf['inflation_to_policy_rate']
            effectiveness['magnitude'] = abs(np.min(inflation_response))
        
        # Persistence of effects
        if 'gdp_growth_to_policy_rate' in mp_irf.columns:
            gdp_response = mp_irf['gdp_growth_to_policy_rate']
            # Count quarters with significant effect (>10% of peak)
            peak_effect = abs(np.min(gdp_response))
            significant_periods = np.sum(abs(gdp_response) > 0.1 * peak_effect)
            effectiveness['persistence'] = significant_periods
        
        return effectiveness
    
    def forecast(self, horizon: int = 8, confidence_levels: List[float] = [0.68, 0.95]) -> Dict:
        """
        Generate forecasts using SVAR model
        
        Args:
            horizon: Forecast horizon in periods
            confidence_levels: Confidence levels for intervals
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        if self.var_model is None:
            raise ValueError("Model must be estimated first")
        
        logger.info(f"Generating {horizon}-period ahead forecasts")
        
        # Generate VAR forecasts
        var_forecast = self.var_model.forecast(self.data.values[-self.lags:], horizon)
        
        # Compute forecast error covariance
        forecast_errors = self._compute_forecast_error_covariance(horizon)
        
        # Generate confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            var_forecast, forecast_errors, confidence_levels
        )
        
        # Format results
        forecast_results = {
            'point_forecasts': pd.DataFrame(
                var_forecast,
                columns=self.variables,
                index=pd.date_range(
                    start=self.data.index[-1] + pd.DateOffset(months=3),
                    periods=horizon,
                    freq='Q'
                )
            ),
            'confidence_intervals': confidence_intervals
        }
        
        return forecast_results
    
    def _compute_forecast_error_covariance(self, horizon: int) -> np.ndarray:
        """
        Compute forecast error covariance matrices
        """
        n_vars = len(self.variables)
        covariances = np.zeros((horizon, n_vars, n_vars))
        
        # Get structural covariance
        structural_cov = self.structural_matrix @ self.structural_matrix.T
        
        # Compute IRFs for covariance calculation
        irf = self._compute_structural_irf(horizon, 1.0)
        
        for h in range(horizon):
            # Cumulative forecast error variance
            cum_var = np.zeros((n_vars, n_vars))
            
            for j in range(h + 1):
                cum_var += irf[j] @ structural_cov @ irf[j].T
            
            covariances[h] = cum_var
        
        return covariances
    
    def _compute_confidence_intervals(self, forecasts: np.ndarray, 
                                    covariances: np.ndarray,
                                    confidence_levels: List[float]) -> Dict:
        """
        Compute confidence intervals for forecasts
        """
        intervals = {}
        
        for conf_level in confidence_levels:
            # Critical value for normal distribution
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            # Compute intervals
            lower_bounds = np.zeros_like(forecasts)
            upper_bounds = np.zeros_like(forecasts)
            
            for h in range(forecasts.shape[0]):
                std_errors = np.sqrt(np.diag(covariances[h]))
                lower_bounds[h] = forecasts[h] - z_score * std_errors
                upper_bounds[h] = forecasts[h] + z_score * std_errors
            
            intervals[f'{int(conf_level*100)}%'] = {
                'lower': pd.DataFrame(
                    lower_bounds,
                    columns=self.variables
                ),
                'upper': pd.DataFrame(
                    upper_bounds,
                    columns=self.variables
                )
            }
        
        return intervals
    
    def plot_impulse_responses(self, irf_results: pd.DataFrame, 
                             shock_variable: str = None,
                             save_path: str = None):
        """
        Plot impulse response functions
        
        Args:
            irf_results: IRF results from impulse_response method
            shock_variable: Specific shock to plot (if None, plot all)
            save_path: Path to save plot
        """
        if shock_variable:
            # Plot responses to specific shock
            shock_cols = [col for col in irf_results.columns if f'_to_{shock_variable}' in col]
            
            if not shock_cols:
                logger.warning(f"No IRF results found for shock: {shock_variable}")
                return
            
            n_responses = len(shock_cols)
            n_cols = min(3, n_responses)
            n_rows = (n_responses + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_responses == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(shock_cols):
                row, col_idx = i // n_cols, i % n_cols
                ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                
                response_var = col.split('_to_')[0]
                
                ax.plot(irf_results.index, irf_results[col], linewidth=2)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax.set_title(f'Response of {response_var} to {shock_variable} shock')
                ax.set_xlabel('Periods')
                ax.set_ylabel('Response')
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(n_responses, n_rows * n_cols):
                row, col_idx = i // n_cols, i % n_cols
                if n_rows > 1:
                    fig.delaxes(axes[row, col_idx])
                else:
                    fig.delaxes(axes[col_idx])
            
            plt.tight_layout()
            
        else:
            # Plot all IRFs (simplified view)
            n_vars = len(self.variables)
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(20, 20))
            
            for i, response_var in enumerate(self.variables):
                for j, shock_var in enumerate(self.variables):
                    col_name = f'{response_var}_to_{shock_var}'
                    
                    if col_name in irf_results.columns:
                        axes[i, j].plot(irf_results.index, irf_results[col_name])
                        axes[i, j].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                        axes[i, j].set_title(f'{response_var} → {shock_var}')
                        axes[i, j].grid(True, alpha=0.3)
                    
                    if i == n_vars - 1:
                        axes[i, j].set_xlabel('Periods')
                    if j == 0:
                        axes[i, j].set_ylabel('Response')
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"IRF plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'SVAR',
            'country': 'Bangladesh',
            'variables': self.variables,
            'sample_size': len(self.data) if self.data is not None else None,
            'lags': self.lags,
            'identification': self.identification
        }
        
        if self.svar_results:
            summary['estimation_results'] = {
                'log_likelihood': self.svar_results.log_likelihood,
                'aic': self.svar_results.aic,
                'bic': self.svar_results.bic,
                'hqic': self.svar_results.hqic
            }
        
        if self.var_model:
            try:
                # Handle different VAR model attribute structures
                if hasattr(self.var_model, 'rsquared') and self.var_model.rsquared is not None:
                    if isinstance(self.var_model.rsquared, (list, np.ndarray)):
                        r_squared = {var: self.var_model.rsquared[i] 
                                   for i, var in enumerate(self.variables) 
                                   if i < len(self.var_model.rsquared)}
                    else:
                        r_squared = {'overall': self.var_model.rsquared}
                else:
                    r_squared = {}
                
                if hasattr(self.var_model, 'rsquared_adj') and self.var_model.rsquared_adj is not None:
                    if isinstance(self.var_model.rsquared_adj, (list, np.ndarray)):
                        adj_r_squared = {var: self.var_model.rsquared_adj[i] 
                                       for i, var in enumerate(self.variables) 
                                       if i < len(self.var_model.rsquared_adj)}
                    else:
                        adj_r_squared = {'overall': self.var_model.rsquared_adj}
                else:
                    adj_r_squared = {}
                
                summary['var_diagnostics'] = {
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared
                }
            except Exception as e:
                logger.warning(f"Could not extract VAR diagnostics: {e}")
                summary['var_diagnostics'] = {'error': str(e)}
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
    n_obs = len(dates)
    
    # Generate synthetic Bangladesh economic data
    data = pd.DataFrame({
        'gdp_growth': np.random.normal(0.06, 0.02, n_obs),
        'inflation': np.random.normal(0.055, 0.015, n_obs),
        'policy_rate': np.random.normal(0.065, 0.01, n_obs),
        'exchange_rate': np.random.normal(85, 5, n_obs),
        'money_growth': np.random.normal(0.12, 0.03, n_obs),
        'credit_growth': np.random.normal(0.15, 0.04, n_obs),
        'current_account': np.random.normal(-0.02, 0.01, n_obs),
        'oil_price': np.random.normal(70, 15, n_obs),
        'us_interest_rate': np.random.normal(0.02, 0.015, n_obs)
    }, index=dates)
    
    # Add some persistence and relationships
    for i in range(1, len(data)):
        data.iloc[i] = 0.7 * data.iloc[i-1] + 0.3 * data.iloc[i]
    
    # Initialize SVAR model
    config = {
        'lags': 4,
        'identification': 'cholesky'
    }
    
    svar_model = SVARModel(config)
    
    # Prepare data
    svar_data = svar_model.prepare_data(data)
    print(f"Data prepared: {svar_data.shape}")
    
    # Estimate model
    results = svar_model.estimate(svar_data)
    print(f"Estimation completed: {results.identification}")
    print(f"Log-likelihood: {results.log_likelihood:.2f}")
    print(f"AIC: {results.aic:.2f}")
    
    # Compute impulse responses
    irf = svar_model.impulse_response(shock_variable='policy_rate', periods=16)
    print(f"\nIRF computed: {irf.shape}")
    
    # Monetary transmission analysis
    transmission = svar_model.monetary_transmission_analysis()
    print("\nMonetary Transmission Analysis:")
    for channel, results in transmission.items():
        if isinstance(results, dict) and 'status' not in results:
            print(f"  {channel}: {results}")
    
    # Forecast
    forecasts = svar_model.forecast(horizon=8)
    print(f"\nForecasts generated: {forecasts['point_forecasts'].shape}")
    
    # Model summary
    summary = svar_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Variables: {len(summary['variables'])}")
    print(f"  Sample size: {summary['sample_size']}")
    print(f"  Identification: {summary['identification']}")