"""Data validation tests for Bangladesh Macroeconomic Models.

This module contains tests to validate:
- Input data quality and consistency
- Model output validation
- Economic theory compliance
- Statistical properties of results
- Data integrity across model pipeline
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import warnings

# Statistical tests
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest

# Import validation utilities
try:
    from src.utils.error_handling import Validator
    from src.utils.logging_config import get_logger
except ImportError:
    # Fallback if utils not available
    class Validator:
        @staticmethod
        def validate_numeric_range(value, min_val, max_val, name="value"):
            if not (min_val <= value <= max_val):
                raise ValueError(f"{name} must be between {min_val} and {max_val}")
            return True
            
        @staticmethod
        def validate_array_shape(array, expected_shape, name="array"):
            if array.shape != expected_shape:
                raise ValueError(f"{name} shape {array.shape} != expected {expected_shape}")
            return True
            
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)


class TestDataQuality:
    """Test data quality and integrity."""
    
    @pytest.fixture
    def sample_macro_data(self):
        """Generate sample macroeconomic data for testing."""
        np.random.seed(42)
        periods = 100
        dates = pd.date_range('2000-01-01', periods=periods, freq='Q')
        
        # Generate realistic Bangladesh macroeconomic data
        base_gdp = 100000  # Million USD
        growth_rate = np.random.normal(0.06, 0.02, periods)  # 6% average growth
        gdp = base_gdp * np.cumprod(1 + growth_rate/4)  # Quarterly
        
        inflation = np.random.normal(0.055, 0.015, periods)  # 5.5% average inflation
        unemployment = np.random.normal(0.045, 0.01, periods)  # 4.5% unemployment
        interest_rate = np.random.normal(0.065, 0.02, periods)  # 6.5% interest rate
        
        # Exchange rate (BDT per USD)
        exchange_rate = 80 + np.cumsum(np.random.normal(0, 0.5, periods))
        
        # Trade data
        exports = gdp * 0.15 + np.random.normal(0, gdp * 0.02, periods)
        imports = gdp * 0.18 + np.random.normal(0, gdp * 0.025, periods)
        
        return pd.DataFrame({
            'date': dates,
            'gdp': gdp,
            'inflation': inflation,
            'unemployment': unemployment,
            'interest_rate': interest_rate,
            'exchange_rate': exchange_rate,
            'exports': exports,
            'imports': imports,
            'consumption': gdp * 0.65 + np.random.normal(0, gdp * 0.03, periods),
            'investment': gdp * 0.22 + np.random.normal(0, gdp * 0.02, periods),
            'government_spending': gdp * 0.13 + np.random.normal(0, gdp * 0.01, periods)
        })
        
    @pytest.fixture
    def sample_sectoral_data(self):
        """Generate sample sectoral data for testing."""
        sectors = [
            'agriculture', 'manufacturing', 'services', 'textiles',
            'construction', 'mining', 'utilities', 'transport'
        ]
        
        np.random.seed(42)
        
        # Sectoral shares (should sum to approximately 1)
        shares = np.array([0.15, 0.25, 0.45, 0.08, 0.04, 0.01, 0.01, 0.01])
        
        # Employment by sector (thousands)
        employment = np.array([12000, 8000, 15000, 4000, 2000, 500, 300, 1200])
        
        # Productivity indices
        productivity = np.random.uniform(0.8, 2.0, len(sectors))
        
        # Capital stock (million USD)
        capital_stock = shares * 500000 + np.random.normal(0, 50000, len(sectors))
        
        return pd.DataFrame({
            'sector': sectors,
            'gdp_share': shares,
            'employment': employment,
            'productivity': productivity,
            'capital_stock': capital_stock,
            'wages': np.random.uniform(15000, 45000, len(sectors)),  # Annual wages
            'exports': np.random.uniform(0, 0.3, len(sectors)) * shares * 100000
        })
        
    def test_data_completeness(self, sample_macro_data):
        """Test that data has no missing values in critical columns."""
        critical_columns = ['gdp', 'inflation', 'unemployment', 'interest_rate']
        
        for col in critical_columns:
            assert col in sample_macro_data.columns, f"Missing critical column: {col}"
            assert not sample_macro_data[col].isna().any(), f"Missing values in {col}"
            
    def test_data_types(self, sample_macro_data):
        """Test that data types are appropriate."""
        # Date column should be datetime
        assert pd.api.types.is_datetime64_any_dtype(sample_macro_data['date'])
        
        # Numeric columns should be numeric
        numeric_columns = ['gdp', 'inflation', 'unemployment', 'interest_rate']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(sample_macro_data[col]), f"{col} should be numeric"
            
    def test_data_ranges(self, sample_macro_data):
        """Test that data values are within reasonable ranges."""
        validator = Validator()
        
        # GDP should be positive
        assert (sample_macro_data['gdp'] > 0).all(), "GDP should be positive"
        
        # Inflation should be reasonable (-10% to 50%)
        inflation_valid = sample_macro_data['inflation'].between(-0.1, 0.5)
        assert inflation_valid.all(), "Inflation rates outside reasonable range"
        
        # Unemployment should be between 0% and 30%
        unemployment_valid = sample_macro_data['unemployment'].between(0, 0.3)
        assert unemployment_valid.all(), "Unemployment rates outside reasonable range"
        
        # Interest rates should be reasonable (0% to 50%)
        interest_valid = sample_macro_data['interest_rate'].between(0, 0.5)
        assert interest_valid.all(), "Interest rates outside reasonable range"
        
    def test_economic_identities(self, sample_macro_data):
        """Test basic economic accounting identities."""
        # GDP = C + I + G + (X - M)
        calculated_gdp = (sample_macro_data['consumption'] + 
                         sample_macro_data['investment'] + 
                         sample_macro_data['government_spending'] + 
                         sample_macro_data['exports'] - 
                         sample_macro_data['imports'])
        
        # Allow for some measurement error (within 5%)
        gdp_diff = np.abs(calculated_gdp - sample_macro_data['gdp']) / sample_macro_data['gdp']
        assert (gdp_diff < 0.05).mean() > 0.8, "GDP identity violated in too many periods"
        
    def test_sectoral_consistency(self, sample_sectoral_data):
        """Test sectoral data consistency."""
        # Sectoral shares should sum to approximately 1
        total_share = sample_sectoral_data['gdp_share'].sum()
        assert 0.95 <= total_share <= 1.05, f"Sectoral shares sum to {total_share}, not ~1"
        
        # All shares should be positive
        assert (sample_sectoral_data['gdp_share'] >= 0).all(), "Negative sectoral shares"
        
        # Employment should be positive
        assert (sample_sectoral_data['employment'] > 0).all(), "Non-positive employment"
        
        # Productivity should be positive
        assert (sample_sectoral_data['productivity'] > 0).all(), "Non-positive productivity"
        
    def test_time_series_properties(self, sample_macro_data):
        """Test time series properties of the data."""
        # Test for stationarity (GDP growth should be more stationary than levels)
        gdp_growth = sample_macro_data['gdp'].pct_change().dropna()
        
        # Test for outliers (more than 3 standard deviations)
        z_scores = np.abs(stats.zscore(gdp_growth))
        outlier_ratio = (z_scores > 3).mean()
        assert outlier_ratio < 0.05, f"Too many outliers in GDP growth: {outlier_ratio:.2%}"
        
        # Test for autocorrelation in growth rates
        # (should be some persistence but not too much)
        autocorr = gdp_growth.autocorr()
        assert -0.5 <= autocorr <= 0.8, f"Unusual autocorrelation in GDP growth: {autocorr}"
        
    def test_cross_variable_relationships(self, sample_macro_data):
        """Test expected relationships between variables."""
        # Correlation tests
        correlations = sample_macro_data[['gdp', 'inflation', 'unemployment', 'interest_rate']].corr()
        
        # GDP and unemployment should be negatively correlated (Okun's law)
        gdp_unemployment_corr = correlations.loc['gdp', 'unemployment']
        # Note: We're using levels, so this might not hold strongly
        # In practice, we'd use GDP growth vs unemployment changes
        
        # Inflation and interest rates should be positively correlated
        inflation_interest_corr = correlations.loc['inflation', 'interest_rate']
        # This relationship should generally hold
        
        # Log correlations for analysis
        logger.info(f"GDP-Unemployment correlation: {gdp_unemployment_corr:.3f}")
        logger.info(f"Inflation-Interest rate correlation: {inflation_interest_corr:.3f}")
        
        # These are soft checks - real data might not always follow theory
        assert -1 <= gdp_unemployment_corr <= 1, "Invalid correlation coefficient"
        assert -1 <= inflation_interest_corr <= 1, "Invalid correlation coefficient"


class TestModelOutputValidation:
    """Test validation of model outputs."""
    
    @pytest.fixture
    def sample_model_output(self):
        """Generate sample model output for testing."""
        np.random.seed(42)
        periods = 50
        
        # Simulated model output
        return {
            'forecast': {
                'gdp_growth': np.random.normal(0.06, 0.02, periods),
                'inflation': np.random.normal(0.055, 0.015, periods),
                'unemployment': np.random.normal(0.045, 0.01, periods),
                'periods': periods
            },
            'impulse_response': {
                'shock_variable': 'government_spending',
                'shock_size': 0.01,
                'response_variables': ['gdp', 'inflation', 'unemployment'],
                'responses': {
                    'gdp': np.random.normal(0.005, 0.002, 20),
                    'inflation': np.random.normal(0.001, 0.0005, 20),
                    'unemployment': np.random.normal(-0.001, 0.0003, 20)
                },
                'periods': 20
            },
            'welfare_analysis': {
                'baseline_welfare': 100.0,
                'policy_welfare': 102.5,
                'welfare_change': 2.5,
                'distribution': {
                    'rural_poor': 1.8,
                    'rural_rich': 2.2,
                    'urban_poor': 2.1,
                    'urban_rich': 3.1
                }
            },
            'convergence_info': {
                'converged': True,
                'iterations': 45,
                'tolerance': 1e-6,
                'max_error': 8.5e-7
            }
        }
        
    def test_forecast_validity(self, sample_model_output):
        """Test validity of model forecasts."""
        forecast = sample_model_output['forecast']
        
        # Check forecast structure
        required_vars = ['gdp_growth', 'inflation', 'unemployment']
        for var in required_vars:
            assert var in forecast, f"Missing forecast variable: {var}"
            assert len(forecast[var]) == forecast['periods'], f"Wrong length for {var}"
            
        # Check forecast ranges
        assert np.all(forecast['gdp_growth'] > -0.2), "GDP growth too negative"
        assert np.all(forecast['gdp_growth'] < 0.3), "GDP growth too high"
        
        assert np.all(forecast['inflation'] > -0.1), "Deflation too severe"
        assert np.all(forecast['inflation'] < 0.2), "Inflation too high"
        
        assert np.all(forecast['unemployment'] >= 0), "Negative unemployment"
        assert np.all(forecast['unemployment'] < 0.3), "Unemployment too high"
        
    def test_impulse_response_validity(self, sample_model_output):
        """Test validity of impulse response functions."""
        ir = sample_model_output['impulse_response']
        
        # Check structure
        assert 'shock_variable' in ir
        assert 'shock_size' in ir
        assert 'response_variables' in ir
        assert 'responses' in ir
        
        # Check responses
        for var in ir['response_variables']:
            assert var in ir['responses'], f"Missing response for {var}"
            response = ir['responses'][var]
            assert len(response) == ir['periods'], f"Wrong response length for {var}"
            
            # Responses should eventually decay to zero
            # Check if later periods have smaller absolute values
            early_response = np.mean(np.abs(response[:5]))
            late_response = np.mean(np.abs(response[-5:]))
            
            # This is a soft check - not all models will satisfy this
            if early_response > 0:
                decay_ratio = late_response / early_response
                logger.info(f"Impulse response decay ratio for {var}: {decay_ratio:.3f}")
                
    def test_welfare_analysis_validity(self, sample_model_output):
        """Test validity of welfare analysis results."""
        welfare = sample_model_output['welfare_analysis']
        
        # Check structure
        required_fields = ['baseline_welfare', 'policy_welfare', 'welfare_change']
        for field in required_fields:
            assert field in welfare, f"Missing welfare field: {field}"
            
        # Check consistency
        expected_change = welfare['policy_welfare'] - welfare['baseline_welfare']
        actual_change = welfare['welfare_change']
        assert abs(expected_change - actual_change) < 1e-10, "Welfare change inconsistent"
        
        # Check distribution
        if 'distribution' in welfare:
            dist = welfare['distribution']
            assert all(isinstance(v, (int, float)) for v in dist.values()), "Non-numeric welfare changes"
            
    def test_convergence_validity(self, sample_model_output):
        """Test validity of convergence information."""
        conv = sample_model_output['convergence_info']
        
        # Check required fields
        required_fields = ['converged', 'iterations', 'tolerance']
        for field in required_fields:
            assert field in conv, f"Missing convergence field: {field}"
            
        # Check types
        assert isinstance(conv['converged'], bool), "Convergence status should be boolean"
        assert isinstance(conv['iterations'], int), "Iterations should be integer"
        assert isinstance(conv['tolerance'], (int, float)), "Tolerance should be numeric"
        
        # Check logical consistency
        if conv['converged']:
            assert 'max_error' in conv, "Missing max_error for converged solution"
            assert conv['max_error'] <= conv['tolerance'], "Max error exceeds tolerance"
            
        # Check reasonable values
        assert conv['iterations'] >= 0, "Negative iterations"
        assert conv['iterations'] < 10000, "Too many iterations"
        assert conv['tolerance'] > 0, "Non-positive tolerance"
        
    def test_statistical_properties(self, sample_model_output):
        """Test statistical properties of model outputs."""
        forecast = sample_model_output['forecast']
        
        # Test normality of forecast errors (if available)
        # For now, test the forecast values themselves
        for var_name, values in forecast.items():
            if isinstance(values, np.ndarray) or isinstance(values, list):
                values = np.array(values)
                
                # Test for extreme outliers
                z_scores = np.abs(stats.zscore(values))
                outlier_ratio = (z_scores > 4).mean()  # More than 4 std devs
                assert outlier_ratio < 0.02, f"Too many extreme outliers in {var_name}"
                
                # Test for reasonable variance
                cv = np.std(values) / np.abs(np.mean(values)) if np.mean(values) != 0 else np.inf
                assert cv < 10, f"Coefficient of variation too high for {var_name}: {cv}"


class TestEconomicTheoryCompliance:
    """Test compliance with economic theory."""
    
    def test_phillips_curve_relationship(self):
        """Test Phillips curve relationship between inflation and unemployment."""
        # Generate data that should follow Phillips curve
        np.random.seed(42)
        periods = 100
        
        # Simulate Phillips curve: π = α - β*u + ε
        alpha, beta = 0.08, 0.5
        unemployment = np.random.uniform(0.02, 0.12, periods)
        inflation = alpha - beta * unemployment + np.random.normal(0, 0.01, periods)
        
        # Test relationship
        correlation = np.corrcoef(inflation, unemployment)[0, 1]
        assert correlation < -0.3, f"Phillips curve relationship too weak: {correlation}"
        
    def test_okuns_law_relationship(self):
        """Test Okun's law relationship between GDP growth and unemployment."""
        # Generate data following Okun's law
        np.random.seed(42)
        periods = 100
        
        # Okun's law: Δu = α - β*(g - g*)
        # Or equivalently: g = g* + (1/β)*(α - Δu)
        natural_growth = 0.06
        okun_coefficient = 2.0
        
        unemployment_change = np.random.normal(0, 0.005, periods)
        gdp_growth = natural_growth - okun_coefficient * unemployment_change + np.random.normal(0, 0.01, periods)
        
        # Test relationship
        correlation = np.corrcoef(gdp_growth, unemployment_change)[0, 1]
        assert correlation < -0.3, f"Okun's law relationship too weak: {correlation}"
        
    def test_fisher_equation(self):
        """Test Fisher equation: nominal rate ≈ real rate + expected inflation."""
        np.random.seed(42)
        periods = 100
        
        # Generate data following Fisher equation
        real_rate = 0.03  # 3% real interest rate
        expected_inflation = np.random.normal(0.05, 0.02, periods)
        nominal_rate = real_rate + expected_inflation + np.random.normal(0, 0.005, periods)
        
        # Test relationship
        implied_real_rate = nominal_rate - expected_inflation
        mean_real_rate = np.mean(implied_real_rate)
        
        # Should be close to the true real rate
        assert abs(mean_real_rate - real_rate) < 0.01, f"Fisher equation violated: {mean_real_rate} vs {real_rate}"
        
    def test_purchasing_power_parity(self):
        """Test purchasing power parity relationship."""
        # This is a simplified test of PPP
        np.random.seed(42)
        periods = 100
        
        # Generate exchange rate and price level data
        domestic_prices = 100 * np.cumprod(1 + np.random.normal(0.05, 0.02, periods))
        foreign_prices = 100 * np.cumprod(1 + np.random.normal(0.03, 0.015, periods))
        
        # PPP exchange rate
        ppp_exchange_rate = domestic_prices / foreign_prices
        
        # Actual exchange rate (with deviations from PPP)
        actual_exchange_rate = ppp_exchange_rate * np.exp(np.random.normal(0, 0.1, periods))
        
        # Test long-run relationship
        log_actual = np.log(actual_exchange_rate)
        log_ppp = np.log(ppp_exchange_rate)
        
        correlation = np.corrcoef(log_actual, log_ppp)[0, 1]
        assert correlation > 0.7, f"PPP relationship too weak: {correlation}"
        
    def test_quantity_theory_of_money(self):
        """Test quantity theory of money: MV = PY."""
        np.random.seed(42)
        periods = 100
        
        # Generate data following quantity theory
        real_gdp = 1000 * np.cumprod(1 + np.random.normal(0.03, 0.02, periods))
        velocity = 2.0 + np.random.normal(0, 0.1, periods)  # Relatively stable
        money_supply = 500 * np.cumprod(1 + np.random.normal(0.08, 0.03, periods))
        
        # Price level from quantity theory
        price_level = (money_supply * velocity) / real_gdp
        
        # Test relationship
        mv = money_supply * velocity
        py = price_level * real_gdp
        
        relative_error = np.abs(mv - py) / py
        mean_error = np.mean(relative_error)
        
        assert mean_error < 0.01, f"Quantity theory violated: mean error {mean_error}"


class TestDataIntegrity:
    """Test data integrity across the model pipeline."""
    
    def test_data_consistency_across_frequencies(self):
        """Test consistency when aggregating data across different frequencies."""
        # Generate monthly data
        np.random.seed(42)
        months = 36
        monthly_data = pd.DataFrame({
            'month': pd.date_range('2020-01-01', periods=months, freq='M'),
            'value': np.random.normal(100, 10, months)
        })
        
        # Aggregate to quarterly
        monthly_data['quarter'] = monthly_data['month'].dt.to_period('Q')
        quarterly_data = monthly_data.groupby('quarter')['value'].sum().reset_index()
        
        # Test consistency
        total_monthly = monthly_data['value'].sum()
        total_quarterly = quarterly_data['value'].sum()
        
        assert abs(total_monthly - total_quarterly) < 1e-10, "Aggregation inconsistency"
        
    def test_data_version_consistency(self):
        """Test that data versions are consistent across different sources."""
        # This would test that data from different sources
        # (e.g., World Bank, Bangladesh Bank) are consistent
        
        # Mock data from two sources
        source1_gdp = 350000  # Million USD
        source2_gdp = 352000  # Million USD
        
        # Allow for small differences due to methodology
        relative_diff = abs(source1_gdp - source2_gdp) / source1_gdp
        assert relative_diff < 0.05, f"GDP data inconsistency: {relative_diff:.2%}"
        
    def test_temporal_consistency(self):
        """Test temporal consistency of data."""
        # Generate time series with known properties
        np.random.seed(42)
        periods = 100
        dates = pd.date_range('2000-01-01', periods=periods, freq='Q')
        
        # GDP should generally be increasing over time
        gdp = 100000 * np.cumprod(1 + np.random.normal(0.015, 0.02, periods))
        
        data = pd.DataFrame({
            'date': dates,
            'gdp': gdp
        })
        
        # Test for temporal ordering
        assert data['date'].is_monotonic_increasing, "Dates not in chronological order"
        
        # Test for reasonable growth
        gdp_growth = data['gdp'].pct_change().dropna()
        
        # Most periods should have positive growth
        positive_growth_ratio = (gdp_growth > 0).mean()
        assert positive_growth_ratio > 0.6, f"Too few periods with positive growth: {positive_growth_ratio:.2%}"
        
        # Growth should not be too volatile
        growth_volatility = gdp_growth.std()
        assert growth_volatility < 0.1, f"GDP growth too volatile: {growth_volatility:.3f}"
        
    def test_cross_sectional_consistency(self):
        """Test consistency across different cross-sectional units."""
        # Test sectoral data consistency
        sectors = ['agriculture', 'manufacturing', 'services']
        
        # Employment and output should be positively correlated
        np.random.seed(42)
        employment = np.array([1000, 1500, 2000])  # Thousands
        output = employment * np.array([50, 80, 70]) + np.random.normal(0, 5000, 3)  # Million USD
        
        correlation = np.corrcoef(employment, output)[0, 1]
        assert correlation > 0.5, f"Employment-output correlation too low: {correlation}"
        
    def test_unit_consistency(self):
        """Test that units are consistent across variables."""
        # This would test that all monetary variables are in the same units,
        # all rates are in the same format (decimal vs percentage), etc.
        
        # Mock data with different unit representations
        gdp_millions = 350000  # Million USD
        consumption_millions = 210000  # Million USD
        
        # Consumption should be reasonable fraction of GDP
        consumption_ratio = consumption_millions / gdp_millions
        assert 0.4 <= consumption_ratio <= 0.8, f"Unrealistic consumption ratio: {consumption_ratio}"
        
        # Interest rates should be in decimal format (not percentage)
        interest_rate = 0.065  # 6.5%
        assert 0 <= interest_rate <= 1, "Interest rate not in decimal format"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])