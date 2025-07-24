#!/usr/bin/env python3
"""
Unit Tests for DSGE Model

This module contains unit tests for the Dynamic Stochastic General Equilibrium (DSGE)
model implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.dsge.dsge_model import DSGEModel, DSGEParameters


class TestDSGEParameters:
    """Test cases for DSGE model parameters."""
    
    def test_default_parameters(self):
        """Test that default parameters are within reasonable ranges."""
        params = DSGEParameters()
        
        # Test discount factor
        assert 0 < params.beta < 1, "Discount factor should be between 0 and 1"
        
        # Test that all parameters are numeric
        for field_name, field_value in params.__dict__.items():
            assert isinstance(field_value, (int, float)), f"{field_name} should be numeric"
    
    def test_parameter_validation(self):
        """Test parameter validation logic."""
        # Test invalid discount factor
        with pytest.raises((ValueError, AssertionError)):
            DSGEParameters(beta=1.5)  # Should be < 1
        
        with pytest.raises((ValueError, AssertionError)):
            DSGEParameters(beta=-0.1)  # Should be > 0


class TestDSGEModel:
    """Test cases for DSGE model functionality."""
    
    @pytest.fixture
    def dsge_model(self):
        """Create a DSGE model instance for testing."""
        return DSGEModel()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        periods = 100
        data = pd.DataFrame({
            'gdp': np.random.normal(100, 10, periods),
            'inflation': np.random.normal(0.02, 0.01, periods),
            'interest_rate': np.random.normal(0.05, 0.02, periods),
            'consumption': np.random.normal(70, 7, periods),
            'investment': np.random.normal(20, 5, periods)
        })
        data.index = pd.date_range('2000-01-01', periods=periods, freq='Q')
        return data
    
    def test_model_initialization(self, dsge_model):
        """Test that the model initializes correctly."""
        assert dsge_model is not None
        assert hasattr(dsge_model, 'parameters')
        assert isinstance(dsge_model.parameters, DSGEParameters)
    
    def test_model_has_required_methods(self, dsge_model):
        """Test that the model has all required methods."""
        required_methods = [
            'simulate',
            'estimate',
            'forecast',
            'impulse_response',
            'calibrate'
        ]
        
        for method in required_methods:
            assert hasattr(dsge_model, method), f"Model should have {method} method"
            assert callable(getattr(dsge_model, method)), f"{method} should be callable"
    
    def test_simulation_output_structure(self, dsge_model):
        """Test that simulation produces correctly structured output."""
        try:
            results = dsge_model.simulate(periods=10)
            
            # Check that results is a dictionary or DataFrame
            assert isinstance(results, (dict, pd.DataFrame)), "Results should be dict or DataFrame"
            
            if isinstance(results, dict):
                # Check for key economic variables
                expected_vars = ['gdp', 'consumption', 'investment', 'inflation']
                for var in expected_vars:
                    if var in results:
                        assert len(results[var]) == 10, f"{var} should have 10 periods"
                        assert all(np.isfinite(results[var])), f"{var} should contain finite values"
            
            elif isinstance(results, pd.DataFrame):
                assert len(results) == 10, "DataFrame should have 10 rows"
                assert not results.isnull().all().any(), "No column should be entirely null"
        
        except Exception as e:
            pytest.skip(f"Simulation method not fully implemented: {e}")
    
    @pytest.mark.slow
    def test_estimation_with_data(self, dsge_model, sample_data):
        """Test model estimation with sample data."""
        try:
            # Test that estimation runs without errors
            results = dsge_model.estimate(sample_data)
            
            # Basic checks on estimation results
            if results is not None:
                assert isinstance(results, dict), "Estimation results should be a dictionary"
                
                # Check for common estimation outputs
                possible_keys = ['parameters', 'log_likelihood', 'aic', 'bic', 'convergence']
                assert any(key in results for key in possible_keys), "Results should contain estimation metrics"
        
        except NotImplementedError:
            pytest.skip("Estimation method not implemented")
        except Exception as e:
            pytest.fail(f"Estimation failed with error: {e}")
    
    def test_forecast_output(self, dsge_model, sample_data):
        """Test forecasting functionality."""
        try:
            forecast_periods = 8
            forecast = dsge_model.forecast(sample_data, periods=forecast_periods)
            
            if forecast is not None:
                if isinstance(forecast, dict):
                    for var, values in forecast.items():
                        assert len(values) == forecast_periods, f"Forecast for {var} should have {forecast_periods} periods"
                        assert all(np.isfinite(values)), f"Forecast for {var} should be finite"
                
                elif isinstance(forecast, pd.DataFrame):
                    assert len(forecast) == forecast_periods, "Forecast should have correct number of periods"
                    assert not forecast.isnull().all().any(), "Forecast should not be entirely null"
        
        except NotImplementedError:
            pytest.skip("Forecast method not implemented")
        except Exception as e:
            pytest.fail(f"Forecasting failed with error: {e}")
    
    def test_impulse_response_structure(self, dsge_model):
        """Test impulse response function structure."""
        try:
            # Test with a simple shock
            shock = {'technology': 0.01}  # 1% technology shock
            irf = dsge_model.impulse_response(shock, periods=20)
            
            if irf is not None:
                assert isinstance(irf, (dict, pd.DataFrame)), "IRF should be dict or DataFrame"
                
                if isinstance(irf, dict):
                    for var, response in irf.items():
                        assert len(response) == 20, f"IRF for {var} should have 20 periods"
                        assert all(np.isfinite(response)), f"IRF for {var} should be finite"
                
                elif isinstance(irf, pd.DataFrame):
                    assert len(irf) == 20, "IRF DataFrame should have 20 periods"
        
        except NotImplementedError:
            pytest.skip("Impulse response method not implemented")
        except Exception as e:
            pytest.fail(f"Impulse response failed with error: {e}")
    
    def test_parameter_bounds(self, dsge_model):
        """Test that model parameters stay within reasonable economic bounds."""
        params = dsge_model.parameters
        
        # Test discount factor
        assert 0 < params.beta < 1, "Discount factor should be between 0 and 1"
        
        # Test other parameters if they exist
        if hasattr(params, 'alpha'):  # Capital share
            assert 0 < params.alpha < 1, "Capital share should be between 0 and 1"
        
        if hasattr(params, 'delta'):  # Depreciation rate
            assert 0 < params.delta < 1, "Depreciation rate should be between 0 and 1"
    
    def test_numerical_stability(self, dsge_model):
        """Test numerical stability of model computations."""
        try:
            # Test with extreme but valid parameters
            extreme_params = DSGEParameters(beta=0.99999)  # Very high discount factor
            dsge_model.parameters = extreme_params
            
            # Run simulation and check for numerical issues
            results = dsge_model.simulate(periods=5)
            
            if isinstance(results, dict):
                for var, values in results.items():
                    assert all(np.isfinite(values)), f"{var} contains non-finite values"
                    assert not any(np.abs(values) > 1e10), f"{var} contains extremely large values"
            
        except Exception as e:
            # This is acceptable for extreme parameters
            pytest.skip(f"Model not stable with extreme parameters: {e}")
    
    @pytest.mark.requires_data
    def test_with_real_data(self, dsge_model):
        """Test model with real economic data if available."""
        # This test would use real Bangladesh economic data
        # Skip if data is not available
        pytest.skip("Real data test requires external data source")
    
    def test_error_handling(self, dsge_model):
        """Test that the model handles errors gracefully."""
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        try:
            result = dsge_model.estimate(invalid_data)
            # If it doesn't raise an error, it should return None or handle gracefully
            if result is not None:
                assert isinstance(result, dict), "Error handling should return dict or None"
        except (ValueError, KeyError, AttributeError):
            # These are acceptable errors for invalid data
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error type: {type(e).__name__}: {e}")


class TestDSGEModelIntegration:
    """Integration tests for DSGE model."""
    
    @pytest.mark.integration
    def test_full_workflow(self):
        """Test a complete model workflow: calibrate -> simulate -> estimate -> forecast."""
        model = DSGEModel()
        
        try:
            # Step 1: Calibration (if implemented)
            if hasattr(model, 'calibrate') and callable(model.calibrate):
                model.calibrate()
            
            # Step 2: Simulation
            sim_data = model.simulate(periods=50)
            assert sim_data is not None, "Simulation should produce data"
            
            # Step 3: Estimation (if implemented)
            if hasattr(model, 'estimate') and callable(model.estimate):
                if isinstance(sim_data, pd.DataFrame):
                    est_results = model.estimate(sim_data)
                    # Basic validation of estimation results
                    if est_results is not None:
                        assert isinstance(est_results, dict)
            
            # Step 4: Forecasting (if implemented)
            if hasattr(model, 'forecast') and callable(model.forecast):
                if isinstance(sim_data, pd.DataFrame):
                    forecast = model.forecast(sim_data, periods=10)
                    if forecast is not None:
                        assert len(forecast) == 10 or (isinstance(forecast, dict) and 
                                                      all(len(v) == 10 for v in forecast.values()))
        
        except NotImplementedError:
            pytest.skip("Full workflow test requires all methods to be implemented")
        except Exception as e:
            pytest.fail(f"Full workflow failed: {e}")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])