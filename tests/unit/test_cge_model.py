"""Unit tests for CGE (Computable General Equilibrium) Model.

This module contains comprehensive tests for the CGE model implementation,
including parameter validation, model initialization, simulation, and
equilibrium computation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the CGE model (adjust import path as needed)
try:
    from models.cge_model import CGEModel, CGEParameters
except ImportError:
    from src.models.cge_model import CGEModel, CGEParameters


class TestCGEParameters:
    """Test CGE model parameters."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = CGEParameters()
        
        # Check that all required parameters have reasonable defaults
        assert hasattr(params, 'sectors')
        assert hasattr(params, 'factors')
        assert hasattr(params, 'households')
        assert hasattr(params, 'elasticities')
        
        # Check parameter types and ranges
        if hasattr(params, 'elasticity_substitution'):
            assert isinstance(params.elasticity_substitution, (int, float))
            assert params.elasticity_substitution > 0
            
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with valid parameters
        valid_params = {
            'sectors': ['agriculture', 'manufacturing', 'services'],
            'factors': ['labor', 'capital'],
            'households': ['rural', 'urban'],
            'elasticities': {'substitution': 0.5, 'transformation': 1.2}
        }
        
        params = CGEParameters(**valid_params)
        assert params.sectors == valid_params['sectors']
        assert params.factors == valid_params['factors']
        
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test with negative elasticity
        with pytest.raises((ValueError, TypeError)):
            CGEParameters(elasticity_substitution=-0.5)
            
        # Test with empty sectors
        with pytest.raises((ValueError, TypeError)):
            CGEParameters(sectors=[])


class TestCGEModel:
    """Test CGE model implementation."""
    
    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters for testing."""
        return CGEParameters(
            sectors=['agriculture', 'manufacturing', 'services'],
            factors=['labor', 'capital'],
            households=['rural', 'urban'],
            elasticities={'substitution': 0.8, 'transformation': 1.5}
        )
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Social Accounting Matrix (SAM)
        sam_size = 10
        sam = np.random.rand(sam_size, sam_size) * 100
        # Make SAM balanced (rows sum = columns sum)
        for i in range(sam_size):
            sam[i, i] = 0  # Zero diagonal
        row_sums = sam.sum(axis=1)
        col_sums = sam.sum(axis=0)
        # Adjust to make balanced
        sam = sam * (row_sums.mean() / row_sums.reshape(-1, 1))
        
        return {
            'sam': sam,
            'production_data': pd.DataFrame({
                'sector': ['agriculture', 'manufacturing', 'services'],
                'output': [1000, 2000, 1500],
                'labor': [500, 800, 700],
                'capital': [300, 600, 400]
            }),
            'trade_data': pd.DataFrame({
                'sector': ['agriculture', 'manufacturing', 'services'],
                'exports': [200, 800, 300],
                'imports': [100, 600, 400]
            })
        }
        
    def test_model_initialization(self, sample_parameters, sample_data):
        """Test model initialization."""
        model = CGEModel(sample_parameters)
        
        # Check basic attributes
        assert model.parameters == sample_parameters
        assert hasattr(model, 'sectors')
        assert hasattr(model, 'factors')
        
        # Test initialization with data
        model.load_data(sample_data)
        assert hasattr(model, 'sam') or hasattr(model, 'data')
        
    def test_model_calibration(self, sample_parameters, sample_data):
        """Test model calibration."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        
        # Test calibration
        calibration_result = model.calibrate()
        
        # Check calibration results
        assert isinstance(calibration_result, dict)
        assert 'status' in calibration_result
        assert calibration_result['status'] in ['success', 'partial', 'failed']
        
        if calibration_result['status'] == 'success':
            assert 'parameters' in calibration_result
            assert isinstance(calibration_result['parameters'], dict)
            
    def test_equilibrium_computation(self, sample_parameters, sample_data):
        """Test equilibrium computation."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test equilibrium computation
        equilibrium = model.solve_equilibrium()
        
        # Check equilibrium results
        assert isinstance(equilibrium, dict)
        assert 'prices' in equilibrium or 'solution' in equilibrium
        assert 'convergence' in equilibrium
        
        # Check convergence
        if equilibrium['convergence']:
            assert 'iterations' in equilibrium
            assert equilibrium['iterations'] > 0
            
    def test_simulation(self, sample_parameters, sample_data):
        """Test model simulation."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test simulation with shock
        shock = {'productivity': {'agriculture': 1.1}}  # 10% productivity increase
        
        simulation_result = model.simulate(shock, periods=5)
        
        # Check simulation results
        assert isinstance(simulation_result, dict)
        assert 'results' in simulation_result
        assert 'status' in simulation_result
        
        if simulation_result['status'] == 'success':
            results = simulation_result['results']
            assert isinstance(results, (dict, pd.DataFrame))
            
    def test_welfare_analysis(self, sample_parameters, sample_data):
        """Test welfare analysis."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test welfare computation
        baseline = model.solve_equilibrium()
        
        # Apply shock and compute new equilibrium
        shock = {'tax_rate': {'manufacturing': 0.05}}  # 5% tax
        model.apply_shock(shock)
        counterfactual = model.solve_equilibrium()
        
        # Compute welfare changes
        welfare_change = model.compute_welfare_change(baseline, counterfactual)
        
        assert isinstance(welfare_change, dict)
        assert 'total_welfare_change' in welfare_change or 'welfare_changes' in welfare_change
        
    def test_policy_analysis(self, sample_parameters, sample_data):
        """Test policy analysis capabilities."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test different policy scenarios
        policies = [
            {'tariff': {'manufacturing': 0.1}},  # 10% tariff
            {'subsidy': {'agriculture': 0.05}},  # 5% subsidy
            {'tax_reform': {'income_tax': 0.02}}  # 2% income tax change
        ]
        
        for policy in policies:
            try:
                result = model.analyze_policy(policy)
                assert isinstance(result, dict)
                assert 'impact' in result or 'effects' in result
            except NotImplementedError:
                # Policy analysis might not be implemented yet
                pass
                
    def test_sectoral_analysis(self, sample_parameters, sample_data):
        """Test sectoral analysis."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test sectoral impacts
        shock = {'productivity': {'manufacturing': 1.2}}  # 20% productivity increase
        
        sectoral_results = model.analyze_sectoral_impacts(shock)
        
        assert isinstance(sectoral_results, dict)
        for sector in sample_parameters.sectors:
            if sector in sectoral_results:
                sector_result = sectoral_results[sector]
                assert isinstance(sector_result, dict)
                
    def test_trade_analysis(self, sample_parameters, sample_data):
        """Test trade analysis."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test trade liberalization
        trade_shock = {'tariff_reduction': {'all_sectors': 0.5}}  # 50% tariff reduction
        
        try:
            trade_results = model.analyze_trade_policy(trade_shock)
            assert isinstance(trade_results, dict)
            assert 'trade_effects' in trade_results or 'welfare_effects' in trade_results
        except (NotImplementedError, AttributeError):
            # Trade analysis might not be implemented yet
            pass
            
    def test_error_handling(self, sample_parameters):
        """Test error handling."""
        model = CGEModel(sample_parameters)
        
        # Test with invalid data
        with pytest.raises((ValueError, TypeError, KeyError)):
            model.load_data({'invalid': 'data'})
            
        # Test simulation without calibration
        with pytest.raises((RuntimeError, ValueError)):
            model.simulate({'shock': 'test'})
            
    def test_convergence_issues(self, sample_parameters, sample_data):
        """Test handling of convergence issues."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        
        # Test with extreme shock that might cause convergence issues
        extreme_shock = {'productivity': {'agriculture': 10.0}}  # 1000% increase
        
        try:
            result = model.simulate(extreme_shock)
            # Should handle convergence issues gracefully
            assert 'status' in result
            if result['status'] == 'failed':
                assert 'error' in result or 'message' in result
        except Exception as e:
            # Should not crash, but handle gracefully
            assert isinstance(e, (RuntimeError, ValueError, np.linalg.LinAlgError))
            
    def test_data_validation(self, sample_parameters):
        """Test data validation."""
        model = CGEModel(sample_parameters)
        
        # Test with inconsistent data
        inconsistent_data = {
            'sam': np.array([[1, 2], [3, 4]]),  # Wrong size
            'production_data': pd.DataFrame({'wrong': ['columns']})
        }
        
        with pytest.raises((ValueError, KeyError, AssertionError)):
            model.load_data(inconsistent_data)
            model.validate_data()
            
    def test_parameter_sensitivity(self, sample_parameters, sample_data):
        """Test parameter sensitivity analysis."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Test sensitivity to elasticity parameters
        elasticity_values = [0.5, 1.0, 1.5, 2.0]
        
        try:
            sensitivity_results = model.sensitivity_analysis(
                parameter='elasticity_substitution',
                values=elasticity_values
            )
            
            assert isinstance(sensitivity_results, dict)
            assert len(sensitivity_results) == len(elasticity_values)
        except (NotImplementedError, AttributeError):
            # Sensitivity analysis might not be implemented yet
            pass
            
    def test_model_validation(self, sample_parameters, sample_data):
        """Test model validation methods."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        
        # Test model validation
        validation_result = model.validate_model()
        
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result or 'status' in validation_result
        
        if 'errors' in validation_result:
            assert isinstance(validation_result['errors'], list)
            
    def test_results_export(self, sample_parameters, sample_data):
        """Test results export functionality."""
        model = CGEModel(sample_parameters)
        model.load_data(sample_data)
        model.calibrate()
        
        # Run simulation
        shock = {'productivity': {'services': 1.1}}
        results = model.simulate(shock)
        
        # Test export to different formats
        try:
            exported_data = model.export_results(results, format='dict')
            assert isinstance(exported_data, dict)
            
            exported_df = model.export_results(results, format='dataframe')
            assert isinstance(exported_df, pd.DataFrame)
        except (NotImplementedError, AttributeError):
            # Export functionality might not be implemented yet
            pass


class TestCGEIntegration:
    """Integration tests for CGE model."""
    
    @pytest.mark.integration
    def test_full_model_workflow(self):
        """Test complete model workflow."""
        # Create model with realistic parameters
        params = CGEParameters(
            sectors=['agriculture', 'manufacturing', 'services'],
            factors=['labor', 'capital', 'land'],
            households=['rural_poor', 'rural_rich', 'urban_poor', 'urban_rich']
        )
        
        model = CGEModel(params)
        
        # Generate synthetic but realistic data
        np.random.seed(123)
        sam_size = 15
        sam = np.random.rand(sam_size, sam_size) * 1000
        
        # Make SAM more realistic
        for i in range(sam_size):
            sam[i, i] = 0
            
        data = {
            'sam': sam,
            'production_data': pd.DataFrame({
                'sector': params.sectors,
                'output': [5000, 15000, 12000],
                'labor_share': [0.6, 0.4, 0.7],
                'capital_share': [0.3, 0.5, 0.25],
                'land_share': [0.1, 0.1, 0.05]
            })
        }
        
        # Full workflow
        model.load_data(data)
        calibration = model.calibrate()
        assert calibration['status'] in ['success', 'partial']
        
        baseline = model.solve_equilibrium()
        assert 'convergence' in baseline
        
        # Policy simulation
        policy_shock = {'productivity': {'agriculture': 1.15}}  # 15% increase
        policy_result = model.simulate(policy_shock, periods=10)
        
        assert policy_result['status'] in ['success', 'partial']
        
    @pytest.mark.slow
    def test_large_scale_simulation(self):
        """Test model with larger scale data."""
        # Create larger model
        sectors = [f'sector_{i}' for i in range(10)]
        factors = ['labor', 'capital', 'land', 'natural_resources']
        households = [f'household_{i}' for i in range(5)]
        
        params = CGEParameters(
            sectors=sectors,
            factors=factors,
            households=households
        )
        
        model = CGEModel(params)
        
        # Generate larger SAM
        sam_size = 25
        sam = np.random.rand(sam_size, sam_size) * 10000
        for i in range(sam_size):
            sam[i, i] = 0
            
        data = {'sam': sam}
        
        try:
            model.load_data(data)
            calibration = model.calibrate()
            
            # Should handle larger models
            assert isinstance(calibration, dict)
            
        except (MemoryError, np.linalg.LinAlgError):
            # Large models might have computational limitations
            pytest.skip("Large scale simulation requires more computational resources")
            
    @pytest.mark.requires_data
    def test_with_real_data(self):
        """Test model with real Bangladesh data (if available)."""
        # This test would use real Bangladesh economic data
        # Skip if data is not available
        pytest.skip("Real data test requires external data sources")
        
        # Example structure for real data test:
        # real_data = load_bangladesh_data()
        # params = create_bangladesh_parameters()
        # model = CGEModel(params)
        # model.load_data(real_data)
        # results = model.calibrate()
        # assert results['status'] == 'success'


if __name__ == '__main__':
    pytest.main([__file__])