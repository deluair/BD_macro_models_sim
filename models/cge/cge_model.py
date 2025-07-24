#!/usr/bin/env python3
"""
Computable General Equilibrium (CGE) Model for Bangladesh

This module implements a multi-sector CGE model specifically designed for the Bangladesh economy.
The model includes detailed sectoral disaggregation, trade flows, and policy instruments
relevant for Bangladesh's economic structure.

Key Features:
- Multi-sector production structure
- Household consumption with different income groups
- Government sector with fiscal policy instruments
- International trade with rest of world
- Labor market with formal/informal segmentation
- Agricultural sector with subsistence and commercial farming
- Manufacturing sector with ready-made garments focus
- Services sector including financial services

Author: Bangladesh Macro Models Team
Date: 2025
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CGEParameters:
    """
    Data class for CGE model parameters
    """
    # Production parameters
    alpha: Dict[str, float]  # Share parameters in CES production
    sigma: Dict[str, float]  # Elasticity of substitution
    beta: Dict[str, float]   # Scale parameters
    
    # Consumption parameters
    gamma: Dict[str, float]  # Marginal budget shares (LES)
    subsistence: Dict[str, float]  # Subsistence consumption
    
    # Trade parameters
    armington_sigma: Dict[str, float]  # Armington elasticity
    export_sigma: Dict[str, float]     # Export transformation elasticity
    
    # Tax rates
    tax_rates: Dict[str, float]
    
    # Other parameters
    depreciation_rate: float
    population_growth: float
    productivity_growth: float

class CGEModel:
    """
    Computable General Equilibrium Model for Bangladesh
    
    This class implements a multi-sector CGE model with detailed representation
    of Bangladesh's economic structure including agriculture, manufacturing
    (with emphasis on ready-made garments), and services sectors.
    """
    
    def __init__(self, config: Dict, data: Optional[Dict] = None):
        """
        Initialize the CGE model
        
        Args:
            config: Configuration dictionary with model parameters
            data: Economic data for calibration
        """
        self.config = config
        self.data = data
        
        # Define sectors
        self.sectors = [
            'agriculture',      # Rice, wheat, other crops
            'livestock',        # Cattle, poultry, fisheries
            'textiles',         # Ready-made garments, textiles
            'manufacturing',    # Other manufacturing
            'construction',     # Construction and real estate
            'services',         # Trade, transport, other services
            'government',       # Government services
            'financial'         # Banking and financial services
        ]
        
        # Define household types
        self.household_types = [
            'rural_poor',       # Rural households below poverty line
            'rural_nonpoor',    # Rural households above poverty line
            'urban_poor',       # Urban households below poverty line
            'urban_nonpoor',    # Urban households above poverty line
            'urban_rich'        # High-income urban households
        ]
        
        # Define factors of production
        self.factors = [
            'labor_unskilled',  # Unskilled labor
            'labor_skilled',    # Skilled labor
            'capital',          # Physical capital
            'land'              # Agricultural land
        ]
        
        # Initialize model components
        self.parameters = None
        self.variables = None
        self.equations = None
        self.solution = None
        self.baseline = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize model parameters and structure
        """
        logger.info("Initializing CGE model for Bangladesh")
        
        # Set up parameters
        self._setup_parameters()
        
        # Initialize variables
        self._initialize_variables()
        
        # Set up equations
        self._setup_equations()
        
        logger.info("CGE model initialization completed")
    
    def _setup_parameters(self):
        """
        Set up model parameters from config and data
        """
        # Production function parameters (CES)
        alpha = {
            'agriculture': {'labor_unskilled': 0.6, 'labor_skilled': 0.1, 'capital': 0.2, 'land': 0.1},
            'livestock': {'labor_unskilled': 0.5, 'labor_skilled': 0.1, 'capital': 0.3, 'land': 0.1},
            'textiles': {'labor_unskilled': 0.7, 'labor_skilled': 0.2, 'capital': 0.1},
            'manufacturing': {'labor_unskilled': 0.4, 'labor_skilled': 0.3, 'capital': 0.3},
            'construction': {'labor_unskilled': 0.6, 'labor_skilled': 0.2, 'capital': 0.2},
            'services': {'labor_unskilled': 0.3, 'labor_skilled': 0.5, 'capital': 0.2},
            'government': {'labor_unskilled': 0.2, 'labor_skilled': 0.6, 'capital': 0.2},
            'financial': {'labor_unskilled': 0.1, 'labor_skilled': 0.7, 'capital': 0.2}
        }
        
        # Elasticity of substitution
        sigma = {
            'agriculture': 0.8, 'livestock': 0.8, 'textiles': 1.2,
            'manufacturing': 1.0, 'construction': 0.9, 'services': 1.1,
            'government': 0.7, 'financial': 1.3
        }
        
        # Consumption parameters (Linear Expenditure System)
        gamma = {
            'rural_poor': {
                'agriculture': 0.4, 'livestock': 0.1, 'textiles': 0.1,
                'manufacturing': 0.2, 'construction': 0.0, 'services': 0.2
            },
            'rural_nonpoor': {
                'agriculture': 0.3, 'livestock': 0.15, 'textiles': 0.15,
                'manufacturing': 0.25, 'construction': 0.05, 'services': 0.1
            },
            'urban_poor': {
                'agriculture': 0.35, 'livestock': 0.1, 'textiles': 0.15,
                'manufacturing': 0.25, 'construction': 0.0, 'services': 0.15
            },
            'urban_nonpoor': {
                'agriculture': 0.25, 'livestock': 0.15, 'textiles': 0.2,
                'manufacturing': 0.25, 'construction': 0.05, 'services': 0.1
            },
            'urban_rich': {
                'agriculture': 0.15, 'livestock': 0.1, 'textiles': 0.25,
                'manufacturing': 0.25, 'construction': 0.1, 'services': 0.15
            }
        }
        
        # Subsistence consumption
        subsistence = {
            'rural_poor': {'agriculture': 100, 'livestock': 20},
            'rural_nonpoor': {'agriculture': 80, 'livestock': 30},
            'urban_poor': {'agriculture': 90, 'livestock': 25},
            'urban_nonpoor': {'agriculture': 70, 'livestock': 35},
            'urban_rich': {'agriculture': 50, 'livestock': 40}
        }
        
        # Trade parameters
        armington_sigma = {
            'agriculture': 2.0, 'livestock': 1.8, 'textiles': 3.0,
            'manufacturing': 2.5, 'services': 1.5, 'financial': 2.0
        }
        
        export_sigma = {
            'agriculture': 2.5, 'livestock': 2.0, 'textiles': 4.0,
            'manufacturing': 3.0, 'services': 2.0, 'financial': 2.5
        }
        
        # Tax rates
        tax_rates = {
            'income_tax': 0.15,
            'corporate_tax': 0.25,
            'vat': 0.15,
            'import_tariff': 0.12,
            'export_tax': 0.02
        }
        
        # Create parameters object
        self.parameters = CGEParameters(
            alpha=alpha,
            sigma=sigma,
            beta={sector: 1.0 for sector in self.sectors},
            gamma=gamma,
            subsistence=subsistence,
            armington_sigma=armington_sigma,
            export_sigma=export_sigma,
            tax_rates=tax_rates,
            depreciation_rate=0.05,
            population_growth=0.012,
            productivity_growth=0.02
        )
        
        # Update with config values if provided
        if self.config:
            self._update_parameters_from_config()
    
    def _update_parameters_from_config(self):
        """
        Update parameters from configuration file
        """
        config_params = self.config.get('parameters', {})
        
        for param_name, param_value in config_params.items():
            if hasattr(self.parameters, param_name):
                setattr(self.parameters, param_name, param_value)
    
    def _initialize_variables(self):
        """
        Initialize model variables
        """
        self.variables = {
            # Production variables
            'output': {sector: 100.0 for sector in self.sectors},
            'factor_demand': {
                sector: {factor: 10.0 for factor in self.factors}
                for sector in self.sectors
            },
            
            # Price variables
            'prices': {sector: 1.0 for sector in self.sectors},
            'factor_prices': {factor: 1.0 for factor in self.factors},
            'exchange_rate': 85.0,  # BDT per USD
            
            # Consumption variables
            'consumption': {
                hh: {sector: 10.0 for sector in self.sectors}
                for hh in self.household_types
            },
            
            # Trade variables
            'exports': {sector: 5.0 for sector in self.sectors},
            'imports': {sector: 5.0 for sector in self.sectors},
            
            # Income variables
            'household_income': {hh: 1000.0 for hh in self.household_types},
            'government_revenue': 500.0,
            
            # Investment and savings
            'investment': {sector: 20.0 for sector in self.sectors},
            'savings': {hh: 100.0 for hh in self.household_types}
        }
    
    def _setup_equations(self):
        """
        Set up model equations
        """
        self.equations = {
            'production': self._production_equations,
            'factor_demand': self._factor_demand_equations,
            'consumption': self._consumption_equations,
            'trade': self._trade_equations,
            'income': self._income_equations,
            'market_clearing': self._market_clearing_equations,
            'government': self._government_equations
        }
    
    def _production_equations(self, variables: Dict) -> Dict:
        """
        Production function equations (CES)
        """
        equations = {}
        
        for sector in self.sectors:
            # CES production function
            factor_inputs = variables['factor_demand'][sector]
            alpha = self.parameters.alpha[sector]
            sigma = self.parameters.sigma[sector]
            beta = self.parameters.beta[sector]
            
            # Calculate CES aggregate
            ces_sum = 0
            for factor in factor_inputs:
                if factor in alpha:
                    # Avoid division by zero when sigma is close to 1 or 0
                    if abs(sigma) < 1e-6:
                        sigma_safe = 1e-6 if sigma >= 0 else -1e-6
                    else:
                        sigma_safe = sigma
                    
                    if abs(sigma_safe - 1) < 1e-6:
                        # Cobb-Douglas case (sigma = 1)
                        ces_sum += alpha[factor] * np.log(factor_inputs[factor] + 1e-9)
                    else:
                        ces_sum += alpha[factor] * (factor_inputs[factor] ** ((sigma_safe - 1) / sigma_safe))
            
            if abs(sigma - 1) < 1e-6:
                # Cobb-Douglas case
                output = beta * np.exp(ces_sum)
            else:
                sigma_safe = sigma if abs(sigma - 1) > 1e-6 else (1 + 1e-6)
                output = beta * (ces_sum ** (sigma_safe / (sigma_safe - 1)))
            equations[f'output_{sector}'] = output - variables['output'][sector]
        
        return equations
    
    def _factor_demand_equations(self, variables: Dict) -> Dict:
        """
        Factor demand equations (first-order conditions)
        """
        equations = {}
        
        for sector in self.sectors:
            for factor in self.factors:
                if factor in self.parameters.alpha[sector]:
                    # Marginal productivity condition
                    alpha = self.parameters.alpha[sector][factor]
                    sigma = self.parameters.sigma[sector]
                    
                    factor_demand = variables['factor_demand'][sector][factor]
                    output = variables['output'][sector]
                    factor_price = variables['factor_prices'][factor]
                    output_price = variables['prices'][sector]
                    
                    # First-order condition: MP = factor price / output price
                    # Avoid division by zero when sigma is close to 0
                    if abs(sigma) < 1e-6:
                        sigma_safe = 1e-6 if sigma >= 0 else -1e-6
                    else:
                        sigma_safe = sigma
                    
                    mp = alpha * (output / (factor_demand + 1e-9)) ** (1 / sigma_safe)
                    
                    equations[f'factor_demand_{sector}_{factor}'] = (
                        mp * output_price - factor_price
                    )
        
        return equations
    
    def _consumption_equations(self, variables: Dict) -> Dict:
        """
        Household consumption equations (Linear Expenditure System)
        """
        equations = {}
        
        for hh in self.household_types:
            income = variables['household_income'][hh]
            
            # Calculate total subsistence expenditure
            subsistence_exp = 0
            if hh in self.parameters.subsistence:
                for sector, quantity in self.parameters.subsistence[hh].items():
                    if sector in variables['prices']:
                        subsistence_exp += quantity * variables['prices'][sector]
            
            # Supernumerary income
            super_income = max(0, income - subsistence_exp)
            
            for sector in self.sectors:
                # LES consumption function
                subsistence_qty = 0
                if (hh in self.parameters.subsistence and 
                    sector in self.parameters.subsistence[hh]):
                    subsistence_qty = self.parameters.subsistence[hh][sector]
                
                gamma = 0
                if (hh in self.parameters.gamma and 
                    sector in self.parameters.gamma[hh]):
                    gamma = self.parameters.gamma[hh][sector]
                
                price = variables['prices'][sector]
                
                consumption = (subsistence_qty + 
                             gamma * super_income / (price + 1e-9))
                
                equations[f'consumption_{hh}_{sector}'] = (
                    consumption - variables['consumption'][hh][sector]
                )
        
        return equations
    
    def _trade_equations(self, variables: Dict) -> Dict:
        """
        International trade equations (Armington and CET)
        """
        equations = {}
        
        for sector in self.sectors:
            if sector in self.parameters.armington_sigma:
                # Armington import demand
                sigma_m = self.parameters.armington_sigma[sector]
                
                # Import/domestic substitution
                domestic_price = variables['prices'][sector]
                import_price = variables['exchange_rate']  # Simplified
                
                # Armington demand ratio
                import_share = 0.3  # Calibrated parameter
                domestic_share = 1 - import_share
                
                armington_ratio = ((import_share / domestic_share) * 
                                 (domestic_price / import_price) ** sigma_m)
                
                equations[f'armington_{sector}'] = (
                    armington_ratio - variables['imports'][sector] / 
                    (variables['output'][sector] - variables['exports'][sector] + 1e-9)
                )
            
            if sector in self.parameters.export_sigma:
                # CET export supply
                sigma_e = self.parameters.export_sigma[sector]
                
                # Export/domestic transformation
                export_price = variables['exchange_rate']  # Simplified
                domestic_price = variables['prices'][sector]
                
                export_share = 0.2  # Calibrated parameter
                domestic_share = 1 - export_share
                
                cet_ratio = ((export_share / domestic_share) * 
                           (export_price / domestic_price) ** sigma_e)
                
                equations[f'cet_{sector}'] = (
                    cet_ratio - variables['exports'][sector] / 
                    (variables['output'][sector] - variables['exports'][sector] + 1e-9)
                )
        
        return equations
    
    def _income_equations(self, variables: Dict) -> Dict:
        """
        Income distribution equations
        """
        equations = {}
        
        # Factor income shares by household type
        factor_shares = {
            'rural_poor': {'labor_unskilled': 0.4, 'land': 0.1},
            'rural_nonpoor': {'labor_unskilled': 0.3, 'labor_skilled': 0.1, 'land': 0.3, 'capital': 0.1},
            'urban_poor': {'labor_unskilled': 0.5},
            'urban_nonpoor': {'labor_unskilled': 0.2, 'labor_skilled': 0.4, 'capital': 0.2},
            'urban_rich': {'labor_skilled': 0.3, 'capital': 0.6}
        }
        
        for hh in self.household_types:
            income = 0
            
            # Factor income
            if hh in factor_shares:
                for factor, share in factor_shares[hh].items():
                    if factor in variables['factor_prices']:
                        # Total factor income
                        total_factor_income = 0
                        for sector in self.sectors:
                            if factor in variables['factor_demand'][sector]:
                                total_factor_income += (
                                    variables['factor_prices'][factor] * 
                                    variables['factor_demand'][sector][factor]
                                )
                        
                        income += share * total_factor_income
            
            # Transfer income (simplified)
            if 'poor' in hh:
                income += 50  # Government transfers
            
            equations[f'income_{hh}'] = income - variables['household_income'][hh]
        
        return equations
    
    def _market_clearing_equations(self, variables: Dict) -> Dict:
        """
        Market clearing conditions
        """
        equations = {}
        
        # Goods market clearing
        for sector in self.sectors:
            supply = variables['output'][sector] + variables['imports'][sector]
            
            demand = variables['exports'][sector]
            
            # Household consumption
            for hh in self.household_types:
                demand += variables['consumption'][hh][sector]
            
            # Investment demand
            demand += variables['investment'][sector]
            
            # Government consumption (simplified)
            if sector == 'government':
                demand += 100  # Government consumption
            
            equations[f'market_clearing_{sector}'] = supply - demand
        
        # Factor market clearing
        for factor in self.factors:
            total_demand = 0
            for sector in self.sectors:
                if factor in variables['factor_demand'][sector]:
                    total_demand += variables['factor_demand'][sector][factor]
            
            # Factor supply (exogenous for now)
            factor_supply = {
                'labor_unskilled': 1000,
                'labor_skilled': 500,
                'capital': 800,
                'land': 200
            }
            
            equations[f'factor_clearing_{factor}'] = (
                factor_supply[factor] - total_demand
            )
        
        return equations
    
    def _government_equations(self, variables: Dict) -> Dict:
        """
        Government budget and fiscal equations
        """
        equations = {}
        
        # Government revenue
        revenue = 0
        
        # Tax revenue
        for hh in self.household_types:
            income_tax = (self.parameters.tax_rates['income_tax'] * 
                         variables['household_income'][hh])
            revenue += income_tax
        
        # VAT revenue
        for sector in self.sectors:
            for hh in self.household_types:
                vat = (self.parameters.tax_rates['vat'] * 
                      variables['prices'][sector] * 
                      variables['consumption'][hh][sector])
                revenue += vat
        
        # Trade taxes
        for sector in self.sectors:
            import_tariff = (self.parameters.tax_rates['import_tariff'] * 
                           variables['exchange_rate'] * 
                           variables['imports'][sector])
            export_tax = (self.parameters.tax_rates['export_tax'] * 
                         variables['exchange_rate'] * 
                         variables['exports'][sector])
            revenue += import_tariff + export_tax
        
        equations['government_revenue'] = revenue - variables['government_revenue']
        
        # Savings-investment balance
        total_savings = sum(variables['savings'][hh] for hh in self.household_types)
        total_investment = sum(variables['investment'][sector] for sector in self.sectors)
        equations['savings_investment_balance'] = total_savings - total_investment
        
        # External balance (current account)
        export_value = sum(variables['exports'][sector] * variables['exchange_rate'] for sector in self.sectors)
        import_value = sum(variables['imports'][sector] * variables['exchange_rate'] for sector in self.sectors)
        equations['external_balance'] = export_value - import_value
        
        # Price normalization (numeraire)
        equations['price_numeraire'] = variables['prices']['agriculture'] - 1.0
        
        # Exchange rate anchor (if needed)
        equations['exchange_rate_anchor'] = variables['exchange_rate'] - 85.0
        
        # Walras' law check (one redundant equation)
        equations['walras_law'] = sum(variables['household_income'][hh] for hh in self.household_types) - 5000.0
        
        return equations
    
    def calibrate(self, base_year: int = 2020) -> Dict:
        """
        Calibrate the model to base year data
        
        Args:
            base_year: Base year for calibration
            
        Returns:
            Calibration results
        """
        logger.info(f"Calibrating CGE model to base year {base_year}")
        
        if self.data is None:
            logger.warning("No data provided for calibration. Using default values.")
            return self._default_calibration()
        
        # Extract base year data
        base_data = self._extract_base_year_data(base_year)
        
        # Calibrate parameters to match base year
        calibration_results = self._calibrate_parameters(base_data)
        
        logger.info("Model calibration completed")
        return calibration_results
    
    def _default_calibration(self) -> Dict:
        """
        Default calibration when no data is available
        """
        # Use default parameter values
        return {
            'status': 'default',
            'message': 'Used default parameter values',
            'parameters': self.parameters
        }
    
    def _extract_base_year_data(self, base_year: int) -> Dict:
        """
        Extract base year data for calibration
        """
        # This would extract actual data for the base year
        # For now, return synthetic data
        return {
            'gdp_by_sector': {
                'agriculture': 15000, 'livestock': 3000, 'textiles': 25000,
                'manufacturing': 20000, 'construction': 8000, 'services': 30000,
                'government': 12000, 'financial': 7000
            },
            'employment_by_sector': {
                'agriculture': 400, 'livestock': 80, 'textiles': 300,
                'manufacturing': 200, 'construction': 100, 'services': 250,
                'government': 150, 'financial': 70
            },
            'trade_flows': {
                'exports': {'textiles': 15000, 'agriculture': 2000, 'manufacturing': 3000},
                'imports': {'manufacturing': 8000, 'services': 2000, 'agriculture': 1000}
            }
        }
    
    def _calibrate_parameters(self, base_data: Dict) -> Dict:
        """
        Calibrate model parameters to match base year data
        """
        # Calibrate scale parameters to match output levels
        for sector in self.sectors:
            if sector in base_data['gdp_by_sector']:
                target_output = base_data['gdp_by_sector'][sector]
                self.variables['output'][sector] = target_output
        
        # Calibrate other parameters as needed
        return {
            'status': 'calibrated',
            'base_year_data': base_data,
            'calibrated_parameters': self.parameters
        }
    
    def solve_baseline(self) -> Dict:
        """
        Solve the baseline equilibrium
        
        Returns:
            Baseline solution
        """
        logger.info("Solving baseline equilibrium")
        
        # Set up system of equations
        def equation_system(x):
            # Unpack variables
            variables = self._unpack_variables(x)
            
            # Compute all equations
            all_equations = {}
            for eq_type, eq_func in self.equations.items():
                equations = eq_func(variables)
                all_equations.update(equations)
            
            # Debug logging
            logger.info(f"Number of variables: {len(x)}")
            logger.info(f"Number of equations: {len(all_equations)}")
            
            # Return residuals
            return list(all_equations.values())
        
        # Initial guess
        x0 = self._pack_variables(self.variables)
        logger.info(f"Initial variable vector length: {len(x0)}")
        
        # Solve system
        try:
            solution = opt.fsolve(equation_system, x0, xtol=1e-8)
            
            # Unpack solution
            self.baseline = self._unpack_variables(solution)
            
            # Check convergence
            residuals = equation_system(solution)
            max_residual = max(abs(r) for r in residuals)
            
            if max_residual < 1e-6:
                status = 'converged'
            else:
                status = 'not_converged'
                logger.warning(f"Solution may not have converged. Max residual: {max_residual}")
            
            logger.info(f"Baseline solution completed with status: {status}")
            
            return {
                'status': status,
                'solution': self.baseline,
                'max_residual': max_residual,
                'iterations': 100  # Would be actual iterations from solver
            }
            
        except Exception as e:
            logger.error(f"Error solving baseline: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _pack_variables(self, variables: Dict) -> np.ndarray:
        """
        Pack variables dictionary into array for solver
        """
        x = []
        
        # Output variables
        for sector in self.sectors:
            x.append(variables['output'][sector])
        
        # Price variables
        for sector in self.sectors:
            x.append(variables['prices'][sector])
        
        # Factor prices
        for factor in self.factors:
            x.append(variables['factor_prices'][factor])
        
        # Factor demands
        for sector in self.sectors:
            for factor in self.factors:
                if factor in self.parameters.alpha[sector]:
                    x.append(variables['factor_demand'][sector][factor])
        
        # Consumption
        for hh in self.household_types:
            for sector in self.sectors:
                x.append(variables['consumption'][hh][sector])
        
        # Trade variables
        for sector in self.sectors:
            x.append(variables['exports'][sector])
            x.append(variables['imports'][sector])
        
        # Income variables
        for hh in self.household_types:
            x.append(variables['household_income'][hh])
        
        x.append(variables['government_revenue'])
        x.append(variables['exchange_rate'])
        
        return np.array(x)
    
    def _unpack_variables(self, x: np.ndarray) -> Dict:
        """
        Unpack array into variables dictionary
        """
        variables = {
            'output': {},
            'prices': {},
            'factor_prices': {},
            'factor_demand': {sector: {} for sector in self.sectors},
            'consumption': {hh: {} for hh in self.household_types},
            'exports': {},
            'imports': {},
            'household_income': {},
            'investment': {sector: 20.0 for sector in self.sectors},
            'savings': {hh: 100.0 for hh in self.household_types}
        }
        
        idx = 0
        
        # Output variables
        for sector in self.sectors:
            variables['output'][sector] = x[idx]
            idx += 1
        
        # Price variables
        for sector in self.sectors:
            variables['prices'][sector] = x[idx]
            idx += 1
        
        # Factor prices
        for factor in self.factors:
            variables['factor_prices'][factor] = x[idx]
            idx += 1
        
        # Factor demands
        for sector in self.sectors:
            for factor in self.factors:
                if factor in self.parameters.alpha[sector]:
                    variables['factor_demand'][sector][factor] = x[idx]
                    idx += 1
        
        # Consumption
        for hh in self.household_types:
            for sector in self.sectors:
                variables['consumption'][hh][sector] = x[idx]
                idx += 1
        
        # Trade variables
        for sector in self.sectors:
            variables['exports'][sector] = x[idx]
            idx += 1
            variables['imports'][sector] = x[idx]
            idx += 1
        
        # Income variables
        for hh in self.household_types:
            variables['household_income'][hh] = x[idx]
            idx += 1
        
        variables['government_revenue'] = x[idx]
        idx += 1
        variables['exchange_rate'] = x[idx]
        
        return variables
    
    def policy_simulation(self, policy_changes: Dict) -> Dict:
        """
        Run policy simulation
        
        Args:
            policy_changes: Dictionary of policy changes
            
        Returns:
            Policy simulation results
        """
        logger.info(f"Running policy simulation: {policy_changes}")
        
        if self.baseline is None:
            logger.error("No baseline solution available. Run solve_baseline() first.")
            return {'status': 'error', 'message': 'No baseline solution'}
        
        # Save original parameters
        original_params = self._copy_parameters()
        
        try:
            # Apply policy changes
            self._apply_policy_changes(policy_changes)
            
            # Solve new equilibrium
            policy_solution = self.solve_baseline()
            
            # Calculate changes from baseline
            changes = self._calculate_changes(self.baseline, policy_solution['solution'])
            
            # Restore original parameters
            self.parameters = original_params
            
            logger.info("Policy simulation completed")
            
            return {
                'status': 'completed',
                'policy_changes': policy_changes,
                'baseline': self.baseline,
                'policy_solution': policy_solution['solution'],
                'changes': changes,
                'welfare_effects': self._calculate_welfare_effects(changes)
            }
            
        except Exception as e:
            # Restore original parameters
            self.parameters = original_params
            logger.error(f"Error in policy simulation: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _copy_parameters(self) -> CGEParameters:
        """
        Create a deep copy of parameters
        """
        import copy
        return copy.deepcopy(self.parameters)
    
    def _apply_policy_changes(self, policy_changes: Dict):
        """
        Apply policy changes to model parameters
        """
        for policy, value in policy_changes.items():
            if policy == 'tariff_reduction':
                # Reduce import tariffs
                current_tariff = self.parameters.tax_rates['import_tariff']
                self.parameters.tax_rates['import_tariff'] = current_tariff * (1 - value)
            
            elif policy == 'export_subsidy':
                # Add export subsidy (negative export tax)
                self.parameters.tax_rates['export_tax'] = -value
            
            elif policy == 'vat_change':
                # Change VAT rate
                self.parameters.tax_rates['vat'] += value
            
            elif policy == 'income_tax_change':
                # Change income tax rate
                self.parameters.tax_rates['income_tax'] += value
            
            elif policy == 'productivity_shock':
                # Productivity shock to specific sectors
                if isinstance(value, dict):
                    for sector, shock in value.items():
                        if sector in self.parameters.beta:
                            self.parameters.beta[sector] *= (1 + shock)
                else:
                    # Apply to all sectors
                    for sector in self.parameters.beta:
                        self.parameters.beta[sector] *= (1 + value)
    
    def _calculate_changes(self, baseline: Dict, policy_solution: Dict) -> Dict:
        """
        Calculate percentage changes from baseline
        """
        changes = {}
        
        # GDP changes
        baseline_gdp = sum(baseline['output'].values())
        policy_gdp = sum(policy_solution['output'].values())
        changes['gdp_change'] = (policy_gdp - baseline_gdp) / baseline_gdp
        
        # Sectoral output changes
        changes['sectoral_changes'] = {}
        for sector in self.sectors:
            baseline_output = baseline['output'][sector]
            policy_output = policy_solution['output'][sector]
            changes['sectoral_changes'][sector] = (
                (policy_output - baseline_output) / baseline_output
            )
        
        # Trade changes
        baseline_exports = sum(baseline['exports'].values())
        policy_exports = sum(policy_solution['exports'].values())
        changes['export_change'] = (policy_exports - baseline_exports) / baseline_exports
        
        baseline_imports = sum(baseline['imports'].values())
        policy_imports = sum(policy_solution['imports'].values())
        changes['import_change'] = (policy_imports - baseline_imports) / baseline_imports
        
        # Price changes
        changes['price_changes'] = {}
        for sector in self.sectors:
            baseline_price = baseline['prices'][sector]
            policy_price = policy_solution['prices'][sector]
            changes['price_changes'][sector] = (
                (policy_price - baseline_price) / baseline_price
            )
        
        # Factor price changes
        changes['factor_price_changes'] = {}
        for factor in self.factors:
            baseline_price = baseline['factor_prices'][factor]
            policy_price = policy_solution['factor_prices'][factor]
            changes['factor_price_changes'][factor] = (
                (policy_price - baseline_price) / baseline_price
            )
        
        # Household income changes
        changes['income_changes'] = {}
        for hh in self.household_types:
            baseline_income = baseline['household_income'][hh]
            policy_income = policy_solution['household_income'][hh]
            changes['income_changes'][hh] = (
                (policy_income - baseline_income) / baseline_income
            )
        
        return changes
    
    def _calculate_welfare_effects(self, changes: Dict) -> Dict:
        """
        Calculate welfare effects using equivalent variation
        """
        welfare_effects = {}
        
        # Simple welfare approximation using income changes
        for hh in self.household_types:
            income_change = changes['income_changes'][hh]
            
            # Adjust for price changes (simplified)
            price_index_change = np.mean(list(changes['price_changes'].values()))
            real_income_change = income_change - price_index_change
            
            welfare_effects[hh] = real_income_change
        
        # Aggregate welfare change (weighted by population)
        population_weights = {
            'rural_poor': 0.3, 'rural_nonpoor': 0.2,
            'urban_poor': 0.2, 'urban_nonpoor': 0.2, 'urban_rich': 0.1
        }
        
        aggregate_welfare = sum(
            population_weights[hh] * welfare_effects[hh]
            for hh in self.household_types
        )
        
        welfare_effects['aggregate'] = aggregate_welfare
        
        return welfare_effects
    
    def trade_policy_analysis(self, trade_policy: Dict) -> Dict:
        """
        Analyze trade policy impacts
        
        Args:
            trade_policy: Trade policy parameters
            
        Returns:
            Trade policy analysis results
        """
        logger.info("Running trade policy analysis")
        
        # Convert trade policy to CGE policy changes
        policy_changes = {}
        
        if 'tariffs' in trade_policy:
            tariff_change = trade_policy['tariffs']
            if tariff_change == 0:
                # Free trade scenario
                policy_changes['tariff_reduction'] = 1.0  # 100% reduction
            else:
                # Set new tariff level
                current_tariff = self.parameters.tax_rates['import_tariff']
                policy_changes['tariff_reduction'] = 1 - (tariff_change / current_tariff)
        
        if 'export_subsidies' in trade_policy:
            policy_changes['export_subsidy'] = trade_policy['export_subsidies']
        
        # Run simulation
        results = self.policy_simulation(policy_changes)
        
        # Add trade-specific analysis
        if results['status'] == 'completed':
            results['trade_analysis'] = self._analyze_trade_effects(results['changes'])
        
        return results
    
    def _analyze_trade_effects(self, changes: Dict) -> Dict:
        """
        Analyze specific trade effects
        """
        trade_effects = {
            'trade_creation': {},
            'trade_diversion': {},
            'terms_of_trade': {},
            'competitiveness': {}
        }
        
        # Trade creation/diversion analysis
        for sector in self.sectors:
            export_change = changes['sectoral_changes'][sector]
            import_change = changes['sectoral_changes'][sector]  # Simplified
            
            if export_change > 0.05:  # 5% threshold
                trade_effects['trade_creation'][sector] = export_change
            
            if import_change > 0.05:
                trade_effects['trade_diversion'][sector] = import_change
        
        # Terms of trade
        export_price_change = np.mean([
            changes['price_changes'][sector] for sector in ['textiles', 'agriculture']
        ])
        import_price_change = np.mean([
            changes['price_changes'][sector] for sector in ['manufacturing', 'services']
        ])
        
        trade_effects['terms_of_trade']['change'] = export_price_change - import_price_change
        
        # Competitiveness indicators
        for sector in self.sectors:
            price_change = changes['price_changes'][sector]
            productivity_proxy = changes['sectoral_changes'][sector] - price_change
            trade_effects['competitiveness'][sector] = productivity_proxy
        
        return trade_effects
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'CGE',
            'country': 'Bangladesh',
            'sectors': self.sectors,
            'household_types': self.household_types,
            'factors': self.factors,
            'parameters': {
                'production_elasticities': self.parameters.sigma,
                'trade_elasticities': {
                    'armington': self.parameters.armington_sigma,
                    'export': self.parameters.export_sigma
                },
                'tax_rates': self.parameters.tax_rates
            }
        }
        
        if self.baseline:
            summary['baseline_solution'] = {
                'total_gdp': sum(self.baseline['output'].values()),
                'sectoral_shares': {
                    sector: self.baseline['output'][sector] / sum(self.baseline['output'].values())
                    for sector in self.sectors
                },
                'trade_balance': (
                    sum(self.baseline['exports'].values()) - 
                    sum(self.baseline['imports'].values())
                )
            }
        
        return summary
    
    def export_results(self, filepath: str, results: Dict):
        """
        Export results to file
        
        Args:
            filepath: Output file path
            results: Results dictionary to export
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        elif filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            # Default to pickle
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        
        logger.info(f"Results exported to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'parameters': {
            'sigma': {
                'textiles': 1.5,  # Higher elasticity for textiles
                'agriculture': 0.7  # Lower elasticity for agriculture
            }
        }
    }
    
    # Initialize model
    cge_model = CGEModel(config)
    
    # Calibrate model
    calibration = cge_model.calibrate(base_year=2020)
    print("Calibration status:", calibration['status'])
    
    # Solve baseline
    baseline = cge_model.solve_baseline()
    print("Baseline status:", baseline['status'])
    
    # Run policy simulation
    policy = {
        'tariff_reduction': 0.5,  # 50% tariff reduction
        'export_subsidy': 0.1     # 10% export subsidy
    }
    
    results = cge_model.policy_simulation(policy)
    print("Policy simulation status:", results['status'])
    
    if results['status'] == 'completed':
        print(f"GDP change: {results['changes']['gdp_change']:.2%}")
        print(f"Export change: {results['changes']['export_change']:.2%}")
        print(f"Welfare effects: {results['welfare_effects']}")
    
    # Get model summary
    summary = cge_model.get_model_summary()
    print("\nModel Summary:")
    print(f"Sectors: {len(summary['sectors'])}")
    print(f"Household types: {len(summary['household_types'])}")
    print(f"Factors: {len(summary['factors'])}")