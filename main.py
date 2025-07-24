#!/usr/bin/env python3
"""
Bangladesh Macroeconomic Models Simulation
Main entry point for running macroeconomic models and simulations

Author: Bangladesh Macro Models Team
Date: 2025
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import model modules
from models.dsge.dsge_model import DSGEModel
from models.cge.cge_model import CGEModel
from models.svar.svar_model import SVARModel
from models.qmm.qmm_model import QuarterlyMacroModel
from models.financial.financial_model import FinancialSectorModel
from models.olg.olg_model import OLGModel
from models.hank.hank_model import HANKModel
from models.game_theory.game_model import GameTheoryModel
from models.rbc.rbc_model import RBCModel
from models.neg.neg_model import NEGModel
from models.search_matching.search_model import SearchMatchingModel
from models.behavioral.behavioral_model import BehavioralModel
from models.abm.abm_model import AgentBasedModel
from models.iam.iam_model import IntegratedAssessmentModel
from models.small_open_economy.soe_model import SmallOpenEconomyModel

# Import utilities
from utils.data_processing.data_loader import DataLoader
from utils.calibration.calibrator import ModelCalibrator
from utils.estimation.estimator import ModelEstimator
from utils.optimization.optimizer import ModelOptimizer
from analysis.simulations.simulator import ModelSimulator
from analysis.forecasting.forecaster import ModelForecaster
from analysis.policy_analysis.policy_analyzer import PolicyAnalyzer
from visualization.dashboards.dashboard import MacroDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bangladesh_macro_models.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BangladeshMacroModels:
    """
    Main class for Bangladesh Macroeconomic Models Simulation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the macro models framework
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader = DataLoader(self.config)
        self.models = {}
        self.results = {}
        
        logger.info("Bangladesh Macroeconomic Models Framework Initialized")
    
    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def load_data(self) -> None:
        """
        Load and process economic data for Bangladesh
        """
        logger.info("Loading Bangladesh economic data...")
        
        try:
            # Load data from various sources
            self.data = self.data_loader.load_all_data()
            logger.info("Data loading completed successfully")
            
            # Display data summary
            self._display_data_summary()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _display_data_summary(self) -> None:
        """
        Display summary of loaded data
        """
        logger.info("=== DATA SUMMARY ===")
        for source, data in self.data.items():
            if hasattr(data, 'shape'):
                logger.info(f"{source}: {data.shape[0]} observations, {data.shape[1]} variables")
            else:
                logger.info(f"{source}: {len(data)} datasets")
    
    def initialize_models(self, model_list: Optional[List[str]] = None) -> None:
        """
        Initialize specified macroeconomic models
        
        Args:
            model_list: List of models to initialize. If None, initialize all models.
        """
        if model_list is None:
            model_list = [
                'dsge', 'cge', 'svar', 'qmm', 'financial', 'olg', 'hank',
                'game_theory', 'rbc', 'neg', 'search_matching', 'behavioral',
                'abm', 'iam', 'small_open_economy'
            ]
        
        logger.info(f"Initializing models: {', '.join(model_list)}")
        
        model_classes = {
            'dsge': DSGEModel,
            'cge': CGEModel,
            'svar': SVARModel,
            'qmm': QuarterlyMacroModel,
            'financial': FinancialSectorModel,
            'olg': OLGModel,
            'hank': HANKModel,
            'game_theory': GameTheoryModel,
            'rbc': RBCModel,
            'neg': NEGModel,
            'search_matching': SearchMatchingModel,
            'behavioral': BehavioralModel,
            'abm': AgentBasedModel,
            'iam': IntegratedAssessmentModel,
            'small_open_economy': SmallOpenEconomyModel
        }
        
        for model_name in model_list:
            if model_name in model_classes:
                try:
                    model_config = self.config.get(model_name, {})
                    self.models[model_name] = model_classes[model_name](
                        config=model_config,
                        data=self.data
                    )
                    logger.info(f"{model_name.upper()} model initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing {model_name} model: {e}")
            else:
                logger.warning(f"Unknown model: {model_name}")
    
    def calibrate_models(self, model_list: Optional[List[str]] = None) -> None:
        """
        Calibrate specified models
        
        Args:
            model_list: List of models to calibrate. If None, calibrate all initialized models.
        """
        if model_list is None:
            model_list = list(self.models.keys())
        
        logger.info(f"Calibrating models: {', '.join(model_list)}")
        
        calibrator = ModelCalibrator(self.config)
        
        for model_name in model_list:
            if model_name in self.models:
                try:
                    logger.info(f"Calibrating {model_name.upper()} model...")
                    calibrator.calibrate(self.models[model_name], model_name)
                    logger.info(f"{model_name.upper()} model calibrated successfully")
                except Exception as e:
                    logger.error(f"Error calibrating {model_name} model: {e}")
    
    def estimate_models(self, model_list: Optional[List[str]] = None) -> None:
        """
        Estimate specified models
        
        Args:
            model_list: List of models to estimate. If None, estimate all initialized models.
        """
        if model_list is None:
            model_list = list(self.models.keys())
        
        logger.info(f"Estimating models: {', '.join(model_list)}")
        
        estimator = ModelEstimator(self.config)
        
        for model_name in model_list:
            if model_name in self.models:
                try:
                    logger.info(f"Estimating {model_name.upper()} model...")
                    estimation_results = estimator.estimate(self.models[model_name], model_name)
                    self.results[f"{model_name}_estimation"] = estimation_results
                    logger.info(f"{model_name.upper()} model estimated successfully")
                except Exception as e:
                    logger.error(f"Error estimating {model_name} model: {e}")
    
    def run_simulations(self, model_list: Optional[List[str]] = None, 
                       scenarios: Optional[List[str]] = None) -> None:
        """
        Run simulations for specified models and scenarios
        
        Args:
            model_list: List of models to simulate
            scenarios: List of scenarios to simulate
        """
        if model_list is None:
            model_list = list(self.models.keys())
        
        if scenarios is None:
            scenarios = ['baseline', 'optimistic', 'pessimistic']
        
        logger.info(f"Running simulations for models: {', '.join(model_list)}")
        logger.info(f"Scenarios: {', '.join(scenarios)}")
        
        simulator = ModelSimulator(self.config)
        
        for model_name in model_list:
            if model_name in self.models:
                for scenario in scenarios:
                    try:
                        logger.info(f"Running {scenario} scenario for {model_name.upper()} model...")
                        simulation_results = simulator.simulate(
                            self.models[model_name], 
                            model_name, 
                            scenario
                        )
                        self.results[f"{model_name}_{scenario}"] = simulation_results
                        logger.info(f"Simulation completed: {model_name.upper()} - {scenario}")
                    except Exception as e:
                        logger.error(f"Error in simulation {model_name} - {scenario}: {e}")
    
    def generate_forecasts(self, model_list: Optional[List[str]] = None, 
                          horizon: int = 8) -> None:
        """
        Generate forecasts using specified models
        
        Args:
            model_list: List of models to use for forecasting
            horizon: Forecast horizon in periods
        """
        if model_list is None:
            model_list = list(self.models.keys())
        
        logger.info(f"Generating forecasts for models: {', '.join(model_list)}")
        logger.info(f"Forecast horizon: {horizon} periods")
        
        forecaster = ModelForecaster(self.config)
        
        for model_name in model_list:
            if model_name in self.models:
                try:
                    logger.info(f"Generating forecasts with {model_name.upper()} model...")
                    forecast_results = forecaster.forecast(
                        self.models[model_name], 
                        model_name, 
                        horizon
                    )
                    self.results[f"{model_name}_forecast"] = forecast_results
                    logger.info(f"Forecasts generated: {model_name.upper()}")
                except Exception as e:
                    logger.error(f"Error generating forecasts with {model_name} model: {e}")
    
    def analyze_policy(self, policy_experiments: List[Dict]) -> None:
        """
        Conduct policy analysis experiments
        
        Args:
            policy_experiments: List of policy experiment configurations
        """
        logger.info(f"Conducting {len(policy_experiments)} policy experiments")
        
        policy_analyzer = PolicyAnalyzer(self.config)
        
        for i, experiment in enumerate(policy_experiments):
            try:
                logger.info(f"Running policy experiment {i+1}: {experiment.get('name', 'Unnamed')}")
                policy_results = policy_analyzer.analyze(
                    self.models, 
                    experiment
                )
                self.results[f"policy_experiment_{i+1}"] = policy_results
                logger.info(f"Policy experiment {i+1} completed")
            except Exception as e:
                logger.error(f"Error in policy experiment {i+1}: {e}")
    
    def create_dashboard(self, port: int = 8050) -> None:
        """
        Create interactive dashboard for results visualization
        
        Args:
            port: Port number for dashboard server
        """
        logger.info(f"Creating interactive dashboard on port {port}")
        
        try:
            dashboard = MacroDashboard(self.config, self.results)
            dashboard.run(port=port, debug=False)
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save all results to files
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        
        for result_name, result_data in self.results.items():
            try:
                # Save as multiple formats
                if hasattr(result_data, 'to_csv'):
                    result_data.to_csv(output_path / f"{result_name}.csv")
                if hasattr(result_data, 'to_excel'):
                    result_data.to_excel(output_path / f"{result_name}.xlsx")
                
                logger.info(f"Saved: {result_name}")
            except Exception as e:
                logger.error(f"Error saving {result_name}: {e}")
        
        logger.info("All results saved successfully")
    
    def generate_report(self, output_file: str = "bangladesh_macro_report.pdf") -> None:
        """
        Generate comprehensive analysis report
        
        Args:
            output_file: Output file name for the report
        """
        logger.info(f"Generating comprehensive report: {output_file}")
        
        try:
            from utils.reporting.report_generator import ReportGenerator
            
            report_generator = ReportGenerator(self.config)
            report_generator.generate_report(
                models=self.models,
                results=self.results,
                output_file=output_file
            )
            
            logger.info(f"Report generated successfully: {output_file}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

def main():
    """
    Main function for command-line interface
    """
    parser = argparse.ArgumentParser(
        description="Bangladesh Macroeconomic Models Simulation"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--models', 
        nargs='+',
        help='List of models to run (default: all models)'
    )
    
    parser.add_argument(
        '--scenarios', 
        nargs='+',
        default=['baseline', 'optimistic', 'pessimistic'],
        help='List of scenarios to simulate'
    )
    
    parser.add_argument(
        '--forecast-horizon', 
        type=int, 
        default=8,
        help='Forecast horizon in periods'
    )
    
    parser.add_argument(
        '--dashboard-port', 
        type=int, 
        default=8050,
        help='Port for dashboard server'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--skip-estimation', 
        action='store_true',
        help='Skip model estimation step'
    )
    
    parser.add_argument(
        '--skip-simulation', 
        action='store_true',
        help='Skip simulation step'
    )
    
    parser.add_argument(
        '--skip-forecasting', 
        action='store_true',
        help='Skip forecasting step'
    )
    
    parser.add_argument(
        '--dashboard-only', 
        action='store_true',
        help='Only run dashboard (skip modeling)'
    )
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = BangladeshMacroModels(config_path=args.config)
    
    if not args.dashboard_only:
        # Load data
        framework.load_data()
        
        # Initialize models
        framework.initialize_models(model_list=args.models)
        
        # Calibrate models
        framework.calibrate_models(model_list=args.models)
        
        # Estimate models (if not skipped)
        if not args.skip_estimation:
            framework.estimate_models(model_list=args.models)
        
        # Run simulations (if not skipped)
        if not args.skip_simulation:
            framework.run_simulations(
                model_list=args.models, 
                scenarios=args.scenarios
            )
        
        # Generate forecasts (if not skipped)
        if not args.skip_forecasting:
            framework.generate_forecasts(
                model_list=args.models, 
                horizon=args.forecast_horizon
            )
        
        # Save results
        framework.save_results(output_dir=args.output_dir)
        
        # Generate report
        framework.generate_report()
    
    # Create dashboard
    framework.create_dashboard(port=args.dashboard_port)

if __name__ == "__main__":
    main()