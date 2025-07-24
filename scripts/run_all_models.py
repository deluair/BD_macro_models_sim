#!/usr/bin/env python3
"""
Bangladesh Economic Models Runner

This script demonstrates running all economic models with real Bangladesh data.
It loads the processed data and runs simulations with each model type.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import yaml
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import all available models
model_imports = {
    'DSGEModel': ('models.dsge.dsge_model', 'DSGEModel'),
    'SVARModel': ('models.svar.svar_model', 'SVARModel'),
    'CGEModel': ('models.cge.cge_model', 'CGEModel'),
    'RBCModel': ('models.rbc.rbc_model', 'RealBusinessCycleModel'),
    'SOEModel': ('models.small_open_economy.soe_model', 'SmallOpenEconomyModel'),
    'HANKModel': ('models.hank.hank_model', 'HANKModel'),
    'FinancialModel': ('models.financial.financial_model', 'FinancialSectorModel'),
    'ABMModel': ('models.abm.abm_model', 'AgentBasedModel'),
    'BehavioralModel': ('models.behavioral.behavioral_model', 'BehavioralEconomicsModel'),
    'GameTheoryModel': ('models.game_theory.game_theory_model', 'GameTheoreticModel'),
    'IAMModel': ('models.iam.iam_model', 'IntegratedAssessmentModel'),
    'NEGModel': ('models.neg.neg_model', 'NewEconomicGeographyModel'),
    'OLGModel': ('models.olg.olg_model', 'OverlappingGenerationsModel'),
    'QMMModel': ('models.qmm.qmm_model', 'QuarterlyMacroModel'),
    'SearchMatchingModel': ('models.search_matching.search_matching_model', 'SearchMatchingModel')
}

# Import models dynamically
available_models = {}
for model_name, (module_path, class_name) in model_imports.items():
    try:
        module = __import__(module_path, fromlist=[class_name])
        available_models[model_name] = getattr(module, class_name)
        print(f"✅ {model_name} imported successfully")
    except ImportError as e:
        print(f"⚠️  {model_name} not available: {e}")
        available_models[model_name] = None
    except AttributeError as e:
        print(f"⚠️  {model_name} class not found: {e}")
        available_models[model_name] = None

# Set individual model variables for backward compatibility
DSGEModel = available_models.get('DSGEModel')
SVARModel = available_models.get('SVARModel')
CGEModel = available_models.get('CGEModel')
RBCModel = available_models.get('RBCModel')
SOEModel = available_models.get('SOEModel')
HANKModel = available_models.get('HANKModel')
FinancialModel = available_models.get('FinancialModel')
ABMModel = available_models.get('ABMModel')
BehavioralModel = available_models.get('BehavioralModel')
GameTheoryModel = available_models.get('GameTheoryModel')
IAMModel = available_models.get('IAMModel')
NEGModel = available_models.get('NEGModel')
OLGModel = available_models.get('OLGModel')
QMMModel = available_models.get('QMMModel')
SearchMatchingModel = available_models.get('SearchMatchingModel')

class ModelRunner:
    """
    Runs all economic models with Bangladesh data.
    """
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_dir = self.project_root / "data" / "processed"
        self.results_dir = self.project_root / "results"
        self.viz_dir = self.project_root / "visualization" / "charts"
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.load_config()
        
        # Load data
        self.load_data()
    
    def load_config(self):
        """Load configuration from config.yaml file."""
        config_file = self.project_root / "config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_file}")
        else:
            print(f"⚠️  Config file not found: {config_file}")
            # Create default config
            self.config = {
                'dsge': {'calibration': {'beta': 0.99, 'sigma': 2.0, 'phi': 1.0, 'alpha': 0.33}},
                'svar': {'lags': 4, 'identification': 'cholesky'},
                'cge': {'sectors': ['agriculture', 'manufacturing', 'services']},
                'rbc': {'technology': {'capital_share': 0.33, 'depreciation': 0.025}},
                'hank': {'heterogeneity': {'income_states': 7, 'asset_grid_size': 100}},
                'olg': {'demographics': {'retirement_age': 60, 'life_expectancy': 72}},
                'behavioral': {'learning': {'adaptive_parameter': 0.1}},
                'game_theory': {'players': ['central_bank', 'government']},
                'iam': {'climate': {'carbon_sensitivity': 3.0}},
                'neg': {'regions': ['dhaka', 'chittagong', 'sylhet']},
                'qmm': {'frequency': 'quarterly'},
                'search_matching': {'labor_market': {'matching_elasticity': 0.5}},
                'financial': {'banking': {'capital_ratio': 0.1}},
                'abm': {'agents': {'households': 1000, 'firms': 100}}
            }
    
    def load_data(self):
        """Load Bangladesh macroeconomic data."""
        data_file = self.data_dir / "bangladesh_macroeconomic_data.csv"
        
        if data_file.exists():
            self.data = pd.read_csv(data_file)
            print(f"\n📊 Loaded Bangladesh data: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Load model-specific data
            model_exports_dir = self.data_dir / "model_exports"
            
            if (model_exports_dir / "bangladesh_dsge_data.csv").exists():
                self.dsge_data = pd.read_csv(model_exports_dir / "bangladesh_dsge_data.csv")
                print(f"✅ DSGE data loaded: {self.dsge_data.shape}")
            
            if (model_exports_dir / "bangladesh_svar_data.csv").exists():
                self.svar_data = pd.read_csv(model_exports_dir / "bangladesh_svar_data.csv")
                print(f"✅ SVAR data loaded: {self.svar_data.shape}")
            
            if (model_exports_dir / "bangladesh_cge_data.csv").exists():
                self.cge_data = pd.read_csv(model_exports_dir / "bangladesh_cge_data.csv")
                print(f"✅ CGE data loaded: {self.cge_data.shape}")
        else:
            print(f"❌ Data file not found: {data_file}")
            self.data = None
    
    def run_dsge_model(self):
        """Run DSGE model simulation."""
        print("\n" + "="*50)
        print("🏛️  RUNNING DSGE MODEL")
        print("="*50)
        
        if DSGEModel is None:
            print("❌ DSGE Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('dsge', {})
            
            # Initialize model with configuration and data
            data = getattr(self, 'dsge_data', None) if hasattr(self, 'dsge_data') else self.data
            model = DSGEModel(config, data)
            
            print("📋 Model Configuration:")
            for section, params in config.items():
                print(f"  {section}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        print(f"    {key}: {value}")
                elif isinstance(params, list):
                    for item in params:
                        print(f"    - {item}")
                else:
                    print(f"    {params}")
            
            # Run simulation
            print("\n🔄 Running DSGE simulation...")
            results = model.simulate(periods=100)
            
            if results is not None:
                print("✅ DSGE simulation completed")
                
                # Save results
                results_file = self.results_dir / "dsge_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                else:
                    results.to_csv(results_file, index=False)
                
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ DSGE simulation failed")
                return None
                
        except Exception as e:
            print(f"❌ DSGE model error: {e}")
            return None
    
    def run_svar_model(self):
        """Run SVAR model estimation."""
        print("\n" + "="*50)
        print("📈 RUNNING SVAR MODEL")
        print("="*50)
        
        if SVARModel is None:
            print("❌ SVAR Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('svar', {})
            
            # Prepare data for SVAR
            if hasattr(self, 'svar_data') and not self.svar_data.empty:
                # Use real data if available
                data_df = self.svar_data.drop('Year', axis=1, errors='ignore')
                print(f"📊 Using real Bangladesh data: {data_df.shape}")
            else:
                # Generate synthetic data for demonstration
                print("📊 Generating synthetic data for demonstration")
                T = 100
                data = np.random.randn(T, 3)  # GDP Growth, Inflation, Current Account
                data[:, 0] = np.cumsum(data[:, 0]) * 0.5 + 6  # GDP Growth around 6%
                data[:, 1] = np.cumsum(data[:, 1]) * 0.3 + 5  # Inflation around 5%
                data[:, 2] = np.cumsum(data[:, 2]) * 0.2 - 2  # Current Account around -2%
                data_df = pd.DataFrame(data, columns=['gdp_growth', 'inflation', 'current_account'])
            
            # Initialize model with configuration
            model = SVARModel(config)
            
            # Prepare data for SVAR
            model.prepare_data(data_df)
            
            print("📋 Model Configuration:")
            for section, params in config.items():
                print(f"  {section}: {params}")
            
            print("\n🔄 Estimating SVAR model...")
            results = model.estimate(lags=2)
            
            if results is not None:
                print("✅ SVAR estimation completed")
                
                # Save results
                results_file = self.results_dir / "svar_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ SVAR estimation failed")
                return None
                
        except Exception as e:
            print(f"❌ SVAR model error: {e}")
            return None
    
    def run_cge_model(self):
        """Run CGE model simulation."""
        print("\n" + "="*50)
        print("🌐 RUNNING CGE MODEL")
        print("="*50)
        
        if CGEModel is None:
            print("❌ CGE Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('cge', {})
            
            # Initialize model with configuration and data
            data = getattr(self, 'cge_data', None) if hasattr(self, 'cge_data') else self.data
            model = CGEModel(config, data)
            
            print("📋 Model Configuration:")
            for section, params in config.items():
                print(f"  {section}:")
                if isinstance(params, dict):
                    for key, value in params.items():
                        print(f"    {key}: {value}")
                elif isinstance(params, list):
                    for i, item in enumerate(params):
                        print(f"    [{i}]: {item}")
                else:
                    print(f"    {params}")
            
            print("\n🔄 Running CGE simulation...")
            results = model.solve_baseline()
            
            if results is not None:
                print("✅ CGE simulation completed")
                
                # Save results
                results_file = self.results_dir / "cge_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ CGE simulation failed")
                return None
                
        except Exception as e:
            print(f"❌ CGE model error: {e}")
            return None
    
    def run_rbc_model(self):
        """Run RBC model simulation."""
        print("\n" + "="*50)
        print("🔄 RUNNING RBC MODEL")
        print("="*50)
        
        if RBCModel is None:
            print("❌ RBC Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('rbc', {})
            
            # Initialize model with configuration
            model = RBCModel(config)
            
            print("📋 Model Configuration:")
            print(f"  config: {config}")
            
            print("\n🔄 Running RBC simulation...")
            results = model.simulate_model(periods=100, n_simulations=10)
            
            if results is not None:
                print("✅ RBC simulation completed")
                
                # Save results
                results_file = self.results_dir / "rbc_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ RBC simulation failed")
                return None
                
        except Exception as e:
            print(f"❌ RBC model error: {e}")
            return None
    
    def run_soe_model(self):
        """Run Small Open Economy model."""
        print("\n" + "="*50)
        print("🌍 RUNNING SMALL OPEN ECONOMY MODEL")
        print("="*50)
        
        if SOEModel is None:
            print("❌ SOE Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('soe', {})
            
            # Initialize model with configuration
            model = SOEModel(config)
            
            print("📋 Model Configuration:")
            print(f"  config: {config}")
            
            print("\n🔄 Running SOE simulation...")
            results = model.simulate_economy(periods=100)
            
            if results is not None:
                print("✅ SOE simulation completed")
                
                # Save results
                results_file = self.results_dir / "soe_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ SOE simulation failed")
                return None
                
        except Exception as e:
            print(f"❌ SOE model error: {e}")
            return None
    
    def run_hank_model(self):
        """Run HANK (Heterogeneous Agent New Keynesian) model."""
        print("\n" + "="*50)
        print("👥 RUNNING HANK MODEL")
        print("="*50)
        
        if HANKModel is None:
            print("❌ HANK Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('hank', {})
            model = HANKModel(config)
            print("🔄 Running HANK simulation...")
            results = model.simulate(periods=100)
            
            if results is not None:
                print("✅ HANK simulation completed")
                results_file = self.results_dir / "hank_results.csv"
                if isinstance(results, dict) and 'time_series' in results:
                    results['time_series'].to_csv(results_file, index=False)
                elif isinstance(results, dict):
                    # Fallback: try to create DataFrame from simple dict
                    try:
                        pd.DataFrame(results).to_csv(results_file, index=False)
                    except:
                        print("⚠️ Could not save results to CSV")
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ HANK simulation failed")
                return None
        except Exception as e:
            print(f"❌ HANK model error: {e}")
            return None
    
    def run_financial_model(self):
        """Run Financial Sector model."""
        print("\n" + "="*50)
        print("🏦 RUNNING FINANCIAL SECTOR MODEL")
        print("="*50)
        
        if FinancialModel is None:
            print("❌ Financial Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('financial', {})
            model = FinancialModel(config)
            print("🔄 Running Financial simulation...")
            results = model.simulate_financial_system(periods=100)
            
            if results is not None:
                print("✅ Financial simulation completed")
                results_file = self.results_dir / "financial_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ Financial simulation failed")
                return None
        except Exception as e:
            print(f"❌ Financial model error: {e}")
            return None
    
    def run_abm_model(self):
        """Run Agent-Based model."""
        print("\n" + "="*50)
        print("🤖 RUNNING AGENT-BASED MODEL")
        print("="*50)
        
        if ABMModel is None:
            print("❌ ABM Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('abm', {})
            model = ABMModel(config)
            print("🔄 Running ABM simulation...")
            results = model.simulate(periods=100)
            
            if results is not None:
                print("✅ ABM simulation completed")
                results_file = self.results_dir / "abm_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ ABM simulation failed")
                return None
        except Exception as e:
            print(f"❌ ABM model error: {e}")
            return None
    
    def run_behavioral_model(self):
        """Run Behavioral Economics model."""
        print("\n" + "="*50)
        print("🧠 RUNNING BEHAVIORAL ECONOMICS MODEL")
        print("="*50)
        
        if BehavioralModel is None:
            print("❌ Behavioral Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('behavioral', {})
            model = BehavioralModel(config)
            print("🔄 Running Behavioral simulation...")
            results = model.simulate_economy(periods=100)
            
            if results is not None:
                print("✅ Behavioral simulation completed")
                results_file = self.results_dir / "behavioral_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ Behavioral simulation failed")
                return None
        except Exception as e:
            print(f"❌ Behavioral model error: {e}")
            return None
    
    def run_game_theory_model(self):
        """Run Game Theory model."""
        print("\n" + "="*50)
        print("🎯 RUNNING GAME THEORY MODEL")
        print("="*50)
        
        if GameTheoryModel is None:
            print("❌ Game Theory Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('game_theory', {})
            model = GameTheoryModel(config)
            print("🔄 Running Game Theory simulation...")
            # Set up a simple game first
            model.setup_trade_negotiation_game()
            results = model.simulate_dynamic_game('trade_negotiation', periods=100)
            
            if results is not None:
                print("✅ Game Theory simulation completed")
                results_file = self.results_dir / "game_theory_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ Game Theory simulation failed")
                return None
        except Exception as e:
            print(f"❌ Game Theory model error: {e}")
            return None
    
    def run_iam_model(self):
        """Run Integrated Assessment model."""
        print("\n" + "="*50)
        print("🌍 RUNNING INTEGRATED ASSESSMENT MODEL")
        print("="*50)
        
        if IAMModel is None:
            print("❌ IAM Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('iam', {})
            model = IAMModel(config)
            print("🔄 Running IAM simulation...")
            results = model.run_baseline_scenario()
            
            if results is not None:
                print("✅ IAM simulation completed")
                results_file = self.results_dir / "iam_results.csv"
                if isinstance(results, dict):
                    pd.DataFrame(results).to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ IAM simulation failed")
                return None
        except Exception as e:
            print(f"❌ IAM model error: {e}")
            return None
    
    def run_neg_model(self):
        """Run New Economic Geography model."""
        print("\n" + "="*50)
        print("🗺️ RUNNING NEW ECONOMIC GEOGRAPHY MODEL")
        print("="*50)
        
        if NEGModel is None:
            print("❌ NEG Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('neg', {})
            model = NEGModel(config)
            print("🔄 Running NEG migration dynamics simulation...")
            results = model.simulate_migration_dynamics(periods=100)
            
            if results is not None:
                print("✅ NEG simulation completed")
                results_file = self.results_dir / "neg_results.csv"
                results.to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ NEG simulation failed")
                return None
        except Exception as e:
            print(f"❌ NEG model error: {e}")
            return None
    
    def run_olg_model(self):
        """Run Overlapping Generations model."""
        print("\n" + "="*50)
        print("👴👶 RUNNING OVERLAPPING GENERATIONS MODEL")
        print("="*50)
        
        if OLGModel is None:
            print("❌ OLG Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('olg', {})
            model = OLGModel(config)
            
            # Initialize the model properly
            print("🔄 Loading demographic data...")
            model.load_demographic_data()
            
            print("🔄 Setting up economic environment...")
            model.setup_economy()
            
            print("🔄 Finding OLG equilibrium...")
            results = model.find_equilibrium()
            
            if results is not None and results.get('convergence', False):
                print("✅ OLG equilibrium found")
                results_file = self.results_dir / "olg_results.csv"
                # Convert equilibrium results to DataFrame
                equilibrium_data = [{
                    'variable': key,
                    'value': value if isinstance(value, (int, float)) else str(value)
                } for key, value in results.get('prices', {}).items()]
                pd.DataFrame(equilibrium_data).to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ OLG equilibrium not found")
                return None
        except Exception as e:
            print(f"❌ OLG model error: {e}")
            return None
    
    def run_qmm_model(self):
        """Run Quarterly Macro model."""
        print("\n" + "="*50)
        print("📊 RUNNING QUARTERLY MACRO MODEL")
        print("="*50)
        
        if QMMModel is None:
            print("❌ QMM Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('qmm', {})
            model = QMMModel(config)
            print("🔄 Preparing data for QMM model...")
            
            # Prepare data with DatetimeIndex for QMM
            if self.data is not None:
                qmm_data = self.data.copy()
                # Create a quarterly date range
                start_date = '2010-01-01'
                periods = len(qmm_data)
                date_index = pd.date_range(start=start_date, periods=periods, freq='Q')
                qmm_data.index = date_index
                
                print("🔄 Loading data into QMM model...")
                model.load_data(qmm_data)
                print("🔄 Estimating QMM model...")
                estimation_results = model.estimate()
                print("🔄 Generating QMM forecasts...")
                results = model.forecast(horizon=8)
            else:
                print("❌ No data available for QMM model")
                return None
            
            if results is not None:
                print("✅ QMM forecasts completed")
                results_file = self.results_dir / "qmm_results.csv"
                results.to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ QMM forecasts failed")
                return None
        except Exception as e:
            print(f"❌ QMM model error: {e}")
            return None
    
    def run_search_matching_model(self):
        """Run Search and Matching model."""
        print("\n" + "="*50)
        print("🔍 RUNNING SEARCH AND MATCHING MODEL")
        print("="*50)
        
        if SearchMatchingModel is None:
            print("❌ Search Matching Model not available")
            return None
        
        try:
            # Use configuration from config.yaml
            config = self.config.get('search_matching', {})
            model = SearchMatchingModel(config)
            print("🔄 Running Search Matching simulation...")
            results = model.simulate_model(periods=100, n_simulations=10)
            
            if results is not None:
                print("✅ Search Matching simulation completed")
                results_file = self.results_dir / "search_matching_results.csv"
                results.to_csv(results_file, index=False)
                print(f"💾 Results saved to {results_file}")
                return results
            else:
                print("❌ Search Matching simulation failed")
                return None
        except Exception as e:
            print(f"❌ Search Matching model error: {e}")
            return None
    
    def run_all_models(self):
        """
        Run all available models.
        """
        print("\n🇧🇩 BANGLADESH ECONOMIC MODELS SIMULATION")
        print("=" * 60)
        
        results = {}
        
        # Define all model runners
        all_models = {
            'DSGE': (DSGEModel, self.run_dsge_model),
            'SVAR': (SVARModel, self.run_svar_model),
            'CGE': (CGEModel, self.run_cge_model),
            'RBC': (RBCModel, self.run_rbc_model),
            'SOE': (SOEModel, self.run_soe_model),
            'HANK': (HANKModel, self.run_hank_model),
            'Financial': (FinancialModel, self.run_financial_model),
            'ABM': (ABMModel, self.run_abm_model),
            'Behavioral': (BehavioralModel, self.run_behavioral_model),
            'GameTheory': (GameTheoryModel, self.run_game_theory_model),
            'IAM': (IAMModel, self.run_iam_model),
            'NEG': (NEGModel, self.run_neg_model),
            'OLG': (OLGModel, self.run_olg_model),
            'QMM': (QMMModel, self.run_qmm_model),
            'SearchMatching': (SearchMatchingModel, self.run_search_matching_model)
        }
        
        successful_runs = 0
        total_available = 0
        
        for model_name, (model_class, model_func) in all_models.items():
            if model_class is not None:
                total_available += 1
                try:
                    result = model_func()
                    if result is not None:
                        results[model_name] = result
                        successful_runs += 1
                except Exception as e:
                    print(f"❌ {model_name} model failed: {e}")
            else:
                print(f"⚠️  {model_name} model not available (import failed)")
        
        # Summary
        print("\n" + "="*60)
        print("📊 SIMULATION SUMMARY")
        print("="*60)
        print(f"📦 Total models available: {total_available}/{len(all_models)}")
        print(f"✅ Successful runs: {successful_runs}/{total_available}")
        print(f"📁 Results saved to: {self.results_dir}")
        
        if successful_runs > 0:
            print("\n🎯 Successfully executed models:")
            for model_name in results.keys():
                print(f"  - {model_name} Model")
        
        return results
    
    def create_comparison_chart(self, results):
        """Create a comparison chart of model results."""
        if not results:
            print("No results to compare")
            return
        
        print("\n📈 Creating model comparison chart...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simple comparison - just show which models ran successfully
        model_names = list(results.keys())
        success_rates = [1] * len(model_names)  # All successful models
        
        bars = ax.bar(model_names, success_rates, color=['blue', 'green', 'red', 'orange', 'purple'][:len(model_names)])
        
        ax.set_title('Bangladesh Economic Models - Successful Simulations', fontsize=16, fontweight='bold')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   '✅ Success', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_file = self.viz_dir / "model_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison chart saved to {chart_file}")
        
        plt.show()

def main():
    """Main function to run all models."""
    print("🇧🇩 Bangladesh Economic Models Runner")
    print("=" * 50)
    
    # Initialize runner
    runner = ModelRunner()
    
    if runner.data is None:
        print("❌ No data available. Run data_manager.py first.")
        return
    
    # Run all models
    results = runner.run_all_models()
    
    # Create comparison chart
    if results:
        runner.create_comparison_chart(results)
    
    print("\n🎉 All model simulations completed!")
    print("\n📋 Next steps:")
    print("  1. Check the results/ folder for detailed outputs")
    print("  2. Review the visualization/charts/ folder for plots")
    print("  3. Analyze model-specific results for policy insights")
    print("  4. Compare model predictions and assumptions")

if __name__ == "__main__":
    main()