#!/usr/bin/env python3
"""
Comprehensive Analysis Framework

This module provides a unified framework for running comprehensive economic analysis
across forecasting, policy analysis, simulations, and validation for Bangladesh
macroeconomic models.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import warnings
import json
from datetime import datetime
from tqdm import tqdm

# Add subdirectories to path
sys.path.append(str(Path(__file__).parent / 'forecasting'))
sys.path.append(str(Path(__file__).parent / 'policy_analysis'))
sys.path.append(str(Path(__file__).parent / 'simulations'))
sys.path.append(str(Path(__file__).parent / 'validation'))

# Import analysis modules
try:
    from comparative_forecasting import ComparativeForecastingAnalyzer
except ImportError:
    print("Warning: Comparative forecasting module not available")
    ComparativeForecastingAnalyzer = None

try:
    from policy_scenario_analyzer import PolicyScenarioAnalyzer
except ImportError:
    print("Warning: Policy scenario analyzer module not available")
    PolicyScenarioAnalyzer = None

try:
    from monte_carlo_simulator import MonteCarloSimulator
except ImportError:
    print("Warning: Monte Carlo simulator module not available")
    MonteCarloSimulator = None

try:
    from model_validator import ModelValidator
except ImportError:
    print("Warning: Model validator module not available")
    ModelValidator = None

class AnalysisFramework:
    """
    Comprehensive analysis framework for Bangladesh macroeconomic models.
    """
    
    def __init__(self, base_dir: str = ".", output_dir: str = "./comprehensive_analysis"):
        """
        Initialize the analysis framework.
        
        Args:
            base_dir: Base directory containing model results and data
            output_dir: Directory for saving comprehensive analysis outputs
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different analysis types
        self.forecasting_dir = self.output_dir / 'forecasting'
        self.policy_dir = self.output_dir / 'policy_analysis'
        self.simulation_dir = self.output_dir / 'simulations'
        self.validation_dir = self.output_dir / 'validation'
        self.reports_dir = self.output_dir / 'reports'
        
        for dir_path in [self.forecasting_dir, self.policy_dir, 
                        self.simulation_dir, self.validation_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize analysis modules
        self.forecasting_analyzer = None
        self.policy_analyzer = None
        self.monte_carlo_simulator = None
        self.model_validator = None
        
        self._initialize_modules()
        
        # Analysis configuration
        self.analysis_config = {
            'run_forecasting': True,
            'run_policy_analysis': True,
            'run_simulations': True,
            'run_validation': True,
            'create_summary_report': True,
            'generate_visualizations': True
        }
        
        # Key economic indicators for Bangladesh
        self.key_indicators = [
            'gdp_growth', 'inflation', 'unemployment', 'current_account',
            'government_debt', 'trade_balance', 'real_exchange_rate',
            'investment_rate', 'consumption_growth', 'exports_growth'
        ]
        
        # Analysis results storage
        self.analysis_results = {
            'forecasting': {},
            'policy_analysis': {},
            'simulations': {},
            'validation': {},
            'summary': {}
        }
    
    def _initialize_modules(self) -> None:
        """
        Initialize analysis modules with appropriate configurations.
        """
        try:
            if ComparativeForecastingAnalyzer:
                self.forecasting_analyzer = ComparativeForecastingAnalyzer(
                    results_dir=str(self.base_dir / 'results'),
                    output_dir=str(self.forecasting_dir)
                )
                print("✓ Forecasting analyzer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize forecasting analyzer: {e}")
        
        try:
            if PolicyScenarioAnalyzer:
                self.policy_analyzer = PolicyScenarioAnalyzer(
                    results_dir=str(self.base_dir / 'results'),
                    output_dir=str(self.policy_dir)
                )
                print("✓ Policy analyzer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize policy analyzer: {e}")
        
        try:
            if MonteCarloSimulator:
                self.monte_carlo_simulator = MonteCarloSimulator(
                    results_dir=str(self.base_dir / 'results'),
                    output_dir=str(self.simulation_dir)
                )
                print("✓ Monte Carlo simulator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Monte Carlo simulator: {e}")
        
        try:
            if ModelValidator:
                self.model_validator = ModelValidator(
                    results_dir=str(self.base_dir / 'results'),
                    data_dir=str(self.base_dir / 'data'),
                    output_dir=str(self.validation_dir)
                )
                print("✓ Model validator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize model validator: {e}")
    
    def load_model_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available model results.
        
        Returns:
            Dictionary of model results DataFrames
        """
        results_dir = self.base_dir / 'results'
        results = {}
        
        if not results_dir.exists():
            print(f"Warning: Results directory {results_dir} does not exist")
            return results
        
        for file_path in results_dir.glob("*.csv"):
            model_name = file_path.stem.replace('_results', '')
            try:
                df = pd.read_csv(file_path)
                results[model_name] = df
                print(f"Loaded {model_name}: {df.shape}")
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        return results
    
    def run_forecasting_analysis(self) -> Dict[str, any]:
        """
        Run comprehensive forecasting analysis.
        
        Returns:
            Dictionary with forecasting analysis results
        """
        print("\n" + "="*60)
        print("RUNNING FORECASTING ANALYSIS")
        print("="*60)
        
        if not self.forecasting_analyzer:
            print("Forecasting analyzer not available. Skipping...")
            return {}
        
        try:
            # Run the complete forecasting analysis
            report_path = self.forecasting_analyzer.run_complete_analysis()
            
            results = {
                'status': 'completed',
                'report_path': report_path,
                'output_directory': str(self.forecasting_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Forecasting analysis completed successfully")
            print(f"  Report: {report_path}")
            
            return results
            
        except Exception as e:
            print(f"✗ Forecasting analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_policy_analysis(self) -> Dict[str, any]:
        """
        Run comprehensive policy scenario analysis.
        
        Returns:
            Dictionary with policy analysis results
        """
        print("\n" + "="*60)
        print("RUNNING POLICY SCENARIO ANALYSIS")
        print("="*60)
        
        if not self.policy_analyzer:
            print("Policy analyzer not available. Skipping...")
            return {}
        
        try:
            # Run the complete policy analysis
            report_path = self.policy_analyzer.run_complete_analysis()
            
            results = {
                'status': 'completed',
                'report_path': report_path,
                'output_directory': str(self.policy_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Policy analysis completed successfully")
            print(f"  Report: {report_path}")
            
            return results
            
        except Exception as e:
            print(f"✗ Policy analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_simulation_analysis(self) -> Dict[str, any]:
        """
        Run comprehensive Monte Carlo simulation analysis.
        
        Returns:
            Dictionary with simulation analysis results
        """
        print("\n" + "="*60)
        print("RUNNING MONTE CARLO SIMULATION ANALYSIS")
        print("="*60)
        
        if not self.monte_carlo_simulator:
            print("Monte Carlo simulator not available. Skipping...")
            return {}
        
        try:
            # Run the complete simulation analysis
            report_path = self.monte_carlo_simulator.run_complete_simulation()
            
            results = {
                'status': 'completed',
                'report_path': report_path,
                'output_directory': str(self.simulation_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Simulation analysis completed successfully")
            print(f"  Report: {report_path}")
            
            return results
            
        except Exception as e:
            print(f"✗ Simulation analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_validation_analysis(self) -> Dict[str, any]:
        """
        Run comprehensive model validation analysis.
        
        Returns:
            Dictionary with validation analysis results
        """
        print("\n" + "="*60)
        print("RUNNING MODEL VALIDATION ANALYSIS")
        print("="*60)
        
        if not self.model_validator:
            print("Model validator not available. Skipping...")
            return {}
        
        try:
            # Run the complete validation analysis
            report_path = self.model_validator.run_complete_validation()
            
            results = {
                'status': 'completed',
                'report_path': report_path,
                'output_directory': str(self.validation_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Validation analysis completed successfully")
            print(f"  Report: {report_path}")
            
            return results
            
        except Exception as e:
            print(f"✗ Validation analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def create_summary_dashboard(self) -> str:
        """
        Create a comprehensive summary dashboard.
        
        Returns:
            Path to the summary dashboard
        """
        print("\n" + "="*60)
        print("CREATING SUMMARY DASHBOARD")
        print("="*60)
        
        # Load model results for summary statistics
        model_results = self.load_model_results()
        
        # Create summary visualizations
        self._create_summary_visualizations(model_results)
        
        # Generate summary report
        dashboard_path = self._generate_summary_report(model_results)
        
        print(f"✓ Summary dashboard created: {dashboard_path}")
        return dashboard_path
    
    def _create_summary_visualizations(self, model_results: Dict[str, pd.DataFrame]) -> None:
        """
        Create summary visualizations across all models.
        
        Args:
            model_results: Dictionary of model results
        """
        plt.style.use('seaborn-v0_8')
        
        # 1. Model comparison overview
        self._plot_model_overview(model_results)
        
        # 2. Key indicators summary
        self._plot_indicators_summary(model_results)
        
        # 3. Analysis modules status
        self._plot_analysis_status()
        
        # 4. Performance metrics comparison
        self._plot_performance_comparison(model_results)
    
    def _plot_model_overview(self, model_results: Dict[str, pd.DataFrame]) -> None:
        """
        Plot model overview comparison.
        """
        if not model_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Model data availability
        model_sizes = {name: len(df) for name, df in model_results.items()}
        axes[0, 0].bar(model_sizes.keys(), model_sizes.values())
        axes[0, 0].set_title('Model Data Availability')
        axes[0, 0].set_ylabel('Number of Records')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Variable coverage
        variable_coverage = {}
        for name, df in model_results.items():
            coverage = sum(1 for var in self.key_indicators if var in df.columns)
            variable_coverage[name] = coverage
        
        axes[0, 1].bar(variable_coverage.keys(), variable_coverage.values())
        axes[0, 1].set_title('Variable Coverage')
        axes[0, 1].set_ylabel('Number of Key Variables')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model types distribution
        model_types = {
            'Structural': ['dsge', 'svar', 'rbc'],
            'Equilibrium': ['cge', 'olg'],
            'Behavioral': ['behavioral', 'abm'],
            'Financial': ['financial', 'hank'],
            'Other': ['neg', 'qmm', 'iam', 'game_theory', 'search_matching', 'soe']
        }
        
        type_counts = {}
        for type_name, models in model_types.items():
            count = sum(1 for model in models if model in model_results)
            type_counts[type_name] = count
        
        axes[1, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Model Types Distribution')
        
        # Analysis modules status
        module_status = {
            'Forecasting': self.forecasting_analyzer is not None,
            'Policy Analysis': self.policy_analyzer is not None,
            'Simulations': self.monte_carlo_simulator is not None,
            'Validation': self.model_validator is not None
        }
        
        colors = ['green' if status else 'red' for status in module_status.values()]
        axes[1, 1].bar(module_status.keys(), [1 if status else 0 for status in module_status.values()], color=colors)
        axes[1, 1].set_title('Analysis Modules Status')
        axes[1, 1].set_ylabel('Available (1=Yes, 0=No)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'model_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_indicators_summary(self, model_results: Dict[str, pd.DataFrame]) -> None:
        """
        Plot key indicators summary across models.
        """
        if not model_results:
            return
        
        # Collect indicator data
        indicator_data = {indicator: [] for indicator in self.key_indicators}
        model_names = []
        
        for model_name, df in model_results.items():
            model_names.append(model_name)
            for indicator in self.key_indicators:
                if indicator in df.columns:
                    # Use first non-null value as representative
                    value = df[indicator].dropna().iloc[0] if not df[indicator].dropna().empty else 0
                    indicator_data[indicator].append(value)
                else:
                    indicator_data[indicator].append(0)
        
        # Create heatmap
        indicator_matrix = np.array([indicator_data[ind] for ind in self.key_indicators])
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(indicator_matrix, 
                   xticklabels=model_names,
                   yticklabels=self.key_indicators,
                   annot=True, fmt='.1f', cmap='RdYlBu_r')
        plt.title('Key Economic Indicators Across Models')
        plt.xlabel('Economic Models')
        plt.ylabel('Key Indicators')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'indicators_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_analysis_status(self) -> None:
        """
        Plot analysis execution status.
        """
        analysis_types = ['Forecasting', 'Policy Analysis', 'Simulations', 'Validation']
        status_data = []
        
        for analysis_type in analysis_types:
            key = analysis_type.lower().replace(' ', '_')
            if key in self.analysis_results:
                status = self.analysis_results[key].get('status', 'not_run')
                status_data.append(1 if status == 'completed' else 0)
            else:
                status_data.append(0)
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if status else 'red' for status in status_data]
        bars = plt.bar(analysis_types, status_data, color=colors, alpha=0.7)
        
        plt.title('Analysis Execution Status')
        plt.ylabel('Completed (1=Yes, 0=No)')
        plt.ylim(0, 1.2)
        
        # Add status labels
        for bar, status in zip(bars, status_data):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    'Completed' if status else 'Not Run',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'analysis_status.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, model_results: Dict[str, pd.DataFrame]) -> None:
        """
        Plot performance comparison across models.
        """
        if not model_results:
            return
        
        # Calculate simple performance metrics
        performance_data = []
        
        for model_name, df in model_results.items():
            # Calculate basic statistics for available indicators
            model_stats = {
                'model': model_name,
                'data_points': len(df),
                'variables_count': len([col for col in df.columns if col in self.key_indicators]),
                'completeness': df.notna().mean().mean() * 100
            }
            
            # Add indicator-specific metrics
            for indicator in ['gdp_growth', 'inflation', 'unemployment']:
                if indicator in df.columns:
                    values = df[indicator].dropna()
                    if len(values) > 0:
                        model_stats[f'{indicator}_mean'] = values.mean()
                        model_stats[f'{indicator}_std'] = values.std()
                    else:
                        model_stats[f'{indicator}_mean'] = 0
                        model_stats[f'{indicator}_std'] = 0
                else:
                    model_stats[f'{indicator}_mean'] = 0
                    model_stats[f'{indicator}_std'] = 0
            
            performance_data.append(model_stats)
        
        if not performance_data:
            return
        
        performance_df = pd.DataFrame(performance_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Data completeness
        axes[0, 0].bar(performance_df['model'], performance_df['completeness'])
        axes[0, 0].set_title('Data Completeness by Model')
        axes[0, 0].set_ylabel('Completeness (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Variable count
        axes[0, 1].bar(performance_df['model'], performance_df['variables_count'])
        axes[0, 1].set_title('Key Variables Count by Model')
        axes[0, 1].set_ylabel('Number of Variables')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # GDP growth comparison
        if 'gdp_growth_mean' in performance_df.columns:
            axes[1, 0].bar(performance_df['model'], performance_df['gdp_growth_mean'])
            axes[1, 0].set_title('Average GDP Growth by Model')
            axes[1, 0].set_ylabel('GDP Growth (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Inflation comparison
        if 'inflation_mean' in performance_df.columns:
            axes[1, 1].bar(performance_df['model'], performance_df['inflation_mean'])
            axes[1, 1].set_title('Average Inflation by Model')
            axes[1, 1].set_ylabel('Inflation (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, model_results: Dict[str, pd.DataFrame]) -> str:
        """
        Generate comprehensive summary report.
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Path to generated report
        """
        report_content = f"""
# Comprehensive Economic Analysis Framework Report
## Bangladesh Macroeconomic Models Analysis

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## Executive Summary

This report presents a comprehensive analysis of {len(model_results)} economic models for Bangladesh, covering forecasting, policy analysis, Monte Carlo simulations, and model validation. The analysis framework provides insights into model performance, policy implications, and risk assessments.

### Key Findings

- **Models Analyzed**: {len(model_results)} economic models
- **Key Variables**: {len(self.key_indicators)} macroeconomic indicators
- **Analysis Modules**: 4 comprehensive analysis frameworks
- **Output Files**: Multiple reports, visualizations, and data exports

---

## Model Portfolio Overview

### Available Models

"""
        
        # Add model details
        model_categories = {
            'Structural Models': ['dsge', 'svar', 'rbc'],
            'Equilibrium Models': ['cge', 'olg'],
            'Behavioral Models': ['behavioral', 'abm'],
            'Financial Models': ['financial', 'hank'],
            'Specialized Models': ['neg', 'qmm', 'iam', 'game_theory', 'search_matching', 'soe']
        }
        
        for category, models in model_categories.items():
            available_models = [model for model in models if model in model_results]
            if available_models:
                report_content += f"""
#### {category}
"""
                for model in available_models:
                    df = model_results[model]
                    report_content += f"- **{model.upper()}**: {len(df)} data points, {len([col for col in df.columns if col in self.key_indicators])} key variables\n"
        
        # Add analysis results summary
        report_content += """

---

## Analysis Results Summary

"""
        
        # Forecasting results
        if 'forecasting' in self.analysis_results and self.analysis_results['forecasting']:
            forecasting_result = self.analysis_results['forecasting']
            status = forecasting_result.get('status', 'not_run')
            report_content += f"""
### 1. Forecasting Analysis
- **Status**: {status.title()}
- **Output Directory**: `{self.forecasting_dir.name}/`
"""
            if status == 'completed':
                report_content += f"- **Report**: Available in output directory\n"
            elif status == 'failed':
                report_content += f"- **Error**: {forecasting_result.get('error', 'Unknown error')}\n"
        
        # Policy analysis results
        if 'policy_analysis' in self.analysis_results and self.analysis_results['policy_analysis']:
            policy_result = self.analysis_results['policy_analysis']
            status = policy_result.get('status', 'not_run')
            report_content += f"""
### 2. Policy Scenario Analysis
- **Status**: {status.title()}
- **Output Directory**: `{self.policy_dir.name}/`
"""
            if status == 'completed':
                report_content += f"- **Report**: Available in output directory\n"
            elif status == 'failed':
                report_content += f"- **Error**: {policy_result.get('error', 'Unknown error')}\n"
        
        # Simulation results
        if 'simulations' in self.analysis_results and self.analysis_results['simulations']:
            simulation_result = self.analysis_results['simulations']
            status = simulation_result.get('status', 'not_run')
            report_content += f"""
### 3. Monte Carlo Simulations
- **Status**: {status.title()}
- **Output Directory**: `{self.simulation_dir.name}/`
"""
            if status == 'completed':
                report_content += f"- **Report**: Available in output directory\n"
            elif status == 'failed':
                report_content += f"- **Error**: {simulation_result.get('error', 'Unknown error')}\n"
        
        # Validation results
        if 'validation' in self.analysis_results and self.analysis_results['validation']:
            validation_result = self.analysis_results['validation']
            status = validation_result.get('status', 'not_run')
            report_content += f"""
### 4. Model Validation
- **Status**: {status.title()}
- **Output Directory**: `{self.validation_dir.name}/`
"""
            if status == 'completed':
                report_content += f"- **Report**: Available in output directory\n"
            elif status == 'failed':
                report_content += f"- **Error**: {validation_result.get('error', 'Unknown error')}\n"
        
        # Add key indicators analysis
        report_content += f"""

---

## Key Economic Indicators Analysis

### Bangladesh Macroeconomic Indicators

"""
        
        # Calculate indicator statistics
        indicator_stats = {}
        for indicator in self.key_indicators:
            values = []
            for model_name, df in model_results.items():
                if indicator in df.columns:
                    indicator_values = df[indicator].dropna()
                    if len(indicator_values) > 0:
                        values.extend(indicator_values.tolist())
            
            if values:
                indicator_stats[indicator] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'models_count': sum(1 for _, df in model_results.items() if indicator in df.columns)
                }
        
        if indicator_stats:
            report_content += "| Indicator | Mean | Std Dev | Min | Max | Models |\n"
            report_content += "|-----------|------|---------|-----|-----|--------|\n"
            
            for indicator, stats in indicator_stats.items():
                report_content += f"| {indicator.replace('_', ' ').title()} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['models_count']} |\n"
        
        # Add recommendations
        report_content += """

---

## Recommendations

### Model Usage Guidelines

1. **Short-term Forecasting (1-4 quarters)**
   - Primary: SVAR models for dynamic relationships
   - Secondary: Financial models for market conditions
   - Validation: Compare with behavioral models

2. **Medium-term Analysis (1-3 years)**
   - Primary: DSGE models for structural relationships
   - Secondary: RBC models for technology effects
   - Policy: CGE models for sectoral impacts

3. **Long-term Planning (3+ years)**
   - Primary: CGE models for structural transformation
   - Secondary: OLG models for demographic effects
   - Climate: IAM models for environmental impacts

4. **Policy Analysis**
   - Monetary Policy: DSGE and SVAR models
   - Fiscal Policy: CGE and DSGE models
   - Financial Stability: Financial and HANK models
   - Trade Policy: CGE and SOE models

### Risk Management

1. **Model Risk**: Use ensemble approaches combining multiple models
2. **Parameter Uncertainty**: Regular Monte Carlo simulations
3. **Structural Breaks**: Continuous model validation and updating
4. **External Shocks**: Stress testing with extreme scenarios

### Implementation Strategy

1. **Regular Updates**: Quarterly model re-estimation and validation
2. **Data Quality**: Continuous improvement of data sources
3. **Model Development**: Ongoing enhancement of model specifications
4. **Capacity Building**: Training for model users and policymakers

---

## Technical Documentation

### Analysis Framework Components

1. **Forecasting Module**: Comparative forecasting across models
2. **Policy Analysis Module**: Scenario analysis and policy simulation
3. **Simulation Module**: Monte Carlo uncertainty analysis
4. **Validation Module**: Statistical testing and backtesting

### Output Structure

```
comprehensive_analysis/
├── forecasting/          # Forecasting analysis outputs
├── policy_analysis/      # Policy scenario outputs
├── simulations/          # Monte Carlo simulation outputs
├── validation/           # Model validation outputs
└── reports/              # Summary reports and visualizations
```

### Data Requirements

- **Model Results**: CSV files with model outputs
- **Historical Data**: Time series data for validation
- **Configuration**: JSON files with analysis parameters

---

*This analysis framework provides a comprehensive foundation for evidence-based economic policy making in Bangladesh. Regular updates and continuous improvement ensure the framework remains relevant and accurate.*
"""
        
        # Save report
        report_path = self.reports_dir / 'comprehensive_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def export_analysis_summary(self) -> None:
        """
        Export comprehensive analysis summary to various formats.
        """
        # Export analysis configuration
        config_data = {
            'framework_version': '1.0.0',
            'analysis_timestamp': datetime.now().isoformat(),
            'base_directory': str(self.base_dir),
            'output_directory': str(self.output_dir),
            'analysis_configuration': self.analysis_config,
            'key_indicators': self.key_indicators,
            'analysis_results': self.analysis_results
        }
        
        with open(self.reports_dir / 'analysis_configuration.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Export analysis summary
        summary_data = []
        for analysis_type, results in self.analysis_results.items():
            if results:
                summary_data.append({
                    'analysis_type': analysis_type,
                    'status': results.get('status', 'unknown'),
                    'timestamp': results.get('timestamp', ''),
                    'output_directory': results.get('output_directory', ''),
                    'report_available': 'report_path' in results
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.reports_dir / 'analysis_summary.csv', index=False)
    
    def run_complete_analysis(self, config: Optional[Dict[str, bool]] = None) -> str:
        """
        Run the complete comprehensive analysis framework.
        
        Args:
            config: Optional configuration dictionary to override defaults
            
        Returns:
            Path to the comprehensive analysis report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ECONOMIC ANALYSIS FRAMEWORK")
        print("Bangladesh Macroeconomic Models Analysis")
        print("="*80)
        
        # Update configuration if provided
        if config:
            self.analysis_config.update(config)
        
        # Load model results
        print("\nLoading model results...")
        model_results = self.load_model_results()
        
        if not model_results:
            print("Warning: No model results found. Some analyses may not be available.")
        
        # Run individual analyses based on configuration
        if self.analysis_config.get('run_forecasting', True):
            self.analysis_results['forecasting'] = self.run_forecasting_analysis()
        
        if self.analysis_config.get('run_policy_analysis', True):
            self.analysis_results['policy_analysis'] = self.run_policy_analysis()
        
        if self.analysis_config.get('run_simulations', True):
            self.analysis_results['simulations'] = self.run_simulation_analysis()
        
        if self.analysis_config.get('run_validation', True):
            self.analysis_results['validation'] = self.run_validation_analysis()
        
        # Create summary dashboard
        if self.analysis_config.get('create_summary_report', True):
            dashboard_path = self.create_summary_dashboard()
        else:
            dashboard_path = "Summary dashboard not created"
        
        # Export analysis summary
        self.export_analysis_summary()
        
        # Final summary
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETED")
        print("="*80)
        
        completed_analyses = sum(1 for result in self.analysis_results.values() 
                               if result and result.get('status') == 'completed')
        total_analyses = len([k for k, v in self.analysis_config.items() 
                            if k.startswith('run_') and v])
        
        print(f"\n✓ Analysis Summary:")
        print(f"  - Completed: {completed_analyses}/{total_analyses} analysis modules")
        print(f"  - Models Analyzed: {len(model_results)}")
        print(f"  - Output Directory: {self.output_dir}")
        print(f"  - Summary Report: {dashboard_path}")
        
        print(f"\n📊 Output Structure:")
        for subdir in [self.forecasting_dir, self.policy_dir, self.simulation_dir, 
                      self.validation_dir, self.reports_dir]:
            file_count = len(list(subdir.glob('*'))) if subdir.exists() else 0
            print(f"  - {subdir.name}/: {file_count} files")
        
        return dashboard_path


if __name__ == "__main__":
    # Run the comprehensive analysis framework
    # Set base directory to parent directory to access results
    base_dir = Path(__file__).parent.parent
    framework = AnalysisFramework(base_dir=str(base_dir))
    
    # Optional: Customize analysis configuration
    custom_config = {
        'run_forecasting': True,
        'run_policy_analysis': True,
        'run_simulations': True,
        'run_validation': True,
        'create_summary_report': True
    }
    
    # Run complete analysis
    report_path = framework.run_complete_analysis(custom_config)
    
    print(f"\n🎉 Comprehensive analysis framework completed successfully!")
    print(f"📋 Main report: {report_path}")
    print(f"📁 All outputs: {framework.output_dir}")