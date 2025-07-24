#!/usr/bin/env python3
"""
Example Analysis: Bangladesh Macroeconomic Models

This script demonstrates how to use the Bangladesh macroeconomic modeling framework
to analyze the economy using various models with real data.

Example analyses:
1. DSGE model simulation and policy analysis
2. Data loading and processing
3. Model comparison
4. Forecasting
5. Policy scenario analysis

Author: Bangladesh Macro Models Team
Date: 2025
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import framework modules
from utils.data_processing.data_loader import BangladeshDataLoader
from models.dsge.dsge_model import DSGEModel
import yaml

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_config():
    """
    Load configuration file
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def example_1_data_loading():
    """
    Example 1: Load and explore Bangladesh economic data
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: DATA LOADING AND EXPLORATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Initialize data loader
    data_loader = BangladeshDataLoader(config)
    
    # Load all data
    print("Loading Bangladesh economic data...")
    all_data = data_loader.load_all_data()
    
    # Display data summary
    print("\nData Summary:")
    summary = data_loader.get_data_summary(all_data)
    print(summary)
    
    # Explore key indicators
    if 'combined' in all_data:
        combined_data = all_data['combined']
        print(f"\nCombined dataset shape: {combined_data.shape}")
        print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        # Display key economic indicators
        key_indicators = [
            col for col in combined_data.columns 
            if any(keyword in col.lower() for keyword in 
                  ['gdp', 'inflation', 'unemployment', 'exchange', 'growth'])
        ]
        
        if key_indicators:
            print(f"\nKey Economic Indicators ({len(key_indicators)} found):")
            for indicator in key_indicators[:10]:  # Show first 10
                print(f"  - {indicator}")
            
            # Plot key indicators
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Bangladesh Key Economic Indicators', fontsize=16)
            
            # GDP Growth
            gdp_cols = [col for col in key_indicators if 'gdp' in col.lower() and 'growth' in col.lower()]
            if gdp_cols:
                combined_data[gdp_cols[0]].plot(ax=axes[0,0], title='GDP Growth (%)')
                axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Inflation
            inflation_cols = [col for col in key_indicators if 'inflation' in col.lower() or 'cpi' in col.lower()]
            if inflation_cols:
                combined_data[inflation_cols[0]].plot(ax=axes[0,1], title='Inflation (%)', color='orange')
                axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Exchange Rate
            exchange_cols = [col for col in key_indicators if 'exchange' in col.lower() or 'usd' in col.lower()]
            if exchange_cols:
                combined_data[exchange_cols[0]].plot(ax=axes[1,0], title='Exchange Rate (BDT/USD)', color='green')
            
            # Interest Rate
            rate_cols = [col for col in combined_data.columns if 'rate' in col.lower() and 'policy' in col.lower()]
            if rate_cols:
                combined_data[rate_cols[0]].plot(ax=axes[1,1], title='Policy Rate (%)', color='purple')
            
            plt.tight_layout()
            plt.savefig('results/bangladesh_key_indicators.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    return all_data

def example_2_dsge_analysis(data):
    """
    Example 2: DSGE Model Analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: DSGE MODEL ANALYSIS")
    print("="*60)
    
    # Load configuration
    config = load_config()
    dsge_config = config.get('dsge', {})
    
    # Initialize DSGE model
    print("Initializing DSGE model for Bangladesh...")
    dsge_model = DSGEModel(dsge_config, data)
    
    # Compute steady state
    print("\nComputing steady state...")
    steady_state = dsge_model.compute_steady_state()
    
    print("Steady State Values:")
    for var, value in steady_state.items():
        print(f"  {var}: {value:.4f}")
    
    # Solve model
    print("\nSolving DSGE model...")
    solution = dsge_model.solve_model()
    print(f"Model stability: {solution['stable']}")
    
    # Run simulation
    print("\nRunning model simulation...")
    simulation = dsge_model.simulate(periods=100)
    
    print(f"Simulation completed. Data shape: {simulation.shape}")
    
    # Plot simulation results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DSGE Model Simulation Results for Bangladesh', fontsize=16)
    
    # Key variables to plot
    plot_vars = ['y', 'c', 'pi', 'r', 'k', 'a']
    var_titles = ['Output', 'Consumption', 'Inflation', 'Interest Rate', 'Capital', 'Technology']
    
    for i, (var, title) in enumerate(zip(plot_vars, var_titles)):
        row, col = i // 3, i % 3
        if var in simulation.columns:
            simulation[var].plot(ax=axes[row, col], title=title)
            axes[row, col].axhline(y=steady_state.get(var, 0), color='red', 
                                 linestyle='--', alpha=0.7, label='Steady State')
            axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('results/dsge_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dsge_model, simulation

def example_3_impulse_responses(dsge_model):
    """
    Example 3: Impulse Response Analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: IMPULSE RESPONSE ANALYSIS")
    print("="*60)
    
    # Compute impulse responses for different shocks
    shocks = ['technology', 'monetary', 'fiscal', 'foreign']
    shock_titles = ['Technology Shock', 'Monetary Policy Shock', 'Fiscal Shock', 'Foreign Shock']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impulse Response Functions - Bangladesh DSGE Model', fontsize=16)
    
    for i, (shock, title) in enumerate(zip(shocks, shock_titles)):
        print(f"\nComputing impulse response to {shock} shock...")
        
        irf = dsge_model.impulse_response(shock_type=shock, shock_size=1.0, periods=20)
        
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot key variables' responses
        variables_to_plot = ['y', 'pi', 'r', 'c']
        var_labels = ['Output', 'Inflation', 'Interest Rate', 'Consumption']
        
        for var, label in zip(variables_to_plot, var_labels):
            if var in irf.columns:
                ax.plot(irf.index[:20], irf[var][:20], label=label, linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Deviation from Steady State')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/impulse_responses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {shock: dsge_model.impulse_response(shock) for shock in shocks}

def example_4_forecasting(dsge_model):
    """
    Example 4: Economic Forecasting
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: ECONOMIC FORECASTING")
    print("="*60)
    
    # Generate forecasts for different scenarios
    scenarios = ['baseline', 'optimistic', 'pessimistic']
    forecasts = {}
    
    for scenario in scenarios:
        print(f"\nGenerating {scenario} forecast...")
        forecast = dsge_model.forecast(horizon=12, scenario=scenario)
        forecasts[scenario] = forecast
        print(f"Forecast completed for {scenario} scenario")
    
    # Plot forecasts
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Economic Forecasts for Bangladesh (12 Quarters Ahead)', fontsize=16)
    
    forecast_vars = ['y', 'pi', 'r', 'c']
    var_titles = ['GDP Growth', 'Inflation', 'Interest Rate', 'Consumption Growth']
    colors = ['blue', 'green', 'red']
    
    for i, (var, title) in enumerate(zip(forecast_vars, var_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        for j, (scenario, color) in enumerate(zip(scenarios, colors)):
            if var in forecasts[scenario].columns:
                forecast_data = forecasts[scenario][var]
                ax.plot(forecast_data.index, forecast_data.values, 
                       label=scenario.capitalize(), color=color, linewidth=2)
                
                # Add confidence intervals for baseline
                if scenario == 'baseline' and f"{var}_lower" in forecasts[scenario].columns:
                    lower = forecasts[scenario][f"{var}_lower"]
                    upper = forecasts[scenario][f"{var}_upper"]
                    ax.fill_between(forecast_data.index, lower, upper, 
                                  alpha=0.2, color=color)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/economic_forecasts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return forecasts

def example_5_policy_analysis(dsge_model):
    """
    Example 5: Policy Analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: POLICY ANALYSIS")
    print("="*60)
    
    # Define policy experiments
    policy_experiments = [
        {
            'name': 'More Aggressive Monetary Policy',
            'changes': {'phi_pi': 2.0, 'phi_y': 0.8},
            'description': 'Increase inflation and output response in Taylor rule'
        },
        {
            'name': 'Reduced Interest Rate Smoothing',
            'changes': {'rho_r': 0.5},
            'description': 'Reduce interest rate smoothing parameter'
        },
        {
            'name': 'Higher Price Flexibility',
            'changes': {'theta': 0.6},
            'description': 'Reduce price stickiness (lower Calvo parameter)'
        }
    ]
    
    policy_results = {}
    
    for experiment in policy_experiments:
        print(f"\nAnalyzing policy: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Parameter changes: {experiment['changes']}")
        
        # Run policy analysis
        results = dsge_model.policy_analysis(experiment['changes'])
        policy_results[experiment['name']] = results
        
        print("Policy analysis completed")
    
    # Plot policy comparison for monetary policy shock
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Policy Analysis: Response to Monetary Policy Shock', fontsize=16)
    
    variables = ['y', 'pi', 'r', 'c']
    var_titles = ['Output', 'Inflation', 'Interest Rate', 'Consumption']
    
    # Get baseline response (original parameters)
    baseline_irf = dsge_model.impulse_response('monetary', periods=16)
    
    for i, (var, title) in enumerate(zip(variables, var_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot baseline
        if var in baseline_irf.columns:
            ax.plot(baseline_irf.index[:16], baseline_irf[var][:16], 
                   label='Baseline', linewidth=3, color='black')
        
        # Plot policy alternatives
        colors = ['red', 'blue', 'green']
        for j, (policy_name, color) in enumerate(zip(policy_results.keys(), colors)):
            policy_irf = policy_results[policy_name]['impulse_responses']['monetary']
            if var in policy_irf.columns:
                ax.plot(policy_irf.index[:16], policy_irf[var][:16], 
                       label=policy_name, linewidth=2, color=color, linestyle='--')
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel('Quarters')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/policy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return policy_results

def example_6_model_summary(dsge_model, data):
    """
    Example 6: Model Summary and Diagnostics
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: MODEL SUMMARY AND DIAGNOSTICS")
    print("="*60)
    
    # Get model summary
    summary = dsge_model.get_model_summary()
    
    print("\nDSGE Model Summary:")
    print(f"Model Type: {summary['model_type']}")
    print(f"Country: {summary['country']}")
    print(f"Frequency: {summary['frequency']}")
    print(f"Number of Variables: {len(summary['variables'])}")
    print(f"State Variables: {len(summary['state_variables'])}")
    print(f"Control Variables: {len(summary['control_variables'])}")
    
    print("\nKey Parameters:")
    for param, value in summary['parameters'].items():
        print(f"  {param}: {value:.4f}")
    
    print("\nSteady State Values:")
    for var, value in summary['steady_state'].items():
        if isinstance(value, (int, float)):
            print(f"  {var}: {value:.4f}")
    
    # Data coverage analysis
    if data and 'combined' in data:
        combined_data = data['combined']
        print(f"\nData Coverage:")
        print(f"  Total observations: {len(combined_data)}")
        print(f"  Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        print(f"  Variables available: {len(combined_data.columns)}")
        print(f"  Missing data: {combined_data.isnull().sum().sum()} cells ({combined_data.isnull().sum().sum()/combined_data.size*100:.1f}%)")
    
    # Model diagnostics
    if dsge_model.solution:
        eigenvalues = dsge_model.solution.get('eigenvalues', [])
        if len(eigenvalues) > 0:
            print(f"\nModel Stability:")
            print(f"  Stable: {dsge_model.solution['stable']}")
            print(f"  Max eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")
            print(f"  Number of eigenvalues: {len(eigenvalues)}")

def main():
    """
    Main function to run all examples
    """
    print("Bangladesh Macroeconomic Models - Example Analysis")
    print("=" * 60)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    try:
        # Example 1: Data Loading
        data = example_1_data_loading()
        
        # Example 2: DSGE Analysis
        dsge_model, simulation = example_2_dsge_analysis(data)
        
        # Example 3: Impulse Responses
        impulse_responses = example_3_impulse_responses(dsge_model)
        
        # Example 4: Forecasting
        forecasts = example_4_forecasting(dsge_model)
        
        # Example 5: Policy Analysis
        policy_results = example_5_policy_analysis(dsge_model)
        
        # Example 6: Model Summary
        example_6_model_summary(dsge_model, data)
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nResults saved to:")
        print("  - results/bangladesh_key_indicators.png")
        print("  - results/dsge_simulation.png")
        print("  - results/impulse_responses.png")
        print("  - results/economic_forecasts.png")
        print("  - results/policy_analysis.png")
        print("\nData files saved to:")
        print("  - data/processed/")
        print("  - data/raw/")
        
    except Exception as e:
        print(f"\nError in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()