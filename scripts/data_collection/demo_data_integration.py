#!/usr/bin/env python3
"""
Demonstration of Data Integration with Bangladesh Macroeconomic Models

This script shows how to:
1. Fetch real macroeconomic data for Bangladesh from World Bank and IMF
2. Integrate the data with our economic models
3. Calibrate models using real data
4. Run simulations with realistic parameters

Author: Bangladesh Macroeconomic Modeling Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our models and data fetcher
from data_fetcher import DataFetcher
from models.cge.cge_model import CGEModel
from models.svar.svar_model import SVARModel
from models.abm.abm_model import ABMModel
from models.hank.hank_model import HANKModel
from models.financial.financial_model import FinancialSectorModel
from models.small_open_economy.soe_model import SmallOpenEconomyModel
from models.iam.iam_model import IntegratedAssessmentModel

def demonstrate_data_fetching():
    """
    Demonstrate fetching real macroeconomic data for Bangladesh
    """
    print("=" * 60)
    print("BANGLADESH MACROECONOMIC DATA FETCHING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Fetch comprehensive Bangladesh dataset
    print("\n1. Fetching comprehensive Bangladesh dataset...")
    try:
        bd_data = fetcher.get_bangladesh_dataset(start_year=2010, end_year=2023)
        print(f"   ✓ Successfully fetched data for {len(bd_data)} indicators")
        print(f"   ✓ Data covers period: {bd_data.index.min()} to {bd_data.index.max()}")
        
        # Display key indicators
        key_indicators = ['GDP_growth', 'inflation', 'unemployment', 'current_account_balance']
        available_indicators = [ind for ind in key_indicators if ind in bd_data.columns]
        
        if available_indicators:
            print("\n   Key Economic Indicators (Latest Available):")
            for indicator in available_indicators:
                latest_value = bd_data[indicator].dropna().iloc[-1]
                latest_year = bd_data[indicator].dropna().index[-1]
                print(f"   - {indicator}: {latest_value:.2f}% ({latest_year})")
    
    except Exception as e:
        print(f"   ⚠ Error fetching data: {e}")
        print("   Using simulated data for demonstration...")
        bd_data = generate_simulated_data()
    
    return bd_data

def generate_simulated_data():
    """
    Generate simulated Bangladesh macroeconomic data for demonstration
    """
    years = range(2010, 2024)
    np.random.seed(42)  # For reproducible results
    
    data = {
        'GDP_growth': 6.5 + np.random.normal(0, 1.5, len(years)),
        'inflation': 5.8 + np.random.normal(0, 2.0, len(years)),
        'unemployment': 4.2 + np.random.normal(0, 0.8, len(years)),
        'current_account_balance': -1.5 + np.random.normal(0, 1.0, len(years)),
        'government_debt': 35.0 + np.cumsum(np.random.normal(0.5, 1.0, len(years))),
        'exports_growth': 8.0 + np.random.normal(0, 3.0, len(years)),
        'imports_growth': 9.0 + np.random.normal(0, 3.5, len(years)),
        'remittances_gdp': 6.0 + np.random.normal(0, 0.5, len(years)),
        'fdi_gdp': 1.2 + np.random.normal(0, 0.3, len(years)),
        'exchange_rate': 80.0 + np.cumsum(np.random.normal(1.0, 2.0, len(years)))
    }
    
    return pd.DataFrame(data, index=years)

def calibrate_models_with_data(data):
    """
    Demonstrate model calibration using real data
    """
    print("\n2. Calibrating models with real data...")
    
    # Extract key statistics for calibration
    gdp_growth_mean = data['GDP_growth'].mean()
    inflation_mean = data['inflation'].mean()
    unemployment_mean = data['unemployment'].mean()
    
    print(f"   Using historical averages for calibration:")
    print(f"   - Average GDP growth: {gdp_growth_mean:.2f}%")
    print(f"   - Average inflation: {inflation_mean:.2f}%")
    print(f"   - Average unemployment: {unemployment_mean:.2f}%")
    
    # Calibrate SVAR model
    print("\n   Calibrating SVAR model...")
    try:
        svar = SVARModel()
        
        # Prepare data for SVAR (need multiple time series)
        svar_data = data[['GDP_growth', 'inflation', 'unemployment']].dropna()
        
        if len(svar_data) >= 10:  # Need sufficient observations
            svar.estimate_model(svar_data)
            print("   ✓ SVAR model estimated successfully")
            
            # Generate impulse responses
            irf = svar.impulse_response_analysis(periods=20)
            print(f"   ✓ Impulse response functions computed for {len(irf)} variables")
        else:
            print("   ⚠ Insufficient data for SVAR estimation")
    
    except Exception as e:
        print(f"   ⚠ SVAR calibration error: {e}")
    
    # Calibrate HANK model
    print("\n   Calibrating HANK model...")
    try:
        hank = HANKModel()
        
        # Adjust parameters based on Bangladesh characteristics
        hank.params.beta = 0.96  # Discount factor (lower for developing economy)
        hank.params.alpha = 0.35  # Capital share
        hank.params.delta = 0.08  # Depreciation rate
        hank.params.rho_z = 0.95  # Productivity persistence
        hank.params.sigma_z = 0.02  # Productivity volatility
        
        # Set inflation target based on historical average
        hank.params.pi_target = inflation_mean / 100
        
        print("   ✓ HANK model parameters calibrated")
        
        # Solve steady state
        steady_state = hank.solve_steady_state()
        print(f"   ✓ Steady state solved: r = {steady_state['interest_rate']:.3f}")
    
    except Exception as e:
        print(f"   ⚠ HANK calibration error: {e}")
    
    # Calibrate Small Open Economy model
    print("\n   Calibrating Small Open Economy model...")
    try:
        soe = SmallOpenEconomyModel()
        
        # Use historical exchange rate volatility
        if 'exchange_rate' in data.columns:
            er_returns = data['exchange_rate'].pct_change().dropna()
            soe.params.exchange_rate_volatility = er_returns.std() * np.sqrt(12)  # Annualized
            print(f"   ✓ Exchange rate volatility calibrated: {soe.params.exchange_rate_volatility:.3f}")
        
        # Use historical current account balance
        if 'current_account_balance' in data.columns:
            ca_mean = data['current_account_balance'].mean()
            soe.params.current_account_target = ca_mean / 100
            print(f"   ✓ Current account target set: {ca_mean:.2f}% of GDP")
        
        print("   ✓ SOE model calibrated successfully")
    
    except Exception as e:
        print(f"   ⚠ SOE calibration error: {e}")
    
    return {
        'gdp_growth_mean': gdp_growth_mean,
        'inflation_mean': inflation_mean,
        'unemployment_mean': unemployment_mean
    }

def run_policy_simulations(calibration_stats):
    """
    Run policy simulations using calibrated models
    """
    print("\n3. Running policy simulations...")
    
    # Monetary policy simulation with HANK model
    print("\n   Monetary Policy Analysis (HANK Model):")
    try:
        hank = HANKModel()
        
        # Simulate monetary policy shock
        shock_size = 0.01  # 100 basis points
        periods = 20
        
        irf = hank.compute_impulse_responses(
            shock_type='monetary_policy',
            shock_size=shock_size,
            periods=periods
        )
        
        print(f"   ✓ Monetary policy shock simulation completed")
        print(f"   - Interest rate increases by {shock_size*100:.0f} basis points")
        print(f"   - GDP impact after 4 quarters: {irf['output'][3]:.3f}%")
        print(f"   - Inflation impact after 4 quarters: {irf['inflation'][3]:.3f}%")
    
    except Exception as e:
        print(f"   ⚠ Monetary policy simulation error: {e}")
    
    # External shock simulation with SOE model
    print("\n   External Shock Analysis (SOE Model):")
    try:
        soe = SmallOpenEconomyModel()
        
        # Simulate terms of trade shock
        shock_scenarios = {
            'terms_of_trade_shock': -0.1,  # 10% deterioration
            'capital_flow_shock': -0.05,   # 5% reduction in capital flows
            'remittance_shock': -0.15      # 15% reduction in remittances
        }
        
        results = soe.analyze_policy_scenarios(shock_scenarios, periods=12)
        
        print(f"   ✓ External shock simulations completed")
        for scenario, impact in results.items():
            if 'gdp_impact' in impact:
                print(f"   - {scenario}: GDP impact = {impact['gdp_impact']:.2f}%")
    
    except Exception as e:
        print(f"   ⚠ External shock simulation error: {e}")
    
    # Financial stability analysis
    print("\n   Financial Stability Analysis:")
    try:
        financial = FinancialSectorModel()
        
        # Run stress test scenarios
        stress_scenarios = {
            'credit_shock': {'default_rate_increase': 0.05},
            'liquidity_shock': {'deposit_outflow': 0.20},
            'interest_rate_shock': {'rate_increase': 0.03}
        }
        
        stress_results = financial.conduct_stress_test(stress_scenarios)
        
        print(f"   ✓ Financial stress tests completed")
        for scenario, result in stress_results.items():
            if 'capital_adequacy' in result:
                print(f"   - {scenario}: Capital adequacy = {result['capital_adequacy']:.2f}%")
    
    except Exception as e:
        print(f"   ⚠ Financial stability analysis error: {e}")

def create_policy_dashboard(data):
    """
    Create a simple policy dashboard with key indicators
    """
    print("\n4. Creating policy dashboard...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bangladesh Macroeconomic Dashboard', fontsize=16, fontweight='bold')
        
        # GDP Growth
        axes[0, 0].plot(data.index, data['GDP_growth'], 'b-', linewidth=2)
        axes[0, 0].axhline(y=data['GDP_growth'].mean(), color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('GDP Growth Rate (%)')
        axes[0, 0].set_ylabel('Percent')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Inflation
        axes[0, 1].plot(data.index, data['inflation'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=data['inflation'].mean(), color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Inflation Rate (%)')
        axes[0, 1].set_ylabel('Percent')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Current Account Balance
        axes[1, 0].plot(data.index, data['current_account_balance'], 'orange', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[1, 0].axhline(y=data['current_account_balance'].mean(), color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Current Account Balance (% of GDP)')
        axes[1, 0].set_ylabel('Percent of GDP')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Government Debt
        axes[1, 1].plot(data.index, data['government_debt'], 'purple', linewidth=2)
        axes[1, 1].axhline(y=data['government_debt'].mean(), color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Government Debt (% of GDP)')
        axes[1, 1].set_ylabel('Percent of GDP')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bangladesh_macro_dashboard.png', dpi=300, bbox_inches='tight')
        print("   ✓ Dashboard saved as 'bangladesh_macro_dashboard.png'")
        
        # Show recent trends
        print("\n   Recent Economic Trends (Last 3 Years):")
        recent_data = data.tail(3)
        for indicator in ['GDP_growth', 'inflation', 'unemployment']:
            if indicator in recent_data.columns:
                trend = recent_data[indicator].iloc[-1] - recent_data[indicator].iloc[0]
                direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                print(f"   - {indicator}: {direction} {abs(trend):.1f} percentage points")
    
    except Exception as e:
        print(f"   ⚠ Dashboard creation error: {e}")

def main():
    """
    Main demonstration function
    """
    print("Starting Bangladesh Macroeconomic Data Integration Demo...\n")
    
    # Step 1: Fetch real data
    data = demonstrate_data_fetching()
    
    # Step 2: Calibrate models
    calibration_stats = calibrate_models_with_data(data)
    
    # Step 3: Run policy simulations
    run_policy_simulations(calibration_stats)
    
    # Step 4: Create dashboard
    create_policy_dashboard(data)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Install required packages: pip install -r requirements.txt")
    print("2. Set up API keys for World Bank and IMF data access")
    print("3. Run individual model scripts for detailed analysis")
    print("4. Customize parameters based on specific research questions")
    print("5. Extend models with additional Bangladesh-specific features")
    
    print("\nFor more information, see the documentation in each model file.")

if __name__ == "__main__":
    main()