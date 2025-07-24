#!/usr/bin/env python3
"""
Working Demonstration of Bangladesh Macroeconomic Data Integration

This script shows a working example of:
1. Fetching real Bangladesh data from World Bank
2. Processing and analyzing the data
3. Creating visualizations
4. Demonstrating how this data can be used for model calibration

Author: Bangladesh Macroeconomic Modeling Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fetch_bangladesh_data():
    """
    Fetch real Bangladesh macroeconomic data from World Bank
    """
    print("Fetching Bangladesh macroeconomic data from World Bank...")
    
    try:
        import wbgapi as wb
        
        # Define key indicators for Bangladesh
        indicators = {
            'NY.GDP.MKTP.KD.ZG': 'GDP Growth (%)',
            'FP.CPI.TOTL.ZG': 'Inflation (%)',
            'SL.UEM.TOTL.ZS': 'Unemployment (%)',
            'BN.CAB.XOKA.GD.ZS': 'Current Account Balance (% GDP)',
            'GC.DOD.TOTL.GD.ZS': 'Government Debt (% GDP)',
            'NE.EXP.GNFS.KD.ZG': 'Exports Growth (%)',
            'NE.IMP.GNFS.KD.ZG': 'Imports Growth (%)',
            'BX.TRF.PWKR.DT.GD.ZS': 'Remittances (% GDP)',
            'NY.GDP.PCAP.KD.ZG': 'GDP per Capita Growth (%)'
        }
        
        # Fetch data for each indicator
        all_data = {}
        years = range(2010, 2024)
        
        for code, name in indicators.items():
            try:
                data = wb.data.DataFrame(code, 'BGD', time=years)
                if not data.empty:
                    # Extract the data series
                    series = data.iloc[:, 0].dropna()
                    if not series.empty:
                        all_data[name] = series
                        print(f"   ✓ {name}: {len(series)} observations")
                    else:
                        print(f"   ⚠ {name}: No data available")
                else:
                    print(f"   ⚠ {name}: No data returned")
            except Exception as e:
                print(f"   ❌ {name}: Error - {str(e)[:50]}...")
        
        if all_data:
            # Combine all data into a single DataFrame
            df = pd.DataFrame(all_data)
            df.index.name = 'Year'
            print(f"\n✓ Successfully fetched data for {len(df.columns)} indicators")
            print(f"✓ Data covers {len(df)} years: {df.index.min()} to {df.index.max()}")
            return df
        else:
            print("\n❌ No data could be fetched")
            return None
    
    except ImportError:
        print("❌ wbgapi not available. Please install: pip install wbgapi")
        return None
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def analyze_bangladesh_economy(data):
    """
    Analyze Bangladesh economic data and extract key insights
    """
    print("\nAnalyzing Bangladesh economic data...")
    
    if data is None or data.empty:
        print("❌ No data available for analysis")
        return None
    
    # Calculate summary statistics
    print("\n1. Summary Statistics (2010-2023):")
    summary = data.describe()
    
    key_stats = ['mean', 'std', 'min', 'max']
    for stat in key_stats:
        if stat in summary.index:
            print(f"\n   {stat.upper()}:")
            for col in data.columns:
                if col in summary.columns:
                    value = summary.loc[stat, col]
                    print(f"   - {col}: {value:.2f}")
    
    # Identify trends
    print("\n2. Recent Trends (Last 5 Years vs Previous 5 Years):")
    if len(data) >= 10:
        recent_period = data.tail(5)
        earlier_period = data.iloc[-10:-5]
        
        for col in data.columns:
            if col in recent_period.columns and col in earlier_period.columns:
                recent_avg = recent_period[col].mean()
                earlier_avg = earlier_period[col].mean()
                change = recent_avg - earlier_avg
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   - {col}: {direction} {abs(change):.2f} percentage points")
    
    # Economic insights
    print("\n3. Key Economic Insights:")
    
    # GDP Growth analysis
    if 'GDP Growth (%)' in data.columns:
        gdp_data = data['GDP Growth (%)'].dropna()
        if not gdp_data.empty:
            avg_growth = gdp_data.mean()
            volatility = gdp_data.std()
            print(f"   - Average GDP growth: {avg_growth:.2f}% (volatility: {volatility:.2f}%)")
            
            if avg_growth > 6:
                print("   - Bangladesh shows strong economic growth")
            elif avg_growth > 4:
                print("   - Bangladesh shows moderate economic growth")
            else:
                print("   - Bangladesh shows slow economic growth")
    
    # Inflation analysis
    if 'Inflation (%)' in data.columns:
        inflation_data = data['Inflation (%)'].dropna()
        if not inflation_data.empty:
            avg_inflation = inflation_data.mean()
            print(f"   - Average inflation: {avg_inflation:.2f}%")
            
            if avg_inflation > 8:
                print("   - High inflation environment")
            elif avg_inflation > 4:
                print("   - Moderate inflation environment")
            else:
                print("   - Low inflation environment")
    
    # External sector analysis
    if 'Current Account Balance (% GDP)' in data.columns:
        ca_data = data['Current Account Balance (% GDP)'].dropna()
        if not ca_data.empty:
            avg_ca = ca_data.mean()
            print(f"   - Average current account balance: {avg_ca:.2f}% of GDP")
            
            if avg_ca < -3:
                print("   - Large current account deficit - external vulnerability")
            elif avg_ca < 0:
                print("   - Current account deficit - manageable")
            else:
                print("   - Current account surplus - strong external position")
    
    return data

def create_comprehensive_dashboard(data):
    """
    Create a comprehensive economic dashboard
    """
    print("\nCreating comprehensive economic dashboard...")
    
    if data is None or data.empty:
        print("❌ No data available for dashboard")
        return False
    
    try:
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 12))
        
        # Define the layout
        plots = [
            ('GDP Growth (%)', 'blue', 'GDP Growth Rate'),
            ('Inflation (%)', 'red', 'Inflation Rate'),
            ('Unemployment (%)', 'green', 'Unemployment Rate'),
            ('Current Account Balance (% GDP)', 'orange', 'Current Account Balance'),
            ('Government Debt (% GDP)', 'purple', 'Government Debt'),
            ('Remittances (% GDP)', 'brown', 'Remittances')
        ]
        
        # Create subplots
        for i, (col, color, title) in enumerate(plots):
            if col in data.columns:
                plt.subplot(2, 3, i + 1)
                series = data[col].dropna()
                if not series.empty:
                    plt.plot(series.index, series.values, color=color, linewidth=2, marker='o')
                    plt.title(title, fontsize=12, fontweight='bold')
                    plt.ylabel('Percent')
                    plt.grid(True, alpha=0.3)
                    
                    # Add trend line
                    if len(series) > 2:
                        z = np.polyfit(range(len(series)), series.values, 1)
                        p = np.poly1d(z)
                        plt.plot(series.index, p(range(len(series))), "--", alpha=0.7, color='gray')
                    
                    # Add average line
                    plt.axhline(y=series.mean(), color='red', linestyle=':', alpha=0.7)
                    
                    plt.xticks(rotation=45)
        
        plt.suptitle('Bangladesh Macroeconomic Dashboard (2010-2023)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('bangladesh_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("   ✓ Comprehensive dashboard saved as 'bangladesh_comprehensive_dashboard.png'")
        
        # Create correlation matrix
        plt.figure(figsize=(12, 8))
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            
            # Create heatmap
            import matplotlib.colors as mcolors
            plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            
            # Add labels
            plt.xticks(range(len(correlation_matrix.columns)), 
                      [col.replace(' (%)', '').replace(' (% GDP)', '') for col in correlation_matrix.columns], 
                      rotation=45, ha='right')
            plt.yticks(range(len(correlation_matrix.columns)), 
                      [col.replace(' (%)', '').replace(' (% GDP)', '') for col in correlation_matrix.columns])
            
            # Add correlation values
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8)
            
            plt.title('Bangladesh Economic Indicators - Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('bangladesh_correlation_matrix.png', dpi=300, bbox_inches='tight')
            print("   ✓ Correlation matrix saved as 'bangladesh_correlation_matrix.png'")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error creating dashboard: {e}")
        return False

def demonstrate_model_calibration(data):
    """
    Demonstrate how to use real data for model calibration
    """
    print("\nDemonstrating model calibration with real data...")
    
    if data is None or data.empty:
        print("❌ No data available for calibration")
        return
    
    print("\n1. Key Parameters for Model Calibration:")
    
    # Extract key parameters
    params = {}
    
    if 'GDP Growth (%)' in data.columns:
        gdp_data = data['GDP Growth (%)'].dropna()
        if not gdp_data.empty:
            params['avg_gdp_growth'] = gdp_data.mean() / 100
            params['gdp_volatility'] = gdp_data.std() / 100
            print(f"   - Average GDP growth rate: {params['avg_gdp_growth']:.3f}")
            print(f"   - GDP growth volatility: {params['gdp_volatility']:.3f}")
    
    if 'Inflation (%)' in data.columns:
        inflation_data = data['Inflation (%)'].dropna()
        if not inflation_data.empty:
            params['avg_inflation'] = inflation_data.mean() / 100
            params['inflation_volatility'] = inflation_data.std() / 100
            print(f"   - Average inflation rate: {params['avg_inflation']:.3f}")
            print(f"   - Inflation volatility: {params['inflation_volatility']:.3f}")
    
    if 'Current Account Balance (% GDP)' in data.columns:
        ca_data = data['Current Account Balance (% GDP)'].dropna()
        if not ca_data.empty:
            params['avg_current_account'] = ca_data.mean() / 100
            print(f"   - Average current account balance: {params['avg_current_account']:.3f}")
    
    if 'Government Debt (% GDP)' in data.columns:
        debt_data = data['Government Debt (% GDP)'].dropna()
        if not debt_data.empty:
            params['avg_debt_ratio'] = debt_data.mean() / 100
            print(f"   - Average debt-to-GDP ratio: {params['avg_debt_ratio']:.3f}")
    
    print("\n2. Model Calibration Recommendations:")
    print("\nFor DSGE/RBC Models:")
    if 'avg_gdp_growth' in params:
        print(f"   - Set steady-state growth rate: {params['avg_gdp_growth']:.3f}")
    if 'gdp_volatility' in params:
        print(f"   - Calibrate productivity shock volatility: {params['gdp_volatility']:.3f}")
    
    print("\nFor Monetary Policy Models:")
    if 'avg_inflation' in params:
        print(f"   - Set inflation target: {params['avg_inflation']:.3f}")
    if 'inflation_volatility' in params:
        print(f"   - Calibrate price rigidity based on volatility: {params['inflation_volatility']:.3f}")
    
    print("\nFor Fiscal Policy Models:")
    if 'avg_debt_ratio' in params:
        print(f"   - Set initial debt-to-GDP ratio: {params['avg_debt_ratio']:.3f}")
    
    print("\nFor Open Economy Models:")
    if 'avg_current_account' in params:
        print(f"   - Set current account target: {params['avg_current_account']:.3f}")
    
    return params

def main():
    """
    Main demonstration function
    """
    print("=" * 70)
    print("BANGLADESH MACROECONOMIC DATA INTEGRATION - WORKING DEMO")
    print("=" * 70)
    
    # Step 1: Fetch real data
    data = fetch_bangladesh_data()
    
    if data is not None and not data.empty:
        # Step 2: Analyze the data
        analyzed_data = analyze_bangladesh_economy(data)
        
        # Step 3: Create visualizations
        dashboard_success = create_comprehensive_dashboard(data)
        
        # Step 4: Demonstrate model calibration
        calibration_params = demonstrate_model_calibration(data)
        
        # Step 5: Save the data
        try:
            data.to_csv('bangladesh_macroeconomic_data.csv')
            print(f"\n✓ Data saved to 'bangladesh_macroeconomic_data.csv'")
        except Exception as e:
            print(f"\n❌ Error saving data: {e}")
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        print("\n📊 Generated Files:")
        print("   - bangladesh_macroeconomic_data.csv (raw data)")
        print("   - bangladesh_comprehensive_dashboard.png (main dashboard)")
        print("   - bangladesh_correlation_matrix.png (correlation analysis)")
        
        print("\n🎯 Next Steps:")
        print("   1. Use the calibration parameters for your specific models")
        print("   2. Run policy simulations with realistic Bangladesh parameters")
        print("   3. Compare model predictions with historical data")
        print("   4. Conduct scenario analysis for policy recommendations")
        
        print("\n📈 Key Insights:")
        if 'GDP Growth (%)' in data.columns:
            gdp_avg = data['GDP Growth (%)'].mean()
            print(f"   - Bangladesh maintains strong GDP growth averaging {gdp_avg:.1f}%")
        if 'Inflation (%)' in data.columns:
            inf_avg = data['Inflation (%)'].mean()
            print(f"   - Inflation averages {inf_avg:.1f}%, indicating price stability challenges")
        if 'Remittances (% GDP)' in data.columns:
            rem_avg = data['Remittances (% GDP)'].mean()
            print(f"   - Remittances contribute significantly at {rem_avg:.1f}% of GDP")
    
    else:
        print("\n❌ Could not fetch data. Please check:")
        print("   1. Internet connectivity")
        print("   2. World Bank API availability")
        print("   3. Package installation: pip install wbgapi")
        
        print("\n🔄 Alternative: The system can work with simulated data for testing")
    
    print("\nFor technical support, refer to the documentation or contact the team.")

if __name__ == "__main__":
    main()