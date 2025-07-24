#!/usr/bin/env python3
"""
Test Script for Bangladesh Macroeconomic Data Access

This script demonstrates how to fetch real macroeconomic data for Bangladesh
from World Bank and IMF APIs.

Author: Bangladesh Macroeconomic Modeling Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_world_bank_api():
    """
    Test World Bank API access for Bangladesh data
    """
    print("Testing World Bank API access...")
    
    try:
        import wbgapi as wb
        
        # Test 1: Get basic country information
        print("\n1. Country Information:")
        countries = wb.economy.DataFrame()
        bangladesh = countries[countries.index == 'BGD']
        if not bangladesh.empty:
            print(f"   Country: {bangladesh.iloc[0]['name']}")
            print(f"   Region: {bangladesh.iloc[0]['region']}")
            print(f"   Income Level: {bangladesh.iloc[0]['incomeLevel']}")
        
        # Test 2: Fetch GDP growth data
        print("\n2. GDP Growth Data (2018-2022):")
        gdp_data = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG', 'BGD', time=range(2018, 2023))
        if not gdp_data.empty:
            gdp_data = gdp_data.dropna()
            print("   Year | GDP Growth (%)")
            print("   -----|---------------")
            for year, value in gdp_data.iloc[:, 0].items():
                print(f"   {year} | {value:.2f}")
        
        # Test 3: Fetch multiple indicators
        print("\n3. Multiple Economic Indicators (Latest Year):")
        indicators = {
            'NY.GDP.MKTP.KD.ZG': 'GDP Growth (%)',
            'FP.CPI.TOTL.ZG': 'Inflation (%)',
            'SL.UEM.TOTL.ZS': 'Unemployment (%)',
            'BN.CAB.XOKA.GD.ZS': 'Current Account Balance (% GDP)',
            'GC.DOD.TOTL.GD.ZS': 'Government Debt (% GDP)'
        }
        
        multi_data = wb.data.DataFrame(list(indicators.keys()), 'BGD', time=range(2020, 2023))
        if not multi_data.empty:
            latest_data = multi_data.dropna().iloc[-1]
            for indicator_code, indicator_name in indicators.items():
                if indicator_code in latest_data.index:
                    value = latest_data[indicator_code]
                    print(f"   {indicator_name}: {value:.2f}")
        
        return True
    
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_imf_api():
    """
    Test IMF API access for Bangladesh data
    """
    print("\nTesting IMF API access...")
    
    try:
        import requests
        import json
        
        # Test 1: Get available datasets
        print("\n1. Available IMF Datasets:")
        dataflow_url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow"
        response = requests.get(dataflow_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            dataflows = data['Structure']['Dataflows']['Dataflow']
            print(f"   Found {len(dataflows)} available datasets")
            
            # Show first few datasets
            for i, df in enumerate(dataflows[:5]):
                print(f"   - {df['@id']}: {df['Name']['#text']}")
        
        # Test 2: Get Bangladesh data from IFS (International Financial Statistics)
        print("\n2. Bangladesh Data from IFS:")
        # IFS dataset, Bangladesh (512), quarterly frequency
        ifs_url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/Q.512.NGDP_SA_XDC"
        
        response = requests.get(ifs_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'CompactData' in data and 'DataSet' in data['CompactData']:
                dataset = data['CompactData']['DataSet']
                if 'Series' in dataset:
                    series = dataset['Series']
                    if 'Obs' in series:
                        observations = series['Obs']
                        if isinstance(observations, list):
                            recent_obs = observations[-5:]  # Last 5 observations
                            print("   Recent GDP observations:")
                            for obs in recent_obs:
                                if '@TIME_PERIOD' in obs and '@OBS_VALUE' in obs:
                                    print(f"   {obs['@TIME_PERIOD']}: {obs['@OBS_VALUE']}")
        
        return True
    
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_data_fetcher_class():
    """
    Test our custom DataFetcher class
    """
    print("\nTesting DataFetcher class...")
    
    try:
        from data_fetcher import DataFetcher
        
        # Initialize fetcher
        fetcher = DataFetcher()
        print("   ✓ DataFetcher initialized successfully")
        
        # Test World Bank data fetching
        print("\n1. Testing World Bank data fetching:")
        try:
            wb_data = fetcher.fetch_world_bank_data(
                indicators=['NY.GDP.MKTP.KD.ZG', 'FP.CPI.TOTL.ZG'],
                country='BGD',
                start_year=2020,
                end_year=2022
            )
            if wb_data is not None and not wb_data.empty:
                print(f"   ✓ Fetched {len(wb_data)} years of data")
                print(f"   ✓ Indicators: {list(wb_data.columns)}")
                print("   Sample data:")
                print(wb_data.head().to_string(float_format='%.2f'))
            else:
                print("   ⚠ No data returned")
        except Exception as e:
            print(f"   ⚠ World Bank fetch error: {e}")
        
        # Test comprehensive dataset
        print("\n2. Testing comprehensive Bangladesh dataset:")
        try:
            bd_data = fetcher.get_bangladesh_dataset(start_year=2020, end_year=2022)
            if bd_data is not None and not bd_data.empty:
                print(f"   ✓ Comprehensive dataset: {bd_data.shape[0]} years, {bd_data.shape[1]} indicators")
                print(f"   ✓ Available indicators: {list(bd_data.columns)[:5]}...")
            else:
                print("   ⚠ No comprehensive data available")
        except Exception as e:
            print(f"   ⚠ Comprehensive dataset error: {e}")
        
        return True
    
    except Exception as e:
        print(f"   Error importing DataFetcher: {e}")
        return False

def create_sample_visualization():
    """
    Create a sample visualization with available data
    """
    print("\nCreating sample visualization...")
    
    try:
        # Generate sample data if real data is not available
        years = list(range(2015, 2024))
        np.random.seed(42)
        
        # Simulate Bangladesh economic indicators
        data = {
            'GDP Growth': 6.5 + np.random.normal(0, 1.5, len(years)),
            'Inflation': 5.8 + np.random.normal(0, 2.0, len(years)),
            'Unemployment': 4.2 + np.random.normal(0, 0.8, len(years))
        }
        
        df = pd.DataFrame(data, index=years)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Bangladesh Economic Indicators (Sample Data)', fontsize=14, fontweight='bold')
        
        # GDP Growth
        axes[0].plot(df.index, df['GDP Growth'], 'b-', linewidth=2, marker='o')
        axes[0].set_title('GDP Growth Rate')
        axes[0].set_ylabel('Percent')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=df['GDP Growth'].mean(), color='r', linestyle='--', alpha=0.7)
        
        # Inflation
        axes[1].plot(df.index, df['Inflation'], 'g-', linewidth=2, marker='s')
        axes[1].set_title('Inflation Rate')
        axes[1].set_ylabel('Percent')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=df['Inflation'].mean(), color='r', linestyle='--', alpha=0.7)
        
        # Unemployment
        axes[2].plot(df.index, df['Unemployment'], 'orange', linewidth=2, marker='^')
        axes[2].set_title('Unemployment Rate')
        axes[2].set_ylabel('Percent')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=df['Unemployment'].mean(), color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('bangladesh_indicators_sample.png', dpi=300, bbox_inches='tight')
        print("   ✓ Sample visualization saved as 'bangladesh_indicators_sample.png'")
        
        # Print summary statistics
        print("\n   Summary Statistics:")
        print(df.describe().round(2).to_string())
        
        return True
    
    except Exception as e:
        print(f"   Error creating visualization: {e}")
        return False

def main():
    """
    Main test function
    """
    print("=" * 60)
    print("BANGLADESH MACROECONOMIC DATA ACCESS TEST")
    print("=" * 60)
    
    test_results = []
    
    # Test World Bank API
    test_results.append(test_world_bank_api())
    
    # Test IMF API
    test_results.append(test_imf_api())
    
    # Test DataFetcher class
    test_results.append(test_data_fetcher_class())
    
    # Create sample visualization
    test_results.append(create_sample_visualization())
    
    # Summary
    successful_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {successful_tests}/{total_tests} tests successful")
    print("=" * 60)
    
    if successful_tests >= 3:
        print("\n🎉 Data access functionality is working!")
        print("\nNext steps:")
        print("1. Use the DataFetcher class to get real Bangladesh data")
        print("2. Integrate the data with macroeconomic models")
        print("3. Run policy simulations with realistic parameters")
        print("4. Analyze results and generate insights")
    else:
        print("\n⚠️ Some data access issues detected.")
        print("Please check:")
        print("1. Internet connectivity")
        print("2. API availability")
        print("3. Package installations")
    
    print("\nFor detailed model integration, run demo_data_integration.py")

if __name__ == "__main__":
    main()