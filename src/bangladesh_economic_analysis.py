#!/usr/bin/env python3
"""
Bangladesh Economic Analysis
Real-time analysis of Bangladesh's macroeconomic indicators and trends

Author: Economic Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_bangladesh_data():
    """
    Load processed Bangladesh economic data
    """
    try:
        data_path = Path('data/processed/bangladesh_macroeconomic_data.csv')
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"✅ Data loaded successfully: {df.shape}")
            return df
        else:
            print("❌ Processed data file not found")
            return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_economic_growth(df):
    """
    Analyze Bangladesh's economic growth patterns
    """
    print("\n🔍 ECONOMIC GROWTH ANALYSIS")
    print("=" * 50)
    
    if 'gdp_growth' in df.columns:
        recent_growth = df['gdp_growth'].tail(5).mean()
        historical_growth = df['gdp_growth'].mean()
        growth_volatility = df['gdp_growth'].std()
        
        print(f"📈 Average GDP Growth (2000-2023): {historical_growth:.2f}%")
        print(f"📈 Recent GDP Growth (last 5 years): {recent_growth:.2f}%")
        print(f"📊 Growth Volatility (std dev): {growth_volatility:.2f}%")
        
        # Growth trend analysis
        if recent_growth > historical_growth:
            print("✅ Economic growth has accelerated in recent years")
        else:
            print("⚠️ Economic growth has decelerated in recent years")
    
    return {
        'avg_growth': historical_growth,
        'recent_growth': recent_growth,
        'volatility': growth_volatility
    }

def analyze_inflation_trends(df):
    """
    Analyze inflation patterns and monetary stability
    """
    print("\n🔍 INFLATION ANALYSIS")
    print("=" * 50)
    
    if 'inflation' in df.columns:
        avg_inflation = df['inflation'].mean()
        recent_inflation = df['inflation'].tail(3).mean()
        inflation_volatility = df['inflation'].std()
        
        print(f"💰 Average Inflation (2000-2023): {avg_inflation:.2f}%")
        print(f"💰 Recent Inflation (last 3 years): {recent_inflation:.2f}%")
        print(f"📊 Inflation Volatility: {inflation_volatility:.2f}%")
        
        # Inflation assessment
        if recent_inflation > 6:
            print("🚨 High inflation - monetary policy attention needed")
        elif recent_inflation > 4:
            print("⚠️ Moderate inflation - monitor closely")
        else:
            print("✅ Low and stable inflation")
    
    return {
        'avg_inflation': avg_inflation,
        'recent_inflation': recent_inflation,
        'volatility': inflation_volatility
    }

def analyze_external_sector(df):
    """
    Analyze trade balance, current account, and external sustainability
    """
    print("\n🔍 EXTERNAL SECTOR ANALYSIS")
    print("=" * 50)
    
    # Trade analysis
    if 'trade_balance' in df.columns:
        avg_trade_balance = df['trade_balance'].mean()
        recent_trade_balance = df['trade_balance'].tail(3).mean()
        
        print(f"🌍 Average Trade Balance: {avg_trade_balance:.2f}% of GDP")
        print(f"🌍 Recent Trade Balance: {recent_trade_balance:.2f}% of GDP")
        
        if recent_trade_balance < -5:
            print("🚨 Large trade deficit - external vulnerability")
        elif recent_trade_balance < 0:
            print("⚠️ Trade deficit - monitor external balance")
        else:
            print("✅ Trade surplus or balanced trade")
    
    # Current account analysis
    if 'current_account' in df.columns:
        avg_ca = df['current_account'].mean()
        recent_ca = df['current_account'].tail(3).mean()
        
        print(f"💱 Average Current Account: {avg_ca:.2f}% of GDP")
        print(f"💱 Recent Current Account: {recent_ca:.2f}% of GDP")
    
    # Remittances analysis
    if 'remittances' in df.columns:
        avg_remittances = df['remittances'].mean()
        recent_remittances = df['remittances'].tail(3).mean()
        
        print(f"💸 Average Remittances: {avg_remittances:.2f}% of GDP")
        print(f"💸 Recent Remittances: {recent_remittances:.2f}% of GDP")
        
        if recent_remittances > 5:
            print("✅ Strong remittance inflows supporting external balance")
    
    return {
        'trade_balance': recent_trade_balance if 'trade_balance' in df.columns else None,
        'current_account': recent_ca if 'current_account' in df.columns else None,
        'remittances': recent_remittances if 'remittances' in df.columns else None
    }

def analyze_fiscal_position(df):
    """
    Analyze government fiscal health and sustainability
    """
    print("\n🔍 FISCAL ANALYSIS")
    print("=" * 50)
    
    if 'fiscal_balance' in df.columns:
        avg_fiscal = df['fiscal_balance'].mean()
        recent_fiscal = df['fiscal_balance'].tail(3).mean()
        
        print(f"🏛️ Average Fiscal Balance: {avg_fiscal:.2f}% of GDP")
        print(f"🏛️ Recent Fiscal Balance: {recent_fiscal:.2f}% of GDP")
        
        if recent_fiscal < -5:
            print("🚨 Large fiscal deficit - sustainability concerns")
        elif recent_fiscal < -3:
            print("⚠️ Moderate fiscal deficit - monitor debt dynamics")
        else:
            print("✅ Manageable fiscal position")
    
    if 'government_debt' in df.columns:
        recent_debt = df['government_debt'].tail(1).iloc[0]
        print(f"💳 Government Debt: {recent_debt:.1f}% of GDP")
        
        if recent_debt > 60:
            print("🚨 High debt level - fiscal consolidation needed")
        elif recent_debt > 40:
            print("⚠️ Moderate debt level - monitor closely")
        else:
            print("✅ Low debt level - fiscal space available")
    
    return {
        'fiscal_balance': recent_fiscal if 'fiscal_balance' in df.columns else None,
        'debt_level': recent_debt if 'government_debt' in df.columns else None
    }

def create_economic_dashboard(df):
    """
    Create comprehensive economic dashboard
    """
    print("\n📊 Creating Economic Dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bangladesh Economic Dashboard (2000-2023)', fontsize=16, fontweight='bold')
    
    # GDP Growth
    if 'gdp_growth' in df.columns:
        axes[0, 0].plot(df.index, df['gdp_growth'], linewidth=2, color='blue')
        axes[0, 0].axhline(y=df['gdp_growth'].mean(), color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('GDP Growth Rate (%)')
        axes[0, 0].set_ylabel('Growth Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Inflation
    if 'inflation' in df.columns:
        axes[0, 1].plot(df.index, df['inflation'], linewidth=2, color='orange')
        axes[0, 1].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% Target')
        axes[0, 1].set_title('Inflation Rate (%)')
        axes[0, 1].set_ylabel('Inflation (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Current Account
    if 'current_account' in df.columns:
        axes[0, 2].plot(df.index, df['current_account'], linewidth=2, color='green')
        axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 2].set_title('Current Account (% of GDP)')
        axes[0, 2].set_ylabel('% of GDP')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Fiscal Balance
    if 'fiscal_balance' in df.columns:
        axes[1, 0].plot(df.index, df['fiscal_balance'], linewidth=2, color='purple')
        axes[1, 0].axhline(y=-3, color='red', linestyle='--', alpha=0.7, label='-3% Threshold')
        axes[1, 0].set_title('Fiscal Balance (% of GDP)')
        axes[1, 0].set_ylabel('% of GDP')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Government Debt
    if 'government_debt' in df.columns:
        axes[1, 1].plot(df.index, df['government_debt'], linewidth=2, color='red')
        axes[1, 1].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60% Threshold')
        axes[1, 1].set_title('Government Debt (% of GDP)')
        axes[1, 1].set_ylabel('% of GDP')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    # Remittances
    if 'remittances' in df.columns:
        axes[1, 2].plot(df.index, df['remittances'], linewidth=2, color='brown')
        axes[1, 2].set_title('Remittances (% of GDP)')
        axes[1, 2].set_ylabel('% of GDP')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/dashboards/bangladesh_economic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Dashboard saved to visualization/dashboards/bangladesh_economic_analysis.png")

def economic_outlook_assessment(growth_data, inflation_data, external_data, fiscal_data):
    """
    Provide overall economic outlook assessment
    """
    print("\n🔮 ECONOMIC OUTLOOK ASSESSMENT")
    print("=" * 50)
    
    score = 0
    max_score = 0
    
    # Growth assessment
    if growth_data['recent_growth'] > 5:
        score += 2
        print("✅ Strong economic growth momentum")
    elif growth_data['recent_growth'] > 3:
        score += 1
        print("⚠️ Moderate economic growth")
    else:
        print("🚨 Weak economic growth")
    max_score += 2
    
    # Inflation assessment
    if inflation_data['recent_inflation'] < 6:
        score += 2
        print("✅ Inflation under control")
    elif inflation_data['recent_inflation'] < 8:
        score += 1
        print("⚠️ Moderate inflation pressure")
    else:
        print("🚨 High inflation risk")
    max_score += 2
    
    # External assessment
    if external_data['current_account'] and external_data['current_account'] > -3:
        score += 1
        print("✅ Manageable external position")
    elif external_data['current_account'] and external_data['current_account'] < -5:
        print("🚨 External vulnerability")
    else:
        print("⚠️ External balance needs monitoring")
    max_score += 1
    
    # Fiscal assessment
    if fiscal_data['fiscal_balance'] and fiscal_data['fiscal_balance'] > -3:
        score += 1
        print("✅ Sustainable fiscal position")
    elif fiscal_data['fiscal_balance'] and fiscal_data['fiscal_balance'] < -5:
        print("🚨 Fiscal sustainability concerns")
    else:
        print("⚠️ Fiscal consolidation needed")
    max_score += 1
    
    # Overall assessment
    overall_score = (score / max_score) * 100
    print(f"\n📊 Overall Economic Health Score: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("🟢 STRONG: Economy is performing well across key indicators")
    elif overall_score >= 60:
        print("🟡 MODERATE: Mixed economic performance with some areas of concern")
    else:
        print("🔴 WEAK: Significant economic challenges requiring policy attention")
    
    return overall_score

def main():
    """
    Main analysis function
    """
    print("🇧🇩 BANGLADESH ECONOMIC ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_bangladesh_data()
    if df is None:
        print("❌ Cannot proceed without data")
        return
    
    print(f"📅 Analysis Period: {df.index[0]} to {df.index[-1]}")
    print(f"📊 Data Points: {len(df)} years")
    print(f"📈 Indicators: {len(df.columns)} variables")
    
    # Perform analyses
    growth_data = analyze_economic_growth(df)
    inflation_data = analyze_inflation_trends(df)
    external_data = analyze_external_sector(df)
    fiscal_data = analyze_fiscal_position(df)
    
    # Create dashboard
    create_economic_dashboard(df)
    
    # Overall assessment
    economic_outlook_assessment(growth_data, inflation_data, external_data, fiscal_data)
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Check visualization/dashboards/ for charts")

if __name__ == "__main__":
    main()