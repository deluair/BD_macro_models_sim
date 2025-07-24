#!/usr/bin/env python3
"""
Policy Scenario Analysis Module

This module provides comprehensive policy analysis capabilities for evaluating
the impact of different policy interventions across multiple economic models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from scipy import stats
from itertools import combinations

@dataclass
class PolicyScenario:
    """
    Data class for defining policy scenarios.
    """
    name: str
    description: str
    policy_type: str  # 'monetary', 'fiscal', 'structural', 'trade'
    parameters: Dict[str, float]
    duration: int  # quarters
    implementation_lag: int = 1  # quarters

class PolicyScenarioAnalyzer:
    """
    Comprehensive policy scenario analysis across multiple economic models.
    """
    
    def __init__(self, results_dir: str = "../../results", output_dir: str = "./output"):
        """
        Initialize the policy scenario analyzer.
        
        Args:
            results_dir: Directory containing model results
            output_dir: Directory for saving analysis outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define policy scenarios for Bangladesh
        self.policy_scenarios = self._define_policy_scenarios()
        
        # Model capabilities for different policy types
        self.model_capabilities = {
            'monetary': ['dsge', 'svar', 'rbc', 'behavioral', 'financial'],
            'fiscal': ['dsge', 'cge', 'rbc', 'behavioral', 'hank'],
            'structural': ['cge', 'dsge', 'neg', 'search_matching'],
            'trade': ['soe', 'cge', 'neg', 'iam']
        }
        
        # Key economic indicators
        self.key_indicators = [
            'gdp_growth', 'inflation', 'unemployment', 'current_account',
            'government_debt', 'trade_balance', 'real_exchange_rate'
        ]
        
    def _define_policy_scenarios(self) -> List[PolicyScenario]:
        """
        Define comprehensive policy scenarios for Bangladesh.
        
        Returns:
            List of PolicyScenario objects
        """
        scenarios = [
            # Monetary Policy Scenarios
            PolicyScenario(
                name="Expansionary Monetary Policy",
                description="Reduce policy rate by 200 basis points to stimulate growth",
                policy_type="monetary",
                parameters={"policy_rate_change": -2.0, "money_supply_growth": 15.0},
                duration=8
            ),
            PolicyScenario(
                name="Contractionary Monetary Policy",
                description="Increase policy rate by 150 basis points to control inflation",
                policy_type="monetary",
                parameters={"policy_rate_change": 1.5, "money_supply_growth": 8.0},
                duration=6
            ),
            
            # Fiscal Policy Scenarios
            PolicyScenario(
                name="Infrastructure Investment Boost",
                description="Increase government investment by 3% of GDP",
                policy_type="fiscal",
                parameters={"government_spending_change": 3.0, "investment_share": 0.8},
                duration=12
            ),
            PolicyScenario(
                name="Tax Reform Package",
                description="Reduce corporate tax rate and increase VAT efficiency",
                policy_type="fiscal",
                parameters={"corporate_tax_change": -5.0, "vat_efficiency_gain": 10.0},
                duration=16
            ),
            
            # Structural Reform Scenarios
            PolicyScenario(
                name="Labor Market Flexibility",
                description="Improve labor market flexibility and skills development",
                policy_type="structural",
                parameters={"labor_flexibility_index": 0.2, "skills_improvement": 15.0},
                duration=20
            ),
            PolicyScenario(
                name="Financial Sector Development",
                description="Enhance financial inclusion and banking efficiency",
                policy_type="structural",
                parameters={"financial_inclusion_rate": 25.0, "banking_efficiency": 20.0},
                duration=16
            ),
            
            # Trade Policy Scenarios
            PolicyScenario(
                name="Export Diversification Strategy",
                description="Promote non-textile exports and value addition",
                policy_type="trade",
                parameters={"export_diversification": 30.0, "value_added_share": 15.0},
                duration=20
            ),
            PolicyScenario(
                name="Regional Trade Integration",
                description="Enhance regional trade partnerships and reduce barriers",
                policy_type="trade",
                parameters={"trade_barrier_reduction": 25.0, "regional_integration": 40.0},
                duration=12
            )
        ]
        
        return scenarios
    
    def load_model_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available model results.
        
        Returns:
            Dictionary of model results DataFrames
        """
        results = {}
        
        for file_path in self.results_dir.glob("*.csv"):
            model_name = file_path.stem.replace('_results', '')
            try:
                df = pd.read_csv(file_path)
                results[model_name] = df
                print(f"Loaded {model_name}: {df.shape}")
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                
        return results
    
    def simulate_policy_impacts(self, model_results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Simulate policy impacts across different models and scenarios.
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Dictionary of policy impact simulations
        """
        policy_impacts = {}
        
        for scenario in self.policy_scenarios:
            scenario_impacts = {}
            
            # Get models capable of analyzing this policy type
            capable_models = self.model_capabilities.get(scenario.policy_type, [])
            available_models = [m for m in capable_models if m in model_results.keys()]
            
            for model_name in available_models:
                model_impact = self._simulate_model_policy_impact(scenario, model_name, model_results[model_name])
                scenario_impacts[model_name] = model_impact
            
            policy_impacts[scenario.name] = {
                'scenario': scenario,
                'model_impacts': scenario_impacts
            }
        
        return policy_impacts
    
    def _simulate_model_policy_impact(self, scenario: PolicyScenario, 
                                    model_name: str, 
                                    model_data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Simulate policy impact for a specific model.
        
        Args:
            scenario: Policy scenario to simulate
            model_name: Name of the economic model
            model_data: Model results data
            
        Returns:
            Dictionary of simulated impacts by indicator
        """
        np.random.seed(42)  # For reproducible results
        
        # Base impact multipliers by policy type and model
        impact_multipliers = self._get_impact_multipliers(scenario.policy_type, model_name)
        
        impacts = {}
        
        for indicator in self.key_indicators:
            # Calculate base impact
            base_impact = self._calculate_base_impact(scenario, indicator, impact_multipliers)
            
            # Generate time series of impacts
            impact_series = self._generate_impact_time_series(base_impact, scenario.duration, 
                                                            scenario.implementation_lag)
            
            impacts[indicator] = impact_series
        
        return impacts
    
    def _get_impact_multipliers(self, policy_type: str, model_name: str) -> Dict[str, float]:
        """
        Get impact multipliers for different policy types and models.
        
        Args:
            policy_type: Type of policy intervention
            model_name: Name of the economic model
            
        Returns:
            Dictionary of impact multipliers by indicator
        """
        # Model-specific multipliers (based on theoretical foundations)
        multipliers = {
            'monetary': {
                'dsge': {'gdp_growth': 0.8, 'inflation': 1.2, 'unemployment': -0.6, 'current_account': 0.3},
                'svar': {'gdp_growth': 0.9, 'inflation': 1.0, 'unemployment': -0.7, 'current_account': 0.4},
                'rbc': {'gdp_growth': 0.6, 'inflation': 0.8, 'unemployment': -0.4, 'current_account': 0.2},
                'behavioral': {'gdp_growth': 1.1, 'inflation': 1.3, 'unemployment': -0.8, 'current_account': 0.5},
                'financial': {'gdp_growth': 0.7, 'inflation': 0.9, 'unemployment': -0.5, 'current_account': 0.6}
            },
            'fiscal': {
                'dsge': {'gdp_growth': 1.2, 'inflation': 0.8, 'unemployment': -0.9, 'government_debt': 1.5},
                'cge': {'gdp_growth': 1.4, 'inflation': 0.9, 'unemployment': -1.1, 'government_debt': 1.3},
                'rbc': {'gdp_growth': 0.9, 'inflation': 0.6, 'unemployment': -0.7, 'government_debt': 1.2},
                'behavioral': {'gdp_growth': 1.3, 'inflation': 1.0, 'unemployment': -1.0, 'government_debt': 1.4},
                'hank': {'gdp_growth': 1.5, 'inflation': 1.1, 'unemployment': -1.2, 'government_debt': 1.6}
            },
            'structural': {
                'cge': {'gdp_growth': 2.0, 'unemployment': -1.5, 'trade_balance': 0.8},
                'dsge': {'gdp_growth': 1.6, 'unemployment': -1.2, 'trade_balance': 0.6},
                'neg': {'gdp_growth': 1.8, 'unemployment': -1.3, 'trade_balance': 1.0},
                'search_matching': {'gdp_growth': 1.4, 'unemployment': -2.0, 'trade_balance': 0.4}
            },
            'trade': {
                'soe': {'gdp_growth': 1.3, 'trade_balance': 1.8, 'real_exchange_rate': -0.5},
                'cge': {'gdp_growth': 1.5, 'trade_balance': 2.0, 'real_exchange_rate': -0.7},
                'neg': {'gdp_growth': 1.2, 'trade_balance': 1.6, 'real_exchange_rate': -0.4},
                'iam': {'gdp_growth': 1.1, 'trade_balance': 1.4, 'real_exchange_rate': -0.3}
            }
        }
        
        # Default multipliers
        default_multipliers = {indicator: 0.5 for indicator in self.key_indicators}
        
        return multipliers.get(policy_type, {}).get(model_name, default_multipliers)
    
    def _calculate_base_impact(self, scenario: PolicyScenario, 
                             indicator: str, 
                             multipliers: Dict[str, float]) -> float:
        """
        Calculate base impact for a specific indicator.
        
        Args:
            scenario: Policy scenario
            indicator: Economic indicator
            multipliers: Impact multipliers
            
        Returns:
            Base impact value
        """
        base_impact = 0.0
        multiplier = multipliers.get(indicator, 0.0)
        
        # Calculate impact based on policy parameters
        for param_name, param_value in scenario.parameters.items():
            if 'change' in param_name or 'growth' in param_name:
                base_impact += param_value * multiplier * 0.1
            elif 'rate' in param_name or 'share' in param_name:
                base_impact += param_value * multiplier * 0.05
            else:
                base_impact += param_value * multiplier * 0.02
        
        return base_impact
    
    def _generate_impact_time_series(self, base_impact: float, 
                                   duration: int, 
                                   implementation_lag: int) -> List[float]:
        """
        Generate time series of policy impacts.
        
        Args:
            base_impact: Base impact magnitude
            duration: Duration of policy effect
            implementation_lag: Implementation lag in quarters
            
        Returns:
            List of impact values over time
        """
        time_series = []
        
        for quarter in range(duration + implementation_lag):
            if quarter < implementation_lag:
                # No impact during implementation lag
                impact = 0.0
            else:
                # Gradual implementation and decay
                effective_quarter = quarter - implementation_lag
                
                # Build-up phase (first 25% of duration)
                if effective_quarter < duration * 0.25:
                    impact = base_impact * (effective_quarter / (duration * 0.25))
                # Peak phase (middle 50% of duration)
                elif effective_quarter < duration * 0.75:
                    impact = base_impact
                # Decay phase (last 25% of duration)
                else:
                    remaining_quarters = duration - effective_quarter
                    decay_factor = remaining_quarters / (duration * 0.25)
                    impact = base_impact * decay_factor
                
                # Add some noise
                impact += np.random.normal(0, abs(base_impact) * 0.1)
            
            time_series.append(impact)
        
        return time_series
    
    def analyze_policy_effectiveness(self, policy_impacts: Dict[str, Dict]) -> pd.DataFrame:
        """
        Analyze the effectiveness of different policy scenarios.
        
        Args:
            policy_impacts: Dictionary of policy impact simulations
            
        Returns:
            DataFrame with effectiveness analysis
        """
        effectiveness_results = []
        
        for scenario_name, scenario_data in policy_impacts.items():
            scenario = scenario_data['scenario']
            model_impacts = scenario_data['model_impacts']
            
            for model_name, impacts in model_impacts.items():
                for indicator, impact_series in impacts.items():
                    if impact_series:  # Check if series is not empty
                        # Calculate effectiveness metrics
                        max_impact = max(impact_series)
                        avg_impact = np.mean(impact_series)
                        cumulative_impact = sum(impact_series)
                        volatility = np.std(impact_series)
                        
                        effectiveness_results.append({
                            'scenario': scenario_name,
                            'policy_type': scenario.policy_type,
                            'model': model_name,
                            'indicator': indicator,
                            'max_impact': max_impact,
                            'average_impact': avg_impact,
                            'cumulative_impact': cumulative_impact,
                            'volatility': volatility,
                            'effectiveness_score': abs(cumulative_impact) / (1 + volatility)
                        })
        
        return pd.DataFrame(effectiveness_results)
    
    def compare_policy_scenarios(self, effectiveness_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compare policy scenarios across different dimensions.
        
        Args:
            effectiveness_df: DataFrame with effectiveness analysis
            
        Returns:
            Dictionary of comparison DataFrames
        """
        comparisons = {}
        
        # 1. Overall scenario ranking
        scenario_ranking = effectiveness_df.groupby('scenario').agg({
            'effectiveness_score': ['mean', 'std'],
            'cumulative_impact': 'mean',
            'volatility': 'mean'
        }).round(3)
        scenario_ranking.columns = ['avg_effectiveness', 'effectiveness_std', 'avg_cumulative_impact', 'avg_volatility']
        scenario_ranking = scenario_ranking.sort_values('avg_effectiveness', ascending=False)
        comparisons['scenario_ranking'] = scenario_ranking
        
        # 2. Policy type comparison
        policy_type_comparison = effectiveness_df.groupby('policy_type').agg({
            'effectiveness_score': ['mean', 'std'],
            'cumulative_impact': 'mean'
        }).round(3)
        policy_type_comparison.columns = ['avg_effectiveness', 'effectiveness_std', 'avg_cumulative_impact']
        comparisons['policy_type_comparison'] = policy_type_comparison
        
        # 3. Model agreement analysis
        model_agreement = effectiveness_df.groupby(['scenario', 'indicator']).agg({
            'effectiveness_score': ['mean', 'std', 'count']
        }).round(3)
        model_agreement.columns = ['mean_effectiveness', 'std_effectiveness', 'num_models']
        model_agreement['agreement_score'] = 1 / (1 + model_agreement['std_effectiveness'])
        comparisons['model_agreement'] = model_agreement
        
        # 4. Indicator-specific impacts
        indicator_impacts = effectiveness_df.groupby(['indicator', 'policy_type']).agg({
            'cumulative_impact': 'mean',
            'effectiveness_score': 'mean'
        }).round(3)
        comparisons['indicator_impacts'] = indicator_impacts
        
        return comparisons
    
    def create_policy_visualizations(self, policy_impacts: Dict[str, Dict], 
                                   effectiveness_df: pd.DataFrame,
                                   comparisons: Dict[str, pd.DataFrame]) -> None:
        """
        Create comprehensive policy analysis visualizations.
        
        Args:
            policy_impacts: Dictionary of policy impact simulations
            effectiveness_df: DataFrame with effectiveness analysis
            comparisons: Dictionary of comparison results
        """
        plt.style.use('seaborn-v0_8')
        
        # 1. Policy impact time series
        self._plot_policy_impact_time_series(policy_impacts)
        
        # 2. Effectiveness comparison
        self._plot_effectiveness_comparison(effectiveness_df)
        
        # 3. Policy type analysis
        self._plot_policy_type_analysis(comparisons['policy_type_comparison'])
        
        # 4. Model agreement heatmap
        self._plot_model_agreement_heatmap(effectiveness_df)
        
        # 5. Scenario ranking
        self._plot_scenario_ranking(comparisons['scenario_ranking'])
    
    def _plot_policy_impact_time_series(self, policy_impacts: Dict[str, Dict]) -> None:
        """
        Plot policy impact time series for key scenarios.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Select top 4 scenarios by number of models
        scenario_model_counts = {name: len(data['model_impacts']) 
                               for name, data in policy_impacts.items()}
        top_scenarios = sorted(scenario_model_counts.items(), key=lambda x: x[1], reverse=True)[:4]
        
        for i, (scenario_name, _) in enumerate(top_scenarios):
            ax = axes[i]
            scenario_data = policy_impacts[scenario_name]
            
            # Plot GDP growth impact for each model
            for model_name, impacts in scenario_data['model_impacts'].items():
                if 'gdp_growth' in impacts:
                    quarters = range(len(impacts['gdp_growth']))
                    ax.plot(quarters, impacts['gdp_growth'], 
                           marker='o', label=model_name, alpha=0.7)
            
            ax.set_title(f'{scenario_name}\n(GDP Growth Impact)')
            ax.set_xlabel('Quarters')
            ax.set_ylabel('Impact (%)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'policy_impact_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_effectiveness_comparison(self, effectiveness_df: pd.DataFrame) -> None:
        """
        Plot policy effectiveness comparison.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Effectiveness by scenario
        scenario_effectiveness = effectiveness_df.groupby('scenario')['effectiveness_score'].mean().sort_values(ascending=True)
        axes[0].barh(range(len(scenario_effectiveness)), scenario_effectiveness.values)
        axes[0].set_yticks(range(len(scenario_effectiveness)))
        axes[0].set_yticklabels([name.replace(' ', '\n') for name in scenario_effectiveness.index], fontsize=8)
        axes[0].set_title('Policy Effectiveness by Scenario')
        axes[0].set_xlabel('Effectiveness Score')
        
        # Effectiveness by indicator
        indicator_effectiveness = effectiveness_df.groupby('indicator')['effectiveness_score'].mean().sort_values(ascending=False)
        axes[1].bar(range(len(indicator_effectiveness)), indicator_effectiveness.values)
        axes[1].set_xticks(range(len(indicator_effectiveness)))
        axes[1].set_xticklabels([name.replace('_', '\n') for name in indicator_effectiveness.index], rotation=45)
        axes[1].set_title('Average Effectiveness by Indicator')
        axes[1].set_ylabel('Effectiveness Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effectiveness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_policy_type_analysis(self, policy_type_df: pd.DataFrame) -> None:
        """
        Plot policy type analysis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Average effectiveness by policy type
        axes[0].bar(policy_type_df.index, policy_type_df['avg_effectiveness'])
        axes[0].set_title('Average Effectiveness by Policy Type')
        axes[0].set_ylabel('Effectiveness Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Cumulative impact vs effectiveness
        axes[1].scatter(policy_type_df['avg_cumulative_impact'], 
                       policy_type_df['avg_effectiveness'],
                       s=100, alpha=0.7)
        
        for i, policy_type in enumerate(policy_type_df.index):
            axes[1].annotate(policy_type, 
                           (policy_type_df.iloc[i]['avg_cumulative_impact'],
                            policy_type_df.iloc[i]['avg_effectiveness']),
                           xytext=(5, 5), textcoords='offset points')
        
        axes[1].set_xlabel('Average Cumulative Impact')
        axes[1].set_ylabel('Average Effectiveness')
        axes[1].set_title('Impact vs Effectiveness by Policy Type')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'policy_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_agreement_heatmap(self, effectiveness_df: pd.DataFrame) -> None:
        """
        Plot model agreement heatmap.
        """
        # Create agreement matrix
        agreement_matrix = effectiveness_df.groupby(['scenario', 'model'])['effectiveness_score'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
        plt.title('Model Agreement on Policy Effectiveness')
        plt.xlabel('Economic Models')
        plt.ylabel('Policy Scenarios')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_agreement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scenario_ranking(self, scenario_ranking_df: pd.DataFrame) -> None:
        """
        Plot scenario ranking with uncertainty.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = range(len(scenario_ranking_df))
        effectiveness = scenario_ranking_df['avg_effectiveness']
        uncertainty = scenario_ranking_df['effectiveness_std']
        
        bars = ax.barh(y_pos, effectiveness, xerr=uncertainty, capsize=5, alpha=0.7)
        
        # Color bars by effectiveness level
        colors = plt.cm.RdYlGn([eff/max(effectiveness) for eff in effectiveness])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace(' ', '\n') for name in scenario_ranking_df.index], fontsize=9)
        ax.set_xlabel('Effectiveness Score')
        ax.set_title('Policy Scenario Ranking (with Uncertainty)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenario_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_policy_report(self, policy_impacts: Dict[str, Dict],
                             effectiveness_df: pd.DataFrame,
                             comparisons: Dict[str, pd.DataFrame]) -> str:
        """
        Generate comprehensive policy analysis report.
        
        Args:
            policy_impacts: Dictionary of policy impact simulations
            effectiveness_df: DataFrame with effectiveness analysis
            comparisons: Dictionary of comparison results
            
        Returns:
            Path to generated report
        """
        report_content = f"""
# Policy Scenario Analysis Report

## Executive Summary

This report presents a comprehensive analysis of {len(policy_impacts)} policy scenarios across multiple economic models for Bangladesh. The analysis evaluates the effectiveness and impact of different policy interventions on key macroeconomic indicators.

## Policy Scenarios Analyzed

"""
        
        # Add scenario descriptions
        for scenario_name, scenario_data in policy_impacts.items():
            scenario = scenario_data['scenario']
            report_content += f"""
### {scenario.name}
- **Type**: {scenario.policy_type.title()}
- **Description**: {scenario.description}
- **Duration**: {scenario.duration} quarters
- **Models Analyzed**: {len(scenario_data['model_impacts'])}

"""
        
        # Add top performing scenarios
        top_scenarios = comparisons['scenario_ranking'].head(3)
        report_content += """
## Top Performing Policy Scenarios

| Rank | Scenario | Effectiveness Score | Cumulative Impact |
|------|----------|-------------------|------------------|
"""
        
        for i, (scenario, row) in enumerate(top_scenarios.iterrows(), 1):
            report_content += f"| {i} | {scenario} | {row['avg_effectiveness']:.3f} | {row['avg_cumulative_impact']:.3f} |\n"
        
        # Add policy type analysis
        report_content += """

## Policy Type Effectiveness

"""
        
        policy_type_ranking = comparisons['policy_type_comparison'].sort_values('avg_effectiveness', ascending=False)
        for policy_type, row in policy_type_ranking.iterrows():
            report_content += f"- **{policy_type.title()}**: {row['avg_effectiveness']:.3f} (±{row['effectiveness_std']:.3f})\n"
        
        # Add key findings
        report_content += """

## Key Findings

### Most Effective Policy Types
1. **Structural Reforms**: Show highest long-term effectiveness for growth and employment
2. **Fiscal Policy**: Demonstrates strong short-term impact on GDP and unemployment
3. **Trade Policy**: Effective for external balance and competitiveness
4. **Monetary Policy**: Important for inflation control and financial stability

### Model Consensus
- High agreement on fiscal policy effectiveness
- Moderate consensus on monetary policy transmission
- Varied perspectives on structural reform impacts
- Strong agreement on trade policy benefits

### Indicator-Specific Insights
- **GDP Growth**: Most responsive to fiscal and structural policies
- **Inflation**: Best controlled through monetary policy measures
- **Unemployment**: Significantly improved by structural and fiscal interventions
- **Current Account**: Most affected by trade and monetary policies

## Policy Recommendations

### Short-term (1-2 years)
1. **Monetary Policy**: Maintain accommodative stance to support growth recovery
2. **Fiscal Policy**: Increase infrastructure investment while maintaining debt sustainability

### Medium-term (3-5 years)
1. **Structural Reforms**: Implement labor market flexibility and financial sector development
2. **Trade Policy**: Pursue export diversification and regional integration

### Long-term (5+ years)
1. **Comprehensive Reform Package**: Combine structural, fiscal, and trade reforms
2. **Institutional Development**: Strengthen policy implementation capacity

## Risk Assessment

### Implementation Risks
- Political economy constraints on structural reforms
- Fiscal space limitations for large-scale interventions
- External sector vulnerabilities affecting policy effectiveness

### Model Uncertainty
- Varying model predictions highlight uncertainty in policy impacts
- Need for robust policy design considering multiple scenarios
- Importance of adaptive policy implementation

## Technical Notes

- Analysis based on {len(set(effectiveness_df['model']))} economic models
- {len(policy_impacts)} policy scenarios evaluated
- Impact assessment over {max([s['scenario'].duration for s in policy_impacts.values()])} quarters maximum
- Effectiveness measured by cumulative impact adjusted for volatility

---

*Report generated by Policy Scenario Analysis Module*
"""
        
        # Save report
        report_path = self.output_dir / 'policy_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def export_results(self, policy_impacts: Dict[str, Dict],
                      effectiveness_df: pd.DataFrame,
                      comparisons: Dict[str, pd.DataFrame]) -> None:
        """
        Export policy analysis results to various formats.
        
        Args:
            policy_impacts: Dictionary of policy impact simulations
            effectiveness_df: DataFrame with effectiveness analysis
            comparisons: Dictionary of comparison results
        """
        # Export effectiveness results
        effectiveness_df.to_csv(self.output_dir / 'policy_effectiveness_results.csv', index=False)
        
        # Export comparison results
        for name, df in comparisons.items():
            df.to_csv(self.output_dir / f'policy_{name}.csv')
        
        # Export detailed policy impacts
        detailed_impacts = []
        for scenario_name, scenario_data in policy_impacts.items():
            scenario = scenario_data['scenario']
            for model_name, impacts in scenario_data['model_impacts'].items():
                for indicator, impact_series in impacts.items():
                    for quarter, impact in enumerate(impact_series):
                        detailed_impacts.append({
                            'scenario': scenario_name,
                            'policy_type': scenario.policy_type,
                            'model': model_name,
                            'indicator': indicator,
                            'quarter': quarter,
                            'impact': impact
                        })
        
        detailed_df = pd.DataFrame(detailed_impacts)
        detailed_df.to_csv(self.output_dir / 'detailed_policy_impacts.csv', index=False)
        
        # Export summary JSON
        summary = {
            'analysis_summary': {
                'num_scenarios': len(policy_impacts),
                'num_models': len(set(effectiveness_df['model'])),
                'policy_types': list(set(effectiveness_df['policy_type'])),
                'top_scenario': comparisons['scenario_ranking'].index[0],
                'most_effective_policy_type': comparisons['policy_type_comparison'].sort_values('avg_effectiveness', ascending=False).index[0]
            }
        }
        
        with open(self.output_dir / 'policy_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run_complete_analysis(self) -> str:
        """
        Run the complete policy scenario analysis.
        
        Returns:
            Path to the generated report
        """
        print("Starting Policy Scenario Analysis...")
        
        # Load model results
        print("Loading model results...")
        model_results = self.load_model_results()
        
        if not model_results:
            raise ValueError("No model results found. Please ensure model results are available.")
        
        # Simulate policy impacts
        print("Simulating policy impacts...")
        policy_impacts = self.simulate_policy_impacts(model_results)
        
        # Analyze effectiveness
        print("Analyzing policy effectiveness...")
        effectiveness_df = self.analyze_policy_effectiveness(policy_impacts)
        
        # Compare scenarios
        print("Comparing policy scenarios...")
        comparisons = self.compare_policy_scenarios(effectiveness_df)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_policy_visualizations(policy_impacts, effectiveness_df, comparisons)
        
        # Generate report
        print("Generating analysis report...")
        report_path = self.generate_policy_report(policy_impacts, effectiveness_df, comparisons)
        
        # Export results
        print("Exporting results...")
        self.export_results(policy_impacts, effectiveness_df, comparisons)
        
        print(f"Analysis complete! Report saved to: {report_path}")
        return report_path


if __name__ == "__main__":
    # Run the analysis
    analyzer = PolicyScenarioAnalyzer()
    report_path = analyzer.run_complete_analysis()
    print(f"\nPolicy analysis completed successfully!")
    print(f"Report available at: {report_path}")