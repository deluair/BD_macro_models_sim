# Bangladesh Macroeconomic Models - User Guide

This comprehensive guide explains how to use the Bangladesh Macroeconomic Models framework to analyze the economy using various theoretical models with real data.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Sources and Loading](#data-sources-and-loading)
3. [Model Overview](#model-overview)
4. [DSGE Model Usage](#dsge-model-usage)
5. [CGE Model Usage](#cge-model-usage)
6. [SVAR Model Usage](#svar-model-usage)
7. [Life Cycle Models (OLG)](#life-cycle-models-olg)
8. [Game Theoretic Models](#game-theoretic-models)
9. [Agent-Based Models](#agent-based-models)
10. [Policy Analysis](#policy-analysis)
11. [Forecasting](#forecasting)
12. [Visualization](#visualization)
13. [Advanced Usage](#advanced-usage)
14. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone or download the project
cd BD_macro_models_sim

# Install dependencies
pip install -r requirements.txt

# Run example analysis
python example_analysis.py
```

### Basic Usage

```python
from main import BangladeshMacroModels
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize framework
models = BangladeshMacroModels(config)

# Load data
models.load_data()

# Run DSGE analysis
dsge_results = models.run_dsge_analysis()

# Generate forecasts
forecasts = models.generate_forecasts(horizon=12)

# Create dashboard
models.create_dashboard()
```

## Data Sources and Loading

### Available Data Sources

1. **Bangladesh Bank (Central Bank)**
   - Monetary policy rates
   - Exchange rates
   - Money supply
   - Banking sector data

2. **Bangladesh Bureau of Statistics (BBS)**
   - GDP data
   - Inflation (CPI)
   - Employment statistics
   - Trade data

3. **International Sources**
   - World Bank
   - IMF
   - Asian Development Bank
   - Trading Economics
   - FRED (Federal Reserve Economic Data)

### Data Loading Example

```python
from utils.data_processing.data_loader import BangladeshDataLoader

# Initialize data loader
data_loader = BangladeshDataLoader(config)

# Load specific source
bb_data = data_loader.load_bangladesh_bank_data()
bbs_data = data_loader.load_bbs_data()
world_bank_data = data_loader.load_world_bank_data()

# Load all data
all_data = data_loader.load_all_data()

# Get data summary
summary = data_loader.get_data_summary(all_data)
print(summary)
```

### Data Processing

```python
# Clean and process data
processed_data = data_loader.process_data(all_data)

# Handle missing values
filled_data = data_loader.fill_missing_values(processed_data)

# Create quarterly aggregates
quarterly_data = data_loader.to_quarterly(filled_data)
```

## Model Overview

### 1. Dynamic Stochastic General Equilibrium (DSGE)
- **Purpose**: Business cycle analysis, monetary policy
- **Features**: Microfounded, forward-looking agents
- **Best for**: Short to medium-term analysis

### 2. Computable General Equilibrium (CGE)
- **Purpose**: Structural policy analysis
- **Features**: Detailed sectoral interactions
- **Best for**: Trade policy, tax reform analysis

### 3. Structural Vector Autoregression (SVAR)
- **Purpose**: Empirical policy analysis
- **Features**: Data-driven identification
- **Best for**: Monetary transmission analysis

### 4. Overlapping Generations (OLG)
- **Purpose**: Long-term demographic analysis
- **Features**: Life-cycle behavior
- **Best for**: Pension reform, aging population

### 5. Game Theoretic Models
- **Purpose**: Strategic interactions
- **Features**: Multiple players, Nash equilibrium
- **Best for**: International trade negotiations

### 6. Agent-Based Models (ABM)
- **Purpose**: Complex adaptive systems
- **Features**: Heterogeneous agents, emergent behavior
- **Best for**: Financial stability, inequality

## DSGE Model Usage

### Basic DSGE Analysis

```python
from models.dsge.dsge_model import DSGEModel

# Initialize model
dsge = DSGEModel(config['dsge'], data)

# Compute steady state
steady_state = dsge.compute_steady_state()
print("Steady State:", steady_state)

# Solve model
solution = dsge.solve_model()
print("Model stable:", solution['stable'])

# Run simulation
simulation = dsge.simulate(periods=100)
```

### Impulse Response Analysis

```python
# Technology shock
tech_irf = dsge.impulse_response(
    shock_type='technology',
    shock_size=1.0,
    periods=20
)

# Monetary policy shock
monetary_irf = dsge.impulse_response(
    shock_type='monetary',
    shock_size=0.25,  # 25 basis points
    periods=20
)

# Plot responses
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
variables = ['y', 'pi', 'r', 'c']
titles = ['Output', 'Inflation', 'Interest Rate', 'Consumption']

for i, (var, title) in enumerate(zip(variables, titles)):
    ax = axes[i//2, i%2]
    ax.plot(tech_irf[var], label='Technology Shock')
    ax.plot(monetary_irf[var], label='Monetary Shock')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

### Parameter Estimation

```python
# Maximum Likelihood Estimation
mle_results = dsge.estimate_mle(
    data=quarterly_data,
    method='BFGS'
)

# Bayesian Estimation
priors = {
    'beta': {'dist': 'beta', 'params': [0.99, 0.005]},
    'sigma': {'dist': 'gamma', 'params': [1.0, 0.5]},
    'phi_pi': {'dist': 'gamma', 'params': [1.5, 0.25]}
}

bayesian_results = dsge.estimate_bayesian(
    data=quarterly_data,
    priors=priors,
    draws=10000,
    chains=4
)
```

### Policy Analysis

```python
# Define policy experiment
policy_changes = {
    'phi_pi': 2.0,  # More aggressive inflation targeting
    'phi_y': 0.8,   # Stronger output response
    'rho_r': 0.5    # Less interest rate smoothing
}

# Run policy analysis
policy_results = dsge.policy_analysis(policy_changes)

# Compare with baseline
baseline_irf = dsge.impulse_response('monetary')
policy_irf = policy_results['impulse_responses']['monetary']

# Plot comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(baseline_irf['y'], label='Baseline')
plt.plot(policy_irf['y'], label='Policy Change')
plt.title('Output Response to Monetary Shock')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(baseline_irf['pi'], label='Baseline')
plt.plot(policy_irf['pi'], label='Policy Change')
plt.title('Inflation Response to Monetary Shock')
plt.legend()

plt.tight_layout()
plt.show()
```

## CGE Model Usage

### Basic CGE Analysis

```python
from models.cge.cge_model import CGEModel

# Initialize CGE model
cge = CGEModel(config['cge'], data)

# Calibrate model
cge.calibrate(base_year=2020)

# Solve baseline
baseline = cge.solve_baseline()

# Run policy simulation
policy_scenario = {
    'tariff_reduction': 0.5,  # 50% tariff reduction
    'export_subsidy': 0.1     # 10% export subsidy
}

policy_results = cge.policy_simulation(policy_scenario)

# Compare results
print("GDP change:", policy_results['gdp_change'])
print("Welfare change:", policy_results['welfare_change'])
print("Sectoral output changes:", policy_results['sectoral_changes'])
```

### Trade Policy Analysis

```python
# Analyze different trade scenarios
trade_scenarios = {
    'free_trade': {'tariffs': 0.0, 'quotas': None},
    'protection': {'tariffs': 0.25, 'quotas': {'textiles': 0.8}},
    'regional_fta': {'tariffs_asia': 0.0, 'tariffs_others': 0.15}
}

trade_results = {}
for scenario, params in trade_scenarios.items():
    results = cge.trade_policy_analysis(params)
    trade_results[scenario] = results

# Visualize results
import pandas as pd

results_df = pd.DataFrame({
    scenario: {
        'GDP': results['gdp_change'],
        'Exports': results['export_change'],
        'Imports': results['import_change'],
        'Welfare': results['welfare_change']
    }
    for scenario, results in trade_results.items()
}).T

results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Trade Policy Scenarios - Economic Impact')
plt.ylabel('Percentage Change from Baseline')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## SVAR Model Usage

### Basic SVAR Analysis

```python
from models.svar.svar_model import SVARModel

# Prepare data
svar_data = quarterly_data[['gdp_growth', 'inflation', 'policy_rate', 'exchange_rate']]

# Initialize SVAR
svar = SVARModel(config['svar'])

# Estimate model
svar.estimate(svar_data, lags=4)

# Identify structural shocks
identification = {
    'method': 'cholesky',
    'ordering': ['policy_rate', 'exchange_rate', 'gdp_growth', 'inflation']
}

svar.identify_shocks(identification)

# Compute impulse responses
irf_results = svar.impulse_response(periods=20)

# Plot results
svar.plot_impulse_responses(irf_results)
```

### Monetary Policy Transmission

```python
# Analyze monetary transmission channels
transmission_results = svar.monetary_transmission_analysis()

# Interest rate channel
interest_channel = transmission_results['interest_rate_channel']
print("Interest rate pass-through:", interest_channel['pass_through'])
print("Peak effect on GDP:", interest_channel['peak_gdp_effect'])

# Exchange rate channel
exchange_channel = transmission_results['exchange_rate_channel']
print("Exchange rate elasticity:", exchange_channel['elasticity'])
print("Import price effect:", exchange_channel['import_price_effect'])

# Credit channel
credit_channel = transmission_results['credit_channel']
print("Credit growth response:", credit_channel['credit_response'])
print("Investment effect:", credit_channel['investment_effect'])
```

## Life Cycle Models (OLG)

### Basic OLG Analysis

```python
from models.olg.olg_model import OLGModel

# Initialize OLG model
olg = OLGModel(config['olg'], data)

# Set demographic parameters
demographics = {
    'birth_rate': 0.02,
    'death_rate': 0.015,
    'retirement_age': 65,
    'life_expectancy': 75
}

olg.set_demographics(demographics)

# Solve steady state
steady_state = olg.solve_steady_state()

# Analyze pension reform
pension_reform = {
    'contribution_rate': 0.12,  # Increase from 10% to 12%
    'replacement_rate': 0.6,    # 60% replacement rate
    'retirement_age': 67        # Increase retirement age
}

reform_results = olg.pension_reform_analysis(pension_reform)

print("Welfare effect:", reform_results['welfare_change'])
print("Fiscal impact:", reform_results['fiscal_impact'])
print("Intergenerational effects:", reform_results['generational_effects'])
```

### Demographic Transition Analysis

```python
# Analyze aging population effects
aging_scenarios = {
    'baseline': {'fertility_rate': 2.1, 'life_expectancy': 75},
    'aging': {'fertility_rate': 1.8, 'life_expectancy': 80},
    'rapid_aging': {'fertility_rate': 1.5, 'life_expectancy': 85}
}

aging_results = {}
for scenario, params in aging_scenarios.items():
    results = olg.demographic_transition_analysis(params)
    aging_results[scenario] = results

# Plot demographic dividend
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, (scenario, results) in enumerate(aging_results.items()):
    ax = axes[i//2, i%2]
    ax.plot(results['years'], results['dependency_ratio'], label='Dependency Ratio')
    ax.plot(results['years'], results['gdp_per_capita'], label='GDP per Capita')
    ax.set_title(f'{scenario.title()} Scenario')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Game Theoretic Models

### Trade Negotiation Game

```python
from models.game_theory.trade_game import TradeNegotiationGame

# Initialize trade game
trade_game = TradeNegotiationGame(config['game_theory'])

# Define players (countries)
players = {
    'bangladesh': {
        'export_sectors': ['textiles', 'leather', 'jute'],
        'import_needs': ['machinery', 'oil', 'food'],
        'bargaining_power': 0.3
    },
    'india': {
        'export_sectors': ['machinery', 'chemicals', 'services'],
        'import_needs': ['textiles', 'raw_materials'],
        'bargaining_power': 0.7
    }
}

# Set up game
trade_game.setup_players(players)

# Define payoff structure
payoffs = trade_game.calculate_payoffs()

# Find Nash equilibrium
equilibrium = trade_game.find_nash_equilibrium()

print("Nash Equilibrium:")
for player, strategy in equilibrium.items():
    print(f"{player}: {strategy}")

# Analyze cooperative solution
cooperative = trade_game.cooperative_solution()
print("\nCooperative Solution:")
print(f"Total welfare: {cooperative['total_welfare']}")
print(f"Welfare distribution: {cooperative['welfare_distribution']}")
```

### Monetary Policy Coordination

```python
from models.game_theory.policy_coordination import PolicyCoordinationGame

# Central bank coordination game
policy_game = PolicyCoordinationGame(config['game_theory'])

# Define central banks
central_banks = {
    'bangladesh_bank': {
        'inflation_target': 0.055,  # 5.5%
        'output_weight': 0.5,
        'exchange_rate_concern': 0.3
    },
    'fed': {
        'inflation_target': 0.02,   # 2%
        'output_weight': 0.5,
        'exchange_rate_concern': 0.1
    },
    'ecb': {
        'inflation_target': 0.02,   # 2%
        'output_weight': 0.4,
        'exchange_rate_concern': 0.2
    }
}

# Analyze coordination benefits
coordination_results = policy_game.analyze_coordination(central_banks)

print("Coordination Benefits:")
print(f"Global welfare gain: {coordination_results['welfare_gain']}")
print(f"Spillover reduction: {coordination_results['spillover_reduction']}")
print(f"Optimal policies: {coordination_results['optimal_policies']}")
```

## Agent-Based Models

### Financial Market ABM

```python
from models.abm.financial_abm import FinancialMarketABM

# Initialize ABM
abm = FinancialMarketABM(config['abm'])

# Define agent types
agent_types = {
    'fundamentalists': {
        'proportion': 0.4,
        'strategy': 'fundamental_value',
        'risk_aversion': 0.5
    },
    'chartists': {
        'proportion': 0.3,
        'strategy': 'technical_analysis',
        'risk_aversion': 0.3
    },
    'noise_traders': {
        'proportion': 0.3,
        'strategy': 'random',
        'risk_aversion': 0.8
    }
}

# Run simulation
abm.setup_agents(agent_types, n_agents=1000)
simulation_results = abm.run_simulation(periods=1000)

# Analyze results
print("Market Statistics:")
print(f"Average return: {simulation_results['returns'].mean():.4f}")
print(f"Volatility: {simulation_results['returns'].std():.4f}")
print(f"Sharpe ratio: {simulation_results['sharpe_ratio']:.4f}")

# Plot price dynamics
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(simulation_results['prices'])
plt.title('Price Dynamics')
plt.ylabel('Price')

plt.subplot(2, 1, 2)
plt.plot(simulation_results['returns'])
plt.title('Returns')
plt.ylabel('Return')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
```

### Banking System ABM

```python
from models.abm.banking_abm import BankingSystemABM

# Initialize banking ABM
banking_abm = BankingSystemABM(config['abm'])

# Define bank characteristics
bank_types = {
    'large_banks': {
        'number': 10,
        'capital_ratio': 0.12,
        'risk_appetite': 0.6,
        'market_share': 0.7
    },
    'small_banks': {
        'number': 30,
        'capital_ratio': 0.10,
        'risk_appetite': 0.8,
        'market_share': 0.3
    }
}

# Run stress test
stress_scenarios = {
    'mild_recession': {'gdp_shock': -0.02, 'default_rate_increase': 0.01},
    'severe_recession': {'gdp_shock': -0.05, 'default_rate_increase': 0.03},
    'financial_crisis': {'gdp_shock': -0.08, 'default_rate_increase': 0.05}
}

stress_results = {}
for scenario, params in stress_scenarios.items():
    results = banking_abm.stress_test(bank_types, params)
    stress_results[scenario] = results

# Analyze systemic risk
for scenario, results in stress_results.items():
    print(f"\n{scenario.title()}:")
    print(f"  Banks failed: {results['failed_banks']}")
    print(f"  System capital ratio: {results['system_capital_ratio']:.3f}")
    print(f"  Credit contraction: {results['credit_contraction']:.2%}")
    print(f"  Contagion risk: {results['contagion_risk']:.3f}")
```

## Policy Analysis

### Comprehensive Policy Framework

```python
from analysis.policy_analysis import PolicyAnalysisFramework

# Initialize policy framework
policy_framework = PolicyAnalysisFramework(config)

# Load all models
policy_framework.load_models(data)

# Define policy package
policy_package = {
    'monetary_policy': {
        'interest_rate_change': -0.005,  # 50 bps cut
        'reserve_requirement_change': -0.01  # 1pp reduction
    },
    'fiscal_policy': {
        'government_spending_increase': 0.02,  # 2% of GDP
        'tax_rate_change': -0.01  # 1pp reduction
    },
    'structural_reforms': {
        'trade_liberalization': 0.3,  # 30% tariff reduction
        'financial_deepening': 0.2,   # 20% increase in credit
        'labor_market_flexibility': 0.15  # 15% improvement
    }
}

# Analyze policy package across models
policy_results = policy_framework.analyze_policy_package(policy_package)

# Compare model predictions
comparison = policy_framework.compare_model_predictions(policy_results)

print("Policy Impact Comparison:")
for indicator in ['gdp_growth', 'inflation', 'unemployment', 'current_account']:
    print(f"\n{indicator.title()}:")
    for model, result in comparison[indicator].items():
        print(f"  {model}: {result:.3f}")
```

### Scenario Analysis

```python
# Define multiple scenarios
scenarios = {
    'baseline': {
        'global_growth': 0.03,
        'oil_price_change': 0.0,
        'trade_war_impact': 0.0,
        'climate_shock': 0.0
    },
    'optimistic': {
        'global_growth': 0.04,
        'oil_price_change': -0.1,
        'trade_war_impact': 0.0,
        'climate_shock': 0.0
    },
    'pessimistic': {
        'global_growth': 0.02,
        'oil_price_change': 0.2,
        'trade_war_impact': -0.02,
        'climate_shock': -0.01
    },
    'climate_stress': {
        'global_growth': 0.025,
        'oil_price_change': 0.05,
        'trade_war_impact': 0.0,
        'climate_shock': -0.03
    }
}

# Run scenario analysis
scenario_results = {}
for scenario_name, scenario_params in scenarios.items():
    results = policy_framework.scenario_analysis(scenario_params)
    scenario_results[scenario_name] = results

# Visualize scenario outcomes
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
indicators = ['gdp_growth', 'inflation', 'unemployment', 'fiscal_balance']
titles = ['GDP Growth (%)', 'Inflation (%)', 'Unemployment (%)', 'Fiscal Balance (% GDP)']

for i, (indicator, title) in enumerate(zip(indicators, titles)):
    ax = axes[i//2, i%2]
    
    scenario_values = [scenario_results[scenario][indicator] for scenario in scenarios.keys()]
    scenario_names = list(scenarios.keys())
    
    bars = ax.bar(scenario_names, scenario_values)
    ax.set_title(title)
    ax.set_ylabel('Percentage')
    
    # Color bars based on performance
    for j, bar in enumerate(bars):
        if scenario_values[j] > 0 and indicator in ['gdp_growth']:
            bar.set_color('green')
        elif scenario_values[j] < 0 and indicator in ['inflation', 'unemployment']:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/scenario_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Forecasting

### Multi-Model Forecasting

```python
from analysis.forecasting import MultiModelForecaster

# Initialize forecaster
forecaster = MultiModelForecaster(config)

# Load models
forecaster.load_models(data)

# Generate forecasts from different models
forecast_horizon = 12  # 12 quarters

model_forecasts = {
    'dsge': forecaster.dsge_forecast(horizon=forecast_horizon),
    'svar': forecaster.svar_forecast(horizon=forecast_horizon),
    'var': forecaster.var_forecast(horizon=forecast_horizon),
    'arima': forecaster.arima_forecast(horizon=forecast_horizon)
}

# Combine forecasts
combined_forecast = forecaster.combine_forecasts(
    model_forecasts,
    weights={'dsge': 0.3, 'svar': 0.3, 'var': 0.2, 'arima': 0.2}
)

# Evaluate forecast accuracy (if historical data available)
if len(data['combined']) > forecast_horizon:
    accuracy_metrics = forecaster.evaluate_forecasts(
        model_forecasts,
        actual_data=data['combined'][-forecast_horizon:]
    )
    
    print("Forecast Accuracy (RMSE):")
    for model, metrics in accuracy_metrics.items():
        print(f"  {model}: {metrics['rmse']:.4f}")

# Plot forecasts
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
variables = ['gdp_growth', 'inflation', 'unemployment', 'current_account']
titles = ['GDP Growth', 'Inflation', 'Unemployment', 'Current Account']

for i, (var, title) in enumerate(zip(variables, titles)):
    ax = axes[i//2, i%2]
    
    # Plot historical data
    historical = data['combined'][var][-20:]  # Last 20 quarters
    ax.plot(historical.index, historical.values, 'k-', linewidth=2, label='Historical')
    
    # Plot model forecasts
    colors = ['blue', 'red', 'green', 'orange']
    for j, (model, color) in enumerate(zip(model_forecasts.keys(), colors)):
        if var in model_forecasts[model].columns:
            forecast_data = model_forecasts[model][var]
            ax.plot(forecast_data.index, forecast_data.values, 
                   color=color, linestyle='--', label=f'{model.upper()}')
    
    # Plot combined forecast
    if var in combined_forecast.columns:
        combined_data = combined_forecast[var]
        ax.plot(combined_data.index, combined_data.values, 
               'purple', linewidth=3, label='Combined')
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/multi_model_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Density Forecasting

```python
# Generate density forecasts with uncertainty bands
density_forecasts = forecaster.density_forecast(
    horizon=forecast_horizon,
    confidence_levels=[0.68, 0.90, 0.95]
)

# Plot density forecasts
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, (var, title) in enumerate(zip(variables, titles)):
    ax = axes[i//2, i%2]
    
    # Historical data
    historical = data['combined'][var][-20:]
    ax.plot(historical.index, historical.values, 'k-', linewidth=2, label='Historical')
    
    # Density forecast
    if var in density_forecasts:
        forecast_data = density_forecasts[var]
        
        # Central forecast
        ax.plot(forecast_data.index, forecast_data['median'], 
               'blue', linewidth=2, label='Median Forecast')
        
        # Confidence bands
        ax.fill_between(forecast_data.index, 
                       forecast_data['lower_95'], forecast_data['upper_95'],
                       alpha=0.2, color='blue', label='95% CI')
        ax.fill_between(forecast_data.index, 
                       forecast_data['lower_68'], forecast_data['upper_68'],
                       alpha=0.4, color='blue', label='68% CI')
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/density_forecasts.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Visualization

### Interactive Dashboard

```python
from visualization.dashboard import BangladeshEconomyDashboard
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize dashboard
dashboard = BangladeshEconomyDashboard(config)

# Load data and models
dashboard.load_data(data)
dashboard.load_models()

# Create interactive plots
interactive_plots = dashboard.create_interactive_plots()

# Economic indicators dashboard
indicators_dash = dashboard.create_indicators_dashboard()

# Model comparison dashboard
model_comparison_dash = dashboard.create_model_comparison_dashboard()

# Policy analysis dashboard
policy_dash = dashboard.create_policy_dashboard()

# Launch dashboard server
dashboard.launch_server(port=8050)
print("Dashboard available at: http://localhost:8050")
```

### Custom Visualizations

```python
from visualization.charts import EconomicCharts

# Initialize chart generator
charts = EconomicCharts(config)

# Business cycle analysis
business_cycle_chart = charts.business_cycle_analysis(
    data['combined'],
    variables=['gdp_growth', 'unemployment', 'inflation']
)

# Phillips curve
phillips_curve = charts.phillips_curve(
    data['combined']['unemployment'],
    data['combined']['inflation']
)

# Yield curve analysis
yield_curve = charts.yield_curve_analysis(
    data['combined'],
    maturities=['3m', '6m', '1y', '2y', '5y', '10y']
)

# Sectoral analysis
sectoral_chart = charts.sectoral_analysis(
    data['combined'],
    sectors=['agriculture', 'manufacturing', 'services']
)

# Export charts
charts.export_all_charts('results/charts/')
```

## Advanced Usage

### Custom Model Development

```python
from models.base_model import BaseModel

class CustomBangladeshModel(BaseModel):
    """
    Custom model for Bangladesh-specific analysis
    """
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.model_type = 'custom_bangladesh'
        
    def setup_equations(self):
        """
        Define model equations specific to Bangladesh
        """
        # Add Bangladesh-specific features
        self.equations = {
            'remittances': self.remittances_equation,
            'garments_exports': self.garments_equation,
            'rice_production': self.agriculture_equation,
            'flood_impact': self.climate_equation
        }
    
    def remittances_equation(self, variables):
        """
        Model remittances impact on economy
        """
        # Implementation here
        pass
    
    def garments_equation(self, variables):
        """
        Model ready-made garments sector
        """
        # Implementation here
        pass
    
    def solve_model(self):
        """
        Solve the custom model
        """
        # Implementation here
        pass

# Use custom model
custom_model = CustomBangladeshModel(config, data)
custom_results = custom_model.solve_model()
```

### Model Ensemble Methods

```python
from analysis.ensemble import ModelEnsemble

# Create model ensemble
ensemble = ModelEnsemble(config)

# Add models to ensemble
ensemble.add_model('dsge', dsge_model)
ensemble.add_model('svar', svar_model)
ensemble.add_model('cge', cge_model)
ensemble.add_model('olg', olg_model)

# Train ensemble weights
ensemble.train_weights(
    training_data=data['combined'][:-12],  # Use all but last 12 quarters
    validation_data=data['combined'][-12:],  # Last 12 quarters for validation
    objective='forecast_accuracy'
)

# Generate ensemble forecasts
ensemble_forecast = ensemble.forecast(horizon=8)

# Analyze ensemble performance
performance = ensemble.evaluate_performance()
print("Ensemble Performance:")
print(f"  RMSE: {performance['rmse']:.4f}")
print(f"  MAE: {performance['mae']:.4f}")
print(f"  Model weights: {performance['weights']}")
```

### Real-time Analysis

```python
from utils.real_time import RealTimeAnalyzer

# Initialize real-time analyzer
rt_analyzer = RealTimeAnalyzer(config)

# Set up data feeds
rt_analyzer.setup_data_feeds([
    'bangladesh_bank_api',
    'bbs_releases',
    'trading_economics',
    'news_sentiment'
])

# Start real-time monitoring
rt_analyzer.start_monitoring()

# Define alert conditions
alerts = {
    'high_inflation': {'condition': 'inflation > 0.08', 'action': 'send_email'},
    'currency_pressure': {'condition': 'exchange_rate_change > 0.05', 'action': 'run_analysis'},
    'growth_slowdown': {'condition': 'gdp_nowcast < 0.05', 'action': 'update_forecasts'}
}

rt_analyzer.setup_alerts(alerts)

# Generate nowcasts
nowcast = rt_analyzer.generate_nowcast()
print(f"Current quarter GDP growth nowcast: {nowcast['gdp_growth']:.2%}")
print(f"Inflation nowcast: {nowcast['inflation']:.2%}")
```

## Troubleshooting

### Common Issues

1. **Data Loading Problems**
   ```python
   # Check API keys
   from utils.data_processing.data_loader import check_api_access
   api_status = check_api_access(config)
   print("API Status:", api_status)
   
   # Use sample data if APIs unavailable
   if not api_status['all_available']:
       data_loader.use_sample_data = True
   ```

2. **Model Convergence Issues**
   ```python
   # Check model stability
   if not dsge.solution['stable']:
       print("Model unstable. Adjusting parameters...")
       dsge.config['beta'] = 0.99  # Adjust discount factor
       dsge.config['sigma'] = 1.0  # Adjust risk aversion
       dsge.solve_model()
   ```

3. **Memory Issues with Large Models**
   ```python
   # Use chunked processing
   from utils.optimization import ChunkedProcessor
   
   processor = ChunkedProcessor(chunk_size=1000)
   results = processor.process_large_dataset(data, model.simulate)
   ```

4. **Slow Estimation**
   ```python
   # Use parallel processing
   from utils.optimization import ParallelEstimator
   
   estimator = ParallelEstimator(n_cores=4)
   results = estimator.estimate_parallel(model, data)
   ```

### Performance Optimization

```python
# Enable caching
from utils.caching import enable_caching
enable_caching(cache_dir='cache/')

# Use GPU acceleration (if available)
from utils.gpu import enable_gpu
if enable_gpu():
    print("GPU acceleration enabled")

# Optimize numerical settings
from utils.optimization import optimize_numerical_settings
optimize_numerical_settings()
```

### Getting Help

- Check the [API documentation](api_reference.md)
- Review [example notebooks](../examples/)
- Submit issues on GitHub
- Contact the development team

---

*This user guide provides comprehensive coverage of the Bangladesh Macroeconomic Models framework. For specific technical details, refer to the API documentation and example notebooks.*