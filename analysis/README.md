# Bangladesh Macroeconomic Models - Comprehensive Analysis Framework

This directory contains a comprehensive analysis framework for Bangladesh macroeconomic models, providing integrated forecasting, policy analysis, Monte Carlo simulations, and model validation capabilities.

## 🎯 Overview

The analysis framework integrates **17 economic models** covering various aspects of the Bangladesh economy:

### Model Categories
- **Structural Models**: DSGE, SVAR, RBC
- **Equilibrium Models**: CGE, OLG
- **Behavioral Models**: ABM, Behavioral
- **Financial Models**: Financial, HANK
- **Specialized Models**: NEG, QMM, IAM, Game Theory, Search & Matching, SOE

## 📁 Directory Structure

```
analysis/
├── forecasting/              # Comparative forecasting analysis
│   └── comparative_forecasting.py
├── policy_analysis/          # Policy scenario analysis
│   └── policy_scenario_analyzer.py
├── simulations/              # Monte Carlo simulations
│   └── monte_carlo_simulator.py
├── validation/               # Model validation and backtesting
│   └── model_validator.py
├── analysis_framework.py     # Main comprehensive framework
├── comprehensive_analysis/   # Generated analysis outputs
│   ├── forecasting/
│   ├── policy_analysis/
│   ├── simulations/
│   ├── validation/
│   └── reports/
└── README.md                # This file
```

## 🚀 Quick Start

### Running the Complete Analysis

```bash
cd analysis
python analysis_framework.py
```

This will execute all analysis modules and generate comprehensive reports.

### Running Individual Modules

```bash
# Forecasting analysis
cd forecasting
python comparative_forecasting.py

# Policy analysis
cd policy_analysis
python policy_scenario_analyzer.py

# Monte Carlo simulations
cd simulations
python monte_carlo_simulator.py

# Model validation
cd validation
python model_validator.py
```

## 📊 Analysis Modules

### 1. Comparative Forecasting (`forecasting/`)
- **Purpose**: Compare forecasting performance across models
- **Features**:
  - Synthetic forecast generation
  - Accuracy metrics (RMSE, MAE, MAPE)
  - Consensus forecasting
  - Visualization and reporting

### 2. Policy Scenario Analysis (`policy_analysis/`)
- **Purpose**: Evaluate policy interventions across models
- **Scenarios**:
  - Monetary Policy (Expansionary/Contractionary)
  - Fiscal Policy (Infrastructure Investment, Tax Reform)
  - Structural Reforms (Labor Market, Financial Sector)
  - Trade Policy (Export Promotion, Import Substitution)
- **Outputs**: Impact assessments, effectiveness rankings, visualizations

### 3. Monte Carlo Simulations (`simulations/`)
- **Purpose**: Uncertainty and risk analysis
- **Scenarios**:
  - Baseline Economic Uncertainty
  - External Shock Analysis
  - Climate Risk Assessment
  - Financial Stress Testing
  - Policy Uncertainty Analysis
- **Metrics**: VaR, Expected Shortfall, Risk distributions

### 4. Model Validation (`validation/`)
- **Purpose**: Statistical testing and backtesting
- **Tests**:
  - Normality (Shapiro-Wilk, Jarque-Bera)
  - Stationarity (ADF, KPSS)
  - Autocorrelation (Ljung-Box)
  - Heteroskedasticity (Breusch-Pagan)
  - Structural Breaks (Chow test)
- **Metrics**: RMSE, MAE, R², Directional Accuracy

## 📈 Key Economic Indicators

The framework analyzes the following key indicators for Bangladesh:

- GDP Growth
- Inflation
- Unemployment
- Current Account
- Government Debt
- Trade Balance
- Real Exchange Rate
- Investment Rate
- Consumption Growth
- Exports Growth

## 📋 Generated Outputs

### Main Reports
- **Comprehensive Analysis Report**: `comprehensive_analysis/reports/comprehensive_analysis_report.md`
- **Policy Analysis Report**: `comprehensive_analysis/policy_analysis/policy_analysis_report.md`
- **Simulation Analysis Report**: `comprehensive_analysis/simulations/simulation_analysis_report.md`

### Visualizations
- Model overview and comparison charts
- Policy impact time series
- Risk distribution plots
- Performance comparison heatmaps
- Scenario ranking visualizations

### Data Exports
- CSV files with detailed results
- JSON configuration files
- Summary statistics tables

## 🔧 Technical Requirements

### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0
tqdm>=4.62.0
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels tqdm
```

## 📊 Analysis Results Summary

### Latest Run Results
- **Models Analyzed**: 17
- **Analysis Modules Completed**: 2/4
- **Policy Scenarios**: 8
- **Simulation Scenarios**: 5
- **Generated Files**: 34

### Successful Modules
✅ **Policy Analysis**: 8 scenarios analyzed across multiple models
✅ **Monte Carlo Simulations**: 5 risk scenarios with comprehensive metrics
✅ **Summary Dashboard**: Comprehensive visualizations and reports

### Module Status
❌ **Forecasting Analysis**: Module dependencies need verification
❌ **Model Validation**: Scale parameter issue in synthetic data generation

## 🎯 Policy Recommendations

### Short-term (1-4 quarters)
- **Primary**: SVAR models for dynamic relationships
- **Secondary**: Financial models for market conditions
- **Validation**: Behavioral models for cross-verification

### Medium-term (1-3 years)
- **Primary**: DSGE models for structural relationships
- **Secondary**: RBC models for technology effects
- **Policy**: CGE models for sectoral impacts

### Long-term (3+ years)
- **Primary**: CGE models for structural transformation
- **Secondary**: OLG models for demographic effects
- **Climate**: IAM models for environmental impacts

## 🔄 Risk Management Framework

1. **Model Risk**: Ensemble approaches combining multiple models
2. **Parameter Uncertainty**: Regular Monte Carlo simulations
3. **Structural Breaks**: Continuous model validation and updating
4. **External Shocks**: Stress testing with extreme scenarios

## 📚 Usage Guidelines

### For Policymakers
- Use policy analysis module for intervention assessment
- Review simulation results for risk evaluation
- Consult model recommendations for appropriate time horizons

### For Researchers
- Extend individual modules for specific research questions
- Use validation framework for model development
- Leverage ensemble approaches for robust analysis

### For Analysts
- Run regular updates with new data
- Monitor model performance metrics
- Generate periodic comprehensive reports

## 🔧 Customization

### Adding New Models
1. Place model results CSV in `../results/` directory
2. Ensure consistent variable naming
3. Re-run analysis framework

### Modifying Scenarios
1. Edit scenario definitions in respective modules
2. Adjust parameters and time horizons
3. Update configuration files

### Extending Analysis
1. Add new analysis modules following existing patterns
2. Integrate with main framework
3. Update documentation

## 📞 Support

For technical issues or questions:
1. Check module-specific documentation
2. Review error logs in output directories
3. Verify data requirements and dependencies

---

*This comprehensive analysis framework provides a robust foundation for evidence-based economic policy making in Bangladesh. Regular updates and continuous improvement ensure the framework remains relevant and accurate.*

**Last Updated**: 2025-07-24
**Framework Version**: 1.0.0
**Models Supported**: 17
**Analysis Modules**: 4