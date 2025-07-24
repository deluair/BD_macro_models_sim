# Bangladesh Macroeconomic Models Simulation Project

🎉 **Status: All 15 Models Successfully Implemented and Operational** 🎉

This project implements a comprehensive suite of 15 major macroeconomic models for Bangladesh using real economic data from various sources including Bangladesh Bank, World Bank, IMF, and other official statistics. All models have been successfully tested and are generating reliable simulation results.

## Overview

Bangladesh has experienced robust economic growth averaging 6% annually over the past two decades, transitioning from one of the poorest nations to lower-middle income status in 2015. This project provides comprehensive macroeconomic modeling tools to analyze Bangladesh's economy using various modeling approaches.

## Implemented Models

### 1. Dynamic Stochastic General Equilibrium (DSGE) Model
- **Purpose**: Business cycle analysis, monetary policy evaluation
- **Features**: Microfounded, rational expectations, stochastic shocks
- **Applications**: Monetary policy transmission, inflation dynamics, output gaps

### 2. Computable General Equilibrium (CGE) Model
- **Purpose**: Long-term policy analysis, trade policy evaluation
- **Features**: Multi-sectoral, price equilibrium, welfare analysis
- **Applications**: Trade liberalization, tax policy, structural reforms

### 3. Structural Vector Autoregression (SVAR) Model
- **Purpose**: Empirical analysis of macroeconomic relationships
- **Features**: Data-driven, impulse response analysis, variance decomposition
- **Applications**: Monetary transmission channels, shock identification

### 4. Quarterly Macroeconomic Model (QMM)
- **Purpose**: Short-term forecasting and policy simulation
- **Features**: Reduced-form equations, quarterly frequency
- **Applications**: GDP forecasting, inflation projection, policy scenarios

### 5. Financial Sector Model
- **Purpose**: Banking sector analysis, financial stability
- **Features**: Credit channels, NPL dynamics, bank behavior
- **Applications**: Financial sector reforms, credit policy analysis

### 6. Overlapping Generations (OLG) Life Cycle Model
- **Purpose**: Demographic transitions, pension systems, intergenerational equity
- **Features**: Age-structured population, life cycle consumption, savings behavior
- **Applications**: Population aging effects, social security reform, fiscal sustainability

### 7. New Keynesian Model with Heterogeneous Agents (HANK)
- **Purpose**: Inequality and aggregate demand, distributional effects of policy
- **Features**: Heterogeneous households, incomplete markets, idiosyncratic risk
- **Applications**: Fiscal multipliers, monetary policy distribution effects, inequality dynamics

### 8. Game Theoretic Models
- **Purpose**: Strategic interactions, policy coordination, institutional analysis
- **Features**: Nash equilibrium, cooperative/non-cooperative games, mechanism design
- **Applications**: Central bank independence, fiscal federalism, international coordination

### 9. Real Business Cycle (RBC) Model
- **Purpose**: Technology-driven fluctuations, productivity analysis
- **Features**: Perfect competition, flexible prices, technology shocks
- **Applications**: Growth accounting, productivity trends, structural transformation

### 10. New Economic Geography (NEG) Model
- **Purpose**: Spatial economics, regional development, agglomeration effects
- **Features**: Transport costs, increasing returns, core-periphery patterns
- **Applications**: Regional inequality, industrial clusters, infrastructure investment

### 11. Search and Matching Model
- **Purpose**: Labor market dynamics, unemployment analysis
- **Features**: Job search frictions, matching functions, wage bargaining
- **Applications**: Employment policies, labor market reforms, skills mismatch

### 12. Behavioral Macroeconomic Model
- **Purpose**: Psychological factors in economic decisions
- **Features**: Bounded rationality, behavioral biases, adaptive learning
- **Applications**: Financial bubbles, policy communication, expectation management

### 13. Agent-Based Model (ABM)
- **Purpose**: Complex adaptive systems, emergent phenomena
- **Features**: Heterogeneous agents, local interactions, non-linear dynamics
- **Applications**: Financial contagion, network effects, systemic risk

### 14. Integrated Assessment Model (IAM)
- **Purpose**: Climate-economy interactions, environmental policy
- **Features**: Climate damages, carbon pricing, green transition
- **Applications**: Carbon tax policy, climate adaptation, sustainable development

### 15. Small Open Economy Model
- **Purpose**: External sector analysis, exchange rate dynamics
- **Features**: Foreign sector interactions, capital flows, terms of trade
- **Applications**: Exchange rate policy, capital account liberalization, external shocks

## Key Economic Features of Bangladesh

- **GDP**: $467.2 billion (nominal, 2025), $1.78 trillion (PPP, 2025)
- **Growth Rate**: 5.82% (2024), historically averaging 6%+
- **Key Sectors**: 
  - Services: 51.4% of GDP
  - Industry: 34.1% of GDP (RMG sector dominates exports)
  - Agriculture: 14.5% of GDP
- **Inflation**: 8.48% (June 2025)
- **Unemployment**: 4.7% (Dec 2024)
- **Current Account**: Deficit due to import dependency
- **Exchange Rate**: Managed float regime with multiple rates

## Data Sources

### Primary Sources
1. **Bangladesh Bank** - Central bank data, monetary policy, financial statistics
2. **Bangladesh Bureau of Statistics (BBS)** - National accounts, employment, prices
3. **Ministry of Finance** - Fiscal data, budget, public debt
4. **Export Promotion Bureau** - Trade statistics

### International Sources
1. **World Bank** - Development indicators, poverty statistics
2. **IMF** - Balance of payments, fiscal data, projections
3. **Asian Development Bank** - Regional economic data
4. **OECD** - Trade and investment data

## Project Structure

```
BD_macro_models_sim/
├── data/                    # Real economic data
├── models/                  # Model implementations
├── analysis/               # Analysis scripts and results
├── visualization/          # Charts and dashboards
├── docs/                   # Documentation and papers
├── tests/                  # Model validation tests
└── utils/                  # Utility functions
```

## Installation and Setup

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run individual models: `python scripts/run_individual_model.py <model_name>`
4. Run all models: `python scripts/run_all_models.py`
5. View results in the `results/` directory

### Quick Start
```bash
# Run all models at once
python scripts/run_all_models.py

# Run specific model (e.g., HANK model)
python scripts/run_individual_model.py hank

# Check results
ls results/
```

## Key Research Applications

1. **Monetary Policy Analysis**: DSGE and SVAR models for transmission mechanisms
2. **Trade Policy Evaluation**: CGE models for WTO commitments, FTA impacts
3. **Fiscal Policy Assessment**: QMM for budget impact analysis
4. **Financial Stability**: Banking sector models for NPL analysis
5. **Development Planning**: Long-term growth projections and scenarios

## Model Validation and Performance

✅ **All 15 models successfully validated and operational**

Validation completed against historical data (2000-2024) with:
- ✅ In-sample fit statistics
- ✅ Out-of-sample forecasting accuracy  
- ✅ Policy simulation consistency
- ✅ Cross-model comparison
- ✅ Performance optimization (especially HANK model)
- ✅ Comprehensive result generation

### Recent Achievements
- **HANK Model Optimization**: Resolved performance issues, reduced computation time
- **Complete Model Suite**: All 15 models running without errors
- **Automated Testing**: Full simulation suite executes successfully
- **Result Generation**: Comprehensive CSV outputs and visualization charts

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for:
- New model implementations
- Data source integrations
- Analysis improvements
- Documentation updates

## License

MIT License - see LICENSE file for details.

## Contact

For questions and collaboration opportunities, please contact the research team.