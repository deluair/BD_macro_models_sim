# Bangladesh Macroeconomic Models Simulation Framework - Project Structure

## Overview
This document outlines the professional, organized structure of the Bangladesh Macroeconomic Models Simulation Framework after comprehensive reorganization and duplicate removal.

## Directory Structure

```
BD_macro_models_sim/
├── .github/                    # GitHub workflows and CI/CD
│   └── workflows/
│       └── ci.yml
├── .gitignore                  # Git ignore patterns
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   └── default.yaml           # Default settings
├── data/                       # Data management
│   ├── external/              # External data sources
│   ├── processed/             # Processed datasets
│   └── raw/                   # Raw data files
├── docs/                       # Documentation
│   ├── api/                   # API documentation
│   ├── methodology/           # Methodology documentation
│   ├── papers/                # Research papers
│   ├── user_guide.md          # User guide
│   └── user_guides/           # Additional user guides
├── logs/                       # Application logs
├── outputs/                    # All output files
│   ├── analysis_outputs/      # Analysis results
│   │   ├── forecasting/
│   │   ├── policy_analysis/
│   │   ├── reports/
│   │   ├── simulations/
│   │   └── validation/
│   ├── benchmarks/            # Benchmark results
│   ├── plots/                 # Visualization outputs
│   │   ├── charts/
│   │   ├── dashboards/
│   │   ├── interactive/
│   │   └── visualization/
│   ├── reports/               # Generated reports
│   └── results/               # Model simulation results
├── scripts/                    # Utility scripts
│   ├── analysis_runners/      # Analysis execution scripts
│   ├── data_collection/       # Data collection utilities
│   │   ├── data_manager.py
│   │   ├── demo_data_integration.py
│   │   ├── fetch_real_data.py
│   │   ├── fetch_time_series.py
│   │   ├── test_data_access.py
│   │   └── working_data_demo.py
│   ├── model_execution/        # Model execution scripts
│   │   ├── run_all_models.py
│   │   └── run_individual_model.py
│   ├── benchmark_models.py    # Model benchmarking
│   ├── collect_bd_data.py     # Bangladesh data collection
│   ├── evaluate_models.py     # Model evaluation
│   ├── preprocess_data.py     # Data preprocessing
│   ├── setup_environment.py   # Environment setup
│   └── train_models.py        # Model training
├── src/                        # Source code
│   ├── analysis/              # Analysis framework
│   │   ├── comprehensive_analysis/
│   │   ├── forecasting/
│   │   ├── policy_analysis/
│   │   ├── simulations/
│   │   ├── validation/
│   │   └── analysis_framework.py
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   └── config_manager.py
│   ├── data_processing/       # Data processing utilities
│   ├── models/                # Economic models
│   │   ├── abm/              # Agent-Based Models
│   │   ├── behavioral/       # Behavioral Economics
│   │   ├── cge/              # Computable General Equilibrium
│   │   ├── dsge/             # Dynamic Stochastic General Equilibrium
│   │   ├── financial/        # Financial Sector Models
│   │   ├── game_theory/      # Game Theory Models
│   │   ├── iam/              # Integrated Assessment Models
│   │   ├── neg/              # New Economic Geography
│   │   ├── olg/              # Overlapping Generations
│   │   ├── qmm/              # Quantitative Monetary Models
│   │   ├── rbc/              # Real Business Cycle
│   │   ├── search_matching/  # Search and Matching
│   │   ├── small_open_economy/ # Small Open Economy
│   │   └── svar/             # Structural Vector Autoregression
│   ├── utils/                 # Utility functions
│   ├── visualization/         # Visualization tools
│   ├── bangladesh_economic_analysis.py
│   ├── bangladesh_economic_outlook_2025.py
│   ├── data_fetcher.py
│   ├── example_analysis.py
│   └── policy_scenario_analysis.py
├── tests/                      # Test suite
├── tools/                      # Development tools
├── main.py                     # Main application entry point
├── pyproject.toml             # Python project configuration
├── pytest.ini                # Pytest configuration
├── requirements.txt           # Python dependencies
├── DEVELOPMENT_GUIDE.md       # Development guidelines
├── ORGANIZATION_STATUS.md     # Organization status
├── PROJECT_IMPROVEMENT_PLAN.md # Improvement roadmap
├── PROJECT_STRUCTURE.md       # Project structure documentation
└── README.md                  # Project overview
```

## Key Improvements Made

### 1. Eliminated Duplicates
- Removed duplicate model directories (kept `src/models/` over root `models/`)
- Removed duplicate analysis directories (kept `src/analysis/` over root `analysis/`)
- Consolidated duplicate scripts (organized in `scripts/` subdirectories)
- Moved visualization files to `outputs/plots/charts/`

### 2. Professional Organization
- Clear separation of source code (`src/`) and outputs (`outputs/`)
- Organized scripts into logical subdirectories
- Centralized configuration in `config/` directory
- Proper documentation structure in `docs/`

### 3. Standardized Structure
- All model implementations in `src/models/` with consistent naming
- All outputs consolidated in `outputs/` with clear categorization
- Development tools and utilities properly organized
- Clean root directory with only essential files

## Usage

### Running the Framework
```bash
python main.py
```

### Running Individual Models
```bash
python scripts/model_execution/run_individual_model.py <model_name>
```

### Running All Models
```bash
python scripts/model_execution/run_all_models.py
```

### Benchmarking
```bash
python scripts/benchmark_models.py --all --detailed
```

## Development Guidelines

1. **Source Code**: All new code should go in the `src/` directory
2. **Models**: New economic models should be added to `src/models/`
3. **Scripts**: Utility scripts should be organized in appropriate `scripts/` subdirectories
4. **Outputs**: All generated files should go to appropriate `outputs/` subdirectories
5. **Documentation**: Update relevant documentation when adding new features

## Configuration

The framework uses YAML configuration files in the `config/` directory:
- `config.yaml`: Main configuration file
- `default.yaml`: Default settings and fallback values

Configuration is managed through the `ConfigManager` class in `src/config/config_manager.py`.

## Data Management

- **Raw Data**: Store in `data/raw/`
- **Processed Data**: Store in `data/processed/`
- **External Data**: Store in `data/external/`
- **Model Results**: Automatically saved to `outputs/results/`

## Testing

Run tests using pytest:
```bash
pytest tests/
```

Test configuration is managed through `pytest.ini`.

---

*This structure ensures maintainability, scalability, and professional organization of the Bangladesh Macroeconomic Models Simulation Framework.*