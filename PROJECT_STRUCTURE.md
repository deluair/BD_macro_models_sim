
# Bangladesh Macroeconomic Models - Project Structure

## New Organized Structure

```
BD_macro_models_sim/
├── src/                          # Source code
│   ├── models/                   # Economic model implementations
│   ├── analysis/                 # Analysis modules
│   ├── utils/                    # Utility functions
│   ├── data_processing/          # Data processing modules
│   └── visualization/            # Visualization modules
├── data/                         # Data storage
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data
│   └── external/                 # External data sources
├── outputs/                      # All generated outputs
│   ├── results/                  # Model results (CSV files)
│   ├── reports/                  # Analysis reports (MD files)
│   ├── plots/                    # Charts and visualizations
│   └── analysis_outputs/         # Comprehensive analysis outputs
├── scripts/                      # Utility and execution scripts
├── tests/                        # Test files
├── docs/                         # Documentation
├── config/                       # Configuration files
├── logs/                         # Log files
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## Key Improvements

1. **Clear Separation**: Source code is now in `src/` directory
2. **Organized Outputs**: All outputs are consolidated in `outputs/`
3. **Better Structure**: Logical grouping of related files
4. **Maintainability**: Easier to navigate and maintain
5. **Scalability**: Structure supports future growth

## Usage

- Run models: `python main.py`
- Access results: Check `outputs/` directory
- View reports: Check `outputs/reports/`
- See visualizations: Check `outputs/plots/`
