
# Configuration file for Bangladesh Macroeconomic Models
# Copy this file to 'config.py' and customize as needed

# Data API Settings
DATA_APIS = {
    'world_bank': {
        'enabled': True,
        'api_key': None,  # Usually not required for World Bank
        'base_url': 'https://api.worldbank.org/v2/',
        'timeout': 30
    },
    'imf': {
        'enabled': True,
        'api_key': None,  # Usually not required for IMF
        'base_url': 'http://dataservices.imf.org/REST/SDMX_JSON.svc/',
        'timeout': 30
    },
    'bangladesh_bank': {
        'enabled': False,  # Implement when API becomes available
        'api_key': None,
        'base_url': 'https://www.bb.org.bd/api/',  # Placeholder
        'timeout': 30
    }
}

# Model Default Parameters
MODEL_DEFAULTS = {
    'country_code': 'BGD',
    'country_name': 'Bangladesh',
    'currency': 'BDT',
    'fiscal_year_start': 7,  # July (Bangladesh fiscal year)
    'data_start_year': 2000,
    'data_end_year': 2023
}

# Simulation Settings
SIMULATION_SETTINGS = {
    'default_periods': 20,
    'monte_carlo_runs': 1000,
    'confidence_intervals': [0.05, 0.95],
    'random_seed': 42
}

# Output Settings
OUTPUT_SETTINGS = {
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300,
    'results_directory': 'results/',
    'data_directory': 'data/'
}

# Bangladesh-Specific Parameters
BANGLADESH_PARAMS = {
    'informal_sector_share': 0.35,  # Approximate share of informal economy
    'remittance_gdp_ratio': 0.06,   # Remittances as % of GDP
    'agriculture_gdp_share': 0.13,  # Agriculture share in GDP
    'manufacturing_gdp_share': 0.20, # Manufacturing share in GDP
    'services_gdp_share': 0.52,     # Services share in GDP
    'rural_population_share': 0.62, # Rural population percentage
    'financial_inclusion_rate': 0.50, # Financial inclusion rate
    'mobile_banking_penetration': 0.35 # Mobile banking usage
}
