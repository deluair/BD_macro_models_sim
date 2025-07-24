#!/usr/bin/env python3
"""
Environment Setup Script for Bangladesh Macroeconomic Models

This script helps set up the environment for running the Bangladesh macroeconomic
modeling suite, including:
1. Installing required packages
2. Testing data API connections
3. Configuring environment variables
4. Running basic validation tests

Usage:
    python setup_environment.py

Author: Bangladesh Macroeconomic Modeling Team
Date: 2024
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_python_version():
    """
    Check if Python version is compatible
    """
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True

def install_requirements():
    """
    Install required packages from requirements.txt
    """
    print("\nInstalling required packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install packages
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ All packages installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_package_imports():
    """
    Test if critical packages can be imported
    """
    print("\nTesting package imports...")
    
    critical_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'statsmodels': 'Statsmodels',
        'requests': 'Requests'
    }
    
    optional_packages = {
        'wbgapi': 'World Bank API',
        'world_bank_data': 'World Bank Data',
        'arch': 'ARCH (econometrics)',
        'linearmodels': 'Linear Models',
        'cvxpy': 'CVXPY (optimization)',
        'networkx': 'NetworkX'
    }
    
    success_count = 0
    total_count = len(critical_packages) + len(optional_packages)
    
    # Test critical packages
    for package, name in critical_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name} - CRITICAL")
    
    # Test optional packages
    for package, name in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"⚠️  {name} - Optional (install if needed)")
    
    print(f"\nPackage import summary: {success_count}/{total_count} successful")
    return success_count >= len(critical_packages)

def test_data_apis():
    """
    Test connections to data APIs
    """
    print("\nTesting data API connections...")
    
    # Test World Bank API
    try:
        import wbgapi as wb
        # Try to fetch a simple indicator
        data = wb.data.fetch('NY.GDP.MKTP.KD.ZG', 'BGD', time=range(2020, 2022))
        if not data.empty:
            print("✅ World Bank API (wbgapi) - Connected")
        else:
            print("⚠️  World Bank API (wbgapi) - Connected but no data returned")
    except Exception as e:
        print(f"❌ World Bank API (wbgapi) - Error: {str(e)[:50]}...")
    
    # Test alternative World Bank package
    try:
        import world_bank_data as wb_alt
        countries = wb_alt.get_countries()
        if 'Bangladesh' in countries['name'].values:
            print("✅ World Bank Data (world_bank_data) - Connected")
        else:
            print("⚠️  World Bank Data (world_bank_data) - Connected but Bangladesh not found")
    except Exception as e:
        print(f"❌ World Bank Data (world_bank_data) - Error: {str(e)[:50]}...")
    
    # Test IMF API (basic connection)
    try:
        import requests
        response = requests.get(
            "http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow",
            timeout=10
        )
        if response.status_code == 200:
            print("✅ IMF API - Connected")
        else:
            print(f"⚠️  IMF API - Response code: {response.status_code}")
    except Exception as e:
        print(f"❌ IMF API - Error: {str(e)[:50]}...")
    
    # Test general internet connectivity
    try:
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connectivity - OK")
        else:
            print("⚠️  Internet connectivity - Limited")
    except Exception as e:
        print("❌ Internet connectivity - No connection")

def create_config_template():
    """
    Create a configuration template file
    """
    print("\nCreating configuration template...")
    
    config_content = """
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
"""
    
    try:
        with open('config_template.py', 'w') as f:
            f.write(config_content)
        print("✅ Configuration template created: config_template.py")
        print("   Copy to 'config.py' and customize as needed")
        return True
    except Exception as e:
        print(f"❌ Error creating config template: {e}")
        return False

def run_basic_tests():
    """
    Run basic functionality tests
    """
    print("\nRunning basic functionality tests...")
    
    # Test data fetcher
    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        print("✅ DataFetcher class imported successfully")
        
        # Test basic functionality (without actual API calls)
        indicators = fetcher.get_wb_indicators()
        if indicators:
            print(f"✅ World Bank indicators loaded: {len(indicators)} available")
        else:
            print("⚠️  World Bank indicators: Using default list")
    
    except Exception as e:
        print(f"❌ DataFetcher test failed: {str(e)[:50]}...")
    
    # Test model imports
    models_to_test = [
        ('cge_model', 'CGEModel'),
        ('svar_model', 'SVARModel'),
        ('abm_model', 'ABMModel'),
        ('hank_model', 'HANKModel'),
        ('financial_model', 'FinancialSectorModel'),
        ('soe_model', 'SmallOpenEconomyModel'),
        ('iam_model', 'IntegratedAssessmentModel')
    ]
    
    successful_models = 0
    for module_name, class_name in models_to_test:
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
            successful_models += 1
        except Exception as e:
            print(f"❌ {class_name} import failed: {str(e)[:30]}...")
    
    print(f"\nModel import summary: {successful_models}/{len(models_to_test)} successful")
    return successful_models == len(models_to_test)

def create_directories():
    """
    Create necessary directories
    """
    print("\nCreating project directories...")
    
    directories = ['results', 'data', 'plots', 'logs']
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Directory created/verified: {directory}/")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")

def main():
    """
    Main setup function
    """
    print("=" * 60)
    print("BANGLADESH MACROECONOMIC MODELS - ENVIRONMENT SETUP")
    print("=" * 60)
    
    success_steps = 0
    total_steps = 7
    
    # Step 1: Check Python version
    if check_python_version():
        success_steps += 1
    
    # Step 2: Install requirements
    if install_requirements():
        success_steps += 1
    
    # Step 3: Test package imports
    if test_package_imports():
        success_steps += 1
    
    # Step 4: Test data APIs
    test_data_apis()  # Always run, don't count as critical
    success_steps += 1
    
    # Step 5: Create config template
    if create_config_template():
        success_steps += 1
    
    # Step 6: Create directories
    create_directories()
    success_steps += 1
    
    # Step 7: Run basic tests
    if run_basic_tests():
        success_steps += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SETUP COMPLETED: {success_steps}/{total_steps} steps successful")
    print("=" * 60)
    
    if success_steps >= 5:
        print("\n🎉 Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and customize config_template.py")
        print("2. Run demo_data_integration.py to test the system")
        print("3. Explore individual model files for specific analyses")
        print("4. Check the results/ directory for outputs")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("Please review the error messages above and:")
        print("1. Install missing packages manually")
        print("2. Check internet connectivity for data APIs")
        print("3. Ensure all model files are present")
    
    print("\nFor support, check the documentation or contact the development team.")

if __name__ == "__main__":
    main()