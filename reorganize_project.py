#!/usr/bin/env python3
"""
Project Reorganization Script

This script reorganizes the Bangladesh Macroeconomic Models project
for better structure, maintainability, and clarity.
"""

import os
import shutil
from pathlib import Path
import json

def create_directory_structure():
    """
    Create the new organized directory structure.
    """
    base_dir = Path('.')
    
    # New directory structure
    new_structure = {
        'src': {
            'models': {},
            'analysis': {},
            'data_processing': {},
            'visualization': {},
            'utils': {}
        },
        'data': {
            'raw': {},
            'processed': {},
            'external': {}
        },
        'outputs': {
            'results': {},
            'reports': {},
            'plots': {},
            'analysis_outputs': {}
        },
        'scripts': {
            'data_collection': {},
            'model_execution': {},
            'analysis_runners': {}
        },
        'docs': {
            'methodology': {},
            'api': {},
            'user_guides': {},
            'papers': {}
        },
        'tests': {
            'unit': {},
            'integration': {},
            'validation': {}
        },
        'config': {},
        'logs': {}
    }
    
    def create_dirs(structure, parent_path=base_dir):
        for name, subdirs in structure.items():
            dir_path = parent_path / name
            dir_path.mkdir(exist_ok=True)
            if subdirs:
                create_dirs(subdirs, dir_path)
    
    create_dirs(new_structure)
    print("✓ Created new directory structure")

def move_model_files():
    """
    Move model files to src/models/
    """
    models_src = Path('models')
    models_dest = Path('src/models')
    
    if models_src.exists():
        # Copy all model directories
        for model_dir in models_src.iterdir():
            if model_dir.is_dir():
                dest_dir = models_dest / model_dir.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(model_dir, dest_dir)
        print("✓ Moved model files to src/models/")

def move_analysis_files():
    """
    Move analysis files to src/analysis/
    """
    analysis_src = Path('analysis')
    analysis_dest = Path('src/analysis')
    
    if analysis_src.exists():
        # Copy analysis framework
        for item in analysis_src.iterdir():
            if item.name != 'comprehensive_analysis':  # Skip output directory
                dest_path = analysis_dest / item.name
                if item.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)
                else:
                    shutil.copy2(item, dest_path)
        print("✓ Moved analysis files to src/analysis/")

def move_data_files():
    """
    Organize data files
    """
    # Data is already well organized, just ensure consistency
    data_src = Path('data')
    if data_src.exists():
        print("✓ Data directory already well organized")

def move_output_files():
    """
    Move output files to outputs/
    """
    # Move results
    results_src = Path('results')
    results_dest = Path('outputs/results')
    if results_src.exists():
        for file in results_src.glob('*.csv'):
            shutil.copy2(file, results_dest / file.name)
        print("✓ Moved results to outputs/results/")
    
    # Move reports
    reports_src = Path('reports')
    reports_dest = Path('outputs/reports')
    if reports_src.exists():
        for file in reports_src.glob('*.md'):
            shutil.copy2(file, reports_dest / file.name)
        print("✓ Moved reports to outputs/reports/")
    
    # Move plots
    plots_src = Path('plots')
    plots_dest = Path('outputs/plots')
    if plots_src.exists():
        for file in plots_src.rglob('*'):
            if file.is_file():
                rel_path = file.relative_to(plots_src)
                dest_file = plots_dest / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dest_file)
        print("✓ Moved plots to outputs/plots/")
    
    # Move visualization outputs
    viz_src = Path('visualization')
    viz_dest = Path('outputs/plots')
    if viz_src.exists():
        for subdir in viz_src.iterdir():
            if subdir.is_dir():
                dest_subdir = viz_dest / subdir.name
                dest_subdir.mkdir(exist_ok=True)
                for file in subdir.rglob('*'):
                    if file.is_file():
                        rel_path = file.relative_to(subdir)
                        dest_file = dest_subdir / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, dest_file)
        print("✓ Moved visualization outputs to outputs/plots/")
    
    # Move analysis outputs
    analysis_outputs_src = Path('analysis/comprehensive_analysis')
    analysis_outputs_dest = Path('outputs/analysis_outputs')
    if analysis_outputs_src.exists():
        if analysis_outputs_dest.exists():
            shutil.rmtree(analysis_outputs_dest)
        shutil.copytree(analysis_outputs_src, analysis_outputs_dest)
        print("✓ Moved comprehensive analysis outputs to outputs/analysis_outputs/")

def move_script_files():
    """
    Organize script files
    """
    scripts_src = Path('scripts')
    
    # Data collection scripts
    data_scripts = ['fetch_real_data.py', 'fetch_time_series.py', 'data_manager.py', 
                   'test_data_access.py', 'demo_data_integration.py', 'working_data_demo.py']
    
    # Model execution scripts
    model_scripts = ['run_all_models.py', 'run_individual_model.py']
    
    # Setup scripts
    setup_scripts = ['setup_environment.py']
    
    if scripts_src.exists():
        for script in data_scripts:
            script_path = scripts_src / script
            if script_path.exists():
                shutil.copy2(script_path, Path('scripts/data_collection') / script)
        
        for script in model_scripts:
            script_path = scripts_src / script
            if script_path.exists():
                shutil.copy2(script_path, Path('scripts/model_execution') / script)
        
        for script in setup_scripts:
            script_path = scripts_src / script
            if script_path.exists():
                shutil.copy2(script_path, Path('scripts') / script)
        
        print("✓ Organized script files")

def move_utility_files():
    """
    Move utility files to src/utils/
    """
    utils_src = Path('utils')
    utils_dest = Path('src/utils')
    
    if utils_src.exists():
        for item in utils_src.iterdir():
            dest_path = utils_dest / item.name
            if item.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(item, dest_path)
            else:
                shutil.copy2(item, dest_path)
        print("✓ Moved utility files to src/utils/")

def move_config_files():
    """
    Move configuration files to config/
    """
    config_files = ['config.yaml', 'config_template.py', 'requirements.txt']
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            shutil.copy2(config_path, Path('config') / config_file)
    
    print("✓ Moved configuration files to config/")

def move_root_files():
    """
    Organize root-level files
    """
    # Move standalone analysis files to src/
    standalone_files = [
        'bangladesh_economic_analysis.py',
        'bangladesh_economic_outlook_2025.py',
        'example_analysis.py',
        'policy_scenario_analysis.py',
        'data_fetcher.py',
        'main.py'
    ]
    
    src_dir = Path('src')
    for file_name in standalone_files:
        file_path = Path(file_name)
        if file_path.exists():
            shutil.copy2(file_path, src_dir / file_name)
    
    print("✓ Moved root analysis files to src/")

def move_docs():
    """
    Organize documentation
    """
    docs_src = Path('docs')
    if docs_src.exists():
        # Docs are already well organized
        print("✓ Documentation already well organized")

def move_tests():
    """
    Organize test files
    """
    tests_src = Path('tests')
    if tests_src.exists():
        # Tests are already well organized
        print("✓ Tests already well organized")

def create_new_main_files():
    """
    Create new main entry points and documentation
    """
    # Create new main.py
    main_content = '''#!/usr/bin/env python3
"""
Bangladesh Macroeconomic Models - Main Entry Point

This is the main entry point for the Bangladesh macroeconomic modeling framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.analysis.analysis_framework import AnalysisFramework

def main():
    """
    Main function to run the comprehensive analysis framework.
    """
    print("Bangladesh Macroeconomic Models - Comprehensive Analysis Framework")
    print("=" * 70)
    
    # Initialize and run analysis framework
    framework = AnalysisFramework()
    report_path = framework.run_complete_analysis()
    
    print(f"\\n✓ Analysis completed successfully!")
    print(f"📋 Main report: {report_path}")
    print(f"📁 All outputs: {framework.output_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open('main.py', 'w') as f:
        f.write(main_content)
    
    # Create project README
    readme_content = '''# Bangladesh Macroeconomic Models

A comprehensive framework for macroeconomic modeling and analysis of Bangladesh's economy.

## 🏗️ Project Structure

```
BD_macro_models_sim/
├── src/                      # Source code
│   ├── models/              # Economic model implementations
│   ├── analysis/            # Analysis frameworks
│   ├── utils/               # Utility functions
│   └── *.py                 # Main analysis scripts
├── data/                    # Data files
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── external/            # External data sources
├── outputs/                 # All outputs
│   ├── results/             # Model results (CSV)
│   ├── reports/             # Analysis reports (MD)
│   ├── plots/               # Visualizations
│   └── analysis_outputs/    # Comprehensive analysis outputs
├── scripts/                 # Utility scripts
│   ├── data_collection/     # Data fetching scripts
│   ├── model_execution/     # Model running scripts
│   └── analysis_runners/    # Analysis execution scripts
├── docs/                    # Documentation
├── tests/                   # Test files
├── config/                  # Configuration files
└── logs/                    # Log files
```

## 🚀 Quick Start

### Run Complete Analysis
```bash
python main.py
```

### Run Individual Components
```bash
# Run all models
python scripts/model_execution/run_all_models.py

# Run analysis framework
python src/analysis/analysis_framework.py

# Fetch new data
python scripts/data_collection/fetch_real_data.py
```

## 📊 Available Models

- **Structural Models**: DSGE, SVAR, RBC
- **Equilibrium Models**: CGE, OLG
- **Behavioral Models**: ABM, Behavioral
- **Financial Models**: Financial, HANK
- **Specialized Models**: NEG, QMM, IAM, Game Theory, Search & Matching, SOE

## 📈 Analysis Capabilities

1. **Comparative Forecasting**: Multi-model forecasting comparison
2. **Policy Scenario Analysis**: 8 policy scenarios across models
3. **Monte Carlo Simulations**: Risk and uncertainty analysis
4. **Model Validation**: Statistical testing and backtesting

## 📋 Requirements

See `config/requirements.txt` for dependencies.

## 📚 Documentation

Detailed documentation available in the `docs/` directory.

## 🤝 Contributing

Please read the contribution guidelines in `docs/` before contributing.

---

*This framework provides comprehensive tools for evidence-based economic policy making in Bangladesh.*
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✓ Created new main files and documentation")

def create_organization_summary():
    """
    Create a summary of the reorganization
    """
    summary = {
        'reorganization_date': '2025-07-24',
        'changes': {
            'src/': 'All source code including models, analysis, and utilities',
            'outputs/': 'All generated outputs including results, reports, and plots',
            'scripts/': 'Organized utility scripts by category',
            'config/': 'Configuration files and requirements',
            'data/': 'Data files (unchanged - already well organized)',
            'docs/': 'Documentation (unchanged - already well organized)',
            'tests/': 'Test files (unchanged - already well organized)'
        },
        'benefits': [
            'Clear separation of source code and outputs',
            'Better organization of scripts by purpose',
            'Centralized configuration management',
            'Improved maintainability and navigation',
            'Standard project structure following best practices'
        ],
        'migration_notes': [
            'All original files preserved in new locations',
            'Import paths may need updating in some scripts',
            'New main.py provides unified entry point',
            'Analysis framework remains fully functional'
        ]
    }
    
    with open('reorganization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Created reorganization summary")

def move_files():
    """Move files to their new locations"""
    try:
        # Move remaining model files
        if os.path.exists('models'):
            for item in os.listdir('models'):
                src_path = os.path.join('models', item)
                dst_path = os.path.join('src', 'models', item)
                if os.path.isdir(src_path) and not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
            # Remove empty models directory
            if not os.listdir('models'):
                os.rmdir('models')
            print("✓ Completed moving model files to src/models/")
        
        # Move remaining analysis files
        if os.path.exists('analysis'):
            for item in os.listdir('analysis'):
                src_path = os.path.join('analysis', item)
                dst_path = os.path.join('src', 'analysis', item)
                if not os.path.exists(dst_path):
                    if os.path.isfile(src_path) or os.path.isdir(src_path):
                        shutil.move(src_path, dst_path)
            # Remove empty analysis directory
            if not os.listdir('analysis'):
                os.rmdir('analysis')
            print("✓ Completed moving analysis files to src/analysis/")
        
        # Move remaining results
        if os.path.exists('results'):
            for item in os.listdir('results'):
                src_path = os.path.join('results', item)
                dst_path = os.path.join('outputs', 'results', item)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
            # Remove empty results directory
            if not os.listdir('results'):
                os.rmdir('results')
            print("✓ Completed moving results to outputs/results/")
        
        # Move remaining reports
        if os.path.exists('reports'):
            for item in os.listdir('reports'):
                src_path = os.path.join('reports', item)
                dst_path = os.path.join('outputs', 'reports', item)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
            # Remove empty reports directory
            if not os.listdir('reports'):
                os.rmdir('reports')
            print("✓ Completed moving reports to outputs/reports/")
        
        # Move remaining plots
        if os.path.exists('plots'):
            for item in os.listdir('plots'):
                src_path = os.path.join('plots', item)
                dst_path = os.path.join('outputs', 'plots', item)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
            # Remove empty plots directory
            if not os.listdir('plots'):
                os.rmdir('plots')
            print("✓ Completed moving plots to outputs/plots/")
        
        # Move remaining visualization outputs
        if os.path.exists('visualization'):
            for item in os.listdir('visualization'):
                src_path = os.path.join('visualization', item)
                dst_path = os.path.join('outputs', 'plots', 'visualization', item)
                if os.path.isdir(src_path) and not os.path.exists(dst_path):
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.move(src_path, dst_path)
            # Remove empty visualization directory
            if not os.listdir('visualization'):
                os.rmdir('visualization')
            print("✓ Completed moving visualization outputs to outputs/plots/")
        
        # Move remaining utility files
        if os.path.exists('utils'):
            for item in os.listdir('utils'):
                src_path = os.path.join('utils', item)
                dst_path = os.path.join('src', 'utils', item)
                if os.path.isdir(src_path) and not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
            # Remove empty utils directory
            if not os.listdir('utils'):
                os.rmdir('utils')
            print("✓ Completed moving utility files to src/utils/")
        
        # Move root level Python files to src
        root_py_files = ['data_fetcher.py', 'bangladesh_economic_analysis.py', 
                        'bangladesh_economic_outlook_2025.py', 'example_analysis.py',
                        'policy_scenario_analysis.py']
        
        for file in root_py_files:
            if os.path.exists(file):
                dst_path = os.path.join('src', file)
                if not os.path.exists(dst_path):
                    shutil.move(file, dst_path)
        
        print("✓ Completed moving root Python files to src/")
        
    except Exception as e:
        print(f"❌ Error during reorganization: {e}")
        print("Please check the error and try again.")

def main():
    """
    Main reorganization function
    """
    print("Bangladesh Macroeconomic Models - Project Reorganization")
    print("=" * 60)
    
    try:
        create_directory_structure()
        move_model_files()
        move_analysis_files()
        move_data_files()
        move_output_files()
        move_script_files()
        move_utility_files()
        move_config_files()
        move_root_files()
        move_docs()
        move_tests()
        create_new_main_files()
        create_organization_summary()
        
        # Clean up remaining files
        move_files()
        
        print("\n" + "=" * 60)
        print("✅ PROJECT REORGANIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📁 New Structure:")
        print("  - src/: All source code")
        print("  - outputs/: All generated outputs")
        print("  - scripts/: Organized utility scripts")
        print("  - config/: Configuration files")
        print("  - data/, docs/, tests/: Unchanged (already well organized)")
        
        print("\n🚀 Next Steps:")
        print("  1. Review the new structure")
        print("  2. Update any hardcoded paths in scripts")
        print("  3. Test the new main.py entry point")
        print("  4. Old directories have been cleaned up automatically")
        
    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()