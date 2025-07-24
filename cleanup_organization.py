#!/usr/bin/env python3
"""
Cleanup script to finalize project organization
"""

import os
import shutil
import sys

def safe_move_directory_contents(src_dir, dst_dir):
    """Safely move contents from source to destination directory"""
    if not os.path.exists(src_dir):
        return
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        try:
            if not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)
                print(f"✓ Moved {item} to {dst_dir}")
            else:
                print(f"⚠ Skipped {item} (already exists in destination)")
        except Exception as e:
            print(f"❌ Error moving {item}: {e}")

def safe_remove_directory(dir_path):
    """Safely remove directory if it's empty or force remove"""
    if not os.path.exists(dir_path):
        return
    
    try:
        if not os.listdir(dir_path):  # Directory is empty
            os.rmdir(dir_path)
            print(f"✓ Removed empty directory: {dir_path}")
        else:
            print(f"⚠ Directory not empty, skipping: {dir_path}")
            # List remaining contents
            print(f"  Contents: {os.listdir(dir_path)}")
    except Exception as e:
        print(f"❌ Error removing {dir_path}: {e}")

def move_remaining_files():
    """Move any remaining files to proper locations"""
    print("\n=== Moving Remaining Files ===")
    
    # Move remaining model directories
    safe_move_directory_contents('models', 'src/models')
    
    # Move remaining analysis files
    safe_move_directory_contents('analysis', 'src/analysis')
    
    # Move remaining results
    safe_move_directory_contents('results', 'outputs/results')
    
    # Move remaining reports
    safe_move_directory_contents('reports', 'outputs/reports')
    
    # Move remaining plots
    safe_move_directory_contents('plots', 'outputs/plots')
    
    # Move remaining visualization
    safe_move_directory_contents('visualization', 'outputs/plots/visualization')
    
    # Move remaining utils
    safe_move_directory_contents('utils', 'src/utils')
    
    # Move root Python files
    root_files = [
        'bangladesh_economic_analysis.py',
        'bangladesh_economic_outlook_2025.py', 
        'data_fetcher.py',
        'example_analysis.py',
        'policy_scenario_analysis.py'
    ]
    
    for file in root_files:
        if os.path.exists(file):
            dst_path = os.path.join('src', file)
            try:
                if not os.path.exists(dst_path):
                    shutil.move(file, dst_path)
                    print(f"✓ Moved {file} to src/")
                else:
                    print(f"⚠ {file} already exists in src/")
            except Exception as e:
                print(f"❌ Error moving {file}: {e}")

def cleanup_empty_directories():
    """Remove empty old directories"""
    print("\n=== Cleaning Up Empty Directories ===")
    
    old_dirs = ['models', 'analysis', 'results', 'reports', 'plots', 'visualization', 'utils']
    
    for dir_name in old_dirs:
        safe_remove_directory(dir_name)

def create_project_structure_summary():
    """Create a summary of the new project structure"""
    structure_summary = """
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
"""
    
    with open('PROJECT_STRUCTURE.md', 'w', encoding='utf-8') as f:
        f.write(structure_summary)
    
    print("✓ Created PROJECT_STRUCTURE.md")

def main():
    """Main cleanup function"""
    print("Bangladesh Macroeconomic Models - Project Cleanup")
    print("=" * 55)
    
    try:
        # Move remaining files
        move_remaining_files()
        
        # Clean up empty directories
        cleanup_empty_directories()
        
        # Create structure summary
        create_project_structure_summary()
        
        print("\n" + "=" * 55)
        print("✅ PROJECT CLEANUP COMPLETED SUCCESSFULLY!")
        print("\n📁 New project structure:")
        print("  • src/ - All source code")
        print("  • outputs/ - All generated outputs")
        print("  • data/ - Data storage")
        print("  • scripts/ - Utility scripts")
        print("  • docs/ - Documentation")
        print("  • tests/ - Test files")
        print("\n📋 Next steps:")
        print("  1. Review PROJECT_STRUCTURE.md")
        print("  2. Test the reorganized structure")
        print("  3. Update any hardcoded paths if needed")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()