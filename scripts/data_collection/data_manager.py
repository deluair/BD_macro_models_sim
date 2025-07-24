#!/usr/bin/env python3
"""
Bangladesh Macroeconomic Data Manager

This script provides a centralized interface for managing and accessing
all working data in the Bangladesh macroeconomic modeling project.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data_fetcher import DataFetcher

class DataManager:
    """
    Centralized data management for Bangladesh macroeconomic modeling project.
    """
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Define data directories
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.external_data_dir = self.data_dir / "external"
        
        # Define visualization directories
        self.viz_dir = self.project_root / "visualization"
        self.dashboard_dir = self.viz_dir / "dashboards"
        self.charts_dir = self.viz_dir / "charts"
        
        # Define results directory
        self.results_dir = self.project_root / "results"
        
        # Initialize data fetcher
        self.fetcher = DataFetcher()
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.dashboard_dir,
            self.charts_dir,
            self.results_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_project_structure(self):
        """Display the current project structure."""
        print("\n=== Bangladesh Macroeconomic Modeling Project Structure ===")
        print(f"Project Root: {self.project_root}")
        print("\nData Organization:")
        print(f"  📁 Raw Data: {self.raw_data_dir}")
        print(f"  📁 Processed Data: {self.processed_data_dir}")
        print(f"  📁 External Data: {self.external_data_dir}")
        print("\nVisualization:")
        print(f"  📁 Dashboards: {self.dashboard_dir}")
        print(f"  📁 Charts: {self.charts_dir}")
        print("\nResults:")
        print(f"  📁 Results: {self.results_dir}")
    
    def list_available_data(self):
        """List all available data files."""
        print("\n=== Available Data Files ===")
        
        # Check processed data
        processed_files = list(self.processed_data_dir.glob("*.csv"))
        if processed_files:
            print("\n📊 Processed Data:")
            for file in processed_files:
                print(f"  - {file.name}")
        
        # Check raw data
        raw_files = list(self.raw_data_dir.glob("*"))
        if raw_files:
            print("\n📥 Raw Data:")
            for file in raw_files:
                print(f"  - {file.name}")
        
        # Check external data
        external_files = list(self.external_data_dir.glob("*"))
        if external_files:
            print("\n🌐 External Data:")
            for file in external_files:
                print(f"  - {file.name}")
    
    def list_visualizations(self):
        """List all available visualizations."""
        print("\n=== Available Visualizations ===")
        
        # Check dashboards
        dashboard_files = list(self.dashboard_dir.glob("*.png"))
        if dashboard_files:
            print("\n📈 Dashboards:")
            for file in dashboard_files:
                print(f"  - {file.name}")
        
        # Check charts
        chart_files = list(self.charts_dir.glob("*.png"))
        if chart_files:
            print("\n📊 Charts:")
            for file in chart_files:
                print(f"  - {file.name}")
    
    def load_bangladesh_data(self):
        """Load the main Bangladesh macroeconomic dataset."""
        # Try time series data first, then fall back to single-point data
        time_series_file = self.processed_data_dir / "bangladesh_time_series_data.csv"
        single_point_file = self.processed_data_dir / "bangladesh_macroeconomic_data.csv"
        
        if time_series_file.exists():
            print(f"\n📊 Loading Bangladesh time series data from {time_series_file}")
            df = pd.read_csv(time_series_file)
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df['Year'].min()} - {df['Year'].max()}")
            print(f"\nColumns: {list(df.columns)}")
            return df
        elif single_point_file.exists():
            print(f"\n📊 Loading Bangladesh macroeconomic data from {single_point_file}")
            df = pd.read_csv(single_point_file)
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df['Year'].min()} - {df['Year'].max()}")
            print(f"\nColumns: {list(df.columns)}")
            return df
        else:
            print(f"\n❌ No data files found")
            print("Run the data fetching script first to generate the data.")
            return None
    
    def get_data_summary(self):
        """Get a comprehensive summary of available data."""
        df = self.load_bangladesh_data()
        
        if df is not None:
            print("\n=== Data Summary ===")
            print(df.describe())
            
            print("\n=== Missing Data Analysis ===")
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            missing_summary = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing Percentage': missing_pct
            })
            print(missing_summary[missing_summary['Missing Count'] > 0])
            
            return df
        return None
    
    def refresh_data(self):
        """Fetch fresh data from APIs and update local files."""
        print("\n🔄 Refreshing Bangladesh macroeconomic data...")
        
        try:
            # Fetch fresh data
            df = self.fetcher.get_bangladesh_dataset()
            
            if df is not None and not df.empty:
                # Save to processed data directory
                output_file = self.processed_data_dir / "bangladesh_macroeconomic_data.csv"
                df.to_csv(output_file, index=False)
                print(f"✅ Data saved to {output_file}")
                
                # Create timestamp file
                timestamp_file = self.processed_data_dir / "last_updated.txt"
                with open(timestamp_file, 'w') as f:
                    f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                return df
            else:
                print("❌ Failed to fetch data")
                return None
                
        except Exception as e:
            print(f"❌ Error refreshing data: {e}")
            return None
    
    def create_quick_dashboard(self):
        """Create a quick dashboard with key indicators."""
        df = self.load_bangladesh_data()
        
        if df is None:
            print("No data available. Run refresh_data() first.")
            return
        
        print("\n📈 Creating quick dashboard...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bangladesh Macroeconomic Indicators - Quick Dashboard', fontsize=16, fontweight='bold')
        
        # GDP Growth
        if 'GDP Growth (%)' in df.columns:
            axes[0, 0].plot(df['Year'], df['GDP Growth (%)'], marker='o', linewidth=2, color='blue')
            axes[0, 0].set_title('GDP Growth Rate (%)')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Growth Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Inflation
        if 'Inflation (%)' in df.columns:
            axes[0, 1].plot(df['Year'], df['Inflation (%)'], marker='s', linewidth=2, color='red')
            axes[0, 1].set_title('Inflation Rate (%)')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Inflation Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Current Account Balance
        if 'Current Account Balance (% GDP)' in df.columns:
            axes[1, 0].plot(df['Year'], df['Current Account Balance (% GDP)'], marker='^', linewidth=2, color='green')
            axes[1, 0].set_title('Current Account Balance (% of GDP)')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Balance (% of GDP)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Unemployment
        if 'Unemployment (%)' in df.columns:
            axes[1, 1].plot(df['Year'], df['Unemployment (%)'], marker='d', linewidth=2, color='orange')
            axes[1, 1].set_title('Unemployment Rate (%)')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Unemployment Rate (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = self.dashboard_dir / "quick_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to {dashboard_file}")
        
        plt.show()
    
    def export_for_models(self, model_type='all'):
        """Export data in formats suitable for different economic models."""
        df = self.load_bangladesh_data()
        
        if df is None:
            print("No data available for export.")
            return
        
        print(f"\n📤 Exporting data for {model_type} models...")
        
        # Create model-specific exports directory
        export_dir = self.processed_data_dir / "model_exports"
        export_dir.mkdir(exist_ok=True)
        
        if model_type in ['all', 'dsge']:
            # DSGE model format
            dsge_data = df[['Year', 'GDP Growth (%)', 'Inflation (%)', 'Unemployment (%)']].dropna()
            dsge_file = export_dir / "bangladesh_dsge_data.csv"
            dsge_data.to_csv(dsge_file, index=False)
            print(f"  ✅ DSGE data: {dsge_file}")
        
        if model_type in ['all', 'svar']:
            # SVAR model format
            svar_data = df[['Year', 'GDP Growth (%)', 'Inflation (%)', 'Current Account Balance (% GDP)']].dropna()
            svar_file = export_dir / "bangladesh_svar_data.csv"
            svar_data.to_csv(svar_file, index=False)
            print(f"  ✅ SVAR data: {svar_file}")
        
        if model_type in ['all', 'cge']:
            # CGE model format
            cge_data = df[['Year', 'GDP Growth (%)', 'Exports Growth (%)', 'Imports Growth (%)']].dropna()
            cge_file = export_dir / "bangladesh_cge_data.csv"
            cge_data.to_csv(cge_file, index=False)
            print(f"  ✅ CGE data: {cge_file}")
        
        print(f"\n📁 All exports saved to: {export_dir}")

def main():
    """Main function to demonstrate data manager functionality."""
    print("🇧🇩 Bangladesh Macroeconomic Data Manager")
    print("=" * 50)
    
    # Initialize data manager
    dm = DataManager()
    
    # Show project structure
    dm.get_project_structure()
    
    # List available data
    dm.list_available_data()
    
    # List visualizations
    dm.list_visualizations()
    
    # Load and summarize data
    df = dm.get_data_summary()
    
    if df is not None:
        # Create quick dashboard
        dm.create_quick_dashboard()
        
        # Export data for models
        dm.export_for_models()
        
        print("\n✅ Data management complete!")
        print("\n📋 Next steps:")
        print("  1. Use the exported data files for model calibration")
        print("  2. Check the visualization/dashboards/ folder for charts")
        print("  3. Run specific model scripts from the models/ directory")
        print("  4. Use dm.refresh_data() to update with latest data")
    else:
        print("\n⚠️  No data found. Run dm.refresh_data() to fetch fresh data.")

if __name__ == "__main__":
    main()