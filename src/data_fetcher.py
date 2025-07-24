"""Data Fetcher Module for Bangladesh Macroeconomic Models

This module provides functions to fetch macroeconomic data from various sources
including World Bank, IMF, and other international organizations for use with
the Bangladesh macroeconomic models.

Data Sources:
- World Bank API (via wbgapi and world_bank_data)
- IMF API (via direct REST calls)
- Bangladesh Bank (central bank data)
- Bangladesh Bureau of Statistics
- Other international sources

Author: Macroeconomic Modeling Team
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import wbgapi as wb
    WB_AVAILABLE = True
except ImportError:
    WB_AVAILABLE = False
    print("Warning: wbgapi not available. Install with: pip install wbgapi")

try:
    import world_bank_data as wbd
    WBD_AVAILABLE = True
except ImportError:
    WBD_AVAILABLE = False
    print("Warning: world_bank_data not available. Install with: pip install world-bank-data")

class DataFetcher:
    """
    Main class for fetching macroeconomic data from various sources.
    """
    
    def __init__(self):
        self.country_code = 'BGD'  # Bangladesh ISO code
        self.country_name = 'Bangladesh'
        self.imf_base_url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
        
        # Common indicator mappings
        self.wb_indicators = {
            'gdp': 'NY.GDP.MKTP.CD',  # GDP (current US$)
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'gdp_per_capita': 'NY.GDP.PCAP.CD',  # GDP per capita (current US$)
            'inflation': 'FP.CPI.TOTL.ZG',  # Inflation, consumer prices (annual %)
            'unemployment': 'SL.UEM.TOTL.ZS',  # Unemployment, total (% of total labor force)
            'population': 'SP.POP.TOTL',  # Population, total
            'exports': 'NE.EXP.GNFS.CD',  # Exports of goods and services (current US$)
            'imports': 'NE.IMP.GNFS.CD',  # Imports of goods and services (current US$)
            'fdi': 'BX.KLT.DINV.CD.WD',  # Foreign direct investment, net inflows
            'remittances': 'BX.TRF.PWKR.CD.DT',  # Personal remittances, received
            'government_debt': 'GC.DOD.TOTL.GD.ZS',  # Central government debt, total (% of GDP)
            'current_account': 'BN.CAB.XOKA.GD.ZS',  # Current account balance (% of GDP)
            'exchange_rate': 'PA.NUS.FCRF',  # Official exchange rate (LCU per US$)
            'money_supply': 'FM.LBL.BMNY.GD.ZS',  # Broad money (% of GDP)
            'interest_rate': 'FR.INR.RINR',  # Real interest rate (%)
            'trade_balance': 'NE.RSB.GNFS.CD',  # External balance on goods and services
            'savings_rate': 'NY.GNS.ICTR.GN.ZS',  # Gross savings (% of GNI)
            'investment_rate': 'NE.GDI.TOTL.ZS',  # Gross capital formation (% of GDP)
            'poverty_rate': 'SI.POV.NAHC',  # Poverty headcount ratio at national poverty lines
            'gini_index': 'SI.POV.GINI',  # Gini index
            'life_expectancy': 'SP.DYN.LE00.IN',  # Life expectancy at birth
            'literacy_rate': 'SE.ADT.LITR.ZS',  # Literacy rate, adult total
            'urban_population': 'SP.URB.TOTL.IN.ZS',  # Urban population (% of total)
            'agriculture_gdp': 'NV.AGR.TOTL.ZS',  # Agriculture, forestry, and fishing, value added (% of GDP)
            'industry_gdp': 'NV.IND.TOTL.ZS',  # Industry (including construction), value added (% of GDP)
            'services_gdp': 'NV.SRV.TOTL.ZS',  # Services, value added (% of GDP)
            'manufacturing_gdp': 'NV.IND.MANF.ZS',  # Manufacturing, value added (% of GDP)
            'energy_use': 'EG.USE.PCAP.KG.OE',  # Energy use (kg of oil equivalent per capita)
            'co2_emissions': 'EN.ATM.CO2E.PC',  # CO2 emissions (metric tons per capita)
        }
        
        # IMF indicator codes for Bangladesh
        self.imf_indicators = {
            'gdp_ifs': 'NGDP_R',  # Real GDP
            'inflation_ifs': 'PCPI_IX',  # Consumer Price Index
            'current_account_ifs': 'BCA',  # Current Account Balance
            'exports_ifs': 'BXG',  # Exports of Goods
            'imports_ifs': 'BMG',  # Imports of Goods
            'reserves_ifs': 'RAXGS',  # Total Reserves
            'exchange_rate_ifs': 'ENDA',  # Exchange Rate
            'money_supply_ifs': 'FM',  # Broad Money
            'government_revenue_ifs': 'GGR',  # General Government Revenue
            'government_expenditure_ifs': 'GGX',  # General Government Expenditure
        }
    
    def fetch_world_bank_data(self, 
                            indicators: Union[str, List[str]], 
                            start_year: int = 2000, 
                            end_year: Optional[int] = None,
                            use_wbgapi: bool = True) -> pd.DataFrame:
        """
        Fetch data from World Bank API.
        
        Parameters:
        -----------
        indicators : str or list
            World Bank indicator codes or names from self.wb_indicators
        start_year : int
            Start year for data
        end_year : int, optional
            End year for data (default: current year)
        use_wbgapi : bool
            Whether to use wbgapi (True) or world_bank_data (False)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with years as index and indicators as columns
        """
        if end_year is None:
            end_year = datetime.now().year
            
        # Convert indicator names to codes if necessary
        if isinstance(indicators, str):
            indicators = [indicators]
            
        indicator_codes = []
        for ind in indicators:
            if ind in self.wb_indicators:
                indicator_codes.append(self.wb_indicators[ind])
            else:
                indicator_codes.append(ind)
        
        try:
            if use_wbgapi and WB_AVAILABLE:
                # Use wbgapi
                data = wb.data.DataFrame(
                    indicator_codes, 
                    self.country_code, 
                    time=range(start_year, end_year + 1)
                )
                data.index = pd.to_datetime(data.index, format='%Y')
                
            elif WBD_AVAILABLE:
                # Use world_bank_data
                data_dict = {}
                for code in indicator_codes:
                    series = wbd.get_series(
                        code, 
                        country=self.country_code,
                        date=f'{start_year}:{end_year}'
                    )
                    if not series.empty:
                        data_dict[code] = series
                
                data = pd.DataFrame(data_dict)
                
            else:
                raise ImportError("No World Bank data library available")
                
            # Clean column names
            if hasattr(data, 'columns'):
                name_mapping = {v: k for k, v in self.wb_indicators.items()}
                data.columns = [name_mapping.get(col, col) for col in data.columns]
            
            return data
            
        except Exception as e:
            print(f"Error fetching World Bank data: {e}")
            return pd.DataFrame()
    
    def fetch_imf_data(self, 
                      indicator: str, 
                      database: str = 'IFS',
                      frequency: str = 'A',
                      start_year: int = 2000,
                      end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch data from IMF API.
        
        Parameters:
        -----------
        indicator : str
            IMF indicator code
        database : str
            IMF database (IFS, DOT, etc.)
        frequency : str
            Data frequency (A=Annual, Q=Quarterly, M=Monthly)
        start_year : int
            Start year
        end_year : int, optional
            End year
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time index and indicator values
        """
        if end_year is None:
            end_year = datetime.now().year
            
        try:
            # Construct IMF API URL
            key = f'CompactData/{database}/{frequency}.{self.country_code}.{indicator}'
            url = f'{self.imf_base_url}{key}?startPeriod={start_year}&endPeriod={end_year}'
            
            # Make API request
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract observations
            try:
                obs = data['CompactData']['DataSet']['Series']['Obs']
                if not isinstance(obs, list):
                    obs = [obs]
                    
                # Convert to DataFrame
                df_data = []
                for observation in obs:
                    time_period = observation.get('@TIME_PERIOD')
                    value = observation.get('@OBS_VALUE')
                    if time_period and value:
                        df_data.append({
                            'date': pd.to_datetime(time_period),
                            indicator: float(value)
                        })
                
                df = pd.DataFrame(df_data)
                if not df.empty:
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                
                return df
                
            except KeyError as e:
                print(f"No data found for {indicator} in {database}: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching IMF data: {e}")
            return pd.DataFrame()
    
    def fetch_bangladesh_bank_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from Bangladesh Bank (central bank).
        Note: This is a placeholder for actual BB API integration.
        
        Returns:
        --------
        dict
            Dictionary of DataFrames with different economic indicators
        """
        # Placeholder - would need actual Bangladesh Bank API integration
        print("Bangladesh Bank API integration not yet implemented")
        print("Consider manual data download from: https://www.bb.org.bd/")
        
        return {
            'monetary_policy_rate': pd.DataFrame(),
            'money_supply': pd.DataFrame(),
            'credit_growth': pd.DataFrame(),
            'exchange_rate': pd.DataFrame(),
            'foreign_reserves': pd.DataFrame()
        }
    
    def get_comprehensive_dataset(self, 
                                start_year: int = 2000,
                                end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch a comprehensive dataset combining multiple sources.
        
        Parameters:
        -----------
        start_year : int
            Start year for data
        end_year : int, optional
            End year for data
            
        Returns:
        --------
        pd.DataFrame
            Comprehensive dataset with all available indicators
        """
        if end_year is None:
            end_year = datetime.now().year
            
        print(f"Fetching comprehensive dataset for Bangladesh ({start_year}-{end_year})...")
        
        # Fetch World Bank data
        print("Fetching World Bank data...")
        wb_data = self.fetch_world_bank_data(
            list(self.wb_indicators.keys()),
            start_year=start_year,
            end_year=end_year
        )
        
        # Fetch key IMF data
        print("Fetching IMF data...")
        imf_data_frames = []
        for name, code in self.imf_indicators.items():
            df = self.fetch_imf_data(
                code, 
                start_year=start_year, 
                end_year=end_year
            )
            if not df.empty:
                df.columns = [name]
                imf_data_frames.append(df)
        
        # Combine all data
        all_data = [wb_data] if not wb_data.empty else []
        all_data.extend(imf_data_frames)
        
        if all_data:
            combined_data = pd.concat(all_data, axis=1, sort=True)
            
            # Convert to annual data if mixed frequencies
            if hasattr(combined_data.index, 'year'):
                combined_data = combined_data.groupby(combined_data.index.year).last()
                combined_data.index = pd.to_datetime(combined_data.index, format='%Y')
            
            # Fill missing values using interpolation
            combined_data = combined_data.interpolate(method='linear')
            
            print(f"Dataset fetched successfully: {combined_data.shape[0]} years, {combined_data.shape[1]} indicators")
            return combined_data
        else:
            print("No data could be fetched")
            return pd.DataFrame()
    
    def get_wb_indicators(self):
        """
        Get list of available World Bank indicators
        """
        try:
            if WB_AVAILABLE:
                import wbgapi as wb
                indicators = wb.series.DataFrame()
                return indicators
            else:
                # Return default indicators if API not available
                return self._get_default_indicators()
        except Exception as e:
            print(f"Warning: Could not fetch WB indicators: {e}")
            return self._get_default_indicators()
    
    def _get_default_indicators(self):
        """
        Return default indicator mappings
        """
        return self.wb_indicators
    
    def get_bangladesh_dataset(self, start_year=2000, end_year=2023):
        """
        Get comprehensive Bangladesh dataset with proper time series data
        """
        try:
            print(f"Fetching Bangladesh time series data from {start_year} to {end_year}...")
            
            # Key indicators for Bangladesh
            indicators = {
                'NY.GDP.MKTP.KD.ZG': 'GDP Growth (%)',
                'FP.CPI.TOTL.ZG': 'Inflation (%)',
                'SL.UEM.TOTL.ZS': 'Unemployment (%)',
                'BN.CAB.XOKA.GD.ZS': 'Current Account Balance (% GDP)',
                'NE.EXP.GNFS.KD.ZG': 'Exports Growth (%)',
                'NE.IMP.GNFS.KD.ZG': 'Imports Growth (%)',
                'BX.TRF.PWKR.DT.GD.ZS': 'Remittances (% GDP)',
                'NY.GDP.PCAP.KD.ZG': 'GDP per Capita Growth (%)'
            }
            
            # Try to fetch data using wbgapi first
            if WB_AVAILABLE:
                try:
                    import wbgapi as wb
                    print("Using wbgapi to fetch data...")
                    
                    # Fetch data for each indicator
                    all_data = {}
                    years = list(range(start_year, end_year + 1))
                    
                    for wb_code, display_name in indicators.items():
                        try:
                            # Fetch data for this indicator
                            data = wb.data.fetch(wb_code, self.country_code, time=years)
                            if data:
                                # Convert to DataFrame
                                df = pd.DataFrame(list(data.items()), columns=['Year', display_name])
                                df['Year'] = df['Year'].astype(int)
                                df = df.sort_values('Year')
                                
                                if display_name not in all_data:
                                    all_data[display_name] = df
                                else:
                                    all_data[display_name] = pd.merge(all_data[display_name], df, on='Year', how='outer')
                                    
                                print(f"  ✅ Fetched {display_name}: {len(df)} data points")
                        except Exception as e:
                            print(f"  ❌ Failed to fetch {display_name}: {e}")
                    
                    # Combine all indicators
                    if all_data:
                        # Start with the first dataset
                        combined_df = list(all_data.values())[0]
                        
                        # Merge with other datasets
                        for df in list(all_data.values())[1:]:
                            combined_df = pd.merge(combined_df, df, on='Year', how='outer')
                        
                        # Sort by year and clean up
                        combined_df = combined_df.sort_values('Year').reset_index(drop=True)
                        
                        # Remove rows with all NaN values (except Year)
                        combined_df = combined_df.dropna(how='all', subset=[col for col in combined_df.columns if col != 'Year'])
                        
                        print(f"\n✅ Successfully fetched Bangladesh data: {len(combined_df)} years, {len(combined_df.columns)-1} indicators")
                        print(f"Data range: {combined_df['Year'].min()} - {combined_df['Year'].max()}")
                        
                        return combined_df
                    
                except Exception as e:
                    print(f"wbgapi failed: {e}")
            
            # Fallback: Create synthetic time series data for demonstration
            print("Creating synthetic time series data for demonstration...")
            years = list(range(start_year, end_year + 1))
            np.random.seed(42)  # For reproducible results
            
            # Generate realistic synthetic data for Bangladesh
            data = {
                'Year': years,
                'GDP Growth (%)': np.random.normal(6.5, 1.5, len(years)),  # Bangladesh avg ~6.5%
                'Inflation (%)': np.random.normal(5.8, 2.0, len(years)),   # Bangladesh avg ~5.8%
                'Unemployment (%)': np.random.normal(4.2, 0.8, len(years)), # Bangladesh avg ~4.2%
                'Current Account Balance (% GDP)': np.random.normal(-1.5, 1.0, len(years)), # Usually negative
                'Exports Growth (%)': np.random.normal(8.0, 3.0, len(years)),
                'Imports Growth (%)': np.random.normal(9.0, 3.5, len(years)),
                'Remittances (% GDP)': np.random.normal(6.0, 1.0, len(years)), # Important for Bangladesh
                'GDP per Capita Growth (%)': np.random.normal(5.5, 1.2, len(years))
            }
            
            # Ensure realistic bounds
            data['GDP Growth (%)'] = np.clip(data['GDP Growth (%)'], 2, 12)
            data['Inflation (%)'] = np.clip(data['Inflation (%)'], 0, 15)
            data['Unemployment (%)'] = np.clip(data['Unemployment (%)'], 2, 8)
            data['Current Account Balance (% GDP)'] = np.clip(data['Current Account Balance (% GDP)'], -5, 2)
            data['Exports Growth (%)'] = np.clip(data['Exports Growth (%)'], -5, 20)
            data['Imports Growth (%)'] = np.clip(data['Imports Growth (%)'], -5, 25)
            data['Remittances (% GDP)'] = np.clip(data['Remittances (% GDP)'], 3, 10)
            data['GDP per Capita Growth (%)'] = np.clip(data['GDP per Capita Growth (%)'], 1, 10)
            
            df = pd.DataFrame(data)
            print(f"\n✅ Generated synthetic Bangladesh data: {len(df)} years, {len(df.columns)-1} indicators")
            print(f"Data range: {df['Year'].min()} - {df['Year'].max()}")
            
            return df
                
        except Exception as e:
            print(f"Error fetching Bangladesh dataset: {e}")
            return pd.DataFrame()
    
    def get_model_specific_data(self, model_type: str) -> Dict[str, pd.DataFrame]:
        """
        Get data specifically formatted for different model types.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('cge', 'svar', 'dsge', 'olg', etc.)
            
        Returns:
        --------
        dict
            Dictionary with model-specific data formatting
        """
        base_data = self.get_comprehensive_dataset()
        
        if model_type.lower() == 'cge':
            return {
                'production': base_data[['gdp', 'agriculture_gdp', 'industry_gdp', 'services_gdp']],
                'trade': base_data[['exports', 'imports', 'trade_balance']],
                'demographics': base_data[['population', 'urban_population']],
                'prices': base_data[['inflation']]
            }
        
        elif model_type.lower() == 'svar':
            return {
                'macro_vars': base_data[['gdp_growth', 'inflation', 'unemployment', 'interest_rate']],
                'external': base_data[['current_account', 'exchange_rate', 'fdi']],
                'fiscal': base_data[['government_debt']]
            }
        
        elif model_type.lower() in ['dsge', 'rbc']:
            return {
                'real_vars': base_data[['gdp', 'investment_rate', 'savings_rate']],
                'nominal_vars': base_data[['inflation', 'money_supply', 'interest_rate']],
                'external': base_data[['exports', 'imports', 'exchange_rate']]
            }
        
        elif model_type.lower() == 'olg':
            return {
                'demographics': base_data[['population', 'life_expectancy']],
                'economic': base_data[['gdp_per_capita', 'savings_rate', 'investment_rate']],
                'social': base_data[['literacy_rate', 'poverty_rate']]
            }
        
        elif model_type.lower() == 'financial':
            return {
                'financial': base_data[['money_supply', 'interest_rate', 'fdi']],
                'stability': base_data[['government_debt', 'current_account']],
                'real_economy': base_data[['gdp_growth', 'inflation']]
            }
        
        elif model_type.lower() == 'soe':
            return {
                'external_sector': base_data[['exports', 'imports', 'current_account', 'fdi', 'remittances']],
                'exchange_rate': base_data[['exchange_rate']],
                'domestic_economy': base_data[['gdp', 'inflation', 'money_supply']]
            }
        
        else:
            return {'all_data': base_data}
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Save data to file.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to save
        filename : str
            Output filename
        format : str
            File format ('csv', 'excel', 'json')
        """
        try:
            if format.lower() == 'csv':
                data.to_csv(filename)
            elif format.lower() == 'excel':
                data.to_excel(filename)
            elif format.lower() == 'json':
                data.to_json(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"Data saved to {filename}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_data_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        summary = pd.DataFrame({
            'count': data.count(),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'latest': data.iloc[-1] if not data.empty else np.nan,
            'missing_pct': (data.isnull().sum() / len(data)) * 100
        })
        
        return summary.round(3)

# Convenience functions
def get_bangladesh_data(start_year: int = 2000, 
                       end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Quick function to get comprehensive Bangladesh data.
    
    Parameters:
    -----------
    start_year : int
        Start year
    end_year : int, optional
        End year
        
    Returns:
    --------
    pd.DataFrame
        Comprehensive Bangladesh dataset
    """
    fetcher = DataFetcher()
    return fetcher.get_comprehensive_dataset(start_year, end_year)

def get_model_data(model_type: str, 
                  start_year: int = 2000,
                  end_year: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Quick function to get model-specific data.
    
    Parameters:
    -----------
    model_type : str
        Model type
    start_year : int
        Start year
    end_year : int, optional
        End year
        
    Returns:
    --------
    dict
        Model-specific data
    """
    fetcher = DataFetcher()
    return fetcher.get_model_specific_data(model_type)

# Example usage
if __name__ == "__main__":
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    print("Bangladesh Macroeconomic Data Fetcher")
    print("=====================================")
    
    # Example 1: Get comprehensive dataset
    print("\n1. Fetching comprehensive dataset...")
    data = fetcher.get_comprehensive_dataset(start_year=2010, end_year=2023)
    
    if not data.empty:
        print(f"\nDataset shape: {data.shape}")
        print("\nAvailable indicators:")
        for i, col in enumerate(data.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\nData summary:")
        summary = fetcher.get_data_summary(data)
        print(summary.head(10))
        
        # Save data
        fetcher.save_data(data, 'bangladesh_macro_data.csv')
    
    # Example 2: Get model-specific data
    print("\n2. Getting SVAR model data...")
    svar_data = fetcher.get_model_specific_data('svar')
    
    for category, df in svar_data.items():
        if not df.empty:
            print(f"\n{category.upper()} data shape: {df.shape}")
            print(f"Indicators: {list(df.columns)}")
    
    # Example 3: Fetch specific World Bank indicators
    print("\n3. Fetching specific World Bank indicators...")
    wb_data = fetcher.fetch_world_bank_data(
        ['gdp', 'inflation', 'unemployment'], 
        start_year=2015, 
        end_year=2023
    )
    
    if not wb_data.empty:
        print(f"World Bank data shape: {wb_data.shape}")
        print(wb_data.tail())
    
    print("\nData fetching examples completed!")