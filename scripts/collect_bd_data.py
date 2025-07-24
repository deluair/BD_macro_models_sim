#!/usr/bin/env python3
"""Data collection script for Bangladesh economic data.

This script collects macroeconomic data for Bangladesh from various sources:
- World Bank Open Data API
- Bangladesh Bank (Central Bank) data
- IMF data
- Local data files

Usage:
    python scripts/collect_bd_data.py --source worldbank --indicators GDP,INFLATION
    python scripts/collect_bd_data.py --all --start-year 2000 --end-year 2023
    python scripts/collect_bd_data.py --update --output data/bd_macro_data.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from utils.logging_config import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

try:
    from config.config_manager import ConfigManager
except ImportError:
    class ConfigManager:
        def get(self, key, default=None):
            return default

# Optional dependencies
try:
    import wbdata
    HAS_WBDATA = True
except ImportError:
    HAS_WBDATA = False
    warnings.warn("wbdata not available. World Bank data collection will be limited.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinance not available. Financial data collection will be limited.")

try:
    import quandl
    HAS_QUANDL = True
except ImportError:
    HAS_QUANDL = False
    warnings.warn("quandl not available. Some data sources will be unavailable.")

logger = get_logger(__name__)


class DataCollector:
    """Comprehensive data collector for Bangladesh economic data."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.session = self._create_session()
        
        # Data source configurations
        self.world_bank_indicators = {
            'GDP': 'NY.GDP.MKTP.CD',  # GDP (current US$)
            'GDP_GROWTH': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'GDP_PER_CAPITA': 'NY.GDP.PCAP.CD',  # GDP per capita (current US$)
            'INFLATION': 'FP.CPI.TOTL.ZG',  # Inflation, consumer prices (annual %)
            'UNEMPLOYMENT': 'SL.UEM.TOTL.ZS',  # Unemployment, total (% of total labor force)
            'POPULATION': 'SP.POP.TOTL',  # Population, total
            'EXPORTS': 'NE.EXP.GNFS.CD',  # Exports of goods and services (current US$)
            'IMPORTS': 'NE.IMP.GNFS.CD',  # Imports of goods and services (current US$)
            'FDI': 'BX.KLT.DINV.CD.WD',  # Foreign direct investment, net inflows
            'REMITTANCES': 'BX.TRF.PWKR.CD.DT',  # Personal remittances, received
            'GOVERNMENT_DEBT': 'GC.DOD.TOTL.GD.ZS',  # Central government debt, total (% of GDP)
            'CURRENT_ACCOUNT': 'BN.CAB.XOKA.CD',  # Current account balance (current US$)
            'RESERVES': 'FI.RES.TOTL.CD',  # Total reserves (current US$)
            'EXCHANGE_RATE': 'PA.NUS.FCRF',  # Official exchange rate (LCU per US$, period average)
            'AGRICULTURE_VA': 'NV.AGR.TOTL.ZS',  # Agriculture, value added (% of GDP)
            'INDUSTRY_VA': 'NV.IND.TOTL.ZS',  # Industry, value added (% of GDP)
            'SERVICES_VA': 'NV.SRV.TOTL.ZS',  # Services, value added (% of GDP)
            'LABOR_FORCE': 'SL.TLF.TOTL.IN',  # Labor force, total
            'URBAN_POPULATION': 'SP.URB.TOTL.IN.ZS',  # Urban population (% of total)
            'POVERTY_RATE': 'SI.POV.NAHC',  # Poverty headcount ratio at national poverty lines
            'GINI_INDEX': 'SI.POV.GINI',  # Gini index
            'EDUCATION_EXPENDITURE': 'SE.XPD.TOTL.GD.ZS',  # Government expenditure on education
            'HEALTH_EXPENDITURE': 'SH.XPD.CHEX.GD.ZS',  # Current health expenditure (% of GDP)
            'INTERNET_USERS': 'IT.NET.USER.ZS',  # Individuals using the Internet (% of population)
            'MOBILE_SUBSCRIPTIONS': 'IT.CEL.SETS.P2'  # Mobile cellular subscriptions (per 100 people)
        }
        
        # Bangladesh Bank indicators (if API becomes available)
        self.bb_indicators = {
            'REPO_RATE': 'repo_rate',
            'CALL_MONEY_RATE': 'call_money_rate',
            'TREASURY_BILL_91': 'tb_91_day',
            'TREASURY_BILL_182': 'tb_182_day',
            'TREASURY_BILL_364': 'tb_364_day',
            'BROAD_MONEY': 'broad_money_m2',
            'NARROW_MONEY': 'narrow_money_m1',
            'CREDIT_PRIVATE': 'credit_to_private_sector',
            'CREDIT_GOVERNMENT': 'credit_to_government'
        }
        
        # IMF indicators
        self.imf_indicators = {
            'REAL_GDP': 'NGDP_R',
            'NOMINAL_GDP': 'NGDP',
            'CPI': 'PCPI',
            'CURRENT_ACCOUNT_GDP': 'BCA_NGDPD',
            'GOVERNMENT_BALANCE': 'GGR_NGDP'
        }
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def collect_world_bank_data(
        self, 
        indicators: List[str] = None,
        start_year: int = 2000,
        end_year: int = None
    ) -> pd.DataFrame:
        """Collect data from World Bank Open Data API."""
        if not HAS_WBDATA:
            logger.warning("wbdata not available, using alternative method")
            return self._collect_wb_data_manual(indicators, start_year, end_year)
            
        if indicators is None:
            indicators = list(self.world_bank_indicators.keys())
            
        if end_year is None:
            end_year = datetime.now().year
            
        logger.info(f"Collecting World Bank data for {len(indicators)} indicators ({start_year}-{end_year})")
        
        # Map indicator names to World Bank codes
        wb_indicators = {}
        for indicator in indicators:
            if indicator in self.world_bank_indicators:
                wb_indicators[self.world_bank_indicators[indicator]] = indicator
            else:
                logger.warning(f"Unknown indicator: {indicator}")
                
        try:
            # Collect data using wbdata
            data = wbdata.get_dataframe(
                wb_indicators,
                country='BD',  # Bangladesh country code
                data_date=(datetime(start_year, 1, 1), datetime(end_year, 12, 31))
            )
            
            # Clean and process data
            data = data.reset_index()
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
            
            # Rename columns to use our indicator names
            column_mapping = {v: k for k, v in wb_indicators.items()}
            data = data.rename(columns=column_mapping)
            
            logger.info(f"Successfully collected {len(data)} observations")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting World Bank data: {e}")
            return pd.DataFrame()
            
    def _collect_wb_data_manual(
        self, 
        indicators: List[str] = None,
        start_year: int = 2000,
        end_year: int = None
    ) -> pd.DataFrame:
        """Manually collect World Bank data using API calls."""
        if indicators is None:
            indicators = list(self.world_bank_indicators.keys())
            
        if end_year is None:
            end_year = datetime.now().year
            
        base_url = "https://api.worldbank.org/v2/country/BD/indicator"
        
        all_data = []
        
        for indicator in indicators:
            if indicator not in self.world_bank_indicators:
                continue
                
            wb_code = self.world_bank_indicators[indicator]
            url = f"{base_url}/{wb_code}"
            
            params = {
                'date': f"{start_year}:{end_year}",
                'format': 'json',
                'per_page': 1000
            }
            
            try:
                logger.info(f"Fetching {indicator} ({wb_code})...")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if len(data) > 1 and data[1]:  # Check if data exists
                    for item in data[1]:
                        if item['value'] is not None:
                            all_data.append({
                                'date': item['date'],
                                'indicator': indicator,
                                'value': float(item['value'])
                            })
                            
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
                continue
                
        if all_data:
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Pivot to wide format
            df = df.pivot(index='date', columns='indicator', values='value')
            df = df.reset_index().sort_values('date')
            
            return df
        else:
            return pd.DataFrame()
            
    def collect_bangladesh_bank_data(self) -> pd.DataFrame:
        """Collect data from Bangladesh Bank (placeholder for future implementation)."""
        logger.info("Bangladesh Bank data collection not yet implemented")
        
        # This would be implemented when Bangladesh Bank provides an API
        # For now, return sample data structure
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        
        sample_data = pd.DataFrame({
            'date': dates,
            'repo_rate': np.random.normal(5.5, 0.5, len(dates)),
            'call_money_rate': np.random.normal(4.8, 0.8, len(dates)),
            'broad_money_growth': np.random.normal(12, 2, len(dates))
        })
        
        logger.warning("Using sample Bangladesh Bank data - implement actual API when available")
        return sample_data
        
    def collect_imf_data(
        self, 
        indicators: List[str] = None,
        start_year: int = 2000,
        end_year: int = None
    ) -> pd.DataFrame:
        """Collect data from IMF (placeholder for future implementation)."""
        logger.info("IMF data collection not yet fully implemented")
        
        # This would use IMF's API when properly configured
        # For now, return empty DataFrame
        return pd.DataFrame()
        
    def collect_financial_data(self) -> pd.DataFrame:
        """Collect financial market data."""
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, skipping financial data")
            return pd.DataFrame()
            
        logger.info("Collecting financial market data...")
        
        financial_data = []
        
        # Bangladesh stock market (if available)
        try:
            # Dhaka Stock Exchange indices (if available on Yahoo Finance)
            symbols = ['DSEX.DH', 'DSES.DH', 'DS30.DH']  # These may not be available
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5y")
                    
                    if not hist.empty:
                        hist = hist.reset_index()
                        hist['symbol'] = symbol
                        financial_data.append(hist)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting financial data: {e}")
            
        if financial_data:
            return pd.concat(financial_data, ignore_index=True)
        else:
            logger.warning("No financial data collected")
            return pd.DataFrame()
            
    def collect_trade_data(self) -> pd.DataFrame:
        """Collect international trade data."""
        logger.info("Collecting trade data...")
        
        # This would typically use UN Comtrade API or similar
        # For now, create sample structure
        dates = pd.date_range('2000-01-01', '2023-12-31', freq='M')
        
        # Sample trade data structure
        trade_data = pd.DataFrame({
            'date': dates,
            'exports_total': np.random.normal(3000, 500, len(dates)),  # Million USD
            'imports_total': np.random.normal(4500, 800, len(dates)),  # Million USD
            'exports_rmg': np.random.normal(2500, 400, len(dates)),    # Ready-made garments
            'exports_jute': np.random.normal(100, 20, len(dates)),     # Jute and jute products
            'imports_fuel': np.random.normal(800, 200, len(dates)),    # Fuel imports
            'imports_machinery': np.random.normal(1200, 300, len(dates))  # Machinery imports
        })
        
        logger.warning("Using sample trade data - implement actual API when available")
        return trade_data
        
    def collect_sectoral_data(self) -> pd.DataFrame:
        """Collect sectoral economic data."""
        logger.info("Collecting sectoral data...")
        
        # Sample sectoral data
        sectors = ['agriculture', 'manufacturing', 'services', 'textiles', 'construction']
        years = list(range(2000, 2024))
        
        sectoral_data = []
        
        for year in years:
            for sector in sectors:
                # Sample data - would be replaced with actual data sources
                if sector == 'agriculture':
                    share = np.random.normal(0.15, 0.02)
                    employment = np.random.normal(40, 5)  # Percentage of total employment
                elif sector == 'manufacturing':
                    share = np.random.normal(0.25, 0.03)
                    employment = np.random.normal(20, 3)
                elif sector == 'services':
                    share = np.random.normal(0.50, 0.04)
                    employment = np.random.normal(35, 4)
                elif sector == 'textiles':
                    share = np.random.normal(0.08, 0.01)
                    employment = np.random.normal(12, 2)
                else:  # construction
                    share = np.random.normal(0.02, 0.005)
                    employment = np.random.normal(3, 0.5)
                    
                sectoral_data.append({
                    'year': year,
                    'sector': sector,
                    'gdp_share': max(0, share),
                    'employment_share': max(0, employment),
                    'productivity_index': np.random.normal(100, 10)
                })
                
        df = pd.DataFrame(sectoral_data)
        logger.warning("Using sample sectoral data - implement actual data sources")
        return df
        
    def collect_all_data(
        self, 
        start_year: int = 2000,
        end_year: int = None,
        sources: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Collect data from all available sources."""
        if sources is None:
            sources = ['worldbank', 'bangladesh_bank', 'trade', 'sectoral']
            
        if end_year is None:
            end_year = datetime.now().year
            
        logger.info(f"Collecting data from sources: {', '.join(sources)}")
        
        data_collection = {}
        
        if 'worldbank' in sources:
            try:
                wb_data = self.collect_world_bank_data(
                    start_year=start_year, 
                    end_year=end_year
                )
                if not wb_data.empty:
                    data_collection['worldbank'] = wb_data
            except Exception as e:
                logger.error(f"Error collecting World Bank data: {e}")
                
        if 'bangladesh_bank' in sources:
            try:
                bb_data = self.collect_bangladesh_bank_data()
                if not bb_data.empty:
                    data_collection['bangladesh_bank'] = bb_data
            except Exception as e:
                logger.error(f"Error collecting Bangladesh Bank data: {e}")
                
        if 'financial' in sources:
            try:
                fin_data = self.collect_financial_data()
                if not fin_data.empty:
                    data_collection['financial'] = fin_data
            except Exception as e:
                logger.error(f"Error collecting financial data: {e}")
                
        if 'trade' in sources:
            try:
                trade_data = self.collect_trade_data()
                if not trade_data.empty:
                    data_collection['trade'] = trade_data
            except Exception as e:
                logger.error(f"Error collecting trade data: {e}")
                
        if 'sectoral' in sources:
            try:
                sectoral_data = self.collect_sectoral_data()
                if not sectoral_data.empty:
                    data_collection['sectoral'] = sectoral_data
            except Exception as e:
                logger.error(f"Error collecting sectoral data: {e}")
                
        logger.info(f"Data collection completed. Sources collected: {list(data_collection.keys())}")
        return data_collection
        
    def merge_datasets(self, data_collection: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge datasets from different sources."""
        logger.info("Merging datasets...")
        
        if not data_collection:
            logger.warning("No data to merge")
            return pd.DataFrame()
            
        # Start with the largest time series dataset
        main_df = None
        
        # Find the best base dataset (usually World Bank data)
        if 'worldbank' in data_collection:
            main_df = data_collection['worldbank'].copy()
            logger.info(f"Using World Bank data as base ({len(main_df)} rows)")
        elif 'bangladesh_bank' in data_collection:
            main_df = data_collection['bangladesh_bank'].copy()
            logger.info(f"Using Bangladesh Bank data as base ({len(main_df)} rows)")
        else:
            # Use the first available dataset
            source_name = list(data_collection.keys())[0]
            main_df = data_collection[source_name].copy()
            logger.info(f"Using {source_name} data as base ({len(main_df)} rows)")
            
        # Merge other datasets
        for source_name, df in data_collection.items():
            if df is main_df:
                continue
                
            if 'date' in df.columns:
                # Time series merge
                df['date'] = pd.to_datetime(df['date'])
                main_df['date'] = pd.to_datetime(main_df['date'])
                
                # Merge on date
                main_df = pd.merge(
                    main_df, df, 
                    on='date', 
                    how='outer', 
                    suffixes=('', f'_{source_name}')
                )
                logger.info(f"Merged {source_name} data ({len(df)} rows)")
                
            else:
                logger.warning(f"Cannot merge {source_name} data - no date column")
                
        # Sort by date and clean up
        if 'date' in main_df.columns:
            main_df = main_df.sort_values('date').reset_index(drop=True)
            
        logger.info(f"Final merged dataset: {len(main_df)} rows, {len(main_df.columns)} columns")
        return main_df
        
    def save_data(
        self, 
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """Save collected data to file(s)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, dict):
            # Save multiple datasets
            for source_name, df in data.items():
                if format == 'csv':
                    file_path = output_path.parent / f"{output_path.stem}_{source_name}.csv"
                    df.to_csv(file_path, index=False)
                elif format == 'excel':
                    file_path = output_path.parent / f"{output_path.stem}_{source_name}.xlsx"
                    df.to_excel(file_path, index=False)
                    
                logger.info(f"Saved {source_name} data to {file_path}")
                
        else:
            # Save single dataset
            if format == 'csv':
                data.to_csv(output_path, index=False)
            elif format == 'excel':
                data.to_excel(output_path, index=False)
            elif format == 'parquet':
                data.to_parquet(output_path, index=False)
                
            logger.info(f"Saved merged data to {output_path}")
            
    def update_existing_data(self, existing_file: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Update existing data file with new observations."""
        existing_path = Path(existing_file)
        
        if existing_path.exists():
            logger.info(f"Loading existing data from {existing_path}")
            
            if existing_path.suffix == '.csv':
                existing_data = pd.read_csv(existing_path)
            elif existing_path.suffix == '.xlsx':
                existing_data = pd.read_excel(existing_path)
            elif existing_path.suffix == '.parquet':
                existing_data = pd.read_parquet(existing_path)
            else:
                raise ValueError(f"Unsupported file format: {existing_path.suffix}")
                
            # Convert date columns
            if 'date' in existing_data.columns:
                existing_data['date'] = pd.to_datetime(existing_data['date'])
            if 'date' in new_data.columns:
                new_data['date'] = pd.to_datetime(new_data['date'])
                
            # Find the latest date in existing data
            if 'date' in existing_data.columns and not existing_data.empty:
                latest_date = existing_data['date'].max()
                logger.info(f"Latest date in existing data: {latest_date}")
                
                # Filter new data to only include newer observations
                if 'date' in new_data.columns:
                    new_observations = new_data[new_data['date'] > latest_date]
                    logger.info(f"Found {len(new_observations)} new observations")
                    
                    if not new_observations.empty:
                        # Combine datasets
                        updated_data = pd.concat([existing_data, new_observations], ignore_index=True)
                        updated_data = updated_data.sort_values('date').reset_index(drop=True)
                        
                        logger.info(f"Updated dataset: {len(updated_data)} total rows")
                        return updated_data
                    else:
                        logger.info("No new data to add")
                        return existing_data
                else:
                    logger.warning("New data has no date column, cannot update")
                    return existing_data
            else:
                logger.warning("Existing data has no date column, replacing entirely")
                return new_data
        else:
            logger.info("No existing data file found, creating new file")
            return new_data


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Collect Bangladesh economic data from various sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_bd_data.py --source worldbank --start-year 2010
  python scripts/collect_bd_data.py --all --output data/bd_complete_data.csv
  python scripts/collect_bd_data.py --update --existing data/bd_data.csv
        """
    )
    
    parser.add_argument(
        '--source', 
        choices=['worldbank', 'bangladesh_bank', 'imf', 'financial', 'trade', 'sectoral'],
        help='Specific data source to collect from'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Collect from all available sources'
    )
    parser.add_argument(
        '--indicators', 
        type=str, 
        help='Comma-separated list of indicators to collect'
    )
    parser.add_argument(
        '--start-year', 
        type=int, 
        default=2000,
        help='Start year for data collection (default: 2000)'
    )
    parser.add_argument(
        '--end-year', 
        type=int, 
        help='End year for data collection (default: current year)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/bd_macro_data.csv',
        help='Output file path (default: data/bd_macro_data.csv)'
    )
    parser.add_argument(
        '--format', 
        choices=['csv', 'excel', 'parquet'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--update', 
        action='store_true', 
        help='Update existing data file with new observations'
    )
    parser.add_argument(
        '--existing', 
        type=str, 
        help='Path to existing data file for updates'
    )
    parser.add_argument(
        '--merge', 
        action='store_true', 
        help='Merge data from multiple sources into single file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
        
    # Initialize data collector
    collector = DataCollector()
    
    try:
        if args.source:
            # Collect from specific source
            logger.info(f"Collecting data from {args.source}")
            
            if args.source == 'worldbank':
                indicators = None
                if args.indicators:
                    indicators = [i.strip() for i in args.indicators.split(',')]
                    
                data = collector.collect_world_bank_data(
                    indicators=indicators,
                    start_year=args.start_year,
                    end_year=args.end_year
                )
                
            elif args.source == 'bangladesh_bank':
                data = collector.collect_bangladesh_bank_data()
                
            elif args.source == 'financial':
                data = collector.collect_financial_data()
                
            elif args.source == 'trade':
                data = collector.collect_trade_data()
                
            elif args.source == 'sectoral':
                data = collector.collect_sectoral_data()
                
            else:
                logger.error(f"Source {args.source} not implemented")
                sys.exit(1)
                
            # Handle updates
            if args.update and args.existing:
                data = collector.update_existing_data(args.existing, data)
                
            # Save data
            collector.save_data(data, args.output, args.format)
            
        elif args.all:
            # Collect from all sources
            sources = ['worldbank', 'bangladesh_bank', 'trade', 'sectoral']
            
            data_collection = collector.collect_all_data(
                start_year=args.start_year,
                end_year=args.end_year,
                sources=sources
            )
            
            if args.merge:
                # Merge all datasets
                merged_data = collector.merge_datasets(data_collection)
                
                # Handle updates
                if args.update and args.existing:
                    merged_data = collector.update_existing_data(args.existing, merged_data)
                    
                collector.save_data(merged_data, args.output, args.format)
            else:
                # Save separate files
                collector.save_data(data_collection, args.output, args.format)
                
        else:
            parser.print_help()
            sys.exit(1)
            
        logger.info("Data collection completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()