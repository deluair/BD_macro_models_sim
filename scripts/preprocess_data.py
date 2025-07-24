#!/usr/bin/env python3
"""Data preprocessing script for Bangladesh macroeconomic data.

This script provides comprehensive data preprocessing capabilities:
- Data cleaning and validation
- Missing value handling
- Outlier detection and treatment
- Data transformation and normalization
- Feature engineering
- Time series preparation
- Data quality reporting

Usage:
    python scripts/preprocess_data.py --input data/bd_macro_data.csv --output data/processed/
    python scripts/preprocess_data.py --config config/preprocessing.yaml
    python scripts/preprocess_data.py --validate --report data/quality_report.html
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

try:
    from utils.error_handling import ValidationError, DataError
except ImportError:
    class ValidationError(Exception):
        pass
    class DataError(Exception):
        pass

# Optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

try:
    import ydata_profiling
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False
    warnings.warn("ydata-profiling not available. Automated profiling will be disabled.")

logger = get_logger(__name__)


class DataPreprocessor:
    """Comprehensive data preprocessing for Bangladesh economic data."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.scaler = None
        self.imputer = None
        self.outlier_detector = None
        
        # Economic data validation rules
        self.validation_rules = {
            'GDP': {'min': 0, 'max': 1e12, 'unit': 'USD'},
            'GDP_GROWTH': {'min': -20, 'max': 20, 'unit': 'percent'},
            'INFLATION': {'min': -10, 'max': 50, 'unit': 'percent'},
            'UNEMPLOYMENT': {'min': 0, 'max': 50, 'unit': 'percent'},
            'POPULATION': {'min': 1e8, 'max': 2e8, 'unit': 'count'},
            'EXCHANGE_RATE': {'min': 50, 'max': 150, 'unit': 'BDT_per_USD'},
            'EXPORTS': {'min': 0, 'max': 1e11, 'unit': 'USD'},
            'IMPORTS': {'min': 0, 'max': 1e11, 'unit': 'USD'},
            'REMITTANCES': {'min': 0, 'max': 5e10, 'unit': 'USD'},
            'FDI': {'min': -1e10, 'max': 5e9, 'unit': 'USD'},
            'RESERVES': {'min': 0, 'max': 1e11, 'unit': 'USD'},
            'GOVERNMENT_DEBT': {'min': 0, 'max': 100, 'unit': 'percent_of_GDP'},
            'CURRENT_ACCOUNT': {'min': -2e10, 'max': 1e10, 'unit': 'USD'}
        }
        
        # Economic relationships for validation
        self.economic_relationships = {
            'trade_balance': ('EXPORTS', 'IMPORTS', 'subtract'),
            'gdp_per_capita': ('GDP', 'POPULATION', 'divide'),
            'import_coverage': ('RESERVES', 'IMPORTS', 'months_coverage'),
            'debt_sustainability': ('GOVERNMENT_DEBT', 'GDP_GROWTH', 'sustainability')
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise DataError(f"Error loading data from {file_path}: {e}")
            
    def validate_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic data structure and types."""
        logger.info("Validating data structure...")
        
        validation_report = {
            'structure': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'has_date_column': 'date' in df.columns
            },
            'columns': {},
            'issues': []
        }
        
        # Check for date column
        if 'date' not in df.columns:
            validation_report['issues'].append("No 'date' column found")
        else:
            try:
                df['date'] = pd.to_datetime(df['date'])
                validation_report['structure']['date_range'] = {
                    'start': df['date'].min().isoformat(),
                    'end': df['date'].max().isoformat(),
                    'frequency': self._infer_frequency(df['date'])
                }
            except Exception as e:
                validation_report['issues'].append(f"Date column conversion failed: {e}")
                
        # Analyze each column
        for col in df.columns:
            if col == 'date':
                continue
                
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(df[col].min()) if not df[col].empty else None,
                    'max': float(df[col].max()) if not df[col].empty else None,
                    'mean': float(df[col].mean()) if not df[col].empty else None,
                    'std': float(df[col].std()) if not df[col].empty else None,
                    'zeros': (df[col] == 0).sum(),
                    'negative_values': (df[col] < 0).sum()
                })
                
                # Check against validation rules
                if col in self.validation_rules:
                    rules = self.validation_rules[col]
                    if col_info['min'] is not None and col_info['min'] < rules['min']:
                        validation_report['issues'].append(
                            f"{col}: minimum value {col_info['min']} below expected {rules['min']}"
                        )
                    if col_info['max'] is not None and col_info['max'] > rules['max']:
                        validation_report['issues'].append(
                            f"{col}: maximum value {col_info['max']} above expected {rules['max']}"
                        )
                        
            validation_report['columns'][col] = col_info
            
        logger.info(f"Validation completed. Found {len(validation_report['issues'])} issues")
        return validation_report
        
    def _infer_frequency(self, date_series: pd.Series) -> str:
        """Infer the frequency of a date series."""
        try:
            if len(date_series) < 2:
                return 'unknown'
                
            date_series = date_series.dropna().sort_values()
            diffs = date_series.diff().dropna()
            
            # Most common difference
            mode_diff = diffs.mode().iloc[0] if not diffs.empty else None
            
            if mode_diff is None:
                return 'unknown'
                
            days = mode_diff.days
            
            if days <= 1:
                return 'daily'
            elif 6 <= days <= 8:
                return 'weekly'
            elif 28 <= days <= 32:
                return 'monthly'
            elif 88 <= days <= 95:
                return 'quarterly'
            elif 360 <= days <= 370:
                return 'annual'
            else:
                return f'irregular_{days}days'
                
        except Exception:
            return 'unknown'
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning."""
        logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        # Convert date column
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            
        # Remove completely empty rows and columns
        initial_rows = len(df_clean)
        initial_cols = len(df_clean.columns)
        
        df_clean = df_clean.dropna(how='all')  # Remove empty rows
        df_clean = df_clean.loc[:, df_clean.notna().any()]  # Remove empty columns
        
        logger.info(f"Removed {initial_rows - len(df_clean)} empty rows")
        logger.info(f"Removed {initial_cols - len(df_clean.columns)} empty columns")
        
        # Remove duplicate rows
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {duplicates} duplicate rows")
            
        # Sort by date if available
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
            
        # Clean numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Replace infinite values with NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Remove extreme outliers (beyond 5 standard deviations)
            if df_clean[col].std() > 0:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                extreme_outliers = z_scores > 5
                
                if extreme_outliers.any():
                    outlier_count = extreme_outliers.sum()
                    df_clean.loc[df_clean[col].dropna().index[extreme_outliers], col] = np.nan
                    logger.info(f"Removed {outlier_count} extreme outliers from {col}")
                    
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = 'interpolate',
        **kwargs
    ) -> pd.DataFrame:
        """Handle missing values using various strategies."""
        logger.info(f"Handling missing values using method: {method}")
        
        df_filled = df.copy()
        
        # Separate date and numeric columns
        date_cols = ['date'] if 'date' in df_filled.columns else []
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'drop':
            # Drop rows with any missing values
            initial_rows = len(df_filled)
            df_filled = df_filled.dropna()
            logger.info(f"Dropped {initial_rows - len(df_filled)} rows with missing values")
            
        elif method == 'forward_fill':
            # Forward fill for time series data
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill')
            
        elif method == 'backward_fill':
            # Backward fill
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='bfill')
            
        elif method == 'interpolate':
            # Linear interpolation for time series
            if 'date' in df_filled.columns:
                df_filled = df_filled.set_index('date')
                df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(
                    method='linear', limit_direction='both'
                )
                df_filled = df_filled.reset_index()
            else:
                df_filled[numeric_cols] = df_filled[numeric_cols].interpolate()
                
        elif method == 'mean':
            # Mean imputation
            imputer = SimpleImputer(strategy='mean')
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
            self.imputer = imputer
            
        elif method == 'median':
            # Median imputation
            imputer = SimpleImputer(strategy='median')
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
            self.imputer = imputer
            
        elif method == 'knn':
            # KNN imputation
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
            self.imputer = imputer
            
        elif method == 'seasonal':
            # Seasonal imputation for time series
            if 'date' in df_filled.columns:
                df_filled = self._seasonal_imputation(df_filled, numeric_cols)
            else:
                logger.warning("Seasonal imputation requires date column, falling back to interpolation")
                df_filled[numeric_cols] = df_filled[numeric_cols].interpolate()
                
        else:
            raise ValueError(f"Unknown imputation method: {method}")
            
        # Report missing values after imputation
        remaining_missing = df_filled[numeric_cols].isnull().sum().sum()
        logger.info(f"Missing values after imputation: {remaining_missing}")
        
        return df_filled
        
    def _seasonal_imputation(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Perform seasonal imputation for time series data."""
        df_seasonal = df.copy()
        df_seasonal['date'] = pd.to_datetime(df_seasonal['date'])
        df_seasonal = df_seasonal.set_index('date')
        
        for col in numeric_cols:
            if df_seasonal[col].isnull().any():
                # Try to detect seasonality
                non_null_data = df_seasonal[col].dropna()
                
                if len(non_null_data) > 24:  # Need sufficient data
                    # Use seasonal decomposition if possible
                    try:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        
                        # Resample to regular frequency if needed
                        freq = pd.infer_freq(non_null_data.index)
                        if freq is None:
                            freq = 'M'  # Default to monthly
                            
                        decomposition = seasonal_decompose(
                            non_null_data, 
                            model='additive', 
                            period=12 if freq == 'M' else 4
                        )
                        
                        # Fill missing values using seasonal pattern
                        seasonal_pattern = decomposition.seasonal
                        trend = decomposition.trend
                        
                        for idx in df_seasonal[df_seasonal[col].isnull()].index:
                            # Find similar seasonal period
                            month = idx.month if freq == 'M' else idx.quarter
                            seasonal_value = seasonal_pattern[seasonal_pattern.index.month == month].mean()
                            trend_value = trend.dropna().iloc[-1]  # Use last trend value
                            
                            if not pd.isna(seasonal_value) and not pd.isna(trend_value):
                                df_seasonal.loc[idx, col] = trend_value + seasonal_value
                                
                    except Exception as e:
                        logger.warning(f"Seasonal decomposition failed for {col}: {e}")
                        # Fall back to simple interpolation
                        df_seasonal[col] = df_seasonal[col].interpolate()
                else:
                    # Not enough data for seasonal analysis
                    df_seasonal[col] = df_seasonal[col].interpolate()
                    
        return df_seasonal.reset_index()
        
    def detect_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'isolation_forest',
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect outliers using various methods."""
        logger.info(f"Detecting outliers using method: {method}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No numeric columns found for outlier detection")
            return df, pd.DataFrame()
            
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        
        if method == 'zscore':
            threshold = kwargs.get('threshold', 3)
            
            for col in numeric_cols:
                if df[col].std() > 0:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outlier_mask.loc[df[col].dropna().index, col] = z_scores > threshold
                    
        elif method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
        elif method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            
            # Use only complete cases for training
            complete_data = df[numeric_cols].dropna()
            
            if len(complete_data) > 10:  # Need sufficient data
                detector = IsolationForest(
                    contamination=contamination, 
                    random_state=42
                )
                
                outlier_predictions = detector.fit_predict(complete_data)
                outlier_indices = complete_data.index[outlier_predictions == -1]
                
                for idx in outlier_indices:
                    outlier_mask.loc[idx, :] = True
                    
                self.outlier_detector = detector
            else:
                logger.warning("Insufficient data for isolation forest, using IQR method")
                return self.detect_outliers(df, method='iqr', **kwargs)
                
        elif method == 'modified_zscore':
            threshold = kwargs.get('threshold', 3.5)
            
            for col in numeric_cols:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                
                if mad > 0:
                    modified_z_scores = 0.6745 * (df[col] - median) / mad
                    outlier_mask[col] = np.abs(modified_z_scores) > threshold
                    
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        # Create outlier summary
        outlier_summary = pd.DataFrame({
            'column': numeric_cols,
            'outlier_count': [outlier_mask[col].sum() for col in numeric_cols],
            'outlier_percentage': [(outlier_mask[col].sum() / len(df)) * 100 for col in numeric_cols]
        })
        
        total_outliers = outlier_mask.any(axis=1).sum()
        logger.info(f"Detected {total_outliers} rows with outliers")
        
        return outlier_mask, outlier_summary
        
    def treat_outliers(
        self, 
        df: pd.DataFrame, 
        outlier_mask: pd.DataFrame, 
        method: str = 'winsorize',
        **kwargs
    ) -> pd.DataFrame:
        """Treat detected outliers."""
        logger.info(f"Treating outliers using method: {method}")
        
        df_treated = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'remove':
            # Remove rows with outliers
            outlier_rows = outlier_mask.any(axis=1)
            initial_rows = len(df_treated)
            df_treated = df_treated[~outlier_rows]
            logger.info(f"Removed {initial_rows - len(df_treated)} rows with outliers")
            
        elif method == 'winsorize':
            # Winsorize outliers to percentile limits
            lower_percentile = kwargs.get('lower_percentile', 5)
            upper_percentile = kwargs.get('upper_percentile', 95)
            
            for col in numeric_cols:
                if col in outlier_mask.columns:
                    lower_limit = df_treated[col].quantile(lower_percentile / 100)
                    upper_limit = df_treated[col].quantile(upper_percentile / 100)
                    
                    df_treated[col] = df_treated[col].clip(lower_limit, upper_limit)
                    
        elif method == 'cap':
            # Cap outliers at IQR boundaries
            multiplier = kwargs.get('multiplier', 1.5)
            
            for col in numeric_cols:
                if col in outlier_mask.columns:
                    Q1 = df_treated[col].quantile(0.25)
                    Q3 = df_treated[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    df_treated[col] = df_treated[col].clip(lower_bound, upper_bound)
                    
        elif method == 'transform':
            # Log transform to reduce outlier impact
            for col in numeric_cols:
                if col in outlier_mask.columns and (df_treated[col] > 0).all():
                    df_treated[col] = np.log1p(df_treated[col])
                    
        elif method == 'impute':
            # Replace outliers with median
            for col in numeric_cols:
                if col in outlier_mask.columns:
                    median_value = df_treated[col].median()
                    df_treated.loc[outlier_mask[col], col] = median_value
                    
        else:
            raise ValueError(f"Unknown outlier treatment method: {method}")
            
        logger.info("Outlier treatment completed")
        return df_treated
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data."""
        logger.info("Engineering features...")
        
        df_features = df.copy()
        
        # Time-based features
        if 'date' in df_features.columns:
            df_features['date'] = pd.to_datetime(df_features['date'])
            df_features['year'] = df_features['date'].dt.year
            df_features['month'] = df_features['date'].dt.month
            df_features['quarter'] = df_features['date'].dt.quarter
            
        # Economic indicators
        if 'EXPORTS' in df_features.columns and 'IMPORTS' in df_features.columns:
            df_features['TRADE_BALANCE'] = df_features['EXPORTS'] - df_features['IMPORTS']
            df_features['TRADE_OPENNESS'] = (df_features['EXPORTS'] + df_features['IMPORTS'])
            
            if 'GDP' in df_features.columns:
                df_features['TRADE_OPENNESS_GDP'] = df_features['TRADE_OPENNESS'] / df_features['GDP']
                df_features['EXPORT_GDP_RATIO'] = df_features['EXPORTS'] / df_features['GDP']
                df_features['IMPORT_GDP_RATIO'] = df_features['IMPORTS'] / df_features['GDP']
                
        if 'GDP' in df_features.columns and 'POPULATION' in df_features.columns:
            df_features['GDP_PER_CAPITA_CALC'] = df_features['GDP'] / df_features['POPULATION']
            
        if 'RESERVES' in df_features.columns and 'IMPORTS' in df_features.columns:
            # Import coverage in months
            monthly_imports = df_features['IMPORTS'] / 12
            df_features['IMPORT_COVERAGE_MONTHS'] = df_features['RESERVES'] / monthly_imports
            
        if 'REMITTANCES' in df_features.columns and 'GDP' in df_features.columns:
            df_features['REMITTANCES_GDP_RATIO'] = df_features['REMITTANCES'] / df_features['GDP']
            
        if 'FDI' in df_features.columns and 'GDP' in df_features.columns:
            df_features['FDI_GDP_RATIO'] = df_features['FDI'] / df_features['GDP']
            
        # Growth rates and changes
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['year', 'month', 'quarter']:
                # Year-over-year growth rate
                df_features[f'{col}_YOY_GROWTH'] = df_features[col].pct_change(periods=12) * 100
                
                # Moving averages
                df_features[f'{col}_MA3'] = df_features[col].rolling(window=3).mean()
                df_features[f'{col}_MA12'] = df_features[col].rolling(window=12).mean()
                
                # Volatility (rolling standard deviation)
                df_features[f'{col}_VOLATILITY'] = df_features[col].rolling(window=12).std()
                
        # Cyclical indicators
        if 'GDP' in df_features.columns:
            # GDP trend using Hodrick-Prescott filter (simplified)
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                gdp_cycle, gdp_trend = hpfilter(df_features['GDP'].dropna(), lamb=1600)
                
                # Align with original dataframe
                df_features['GDP_TREND'] = np.nan
                df_features['GDP_CYCLE'] = np.nan
                
                valid_idx = df_features['GDP'].dropna().index
                df_features.loc[valid_idx, 'GDP_TREND'] = gdp_trend
                df_features.loc[valid_idx, 'GDP_CYCLE'] = gdp_cycle
                
            except ImportError:
                logger.warning("statsmodels not available, skipping HP filter")
                
        # Interaction terms
        if 'INFLATION' in df_features.columns and 'UNEMPLOYMENT' in df_features.columns:
            df_features['MISERY_INDEX'] = df_features['INFLATION'] + df_features['UNEMPLOYMENT']
            
        logger.info(f"Feature engineering completed. Added {len(df_features.columns) - len(df.columns)} new features")
        return df_features
        
    def normalize_data(
        self, 
        df: pd.DataFrame, 
        method: str = 'standard',
        exclude_cols: List[str] = None
    ) -> pd.DataFrame:
        """Normalize/scale numeric data."""
        logger.info(f"Normalizing data using method: {method}")
        
        df_normalized = df.copy()
        
        # Columns to exclude from normalization
        if exclude_cols is None:
            exclude_cols = ['date', 'year', 'month', 'quarter']
            
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if not cols_to_normalize:
            logger.warning("No columns to normalize")
            return df_normalized
            
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        # Fit and transform
        df_normalized[cols_to_normalize] = scaler.fit_transform(
            df_normalized[cols_to_normalize]
        )
        
        self.scaler = scaler
        logger.info(f"Normalized {len(cols_to_normalize)} columns")
        
        return df_normalized
        
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series specific features."""
        logger.info("Creating time series features...")
        
        if 'date' not in df.columns:
            logger.warning("No date column found, skipping time series features")
            return df
            
        df_ts = df.copy()
        df_ts['date'] = pd.to_datetime(df_ts['date'])
        df_ts = df_ts.sort_values('date').reset_index(drop=True)
        
        # Lag features
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['year', 'month', 'quarter']:
                # Create lags
                for lag in [1, 3, 6, 12]:
                    df_ts[f'{col}_LAG{lag}'] = df_ts[col].shift(lag)
                    
                # Differencing
                df_ts[f'{col}_DIFF1'] = df_ts[col].diff()
                df_ts[f'{col}_DIFF12'] = df_ts[col].diff(12)
                
                # Seasonal differences
                df_ts[f'{col}_SEASONAL_DIFF'] = df_ts[col] - df_ts[col].shift(12)
                
        # Time trend
        df_ts['TIME_TREND'] = range(len(df_ts))
        
        # Seasonal dummies
        if 'month' in df_ts.columns:
            month_dummies = pd.get_dummies(df_ts['month'], prefix='MONTH')
            df_ts = pd.concat([df_ts, month_dummies], axis=1)
            
        if 'quarter' in df_ts.columns:
            quarter_dummies = pd.get_dummies(df_ts['quarter'], prefix='QUARTER')
            df_ts = pd.concat([df_ts, quarter_dummies], axis=1)
            
        logger.info(f"Created time series features. Final shape: {df_ts.shape}")
        return df_ts
        
    def validate_economic_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate economic relationships and identities."""
        logger.info("Validating economic relationships...")
        
        validation_results = {
            'relationships': {},
            'identities': {},
            'warnings': []
        }
        
        # Check trade balance identity
        if all(col in df.columns for col in ['EXPORTS', 'IMPORTS', 'TRADE_BALANCE']):
            calculated_balance = df['EXPORTS'] - df['IMPORTS']
            balance_diff = np.abs(df['TRADE_BALANCE'] - calculated_balance)
            
            validation_results['identities']['trade_balance'] = {
                'valid': (balance_diff < 0.01 * df['EXPORTS'].abs()).all(),
                'max_error': balance_diff.max(),
                'mean_error': balance_diff.mean()
            }
            
        # Check GDP per capita relationship
        if all(col in df.columns for col in ['GDP', 'POPULATION', 'GDP_PER_CAPITA']):
            calculated_gdp_pc = df['GDP'] / df['POPULATION']
            gdp_pc_diff = np.abs(df['GDP_PER_CAPITA'] - calculated_gdp_pc)
            
            validation_results['identities']['gdp_per_capita'] = {
                'valid': (gdp_pc_diff < 0.01 * df['GDP_PER_CAPITA'].abs()).all(),
                'max_error': gdp_pc_diff.max(),
                'mean_error': gdp_pc_diff.mean()
            }
            
        # Economic relationship checks
        relationships_to_check = [
            ('GDP_GROWTH', 'UNEMPLOYMENT', 'negative'),  # Okun's Law
            ('INFLATION', 'UNEMPLOYMENT', 'negative'),    # Phillips Curve
            ('EXCHANGE_RATE', 'INFLATION', 'positive'),  # Exchange rate pass-through
            ('EXPORTS', 'EXCHANGE_RATE', 'negative'),    # Export competitiveness
            ('IMPORTS', 'GDP', 'positive'),              # Import demand
            ('REMITTANCES', 'EXCHANGE_RATE', 'negative') # Remittance flows
        ]
        
        for var1, var2, expected_relationship in relationships_to_check:
            if var1 in df.columns and var2 in df.columns:
                correlation = df[var1].corr(df[var2])
                
                validation_results['relationships'][f'{var1}_vs_{var2}'] = {
                    'correlation': correlation,
                    'expected': expected_relationship,
                    'consistent': (
                        (expected_relationship == 'positive' and correlation > 0) or
                        (expected_relationship == 'negative' and correlation < 0)
                    )
                }
                
                if not validation_results['relationships'][f'{var1}_vs_{var2}']['consistent']:
                    validation_results['warnings'].append(
                        f"Unexpected relationship between {var1} and {var2}: "
                        f"correlation = {correlation:.3f}, expected {expected_relationship}"
                    )
                    
        logger.info(f"Economic validation completed. Found {len(validation_results['warnings'])} warnings")
        return validation_results
        
    def generate_quality_report(
        self, 
        df: pd.DataFrame, 
        validation_report: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """Generate comprehensive data quality report."""
        logger.info("Generating data quality report...")
        
        report_sections = []
        
        # Header
        report_sections.append("# Data Quality Report\n")
        report_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_sections.append(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n")
        
        # Data Structure Summary
        report_sections.append("## Data Structure Summary\n")
        structure = validation_report['structure']
        
        report_sections.append(f"- **Total Rows:** {structure['rows']:,}")
        report_sections.append(f"- **Total Columns:** {structure['columns']:,}")
        report_sections.append(f"- **Memory Usage:** {structure['memory_usage'] / 1024**2:.2f} MB")
        report_sections.append(f"- **Has Date Column:** {structure['has_date_column']}")
        
        if 'date_range' in structure:
            date_range = structure['date_range']
            report_sections.append(f"- **Date Range:** {date_range['start']} to {date_range['end']}")
            report_sections.append(f"- **Frequency:** {date_range['frequency']}")
            
        report_sections.append("\n")
        
        # Column Analysis
        report_sections.append("## Column Analysis\n")
        
        columns_df = pd.DataFrame(validation_report['columns']).T
        
        # Missing values summary
        missing_summary = columns_df[['null_count', 'null_percentage']].sort_values(
            'null_percentage', ascending=False
        )
        
        report_sections.append("### Missing Values\n")
        report_sections.append(missing_summary.to_markdown())
        report_sections.append("\n")
        
        # Numeric columns summary
        numeric_cols = columns_df[columns_df['dtype'].isin(['int64', 'float64'])]
        
        if not numeric_cols.empty:
            report_sections.append("### Numeric Columns Summary\n")
            numeric_summary = numeric_cols[['min', 'max', 'mean', 'std']].round(2)
            report_sections.append(numeric_summary.to_markdown())
            report_sections.append("\n")
            
        # Issues and Warnings
        if validation_report['issues']:
            report_sections.append("## Data Quality Issues\n")
            for issue in validation_report['issues']:
                report_sections.append(f"- {issue}")
            report_sections.append("\n")
            
        # Recommendations
        report_sections.append("## Recommendations\n")
        
        high_missing = missing_summary[missing_summary['null_percentage'] > 20]
        if not high_missing.empty:
            report_sections.append("### High Missing Values")
            report_sections.append("The following columns have >20% missing values:")
            for col in high_missing.index:
                pct = high_missing.loc[col, 'null_percentage']
                report_sections.append(f"- **{col}:** {pct:.1f}% missing")
            report_sections.append("Consider imputation strategies or removal.\n")
            
        # Generate report
        report_content = "\n".join(report_sections)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"Quality report saved to {output_path}")
            
        return report_content
        
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """Save processed data in multiple formats."""
        if formats is None:
            formats = ['csv', 'parquet']
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"bd_processed_data_{timestamp}"
        
        for format_type in formats:
            if format_type == 'csv':
                file_path = output_dir / f"{base_name}.csv"
                df.to_csv(file_path, index=False)
                
            elif format_type == 'parquet':
                file_path = output_dir / f"{base_name}.parquet"
                df.to_parquet(file_path, index=False)
                
            elif format_type == 'excel':
                file_path = output_dir / f"{base_name}.xlsx"
                df.to_excel(file_path, index=False)
                
            elif format_type == 'json':
                file_path = output_dir / f"{base_name}.json"
                df.to_json(file_path, orient='records', date_format='iso')
                
            else:
                logger.warning(f"Unknown format: {format_type}")
                continue
                
            saved_files[format_type] = str(file_path)
            logger.info(f"Saved {format_type} file: {file_path}")
            
        # Save metadata
        metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'processing_steps': [
                'data_loading',
                'structure_validation', 
                'data_cleaning',
                'missing_value_handling',
                'outlier_detection',
                'feature_engineering',
                'normalization'
            ]
        }
        
        metadata_path = output_dir / f"{base_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        saved_files['metadata'] = str(metadata_path)
        
        return saved_files


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Preprocess Bangladesh economic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess_data.py --input data/bd_macro_data.csv --output data/processed/
  python scripts/preprocess_data.py --input data/raw/ --validate --report data/quality_report.md
  python scripts/preprocess_data.py --input data/bd_data.csv --missing-method knn --outlier-method isolation_forest
        """
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Input data file or directory'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/processed/',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--missing-method', 
        choices=['drop', 'forward_fill', 'backward_fill', 'interpolate', 'mean', 'median', 'knn', 'seasonal'],
        default='interpolate',
        help='Method for handling missing values'
    )
    parser.add_argument(
        '--outlier-method', 
        choices=['zscore', 'iqr', 'isolation_forest', 'modified_zscore'],
        default='isolation_forest',
        help='Method for outlier detection'
    )
    parser.add_argument(
        '--outlier-treatment', 
        choices=['remove', 'winsorize', 'cap', 'transform', 'impute'],
        default='winsorize',
        help='Method for outlier treatment'
    )
    parser.add_argument(
        '--normalize', 
        choices=['standard', 'minmax', 'robust'],
        help='Normalization method'
    )
    parser.add_argument(
        '--validate', 
        action='store_true', 
        help='Perform data validation'
    )
    parser.add_argument(
        '--report', 
        type=str, 
        help='Generate quality report at specified path'
    )
    parser.add_argument(
        '--features', 
        action='store_true', 
        help='Engineer additional features'
    )
    parser.add_argument(
        '--time-series', 
        action='store_true', 
        help='Create time series features'
    )
    parser.add_argument(
        '--formats', 
        nargs='+',
        choices=['csv', 'parquet', 'excel', 'json'],
        default=['csv', 'parquet'],
        help='Output formats'
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
        
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    try:
        # Load data
        df = preprocessor.load_data(args.input)
        
        # Validate data structure
        validation_report = preprocessor.validate_data_structure(df)
        
        if args.validate:
            logger.info("Data validation completed")
            
            if args.report:
                preprocessor.generate_quality_report(df, validation_report, args.report)
                
        # Clean data
        df_clean = preprocessor.clean_data(df)
        
        # Handle missing values
        df_filled = preprocessor.handle_missing_values(df_clean, method=args.missing_method)
        
        # Detect and treat outliers
        outlier_mask, outlier_summary = preprocessor.detect_outliers(
            df_filled, method=args.outlier_method
        )
        
        df_treated = preprocessor.treat_outliers(
            df_filled, outlier_mask, method=args.outlier_treatment
        )
        
        # Feature engineering
        if args.features:
            df_treated = preprocessor.engineer_features(df_treated)
            
        # Time series features
        if args.time_series:
            df_treated = preprocessor.create_time_series_features(df_treated)
            
        # Normalization
        if args.normalize:
            df_treated = preprocessor.normalize_data(df_treated, method=args.normalize)
            
        # Validate economic relationships
        economic_validation = preprocessor.validate_economic_relationships(df_treated)
        
        if economic_validation['warnings']:
            logger.warning("Economic relationship warnings:")
            for warning in economic_validation['warnings']:
                logger.warning(f"  - {warning}")
                
        # Save processed data
        saved_files = preprocessor.save_processed_data(
            df_treated, args.output, formats=args.formats
        )
        
        logger.info("Data preprocessing completed successfully")
        logger.info(f"Processed data shape: {df_treated.shape}")
        logger.info(f"Saved files: {list(saved_files.keys())}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()