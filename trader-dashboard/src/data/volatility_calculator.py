"""
Volatility Calculator Module

Implements various volatility calculation methods including:
- Close-to-Close volatility
- Garman-Klass volatility  
- Rogers-Satchell volatility
- Yang-Zhang volatility
- Parkinson volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta


class VolatilityCalculator:
    """
    Comprehensive volatility calculator with multiple estimation methods
    """
    
    def __init__(self, annualization_factor: int = 252):
        """
        Initialize the volatility calculator
        
        Args:
            annualization_factor: Number of trading days per year (default 252)
        """
        self.annualization_factor = annualization_factor
        self.methods = {
            'close_to_close': self.calculate_close_to_close,
            'garman_klass': self.calculate_garman_klass,
            'rogers_satchell': self.calculate_rogers_satchell,
            'yang_zhang': self.calculate_yang_zhang,
            'parkinson': self.calculate_parkinson
        }
    
    def calculate_close_to_close(self, data: pd.DataFrame, window: int = 20, 
                               annualized: bool = True) -> pd.Series:
        """
        Calculate close-to-close volatility (standard method)
        
        Args:
            data: DataFrame with 'Close' column
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # Calculate log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=window).std()
        
        if annualized:
            volatility = volatility * np.sqrt(self.annualization_factor)
        
        return volatility * 100  # Convert to percentage
    
    def calculate_garman_klass(self, data: pd.DataFrame, window: int = 20,
                             annualized: bool = True) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator
        
        Args:
            data: DataFrame with OHLC columns
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Garman-Klass formula
        hl_ratio = np.log(data['High'] / data['Low'])
        co_ratio = np.log(data['Close'] / data['Open'])
        
        gk_variance = 0.5 * hl_ratio**2 - (2*np.log(2) - 1) * co_ratio**2
        
        # Calculate rolling average
        volatility = np.sqrt(gk_variance.rolling(window=window).mean())
        
        if annualized:
            volatility = volatility * np.sqrt(self.annualization_factor)
        
        return volatility * 100  # Convert to percentage
    
    def calculate_rogers_satchell(self, data: pd.DataFrame, window: int = 20,
                                annualized: bool = True) -> pd.Series:
        """
        Calculate Rogers-Satchell volatility estimator
        
        Args:
            data: DataFrame with OHLC columns
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Rogers-Satchell formula
        ho_ratio = np.log(data['High'] / data['Open'])
        hc_ratio = np.log(data['High'] / data['Close'])
        lo_ratio = np.log(data['Low'] / data['Open'])
        lc_ratio = np.log(data['Low'] / data['Close'])
        
        rs_variance = ho_ratio * hc_ratio + lo_ratio * lc_ratio
        
        # Calculate rolling average
        volatility = np.sqrt(rs_variance.rolling(window=window).mean())
        
        if annualized:
            volatility = volatility * np.sqrt(self.annualization_factor)
        
        return volatility * 100  # Convert to percentage
    
    def calculate_yang_zhang(self, data: pd.DataFrame, window: int = 20,
                           annualized: bool = True) -> pd.Series:
        """
        Calculate Yang-Zhang volatility estimator
        
        Args:
            data: DataFrame with OHLC columns
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Calculate overnight returns
        overnight_returns = np.log(data['Open'] / data['Close'].shift(1))
        
        # Calculate open-to-close returns
        oc_returns = np.log(data['Close'] / data['Open'])
        
        # Calculate Rogers-Satchell component
        rs_component = self.calculate_rogers_satchell(data, window=1, annualized=False) / 100
        
        # Yang-Zhang formula components
        overnight_variance = overnight_returns.rolling(window=window).var()
        oc_variance = oc_returns.rolling(window=window).var()
        rs_variance = (rs_component ** 2).rolling(window=window).mean()
        
        # Scaling factor k
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Yang-Zhang volatility
        yz_variance = overnight_variance + k * oc_variance + (1 - k) * rs_variance
        volatility = np.sqrt(yz_variance)
        
        if annualized:
            volatility = volatility * np.sqrt(self.annualization_factor)
        
        return volatility * 100  # Convert to percentage
    
    def calculate_parkinson(self, data: pd.DataFrame, window: int = 20,
                          annualized: bool = True) -> pd.Series:
        """
        Calculate Parkinson volatility estimator
        
        Args:
            data: DataFrame with High and Low columns
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        required_cols = ['High', 'Low']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Parkinson formula
        hl_ratio = np.log(data['High'] / data['Low'])
        parkinson_variance = hl_ratio**2 / (4 * np.log(2))
        
        # Calculate rolling average
        volatility = np.sqrt(parkinson_variance.rolling(window=window).mean())
        
        if annualized:
            volatility = volatility * np.sqrt(self.annualization_factor)
        
        return volatility * 100  # Convert to percentage
    
    def calculate_all_methods(self, data: pd.DataFrame, window: int = 20,
                            annualized: bool = True) -> pd.DataFrame:
        """
        Calculate volatility using all available methods
        
        Args:
            data: DataFrame with OHLC columns
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            DataFrame with volatility from all methods
        """
        results = pd.DataFrame(index=data.index)
        
        # Close-to-Close (always available if Close exists)
        if 'Close' in data.columns:
            try:
                results['Close_to_Close'] = self.calculate_close_to_close(
                    data, window, annualized)
            except Exception as e:
                warnings.warn(f"Could not calculate Close-to-Close volatility: {e}")
        
        # OHLC-based methods (require all OHLC data)
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in data.columns for col in ohlc_cols):
            try:
                results['Garman_Klass'] = self.calculate_garman_klass(
                    data, window, annualized)
            except Exception as e:
                warnings.warn(f"Could not calculate Garman-Klass volatility: {e}")
            
            try:
                results['Rogers_Satchell'] = self.calculate_rogers_satchell(
                    data, window, annualized)
            except Exception as e:
                warnings.warn(f"Could not calculate Rogers-Satchell volatility: {e}")
            
            try:
                results['Yang_Zhang'] = self.calculate_yang_zhang(
                    data, window, annualized)
            except Exception as e:
                warnings.warn(f"Could not calculate Yang-Zhang volatility: {e}")
        
        # Parkinson (requires High and Low)
        if 'High' in data.columns and 'Low' in data.columns:
            try:
                results['Parkinson'] = self.calculate_parkinson(
                    data, window, annualized)
            except Exception as e:
                warnings.warn(f"Could not calculate Parkinson volatility: {e}")
        
        return results
    
    def calculate_realized_volatility(self, data: pd.DataFrame, 
                                    method: str = 'close_to_close',
                                    window: int = 20) -> Dict:
        """
        Calculate realized volatility statistics
        
        Args:
            data: DataFrame with price data
            method: Volatility calculation method
            window: Rolling window size
            
        Returns:
            Dictionary with volatility statistics
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        # Calculate volatility
        volatility = self.methods[method](data, window=window, annualized=True)
        volatility = volatility.dropna()
        
        if len(volatility) == 0:
            return {'error': 'No valid volatility data calculated'}
        
        # Calculate statistics
        stats = {
            'method': method,
            'window': window,
            'current_volatility': float(volatility.iloc[-1]) if len(volatility) > 0 else None,
            'mean_volatility': float(volatility.mean()),
            'std_volatility': float(volatility.std()),
            'min_volatility': float(volatility.min()),
            'max_volatility': float(volatility.max()),
            'median_volatility': float(volatility.median()),
            'percentile_25': float(volatility.quantile(0.25)),
            'percentile_75': float(volatility.quantile(0.75)),
            'percentile_95': float(volatility.quantile(0.95)),
            'percentile_5': float(volatility.quantile(0.05)),
            'observations': len(volatility),
            'start_date': str(volatility.index[0]),
            'end_date': str(volatility.index[-1])
        }
        
        # Add volatility regime classification
        current_vol = stats['current_volatility']
        if current_vol:
            if current_vol < 15:
                regime = 'Low'
            elif current_vol < 25:
                regime = 'Medium'
            elif current_vol < 40:
                regime = 'High'
            else:
                regime = 'Extreme'
            
            stats['volatility_regime'] = regime
        
        return stats
    
    def compare_methods(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Compare different volatility calculation methods
        
        Args:
            data: DataFrame with OHLC data
            window: Rolling window size
            
        Returns:
            DataFrame comparing different methods
        """
        all_vols = self.calculate_all_methods(data, window=window, annualized=True)
        
        if all_vols.empty:
            return pd.DataFrame()
        
        # Calculate comparison statistics
        comparison = pd.DataFrame()
        
        for method in all_vols.columns:
            vol_series = all_vols[method].dropna()
            if len(vol_series) > 0:
                comparison[method] = [
                    vol_series.mean(),
                    vol_series.std(),
                    vol_series.min(),
                    vol_series.max(),
                    vol_series.iloc[-1] if len(vol_series) > 0 else np.nan
                ]
        
        comparison.index = ['Mean', 'Std', 'Min', 'Max', 'Current']
        
        return comparison.round(2)
    
    def calculate_volatility_percentile(self, data: pd.DataFrame, 
                                      method: str = 'close_to_close',
                                      window: int = 20,
                                      lookback_period: int = 252) -> Dict:
        """
        Calculate where current volatility stands relative to historical distribution
        
        Args:
            data: DataFrame with price data
            method: Volatility calculation method
            window: Rolling window for volatility calculation
            lookback_period: Number of periods to look back for percentile calculation
            
        Returns:
            Dictionary with percentile information
        """
        # Calculate volatility
        volatility = self.methods[method](data, window=window, annualized=True)
        volatility = volatility.dropna()
        
        if len(volatility) < lookback_period:
            lookback_period = len(volatility)
        
        if lookback_period == 0:
            return {'error': 'No volatility data available'}
        
        # Get recent volatility data
        recent_vol = volatility.tail(lookback_period)
        current_vol = volatility.iloc[-1]
        
        # Calculate percentile
        percentile = (recent_vol < current_vol).mean() * 100
        
        return {
            'current_volatility': float(current_vol),
            'percentile': float(percentile),
            'interpretation': self._interpret_percentile(percentile),
            'lookback_period': lookback_period,
            'historical_mean': float(recent_vol.mean()),
            'historical_std': float(recent_vol.std()),
            'z_score': float((current_vol - recent_vol.mean()) / recent_vol.std()) if recent_vol.std() > 0 else 0
        }
    
    def _interpret_percentile(self, percentile: float) -> str:
        """Interpret volatility percentile"""
        if percentile < 10:
            return "Extremely Low"
        elif percentile < 25:
            return "Low"
        elif percentile < 75:
            return "Normal"
        elif percentile < 90:
            return "High"
        else:
            return "Extremely High"
    
    def calculate_volatility_forecast(self, data: pd.DataFrame,
                                    method: str = 'close_to_close',
                                    window: int = 20,
                                    forecast_days: int = 5) -> Dict:
        """
        Simple volatility forecast using mean reversion
        
        Args:
            data: DataFrame with price data
            method: Volatility calculation method
            window: Rolling window size
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast information
        """
        # Calculate volatility
        volatility = self.methods[method](data, window=window, annualized=True)
        volatility = volatility.dropna()
        
        if len(volatility) < 30:
            return {'error': 'Insufficient data for forecasting'}
        
        current_vol = volatility.iloc[-1]
        long_term_mean = volatility.mean()
        volatility_of_volatility = volatility.pct_change().std()
        
        # Simple mean reversion model
        mean_reversion_speed = 0.1  # Assumption: 10% daily mean reversion
        
        forecasts = []
        forecast_vol = current_vol
        
        for day in range(1, forecast_days + 1):
            # Mean reversion formula: next_vol = current_vol + speed * (mean - current_vol)
            forecast_vol = forecast_vol + mean_reversion_speed * (long_term_mean - forecast_vol)
            forecasts.append(forecast_vol)
        
        return {
            'current_volatility': float(current_vol),
            'long_term_mean': float(long_term_mean),
            'forecast_days': forecast_days,
            'forecasts': [float(f) for f in forecasts],
            'mean_reversion_speed': mean_reversion_speed,
            'volatility_of_volatility': float(volatility_of_volatility * 100),
            'model': 'Simple Mean Reversion'
        }
    
    def get_volatility_summary(self, data: pd.DataFrame,
                             method: str = 'close_to_close',
                             window: int = 20) -> str:
        """
        Get a comprehensive volatility summary as formatted string
        
        Args:
            data: DataFrame with price data
            method: Volatility calculation method
            window: Rolling window size
            
        Returns:
            Formatted string with volatility summary
        """
        stats = self.calculate_realized_volatility(data, method, window)
        percentile_info = self.calculate_volatility_percentile(data, method, window)
        
        if 'error' in stats:
            return f"Error calculating volatility: {stats['error']}"
        
        summary = f"""
VOLATILITY ANALYSIS SUMMARY
{'='*50}

Method: {stats['method'].replace('_', ' ').title()}
Window: {stats['window']} periods
Data Range: {stats['start_date']} to {stats['end_date']}
Observations: {stats['observations']}

CURRENT STATUS:
Current Volatility: {stats['current_volatility']:.2f}%
Volatility Regime: {stats.get('volatility_regime', 'Unknown')}
Historical Percentile: {percentile_info.get('percentile', 0):.1f}% ({percentile_info.get('interpretation', 'Unknown')})

HISTORICAL STATISTICS:
Mean Volatility: {stats['mean_volatility']:.2f}%
Std Dev of Volatility: {stats['std_volatility']:.2f}%
Minimum: {stats['min_volatility']:.2f}%
Maximum: {stats['max_volatility']:.2f}%
Median: {stats['median_volatility']:.2f}%

PERCENTILE RANGES:
5th Percentile: {stats['percentile_5']:.2f}%
25th Percentile: {stats['percentile_25']:.2f}%
75th Percentile: {stats['percentile_75']:.2f}%
95th Percentile: {stats['percentile_95']:.2f}%
"""
        
        return summary


# Helper functions for common use cases
def calculate_simple_volatility(prices: Union[pd.Series, List[float]], 
                               window: int = 20,
                               annualized: bool = True) -> pd.Series:
    """
    Simple convenience function to calculate volatility from price series
    
    Args:
        prices: Price series or list
        window: Rolling window size
        annualized: Whether to annualize
        
    Returns:
        Volatility series
    """
    if isinstance(prices, list):
        prices = pd.Series(prices)
    
    data = pd.DataFrame({'Close': prices})
    calculator = VolatilityCalculator()
    return calculator.calculate_close_to_close(data, window, annualized)


def get_current_volatility(data: pd.DataFrame, method: str = 'close_to_close') -> float:
    """
    Get the most recent volatility value
    
    Args:
        data: DataFrame with price data
        method: Calculation method
        
    Returns:
        Current volatility value
    """
    calculator = VolatilityCalculator()
    stats = calculator.calculate_realized_volatility(data, method)
    return stats.get('current_volatility', 0.0)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    n_periods = len(dates)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * np.cumprod(1 + returns)
    
    # Create OHLC data
    sample_data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'Close': prices
    }, index=dates)
    
    # Test the calculator
    calculator = VolatilityCalculator()
    
    print("Testing Volatility Calculator")
    print("=" * 40)
    
    # Test individual methods
    print("\n1. Testing individual methods:")
    cc_vol = calculator.calculate_close_to_close(sample_data)
    print(f"Close-to-Close volatility (latest): {cc_vol.iloc[-1]:.2f}%")
    
    gk_vol = calculator.calculate_garman_klass(sample_data)
    print(f"Garman-Klass volatility (latest): {gk_vol.iloc[-1]:.2f}%")
    
    # Test all methods comparison
    print("\n2. Testing method comparison:")
    comparison = calculator.compare_methods(sample_data)
    print(comparison)
    
    # Test volatility statistics
    print("\n3. Testing volatility statistics:")
    stats = calculator.calculate_realized_volatility(sample_data)
    print(f"Current volatility: {stats['current_volatility']:.2f}%")
    print(f"Volatility regime: {stats['volatility_regime']}")
    
    # Test percentile calculation
    print("\n4. Testing percentile calculation:")
    percentile_info = calculator.calculate_volatility_percentile(sample_data)
    print(f"Current volatility percentile: {percentile_info['percentile']:.1f}%")
    print(f"Interpretation: {percentile_info['interpretation']}")
    
    print("\nTesting completed successfully!")
