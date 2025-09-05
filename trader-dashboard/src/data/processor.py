import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    @staticmethod
    def calculate_statistics(data):
        """Calculate statistical measures for volatility data"""
        if not data:
            return {}
        
        data_array = np.array(data)
        return {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'percentile_25': np.percentile(data_array, 25),
            'percentile_75': np.percentile(data_array, 75),
            'percentile_90': np.percentile(data_array, 90),
            'percentile_95': np.percentile(data_array, 95)
        }
    
    @staticmethod
    def calculate_correlation(data1, data2):
        """Calculate correlation between two datasets"""
        if len(data1) != len(data2) or len(data1) < 2:
            return 0.0
        
        return np.corrcoef(data1, data2)[0, 1]
    
    @staticmethod
    def calculate_spread(data1, data2):
        """Calculate average spread between two datasets"""
        if len(data1) != len(data2):
            return 0.0
        
        spread = np.array(data2) - np.array(data1)
        return np.mean(spread)
    
    @staticmethod
    def calculate_volatility_from_prices(prices, window=20):
        """Calculate historical volatility from price data"""
        if len(prices) < window + 1:
            return []
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Calculate rolling volatility
        volatilities = []
        for i in range(window-1, len(log_returns)):
            window_returns = log_returns[i-window+1:i+1]
            # Annualized volatility (assuming 252 trading days per year)
            vol = np.std(window_returns) * np.sqrt(252) * 100
            volatilities.append(vol)
        
        return volatilities