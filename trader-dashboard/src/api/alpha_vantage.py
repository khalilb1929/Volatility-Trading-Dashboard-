import requests
import json
import numpy as np
from datetime import datetime, timedelta

class AlphaVantageAPI:
    def __init__(self, api_key="demo"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_daily_prices(self, symbol, period_days=30):
        """Fetch daily stock prices from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full' if period_days > 100 else 'compact'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise Exception("API call frequency limit reached")
            
            time_series = data.get('Time Series (Daily)', {})
            
            # Convert to our format
            dates = []
            prices = []
            
            for date_str, price_data in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                if date >= datetime.now() - timedelta(days=period_days):
                    dates.append(date)
                    prices.append(float(price_data['4. close']))
            
            # Sort by date
            sorted_data = sorted(zip(dates, prices))
            dates, prices = zip(*sorted_data) if sorted_data else ([], [])
            
            return list(dates), list(prices)
            
        except requests.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            raise Exception(f"Data processing error: {str(e)}")
    
    def calculate_implied_volatility(self, prices, window=20):
        """Calculate implied volatility from price data"""
        if len(prices) < window:
            raise Exception(f"Insufficient data points. Need at least {window} points.")
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate rolling volatility
        volatilities = []
        for i in range(window-1, len(returns)):
            window_returns = returns[i-window+1:i+1]
            volatility = np.std(window_returns) * np.sqrt(252) * 100  # Annualized %
            volatilities.append(volatility)
        
        return volatilities