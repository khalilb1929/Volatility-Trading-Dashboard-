"""
Market Data Fetcher Module

Handles fetching real-time and historical market data from various sources.
Primary source is Yahoo Finance through yfinance library.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional, Tuple, Union


class MarketDataFetcher:
    """
    Fetches market data from various sources including Yahoo Finance
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize the market data fetcher
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
    
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """
        Fetch real-time market data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            Dictionary containing real-time data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current market data
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            current_price = hist['Close'].iloc[-1]
            
            return {
                'symbol': symbol.upper(),
                'current_price': round(float(current_price), 2),
                'open_price': round(float(hist['Open'].iloc[0]), 2),
                'high_price': round(float(hist['High'].max()), 2),
                'low_price': round(float(hist['Low'].min()), 2),
                'volume': int(hist['Volume'].sum()),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'timestamp': datetime.now(),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'NASDAQ')
            }
            
        except Exception as e:
            raise Exception(f"Error fetching real-time data for {symbol}: {str(e)}")
    
    def fetch_historical_data(self, symbol: str, period: str = "1y", 
                            interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical market data
        
        Args:
            symbol: Stock symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self._cache:
                cached_data, cache_time = self._cache[cache_key]
                if time.time() - cache_time < self._cache_timeout:
                    return cached_data.copy()
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Clean and process data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Add calculated fields
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Price_Change'] = data['Close'] - data['Open']
            data['Price_Change_Pct'] = (data['Close'] - data['Open']) / data['Open'] * 100
            data['Range'] = data['High'] - data['Low']
            data['Range_Pct'] = data['Range'] / data['Open'] * 100
            
            # Cache the data
            self._cache[cache_key] = (data.copy(), time.time())
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching historical data for {symbol}: {str(e)}")
    
    def fetch_multiple_symbols(self, symbols: List[str], period: str = "1y", 
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                data = self.fetch_historical_data(symbol, period, interval)
                results[symbol] = data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                failed_symbols.append((symbol, str(e)))
                continue
        
        if failed_symbols:
            print(f"Failed to fetch data for: {[s[0] for s in failed_symbols]}")
        
        return results
    
    def fetch_intraday_data(self, symbol: str, days: int = 1) -> pd.DataFrame:
        """
        Fetch intraday data (1-minute intervals)
        
        Args:
            symbol: Stock symbol
            days: Number of days of intraday data
            
        Returns:
            DataFrame with 1-minute intraday data
        """
        try:
            period = f"{days}d"
            return self.fetch_historical_data(symbol, period=period, interval="1m")
        except Exception as e:
            raise Exception(f"Error fetching intraday data for {symbol}: {str(e)}")
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol.upper(),
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'average_volume': info.get('averageVolume', 0),
                'business_summary': info.get('businessSummary', ''),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0)
            }
            
        except Exception as e:
            raise Exception(f"Error fetching symbol info for {symbol}: {str(e)}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get recent data
            data = ticker.history(period="5d")
            return not data.empty
        except:
            return False
    
    def search_symbols(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for symbols by company name or symbol
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching symbols with basic info
        """
        # This is a simplified implementation
        # In production, you might want to use a proper search API
        results = []
        
        # Common symbols for demo purposes
        common_symbols = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'PG': 'Procter & Gamble Company',
            'V': 'Visa Inc.',
            'HD': 'Home Depot Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'BAC': 'Bank of America Corporation',
            'MA': 'Mastercard Inc.',
            'DIS': 'Walt Disney Company',
            'ADBE': 'Adobe Inc.',
            'NFLX': 'Netflix Inc.',
            'CRM': 'Salesforce Inc.',
            'XOM': 'Exxon Mobil Corporation'
        }
        
        query_lower = query.lower()
        
        for symbol, name in common_symbols.items():
            if (query_lower in symbol.lower() or 
                query_lower in name.lower()):
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'match_type': 'symbol' if query_lower in symbol.lower() else 'name'
                })
                
                if len(results) >= max_results:
                    break
        
        return results
    
    def get_market_status(self) -> Dict:
        """
        Get current market status
        
        Returns:
            Dictionary with market status information
        """
        try:
            # Use SPY as a proxy for market status
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1d", interval="1m")
            
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Simple market hours check (US Eastern time assumption)
            is_open = (market_open <= now <= market_close and 
                      now.weekday() < 5)  # Monday = 0, Friday = 4
            
            last_update = hist.index[-1] if not hist.empty else None
            
            return {
                'is_open': is_open,
                'market_open': market_open,
                'market_close': market_close,
                'last_update': last_update,
                'timezone': 'US/Eastern',
                'current_time': now
            }
            
        except Exception as e:
            return {
                'is_open': False,
                'error': str(e),
                'current_time': datetime.now()
            }
    
    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache"""
        return {
            'cached_items': len(self._cache),
            'cache_timeout': self._cache_timeout,
            'cache_keys': list(self._cache.keys())
        }
    
    def fetch_options_data(self, symbol: str) -> Dict:
        """
        Fetch options data for a symbol (placeholder implementation)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with options data
        """
        try:
            # This is a placeholder implementation
            # In a real application, you would use an options data provider
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice', 100)
            
            # Generate mock options data
            strikes = []
            calls = []
            puts = []
            
            # Create strikes around current price
            for i in range(-5, 6):
                strike = round(current_price + (i * 5), 2)
                strikes.append(strike)
                
                # Mock call data
                calls.append({
                    'strike': strike,
                    'bid': max(0, current_price - strike + np.random.uniform(-2, 2)),
                    'ask': max(0, current_price - strike + np.random.uniform(0, 4)),
                    'volume': int(np.random.uniform(0, 1000)),
                    'openInterest': int(np.random.uniform(0, 5000)),
                    'impliedVolatility': np.random.uniform(0.15, 0.45)
                })
                
                # Mock put data
                puts.append({
                    'strike': strike,
                    'bid': max(0, strike - current_price + np.random.uniform(-2, 2)),
                    'ask': max(0, strike - current_price + np.random.uniform(0, 4)),
                    'volume': int(np.random.uniform(0, 1000)),
                    'openInterest': int(np.random.uniform(0, 5000)),
                    'impliedVolatility': np.random.uniform(0.15, 0.45)
                })
            
            return {
                'symbol': symbol.upper(),
                'current_price': current_price,
                'calls': calls,
                'puts': puts,
                'expiration_date': '2024-01-19',  # Mock expiration
                'timestamp': datetime.now(),
                'note': 'This is mock options data for demonstration purposes'
            }
            
        except Exception as e:
            return {
                'error': f"Error fetching options data: {str(e)}",
                'symbol': symbol.upper(),
                'calls': [],
                'puts': []
            }


# Helper functions for common use cases
def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Convenience function to get stock data
    
    Args:
        symbol: Stock symbol
        period: Data period
        
    Returns:
        DataFrame with stock data
    """
    fetcher = MarketDataFetcher()
    return fetcher.fetch_historical_data(symbol, period)


def get_current_price(symbol: str) -> float:
    """
    Convenience function to get current stock price
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Current stock price
    """
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_realtime_data(symbol)
    return data['current_price']


def validate_stock_symbol(symbol: str) -> bool:
    """
    Convenience function to validate stock symbol
    
    Args:
        symbol: Stock symbol
        
    Returns:
        True if valid, False otherwise
    """
    fetcher = MarketDataFetcher()
    return fetcher.validate_symbol(symbol)


# Example usage and testing
if __name__ == "__main__":
    # Test the market data fetcher
    fetcher = MarketDataFetcher()
    
    # Test symbol validation
    print("Testing symbol validation:")
    print(f"AAPL is valid: {fetcher.validate_symbol('AAPL')}")
    print(f"INVALID is valid: {fetcher.validate_symbol('INVALID')}")
    
    # Test real-time data
    print("\nTesting real-time data:")
    try:
        real_time = fetcher.fetch_realtime_data('AAPL')
        print(f"AAPL current price: ${real_time['current_price']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test historical data
    print("\nTesting historical data:")
    try:
        hist_data = fetcher.fetch_historical_data('AAPL', period="1mo")
        print(f"Retrieved {len(hist_data)} days of data")
        print(f"Date range: {hist_data.index[0]} to {hist_data.index[-1]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test market status
    print("\nTesting market status:")
    status = fetcher.get_market_status()
    print(f"Market is {'open' if status['is_open'] else 'closed'}")
