"""
Volatility Education & Analysis Dashboard

A comprehensive trading application for learning and analyzing market volatility.
Features include:
- Educational content for volatility beginners
- Real-time market data analysis
- Machine learning volatility predictions
- Interactive charts and visualizations
- Options analysis and implied volatility
"""

__version__ = "1.0.0"
__author__ = "Trading Dashboard Team"
__description__ = "Volatility Education & Analysis Dashboard"

# Import main components for easy access
try:
    from .gui.dashboard import Dashboard
    from .data.market_data_fetcher import MarketDataFetcher
    from .data.volatility_calculator import VolatilityCalculator
    from .ml.volatility_predictor import VolatilityPredictor
except ImportError:
    # Handle import errors gracefully during development
    pass

__all__ = [
    'Dashboard',
    'MarketDataFetcher', 
    'VolatilityCalculator',
    'VolatilityPredictor'
]