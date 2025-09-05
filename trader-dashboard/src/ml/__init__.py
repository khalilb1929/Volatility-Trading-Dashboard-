"""
Machine Learning module for volatility prediction and analysis

This module contains:
- Volatility prediction models using various ML algorithms
- Market regime detection using unsupervised learning
- Feature engineering for time series data
- Model evaluation and backtesting utilities
"""

from .volatility_predictor import VolatilityPredictor
from .regime_detector import RegimeDetector
from .feature_engineering import FeatureEngineer

__all__ = [
    'VolatilityPredictor',
    'RegimeDetector', 
    'FeatureEngineer'
]

__version__ = "1.0.0"

# ML model configurations
DEFAULT_MODELS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'gradient_boost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    },
    'neural_network': {
        'hidden_layer_sizes': (50, 25),
        'max_iter': 500,
        'random_state': 42
    },
    'linear_regression': {
        'fit_intercept': True,
        'normalize': False
    }
}

# Feature engineering parameters
FEATURE_PARAMS = {
    'lookback_periods': [5, 10, 20, 50],
    'volatility_windows': [5, 10, 20],
    'return_periods': [1, 5, 10],
    'technical_indicators': True,
    'regime_features': True
}

# Model validation parameters
VALIDATION_PARAMS = {
    'test_size': 0.2,
    'cv_folds': 5,
    'scoring': 'neg_mean_squared_error',
    'random_state': 42
}