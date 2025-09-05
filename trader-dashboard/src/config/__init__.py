"""
Configuration module for the Trader Dashboard application.

This module provides configuration management, settings handling,
and application preferences.
"""

from .settings import Settings

__all__ = ['Settings']

# Default configuration values
DEFAULT_CONFIG = {
    'data_source': 'yahoo',
    'default_symbol': 'AAPL',
    'default_period': '90 days',
    'ml_model': 'random_forest',
    'chart_theme': 'dark',
    'auto_refresh': True,
    'refresh_interval': 60,
    'cache_enabled': True,
    'cache_duration': 300,  # 5 minutes
    'max_data_points': 10000,
    'gui_theme': 'dark',
    'window_width': 1200,
    'window_height': 800,
    'log_level': 'INFO'
}

# Volatility calculation settings
VOLATILITY_CONFIG = {
    'method': 'close_to_close',
    'window': 20,
    'annualization_factor': 252,
    'min_periods': 10
}

# Machine learning settings
ML_CONFIG = {
    'train_test_split': 0.8,
    'cross_validation_folds': 5,
    'feature_selection': True,
    'model_validation': True,
    'hyperparameter_tuning': False
}

# Chart settings
CHART_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'dark_background',
    'grid': True,
    'legend': True,
    'toolbar': True
}

# Risk management settings
RISK_CONFIG = {
    'max_volatility_threshold': 100,
    'alert_on_regime_change': True,
    'position_size_limit': 1.0,
    'max_drawdown_alert': 0.2
}