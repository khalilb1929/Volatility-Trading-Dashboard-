"""
GUI module for the Volatility Education & Analysis Dashboard

This module contains all the graphical user interface components:
- Main dashboard with tabbed interface
- Volatility education tab with interactive examples
- Custom widgets for data entry and display
- Charts and visualizations
"""

from .dashboard import Dashboard
from .volatility_education import VolatilityEducationTab
from .widgets import (
    StatusBar, 
    SymbolEntry, 
    PeriodSelector, 
    ModelSelector,
    AlertPanel,
    MetricsDisplay
)

__all__ = [
    'Dashboard',
    'VolatilityEducationTab',
    'StatusBar',
    'SymbolEntry', 
    'PeriodSelector',
    'ModelSelector',
    'AlertPanel',
    'MetricsDisplay'
]

__version__ = "1.0.0"