import json
import os
from pathlib import Path
import logging
from typing import Any, Dict, Optional

class Settings:
    """Application settings manager with file persistence"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.settings = {}
        self.logger = logging.getLogger(__name__)
        
        # Load default settings
        self._load_defaults()
        
        # Load settings from file
        self.load()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Try to find config directory relative to this file
        current_dir = Path(__file__).parent
        config_file = current_dir / "config.json"
        
        # If config.json doesn't exist, create it
        if not config_file.exists():
            self._create_default_config(config_file)
        
        return str(config_file)
    
    def _create_default_config(self, config_path: Path):
        """Create default configuration file"""
        from . import DEFAULT_CONFIG, VOLATILITY_CONFIG, ML_CONFIG, CHART_CONFIG, RISK_CONFIG
        
        default_settings = {
            'application': DEFAULT_CONFIG,
            'volatility': VOLATILITY_CONFIG,
            'machine_learning': ML_CONFIG,
            'charts': CHART_CONFIG,
            'risk_management': RISK_CONFIG
        }
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_settings, f, indent=2)
            self.logger.info(f"Created default configuration file: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default config file: {e}")
    
    def _load_defaults(self):
        """Load default settings"""
        from . import DEFAULT_CONFIG, VOLATILITY_CONFIG, ML_CONFIG, CHART_CONFIG, RISK_CONFIG
        
        self.settings = {
            'application': DEFAULT_CONFIG.copy(),
            'volatility': VOLATILITY_CONFIG.copy(),
            'machine_learning': ML_CONFIG.copy(),
            'charts': CHART_CONFIG.copy(),
            'risk_management': RISK_CONFIG.copy()
        }
    
    def load(self) -> bool:
        """Load settings from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_settings = json.load(f)
                
                # Merge with defaults (file settings override defaults)
                for category, values in file_settings.items():
                    if category in self.settings:
                        self.settings[category].update(values)
                    else:
                        self.settings[category] = values
                
                self.logger.info(f"Loaded settings from {self.config_file}")
                return True
            else:
                self.logger.warning(f"Config file not found: {self.config_file}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            return False
    
    def save(self) -> bool:
        """Save settings to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2, sort_keys=True)
            
            self.logger.info(f"Settings saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            return False
    
    def get(self, key: str, category: str = 'application', default: Any = None) -> Any:
        """Get a setting value"""
        try:
            return self.settings.get(category, {}).get(key, default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any, category: str = 'application') -> None:
        """Set a setting value"""
        if category not in self.settings:
            self.settings[category] = {}
        
        self.settings[category][key] = value
        self.logger.debug(f"Setting updated: {category}.{key} = {value}")
    
    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all settings for a category"""
        return self.settings.get(category, {}).copy()
    
    def set_category(self, category: str, values: Dict[str, Any]) -> None:
        """Set multiple values for a category"""
        if category not in self.settings:
            self.settings[category] = {}
        
        self.settings[category].update(values)
        self.logger.debug(f"Category updated: {category}")
    
    def reset_category(self, category: str) -> None:
        """Reset a category to defaults"""
        from . import DEFAULT_CONFIG, VOLATILITY_CONFIG, ML_CONFIG, CHART_CONFIG, RISK_CONFIG
        
        defaults = {
            'application': DEFAULT_CONFIG,
            'volatility': VOLATILITY_CONFIG,
            'machine_learning': ML_CONFIG,
            'charts': CHART_CONFIG,
            'risk_management': RISK_CONFIG
        }
        
        if category in defaults:
            self.settings[category] = defaults[category].copy()
            self.logger.info(f"Reset category to defaults: {category}")
    
    def reset_all(self) -> None:
        """Reset all settings to defaults"""
        self._load_defaults()
        self.logger.info("All settings reset to defaults")
    
    def export_settings(self, filepath: str) -> bool:
        """Export settings to a file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.settings, f, indent=2, sort_keys=True)
            
            self.logger.info(f"Settings exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, filepath: str) -> bool:
        """Import settings from a file"""
        try:
            with open(filepath, 'r') as f:
                imported_settings = json.load(f)
            
            # Validate imported settings
            if self._validate_settings(imported_settings):
                self.settings.update(imported_settings)
                self.logger.info(f"Settings imported from {filepath}")
                return True
            else:
                self.logger.error("Invalid settings format in import file")
                return False
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            return False
    
    def _validate_settings(self, settings: Dict) -> bool:
        """Validate settings structure"""
        required_categories = ['application', 'volatility', 'machine_learning', 'charts', 'risk_management']
        
        if not isinstance(settings, dict):
            return False
        
        # Check if at least one required category exists
        return any(category in settings for category in required_categories)
    
    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration"""
        return {
            'source': self.get('data_source', default='yahoo'),
            'cache_enabled': self.get('cache_enabled', default=True),
            'cache_duration': self.get('cache_duration', default=300),
            'max_data_points': self.get('max_data_points', default=10000)
        }
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration"""
        return self.get_category('machine_learning')
    
    def get_volatility_config(self) -> Dict[str, Any]:
        """Get volatility calculation configuration"""
        return self.get_category('volatility')
    
    def get_chart_config(self) -> Dict[str, Any]:
        """Get chart configuration"""
        return self.get_category('charts')
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.get_category('risk_management')
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration"""
        return {
            'theme': self.get('gui_theme', default='dark'),
            'window_width': self.get('window_width', default=1200),
            'window_height': self.get('window_height', default=800),
            'auto_refresh': self.get('auto_refresh', default=True),
            'refresh_interval': self.get('refresh_interval', default=60)
        }
    
    def update_from_args(self, args) -> None:
        """Update settings from command line arguments"""
        if hasattr(args, 'symbol') and args.symbol:
            self.set('default_symbol', args.symbol)
        
        if hasattr(args, 'period') and args.period:
            self.set('default_period', args.period)
        
        if hasattr(args, 'model') and args.model:
            self.set('ml_model', args.model)
        
        if hasattr(args, 'theme') and args.theme:
            self.set('gui_theme', args.theme)
        
        if hasattr(args, 'config') and args.config:
            self.config_file = args.config
            self.load()
    
    def __str__(self) -> str:
        """String representation of settings"""
        return json.dumps(self.settings, indent=2, sort_keys=True)
    
    def __repr__(self) -> str:
        """Debug representation of settings"""
        return f"Settings(config_file='{self.config_file}', categories={list(self.settings.keys())})"