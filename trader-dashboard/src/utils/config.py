import json
import os

class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            # Look for config file in project root
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), self.config_file)
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Return default config if file doesn't exist
                return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "alpha_vantage": {
                "api_key": "demo",
                "base_url": "https://www.alphavantage.co/query"
            },
            "default_settings": {
                "default_symbol": "AAPL",
                "default_period": "90 days",
                "refresh_interval": 300
            },
            "ml_settings": {
                "default_model": "random_forest",
                "prediction_days": 5,
                "feature_window": 20
            }
        }
    
    def get_api_key(self):
        """Get Alpha Vantage API key"""
        return self.config.get("alpha_vantage", {}).get("api_key", "demo")
    
    def get_default_symbol(self):
        """Get default stock symbol"""
        return self.config.get("default_settings", {}).get("default_symbol", "AAPL")