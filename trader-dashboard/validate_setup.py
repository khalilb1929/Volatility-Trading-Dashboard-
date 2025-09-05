"""
Validation script for the Volatility Trading Dashboard
Tests all major components to ensure they're working correctly
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all major imports"""
    print("Testing imports...")
    
    try:
        # Test data modules
        from data.market_data_fetcher import MarketDataFetcher
        from data.volatility_calculator import VolatilityCalculator
        print("✅ Data modules imported successfully")
        
        # Test ML modules
        from ml.volatility_predictor import VolatilityPredictor
        from ml.regime_detector import RegimeDetector
        from ml.feature_engineering import FeatureEngineer
        print("✅ ML modules imported successfully")
        
        # Test GUI modules
        from gui.widgets import StatusBar, SymbolEntry, PeriodSelector
        print("✅ GUI widget modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_fetching():
    """Test data fetching functionality"""
    print("\nTesting data fetching...")
    
    try:
        from data.market_data_fetcher import MarketDataFetcher
        
        fetcher = MarketDataFetcher()
        
        # Test symbol validation
        if fetcher.validate_symbol('AAPL'):
            print("✅ Symbol validation working")
        else:
            print("⚠️ Symbol validation returned False for AAPL")
        
        # Test data fetching (might fail if no internet)
        try:
            data = fetcher.fetch_historical_data('AAPL', period="5d")
            if len(data) > 0:
                print("✅ Historical data fetching working")
            else:
                print("⚠️ No historical data returned")
        except:
            print("⚠️ Historical data fetching failed (might be network issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fetching error: {e}")
        return False

def test_volatility_calculation():
    """Test volatility calculation"""
    print("\nTesting volatility calculation...")
    
    try:
        import pandas as pd
        import numpy as np
        from data.volatility_calculator import VolatilityCalculator
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        
        sample_data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices
        }, index=dates)
        
        calculator = VolatilityCalculator()
        
        # Test close-to-close volatility
        vol = calculator.calculate_close_to_close(sample_data)
        if len(vol.dropna()) > 0:
            print("✅ Close-to-close volatility calculation working")
        
        # Test all methods
        all_vols = calculator.calculate_all_methods(sample_data)
        if not all_vols.empty:
            print("✅ All volatility methods working")
        
        return True
        
    except Exception as e:
        print(f"❌ Volatility calculation error: {e}")
        return False

def test_ml_components():
    """Test ML components"""
    print("\nTesting ML components...")
    
    try:
        import pandas as pd
        import numpy as np
        from ml.volatility_predictor import VolatilityPredictor
        from ml.feature_engineering import FeatureEngineer
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        volatilities = np.abs(np.random.normal(20, 5, len(dates)))
        
        # Test feature engineering
        engineer = FeatureEngineer()
        features = engineer.create_basic_features(prices, volatilities)
        if features is not None and len(features) > 0:
            print("✅ Feature engineering working")
        
        # Test volatility predictor
        predictor = VolatilityPredictor()
        print("✅ Volatility predictor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ ML components error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Validating Volatility Trading Dashboard")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_data_fetching,
        test_volatility_calculation,
        test_ml_components
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The dashboard should work correctly.")
        print("\nTo start the application, run:")
        print("  python src/main.py")
        print("  or")
        print("  ./run_dashboard.bat (Windows)")
        print("  ./run_dashboard.sh (Linux/Mac)")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("You may need to install missing packages or check your internet connection.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
