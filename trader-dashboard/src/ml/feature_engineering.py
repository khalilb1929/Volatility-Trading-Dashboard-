import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using fallback implementations for technical indicators.")

class FeatureEngineer:
    """Advanced feature engineering for financial time series data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        self.feature_types = {}
        self.is_fitted = False
        
        # Configuration for different feature categories
        self.config = {
            'price_features': True,
            'volatility_features': True,
            'return_features': True,
            'technical_indicators': True,
            'statistical_features': True,
            'regime_features': True,
            'time_features': True,
            'interaction_features': False  # More advanced, can be computationally expensive
        }
        
        # Lookback periods for different feature types
        self.lookback_periods = {
            'short': [3, 5, 7],
            'medium': [10, 15, 20],
            'long': [30, 50, 100]
        }
    
    def set_config(self, **kwargs):
        """Update feature engineering configuration"""
        self.config.update(kwargs)
    
    def create_basic_features(self, prices, volatilities, lookback_periods=None):
        """Create basic features for volatility prediction - simplified version"""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20]
        
        try:
            if isinstance(prices, (list, tuple)):
                prices = np.array(prices)
            if isinstance(volatilities, (list, tuple)):
                volatilities = np.array(volatilities)
            
            if len(prices) < max(lookback_periods) + 1:
                return None
            
            # Create DataFrame for easier manipulation
            df = pd.DataFrame({
                'price': prices,
                'volatility': volatilities
            })
            
            # Calculate returns
            df['returns'] = df['price'].pct_change()
            
            # Basic statistical features for the most recent data point
            features = []
            
            # Price-based features
            for period in lookback_periods:
                if len(df) >= period:
                    recent_prices = df['price'].iloc[-period:]
                    features.extend([
                        recent_prices.mean(),
                        recent_prices.std(),
                        (df['price'].iloc[-1] - recent_prices.mean()) / recent_prices.std() if recent_prices.std() > 0 else 0
                    ])
            
            # Volatility-based features
            for period in lookback_periods:
                if len(df) >= period:
                    recent_vols = df['volatility'].iloc[-period:]
                    features.extend([
                        recent_vols.mean(),
                        recent_vols.std(),
                        recent_vols.min(),
                        recent_vols.max()
                    ])
            
            # Return-based features
            returns = df['returns'].dropna()
            if len(returns) >= 5:
                features.extend([
                    returns.mean(),
                    returns.std(),
                    returns.skew() if len(returns) > 2 else 0,
                    returns.kurtosis() if len(returns) > 3 else 0
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            return np.array(features) if features else None
            
        except Exception as e:
            print(f"Warning: Error in create_basic_features: {e}")
            return None
    
    def create_all_features(self, prices, volumes=None, returns=None, volatilities=None, dates=None):
        """Create comprehensive feature set"""
        if len(prices) < 100:
            raise ValueError("Need at least 100 data points for feature engineering")
        
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame({'price': prices})
        
        if volumes is not None:
            df['volume'] = volumes
        if returns is not None:
            df['returns'] = returns
        else:
            df['returns'] = df['price'].pct_change()
        
        if volatilities is not None:
            df['volatility'] = volatilities
        else:
            # Calculate simple rolling volatility
            df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252) * 100
        
        if dates is not None:
            df['date'] = pd.to_datetime(dates)
        else:
            df['date'] = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
        
        # Initialize feature storage
        all_features = []
        feature_names = []
        
        # Determine the minimum required lookback
        max_lookback = max([max(periods) for periods in self.lookback_periods.values()])
        start_idx = max_lookback
        
        # Generate features for each time point
        for i in range(start_idx, len(df)):
            feature_vector = []
            current_feature_names = []
            
            # 1. Price Features
            if self.config['price_features']:
                price_features, price_names = self._create_price_features(df, i)
                feature_vector.extend(price_features)
                current_feature_names.extend(price_names)
            
            # 2. Volatility Features
            if self.config['volatility_features']:
                vol_features, vol_names = self._create_volatility_features(df, i)
                feature_vector.extend(vol_features)
                current_feature_names.extend(vol_names)
            
            # 3. Return Features
            if self.config['return_features']:
                ret_features, ret_names = self._create_return_features(df, i)
                feature_vector.extend(ret_features)
                current_feature_names.extend(ret_names)
            
            # 4. Technical Indicators
            if self.config['technical_indicators']:
                tech_features, tech_names = self._create_technical_features(df, i)
                feature_vector.extend(tech_features)
                current_feature_names.extend(tech_names)
            
            # 5. Statistical Features
            if self.config['statistical_features']:
                stat_features, stat_names = self._create_statistical_features(df, i)
                feature_vector.extend(stat_features)
                current_feature_names.extend(stat_names)
            
            # 6. Regime Features
            if self.config['regime_features']:
                regime_features, regime_names = self._create_regime_features(df, i)
                feature_vector.extend(regime_features)
                current_feature_names.extend(regime_names)
            
            # 7. Time Features
            if self.config['time_features']:
                time_features, time_names = self._create_time_features(df, i)
                feature_vector.extend(time_features)
                current_feature_names.extend(time_names)
            
            # 8. Interaction Features (optional, computationally expensive)
            if self.config['interaction_features']:
                interact_features, interact_names = self._create_interaction_features(df, i)
                feature_vector.extend(interact_features)
                current_feature_names.extend(interact_names)
            
            all_features.append(feature_vector)
            
            # Store feature names (only for first iteration)
            if i == start_idx:
                feature_names = current_feature_names
        
        self.feature_names = feature_names
        return np.array(all_features), feature_names
    
    def _create_price_features(self, df, i):
        """Create price-based features"""
        features = []
        names = []
        
        current_price = df['price'].iloc[i]
        
        # Price levels and ratios
        for period in self.lookback_periods['short'] + self.lookback_periods['medium']:
            if i >= period:
                past_price = df['price'].iloc[i-period]
                price_ratio = current_price / past_price
                price_change = (current_price - past_price) / past_price
                
                features.extend([price_ratio, price_change])
                names.extend([f'price_ratio_{period}d', f'price_change_{period}d'])
            else:
                features.extend([1.0, 0.0])
                names.extend([f'price_ratio_{period}d', f'price_change_{period}d'])
        
        # Moving averages and deviations
        for period in [5, 10, 20, 50]:
            if i >= period:
                ma = df['price'].iloc[i-period:i].mean()
                ma_deviation = (current_price - ma) / ma
                ma_position = (current_price - df['price'].iloc[i-period:i].min()) / (df['price'].iloc[i-period:i].max() - df['price'].iloc[i-period:i].min()) if df['price'].iloc[i-period:i].max() != df['price'].iloc[i-period:i].min() else 0.5
                
                features.extend([ma_deviation, ma_position])
                names.extend([f'ma_deviation_{period}d', f'ma_position_{period}d'])
            else:
                features.extend([0.0, 0.5])
                names.extend([f'ma_deviation_{period}d', f'ma_position_{period}d'])
        
        # Price momentum
        for period in [3, 7, 14]:
            if i >= period:
                momentum = (current_price - df['price'].iloc[i-period]) / df['price'].iloc[i-period]
                features.append(momentum)
                names.append(f'momentum_{period}d')
            else:
                features.append(0.0)
                names.append(f'momentum_{period}d')
        
        return features, names
    
    def _create_volatility_features(self, df, i):
        """Create volatility-based features"""
        features = []
        names = []
        
        current_vol = df['volatility'].iloc[i]
        
        # Volatility statistics
        for period in [5, 10, 20, 50]:
            if i >= period:
                vol_window = df['volatility'].iloc[i-period:i]
                vol_mean = vol_window.mean()
                vol_std = vol_window.std()
                vol_min = vol_window.min()
                vol_max = vol_window.max()
                vol_percentile = (vol_window <= current_vol).mean()
                vol_zscore = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
                vol_range = vol_max - vol_min
                vol_cv = vol_std / vol_mean if vol_mean > 0 else 0
                
                features.extend([
                    vol_mean, vol_std, vol_min, vol_max,
                    vol_percentile, vol_zscore, vol_range, vol_cv
                ])
                names.extend([
                    f'vol_mean_{period}d', f'vol_std_{period}d',
                    f'vol_min_{period}d', f'vol_max_{period}d',
                    f'vol_percentile_{period}d', f'vol_zscore_{period}d',
                    f'vol_range_{period}d', f'vol_cv_{period}d'
                ])
            else:
                features.extend([current_vol, 0, current_vol, current_vol, 0.5, 0, 0, 0])
                names.extend([
                    f'vol_mean_{period}d', f'vol_std_{period}d',
                    f'vol_min_{period}d', f'vol_max_{period}d',
                    f'vol_percentile_{period}d', f'vol_zscore_{period}d',
                    f'vol_range_{period}d', f'vol_cv_{period}d'
                ])
        
        # Volatility momentum and persistence
        for period in [5, 10, 20]:
            if i >= period:
                vol_change = (current_vol - df['volatility'].iloc[i-period]) / df['volatility'].iloc[i-period] if df['volatility'].iloc[i-period] > 0 else 0
                
                # Volatility persistence (how many consecutive days vol has been above/below average)
                vol_avg = df['volatility'].iloc[max(0, i-50):i].mean()
                persistence = 0
                for j in range(i-1, max(i-period-1, -1), -1):
                    if (df['volatility'].iloc[j] > vol_avg) == (current_vol > vol_avg):
                        persistence += 1
                    else:
                        break
                
                features.extend([vol_change, persistence])
                names.extend([f'vol_change_{period}d', f'vol_persistence_{period}d'])
            else:
                features.extend([0.0, 0])
                names.extend([f'vol_change_{period}d', f'vol_persistence_{period}d'])
        
        return features, names
    
    def _create_return_features(self, df, i):
        """Create return-based features"""
        features = []
        names = []
        
        # Return statistics for different periods
        for period in [5, 10, 20, 50]:
            if i >= period:
                returns_window = df['returns'].iloc[i-period:i].dropna()
                
                if len(returns_window) > 2:
                    ret_mean = returns_window.mean()
                    ret_std = returns_window.std()
                    ret_skew = returns_window.skew()
                    ret_kurt = returns_window.kurtosis()
                    ret_min = returns_window.min()
                    ret_max = returns_window.max()
                    
                    # Sharpe ratio (annualized)
                    sharpe = ret_mean / ret_std * np.sqrt(252) if ret_std > 0 else 0
                    
                    # Downside deviation
                    downside_returns = returns_window[returns_window < 0]
                    downside_dev = downside_returns.std() if len(downside_returns) > 0 else 0
                    
                    # Maximum drawdown
                    cumulative = (1 + returns_window).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = ((cumulative - running_max) / running_max).min()
                    
                    # Positive/negative return ratios
                    pos_returns = (returns_window > 0).sum()
                    neg_returns = (returns_window < 0).sum()
                    win_rate = pos_returns / len(returns_window) if len(returns_window) > 0 else 0
                    
                    features.extend([
                        ret_mean, ret_std, ret_skew, ret_kurt, ret_min, ret_max,
                        sharpe, downside_dev, drawdown, win_rate
                    ])
                    names.extend([
                        f'ret_mean_{period}d', f'ret_std_{period}d',
                        f'ret_skew_{period}d', f'ret_kurt_{period}d',
                        f'ret_min_{period}d', f'ret_max_{period}d',
                        f'sharpe_{period}d', f'downside_dev_{period}d',
                        f'drawdown_{period}d', f'win_rate_{period}d'
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5])
                    names.extend([
                        f'ret_mean_{period}d', f'ret_std_{period}d',
                        f'ret_skew_{period}d', f'ret_kurt_{period}d',
                        f'ret_min_{period}d', f'ret_max_{period}d',
                        f'sharpe_{period}d', f'downside_dev_{period}d',
                        f'drawdown_{period}d', f'win_rate_{period}d'
                    ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5])
                names.extend([
                    f'ret_mean_{period}d', f'ret_std_{period}d',
                    f'ret_skew_{period}d', f'ret_kurt_{period}d',
                    f'ret_min_{period}d', f'ret_max_{period}d',
                    f'sharpe_{period}d', f'downside_dev_{period}d',
                    f'drawdown_{period}d', f'win_rate_{period}d'
                ])
        
        return features, names
    
    def _create_technical_features(self, df, i):
        """Create technical indicator features"""
        features = []
        names = []
        
        # Need enough data for technical indicators
        min_required = 50
        if i < min_required:
            # Return zeros for technical indicators if not enough data
            placeholder_features = [0.0] * 25  # Approximate number of technical features
            placeholder_names = [
                'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'williams_r', 'cci',
                'atr_14', 'atr_ratio', 'adx', 'aroon_up', 'aroon_down',
                'mfi', 'obv_slope', 'vpt_slope', 'trix', 'roc', 'mom', 'dpo'
            ]
            return placeholder_features, placeholder_names
        
        # Extract price arrays for talib
        prices = df['price'].iloc[:i+1].values.astype(float)
        highs = prices  # Simplified - using close as high
        lows = prices   # Simplified - using close as low
        closes = prices
        volumes = df['volume'].iloc[:i+1].values.astype(float) if 'volume' in df.columns else np.ones_like(prices)
        
        try:
            # RSI
            rsi_14 = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
            rsi_30 = talib.RSI(closes, timeperiod=30)[-1] if len(closes) >= 30 else 50
            
            # MACD
            if len(closes) >= 34:  # Need 34 periods for MACD
                macd, macd_signal, macd_hist = talib.MACD(closes)
                macd_val = macd[-1] if not np.isnan(macd[-1]) else 0
                macd_signal_val = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
                macd_hist_val = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            else:
                macd_val = macd_signal_val = macd_hist_val = 0
            
            # Bollinger Bands
            if len(closes) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
                bb_upper_val = bb_upper[-1] if not np.isnan(bb_upper[-1]) else prices[-1]
                bb_lower_val = bb_lower[-1] if not np.isnan(bb_lower[-1]) else prices[-1]
                bb_width = (bb_upper_val - bb_lower_val) / bb_middle[-1] if not np.isnan(bb_middle[-1]) and bb_middle[-1] != 0 else 0
                bb_position = (prices[-1] - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val != bb_lower_val else 0.5
            else:
                bb_upper_val = bb_lower_val = prices[-1]
                bb_width = bb_position = 0
            
            # Stochastic
            if len(closes) >= 14:
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
                stoch_k_val = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
                stoch_d_val = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50
            else:
                stoch_k_val = stoch_d_val = 50
            
            # Williams %R
            williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else -50
            
            # CCI
            cci = talib.CCI(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else 0
            
            # ATR
            if len(closes) >= 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                atr_ratio = atr / prices[-1] if prices[-1] != 0 else 0
            else:
                atr = atr_ratio = 0
            
            # ADX
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else 25
            
            # Aroon
            if len(closes) >= 14:
                aroon_down, aroon_up = talib.AROON(highs, lows, timeperiod=14)
                aroon_up_val = aroon_up[-1] if not np.isnan(aroon_up[-1]) else 50
                aroon_down_val = aroon_down[-1] if not np.isnan(aroon_down[-1]) else 50
            else:
                aroon_up_val = aroon_down_val = 50
            
            # MFI (Money Flow Index) - requires volume
            if len(closes) >= 14 and 'volume' in df.columns:
                mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=14)[-1]
                mfi = mfi if not np.isnan(mfi) else 50
            else:
                mfi = 50
            
            # Volume indicators
            if 'volume' in df.columns and len(volumes) >= 10:
                # OBV slope
                obv = talib.OBV(closes, volumes)
                if len(obv) >= 10:
                    obv_slope = (obv[-1] - obv[-10]) / 10 if obv[-10] != 0 else 0
                else:
                    obv_slope = 0
                
                # Volume Price Trend slope
                vpt = np.cumsum(volumes * (closes / np.roll(closes, 1) - 1))
                vpt_slope = (vpt[-1] - vpt[-10]) / 10 if len(vpt) >= 10 and vpt[-10] != 0 else 0
            else:
                obv_slope = vpt_slope = 0
            
            # TRIX
            trix = talib.TRIX(closes, timeperiod=14)[-1] if len(closes) >= 14 else 0
            trix = trix if not np.isnan(trix) else 0
            
            # Rate of Change
            roc = talib.ROC(closes, timeperiod=10)[-1] if len(closes) >= 10 else 0
            roc = roc if not np.isnan(roc) else 0
            
            # Momentum
            mom = talib.MOM(closes, timeperiod=10)[-1] if len(closes) >= 10 else 0
            mom = mom if not np.isnan(mom) else 0
            
            # Detrended Price Oscillator
            if len(closes) >= 20:
                sma = talib.SMA(closes, timeperiod=20)
                dpo = closes[-1] - sma[-11] if len(sma) >= 11 and not np.isnan(sma[-11]) else 0
            else:
                dpo = 0
            
            features = [
                rsi_14, rsi_30, macd_val, macd_signal_val, macd_hist_val,
                bb_upper_val, bb_lower_val, bb_width, bb_position,
                stoch_k_val, stoch_d_val, williams_r, cci,
                atr, atr_ratio, adx, aroon_up_val, aroon_down_val,
                mfi, obv_slope, vpt_slope, trix, roc, mom, dpo
            ]
            
            names = [
                'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'williams_r', 'cci',
                'atr_14', 'atr_ratio', 'adx', 'aroon_up', 'aroon_down',
                'mfi', 'obv_slope', 'vpt_slope', 'trix', 'roc', 'mom', 'dpo'
            ]
            
        except Exception:
            # If talib fails, return default values
            features = [50, 50, 0, 0, 0, 0, 0, 0, 0.5, 50, 50, -50, 0, 0, 0, 25, 50, 50, 50, 0, 0, 0, 0, 0, 0]
            names = [
                'rsi_14', 'rsi_30', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d', 'williams_r', 'cci',
                'atr_14', 'atr_ratio', 'adx', 'aroon_up', 'aroon_down',
                'mfi', 'obv_slope', 'vpt_slope', 'trix', 'roc', 'mom', 'dpo'
            ]
        
        return features, names
    
    def _create_statistical_features(self, df, i):
        """Create statistical features"""
        features = []
        names = []
        
        # Statistical tests and measures
        for period in [20, 50]:
            if i >= period:
                price_window = df['price'].iloc[i-period:i]
                return_window = df['returns'].iloc[i-period:i].dropna()
                
                # Normality tests (simplified)
                if len(return_window) > 7:
                    # Jarque-Bera test statistic (simplified)
                    jb_stat = stats.jarque_bera(return_window)[0] if len(return_window) > 7 else 0
                    
                    # Autocorrelation
                    autocorr_1 = return_window.autocorr(lag=1) if len(return_window) > 1 else 0
                    autocorr_1 = autocorr_1 if not np.isnan(autocorr_1) else 0
                    
                    # Hurst exponent (simplified calculation)
                    if len(price_window) > 10:
                        lags = range(2, min(20, len(price_window)//2))
                        tau = [np.sqrt(np.std(np.subtract(price_window[lag:], price_window[:-lag]))) for lag in lags]
                        if len(tau) > 2:
                            poly = np.polyfit(np.log(lags), np.log(tau), 1)
                            hurst = poly[0] * 2.0
                        else:
                            hurst = 0.5
                    else:
                        hurst = 0.5
                    
                    # Runs test (simplified)
                    median_return = return_window.median()
                    runs, n1, n2 = 0, 0, 0
                    
                    # Count runs above/below median
                    for j, ret in enumerate(return_window):
                        if ret > median_return:
                            n1 += 1
                            if j == 0 or return_window.iloc[j-1] <= median_return:
                                runs += 1
                        else:
                            n2 += 1
                            if j == 0 or return_window.iloc[j-1] > median_return:
                                runs += 1
                    
                    # Expected runs and test statistic
                    if n1 > 0 and n2 > 0:
                        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
                        runs_z = (runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0
                    else:
                        runs_z = 0
                    
                    features.extend([jb_stat, autocorr_1, hurst, runs_z])
                    names.extend([
                        f'jarque_bera_{period}d', f'autocorr_1_{period}d',
                        f'hurst_{period}d', f'runs_test_{period}d'
                    ])
                else:
                    features.extend([0, 0, 0.5, 0])
                    names.extend([
                        f'jarque_bera_{period}d', f'autocorr_1_{period}d',
                        f'hurst_{period}d', f'runs_test_{period}d'
                    ])
            else:
                features.extend([0, 0, 0.5, 0])
                names.extend([
                    f'jarque_bera_{period}d', f'autocorr_1_{period}d',
                    f'hurst_{period}d', f'runs_test_{period}d'
                ])
        
        return features, names
    
    def _create_regime_features(self, df, i):
        """Create regime-related features"""
        features = []
        names = []
        
        # Market state indicators
        for period in [20, 50, 100]:
            if i >= period:
                vol_window = df['volatility'].iloc[i-period:i]
                price_window = df['price'].iloc[i-period:i]
                return_window = df['returns'].iloc[i-period:i].dropna()
                
                # Volatility regime indicators
                vol_percentile = (vol_window <= df['volatility'].iloc[i]).mean()
                vol_median = vol_window.median()
                vol_regime = 1 if df['volatility'].iloc[i] > vol_median else 0
                
                # Trend regime
                price_trend = (price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0] if price_window.iloc[0] != 0 else 0
                trend_strength = abs(price_trend)
                
                # Market stress indicators
                if len(return_window) > 0:
                    large_moves = (np.abs(return_window) > 2 * return_window.std()).sum()
                    stress_indicator = large_moves / len(return_window)
                    
                    # VIX-like indicator
                    vix_like = return_window.std() * np.sqrt(252) * 100
                else:
                    stress_indicator = 0
                    vix_like = 20  # Default VIX-like value
                
                features.extend([
                    vol_percentile, vol_regime, price_trend, trend_strength,
                    stress_indicator, vix_like
                ])
                names.extend([
                    f'vol_percentile_{period}d', f'vol_regime_{period}d',
                    f'price_trend_{period}d', f'trend_strength_{period}d',
                    f'stress_indicator_{period}d', f'vix_like_{period}d'
                ])
            else:
                features.extend([0.5, 0, 0, 0, 0, 20])
                names.extend([
                    f'vol_percentile_{period}d', f'vol_regime_{period}d',
                    f'price_trend_{period}d', f'trend_strength_{period}d',
                    f'stress_indicator_{period}d', f'vix_like_{period}d'
                ])
        
        return features, names
    
    def _create_time_features(self, df, i):
        """Create time-based features"""
        features = []
        names = []
        
        # Extract time information
        current_date = df['date'].iloc[i]
        
        # Day of week (0 = Monday, 6 = Sunday)
        day_of_week = current_date.weekday()
        
        # Month (1-12)
        month = current_date.month
        
        # Quarter (1-4)
        quarter = (month - 1) // 3 + 1
        
        # Day of month
        day_of_month = current_date.day
        
        # Is month end (last 3 days of month)
        is_month_end = 1 if day_of_month >= 28 else 0
        
        # Is quarter end
        is_quarter_end = 1 if month in [3, 6, 9, 12] and day_of_month >= 28 else 0
        
        # Cyclical encoding for temporal features
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        features = [
            day_of_week, month, quarter, day_of_month,
            is_month_end, is_quarter_end,
            day_sin, day_cos, month_sin, month_cos
        ]
        
        names = [
            'day_of_week', 'month', 'quarter', 'day_of_month',
            'is_month_end', 'is_quarter_end',
            'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        return features, names
    
    def _create_interaction_features(self, df, i):
        """Create interaction features (computationally expensive)"""
        features = []
        names = []
        
        # Simple interactions between key variables
        current_vol = df['volatility'].iloc[i]
        current_price = df['price'].iloc[i]
        
        if i >= 20:
            # Price-volatility interactions
            recent_returns = df['returns'].iloc[i-20:i].dropna()
            vol_20d = df['volatility'].iloc[i-20:i].mean()
            
            if len(recent_returns) > 0:
                # Volatility-adjusted momentum
                momentum_10d = (current_price - df['price'].iloc[i-10]) / df['price'].iloc[i-10] if i >= 10 and df['price'].iloc[i-10] != 0 else 0
                vol_adj_momentum = momentum_10d / current_vol if current_vol > 0 else 0
                
                # Return-volatility correlation
                ret_vol_corr = np.corrcoef(recent_returns, df['volatility'].iloc[i-len(recent_returns):i])[0, 1] if len(recent_returns) > 1 else 0
                ret_vol_corr = ret_vol_corr if not np.isnan(ret_vol_corr) else 0
                
                # Volatility premium (current vs average)
                vol_premium = (current_vol - vol_20d) / vol_20d if vol_20d > 0 else 0
                
                features.extend([vol_adj_momentum, ret_vol_corr, vol_premium])
                names.extend(['vol_adj_momentum', 'ret_vol_corr', 'vol_premium'])
            else:
                features.extend([0, 0, 0])
                names.extend(['vol_adj_momentum', 'ret_vol_corr', 'vol_premium'])
        else:
            features.extend([0, 0, 0])
            names.extend(['vol_adj_momentum', 'ret_vol_corr', 'vol_premium'])
        
        return features, names
    
    def fit_scalers(self, features, scaling_method='standard'):
        """Fit scalers to the features"""
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unknown scaling method")
        
        self.scalers['main'] = scaler.fit(features)
        self.is_fitted = True
        
        return scaler
    
    def transform_features(self, features, scaling_method='standard'):
        """Transform features using fitted scalers"""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted first")
        
        return self.scalers['main'].transform(features)
    
    def fit_transform_features(self, features, scaling_method='standard'):
        """Fit scalers and transform features in one step"""
        scaler = self.fit_scalers(features, scaling_method)
        return scaler.transform(features)
    
    def get_feature_importance_groups(self):
        """Group features by type for importance analysis"""
        groups = {
            'price_features': [],
            'volatility_features': [],
            'return_features': [],
            'technical_features': [],
            'statistical_features': [],
            'regime_features': [],
            'time_features': [],
            'interaction_features': []
        }
        
        for i, name in enumerate(self.feature_names):
            if any(keyword in name for keyword in ['price', 'ma_', 'momentum']):
                groups['price_features'].append((i, name))
            elif 'vol' in name:
                groups['volatility_features'].append((i, name))
            elif any(keyword in name for keyword in ['ret_', 'sharpe', 'drawdown']):
                groups['return_features'].append((i, name))
            elif any(keyword in name for keyword in ['rsi', 'macd', 'bb_', 'stoch', 'atr']):
                groups['technical_features'].append((i, name))
            elif any(keyword in name for keyword in ['jarque', 'autocorr', 'hurst', 'runs']):
                groups['statistical_features'].append((i, name))
            elif any(keyword in name for keyword in ['regime', 'trend', 'stress']):
                groups['regime_features'].append((i, name))
            elif any(keyword in name for keyword in ['day', 'month', 'quarter']):
                groups['time_features'].append((i, name))
            elif any(keyword in name for keyword in ['adj_', 'corr', 'premium']):
                groups['interaction_features'].append((i, name))
        
        return groups
    
    def save_feature_config(self, filepath):
        """Save feature engineering configuration"""
        import json
        
        config_data = {
            'config': self.config,
            'lookback_periods': self.lookback_periods,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def load_feature_config(self, filepath):
        """Load feature engineering configuration"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.config = config_data['config']
        self.lookback_periods = config_data['lookback_periods']
        self.feature_names = config_data['feature_names']
        self.is_fitted = config_data['is_fitted']