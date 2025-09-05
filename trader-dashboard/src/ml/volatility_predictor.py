import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class VolatilityPredictor:
    """Machine learning model for predicting volatility"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.model_type = None
        self.is_trained = False
        self.feature_names = []
        self.training_results = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boost': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'neural_network': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': (50, 25, 10),
                    'max_iter': 1000,
                    'learning_rate': 'adaptive',
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': 42
                }
            },
            'linear_regression': {
                'model': LinearRegression,
                'params': {
                    'fit_intercept': True
                }
            },
            'svr': {
                'model': SVR,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            }
        }
    
    def create_features(self, prices, volatilities, lookback_periods=None):
        """Create features for volatility prediction"""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20]
        if len(prices) < max(lookback_periods) + 1:
            raise ValueError(f"Need at least {max(lookback_periods) + 1} data points")
        
        df = pd.DataFrame({
            'price': prices,
            'volatility': volatilities
        })
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        features = []
        feature_names = []
        
        # Start from the maximum lookback period to have complete features
        start_idx = max(lookback_periods)
        
        for i in range(start_idx, len(df)):
            feature_row = []
            
            # Current market features
            current_price = df['price'].iloc[i]
            
            # Historical volatility features
            for period in lookback_periods:
                # Recent volatility statistics
                recent_vol = df['volatility'].iloc[i-period:i]
                feature_row.extend([
                    recent_vol.mean(),
                    recent_vol.std(),
                    recent_vol.min(),
                    recent_vol.max(),
                    recent_vol.iloc[-1]  # Most recent volatility
                ])
                
                if i == start_idx:  # Only add feature names once
                    feature_names.extend([
                        f'vol_mean_{period}d',
                        f'vol_std_{period}d', 
                        f'vol_min_{period}d',
                        f'vol_max_{period}d',
                        f'vol_last_{period}d'
                    ])
            
            # Return-based features
            for period in lookback_periods:
                recent_returns = df['returns'].iloc[i-period:i].dropna()
                if len(recent_returns) > 0:
                    feature_row.extend([
                        recent_returns.mean(),
                        recent_returns.std(),
                        recent_returns.skew() if len(recent_returns) > 2 else 0,
                        recent_returns.kurtosis() if len(recent_returns) > 3 else 0
                    ])
                    
                    if i == start_idx:
                        feature_names.extend([
                            f'ret_mean_{period}d',
                            f'ret_std_{period}d',
                            f'ret_skew_{period}d',
                            f'ret_kurt_{period}d'
                        ])
                else:
                    feature_row.extend([0, 0, 0, 0])
                    if i == start_idx:
                        feature_names.extend([
                            f'ret_mean_{period}d',
                            f'ret_std_{period}d',
                            f'ret_skew_{period}d', 
                            f'ret_kurt_{period}d'
                        ])
            
            # Technical indicators
            # Moving averages
            for period in [5, 10, 20]:
                if i >= period:
                    ma = df['price'].iloc[i-period:i].mean()
                    feature_row.append((current_price - ma) / ma)  # Price deviation from MA
                    
                    if i == start_idx:
                        feature_names.append(f'price_vs_ma_{period}d')
                else:
                    feature_row.append(0)
                    if i == start_idx:
                        feature_names.append(f'price_vs_ma_{period}d')
            
            # Volatility regime features
            if i >= 60:  # Need enough data for regime detection
                vol_60d = df['volatility'].iloc[i-60:i]
                vol_percentile = (vol_60d <= df['volatility'].iloc[i]).mean()
                vol_zscore = (df['volatility'].iloc[i] - vol_60d.mean()) / vol_60d.std() if vol_60d.std() > 0 else 0
                
                feature_row.extend([vol_percentile, vol_zscore])
                
                if i == start_idx:
                    feature_names.extend(['vol_percentile_60d', 'vol_zscore_60d'])
            else:
                feature_row.extend([0.5, 0])  # Neutral values
                if i == start_idx:
                    feature_names.extend(['vol_percentile_60d', 'vol_zscore_60d'])
            
            # Momentum features
            if i >= 10:
                price_momentum = (df['price'].iloc[i] - df['price'].iloc[i-10]) / df['price'].iloc[i-10]
                vol_momentum = (df['volatility'].iloc[i] - df['volatility'].iloc[i-10]) / df['volatility'].iloc[i-10] if df['volatility'].iloc[i-10] != 0 else 0
                
                feature_row.extend([price_momentum, vol_momentum])
                
                if i == start_idx:
                    feature_names.extend(['price_momentum_10d', 'vol_momentum_10d'])
            else:
                feature_row.extend([0, 0])
                if i == start_idx:
                    feature_names.extend(['price_momentum_10d', 'vol_momentum_10d'])
            
            features.append(feature_row)
        
        self.feature_names = feature_names
        return np.array(features), feature_names
    
    def prepare_targets(self, volatilities, features_start_idx, forecast_horizon=1):
        """Prepare target values for prediction"""
        # Start from features_start_idx and predict forecast_horizon days ahead
        targets = []
        
        for i in range(features_start_idx, len(volatilities) - forecast_horizon):
            # Target is the volatility forecast_horizon days ahead
            target_vol = volatilities[i + forecast_horizon]
            targets.append(target_vol)
        
        return np.array(targets)
    
    def train_model(self, prices, volatilities, model_type='random_forest', test_size=0.2, forecast_horizon=1):
        """Train the volatility prediction model"""
        try:
            if len(prices) != len(volatilities):
                raise ValueError("Prices and volatilities must have the same length")
            
            if len(prices) < 100:
                raise ValueError("Need at least 100 data points for training")
            
            # Create features
            X, feature_names = self.create_features(prices, volatilities)
            
            # Create targets (predict volatility forecast_horizon days ahead)
            lookback_periods = [5, 10, 20]
            features_start_idx = max(lookback_periods)
            y = self.prepare_targets(volatilities, features_start_idx, forecast_horizon)
            
            # Ensure X and y have the same length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            
            if len(X) == 0:
                raise ValueError("No valid feature-target pairs could be created")
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Split data (use time series split to avoid look-ahead bias)
            split_point = int(len(X_scaled) * (1 - test_size))
            X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Initialize model
            if model_type not in self.model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model_class = self.model_configs[model_type]['model']
            model_params = self.model_configs[model_type]['params'].copy()
            
            self.model = model_class(**model_params)
            self.model_type = model_type
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation (using time series split)
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
            cv_scores = -cv_scores  # Convert to positive MSE
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = list(zip(feature_names, self.model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Store results
            self.training_results = {
                'model_type': model_type,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_scaled.shape[1],
                'feature_importance': feature_importance,
                'forecast_horizon': forecast_horizon
            }
            
            self.is_trained = True
            
            return self.training_results
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def predict_volatility(self, prices, volatilities, days_ahead=5):
        """Predict volatility for multiple days ahead"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(prices) != len(volatilities):
            raise ValueError("Prices and volatilities must have the same length")
        
        predictions = []
        
        # Make predictions for each day ahead
        current_prices = prices.copy()
        current_vols = volatilities.copy()
        
        for _ in range(days_ahead):
            # Create features for current state
            X, _ = self.create_features(current_prices, current_vols)
            
            if len(X) == 0:
                # If we can't create features, use last known volatility
                predictions.append(current_vols[-1])
                continue
            
            # Use the most recent feature vector
            X_latest = X[-1:] 
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X_latest)
            
            # Make prediction
            pred_vol = self.model.predict(X_scaled)[0]
            
            # Ensure prediction is reasonable (volatility can't be negative)
            pred_vol = max(0.01, pred_vol)  # Minimum 1% volatility
            pred_vol = min(200.0, pred_vol)  # Maximum 200% volatility
            
            predictions.append(pred_vol)
            
            # Update data for next prediction (simple approach)
            # In practice, you might want more sophisticated methods
            
            # Estimate next price using random walk (for feature creation)
            # This is a simplification - in practice you might use other price prediction methods
            last_return = (current_prices[-1] - current_prices[-2]) / current_prices[-2] if len(current_prices) > 1 else 0
            next_price = current_prices[-1] * (1 + np.random.normal(0, pred_vol/100/np.sqrt(252)))
            
            # Add predicted volatility and estimated price to the series
            current_prices = np.append(current_prices, next_price)
            current_vols = np.append(current_vols, pred_vol)
        
        return np.array(predictions)
    
    def get_feature_importance(self, top_n=10):
        """Get the most important features"""
        if not self.is_trained or not self.training_results.get('feature_importance'):
            return None
        
        importance_list = self.training_results['feature_importance']
        return importance_list[:top_n]
    
    def predict_single_step(self, prices, volatilities):
        """Predict volatility for the next single time step"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        X, _ = self.create_features(prices, volatilities)
        
        if len(X) == 0:
            raise ValueError("Cannot create features from provided data")
        
        # Use the most recent feature vector
        X_latest = X[-1:]
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X_latest)
        
        # Make prediction
        pred_vol = self.model.predict(X_scaled)[0]
        
        # Ensure prediction is reasonable
        pred_vol = max(0.01, min(200.0, pred_vol))
        
        return pred_vol
    
    def evaluate_model(self, prices, volatilities, start_date=None, end_date=None):
        """Evaluate model performance on out-of-sample data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            # Create features and targets
            X, _ = self.create_features(prices, volatilities)
            lookback_periods = [5, 10, 20]
            features_start_idx = max(lookback_periods)
            y = self.prepare_targets(volatilities, features_start_idx, 1)
            
            # Ensure X and y have the same length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            
            if len(X) == 0:
                raise ValueError("No valid data for evaluation")
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Calculate directional accuracy
            actual_direction = np.diff(y) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) if len(actual_direction) > 0 else 0
            
            # Calculate error statistics
            errors = y - y_pred
            
            evaluation_results = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(np.abs(errors)),
                'samples': len(y)
            }
            
            return evaluation_results
            
        except Exception as e:
            raise Exception(f"Evaluation failed: {str(e)}")
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_results': self.training_results,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_scaler = model_data.get('feature_scaler')
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.training_results = model_data['training_results']
        self.is_trained = model_data['is_trained']
    
    def get_model_summary(self):
        """Get a summary of the trained model"""
        if not self.is_trained:
            return "No model trained"
        
        summary = f"""Volatility Prediction Model Summary
{'='*50}
Model Type: {self.model_type}
Training R²: {self.training_results['train_r2']:.4f}
Test R²: {self.training_results['test_r2']:.4f}
Training RMSE: {np.sqrt(self.training_results['train_mse']):.4f}
Test RMSE: {np.sqrt(self.training_results['test_mse']):.4f}
Cross-Validation RMSE: {np.sqrt(self.training_results['cv_mean']):.4f} ± {np.sqrt(self.training_results['cv_std']):.4f}

Training Samples: {self.training_results['train_samples']}
Test Samples: {self.training_results['test_samples']}
Features: {self.training_results['n_features']}
Forecast Horizon: {self.training_results['forecast_horizon']} day(s)
"""
        
        # Add feature importance if available
        if self.training_results.get('feature_importance'):
            summary += "\nTop 5 Most Important Features:\n"
            for i, (feature, importance) in enumerate(self.training_results['feature_importance'][:5]):
                summary += f"{i+1}. {feature}: {importance:.4f}\n"
        
        return summary