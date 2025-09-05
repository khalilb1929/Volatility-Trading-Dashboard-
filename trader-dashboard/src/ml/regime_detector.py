import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class RegimeDetector:
    """Machine learning model for detecting market volatility regimes"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.regime_labels = None
        self.regime_centers = None
        self.n_regimes = None
        self.model_type = None
        self.is_fitted = False
        self.feature_names = []
        self.regime_descriptions = {}
        
        # Model configurations
        self.model_configs = {
            'kmeans': {
                'model': KMeans,
                'params': {
                    'random_state': 42,
                    'n_init': 10,
                    'max_iter': 300
                }
            },
            'gaussian_mixture': {
                'model': GaussianMixture,
                'params': {
                    'random_state': 42,
                    'max_iter': 100,
                    'n_init': 1
                }
            },
            'dbscan': {
                'model': DBSCAN,
                'params': {
                    'eps': 0.5,
                    'min_samples': 5
                }
            }
        }
        
        # Default regime names
        self.regime_names = {
            0: "Low Volatility",
            1: "Medium Volatility", 
            2: "High Volatility",
            3: "Extreme Volatility"
        }
    
    def create_regime_features(self, prices, volatilities, returns=None, window_sizes=None):
        """Create features for regime detection"""
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
        if len(prices) != len(volatilities):
            raise ValueError("Prices and volatilities must have the same length")
        
        if len(prices) < max(window_sizes) + 1:
            raise ValueError(f"Need at least {max(window_sizes) + 1} data points")
        
        df = pd.DataFrame({
            'price': prices,
            'volatility': volatilities
        })
        
        # Calculate returns if not provided
        if returns is None:
            df['returns'] = df['price'].pct_change()
        else:
            df['returns'] = returns
        
        features = []
        feature_names = []
        
        # Start from the maximum window size to have complete features
        start_idx = max(window_sizes)
        
        for i in range(start_idx, len(df)):
            feature_row = []
            
            # Volatility-based features
            current_vol = df['volatility'].iloc[i]
            feature_row.append(current_vol)
            if i == start_idx:
                feature_names.append('current_volatility')
            
            # Rolling volatility statistics
            for window in window_sizes:
                vol_window = df['volatility'].iloc[i-window:i]
                
                # Statistical measures
                vol_mean = vol_window.mean()
                vol_std = vol_window.std()
                vol_min = vol_window.min()
                vol_max = vol_window.max()
                vol_range = vol_max - vol_min
                vol_cv = vol_std / vol_mean if vol_mean != 0 else 0  # Coefficient of variation
                
                # Percentile positions
                vol_percentile = (vol_window <= current_vol).mean()
                
                # Z-score
                vol_zscore = (current_vol - vol_mean) / vol_std if vol_std != 0 else 0
                
                feature_row.extend([
                    vol_mean, vol_std, vol_min, vol_max, vol_range,
                    vol_cv, vol_percentile, vol_zscore
                ])
                
                if i == start_idx:
                    feature_names.extend([
                        f'vol_mean_{window}d', f'vol_std_{window}d',
                        f'vol_min_{window}d', f'vol_max_{window}d', f'vol_range_{window}d',
                        f'vol_cv_{window}d', f'vol_percentile_{window}d', f'vol_zscore_{window}d'
                    ])
            
            # Return-based features
            for window in window_sizes:
                returns_window = df['returns'].iloc[i-window:i].dropna()
                
                if len(returns_window) > 2:
                    ret_mean = returns_window.mean()
                    ret_std = returns_window.std()
                    ret_skew = returns_window.skew()
                    ret_kurt = returns_window.kurtosis()
                    ret_min = returns_window.min()
                    ret_max = returns_window.max()
                    
                    # Drawdown
                    cumulative = (1 + returns_window).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = ((cumulative - running_max) / running_max).min()
                    
                    feature_row.extend([
                        ret_mean, ret_std, ret_skew, ret_kurt,
                        ret_min, ret_max, drawdown
                    ])
                else:
                    feature_row.extend([0, 0, 0, 0, 0, 0, 0])
                
                if i == start_idx:
                    feature_names.extend([
                        f'ret_mean_{window}d', f'ret_std_{window}d',
                        f'ret_skew_{window}d', f'ret_kurt_{window}d',
                        f'ret_min_{window}d', f'ret_max_{window}d', f'drawdown_{window}d'
                    ])
            
            # Price momentum features
            for window in [5, 10, 20]:
                if i >= window:
                    price_change = (df['price'].iloc[i] - df['price'].iloc[i-window]) / df['price'].iloc[i-window]
                    feature_row.append(price_change)
                    
                    if i == start_idx:
                        feature_names.append(f'price_momentum_{window}d')
                else:
                    feature_row.append(0)
                    if i == start_idx:
                        feature_names.append(f'price_momentum_{window}d')
            
            # Volatility regime indicators
            # Volatility clustering (correlation with recent volatility)
            if i >= 20:
                recent_vols = df['volatility'].iloc[i-20:i]
                vol_correlation = np.corrcoef(recent_vols[:-1], recent_vols[1:])[0, 1] if len(recent_vols) > 1 else 0
                
                # Volatility persistence (how long has vol been in current state)
                vol_median = recent_vols.median()
                persistence = 0
                for j in range(len(recent_vols)-1, -1, -1):
                    if (recent_vols.iloc[j] > vol_median) == (current_vol > vol_median):
                        persistence += 1
                    else:
                        break
                
                feature_row.extend([vol_correlation, persistence])
                
                if i == start_idx:
                    feature_names.extend(['vol_correlation', 'vol_persistence'])
            else:
                feature_row.extend([0, 0])
                if i == start_idx:
                    feature_names.extend(['vol_correlation', 'vol_persistence'])
            
            # Market stress indicators
            # Large move frequency
            if i >= 20:
                recent_returns = df['returns'].iloc[i-20:i].dropna()
                if len(recent_returns) > 0:
                    large_moves = (np.abs(recent_returns) > 2 * recent_returns.std()).sum()
                    large_move_freq = large_moves / len(recent_returns)
                    
                    # Gap frequency (large overnight moves)
                    gaps = np.abs(recent_returns) > 0.02  # 2% moves
                    gap_freq = gaps.sum() / len(recent_returns)
                    
                    feature_row.extend([large_move_freq, gap_freq])
                else:
                    feature_row.extend([0, 0])
                
                if i == start_idx:
                    feature_names.extend(['large_move_freq', 'gap_freq'])
            else:
                feature_row.extend([0, 0])
                if i == start_idx:
                    feature_names.extend(['large_move_freq', 'gap_freq'])
            
            features.append(feature_row)
        
        self.feature_names = feature_names
        return np.array(features), feature_names
    
    def find_optimal_regimes(self, features, max_regimes=6, method='kmeans'):
        """Find the optimal number of regimes using various criteria"""
        if method not in self.model_configs:
            raise ValueError(f"Unknown method: {method}")
        
        if method == 'dbscan':
            # DBSCAN doesn't need pre-specified number of clusters
            return self._fit_dbscan(features)
        
        scores = []
        inertias = []
        
        for n_regimes in range(2, max_regimes + 1):
            # Fit model
            model_class = self.model_configs[method]['model']
            params = self.model_configs[method]['params'].copy()
            
            if method == 'kmeans':
                params['n_clusters'] = n_regimes
                model = model_class(**params)
            elif method == 'gaussian_mixture':
                params['n_components'] = n_regimes
                model = model_class(**params)
            
            labels = model.fit_predict(features)
            
            # Calculate silhouette score
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(features, labels)
                scores.append((n_regimes, score))
                
                if hasattr(model, 'inertia_'):
                    inertias.append((n_regimes, model.inertia_))
        
        if not scores:
            return 3  # Default to 3 regimes
        
        # Find optimal number of regimes (highest silhouette score)
        optimal_regimes = max(scores, key=lambda x: x[1])[0]
        
        return optimal_regimes
    
    def _fit_dbscan(self, features):
        """Fit DBSCAN and return the number of clusters found"""
        model = DBSCAN(**self.model_configs['dbscan']['params'])
        labels = model.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points
        return max(2, min(n_clusters, 6))  # Ensure reasonable number of clusters
    
    def fit_regimes(self, prices, volatilities, returns=None, n_regimes='auto', method='kmeans', use_pca=False):
        """Fit regime detection model"""
        try:
            # Create features
            features, feature_names = self.create_regime_features(prices, volatilities, returns)
            
            if len(features) == 0:
                raise ValueError("No features could be created")
            
            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            
            # Apply PCA if requested
            if use_pca:
                self.pca = PCA(n_components=0.95)  # Keep 95% of variance
                features_scaled = self.pca.fit_transform(features_scaled)
            
            # Determine optimal number of regimes
            if n_regimes == 'auto':
                n_regimes = self.find_optimal_regimes(features_scaled, method=method)
            
            self.n_regimes = n_regimes
            self.model_type = method
            
            # Fit the model
            model_class = self.model_configs[method]['model']
            params = self.model_configs[method]['params'].copy()
            
            if method == 'kmeans':
                params['n_clusters'] = n_regimes
                self.model = model_class(**params)
                self.regime_labels = self.model.fit_predict(features_scaled)
                self.regime_centers = self.model.cluster_centers_
                
            elif method == 'gaussian_mixture':
                params['n_components'] = n_regimes
                self.model = model_class(**params)
                self.regime_labels = self.model.fit_predict(features_scaled)
                self.regime_centers = self.model.means_
                
            elif method == 'dbscan':
                self.model = model_class(**params)
                self.regime_labels = self.model.fit_predict(features_scaled)
                
                # For DBSCAN, create pseudo-centers
                unique_labels = set(self.regime_labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # Remove noise label
                
                self.regime_centers = []
                for label in unique_labels:
                    cluster_points = features_scaled[self.regime_labels == label]
                    center = np.mean(cluster_points, axis=0)
                    self.regime_centers.append(center)
                
                self.regime_centers = np.array(self.regime_centers)
                self.n_regimes = len(unique_labels)
            
            # Create regime descriptions
            self._create_regime_descriptions(features, volatilities)
            
            self.is_fitted = True
            
            # Calculate fit statistics
            fit_stats = self._calculate_fit_statistics(features_scaled)
            
            return {
                'n_regimes': self.n_regimes,
                'method': method,
                'fit_statistics': fit_stats,
                'regime_descriptions': self.regime_descriptions
            }
            
        except Exception as e:
            raise Exception(f"Regime fitting failed: {str(e)}")
    
    def _create_regime_descriptions(self, features, volatilities):
        """Create descriptive names and statistics for each regime"""
        if self.regime_labels is None:
            return
        
        unique_labels = sorted(set(self.regime_labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label for DBSCAN
        
        # Calculate statistics for each regime
        for label in unique_labels:
            regime_mask = self.regime_labels == label
            regime_vols = np.array(volatilities)[max(len(volatilities) - len(self.regime_labels), 0):][regime_mask]
            
            if len(regime_vols) > 0:
                vol_mean = np.mean(regime_vols)
                vol_std = np.std(regime_vols)
                vol_min = np.min(regime_vols)
                vol_max = np.max(regime_vols)
                frequency = np.sum(regime_mask) / len(self.regime_labels)
                
                # Classify regime based on volatility level
                if vol_mean < 15:
                    regime_name = "Low Volatility"
                    description = "Calm market conditions with low price fluctuations"
                elif vol_mean < 25:
                    regime_name = "Medium Volatility"
                    description = "Normal market conditions with moderate price movements"
                elif vol_mean < 40:
                    regime_name = "High Volatility"
                    description = "Elevated uncertainty with increased price swings"
                else:
                    regime_name = "Extreme Volatility"
                    description = "Crisis or panic conditions with very large price movements"
                
                self.regime_descriptions[label] = {
                    'name': regime_name,
                    'description': description,
                    'vol_mean': vol_mean,
                    'vol_std': vol_std,
                    'vol_min': vol_min,
                    'vol_max': vol_max,
                    'frequency': frequency,
                    'sample_count': np.sum(regime_mask)
                }
    
    def _calculate_fit_statistics(self, features_scaled):
        """Calculate statistics about the regime fit"""
        if self.regime_labels is None:
            return {}
        
        stats = {}
        
        # Silhouette score
        unique_labels = set(self.regime_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            stats['silhouette_score'] = silhouette_score(features_scaled, self.regime_labels)
        
        # Regime distribution
        label_counts = np.bincount(self.regime_labels[self.regime_labels >= 0])  # Exclude noise
        stats['regime_distribution'] = label_counts / len(self.regime_labels)
        
        # Inertia (for KMeans)
        if hasattr(self.model, 'inertia_'):
            stats['inertia'] = self.model.inertia_
        
        # Log likelihood (for Gaussian Mixture)
        if hasattr(self.model, 'score'):
            stats['log_likelihood'] = self.model.score(features_scaled)
        
        return stats
    
    def predict_regime(self, prices, volatilities, returns=None):
        """Predict the current regime for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create features for the new data
        features, _ = self.create_regime_features(prices, volatilities, returns)
        
        if len(features) == 0:
            raise ValueError("Cannot create features from provided data")
        
        # Use the most recent feature vector
        latest_features = features[-1:] 
        
        # Scale features
        features_scaled = self.scaler.transform(latest_features)
        
        # Apply PCA if used during fitting
        if self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)
        
        # Predict regime
        regime_label = self.model.predict(features_scaled)[0]
        
        # Get regime information
        regime_info = self.regime_descriptions.get(regime_label, {
            'name': f'Regime {regime_label}',
            'description': 'Unknown regime'
        })
        
        # Calculate confidence (for probabilistic models)
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        
        return {
            'regime_label': regime_label,
            'regime_name': regime_info.get('name', f'Regime {regime_label}'),
            'description': regime_info.get('description', ''),
            'confidence': confidence,
            'regime_stats': regime_info
        }
    
    def get_regime_transitions(self):
        """Analyze regime transitions"""
        if not self.is_fitted or self.regime_labels is None:
            return None
        
        transitions = []
        current_regime = self.regime_labels[0]
        transition_start = 0
        
        for i, regime in enumerate(self.regime_labels[1:], 1):
            if regime != current_regime:
                # Record transition
                transitions.append({
                    'from_regime': current_regime,
                    'to_regime': regime,
                    'start_index': transition_start,
                    'end_index': i,
                    'duration': i - transition_start
                })
                
                current_regime = regime
                transition_start = i
        
        # Add final regime period
        transitions.append({
            'from_regime': current_regime,
            'to_regime': None,
            'start_index': transition_start,
            'end_index': len(self.regime_labels),
            'duration': len(self.regime_labels) - transition_start
        })
        
        return transitions
    
    def get_regime_summary(self):
        """Get a summary of the detected regimes"""
        if not self.is_fitted:
            return "No regimes detected. Please fit the model first."
        
        summary = f"""Volatility Regime Detection Summary
{'='*50}
Model Type: {self.model_type}
Number of Regimes: {self.n_regimes}
Total Observations: {len(self.regime_labels)}

REGIME CHARACTERISTICS:
"""
        
        for label, info in self.regime_descriptions.items():
            summary += f"""
Regime {label}: {info['name']}
  Description: {info['description']}
  Average Volatility: {info['vol_mean']:.1f}% ± {info['vol_std']:.1f}%
  Volatility Range: {info['vol_min']:.1f}% - {info['vol_max']:.1f}%
  Frequency: {info['frequency']:.1%} ({info['sample_count']} observations)
"""
        
        # Add transition analysis
        transitions = self.get_regime_transitions()
        if transitions:
            summary += f"\nREGIME TRANSITIONS:\n"
            summary += f"Total Transitions: {len([t for t in transitions if t['to_regime'] is not None])}\n"
            
            # Average duration by regime
            regime_durations = {}
            for transition in transitions:
                regime = transition['from_regime']
                if regime not in regime_durations:
                    regime_durations[regime] = []
                regime_durations[regime].append(transition['duration'])
            
            summary += "Average Regime Duration:\n"
            for regime, durations in regime_durations.items():
                avg_duration = np.mean(durations)
                regime_name = self.regime_descriptions.get(regime, {}).get('name', f'Regime {regime}')
                summary += f"  {regime_name}: {avg_duration:.1f} periods\n"
        
        return summary
    
    def save_model(self, filepath):
        """Save the fitted regime model"""
        if not self.is_fitted:
            raise ValueError("No fitted model to save")
        
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'regime_labels': self.regime_labels,
            'regime_centers': self.regime_centers,
            'n_regimes': self.n_regimes,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'regime_descriptions': self.regime_descriptions,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a fitted regime model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 
        self.pca = model_data.get('pca')
        self.regime_labels = model_data['regime_labels']
        self.regime_centers = model_data['regime_centers']
        self.n_regimes = model_data['n_regimes']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.regime_descriptions = model_data['regime_descriptions']
        self.is_fitted = model_data['is_fitted']