import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime, timedelta
import threading
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from .volatility_education import VolatilityEducationTab
    from .widgets import StatusBar, SymbolEntry, PeriodSelector, AlertPanel, MetricsDisplay
    from data.market_data_fetcher import MarketDataFetcher
    from data.volatility_calculator import VolatilityCalculator
    from ml.volatility_predictor import VolatilityPredictor
except ImportError as e:
    print(f"Import warning in dashboard: {e}")

class Dashboard(tk.Frame):
    """Main dashboard for the Volatility Education & Analysis application"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # Initialize components
        self.data_fetcher = MarketDataFetcher()
        self.vol_calculator = VolatilityCalculator()
        self.ml_predictor = VolatilityPredictor()
        
        # Data storage
        self.current_data = None
        self.current_symbol = "AAPL"
        self.volatilities = {}
        
        # Initialize attributes that will be created later
        self.symbol_entry = None
        self.period_selector = None
        self.notebook = None
        self.education_tab = None
        self.analysis_tab = None
        self.prediction_tab = None
        self.options_tab = None
        self.chart_var = None
        self.charts_frame = None
        self.metrics_display = None
        self.alert_panel = None
        self.model_var = None
        self.pred_days_var = None
        self.predictions_frame = None
        self.options_display = None
        self.status_bar = None
        
        # Setup GUI
        self.setup_gui()
        
        # Load initial data
        self.load_initial_data()
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main container
        main_container = tk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create top toolbar
        self.create_toolbar(main_container)
        
        # Create tabbed interface
        self.create_tabs(main_container)
        
        # Create status bar
        self.status_bar = StatusBar(self)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.set_ready()
    
    def create_toolbar(self, parent):
        """Create the top toolbar with controls"""
        toolbar = tk.Frame(parent, relief=tk.RAISED, bd=1)
        toolbar.pack(fill="x", pady=(0, 10))
        
        # Symbol entry
        self.symbol_entry = SymbolEntry(toolbar, default_symbol=self.current_symbol, 
                                       callback=self.on_symbol_changed)
        self.symbol_entry.pack(side="left", padx=5, pady=5)
        
        # Period selector
        self.period_selector = PeriodSelector(toolbar, default_period="90 days",
                                            callback=self.on_period_changed)
        self.period_selector.pack(side="left", padx=5, pady=5)
        
        # Refresh button
        refresh_btn = tk.Button(toolbar, text="🔄 Refresh Data", 
                               command=self.refresh_data,
                               bg="#3498db", fg="white", font=("Helvetica", 10))
        refresh_btn.pack(side="left", padx=10, pady=5)
        
        # Settings button
        settings_btn = tk.Button(toolbar, text="⚙️ Settings", 
                                command=self.show_settings,
                                bg="#95a5a6", fg="white", font=("Helvetica", 10))
        settings_btn.pack(side="right", padx=5, pady=5)
    
    def create_tabs(self, parent):
        """Create the main tabbed interface"""
        # Create notebook widget
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        
        # Tab 1: Analysis Dashboard
        self.analysis_tab = self.create_analysis_tab()
        self.notebook.add(self.analysis_tab, text="📊 Analysis Dashboard")
        
        # Tab 2: ML Predictions
        self.prediction_tab = self.create_prediction_tab()
        self.notebook.add(self.prediction_tab, text="🤖 ML Predictions")
        
        # Tab 3: Options Analysis
        self.options_tab = self.create_options_tab()
        self.notebook.add(self.options_tab, text="📈 Options Analysis")
        
        # Tab 4: Volatility Education (moved to last)
        self.education_tab = VolatilityEducationTab(self.notebook)
        self.notebook.add(self.education_tab, text="📚 Learn Volatility")
    
    def create_analysis_tab(self):
        """Create the main analysis dashboard tab"""
        tab_frame = tk.Frame(self.notebook)
        
        # Create paned window for layout
        paned = tk.PanedWindow(tab_frame, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left panel for charts
        left_panel = tk.Frame(paned, relief=tk.RAISED, bd=1)
        paned.add(left_panel, width=800)
        
        # Right panel for metrics and alerts
        right_panel = tk.Frame(paned, relief=tk.RAISED, bd=1)
        paned.add(right_panel, width=300)
        
        # Setup left panel (charts)
        self.setup_charts_panel(left_panel)
        
        # Setup right panel (metrics)
        self.setup_metrics_panel(right_panel)
        
        return tab_frame
    
    def setup_charts_panel(self, parent):
        """Setup the charts panel"""
        # Chart selection
        chart_frame = tk.Frame(parent)
        chart_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(chart_frame, text="Chart Type:", font=("Helvetica", 10, "bold")).pack(side="left")
        
        self.chart_var = tk.StringVar(value="Price & Volatility")
        chart_combo = ttk.Combobox(chart_frame, textvariable=self.chart_var,
                                  values=["Price & Volatility", "Volatility Only", "Returns Distribution", "Regime Analysis"],
                                  state="readonly", width=20)
        chart_combo.pack(side="left", padx=5)
        chart_combo.bind('<<ComboboxSelected>>', self.update_charts)
        
        # Charts container
        self.charts_frame = tk.Frame(parent)
        self.charts_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initialize with default chart
        self.create_default_chart()
    
    def setup_metrics_panel(self, parent):
        """Setup the metrics and alerts panel"""
        # Metrics display
        self.metrics_display = MetricsDisplay(parent)
        self.metrics_display.pack(fill="x", padx=5, pady=5)
        
        # Alert panel
        self.alert_panel = AlertPanel(parent)
        self.alert_panel.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add some sample alerts
        self.alert_panel.add_alert("Dashboard initialized", "info")
    
    def create_prediction_tab(self):
        """Create the ML predictions tab"""
        tab_frame = tk.Frame(self.notebook)
        
        # Title
        title_label = tk.Label(tab_frame, text="🤖 Machine Learning Volatility Predictions", 
                              font=("Helvetica", 16, "bold"), fg="#2c3e50")
        title_label.pack(pady=10)
        
        # Model selection frame
        model_frame = tk.LabelFrame(tab_frame, text="Model Configuration", 
                                   font=("Helvetica", 12, "bold"))
        model_frame.pack(fill="x", padx=20, pady=10)
        
        # Model selector
        control_frame = tk.Frame(model_frame)
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="ML Model:", font=("Helvetica", 10, "bold")).pack(side="left")
        
        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                  values=["Random Forest", "Gradient Boosting", "Neural Network", "Linear Regression"],
                                  state="readonly", width=15)
        model_combo.pack(side="left", padx=10)
        
        # Train model button
        train_btn = tk.Button(control_frame, text="🔄 Train Model", 
                             command=self.train_ml_model,
                             bg="#27ae60", fg="white")
        train_btn.pack(side="left", padx=10)
        
        # Prediction days
        tk.Label(control_frame, text="Predict Days:", font=("Helvetica", 10, "bold")).pack(side="left", padx=(20, 5))
        
        self.pred_days_var = tk.StringVar(value="5")
        pred_entry = tk.Entry(control_frame, textvariable=self.pred_days_var, width=5)
        pred_entry.pack(side="left")
        
        # Predictions display
        self.predictions_frame = tk.Frame(tab_frame)
        self.predictions_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Initial message
        initial_label = tk.Label(self.predictions_frame, 
                                text="📊 Load data and train a model to see volatility predictions",
                                font=("Helvetica", 12), fg="gray")
        initial_label.pack(expand=True)
        
        return tab_frame
    
    def create_options_tab(self):
        """Create the options analysis tab"""
        tab_frame = tk.Frame(self.notebook)
        
        # Title
        title_label = tk.Label(tab_frame, text="📈 Options & Implied Volatility Analysis", 
                              font=("Helvetica", 16, "bold"), fg="#2c3e50")
        title_label.pack(pady=10)
        
        # Options info frame
        options_frame = tk.LabelFrame(tab_frame, text="Options Data", 
                                     font=("Helvetica", 12, "bold"))
        options_frame.pack(fill="x", padx=20, pady=10)
        
        # Fetch options button
        fetch_options_btn = tk.Button(options_frame, text="📊 Fetch Options Data", 
                                     command=self.fetch_options_data,
                                     bg="#e74c3c", fg="white")
        fetch_options_btn.pack(pady=10)
        
        # Options display
        self.options_display = tk.Text(options_frame, height=15, wrap=tk.WORD,
                                      font=("Courier", 10))
        options_scrollbar = ttk.Scrollbar(options_frame, orient="vertical", 
                                         command=self.options_display.yview)
        self.options_display.configure(yscrollcommand=options_scrollbar.set)
        
        self.options_display.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        options_scrollbar.pack(side="right", fill="y")
        
        # Initial message
        self.options_display.insert("1.0", 
            "📊 OPTIONS ANALYSIS\n\n"
            "Click 'Fetch Options Data' to load current options chain and implied volatility data.\n\n"
            "This will show:\n"
            "• Current implied volatility levels\n"
            "• Options chain for nearest expiration\n"
            "• Put/Call ratios\n"
            "• Volatility skew analysis\n"
            "• Comparison with historical volatility\n\n"
            "💡 TIP: Options data provides insights into market expectations for future volatility!")
        self.options_display.config(state="disabled")
        
        return tab_frame
    
    def create_default_chart(self):
        """Create the default price and volatility chart"""
        # Clear existing charts
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        
        if self.current_data is not None and not self.current_data.empty:
            # Plot price data
            ax1.plot(self.current_data.index, self.current_data['Close'], 
                    linewidth=2, color='blue', label='Price')
            ax1.set_title(f"{self.current_symbol} - Stock Price", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Price ($)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot volatility
            if 'Volatility' in self.current_data.columns:
                ax2.plot(self.current_data.index, self.current_data['Volatility'], 
                        linewidth=2, color='red', label='Historical Volatility')
                ax2.axhline(y=self.current_data['Volatility'].mean(), 
                           color='red', linestyle='--', alpha=0.7, label='Average')
            else:
                # Calculate simple volatility if not available
                returns = self.current_data['Close'].pct_change().dropna()
                vol = returns.rolling(20).std() * np.sqrt(252) * 100
                ax2.plot(vol.index, vol, linewidth=2, color='red', label='20-day Volatility')
            
            ax2.set_title("Historical Volatility (%)", fontsize=12)
            ax2.set_ylabel("Volatility (%)")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            # Show placeholder
            ax1.text(0.5, 0.5, 'No Data Available\n\nEnter a stock symbol and click Refresh', 
                    transform=ax1.transAxes, ha='center', va='center', 
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat'))
            ax1.set_title("Stock Price Chart")
            
            ax2.text(0.5, 0.5, 'Volatility data will appear here', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
            ax2.set_title("Volatility Chart")
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def load_initial_data(self):
        """Load initial data for the default symbol"""
        self.refresh_data()
    
    def on_symbol_changed(self):
        """Handle symbol change"""
        new_symbol = self.symbol_entry.get_symbol()
        if new_symbol != self.current_symbol:
            self.current_symbol = new_symbol
            self.refresh_data()
    
    def on_period_changed(self):
        """Handle period change"""
        self.refresh_data()
    
    def refresh_data(self):
        """Refresh data in background thread"""
        self.status_bar.set_loading(f"Loading data for {self.current_symbol}...")
        
        # Run data fetching in background thread
        thread = threading.Thread(target=self._fetch_data_background)
        thread.daemon = True
        thread.start()
    
    def _fetch_data_background(self):
        """Fetch data in background thread"""
        try:
            # Get period
            period_days = self.period_selector.get_period_days()
            period_map = {
                7: "7d", 14: "14d", 30: "1mo", 60: "2mo", 
                90: "3mo", 180: "6mo", 365: "1y", 730: "2y"
            }
            period = period_map.get(period_days, "3mo")
            
            # Fetch stock data
            data = self.data_fetcher.fetch_historical_data(self.current_symbol, period=period)
            
            # Calculate volatility
            vol_stats = self.vol_calculator.calculate_realized_volatility(data)
            
            # Add volatility column to data
            volatility_series = self.vol_calculator.calculate_close_to_close(data)
            data['Volatility'] = volatility_series
            
            # Update UI in main thread
            self.parent.after(0, self._update_data_display, data)
            
        except Exception as e:
            self.parent.after(0, self._handle_data_error, str(e))
    
    def _update_data_display(self, data):
        """Update the display with new data (called from main thread)"""
        self.current_data = data
        self.status_bar.set_success(f"Data loaded for {self.current_symbol}")
        
        # Update charts
        self.update_charts()
        
        # Update metrics
        self._update_metrics()
        
        # Add alert
        self.alert_panel.add_alert(f"Data refreshed for {self.current_symbol}", "success")
    
    def _handle_data_error(self, error_message):
        """Handle data fetching errors (called from main thread)"""
        self.status_bar.set_error(error_message)
        self.alert_panel.add_alert(f"Failed to load data: {error_message}", "danger")
        messagebox.showerror("Data Error", f"Failed to load data:\n\n{error_message}")
    
    def update_charts(self, event=None):
        """Update charts based on current selection"""
        chart_type = self.chart_var.get()
        
        if chart_type == "Price & Volatility":
            self.create_default_chart()
        elif chart_type == "Volatility Only":
            self.create_volatility_chart()
        elif chart_type == "Returns Distribution":
            self.create_returns_distribution()
        elif chart_type == "Regime Analysis":
            self.create_regime_chart()
    
    def create_volatility_chart(self):
        """Create volatility-focused chart"""
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if self.current_data is not None and 'Volatility' in self.current_data.columns:
            vol_data = self.current_data['Volatility'].dropna()
            
            ax.plot(vol_data.index, vol_data, linewidth=2, color='red', label='Historical Volatility')
            
            # Add volatility percentiles
            p25 = np.percentile(vol_data, 25)
            p75 = np.percentile(vol_data, 75)
            
            ax.axhline(y=p25, color='green', linestyle='--', alpha=0.7, label='25th Percentile')
            ax.axhline(y=p75, color='orange', linestyle='--', alpha=0.7, label='75th Percentile')
            ax.axhline(y=vol_data.mean(), color='blue', linestyle='-', alpha=0.7, label='Average')
            
            ax.set_title(f"{self.current_symbol} - Volatility Analysis", fontsize=14, fontweight='bold')
            ax.set_ylabel("Volatility (%)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_returns_distribution(self):
        """Create returns distribution chart"""
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        if self.current_data is not None:
            returns = self.current_data['Returns'].dropna() * 100  # Convert to percentage
            
            # Histogram
            ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
            ax1.set_title("Daily Returns Distribution")
            ax1.set_xlabel("Returns (%)")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot for normality check
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title("Q-Q Plot (Normality Check)")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_regime_chart(self):
        """Create regime analysis chart"""
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if self.current_data is not None and 'Volatility' in self.current_data.columns:
            vol_data = self.current_data['Volatility'].dropna()
            
            # Simple regime classification based on volatility percentiles
            low_thresh = np.percentile(vol_data, 33)
            high_thresh = np.percentile(vol_data, 67)
            
            colors = []
            for vol in vol_data:
                if vol < low_thresh:
                    colors.append('green')  # Low volatility
                elif vol > high_thresh:
                    colors.append('red')    # High volatility
                else:
                    colors.append('orange') # Medium volatility
            
            ax.scatter(vol_data.index, vol_data, c=colors, alpha=0.6, s=20)
            
            ax.axhline(y=low_thresh, color='green', linestyle='--', alpha=0.7, label='Low Vol Regime')
            ax.axhline(y=high_thresh, color='red', linestyle='--', alpha=0.7, label='High Vol Regime')
            
            ax.set_title(f"{self.current_symbol} - Volatility Regime Analysis", fontsize=14, fontweight='bold')
            ax.set_ylabel("Volatility (%)")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _update_metrics(self):
        """Update the metrics display"""
        if self.current_data is None or self.current_data.empty:
            return
        
        try:
            current_vol = self.current_data['Volatility'].iloc[-1] if 'Volatility' in self.current_data.columns else 0
            vol_data = self.current_data['Volatility'].dropna() if 'Volatility' in self.current_data.columns else []
            
            if len(vol_data) > 0:
                vol_percentile = (vol_data <= current_vol).mean() * 100
                vol_regime = "High" if vol_percentile > 75 else "Medium" if vol_percentile > 25 else "Low"
            else:
                vol_percentile = 0
                vol_regime = "Unknown"
            
            metrics = {
                "Current Vol": f"{current_vol:.1f}%",
                "Vol Percentile": f"{vol_percentile:.0f}%",
                "Vol Regime": vol_regime,
                "Data Points": len(self.current_data)
            }
            
            self.metrics_display.update_all_metrics(metrics)
            
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def train_ml_model(self):
        """Train the ML model for volatility prediction"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning("No Data", "Please load data first before training the model.")
            return
        
        self.status_bar.set_loading("Training ML model...")
        
        # Run training in background
        thread = threading.Thread(target=self._train_model_background)
        thread.daemon = True
        thread.start()
    
    def _train_model_background(self):
        """Train model in background thread"""
        try:
            prices = self.current_data['Close'].values
            volatilities = self.current_data['Volatility'].values if 'Volatility' in self.current_data.columns else None
            
            if volatilities is None:
                raise ValueError("Volatility data not available")
            
            # Map model names
            model_map = {
                "Random Forest": "random_forest",
                "Gradient Boosting": "gradient_boost", 
                "Neural Network": "neural_network",
                "Linear Regression": "linear_regression"
            }
            
            model_type = model_map[self.model_var.get()]
            
            # Train model
            results = self.ml_predictor.train_model(prices, volatilities, model_type=model_type)
            
            # Update UI in main thread
            self.parent.after(0, self._update_prediction_display, results)
            
        except Exception as e:
            self.parent.after(0, self._handle_training_error, str(e))
    
    def _update_prediction_display(self, results):
        """Update prediction display with training results"""
        self.status_bar.set_success("Model trained successfully")
        
        # Clear predictions frame
        for widget in self.predictions_frame.winfo_children():
            widget.destroy()
        
        # Create results display
        results_frame = tk.LabelFrame(self.predictions_frame, text="Training Results", 
                                     font=("Helvetica", 12, "bold"))
        results_frame.pack(fill="x", padx=10, pady=10)
        
        # Model metrics
        metrics_text = tk.Text(results_frame, height=8, wrap=tk.WORD, font=("Courier", 10))
        metrics_text.pack(fill="x", padx=10, pady=10)
        
        metrics_content = f"""MODEL TRAINING RESULTS
{'='*50}
Model Type: {results['model_type']}
Training R²: {results['train_r2']:.4f}
Test R²: {results['test_r2']:.4f}
Training RMSE: {results['train_mse']**0.5:.4f}
Test RMSE: {results['test_mse']**0.5:.4f}
Cross-Validation Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}

Training Samples: {results['train_samples']}
Test Samples: {results['test_samples']}
Features Used: {results['n_features']}
"""
        
        metrics_text.insert("1.0", metrics_content)
        metrics_text.config(state="disabled")
        
        # Prediction button
        predict_btn = tk.Button(results_frame, text="🔮 Generate Predictions", 
                               command=self.generate_predictions,
                               bg="#9b59b6", fg="white", font=("Helvetica", 11))
        predict_btn.pack(pady=10)
        
        self.alert_panel.add_alert(f"ML model trained: {results['model_type']}", "success")
    
    def _handle_training_error(self, error_message):
        """Handle training errors"""
        self.status_bar.set_error(f"Training failed: {error_message}")
        self.alert_panel.add_alert(f"Model training failed: {error_message}", "danger")
        messagebox.showerror("Training Error", f"Failed to train model:\n\n{error_message}")
    
    def generate_predictions(self):
        """Generate volatility predictions"""
        if not self.ml_predictor.is_trained:
            messagebox.showwarning("Model Not Trained", "Please train a model first.")
            return
        
        try:
            pred_days = int(self.pred_days_var.get())
            if pred_days < 1 or pred_days > 30:
                raise ValueError("Prediction days must be between 1 and 30")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of days (1-30)")
            return
        
        self.status_bar.set_loading("Generating predictions...")
        
        try:
            prices = self.current_data['Close'].values
            volatilities = self.current_data['Volatility'].values
            
            predictions = self.ml_predictor.predict_volatility(prices, volatilities, days_ahead=pred_days)
            
            self._display_predictions(predictions, pred_days)
            self.status_bar.set_success("Predictions generated")
            
        except Exception as e:
            self.status_bar.set_error(f"Prediction failed: {str(e)}")
            messagebox.showerror("Prediction Error", f"Failed to generate predictions:\n\n{str(e)}")
    
    def _display_predictions(self, predictions, days):
        """Display the generated predictions"""
        # Find existing predictions frame or create new one
        pred_display_frame = None
        for widget in self.predictions_frame.winfo_children():
            if hasattr(widget, 'pred_display_marker'):
                pred_display_frame = widget
                break
        
        if pred_display_frame is None:
            pred_display_frame = tk.LabelFrame(self.predictions_frame, text="Volatility Predictions", 
                                              font=("Helvetica", 12, "bold"))
            pred_display_frame.pred_display_marker = True
            pred_display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Clear existing content
        for widget in pred_display_frame.winfo_children():
            widget.destroy()
        
        # Create prediction display
        pred_text = tk.Text(pred_display_frame, height=10, wrap=tk.WORD, font=("Courier", 10))
        pred_scrollbar = ttk.Scrollbar(pred_display_frame, orient="vertical", command=pred_text.yview)
        pred_text.configure(yscrollcommand=pred_scrollbar.set)
        
        pred_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        pred_scrollbar.pack(side="right", fill="y")
        
        # Format predictions
        current_vol = self.current_data['Volatility'].iloc[-1]
        pred_content = f"VOLATILITY PREDICTIONS FOR {self.current_symbol}\n"
        pred_content += "="*60 + "\n\n"
        pred_content += f"Current Volatility: {current_vol:.2f}%\n\n"
        pred_content += "Predicted Volatility:\n"
        pred_content += "-" * 30 + "\n"
        
        for i, pred in enumerate(predictions, 1):
            change = pred - current_vol
            change_pct = (change / current_vol) * 100 if current_vol != 0 else 0
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            
            pred_content += f"Day +{i}: {pred:.2f}% {direction} ({change:+.2f}%, {change_pct:+.1f}%)\n"
        
        pred_content += "\n" + "="*60 + "\n"
        pred_content += "💡 These predictions are based on historical patterns and should be used alongside other analysis methods."
        
        pred_text.insert("1.0", pred_content)
        pred_text.config(state="disabled")
        
        self.alert_panel.add_alert(f"Generated {days}-day volatility forecast", "info")
    
    def fetch_options_data(self):
        """Fetch and display options data"""
        self.status_bar.set_loading("Fetching options data...")
        
        thread = threading.Thread(target=self._fetch_options_background)
        thread.daemon = True
        thread.start()
    
    def _fetch_options_background(self):
        """Fetch options data in background"""
        try:
            options_data = self.data_fetcher.fetch_options_data(self.current_symbol)
            self.parent.after(0, self._display_options_data, options_data)
        except Exception as e:
            self.parent.after(0, self._handle_options_error, str(e))
    
    def _display_options_data(self, options_data):
        """Display options data"""
        self.status_bar.set_success("Options data loaded")
        
        self.options_display.config(state="normal")
        self.options_display.delete("1.0", tk.END)
        
        content = f"OPTIONS ANALYSIS FOR {options_data['symbol']}\n"
        content += "="*60 + "\n\n"
        content += f"Current Stock Price: ${options_data['current_price']:.2f}\n"
        content += f"Expiration Date: {options_data['expiration']}\n\n"
        
        if options_data['atm_call']:
            call = options_data['atm_call']
            content += f"AT-THE-MONEY CALL OPTION:\n"
            content += f"Strike: ${call.get('strike', 0):.2f}\n"
            content += f"Last Price: ${call.get('lastPrice', 0):.2f}\n"
            content += f"Implied Volatility: {call.get('impliedVolatility', 0)*100:.1f}%\n"
            content += f"Volume: {call.get('volume', 0)}\n\n"
        
        if options_data['atm_put']:
            put = options_data['atm_put']
            content += f"AT-THE-MONEY PUT OPTION:\n"
            content += f"Strike: ${put.get('strike', 0):.2f}\n"
            content += f"Last Price: ${put.get('lastPrice', 0):.2f}\n"
            content += f"Implied Volatility: {put.get('impliedVolatility', 0)*100:.1f}%\n"
            content += f"Volume: {put.get('volume', 0)}\n\n"
        
        content += "="*60 + "\n"
        content += "💡 Implied volatility represents market expectations for future price movement."
        
        self.options_display.insert("1.0", content)
        self.options_display.config(state="disabled")
        
        self.alert_panel.add_alert("Options data updated", "success")
    
    def _handle_options_error(self, error_message):
        """Handle options data errors"""
        self.status_bar.set_error(f"Options fetch failed: {error_message}")
        self.alert_panel.add_alert(f"Options data error: {error_message}", "danger")
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("⚙️ Dashboard Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.parent)
        settings_window.grab_set()
        
        # Center the window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (settings_window.winfo_width() // 2)
        y = (settings_window.winfo_screenheight() // 2) - (settings_window.winfo_height() // 2)
        settings_window.geometry(f"+{x}+{y}")
        
        # Settings content
        tk.Label(settings_window, text="⚙️ Dashboard Settings", 
                font=("Helvetica", 16, "bold")).pack(pady=20)
        
        # Cache settings
        cache_frame = tk.LabelFrame(settings_window, text="Data Cache", font=("Helvetica", 12, "bold"))
        cache_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Button(cache_frame, text="Clear Cache", command=self.clear_cache,
                 bg="#e74c3c", fg="white").pack(pady=10)
        
        # About section
        about_frame = tk.LabelFrame(settings_window, text="About", font=("Helvetica", 12, "bold"))
        about_frame.pack(fill="x", padx=20, pady=10)
        
        about_text = """Volatility Education & Analysis Dashboard
Version 1.0.0

A comprehensive tool for learning and analyzing market volatility with machine learning predictions."""
        
        tk.Label(about_frame, text=about_text, justify="left", wraplength=350).pack(pady=10)
        
        # Close button
        tk.Button(settings_window, text="Close", command=settings_window.destroy,
                 bg="#95a5a6", fg="white").pack(pady=20)
    
    def clear_cache(self):
        """Clear data cache"""
        self.data_fetcher.clear_cache()
        self.alert_panel.add_alert("Data cache cleared", "info")
        messagebox.showinfo("Cache Cleared", "Data cache has been cleared successfully.")