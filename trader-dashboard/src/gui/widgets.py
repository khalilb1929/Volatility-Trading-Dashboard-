import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import re

class StatusBar(tk.Frame):
    """Status bar widget for displaying application status"""
    
    def __init__(self, parent):
        super().__init__(parent, relief=tk.SUNKEN, bd=1)
        
        # Status label
        self.status_label = tk.Label(self, text="Ready", anchor="w", 
                                    font=("Helvetica", 9))
        self.status_label.pack(side="left", fill="x", expand=True, padx=5)
        
        # Time label
        self.time_label = tk.Label(self, text="", anchor="e", 
                                  font=("Helvetica", 9))
        self.time_label.pack(side="right", padx=5)
        
        # Update time every second
        self._update_job = None
        self.update_time()
    
    def update_time(self):
        """Update the time display"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            self.time_label.config(text=current_time)
            # Schedule next update
            self._update_job = self.after(1000, self.update_time)
        except tk.TclError:
            # Widget has been destroyed, stop updating
            pass
    
    def destroy(self):
        """Clean up scheduled updates"""
        if self._update_job:
            self.after_cancel(self._update_job)
        super().destroy()
    
    def set_status(self, message, status_type="info"):
        """Set status message with color coding"""
        colors = {
            "info": "#2c3e50",
            "success": "#27ae60", 
            "warning": "#f39c12",
            "error": "#e74c3c",
            "loading": "#3498db"
        }
        
        color = colors.get(status_type, "#2c3e50")
        self.status_label.config(text=message, fg=color)
    
    def set_ready(self):
        """Set ready status"""
        self.set_status("Ready", "info")
    
    def set_loading(self, message="Loading..."):
        """Set loading status"""
        self.set_status(f"⏳ {message}", "loading")
    
    def set_success(self, message):
        """Set success status"""
        self.set_status(f"✅ {message}", "success")
    
    def set_error(self, message):
        """Set error status"""
        self.set_status(f"❌ {message}", "error")
    
    def set_warning(self, message):
        """Set warning status"""
        self.set_status(f"⚠️ {message}", "warning")


class SymbolEntry(tk.Frame):
    """Widget for entering stock symbols with validation"""
    
    def __init__(self, parent, default_symbol="AAPL", callback=None):
        super().__init__(parent)
        
        self.callback = callback
        
        # Label
        tk.Label(self, text="Stock Symbol:", font=("Helvetica", 10, "bold")).pack(side="left")
        
        # Entry with validation
        self.symbol_var = tk.StringVar(value=default_symbol)
        self.entry = tk.Entry(self, textvariable=self.symbol_var, 
                             width=8, font=("Helvetica", 10))
        self.entry.pack(side="left", padx=5)
        
        # Bind events
        self.entry.bind('<Return>', self.on_enter)
        self.entry.bind('<FocusOut>', self.on_focus_out)
        self.symbol_var.trace('w', self.on_symbol_change)
        
        # Validation status
        self.status_label = tk.Label(self, text="", font=("Helvetica", 8))
        self.status_label.pack(side="left", padx=5)
        
        # Track last valid symbol
        self.last_valid_symbol = default_symbol
    
    def on_enter(self, event):
        """Handle Enter key press"""
        self.validate_and_callback()
    
    def on_focus_out(self, event):
        """Handle focus out event"""
        self.validate_and_callback()
    
    def on_symbol_change(self, *args):
        """Handle symbol text change"""
        symbol = self.symbol_var.get().upper()
        
        # Auto-uppercase
        if symbol != self.symbol_var.get():
            self.symbol_var.set(symbol)
            return
        
        # Basic validation
        if not symbol:
            self.status_label.config(text="", fg="black")
            return
        
        # Check if symbol contains only valid characters
        if re.match(r'^[A-Z]{1,5}$', symbol):
            self.status_label.config(text="✓", fg="green")
        elif len(symbol) > 5:
            self.status_label.config(text="Too long", fg="red")
        elif not re.match(r'^[A-Z]*$', symbol):
            self.status_label.config(text="Invalid chars", fg="red")
        else:
            self.status_label.config(text="", fg="black")
    
    def validate_and_callback(self):
        """Validate symbol and trigger callback"""
        symbol = self.symbol_var.get().upper().strip()
        
        if not symbol:
            return
        
        # Basic validation
        if re.match(r'^[A-Z]{1,5}$', symbol):
            if symbol != self.last_valid_symbol:
                self.last_valid_symbol = symbol
                if self.callback:
                    self.callback()
        else:
            # Reset to last valid symbol
            self.symbol_var.set(self.last_valid_symbol)
    
    def get_symbol(self):
        """Get the current valid symbol"""
        return self.last_valid_symbol
    
    def set_symbol(self, symbol):
        """Set the symbol programmatically"""
        symbol = symbol.upper().strip()
        if re.match(r'^[A-Z]{1,5}$', symbol):
            self.symbol_var.set(symbol)
            self.last_valid_symbol = symbol


class PeriodSelector(tk.Frame):
    """Widget for selecting time periods"""
    
    def __init__(self, parent, default_period="90 days", callback=None):
        super().__init__(parent)
        
        self.callback = callback
        
        # Label
        tk.Label(self, text="Period:", font=("Helvetica", 10, "bold")).pack(side="left")
        
        # Period options
        self.periods = [
            ("1 week", 7),
            ("2 weeks", 14), 
            ("1 month", 30),
            ("2 months", 60),
            ("3 months", 90),
            ("6 months", 180),
            ("1 year", 365),
            ("2 years", 730)
        ]
        
        # Combobox
        self.period_var = tk.StringVar(value=default_period)
        self.combo = ttk.Combobox(self, textvariable=self.period_var,
                                 values=[p[0] for p in self.periods],
                                 state="readonly", width=12)
        self.combo.pack(side="left", padx=5)
        
        # Bind selection event
        self.combo.bind('<<ComboboxSelected>>', self.on_period_change)
    
    def on_period_change(self, event):
        """Handle period selection change"""
        if self.callback:
            self.callback()
    
    def get_period_days(self):
        """Get the selected period in days"""
        selected = self.period_var.get()
        for name, days in self.periods:
            if name == selected:
                return days
        return 90  # Default
    
    def get_period_name(self):
        """Get the selected period name"""
        return self.period_var.get()
    
    def set_period(self, period_name):
        """Set the period programmatically"""
        valid_periods = [p[0] for p in self.periods]
        if period_name in valid_periods:
            self.period_var.set(period_name)


class ModelSelector(tk.Frame):
    """Widget for selecting ML models"""
    
    def __init__(self, parent, callback=None):
        super().__init__(parent)
        
        self.callback = callback
        
        # Label
        tk.Label(self, text="ML Model:", font=("Helvetica", 10, "bold")).pack(side="left")
        
        # Model options
        self.models = [
            "Random Forest",
            "Gradient Boosting", 
            "Neural Network",
            "Linear Regression",
            "Support Vector Machine"
        ]
        
        # Combobox
        self.model_var = tk.StringVar(value=self.models[0])
        self.combo = ttk.Combobox(self, textvariable=self.model_var,
                                 values=self.models, state="readonly", width=18)
        self.combo.pack(side="left", padx=5)
        
        # Bind selection event
        self.combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Info button
        self.info_btn = tk.Button(self, text="ℹ️", command=self.show_model_info,
                                 font=("Helvetica", 8), width=2)
        self.info_btn.pack(side="left", padx=2)
    
    def on_model_change(self, event):
        """Handle model selection change"""
        if self.callback:
            self.callback()
    
    def get_selected_model(self):
        """Get the selected model"""
        return self.model_var.get()
    
    def set_model(self, model_name):
        """Set the model programmatically"""
        if model_name in self.models:
            self.model_var.set(model_name)
    
    def show_model_info(self):
        """Show information about the selected model"""
        model = self.model_var.get()
        
        info_texts = {
            "Random Forest": """Random Forest Model:
• Ensemble of decision trees
• Good for non-linear relationships  
• Handles missing data well
• Less prone to overfitting
• Good baseline model choice""",
            
            "Gradient Boosting": """Gradient Boosting Model:
• Sequential tree building
• Each tree corrects previous errors
• Often highest accuracy
• Can overfit if not tuned properly
• Slower to train""",
            
            "Neural Network": """Neural Network Model:
• Deep learning approach
• Can capture complex patterns
• Requires more data
• Longer training time
• Good for time series""",
            
            "Linear Regression": """Linear Regression Model:
• Simple and interpretable
• Fast training and prediction
• Good baseline model
• Assumes linear relationships
• May underfit complex data""",
            
            "Support Vector Machine": """Support Vector Machine:
• Finds optimal decision boundary
• Good for high-dimensional data
• Robust to outliers
• Can be slow with large datasets
• Good generalization"""
        }
        
        info_text = info_texts.get(model, "No information available")
        
        # Create info window
        info_window = tk.Toplevel(self)
        info_window.title(f"Model Info: {model}")
        info_window.geometry("350x200")
        info_window.transient(self.winfo_toplevel())
        info_window.grab_set()
        
        # Center the window
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (info_window.winfo_width() // 2)
        y = (info_window.winfo_screenheight() // 2) - (info_window.winfo_height() // 2)
        info_window.geometry(f"+{x}+{y}")
        
        # Add text
        text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10,
                             font=("Helvetica", 10))
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", info_text)
        text_widget.config(state="disabled")
        
        # Close button
        tk.Button(info_window, text="Close", command=info_window.destroy,
                 bg="#95a5a6", fg="white").pack(pady=5)


class AlertPanel(tk.Frame):
    """Panel for displaying alerts and notifications"""
    
    def __init__(self, parent):
        super().__init__(parent, relief=tk.RAISED, bd=1)
        
        # Title
        title_frame = tk.Frame(self, bg="#34495e")
        title_frame.pack(fill="x")
        
        tk.Label(title_frame, text="📢 Alerts", font=("Helvetica", 12, "bold"),
                bg="#34495e", fg="white").pack(pady=5)
        
        # Scrollable alerts area
        self.setup_scrollable_area()
        
        # Clear button
        clear_btn = tk.Button(self, text="Clear All", command=self.clear_alerts,
                             bg="#e74c3c", fg="white", font=("Helvetica", 9))
        clear_btn.pack(pady=5)
        
        # Store alerts
        self.alerts = []
        self.max_alerts = 50
    
    def setup_scrollable_area(self):
        """Setup scrollable area for alerts"""
        # Create frame for scrollbar and listbox
        scroll_frame = tk.Frame(self)
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Listbox for alerts
        self.alerts_listbox = tk.Listbox(scroll_frame, yscrollcommand=scrollbar.set,
                                        font=("Helvetica", 9), height=15)
        self.alerts_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar.config(command=self.alerts_listbox.yview)
    
    def add_alert(self, message, alert_type="info"):
        """Add a new alert"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format message with icon
        icons = {
            "info": "ℹ️",
            "success": "✅", 
            "warning": "⚠️",
            "danger": "❌",
            "loading": "⏳"
        }
        
        icon = icons.get(alert_type, "ℹ️")
        formatted_message = f"{timestamp} {icon} {message}"
        
        # Add to listbox
        self.alerts_listbox.insert(0, formatted_message)
        
        # Color coding
        colors = {
            "info": "#3498db",
            "success": "#27ae60",
            "warning": "#f39c12", 
            "danger": "#e74c3c",
            "loading": "#9b59b6"
        }
        
        color = colors.get(alert_type, "#3498db")
        self.alerts_listbox.itemconfig(0, {'fg': color})
        
        # Store alert
        self.alerts.append({
            'message': message,
            'type': alert_type,
            'timestamp': datetime.now(),
            'formatted': formatted_message
        })
        
        # Limit number of alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[:self.max_alerts]
            self.refresh_display()
        
        # Auto-scroll to top
        self.alerts_listbox.see(0)
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts_listbox.delete(0, tk.END)
        self.alerts = []
    
    def refresh_display(self):
        """Refresh the alerts display"""
        self.alerts_listbox.delete(0, tk.END)
        
        for alert in self.alerts:
            self.alerts_listbox.insert(tk.END, alert['formatted'])
            
            colors = {
                "info": "#3498db",
                "success": "#27ae60", 
                "warning": "#f39c12",
                "danger": "#e74c3c",
                "loading": "#9b59b6"
            }
            
            color = colors.get(alert['type'], "#3498db")
            self.alerts_listbox.itemconfig(tk.END, {'fg': color})
    
    def get_alerts_by_type(self, alert_type):
        """Get alerts of a specific type"""
        return [alert for alert in self.alerts if alert['type'] == alert_type]
    
    def get_recent_alerts(self, minutes=60):
        """Get alerts from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]


class MetricsDisplay(tk.Frame):
    """Widget for displaying key metrics"""
    
    def __init__(self, parent):
        super().__init__(parent, relief=tk.RAISED, bd=1)
        
        # Title
        title_frame = tk.Frame(self, bg="#2c3e50")
        title_frame.pack(fill="x")
        
        tk.Label(title_frame, text="📊 Key Metrics", font=("Helvetica", 12, "bold"),
                bg="#2c3e50", fg="white").pack(pady=5)
        
        # Metrics container
        self.metrics_frame = tk.Frame(self, bg="white")
        self.metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Store metric widgets
        self.metric_widgets = {}
        
        # Initialize with default metrics
        self.setup_default_metrics()
    
    def setup_default_metrics(self):
        """Setup default metrics display"""
        default_metrics = {
            "Current Vol": "-- %",
            "Vol Percentile": "-- %", 
            "Vol Regime": "--",
            "Data Points": "--"
        }
        
        for i, (name, value) in enumerate(default_metrics.items()):
            self.add_metric(name, value, row=i)
    
    def add_metric(self, name, value, row=None, color="black"):
        """Add a metric to the display"""
        if row is None:
            row = len(self.metric_widgets)
        
        # Create frame for this metric
        metric_frame = tk.Frame(self.metrics_frame, bg="white")
        metric_frame.grid(row=row, column=0, sticky="ew", pady=2)
        self.metrics_frame.grid_columnconfigure(0, weight=1)
        
        # Metric name
        name_label = tk.Label(metric_frame, text=f"{name}:", 
                             font=("Helvetica", 10), bg="white", anchor="w")
        name_label.pack(side="left")
        
        # Metric value
        value_label = tk.Label(metric_frame, text=str(value),
                              font=("Helvetica", 10, "bold"), bg="white", 
                              anchor="e", fg=color)
        value_label.pack(side="right")
        
        # Store references
        self.metric_widgets[name] = {
            'frame': metric_frame,
            'name_label': name_label,
            'value_label': value_label
        }
    
    def update_metric(self, name, value, color="black"):
        """Update a specific metric"""
        if name in self.metric_widgets:
            self.metric_widgets[name]['value_label'].config(text=str(value), fg=color)
    
    def update_all_metrics(self, metrics_dict):
        """Update multiple metrics at once"""
        for name, value in metrics_dict.items():
            if name in self.metric_widgets:
                # Determine color based on metric name and value
                color = self.get_metric_color(name, value)
                self.update_metric(name, value, color)
    
    def get_metric_color(self, name, value):
        """Get appropriate color for a metric based on its value"""
        try:
            # For volatility percentile
            if "percentile" in name.lower():
                num_value = float(str(value).replace('%', ''))
                if num_value > 75:
                    return "#e74c3c"  # Red for high
                elif num_value < 25:
                    return "#27ae60"  # Green for low
                else:
                    return "#f39c12"  # Orange for medium
            
            # For volatility regime
            elif "regime" in name.lower():
                if "high" in str(value).lower():
                    return "#e74c3c"
                elif "low" in str(value).lower():
                    return "#27ae60"
                else:
                    return "#f39c12"
            
            # For current volatility
            elif "vol" in name.lower() and "%" in str(value):
                num_value = float(str(value).replace('%', ''))
                if num_value > 40:
                    return "#e74c3c"  # Red for very high vol
                elif num_value < 15:
                    return "#27ae60"  # Green for low vol
                else:
                    return "#3498db"  # Blue for normal
            
        except (ValueError, AttributeError):
            pass
        
        return "black"  # Default color
    
    def remove_metric(self, name):
        """Remove a metric from the display"""
        if name in self.metric_widgets:
            self.metric_widgets[name]['frame'].destroy()
            del self.metric_widgets[name]
    
    def clear_metrics(self):
        """Clear all metrics"""
        for name in list(self.metric_widgets.keys()):
            self.remove_metric(name)
    
    def get_metric_value(self, name):
        """Get the current value of a metric"""
        if name in self.metric_widgets:
            return self.metric_widgets[name]['value_label'].cget('text')
        return None


class ProgressBar(tk.Frame):
    """Custom progress bar widget"""
    
    def __init__(self, parent, width=200, height=20):
        super().__init__(parent)
        
        self.width = width
        self.height = height
        
        # Create canvas for custom drawing
        self.canvas = tk.Canvas(self, width=width, height=height, 
                               bg="white", highlightthickness=1,
                               highlightbackground="gray")
        self.canvas.pack()
        
        # Progress variables
        self.progress = 0.0  # 0.0 to 1.0
        self.text = ""
        
        # Draw initial state
        self.update_display()
    
    def set_progress(self, progress, text=""):
        """Set progress value (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
        self.text = text
        self.update_display()
    
    def update_display(self):
        """Update the visual display"""
        self.canvas.delete("all")
        
        # Background
        self.canvas.create_rectangle(0, 0, self.width, self.height,
                                   fill="#ecf0f1", outline="#bdc3c7")
        
        # Progress bar
        if self.progress > 0:
            prog_width = int(self.width * self.progress)
            color = self.get_progress_color()
            self.canvas.create_rectangle(1, 1, prog_width, self.height-1,
                                       fill=color, outline="")
        
        # Text
        if self.text:
            self.canvas.create_text(self.width//2, self.height//2,
                                  text=self.text, font=("Helvetica", 8))
    
    def get_progress_color(self):
        """Get color based on progress value"""
        if self.progress < 0.3:
            return "#e74c3c"  # Red
        elif self.progress < 0.7:
            return "#f39c12"  # Orange
        else:
            return "#27ae60"  # Green


class DataTable(tk.Frame):
    """Simple data table widget"""
    
    def __init__(self, parent, columns, height=10):
        super().__init__(parent)
        
        self.columns = columns
        
        # Create treeview
        self.tree = ttk.Treeview(self, columns=columns, show='headings', height=height)
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack widgets
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def insert_row(self, values):
        """Insert a row of data"""
        self.tree.insert("", "end", values=values)
    
    def clear_data(self):
        """Clear all data"""
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def set_data(self, data_rows):
        """Set all data at once"""
        self.clear_data()
        for row in data_rows:
            self.insert_row(row)
    
    def get_selected_row(self):
        """Get the selected row data"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            return self.tree.item(item)['values']
        return None


class TooltipWidget:
    """Tooltip widget for showing help text on hover"""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        
        # Bind events
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
    
    def on_enter(self, event):
        """Show tooltip on mouse enter"""
        self.show_tooltip()
    
    def on_leave(self, event):
        """Hide tooltip on mouse leave"""
        self.hide_tooltip()
    
    def show_tooltip(self):
        """Display the tooltip"""
        if self.tooltip_window:
            return
        
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip_window, text=self.text,
                        background="#ffffe0", relief="solid", borderwidth=1,
                        font=("Helvetica", 9))
        label.pack()
    
    def hide_tooltip(self):
        """Hide the tooltip"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def add_tooltip(widget, text):
    """Helper function to add tooltip to any widget"""
    return TooltipWidget(widget, text)