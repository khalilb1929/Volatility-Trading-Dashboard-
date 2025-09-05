import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

class VolatilityEducationTab(tk.Frame):
    """Tab version for the main dashboard"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # Create button to open education window
        main_frame = tk.Frame(self, bg="white")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = tk.Label(main_frame, text="📚 Volatility Education Center",
                        font=("Helvetica", 16, "bold"), bg="white")
        title.pack(pady=20)
        
        description = tk.Label(main_frame,
                              text="Learn about volatility concepts, calculations, and interpretation.\n"
                                   "Perfect for beginners and advanced traders alike.",
                              font=("Helvetica", 12), bg="white", wraplength=400)
        description.pack(pady=10)
        
        open_btn = tk.Button(main_frame, text="🚀 Open Education Center",
                            command=self.open_education_window,
                            font=("Helvetica", 12, "bold"),
                            bg="#3498db", fg="white", padx=30, pady=10)
        open_btn.pack(pady=20)
        
        # Quick info
        info_frame = tk.Frame(main_frame, bg="#f8f9fa", relief=tk.RAISED, bd=1)
        info_frame.pack(fill="x", pady=20, padx=20)
        
        info_title = tk.Label(info_frame, text="What You'll Learn:",
                             font=("Helvetica", 12, "bold"), bg="#f8f9fa")
        info_title.pack(pady=10)
        
        topics = [
            "• Basic volatility concepts and definitions",
            "• Different calculation methods (Close-to-Close, Garman-Klass, etc.)",
            "• How to interpret volatility levels",
            "• Real market examples and scenarios",
            "• Interactive quiz to test your knowledge"
        ]
        
        for topic in topics:
            topic_label = tk.Label(info_frame, text=topic,
                                  font=("Helvetica", 10), bg="#f8f9fa", anchor="w")
            topic_label.pack(fill="x", padx=20, pady=2)
        
        tk.Label(info_frame, text="", bg="#f8f9fa").pack(pady=5)  # Spacer
    
    def open_education_window(self):
        """Open the education window"""
        VolatilityEducationWindow(self.parent)

class VolatilityEducationWindow:
    """Educational window about volatility concepts and calculations"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Volatility Education Center")
        self.window.geometry("1000x700")
        self.window.configure(bg="#2c3e50")
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Center the window
        self.center_window()
        
        # Setup the interface
        self.setup_interface()
        
        # Educational content
        self.content = self.get_educational_content()
        
        # Load initial content
        self.load_basics()
    
    def center_window(self):
        """Center the window on screen"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"+{x}+{y}")
    
    def setup_interface(self):
        """Setup the user interface"""
        # Main title
        title_frame = tk.Frame(self.window, bg="#34495e", height=60)
        title_frame.pack(fill="x", padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="📚 Volatility Education Center", 
                font=("Helvetica", 18, "bold"), bg="#34495e", fg="white").pack(pady=15)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_basics_tab()
        self.create_calculation_tab()
        self.create_interpretation_tab()
        self.create_examples_tab()
        self.create_quiz_tab()
        
        # Close button
        close_frame = tk.Frame(self.window, bg="#2c3e50")
        close_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(close_frame, text="Close", command=self.window.destroy,
                 bg="#95a5a6", fg="white", font=("Helvetica", 12),
                 padx=20, pady=5).pack(side="right")
    
    def create_basics_tab(self):
        """Create the basics tab"""
        self.basics_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.basics_frame, text="📖 Basics")
        
        # Scrollable text area
        self.basics_text = scrolledtext.ScrolledText(
            self.basics_frame, wrap=tk.WORD, font=("Helvetica", 11),
            bg="white", fg="black", padx=20, pady=20
        )
        self.basics_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_calculation_tab(self):
        """Create the calculation methods tab"""
        self.calc_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.calc_frame, text="🧮 Calculations")
        
        # Split into text and formula sections
        paned = tk.PanedWindow(self.calc_frame, orient=tk.HORIZONTAL, bg="white")
        paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Text section
        text_frame = tk.Frame(paned, bg="white")
        paned.add(text_frame, width=500)
        
        self.calc_text = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, font=("Helvetica", 10),
            bg="white", fg="black"
        )
        self.calc_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Formula/visualization section
        viz_frame = tk.Frame(paned, bg="white")
        paned.add(viz_frame, width=400)
        
        # Method selector
        method_frame = tk.Frame(viz_frame, bg="white")
        method_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(method_frame, text="Calculation Method:", 
                font=("Helvetica", 10, "bold"), bg="white").pack(anchor="w")
        
        self.calc_method = tk.StringVar(value="Close-to-Close")
        methods = ["Close-to-Close", "Garman-Klass", "Rogers-Satchell", "Yang-Zhang"]
        
        for method in methods:
            tk.Radiobutton(method_frame, text=method, variable=self.calc_method, 
                          value=method, bg="white", command=self.show_calculation_method).pack(anchor="w")
        
        # Formula display
        self.formula_text = tk.Text(viz_frame, height=15, font=("Courier", 9),
                                   bg="#f8f9fa", fg="black", wrap=tk.WORD)
        self.formula_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_interpretation_tab(self):
        """Create the interpretation tab"""
        self.interp_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.interp_frame, text="📊 Interpretation")
        
        # Create sections
        sections_frame = tk.Frame(self.interp_frame, bg="white")
        sections_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Section selector
        selector_frame = tk.Frame(sections_frame, bg="white")
        selector_frame.pack(fill="x", pady=5)
        
        tk.Label(selector_frame, text="Topic:", font=("Helvetica", 10, "bold"), 
                bg="white").pack(side="left")
        
        self.interp_topic = tk.StringVar(value="Volatility Levels")
        topics = ["Volatility Levels", "Regime Detection", "Market Conditions", "Trading Implications"]
        
        topic_combo = ttk.Combobox(selector_frame, textvariable=self.interp_topic,
                                  values=topics, state="readonly", width=20)
        topic_combo.pack(side="left", padx=10)
        topic_combo.bind('<<ComboboxSelected>>', self.load_interpretation_topic)
        
        # Content area
        self.interp_text = scrolledtext.ScrolledText(
            sections_frame, wrap=tk.WORD, font=("Helvetica", 11),
            bg="white", fg="black", height=25
        )
        self.interp_text.pack(fill="both", expand=True, pady=10)
    
    def create_examples_tab(self):
        """Create the interactive examples tab"""
        self.examples_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.examples_frame, text="📈 Examples")
        
        # Control panel
        control_frame = tk.Frame(self.examples_frame, bg="#ecf0f1", height=100)
        control_frame.pack(fill="x", padx=10, pady=5)
        control_frame.pack_propagate(False)
        
        # Example selector
        tk.Label(control_frame, text="Example:", font=("Helvetica", 10, "bold"),
                bg="#ecf0f1").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.example_type = tk.StringVar(value="Low Volatility Period")
        examples = ["Low Volatility Period", "High Volatility Period", "Volatility Spike", "Volatility Clustering"]
        
        example_combo = ttk.Combobox(control_frame, textvariable=self.example_type,
                                    values=examples, state="readonly", width=20)
        example_combo.grid(row=0, column=1, padx=5, pady=5)
        example_combo.bind('<<ComboboxSelected>>', self.generate_example)
        
        # Generate button
        tk.Button(control_frame, text="Generate Example", command=self.generate_example,
                 bg="#3498db", fg="white", font=("Helvetica", 10)).grid(row=0, column=2, padx=10, pady=5)
        
        # Chart area
        self.chart_frame = tk.Frame(self.examples_frame, bg="white")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Initialize chart
        self.create_example_chart()
    
    def create_quiz_tab(self):
        """Create the quiz tab"""
        self.quiz_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.quiz_frame, text="🎯 Quiz")
        
        # Quiz header
        header_frame = tk.Frame(self.quiz_frame, bg="#3498db")
        header_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Label(header_frame, text="🎯 Volatility Knowledge Quiz", 
                font=("Helvetica", 16, "bold"), bg="#3498db", fg="white", 
                pady=10).pack()
        
        # Quiz content
        self.quiz_content = tk.Frame(self.quiz_frame, bg="white")
        self.quiz_content.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Quiz state
        self.current_question = 0
        self.quiz_score = 0
        self.quiz_questions = self.get_quiz_questions()
        
        # Start quiz
        self.show_quiz_question()
    
    def load_basics(self):
        """Load basic volatility concepts"""
        content = self.content["basics"]
        self.basics_text.delete(1.0, tk.END)
        self.basics_text.insert(1.0, content)
        
        # Add some formatting
        self.basics_text.tag_configure("title", font=("Helvetica", 14, "bold"), foreground="#2c3e50")
        self.basics_text.tag_configure("subtitle", font=("Helvetica", 12, "bold"), foreground="#34495e")
        self.basics_text.tag_configure("highlight", background="#f1c40f", foreground="black")
        
        # Apply tags (simplified - would need more sophisticated parsing in real implementation)
        lines = content.split('\n')
        current_line = 1.0
        for line in lines:
            if line.startswith('WHAT IS VOLATILITY'):
                self.basics_text.tag_add("title", current_line, f"{current_line} lineend")
            elif line.startswith('Key Points:'):
                self.basics_text.tag_add("subtitle", current_line, f"{current_line} lineend")
            current_line += 1.0
    
    def show_calculation_method(self):
        """Show selected calculation method details"""
        method = self.calc_method.get()
        formulas = self.content["calculations"]
        
        if method in formulas:
            self.formula_text.delete(1.0, tk.END)
            self.formula_text.insert(1.0, formulas[method])
    
    def load_interpretation_topic(self, event=None):
        """Load selected interpretation topic"""
        topic = self.interp_topic.get()
        content = self.content["interpretation"].get(topic, "Content not available.")
        
        self.interp_text.delete(1.0, tk.END)
        self.interp_text.insert(1.0, content)
    
    def create_example_chart(self):
        """Create the initial example chart"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                                     facecolor='white')
        self.fig.suptitle('Volatility Example', fontsize=14, fontweight='bold')
        
        # Initial empty plots
        self.ax1.set_title('Price Movement')
        self.ax1.set_ylabel('Price')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Volatility')
        self.ax2.set_ylabel('Volatility (%)')
        self.ax2.set_xlabel('Time')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Generate initial example
        self.generate_example()
    
    def generate_example(self, event=None):
        """Generate selected volatility example"""
        example_type = self.example_type.get()
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Generate data based on example type
        days = 100
        np.random.seed(42)  # For reproducible examples
        
        if example_type == "Low Volatility Period":
            returns = np.random.normal(0.001, 0.01, days)
            title_suffix = "Low Volatility (σ ≈ 16%)"
            
        elif example_type == "High Volatility Period":
            returns = np.random.normal(0.001, 0.03, days)
            title_suffix = "High Volatility (σ ≈ 48%)"
            
        elif example_type == "Volatility Spike":
            returns = np.random.normal(0.001, 0.01, days)
            # Add spike in middle
            spike_start = days // 2 - 5
            spike_end = days // 2 + 5
            returns[spike_start:spike_end] = np.random.normal(0, 0.05, 10)
            title_suffix = "Volatility Spike Event"
            
        elif example_type == "Volatility Clustering":
            # Create clustered volatility
            returns = []
            vol_regime = 0.01  # Start with low vol
            for i in range(days):
                if i % 20 == 0:  # Change regime every 20 days
                    vol_regime = np.random.choice([0.01, 0.02, 0.03])
                returns.append(np.random.normal(0.001, vol_regime))
            returns = np.array(returns)
            title_suffix = "Volatility Clustering"
        
        # Calculate price series
        prices = 100 * np.cumprod(1 + returns)
        
        # Calculate rolling volatility
        returns_series = pd.Series(returns)
        volatility = returns_series.rolling(20).std() * np.sqrt(252) * 100
        
        # Plot price
        self.ax1.plot(prices, color='#3498db', linewidth=2)
        self.ax1.set_title(f'Price Movement - {title_suffix}')
        self.ax1.set_ylabel('Price ($)')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot volatility
        self.ax2.plot(volatility, color='#e74c3c', linewidth=2)
        self.ax2.set_title('Rolling Volatility (20-day)')
        self.ax2.set_ylabel('Volatility (%)')
        self.ax2.set_xlabel('Days')
        self.ax2.grid(True, alpha=0.3)
        
        # Add volatility regime bands
        self.ax2.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Low Vol')
        self.ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Medium Vol')
        self.ax2.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='High Vol')
        self.ax2.legend()
        
        plt.tight_layout()
        self.canvas.draw()
    
    def show_quiz_question(self):
        """Show current quiz question"""
        # Clear previous question
        for widget in self.quiz_content.winfo_children():
            widget.destroy()
        
        if self.current_question >= len(self.quiz_questions):
            self.show_quiz_results()
            return
        
        question_data = self.quiz_questions[self.current_question]
        
        # Question number and text
        question_frame = tk.Frame(self.quiz_content, bg="white")
        question_frame.pack(fill="x", pady=10)
        
        tk.Label(question_frame, 
                text=f"Question {self.current_question + 1} of {len(self.quiz_questions)}", 
                font=("Helvetica", 12, "bold"), bg="white", fg="#34495e").pack(anchor="w")
        
        tk.Label(question_frame, text=question_data["question"], 
                font=("Helvetica", 11), bg="white", fg="black", 
                wraplength=600, justify="left").pack(anchor="w", pady=5)
        
        # Answer options
        self.quiz_answer = tk.StringVar()
        answers_frame = tk.Frame(self.quiz_content, bg="white")
        answers_frame.pack(fill="x", pady=10)
        
        for i, option in enumerate(question_data["options"]):
            tk.Radiobutton(answers_frame, text=f"{chr(65+i)}. {option}", 
                          variable=self.quiz_answer, value=str(i),
                          font=("Helvetica", 10), bg="white", 
                          wraplength=500, justify="left").pack(anchor="w", pady=2)
        
        # Submit button
        button_frame = tk.Frame(self.quiz_content, bg="white")
        button_frame.pack(fill="x", pady=20)
        
        tk.Button(button_frame, text="Submit Answer", command=self.submit_quiz_answer,
                 bg="#27ae60", fg="white", font=("Helvetica", 11),
                 padx=20, pady=5).pack()
    
    def submit_quiz_answer(self):
        """Submit quiz answer and show result"""
        if not self.quiz_answer.get():
            return
        
        question_data = self.quiz_questions[self.current_question]
        correct_answer = question_data["correct"]
        user_answer = int(self.quiz_answer.get())
        
        # Show result
        result_frame = tk.Frame(self.quiz_content, bg="white")
        result_frame.pack(fill="x", pady=10)
        
        if user_answer == correct_answer:
            self.quiz_score += 1
            tk.Label(result_frame, text="✅ Correct!", 
                    font=("Helvetica", 12, "bold"), bg="white", fg="#27ae60").pack()
        else:
            correct_text = question_data["options"][correct_answer]
            tk.Label(result_frame, text="❌ Incorrect", 
                    font=("Helvetica", 12, "bold"), bg="white", fg="#e74c3c").pack()
            tk.Label(result_frame, text=f"Correct answer: {chr(65+correct_answer)}. {correct_text}", 
                    font=("Helvetica", 10), bg="white", fg="#34495e",
                    wraplength=600, justify="left").pack(pady=5)
        
        # Explanation
        if "explanation" in question_data:
            tk.Label(result_frame, text=f"Explanation: {question_data['explanation']}", 
                    font=("Helvetica", 10), bg="white", fg="#2c3e50",
                    wraplength=600, justify="left").pack(pady=5)
        
        # Next button
        button_frame = tk.Frame(self.quiz_content, bg="white")
        button_frame.pack(fill="x", pady=10)
        
        tk.Button(button_frame, text="Next Question", command=self.next_quiz_question,
                 bg="#3498db", fg="white", font=("Helvetica", 11),
                 padx=20, pady=5).pack()
    
    def next_quiz_question(self):
        """Move to next quiz question"""
        self.current_question += 1
        self.show_quiz_question()
    
    def show_quiz_results(self):
        """Show final quiz results"""
        # Clear content
        for widget in self.quiz_content.winfo_children():
            widget.destroy()
        
        # Results
        results_frame = tk.Frame(self.quiz_content, bg="white")
        results_frame.pack(expand=True)
        
        score_percent = (self.quiz_score / len(self.quiz_questions)) * 100
        
        tk.Label(results_frame, text="🎉 Quiz Complete!", 
                font=("Helvetica", 18, "bold"), bg="white", fg="#2c3e50").pack(pady=20)
        
        tk.Label(results_frame, text=f"Your Score: {self.quiz_score}/{len(self.quiz_questions)} ({score_percent:.1f}%)", 
                font=("Helvetica", 14), bg="white", fg="#34495e").pack(pady=10)
        
        # Performance message
        if score_percent >= 80:
            message = "Excellent! You have a strong understanding of volatility concepts."
            color = "#27ae60"
        elif score_percent >= 60:
            message = "Good job! You understand the basics well."
            color = "#f39c12"
        else:
            message = "Keep studying! Review the basics and try again."
            color = "#e74c3c"
        
        tk.Label(results_frame, text=message, 
                font=("Helvetica", 12), bg="white", fg=color,
                wraplength=400, justify="center").pack(pady=10)
        
        # Restart button
        tk.Button(results_frame, text="Restart Quiz", command=self.restart_quiz,
                 bg="#3498db", fg="white", font=("Helvetica", 11),
                 padx=20, pady=5).pack(pady=20)
    
    def restart_quiz(self):
        """Restart the quiz"""
        self.current_question = 0
        self.quiz_score = 0
        self.show_quiz_question()
    
    def get_educational_content(self):
        """Get all educational content"""
        return {
            "basics": """WHAT IS VOLATILITY?

Volatility measures how much the price of a financial instrument moves up and down over time. It's one of the most important concepts in finance and trading.

Key Points:
• Higher volatility = larger price swings = higher risk/reward potential
• Lower volatility = smaller price movements = more stable, predictable behavior
• Volatility is NOT direction - it measures magnitude of movement, not whether prices go up or down

TYPES OF VOLATILITY:

1. Historical Volatility (Realized Volatility)
   - Calculated from past price movements
   - Shows what actually happened
   - Used to understand past market behavior

2. Implied Volatility
   - Derived from options prices
   - Shows what the market expects
   - Forward-looking measure

3. Annualized Volatility
   - Scaled to represent yearly volatility
   - Standard way to compare volatilities
   - Calculated by multiplying daily volatility by √252

VOLATILITY CHARACTERISTICS:

Volatility Clustering:
Periods of high volatility tend to be followed by high volatility, and periods of low volatility by low volatility. This creates "clusters" of similar volatility levels.

Mean Reversion:
Volatility tends to return to its long-term average over time. Extremely high or low volatility periods are usually temporary.

Asymmetric Response:
Volatility often increases more after market declines than after equivalent market rises. This is called the "volatility skew" or "leverage effect."

PRACTICAL IMPLICATIONS:

For Traders:
• High volatility = more profit opportunities but higher risk
• Position sizing should account for volatility levels
• Stop losses may need to be wider in high volatility periods

For Investors:
• High volatility may present buying opportunities
• Portfolio diversification becomes more important during volatile periods
• Long-term investors can often ignore short-term volatility spikes

For Risk Management:
• Volatility is a key input for Value at Risk (VaR) calculations
• Position limits often scale with volatility
• Stress testing scenarios typically include high volatility periods""",

            "calculations": {
                "Close-to-Close": """CLOSE-TO-CLOSE VOLATILITY

Formula:
σ = √(252 × Σ(ln(Pt/Pt-1))²/(n-1))

Where:
- Pt = Closing price at time t
- n = Number of observations
- 252 = Trading days in a year (annualization factor)

Steps:
1. Calculate daily returns: rt = ln(Pt/Pt-1)
2. Calculate variance: σ² = Σ(rt - μ)²/(n-1)
3. Take square root: σ = √σ²
4. Annualize: σannual = σdaily × √252

Advantages:
• Simple and widely used
• Only requires closing prices
• Easy to understand and calculate

Disadvantages:
• Ignores intraday price movements
• May underestimate true volatility
• Assumes prices only move at market close""",

                "Garman-Klass": """GARMAN-KLASS VOLATILITY

Formula:
σ² = (1/n) × Σ[0.5×(ln(H/L))² - (2×ln(2)-1)×(ln(C/O))²]

Where:
- H = High price
- L = Low price  
- C = Close price
- O = Open price
- n = Number of observations

Features:
• Uses all OHLC data (Open, High, Low, Close)
• More efficient estimator than close-to-close
• Captures intraday volatility information

Advantages:
• More accurate than close-to-close
• Uses all available price information
• Better for high-frequency analysis

Disadvantages:
• Requires OHLC data (not just closing prices)
• More complex calculation
• Assumes no overnight gaps""",

                "Rogers-Satchell": """ROGERS-SATCHELL VOLATILITY

Formula:
σ² = (1/n) × Σ[ln(H/C)×ln(H/O) + ln(L/C)×ln(L/O)]

Where:
- H = High price
- L = Low price
- C = Close price  
- O = Open price

Special Properties:
• Drift-independent estimator
• Doesn't assume zero drift in prices
• Handles trending markets better

Advantages:
• Works well with trending markets
• Theoretically superior to Garman-Klass
• Accounts for price drift

Disadvantages:
• Still assumes no overnight gaps
• Complex calculation
• Less intuitive than simpler methods""",

                "Yang-Zhang": """YANG-ZHANG VOLATILITY

Formula:
σ² = σ²overnight + k×σ²open-to-close + (1-k)×σ²Rogers-Satchell

Where:
- σ²overnight = Variance of overnight returns
- σ²open-to-close = Variance of open-to-close returns
- k = Scaling factor
- σ²Rogers-Satchell = Rogers-Satchell estimator

Components:
1. Overnight volatility: ln(Ot/Ct-1)
2. Open-to-close volatility: ln(Ct/Ot)
3. Intraday Rogers-Satchell component

Advantages:
• Handles overnight gaps
• Most comprehensive estimator
• Combines multiple volatility sources

Disadvantages:
• Most complex calculation
• Requires complete OHLC data
• May be over-engineered for simple applications"""
            },

            "interpretation": {
                "Volatility Levels": """INTERPRETING VOLATILITY LEVELS

Low Volatility (< 15%):
• Market is calm and stable
• Prices move in small, predictable ranges
• Lower profit opportunities but also lower risk
• Often occurs during economic stability
• Good for income strategies, covered calls

Medium Volatility (15-25%):
• Normal market conditions
• Balanced risk/reward environment
• Most common volatility range for many stocks
• Suitable for most trading strategies
• Market functioning normally

High Volatility (25-40%):
• Elevated uncertainty and fear
• Larger price swings, both up and down
• Higher profit potential but greater risk
• Often during earnings, news events
• Requires careful position sizing

Extreme Volatility (> 40%):
• Crisis or panic conditions
• Very large, unpredictable moves
• Extremely high risk environment
• Often during market crashes, major events
• Most strategies become difficult to execute

CONTEXT MATTERS:
• Different assets have different "normal" volatility levels
• Technology stocks typically more volatile than utilities
• Emerging markets more volatile than developed markets
• Consider historical context and asset class norms""",

                "Regime Detection": """VOLATILITY REGIME DETECTION

What are Volatility Regimes?
Persistent periods where volatility remains at similar levels. Markets tend to stay in volatility regimes for extended periods before transitioning.

Common Regimes:
1. Low Volatility Regime
   - Sustained periods of calm
   - Markets trending smoothly
   - Economic stability

2. High Volatility Regime  
   - Persistent uncertainty
   - Frequent large moves
   - Economic or political stress

Detection Methods:
• Statistical: Moving averages, percentiles
• Machine Learning: Clustering algorithms
• Market-based: VIX levels, realized vol

Regime Persistence:
• Volatility regimes can last weeks to years
• Transitions often sudden and dramatic
• Important for risk management and strategy selection

Trading Implications:
• Different strategies work better in different regimes
• Position sizing should adapt to current regime
• Regime changes often signal major market shifts""",

                "Market Conditions": """VOLATILITY AND MARKET CONDITIONS

Bull Markets:
• Generally lower volatility
• Steady, gradual price increases
• Occasional volatility spikes on pullbacks
• "Climbing a wall of worry"

Bear Markets:
• Higher overall volatility
• Sharp, sudden declines
• Relief rallies can be very volatile
• Fear dominates market psychology

Market Corrections:
• Volatility spikes during initial decline
• Often 20-30% increase in volatility
• Creates opportunities for contrarian investors
• Usually temporary but can be severe

Crisis Periods:
• Extreme volatility (often >50%)
• Correlations increase (diversification fails)
• Liquidity can dry up
• Traditional relationships break down

Economic Cycles:
• Recession: Higher volatility
• Recovery: Decreasing volatility  
• Expansion: Low volatility
• Peak: Volatility starting to increase""",

                "Trading Implications": """TRADING WITH VOLATILITY

Position Sizing:
• Higher volatility = smaller position sizes
• Use volatility-adjusted position sizing
• Risk should be consistent across trades
• Example: 1% risk per trade regardless of volatility

Stop Losses:
• Wider stops in high volatility
• ATR-based stops adapt automatically
• Consider volatility when setting targets
• Avoid getting stopped out by normal fluctuations

Strategy Selection:
Low Volatility:
• Momentum strategies
• Carry trades
• Income strategies

High Volatility:
• Mean reversion strategies
• Options strategies
• Contrarian approaches

Timing:
• Volatility often mean-reverts
• Extreme levels often temporary
• Consider fading extreme volatility
• But trends in volatility can persist

Risk Management:
• Monitor volatility changes
• Adjust position sizes dynamically
• Use volatility for portfolio allocation
• Stress test portfolios at different volatility levels"""
            }
        }
    
    def get_quiz_questions(self):
        """Get quiz questions"""
        return [
            {
                "question": "What does volatility measure in financial markets?",
                "options": [
                    "The direction of price movement",
                    "The magnitude of price fluctuations", 
                    "The average price of an asset",
                    "The trading volume"
                ],
                "correct": 1,
                "explanation": "Volatility measures the magnitude of price movements, not their direction."
            },
            {
                "question": "Which volatility calculation method uses all OHLC (Open, High, Low, Close) data?",
                "options": [
                    "Close-to-Close",
                    "Simple Moving Average",
                    "Garman-Klass",
                    "Exponential Smoothing"
                ],
                "correct": 2,
                "explanation": "Garman-Klass uses all OHLC data, making it more accurate than close-to-close methods."
            },
            {
                "question": "What is the typical annualization factor for daily volatility?",
                "options": [
                    "365",
                    "252", 
                    "250",
                    "360"
                ],
                "correct": 1,
                "explanation": "252 is the standard trading days per year used for annualizing daily volatility."
            },
            {
                "question": "High volatility periods are generally associated with:",
                "options": [
                    "Lower risk and lower potential returns",
                    "Higher risk and higher potential returns",
                    "Lower risk and higher potential returns", 
                    "No change in risk or returns"
                ],
                "correct": 1,
                "explanation": "High volatility means larger price swings, creating both higher risk and higher profit potential."
            },
            {
                "question": "What is volatility clustering?",
                "options": [
                    "Volatility being the same across all assets",
                    "High volatility periods followed by high volatility periods",
                    "Volatility only occurring during market hours",
                    "Volatility being caused by trading volume"
                ],
                "correct": 1,
                "explanation": "Volatility clustering means periods of high volatility tend to be followed by more high volatility."
            }
        ]