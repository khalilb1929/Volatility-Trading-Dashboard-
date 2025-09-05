import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from gui.dashboard import Dashboard
except ImportError as e:
    # Handle import error gracefully
    print(f"Import error: {e}")
    main_root = tk.Tk()
    main_root.withdraw()
    messagebox.showerror("Import Error", 
                        f"Could not import dashboard components.\n\n"
                        f"Error: {e}\n\n"
                        f"Please ensure all required packages are installed:\n"
                        f"pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main application entry point"""
    try:
        # Create main window
        root = tk.Tk()
        root.title("📊 Volatility Education & Analysis Dashboard")
        root.geometry("1400x900")
        root.minsize(1000, 700)
        
        # Set window icon (if available)
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.ico")
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except FileNotFoundError:
            pass  # Icon file not found, continue without it
        
        # Create and pack dashboard
        dashboard = Dashboard(root)
        dashboard.pack(fill="both", expand=True)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Handle window closing
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit the Volatility Dashboard?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the application
        root.mainloop()
        
    except ImportError as e:
        # Handle any unexpected errors
        messagebox.showerror("Application Error", 
                           f"An unexpected error occurred:\n\n{str(e)}\n\n"
                           f"Please check your installation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()