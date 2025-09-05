#!/bin/bash

echo "Starting Volatility Trading Dashboard..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.8+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Navigate to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if not already installed
echo "Checking requirements..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet

# Run the application
echo
echo "Launching dashboard..."
$PYTHON_CMD src/main.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "Application exited with an error."
    read -p "Press Enter to continue..."
fi
