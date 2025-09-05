@echo off
echo Starting Volatility Trading Dashboard...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Navigate to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install requirements if not already installed
echo Checking requirements...
pip install -r requirements.txt --quiet

REM Run the application
echo.
echo Launching dashboard...
python src\main.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
