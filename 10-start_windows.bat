@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python backtest.py
if errorlevel 1 goto failed
python validate_demo.py
if errorlevel 1 goto failed
python -m streamlit run app.py
goto end
:failed
echo Setup or backtest failed. Copy the error above for help.
pause
:end
