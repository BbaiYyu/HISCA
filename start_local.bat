@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set GRADIO_ANALYTICS_ENABLED=False
where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Activate the hisca environment first.
  pause
  exit /b 1
)
python check_deployment.py
if errorlevel 1 (
  pause
  exit /b 1
)
python -u app.py
pause
