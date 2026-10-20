@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
where py >nul 2>&1
if errorlevel 1 goto use_python
py -3.10 -c "import sys" >nul 2>&1
if errorlevel 1 goto use_python
py -3.10 bootstrap.py --mode web
goto finish
:use_python
python bootstrap.py --mode web
:finish
if errorlevel 1 pause