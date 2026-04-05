@echo off
cd /d "%~dp0"
"C:\Program Files\Python312\python.exe" -m pytest tests/ -v --tb=short --no-header -q %*
