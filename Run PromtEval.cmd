@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\prompteval.exe" --judge-model groq/llama-3.3-70b-versatile
echo.
pause
