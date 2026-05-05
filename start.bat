@echo off
setlocal

cd /d "%~dp0"

set "PORT=%XP_APP_PORT%"
if not defined PORT set "PORT=7860"

:: Optional OpenAI-compatible API configuration.
:: Set these in the system environment or in the current terminal before running.
:: Example:
::   set "XP_OPENAI_API_KEY=your_api_key"
::   set "XP_OPENAI_BASE_URL=https://openrouter.ai/api/v1"

if not exist "app.py" (
    echo [ERROR] app.py not found in %CD%
    pause
    exit /b 1
)

if not exist "xp_search\ui.py" (
    echo [ERROR] Source folder xp_search is missing in %CD%
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON="
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$cmd=(Get-Command python -ErrorAction SilentlyContinue); if($cmd){$cmd.Source}"`) do set "PYTHON=%%I"
    if not defined PYTHON set "PYTHON=python"
)

if /i "%PYTHON%"=="python" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.11+ or create .venv in this folder.
        pause
        exit /b 1
    )
)

"%PYTHON%" -c "import gradio, requests, telethon, qrcode" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Missing dependencies. Installing from requirements.txt...
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

set "PYTHONIOENCODING=utf-8"

echo [INFO] Starting app...
echo [INFO] URL: http://127.0.0.1:%PORT%
echo.

start "" cmd /c "timeout /t 4 >nul && start """" http://127.0.0.1:%PORT%"

"%PYTHON%" app.py

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] App exited with code: %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
