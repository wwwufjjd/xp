@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "REPO_DIR=E:\xp-main\models\hf-cache\models--Disty0--Qwen3-VL-8B-NSFW-Caption-V4.5"
set "REF_FILE=%REPO_DIR%\refs\main"

if not exist "%REF_FILE%" (
    echo [ERROR] Missing ref file: %REF_FILE%
    pause
    exit /b 1
)

set /p REV=<"%REF_FILE%"
if not defined REV (
    echo [ERROR] Empty model revision in %REF_FILE%
    pause
    exit /b 1
)

set "SNAPSHOT_DIR=%REPO_DIR%\snapshots\%REV%"
if not exist "%SNAPSHOT_DIR%" (
    mkdir "%SNAPSHOT_DIR%"
)

where curl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] curl.exe not found.
    pause
    exit /b 1
)

echo [INFO] Snapshot dir:
echo   %SNAPSHOT_DIR%
echo.
echo [INFO] Downloading missing Qwen shards with resume support.
echo [INFO] You can rerun this script any time to continue.
echo.

call :download model-00001-of-00004.safetensors
if errorlevel 1 goto :fail
call :download model-00002-of-00004.safetensors
if errorlevel 1 goto :fail
call :download model-00003-of-00004.safetensors
if errorlevel 1 goto :fail

echo.
echo [OK] Qwen shard repair finished.
pause
exit /b 0

:download
set "NAME=%~1"
set "URL=https://huggingface.co/Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5/resolve/main/%NAME%"
set "DEST=%SNAPSHOT_DIR%\%NAME%"

echo [INFO] %NAME%
curl.exe -L -C - --fail --output "%DEST%" "%URL%"
if errorlevel 1 (
    echo [ERROR] Download failed: %NAME%
    exit /b 1
)

for %%F in ("%DEST%") do echo [OK] Saved %%~nxF (%%~zF bytes)
echo.
exit /b 0

:fail
echo.
echo [ERROR] Repair stopped before completion.
pause
exit /b 1
