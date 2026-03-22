@echo off
REM run.bat — 一鍵執行 ROM v4 (Split-ROM) 驗證 (Windows)
REM 使用方式: run.bat [config.toml]

setlocal

set "SCRIPT_DIR=%~dp0"
if "%~1"=="" (
    set "CONFIG=%SCRIPT_DIR%workspace\config.toml"
) else (
    set "CONFIG=%~1"
)

if not exist "%CONFIG%" (
    echo ERROR: Config file not found: %CONFIG%
    exit /b 1
)

echo === ROM v4 (Split-ROM) Verification ===
echo Config : %CONFIG%
echo ========================================

cd /d "%SCRIPT_DIR%workspace"
uv run verify_romv4.py --config "%CONFIG%"

endlocal
