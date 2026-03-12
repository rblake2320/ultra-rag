@echo off
echo ============================================
echo  Ultra RAG Client Setup
echo ============================================
echo.

REM Try to find ultra_client.py
set SCRIPT_DIR=%~dp0

REM Install requirements
echo Installing dependencies...
pip install httpx --quiet 2>/dev/null

REM Prompt for hub URL if not set
if "%ULTRA_RAG_URL%"=="" (
    echo.
    echo Hub URL options:
    echo   [1] VPS (permanent): http://76.13.118.222:8300
    echo   [2] LAN (same network): http://192.168.12.198:8300
    echo   [3] Custom URL (Cloudflare tunnel, etc.)
    set /p CHOICE="Enter choice [1]: "
    if "%CHOICE%"=="2" (
        setx ULTRA_RAG_URL "http://192.168.12.198:8300"
    ) else if "%CHOICE%"=="3" (
        set /p CUSTOM_URL="Enter full URL: "
        setx ULTRA_RAG_URL "%CUSTOM_URL%"
    ) else (
        setx ULTRA_RAG_URL "http://76.13.118.222:8300"
    )
    echo URL saved to ULTRA_RAG_URL environment variable.
)

REM Optional: set API key
if "%ULTRA_RAG_API_KEY%"=="" (
    set /p API_KEY="Enter API key (or press Enter to skip): "
    if not "%API_KEY%"=="" (
        setx ULTRA_RAG_API_KEY "%API_KEY%"
        echo API key saved.
    )
)

REM Register this system
set SYSTEM_ID=%COMPUTERNAME%
echo.
echo Registering %SYSTEM_ID% with Ultra RAG hub...
python "%SCRIPT_DIR%ultra_client.py" --register --system-id "%SYSTEM_ID%"

echo.
echo ============================================
echo  Setup complete! Test with:
echo  python ultra_client.py "your question"
echo ============================================
pause
