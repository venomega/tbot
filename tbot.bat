@echo off
setlocal
cd /d "%~dp0"
where python3 >nul 2>nul
if %errorlevel%==0 (
    python3 tbot.py %*
) else (
    python tbot.py %*
)
endlocal
