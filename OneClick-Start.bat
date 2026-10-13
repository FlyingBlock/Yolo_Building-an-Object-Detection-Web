@echo off
rem ---------------------------------------------------------------------------
rem  Double-click to start the YOLO object detection web app.
rem  This is the ASCII-only launcher: cmd.exe renders non-ASCII text
rem  unpredictably across codepages, so every Chinese message is printed by
rem  start.ps1 (UTF-8 with BOM) instead.
rem
rem  Usage examples:
rem      OneClick-Start.bat
rem      OneClick-Start.bat -ApiPort 8010 -UiPort 8511
rem      OneClick-Start.bat -NoBrowser
rem      OneClick-Start.bat -SetupOnly
rem ---------------------------------------------------------------------------
chcp 65001 >nul 2>&1
cd /d "%~dp0" 2>nul
set "PS_EXE=powershell.exe"
where pwsh.exe >nul 2>&1 && set "PS_EXE=pwsh.exe"
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [launcher] exited with code %EXIT_CODE%
    echo [launcher] press any key to close this window...
    pause >nul
)
exit /b %EXIT_CODE%
