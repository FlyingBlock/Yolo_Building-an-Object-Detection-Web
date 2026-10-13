@echo off
rem ===========================================================================
rem  YOLO object detection web app -- one-click start.
rem
rem  Usage: just double-click this file. Keep it in the project root.
rem
rem  It performs, automatically, the manual steps that used to need two
rem  Miniforge Prompt windows:
rem      conda activate yoloweb
rem      uvicorn api:app --reload --port 8000     (backend / API)
rem      streamlit run app.py                     (frontend / UI)
rem  start.ps1 does all of it in one process tree, so no conda activation,
rem  no "cd /d" and no second window are required.
rem
rem  Optional arguments (same as start.ps1), for example:
rem      Launch.bat -ApiPort 8010 -UiPort 8511
rem      Launch.bat -NoBrowser
rem      Launch.bat -CondaEnv yolo
rem      Launch.bat -NoConda
rem
rem  IMPORTANT: this file is deliberately ASCII-only. cmd.exe mis-parses
rem  non-ASCII bytes inside .bat files (a full-width colon breaks "rem" and
rem  swallows the following lines), so every Chinese message is printed by
rem  start.ps1, which is UTF-8 with BOM.
rem ===========================================================================
chcp 65001 >nul 2>&1
cd /d "%~dp0"

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
