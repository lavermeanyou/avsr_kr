@echo off
setlocal
title avsr_kr - 1. Setup
echo ==================================================================
echo  avsr_kr  -  Step 1: install what this PC needs
echo ==================================================================
echo This installs Python 3.12, FFmpeg and the Python packages
echo (about 4 GB to download the first time, 10 to 20 minutes).
echo It is safe to run again: anything already installed is skipped.
echo Everything is written to setup.log in this folder.
echo.
set "ANSWER="
set /p "ANSWER=Press Enter to start. Type C to only check, Q to quit: "
if /i "%ANSWER%"=="Q" goto end
set "MODE="
if /i "%ANSWER%"=="C" set "MODE=-CheckOnly"
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %MODE%
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" goto ok
echo There were problems - read the red lines above or setup.log.
echo Fix them, then run this file again.
goto end
:ok
echo Setup OK. Next: 2_preprocess_part.bat (to preprocess new data), or on
echo the main PC 4_train_sync.bat.
:end
echo.
pause
endlocal
