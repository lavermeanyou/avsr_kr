@echo off
setlocal
title avsr_kr - 4. Train the speaker-lip matching model
echo ==================================================================
echo  avsr_kr  -  Step 4, on the main PC: train the speaker-lip
echo  matching model, which tells which face a voice belongs to
echo ==================================================================
echo Uses the preprocessed data in the work folder and the GPU.
echo Settings: configs\sync.yaml. Result: work\checkpoints_sync\best.pt
echo Log: work\logs\sync_train.log
echo Training takes hours. To stop, press Ctrl+C once and wait: the last
echo state is saved. Run this file again to continue where it stopped.
echo Do not run other GPU work at the same time - it slows training down.
echo.
if not exist "%~dp0work\manifests" goto no_data
set "GO="
set /p "GO=Press Enter to start. Type Q to quit: "
if /i "%GO%"=="Q" goto end
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train_sync.ps1"
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" goto ok
echo Training stopped with exit code %CODE%. Read the messages above.
echo Run this file again to continue from the last saved state.
goto end
:ok
echo Training finished. Next: 5_evaluate_sync.bat
goto end
:no_data
echo There is no work\manifests folder: preprocess the data first.
:end
echo.
pause
endlocal
