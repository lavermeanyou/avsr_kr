@echo off
setlocal
title avsr_kr - 5. Evaluate the speaker-lip matching model
echo ==================================================================
echo  avsr_kr  -  Step 5, on the main PC: test the speaker-lip matching
echo  model on speakers it never saw in training
echo ==================================================================
echo Model: work\checkpoints_sync\best.pt
echo Results: work\eval\sync_val_results.json, sync_test_results.json,
echo sync_heldout_results.json - the tables are printed here as well.
echo   val     = speaker C313, used to choose the off-screen threshold
echo   test    = speakers E014 and C159
echo   heldout = all 3 together, used for 3-person scenes
echo.
if not exist "%~dp0work\checkpoints_sync\best.pt" goto no_model
set "ANS="
set /p "ANS=Press Enter for all three, or type val, test or heldout: "
if defined ANS set "ANS=%ANS:"=%"
set "SPLITS=val test heldout"
if /i "%ANS%"=="val" set "SPLITS=val"
if /i "%ANS%"=="test" set "SPLITS=test"
if /i "%ANS%"=="heldout" set "SPLITS=heldout"
set "CODE=0"
for %%S in (%SPLITS%) do call :one %%S
echo.
if "%CODE%"=="0" goto ok
echo Finished WITH PROBLEMS, exit code %CODE%. Read the messages above.
goto end
:ok
echo Done. The meaning of the numbers is explained in README.md section 11.
goto end
:no_model
echo There is no work\checkpoints_sync\best.pt yet: run 4_train_sync.bat first.
goto end

:one
if not "%CODE%"=="0" goto :eof
echo.
echo ---------------- split %1 ----------------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\evaluate_sync.ps1" --split %1
set "CODE=%ERRORLEVEL%"
goto :eof

:end
echo.
pause
endlocal
