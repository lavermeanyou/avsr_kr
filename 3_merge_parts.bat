@echo off
setlocal
title avsr_kr - 3. Merge the parts
echo ==================================================================
echo  avsr_kr  -  Step 3, on the main PC: add the parts from PCs 2-4
echo ==================================================================
echo Give the folders that hold the parts made on the other PCs, for
echo example the USB drive E:\ or a network folder \\PC2\share
echo Several folders: separate them with ;   for example  E:\;F:\
echo A folder may hold several parts: avsr_part2, avsr_part3, ...
echo Folders named work_shard1 ... work_shard4 next to this file are
echo added automatically. Videos already in the work folder are kept.
echo Nothing is deleted from the part folders.
echo.
set "SRC="
set /p "SRC=Folders with the parts [Enter = only work_shard folders here]: "
if not defined SRC goto run_local
set "SRC=%SRC:"=%"
if "%SRC:~-1%"=="\" set "SRC=%SRC%."
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\merge_shards.ps1" -IncludeLocal -Src "%SRC%"
goto after
:run_local
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\merge_shards.ps1" -IncludeLocal
:after
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" goto ok
echo Finished WITH PROBLEMS, exit code %CODE%. Read the messages above.
echo After fixing them, run this file again: copied files are skipped.
goto end
:ok
echo Done. The work folder is complete. Next: 4_train_sync.bat
:end
echo.
pause
endlocal
