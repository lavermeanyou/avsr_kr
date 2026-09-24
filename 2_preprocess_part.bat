@echo off
setlocal
title avsr_kr - 2. Preprocess one part
echo ==================================================================
echo  avsr_kr  -  Step 2: preprocess this PC's part of the videos
echo ==================================================================
echo Run this on all 4 PCs at the same time. Each PC does 1/4 of the
echo videos: about 30 minutes for 960 videos, instead of 110 minutes on
echo one PC. PC 1 is the main PC, where training runs: its part goes
echo straight into the work folder. PCs 2-4 copy their part to a USB
echo drive or network folder; then run 3_merge_parts.bat on the main PC.
echo.
echo NOTE: videos that are already preprocessed are skipped. The main PC
echo already has all 960 videos of the current dataset, so this is only
echo useful for NEW videos.
echo.

:ask_pc
set "PCNUM="
set /p "PCNUM=Which PC is this? Type 1, 2, 3 or 4: "
if not defined PCNUM goto ask_pc
set "PCNUM=%PCNUM:"=%"
set "PCNUM=%PCNUM: =%"
if "%PCNUM%"=="1" goto pc_ok
if "%PCNUM%"=="2" goto pc_ok
if "%PCNUM%"=="3" goto pc_ok
if "%PCNUM%"=="4" goto pc_ok
echo Please type 1, 2, 3 or 4.
goto ask_pc
:pc_ok

echo.
echo Do all 4 PCs have the SAME videos, the same copy of the dataset?
echo   Y = yes: split the work, this PC does part %PCNUM% of 4.  [default]
echo   N = no, every PC has DIFFERENT videos: this PC does all of its own.
set "SAME="
set /p "SAME=Same videos on all PCs? [Y/n]: "
set "SPLIT=-NumShards 4"
if not defined SAME goto split_done
set "SAME=%SAME:"=%"
if /i "%SAME%"=="N" set "SPLIT=-NoSplit"
if /i "%SAME%"=="NO" set "SPLIT=-NoSplit"
:split_done

echo.
echo Data folder = the downloaded dataset folder. Press Enter to use the
echo folder in your Downloads folder whose name starts with 009.
echo Or paste the full path: in Explorer right-click the folder, choose
echo "Copy as path", then right-click in this window to paste.
set "DATA="
set /p "DATA=Data folder [Enter = Downloads\009...]: "
if not defined DATA goto data_done
set "DATA=%DATA:"=%"
if "%DATA:~-1%"=="\" set "DATA=%DATA%."
:data_done

set "DEST="
if "%PCNUM%"=="1" goto run
echo.
echo Transfer folder = where to copy the result for the main PC, for
echo example the USB drive E:\ or a network folder \\MAINPC\share
echo The result goes into a sub folder avsr_part%PCNUM% there.
echo Press Enter to skip: the result then stays in the folder
echo work_shard%PCNUM% next to this file - copy it to the main PC yourself.
set /p "DEST=Transfer folder [Enter = none]: "
if not defined DEST goto run
set "DEST=%DEST:"=%"
if "%DEST:~-1%"=="\" set "DEST=%DEST%."

:run
set "ARGS=-Shard %PCNUM% %SPLIT%"
if "%PCNUM%"=="1" set "ARGS=%ARGS% -WorkDir work"
echo.
echo Starting. To stop, close this window; run this file again to continue.
echo.
if not defined DATA goto run_nodata
if not defined DEST goto run_data
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\preprocess_shard.ps1" %ARGS% -DataRoot "%DATA%" -Dest "%DEST%"
goto after
:run_data
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\preprocess_shard.ps1" %ARGS% -DataRoot "%DATA%"
goto after
:run_nodata
if not defined DEST goto run_none
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\preprocess_shard.ps1" %ARGS% -Dest "%DEST%"
goto after
:run_none
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\preprocess_shard.ps1" %ARGS%
:after
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" goto ok
echo Finished WITH PROBLEMS, exit code %CODE%. Read the messages above.
echo Running this file again retries what failed; finished videos are skipped.
goto end
:ok
if "%PCNUM%"=="1" echo Done. When PCs 2-4 are finished, run 3_merge_parts.bat on this PC.
if not "%PCNUM%"=="1" echo Done. Bring the result to the main PC and run 3_merge_parts.bat there.
:end
echo.
pause
endlocal
