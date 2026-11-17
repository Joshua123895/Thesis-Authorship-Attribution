@echo off
REM Henry review tasks 19/24/22: IndoBERT with nested epoch selection. Log: results_review\indobert_log.txt
REM Usage: run_indobert_review.bat                         (all three conditions, resumable)
REM        run_indobert_review.bat "question groups"       (one condition only)
cd /d "%~dp0"
if not exist results_review mkdir results_review
set LOG=results_review\indobert_log.txt
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo START %date% %time% >> %LOG%
%PY% -u -c "import torch; ok=torch.cuda.is_available(); print('cuda', ok); raise SystemExit(0 if ok else 3)" >> %LOG% 2>&1
if errorlevel 1 (
  echo ENV_CHECK_FAILED >> %LOG%
  echo Environment check failed - see %LOG%
  pause
  exit /b 1
)
%PY% -u src_review\run_indobert_review.py %* >> %LOG% 2>&1
echo EXIT %errorlevel% %date% %time% >> %LOG%
echo Finished - see %LOG%
pause
