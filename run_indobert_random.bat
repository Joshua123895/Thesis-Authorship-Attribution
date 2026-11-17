@echo off
REM Runs IndoBERT for the random question-group design. Log: results_random\indobert_log.txt
cd /d "%~dp0"
set LOG=results_random\indobert_log.txt
set PY=python
if exist "..\.venv\Scripts\python.exe" set PY=..\.venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo START %date% %time% >> %LOG%
echo python: %PY% >> %LOG%
%PY% -u -c "import torch, transformers, sklearn, pandas; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); raise SystemExit(0 if ok else 3)" >> %LOG% 2>&1
if errorlevel 1 (
  echo ENV_CHECK_FAILED >> %LOG%
  echo Environment check failed - see %LOG%
  pause
  exit /b 1
)
%PY% -u src_random\run_indobert_random.py >> %LOG% 2>&1
echo EXIT %errorlevel% %date% %time% >> %LOG%
echo Finished - see %LOG%
pause
