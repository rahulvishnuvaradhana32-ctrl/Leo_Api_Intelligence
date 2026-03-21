:: ================================
:: LEO Full Dataset Training Runner
:: Place this file at project root:
:: C:\Users\irumb\OneDrive\Documents\FCE_project\train_full.bat
:: Run: .\train_full.bat
:: ================================

@echo off
cd /d C:\Users\irumb\OneDrive\Documents\FCE_project

echo ================================
echo LEO Full Dataset Training Run
echo Started: %date% %time%
echo ================================

:: Prevent Windows sleeping during long training run
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0

:: Backup waker — sends dummy keypress every 2 min via hidden PowerShell
start /min powershell -WindowStyle Hidden -Command "while($true){(New-Object -COM WScript.Shell).SendKeys('+{F15}');Start-Sleep -Seconds 120}"

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Skip training if a good model already exists (avg_auc >= 0.775)
::python -c "import json; d=json.load(open('models/lstm_results.json')); exit(0 if d.get('avg_auc',0) >= 0.775 else 1)" 2>nul
::if %errorlevel%==0 (
::    echo Previous successful training found. Skipping training.
::   goto pipeline
::)

:: Always retrain on full dataset — no skip
echo Retraining on full dataset — no sequence cap...

:: Run full dataset training — no sequence cap this time
:: python scripts/run_lstm_training.py ^
 ::  --data data/banking_api_features_v6.csv ^
::   --epochs 15 ^
::   --hidden_size 128 ^
::   --batch_size 128 ^
::   --patience 6

python scripts/run_lstm_training.py ^
  --data data/banking_api_features_v6.csv ^
  --epochs 15 ^
  --hidden_size 256 ^
  --batch_size 64 ^
  --patience 6 ^
  --sequence_length 60 ^
  --focal_gamma 3.0

:: If training failed stop here — do not run pipeline on a broken model
if %errorlevel% neq 0 (
    echo FAILED: Training did not complete successfully.
    goto cleanup
)

echo ================================
echo Training finished: %date% %time%
echo ================================

:: ── SUCCESS PATH ─────────────────
:pipeline
echo.
echo Training complete. Starting pipeline...
call run_pipeline.bat
goto done

:: ── FAILURE PATH ─────────────────
:cleanup
:: Falls through to :done to restore settings
echo Skipping pipeline due to training failure.

:: ── ALWAYS RUNS ──────────────────
:done
:: Restore normal sleep settings regardless of outcome
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 60
powercfg /change monitor-timeout-ac 15

:: Kill the background PowerShell waker process
taskkill /f /im powershell.exe /fi "WINDOWTITLE eq powershell*" 2>nul

pause