:: ================================
:: LEO Full Pipeline Runner
:: Run after train_full.bat completes
:: Or run standalone if model already trained
:: ================================

@echo off
cd /d C:\Users\irumb\OneDrive\Documents\FCE_project

echo ================================
echo LEO Full Pipeline Run
echo Started: %date% %time%
echo ================================

:: Prevent sleep during pipeline run
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Guard — check trained model exists before running pipeline
python -c "import json; d=json.load(open('models/lstm_results.json')); exit(0 if d.get('avg_auc',0) >= 0.70 else 1)" 2>nul
if %errorlevel%==1 (
    echo ERROR: No valid trained model found.
    echo Please run train_full.bat first.
    goto cleanup
)

:: ── STEP 2 ───────────────────────
echo.
echo ================================
echo Step 2 - Evaluating against baselines...
echo ================================
python scripts/evaluate_lstm.py ^
  --data data/banking_api_features_v6.csv ^
  --end_date 2024-12-31
if %errorlevel% neq 0 (
    echo FAILED: evaluate_lstm.py
    goto cleanup
)
echo DONE: Evaluation complete

:: ── STEP 3 ───────────────────────
echo.
echo ================================
echo Step 3 - Running ablation study...
echo ================================
python scripts/ablation_study.py ^
  --data data/banking_api_features_v6.csv ^
  --recent_rows 300000 ^
  --epochs 5 ^
  --max_seq 50000
if %errorlevel% neq 0 (
    echo FAILED: ablation_study.py
    goto cleanup
)
echo DONE: Ablation complete

:: ── STEP 4 ───────────────────────
echo.
echo ================================
echo Step 4 - Running conformal prediction...
echo ================================
python scripts/conformal_prediction.py ^
  --cal_seq 100000 ^
  --test_seq 50000
if %errorlevel% neq 0 (
    echo FAILED: conformal_prediction.py
    goto cleanup
)
echo DONE: Conformal prediction complete

:: ── STEP 5 ───────────────────────
echo.
echo ================================
echo Step 5 - Running agent simulation...
echo ================================
python scripts/agent_simulation.py ^
  --data data/banking_api_features_v6.csv ^
  --n_transactions 1000 ^
  --seed 42
if %errorlevel% neq 0 (
    echo FAILED: agent_simulation.py
    goto cleanup
)
echo DONE: Agent simulation complete

:: ── STEP 6 ───────────────────────
echo.
echo ================================
echo Step 6 - Running self-improving pipeline dry run...
echo ================================
python scripts/self_improving_pipeline.py ^
  --dry_run ^
  --recent_rows 1000000
if %errorlevel% neq 0 (
    echo FAILED: self_improving_pipeline.py
    goto cleanup
)
echo DONE: Self-heal dry run complete

:: ── ALL STEPS PASSED ─────────────
echo.
echo ================================
echo All steps completed successfully
echo Finished: %date% %time%
echo ================================
goto dashboard

:: ── FAILURE PATH ─────────────────
:cleanup
echo.
echo Pipeline stopped due to failure above.
echo Fix the failing script then re-run run_pipeline.bat
goto done

:: ── DASHBOARD ────────────────────
:dashboard
:: Restore sleep before starting dashboard
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 60
powercfg /change monitor-timeout-ac 15

:: Kill any existing process on port 8000
echo Clearing port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /f /pid %%a 2>nul
)

echo.
echo Starting dashboard...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop the dashboard
echo.
python scripts/production_dashboard.py
goto done

:: ── ALWAYS RUNS ──────────────────
:done
:: Restore sleep settings in all cases
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 60
powercfg /change monitor-timeout-ac 15

pause