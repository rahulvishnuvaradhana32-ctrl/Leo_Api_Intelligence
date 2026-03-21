@echo off
cd /d C:\Users\irumb\OneDrive\Documents\FCE_project

echo ================================
echo FCE Full Pipeline Run
echo Started: %date% %time%
echo ================================

REM Prevent sleep during pipeline
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0

REM Activate venv
call .venv\Scripts\activate.bat

REM Check training completed first
python -c "import json; d=json.load(open('models/lstm_results.json')); exit(0 if d.get('avg_auc',0) >= 0.70 else 1)" 2>nul
if %errorlevel%==1 (
    echo ERROR: No valid trained model found.
    echo Please run train_full.bat first.
    goto cleanup
)

echo.
echo ================================
echo Step 2 - Evaluating against baselines...
echo ================================
python scripts/evaluate_lstm.py --end_date 2024-12-31
if %errorlevel% neq 0 (
    echo FAILED: evaluate_lstm.py
    goto cleanup
)
echo DONE: Evaluation complete

echo.
echo ================================
echo Step 3 - Running ablation study...
echo ================================
python scripts/ablation_study.py --recent_rows 300000 --epochs 5 --max_seq 50000
if %errorlevel% neq 0 (
    echo FAILED: ablation_study.py
    goto cleanup
)
echo DONE: Ablation complete

echo.
echo ================================
echo Step 4 - Running conformal prediction...
echo ================================
python scripts/conformal_prediction.py --cal_seq 50000 --test_seq 50000
if %errorlevel% neq 0 (
    echo FAILED: conformal_prediction.py
    goto cleanup
)
echo DONE: Conformal prediction complete

echo.
echo ================================
echo Step 5 - Running agent simulation...
echo ================================
python scripts/agent_simulation.py --n_transactions 1000 --seed 42
if %errorlevel% neq 0 (
    echo FAILED: agent_simulation.py
    goto cleanup
)
echo DONE: Agent simulation complete

echo.
echo ================================
echo Step 6 - Running self-improving pipeline dry run...
echo ================================
python scripts/self_improving_pipeline.py --dry_run --recent_rows 1000000
if %errorlevel% neq 0 (
    echo FAILED: self_improving_pipeline.py
    goto cleanup
)
echo DONE: Self-heal dry run complete

echo.
echo ================================
echo All steps completed successfully
echo Finished: %date% %time%
echo ================================

echo.
echo Starting dashboard...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop the dashboard
echo.

:cleanup
REM Restore sleep settings
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 60
powercfg /change monitor-timeout-ac 15

REM Step 7 - Start dashboard (always runs last even after cleanup)
python scripts/production_dashboard.py

pause