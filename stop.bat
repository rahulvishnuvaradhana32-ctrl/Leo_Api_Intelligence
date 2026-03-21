:: ================================
:: LEO Stop All Running Processes
:: Run: .\stop.bat
:: ================================

@echo off
cd /d C:\Users\irumb\OneDrive\Documents\FCE_project

echo ================================
echo Stopping all LEO processes...
echo ================================

:: Kill dashboard on port 8000
echo Stopping dashboard (port 8000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /f /pid %%a 2>nul
)

:: Kill any running Python scripts
echo Stopping Python processes...
taskkill /f /im python.exe 2>nul

:: Kill background PowerShell waker
echo Stopping PowerShell waker...
taskkill /f /im powershell.exe 2>nul

:: Restore sleep settings
echo Restoring sleep settings...
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 60
powercfg /change monitor-timeout-ac 15

echo ================================
echo All processes stopped.
echo Sleep settings restored.
echo ================================

pause