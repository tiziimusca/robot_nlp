@echo off
REM ============================================================
REM  Control de Acceso EPP en Tiempo Real (Cámara Web)
REM ============================================================
cd /d "%~dp0"

py -3 mi_desarrollo\control_acceso.py %*
if errorlevel 1 (
    python mi_desarrollo\control_acceso.py %*
)
pause
