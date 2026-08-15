@echo off
REM MiniMax Music 3 Studio launcher
REM Usage:  start.bat            -> auto: whole-component CPU offload, peak ~18GB VRAM
REM         start.bat gpu        -> all weights resident on GPU, fastest.
REM                                 Close Chrome/Cursor/games first - it needs ~22.5GB
REM                                 of the 24GB and desktop apps alone can take 4GB.

setlocal
cd /d "%~dp0"

set OFFLOAD=auto
if not "%~1"=="" set OFFLOAD=%~1

set HF_HOME=%~dp0.hfcache

REM Prefer the bf16 pre-saved weights: same numerics as inference (bf16 either
REM way), but loading skips the fp32->bf16 conversion that dominates startup.
if exist "%~dp0models_bf16\modular_model_index.json" set MUSIC3_MODELS=%~dp0models_bf16

echo Starting MiniMax Music 3 Studio (offload=%OFFLOAD%)...
echo Open http://127.0.0.1:7878 once the model finishes loading.
echo.

start "" http://127.0.0.1:7878
".venv\Scripts\python.exe" "app\server.py" --offload %OFFLOAD%

endlocal
pause
