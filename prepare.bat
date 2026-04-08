@echo off
REM ============================================================
REM  Setup ambiente virtuale e dipendenze per il progetto PID
REM ============================================================

echo ===================================
echo  Particle Identification - Setup
echo ===================================
echo.

REM --- Creazione virtual environment ---
if not exist ".venv" (
    echo [1/3] Creazione virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERRORE: impossibile creare il venv. Assicurati che Python sia installato.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment gia' presente, skip.
)

REM --- Attivazione ---
echo [2/3] Attivazione venv...
.venv\Scripts\Activate.ps1 

REM --- Installazione dipendenze ---
echo [3/3] Installazione dipendenze da requirements.txt...
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip3 install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128  

echo.
echo ===================================
echo  Setup completato!
echo  Esegui:
echo    python main.py --help
echo ===================================
pause
