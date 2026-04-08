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
    call python -m venv .venv
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
.venv\Scripts\activate.bat

REM --- Installazione dipendenze ---
echo [3/3] Installazione dipendenze da requirements.txt...
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===================================
echo  Setup completato!
echo  Attiva il venv con:
echo    .venv\Scripts\activate
echo  Poi esegui:
echo    python main.py --help
echo ===================================
pause
