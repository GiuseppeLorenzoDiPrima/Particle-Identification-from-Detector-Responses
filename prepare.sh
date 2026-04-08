#!/usr/bin/env bash

# ============================================================
# Setup ambiente virtuale e dipendenze per il progetto PID
# ============================================================

set -e

echo "==================================="
echo " Particle Identification - Setup"
echo "==================================="
echo

# --- Creazione virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[1/3] Creazione virtual environment..."
    python3 -m venv .venv
else
    echo "[1/3] Virtual environment gia' presente, skip."
fi

echo "[2/3] Attivazione venv..."
source .venv/bin/activate

echo "[3/3] Installazione dipendenze da requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo
echo "==================================="
echo " Setup completato!"
echo " Esegui:"
echo "   python3 main.py --help"
echo "==================================="
