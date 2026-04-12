# Installazione

## Requisiti di sistema

| Requisito | Versione minima | Note |
|---|---|---|
| Python | 3.10+ | Necessario per type hints `list[int]` e `X \| Y` |
| pip | qualsiasi recente | Per installazione dipendenze |
| Account Kaggle | — | Per il download automatico del dataset |
| GPU CUDA (opzionale) | — | Accelera il training MLP; senza GPU si usa CPU |

---

## Configurazione Kaggle API

Il dataset viene scaricato automaticamente tramite l'API Kaggle ufficiale. È necessario un account Kaggle e un token API.

### 1. Creare l'API token

1. Accedere a [kaggle.com](https://www.kaggle.com)
2. Aprire il profilo → **Settings** → **API** → **Create New Token**
3. Viene scaricato il file `kaggle.json`

### 2. Posizionare il token

**Linux / macOS:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```powershell
mkdir %USERPROFILE%\.kaggle
copy %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

Il file `kaggle.json` deve contenere:
```json
{"username": "il_tuo_username", "key": "la_tua_api_key"}
```

### 3. Accettare i termini del dataset

Prima del primo download è necessario accettare le condizioni del dataset su Kaggle:
- Visitare: `https://www.kaggle.com/datasets/naharrison/particle-identification-from-detector-responses`
- Cliccare su **Download** e accettare le condizioni

---

## Installazione delle dipendenze

### Dipendenze principali

```
numpy
pandas
scikit-learn
xgboost
torch            # PyTorch (vedi sotto per installazione CUDA)
matplotlib
seaborn
shap
tabulate
pyyaml
kaggle
```

### Installazione CPU (universale)

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn shap tabulate pyyaml kaggle
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Installazione con supporto CUDA (GPU NVIDIA)

Sostituire l'ultima riga con la versione appropriata per la propria versione di CUDA.

**CUDA 12.1:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verificare la versione CUDA installata con:
```bash
nvidia-smi
```

### Verifica installazione

```python
import torch
print(torch.__version__)
print("CUDA disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## Clonare il repository

```bash
git clone <url-repository>
cd Particle-Identification-from-Detector-Responses
```

---

## Verifica dell'installazione

```bash
python main.py --quick --phase 1
```

Questo comando:
1. Scarica automaticamente il dataset da Kaggle (prima esecuzione)
2. Carica 100.000 campioni (modalità `--quick`)
3. Esegue solo la Fase 1 (preprocessing + visualizzazioni esplorative)
4. Salva le figure in `outs/imgs/pre-processing/`

Se non vengono segnalati errori, l'installazione è corretta.

---

## Struttura directory dopo l'installazione

```
Particle-Identification-from-Detector-Responses/
├── data/                   # Creata automaticamente al primo run
│   └── pid-5M.csv          # Dataset scaricato da Kaggle
├── outs/                   # Creata automaticamente al primo run
│   ├── imgs/               # Figure generate
│   ├── models/             # Checkpoint modelli
│   ├── results/            # Report e CSV
│   └── logs/               # File di log
└── ...
```

Le directory `data/` e `outs/` sono elencate nel `.gitignore` e non vengono tracciate da git.

---

## Risoluzione problemi comuni

### `kaggle: command not found`

```bash
pip install kaggle
# oppure
python -m kaggle datasets download ...
```

### `401 - Unauthorized` durante il download

Il file `kaggle.json` non è nella posizione corretta o i permessi non sono impostati. Verificare il percorso e ripetere i passi della [Configurazione Kaggle API](#configurazione-kaggle-api).

### `OSError: [Errno 28] No space left on device`

Il dataset completo (5M eventi) occupa circa 600 MB. Assicurarsi di avere sufficiente spazio. In alternativa, usare il parametro `max_samples` in `config.yaml` per lavorare con un sottoinsieme.

### CUDA non rilevata

Se `torch.cuda.is_available()` restituisce `False` anche con una GPU NVIDIA:
1. Verificare che i driver NVIDIA siano aggiornati
2. Reinstallare PyTorch con la versione CUDA corretta
3. Il codice funziona comunque su CPU, con tempi di training più lunghi
