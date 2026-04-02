# Particle Identification from Detector Responses

Benvenuta/o, questo progetto è stato sviluppato per la materia "Principi di Calcolo Tensoriale" prevista dal piano di studi per il Corso di Dottorato di Ricerca in Sistemi Intelligente per l'Ingegneria. Il lavoro proposto prevede l'applicazione di tecniche di Machine Learning alla fisica delle particelle.
Nel dettaglio, si tratta di un problema di classificazione multi-classe, in cui l'obiettivo è identificare 4 specie di particelle (elettrone, pione, kaone o protone) a partire dalle risposte di 6 rivelatori, confrontando il metodo tradizionale a tagli fisici con modelli di ML e DL.

---

## Indice

1. [Prerequisiti](#1-prerequisiti)
2. [Installazione](#2-installazione)
3. [Esecuzione](#3-esecuzione)
4. [Struttura del progetto](#4-struttura-del-progetto)
5. [Le 6 fasi della pipeline](#5-le-6-fasi-della-pipeline)
6. [Configurazione](#6-configurazione)
7. [Output prodotti](#7-output-prodotti)

---

## 1. Prerequisiti

- **Python 3.10+** installato e disponibile da terminale (`python --version`)
- **Connessione internet** (per scaricare il dataset da Kaggle al primo avvio)
- **Account Kaggle** gratuito (serve per le credenziali API)

Il dataset viene scaricato automaticamente da Kaggle tramite API. Se non hai configurato il file **kaggle.json** in precedenza, devi:

### Passo 1 - Crea un account Kaggle

Se non ne hai uno, registrati su [kaggle.com](https://www.kaggle.com/) (è gratuito).

### Passo 2 - Genera la API Key

1. Vai su [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scorri fino alla sezione **API**
3. Clicca su **"Create New Token"**
4. Crea una cartella al path: 

```
C:\Users\<TUO_UTENTE>\.kaggle\kaggle.json
```

5. Crea al suo interno un file `kaggle.json` con questo contenuto:

```json
{"username": "il_tuo_username", "key": "una_stringa_esadecimale"}
```

### Passo 3 - Posiziona il file

Copia il file `kaggle.json` scaricato nella cartella:

```
C:\Users\<TUO_UTENTE>\.kaggle\kaggle.json
```

### Verifica

Dopo aver configurato il file, puoi verificare che funzioni aprendo un terminale e digitando:

```bash
pip install kaggle
kaggle datasets list
```

Se vedi una lista di dataset, le credenziali sono configurate correttamente.

---

## 2. Installazione

### Metodo rapido (Windows)

Doppio click su **`prepare.bat`**, oppure, da terminale: 

```bash
.\prepare.bat
```

Lo script:

1. Crea un virtual environment in `.venv/`
2. Lo attiva
3. Installa tutte le dipendenze da `requirements.txt`

### Metodo manuale

```bash
# Crea il virtual environment
python -m venv .venv

# Attiva il virtual environment
# Su Windows (cmd):
.venv\Scripts\activate
# Su Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Su Linux/Mac:
source .venv/bin/activate

# Installa le dipendenze
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Esecuzione

> **Importante:** assicurati di aver attivato il virtual environment prima di eseguire qualsiasi comando (`(.venv)` deve comparire nel prompt). Se non compare, esegui:

```bash
# Attiva il virtual environment
# Su Windows (cmd):
.venv\Scripts\activate
# Su Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Su Linux/Mac:
source .venv/bin/activate

```

### Pipeline completa

Per eseguire tutte e 6 le fasi in sequenza:

```bash
python main.py
```

### Run veloce (consigliato per il primo test)

Usa solo 100k campioni e meno epoche di training. Ideale per verificare che tutto funzioni prima del run completo:

```bash
python main.py --quick
```

### Eseguire una singola fase

```bash
python main.py --phase 1    # Solo visualizzazione esplorativa
python main.py --phase 2    # Solo baseline a tagli fisici
python main.py --phase 3    # Solo modelli ML classici
python main.py --phase 4    # Solo deep learning (MLP)
python main.py --phase 5    # Solo interpretabilita' e uncertainty
python main.py --phase 6    # Solo valutazione finale e confronto
```

### Eseguire piu' fasi selezionate

```bash
python main.py --phases 1 2 3    # Fasi 1, 2 e 3
```

### Usare una configurazione custom

```bash
python main.py --config mia_config.yaml
```

---

## 5. Struttura del progetto

```
CalcoloTensoriale/
│
├── main.py                  # Entry point - orchestra la pipeline
├── config.yaml              # Configurazione centralizzata
├── requirements.txt         # Dipendenze Python
├── prepare.bat              # Script di setup automatico (Windows)
├── README.md                # Questo file
├── .gitignore
│
├── src/                     # Codice sorgente modulare
│   ├── __init__.py
│   ├── data_loader.py       # Download da Kaggle + preprocessing + split
│   ├── visualization.py     # Grafici: Bethe-Bloch, distribuzioni, ROC, CM
│   ├── baseline.py          # PID tradizionale a tagli fisici
│   ├── classical_models.py  # LR, KNN, Decision Tree, Random Forest e XGBoost
│   ├── deep_learning.py     # MLP con PyTorch (early stopping, BatchNorm)
│   ├── evaluation.py        # Metriche, tabella comparativa, report
│   ├── interpretability.py  # Analisi SHAP values
│   └── uncertainty.py       # MC Dropout e uncertainty quantification
│
├── data/                    # Dataset CSV (scaricato automaticamente)
│
└── outputs/
    ├── figures/             # Tutti i grafici generati (.png)
    ├── models/              # Modelli salvati (es. mlp_best.pt)
    └── results/             # Tabelle CSV, report testuali, log
```

---

## 6. Le 6 fasi della pipeline

### Fase 1 - Caricamento dati e visualizzazione esplorativa

- Scarica il dataset da Kaggle (solo la prima volta)
- Genera il diagramma di **Bethe-Bloch** (energia depositata vs momento):
  le 4 specie di particelle formano bande visivamente separabili
- Distribuzioni di ogni feature per tipo di particella
- Distribuzione delle classi (per verificare eventuali sbilanciamenti)
- Matrice di correlazione tra le feature

### Fase 2 - Baseline a tagli fisici

- Implementa la PID "tradizionale" usata in fisica sperimentale
- Per ogni classe, calcola l'intervallo [5o, 95o percentile] di ogni feature
- Classifica un evento in base a quante feature cadono nell'intervallo atteso
- Questo e' il **benchmark da battere** con il machine learning

### Fase 3 - Modelli di ML classici

- Addestra e confronta 5 modelli: Logistic Regression, K-NN, Decision Tree,
  Random Forest, XGBoost
- Cross-validation stratificata a 5 fold
- Analisi della feature importance (quali segnali del rivelatore contano di piu')

### Fase 4 - Deep Learning (MLP)

- Rete neurale densa (Multi-Layer Perceptron) con PyTorch
- Architettura: 128 → 64 → 32 neuroni, con BatchNorm e Dropout
- Early stopping sulla validation loss
- Salvataggio automatico del miglior modello

### Fase 5 - Interpretabilita' e Uncertainty

- **SHAP values**: analizza quali feature guidano la predizione per ogni tipo di particella (TreeExplainer per RF/XGBoost, KernelExplainer per MLP)
- **MC Dropout**: esegue 50 forward pass con dropout attivo per stimare l'incertezza del modello su ogni evento
- **Rejection curve**: mostra come l'accuracy migliora se si scartano gli eventi ad alta incertezza

### Fase 6 - Valutazione finale

- Tabella comparativa di tutti i modelli (accuracy, F1 macro, AUC-ROC)
- Matrice di confusione per ogni modello
- Curve ROC one-vs-rest per ogni classe
- Classification report dettagliato
- Tutto salvato in `outputs/results/`

---

## 7. Configurazione

Tutti i parametri sono in **`config.yaml`**. Puoi modificarli senza toccare
il codice. Le sezioni principali:

| Sezione | Cosa controlla |
|---------|---------------|
| `paths` | Cartelle di input/output |
| `dataset` | Slug Kaggle, split train/val/test, random seed |
| `classical_models` | Iperparametri di ciascun modello ML |
| `deep_learning` | Architettura MLP, learning rate, epoche, dropout |
| `interpretability` | Numero di campioni per SHAP |
| `uncertainty` | Numero di iterazioni MC Dropout |
| `visualization` | DPI, dimensioni figure, stile grafico |

### Esempio: cambiare l'architettura del MLP

In `config.yaml`:

```yaml
deep_learning:
  hidden_layers: [256, 128, 64]   # rete piu' profonda
  dropout: 0.4
  epochs: 100
```

### Esempio: usare un sottoinsieme del dataset

```yaml
dataset:
  max_samples: 500000   # usa solo 500k eventi su 5M
```

---

## 8. Output prodotti

Dopo un run completo, troverai in `outputs/`:

### Grafici (`outputs/figures/`)

| File | Descrizione |
|------|-------------|
| `bethe_bloch.png` | Diagramma energia vs momento per le 4 particelle |
| `feature_distributions.png` | Distribuzioni di ogni feature per classe |
| `class_distribution.png` | Bilanciamento delle classi |
| `correlation_matrix.png` | Correlazioni tra feature |
| `feature_importance.png` | Importanza delle feature (RF, XGBoost) |
| `mlp_training_history.png` | Loss e accuracy durante il training MLP |
| `cm_*.png` | Matrice di confusione per ogni modello |
| `roc_*.png` | Curve ROC per ogni modello |
| `shap_summary_*.png` | SHAP beeswarm plot |
| `shap_bar_*.png` | SHAP feature importance |
| `uncertainty_entropy.png` | Incertezza: corretti vs errati |
| `rejection_curve.png` | Accuracy vs soglia di rifiuto |
| `uncertainty_per_class.png` | Incertezza per tipo di particella |
| `uncertainty_scatter.png` | Mappa di incertezza nel piano p vs energia |
| `model_comparison.png` | Confronto accuracy tra tutti i modelli |

### Risultati (`outputs/results/`)

| File | Descrizione |
|------|-------------|
| `model_comparison.csv` | Tabella con tutte le metriche per ogni modello |
| `report_*.txt` | Classification report dettagliato per modello |
| `run.log` | Log completo dell'esecuzione |

### Modelli (`outputs/models/`)

| File | Descrizione |
|------|-------------|
| `mlp_best.pt` | Pesi del miglior MLP (checkpoint PyTorch) |
