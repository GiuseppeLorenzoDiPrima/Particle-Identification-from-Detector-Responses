# Particle Identification from Detector Responses
## Machine Learning applicato alla Fisica delle Particelle

Benvenuta/o. Il presente progetto è stato sviluppato nell’ambito della preparazione all’esame di Principi di Calcolo Tensoriale, previsto dal piano di studi del Corso di Dottorato di Ricerca in Sistemi Intelligenti per l’Ingegneria presso l’Università degli Studi di Enna Kore.

L’elaborato propone l’applicazione di tecniche di Machine Learning e Deep Learning al contesto della fisica delle particelle. In particolare, il problema affrontato riguarda la classificazione multi-classe: l’obiettivo consiste nell’identificare la natura di una particella in ingresso — tra elettrone, pione, kaone e protone — sulla base delle risposte fornite da sei rivelatori.

I dati utilizzati sono stati reperiti dalla piattaforma [Kaggle](https://www.kaggle.com/database/naharrison/particle-identification-from-detector-responses) e consistono in 5 milioni di samples, ciascuno descritto da sei features. Tali dati sono stati generati mediante simulazioni basate sul metodo Monte Carlo.

A tal fine, vengono messi a confronto l’approccio tradizionale, basato su tagli selettivi, e modelli avanzati di apprendimento automatico e profondo, al fine di valutarne le prestazioni e l’efficacia nel contesto considerato.

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

1. Crea una cartella al path:

```
C:\Users\<TUO_UTENTE>\.kaggle
```

2. Vai su [kaggle.com/settings](https://www.kaggle.com/settings)
3. Scorri fino alla sezione **API**
4. Clicca su **"Create New Token"**
5. Conserva la chiave esadecimale appena creata

### Passo 3 - Configura la API

6. Vai al path `C:\Users\<TUO_UTENTE>\.kaggle` e crea, al suo interno, un file `kaggle.json`

7. Inserisci, all'interno del file `C:\Users\<TUO_UTENTE>\.kaggle\kaggle.json` il contenuto contenuto:

```json
{"username": "il_tuo_username", "key": "una_stringa_esadecimale"}
```

### Passo 4 - Verifica

Dopo aver configurato il file, puoi verificare che funzioni aprendo un terminale e digitando:

```bash
pip install kaggle
kaggle datasets list
```

Se vedi una lista di dataset, le credenziali sono configurate correttamente.

---

## 2. Installazione

### Metodo rapido (Windows)

Il progetto contiene un file denominato **`prepare.bat`**. Sarà sufficiente effettuare un doppio click, oppure, da terminale, eseguire: 

```bash
.\prepare.bat
```

In questo modo, lo script:

1. Crea un virtual environment in `.venv/`
2. Lo attiva
3. Installa tutte le dipendenze da `requirements.txt`

### Metodo manuale (Windows)

```bash
# Crea il virtual environment
python -m venv .venv

# Attiva il virtual environment
# Su Windows (cmd):
.venv\Scripts\activate
# Su Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Installa le dipendenze
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```
### Metodo rapido (Linux/MacOS)

Il progetto contiene un file denominato **`prepare.sh`**. Sarà sufficiente effettuare un doppio click, oppure, da terminale, eseguire: 

```bash
bash prepare.sh
```

In questo modo, lo script:

1. Crea un virtual environment in `.venv/`
2. Lo attiva
3. Installa tutte le dipendenze da `requirements.txt`


### Metodo manuale (Linux/MacOS)

```bash
# Crea il virtual environment
python -m venv .venv

# Attiva il virtual environment
source .venv/bin/activate

# Installa le dipendenze
python -m pip install --upgrade pip
pip install -r requirements.txt
```
---

## 3. Esecuzione

> **Importante:** assicurati di aver attivato il virtual environment prima di eseguire qualsiasi comando (una cartella denominata `.venv` deve comparire nella root del progetto). Se non compare, esegui:

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
In questo modo potrai eseguire una sola fase:

```bash
python main.py --phase 1    # Solo visualizzazione esplorativa
python main.py --phase 2    # Solo baseline a tagli
python main.py --phase 3    # Solo modelli ML classici
python main.py --phase 4    # Solo deep learning (MLP)
python main.py --phase 5    # Solo interpretabilità e uncertainty
python main.py --phase 6    # Solo valutazione finale e confronto
```

### Eseguire piu' fasi selezionate
In questo modo potrai eseguire più fasi:

```bash
python main.py --phases 1 2 3    # Fasi 1, 2 e 3
```

### Usare una configurazione custom
Puoi anche impostare delle configurazioni custom digitando:

```bash
python main.py --config mia_config.yaml
```

---

## 5. Struttura del progetto

```
Particle-Identification-from-Detector-Responses/
│
├── main.py                  # Entry point
├── config.yaml              # Configurazione centralizzata
├── requirements.txt         # Dipendenze Python
├── prepare.bat              # Script di setup automatico (Windows)
├── prepare.sh               # Script di setup automatico (Linux/MacOS)
├── README.md                # Questo file
├── .gitignore
│
├── src/                     # Codice sorgente modulare
│   ├── __init__.py
│   ├── data_loader.py       # Download da Kaggle + preprocessing + split
│   ├── visualization.py     # Grafici: Bethe-Bloch, distribuzioni, ROC, CM
│   ├── baseline.py          # PID tradizionale a tagli
│   ├── classical_models.py  # LR, KNN, Decision Tree, Random Forest e XGBoost
│   ├── deep_learning.py     # MLP con Framework PyTorch
│   ├── evaluation.py        # Metriche, tabella comparativa, report, comparison
│   ├── interpretability.py  # Analisi SHAP values
│   └── uncertainty.py       # MC Dropout e uncertainty quantification
│
├── data/                    # Dataset CSV (scaricato automaticamente)
│
└── outputs/
    ├── figures/             # Tutti i grafici generati (.png)
    ├── models/              # Modelli salvati (es. mlp_best.pt)
    ├── logs/                # Log di addestramento
    └── results/             # Tabelle CSV e report testuali
```

---

## 6. Le 6 fasi della pipeline

### Fase 1 - Caricamento dati e visualizzazione esplorativa

- Scarica il dataset da Kaggle (solo la prima volta)
- Genera il diagramma di **Bethe-Bloch** (energia depositata vs quantità di moto)
- Distribuzioni di ogni feature per tipo di particella
- Distribuzione delle classi (per verificare eventuali sbilanciamenti)
- Matrice di correlazione tra le feature

### Fase 2 - Baseline a tagli fisici

- Implementa la PID "tradizionale" usata in fisica sperimentale
- Per ogni classe, calcola l'intervallo [10°, 90°] percentile di ogni feature
- Classifica un evento in base a quante feature cadono nell'intervallo atteso
- In caso di parità, calcola la distanza dal centroide
- Questo scenario viene impostato come **benchmark** da comparare con le tecniche di Machine Learning e deep learning

### Fase 3 - Modelli di ML classici

- Addestra e confronta 5 modelli: Logistic Regression, K-NN, Decision Tree,
  Random Forest e XGBoost
- Cross-validation stratificata (di default 5-fold)
- Analisi della feature importance (quali segnali del rivelatore contano di più)

### Fase 4 - Deep Learning (MLP)

- Rete neurale densa (Multi-Layer Perceptron) con PyTorch
- Architettura: 64 → 128 → 256 → 64 neuroni, con BatchNorm e Dropout
- Early stopping sulla validation loss (default 20 epoche di pazienza)
- Salvataggio automatico del miglior modello

### Fase 5 - Interpretabilità e Uncertainty

- **SHAP values**: analizza quali feature guidano la predizione per ogni tipo di particella (TreeExplainer per RF/XGBoost, KernelExplainer per MLP)
- **MC Dropout**: esegue 50 forward pass con dropout attivo per stimare l'incertezza del modello su ogni evento
- **Rejection curve**: mostra come migliora l'accuracy se si scartano gli eventi ad alta incertezza

### Fase 6 - Valutazione finale

- Tabella comparativa di tutti i modelli (accuracy, precision (macro/weighted), recall (macro/weighted), F1-score (macro/weighted) e AUC-ROC)
- Matrice di confusione per ogni modello
- Curve ROC one-vs-rest per ogni classe
- Classification report dettagliato
- Tutti i risultati venfono salvati al path: `outputs/results/`

---

## 7. Configurazione

Tutti i parametri sono configurati in **`config.yaml`**. Puoi modificarli senza toccare il codice. Se non sai cosa modificare, lascia le impostazioni di default. Le sezioni principali:

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
  hidden_layers: [128, 256, 512, 128]   # rete più profonda
  dropout: 0.4                          # Dropout più marcato
  epochs: 500                           # Maggiore numero di epoche
```

### Esempio: usare un sottoinsieme del dataset

```yaml
dataset:
  max_samples: 500000   # usa solo 500k eventi su 5M
```

---

## 8. Output prodotti

Dopo un'esecuzione completa, troverai in `outputs/`:

### Grafici (`outputs/figures/*/`)

| File | Descrizione |
|------|-------------|
| `bethe_bloch.png` | Diagramma energia vs quantità di moto per le 4 particelle |
| `feature_distributions.png` | Distribuzioni di ogni feature per classe |
| `class_distribution.png` | Bilanciamento delle classi |
| `correlation_matrix.png` | Correlazioni tra feature |
| `feature_importance.png` | Importanza delle feature (LR, DT, RF, XGBoost) |
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
| `cube_separability_*.png` | Mapping tridimensionale delle feature per diversi modelli |
| `model_*_comparison.png` | Comparazione di una metrica tra modelli |
| `model_comparison_groups.png` | Comparazione di gruppo per più metriche tra più modelli |

### Risultati (`outputs/results/`)

| File | Descrizione |
|------|-------------|
| `model_comparison.csv` | Tabella con tutte le metriche per ogni modello |
| `report_*.txt` | Classification report dettagliato per modello |
| `cube_separability_*.txt` | Distanza intra-classe ed inter-classe tra feature |
| `report_model_comparison.txt` | Report testuale finale (confronto tra modelli) |

### Logs (`outputs/logs/`)

| `run.log` | Log completo dell'esecuzione |

### Modelli (`outputs/models/`)

| File | Descrizione |
|------|-------------|
| `mlp_best.pt` | Pesi del miglior MLP (checkpoint PyTorch) |

---
Per qualsiasi esigenza, i riferimenti restano:
##### Giuseppe Lorenzo Di Prima
###### Ph.D. Sistemi Intelligenti per l’Ingegneria
###### Università degli Studi di Enna Kore, Italy
##### giuseppelorenzo.diprima@unikorestudent.it
