# Particle Identification via Machine Learning
## Machine Learning applicato alla Fisica delle Particelle

> Classificazione multi-classe di particelle (e, K, π, p) attraverso l'utilizzo di risposte di rivelatori simulate con metodo Monte Carlo. Applicazione di modelli di Machine Learning e Deep Learning.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Pipeline completa per Particle IDentification (PID) basata su:
- 5M eventi simulati (Monte Carlo)
- Confronto tra approccio fisico tradizionale e ML/DL
- Miglior modello: XGBoost con accuracy ~97.10%

###### **Obiettivo: dimostrare il vantaggio quantitativo del ML rispetto all'approccio cut-based**.

##### Risultati ottenuti

<div align="center">

<table>
  <tr>
    <th>Modello</th>
    <th>Accuracy</th>
    <th>F1 Macro</th>
    <th>Precision Macro</th>
    <th>Recall Macro</th>
    <th>AUC ROC Macro</th>
    <th>CV Accuracy</th>
    <th>Train Time (s)</th>
  </tr>

  <tr>
    <td>XGBoost</td>
    <td>0.9710</td>
    <td>0.8625</td>
    <td>0.8880</td>
    <td>0.8406</td>
    <td>0.9951</td>
    <td>0.9711</td>
    <td>317.90</td>
  </tr>

  <tr>
    <td>K-NN</td>
    <td>0.9681</td>
    <td>0.8468</td>
    <td>0.8843</td>
    <td>0.8172</td>
    <td>0.9680</td>
    <td>0.9679</td>
    <td>16.50</td>
  </tr>

  <tr>
    <td>Random Forest</td>
    <td>0.9475</td>
    <td>0.7828</td>
    <td>0.7165</td>
    <td>0.9375</td>
    <td>0.9946</td>
    <td>0.9483</td>
    <td>724.30</td>
  </tr>

  <tr>
    <td>Decision Tree</td>
    <td>0.9398</td>
    <td>0.7402</td>
    <td>0.6792</td>
    <td>0.9313</td>
    <td>0.9757</td>
    <td>0.9404</td>
    <td>70.10</td>
  </tr>

  <tr>
    <td>MLP (PyTorch)</td>
    <td>0.9249</td>
    <td>0.6954</td>
    <td>0.6463</td>
    <td>0.9430</td>
    <td>0.9942</td>
    <td>—</td>
    <td>10307.70</td>
  </tr>

  <tr>
    <td>Logistic Regression</td>
    <td>0.8062</td>
    <td>0.5555</td>
    <td>0.5562</td>
    <td>0.8385</td>
    <td>0.9614</td>
    <td>0.8062</td>
    <td>32.90</td>
  </tr>

  <tr>
    <td>Cuts-Based PID</td>
    <td>0.4433</td>
    <td>0.3272</td>
    <td>0.4474</td>
    <td>0.5520</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
  </tr>

</table>

</div>

<p align="center">
  <img src="github_icon\bethe_bloch.png" height="220" width="45%">
  <img src="github_icon\correlation_matrix.png" height="220" width="45%">
</p>

<p align="center">
  <img src="github_icon\model_comparison_groups.png" height="220" width="45%">
  <img src="github_icon\SHAP_summary_xgboost.png" height="220" width="45%">
</p>

<p align="center">
  <em>Top row: Bethe-Bloch and Correlation Matrix. Bottom row: Model Comparison and SHAP Summary (XGBoost).</em>
</p>

##### Key Insights

- XGBoost si attesta come miglior modello, superando le performance dei metodi tradizionali del circa 53% in termini di accuratezza.
- I modelli tree-based (XGBoost, Random Forest, Decision Tree) superano la baseline in accuratezza del ~50%
- Le feature più discriminanti sono: velocità ridotta e quantità di moto
- L'incertezza (MC Dropout) evidenzia l'andamento dell'accuratezza man mano vengono scartati eventi incerti

##### Reproducibility
Per garantire la riproducibilità del progetto, tutti gli esperimenti sono stati condotto con un seed fisso (default: 42).

---
## Introduzione
Benvenuta/o. Il presente progetto è stato sviluppato nell’ambito della preparazione all’esame di Principi di Calcolo Tensoriale, previsto dal piano di studi del Corso di Dottorato di Ricerca in Sistemi Intelligenti per l’Ingegneria presso l’Università degli Studi di Enna Kore.

L’elaborato propone l’applicazione di tecniche di Machine Learning e Deep Learning al contesto della fisica delle particelle. In particolare, il problema affrontato riguarda la classificazione multi-classe: l’obiettivo consiste nell’identificare la natura di una particella in ingresso (tra elettrone, pione, kaone e protone) sulla base di sei risposte simulate attraverso il metodo Monte Carlo (simulazione di sei rivelatori).

I dati utilizzati provengono dalla piattaforma Open-Source [Kaggle](https://www.kaggle.com/database/naharrison/particle-identification-from-detector-responses) e consistono in 5 milioni di samples, ciascuno descritto da sei features. Tali dati sono stati generati mediante simulazioni basate sul metodo Monte Carlo.

A tal fine, vengono messi a confronto l’approccio tradizionale, basato su tagli selettivi (range percentili), e modelli avanzati di apprendimento automatico e profondo, al fine di valutarne le prestazioni e l’efficacia nel contesto considerato.

---

## Indice

1. [Prerequisiti](#1-prerequisiti)
2. [Installazione](#2-installazione)
3. [Esecuzione](#3-esecuzione)
4. [Struttura del progetto](#4-struttura-del-progetto)
5. [Le 6 fasi della pipeline](#5-le-6-fasi-della-pipeline)
6. [Configurazione](#6-configurazione)
7. [Output prodotti](#7-output-prodotti)
8. [Licenza MIT](#8-license)
9. [Contatti](#9-contatti)

---

## 1. Prerequisiti

- **Python 3.10+** installato e disponibile da terminale (`python --version`)
- **Connessione internet** (per scaricare il dataset da Kaggle alla prima esecuzione)
- **Account Kaggle** gratuito (serve per le credenziali API)

Il dataset viene scaricato automaticamente da Kaggle tramite API. Se non è stato configurato il file **kaggle.json** in precedenza, è necessario:

### Passo 1 - Creare un account Kaggle

Se non si dispone già di un account, registrarsi su [kaggle.com](https://www.kaggle.com/) (è gratuito).

### Passo 2 - Generare la API Key

1. Creare una cartella al path:

```
C:\Users\<TUO_UTENTE>\.kaggle
```

2. Navigare alla pagina [kaggle.com/settings](https://www.kaggle.com/settings)
3. Scorrere fino alla sezione **API**
4. Cliccare su **"Create New Token"**
5. Conservare la chiave esadecimale appena creata

### Passo 3 - Configurare la API

6. Andare al path `C:\Users\<TUO_UTENTE>\.kaggle` e creare, al suo interno, un file `kaggle.json`
7. Inserire, all'interno del file `C:\Users\<TUO_UTENTE>\.kaggle\kaggle.json` il contenuto:

```json
{"username": "il_tuo_username", "key": "una_stringa_esadecimale"}
```

### Passo 4 - Verificare il funzionamento

Dopo aver configurato il file, è possibile verificare che funzioni semplicemente aprendo un terminale e digitando:

```bash
pip install kaggle
kaggle datasets list
```

Se compare una lista di dataset, le credenziali sono configurate correttamente.

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

> **Importante:** assicurarsi di aver attivato il virtual environment prima di eseguire qualsiasi comando (una cartella denominata `.venv` deve comparire nella root del progetto). Se non compare, eseguire:

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

### Pipeline veloce (consigliata per il primo test)

Eseguire le 6 fasi in sequenza solo su 100k campioni e meno epoche di training. Ideale per verificare che tutto funzioni prima dell'esecuzione completa:

```bash
python main.py --quick
```

### Eseguire una singola fase
In questo modo è possibile eseguire una sola fase:

```bash
python main.py --phase 1    # Solo visualizzazione esplorativa
python main.py --phase 2    # Solo baseline a tagli
python main.py --phase 3    # Solo modelli ML classici
python main.py --phase 4    # Solo deep learning (MLP)
python main.py --phase 5    # Solo interpretabilità e uncertainty
python main.py --phase 6    # Solo valutazione finale e confronto
```

### Eseguire più fasi selezionate
In questo modo è possibile eseguire più fasi:

```bash
python main.py --phases 1 2 3    # Fasi 1, 2 e 3
```

### Usare una configurazione custom
È anche possibile impostare delle configurazioni custom digitando:

```bash
python main.py --config mia_config.yaml
```

---

## 4. Struttura del progetto

```
Particle-Identification-from-Detector-Responses/
│
├── main.py                  # Entry point
├── config/
│   └── config.yaml          # Configurazione centralizzata
├── requirements.txt         # Dipendenze Python
├── prepare.bat              # Script di setup automatico (Windows)
├── prepare.sh               # Script di setup automatico (Linux/MacOS)
├── README.md                # Questo file
├── LICENSE                  # MIT License
├── .gitignore
│
├── src/                     # Codice sorgente modulare
│   ├── __init__.py
│   ├── data_loader.py       # Download da Kaggle + preprocessing + split
│   ├── visualization.py     # Grafici: Bethe-Bloch, distribuzioni, ROC, CM, ecc...
│   ├── baseline.py          # PID tradizionale a tagli
│   ├── classical_models.py  # LR, KNN, Decision Tree, Random Forest e XGBoost
│   ├── deep_learning.py     # MLP con Framework PyTorch
│   ├── evaluation.py        # Metriche, tabella comparativa, report, ecc...
│   ├── interpretability.py  # Analisi SHAP values
│   └── uncertainty.py       # MC Dropout e uncertainty quantification
│
├── data/                    # Dataset CSV (scaricato automaticamente)
│
├── github_icon/             # Immagini per il README.md (questo file)
│
└── outputs/
    ├── figures/             # Tutti i grafici generati (.png)
    ├── models/              # Modelli salvati (es. mlp_best.pt)
    ├── logs/                # Log di addestramento
    └── results/             # Tabelle CSV e report testuali
```

---

## 5. Le 6 fasi della pipeline

### Fase 1 - Caricamento dati e visualizzazione esplorativa

- Scarica il dataset da Kaggle (solo per la prima esecuzione)
- Genera il diagramma di **Bethe-Bloch** (energia depositata vs quantità di moto)
- Distribuzioni di ogni feature per tipo di particella
- Distribuzione delle classi completa e sui singoli sets (per verificare eventuali sbilanciamenti)
- Matrice di correlazione tra le features

### Fase 2 - Baseline a tagli fisici

- Implementa la PID "tradizionale" usata in fisica sperimentale
- Per ogni classe, calcola l'intervallo [10°, 90°] percentile di ogni feature
- Classifica un evento in base a quante features cadono nell'intervallo atteso
- In caso di parità, calcola la distanza dal centroide dell'intervallo
- Questo scenario viene utilizzato come **benchmark** da comparare con le tecniche di Machine Learning e Deep Learning implementate

### Fase 3 - Modelli di ML classici

- Addestra e confronta 5 modelli: Logistic Regression, K-NN, Decision Tree, Random Forest e XGBoost
- Cross-validation stratificata (di default 5-fold)
- Analisi della feature importance (quali features contano di più)

### Fase 4 - Deep Learning (MLP)

- Rete neurale densa (Multi-Layer Perceptron) basata sul framework PyTorch
- Architettura: 64 → 128 → 256 → 128 neuroni, con ReLU, BatchNorm e Dropout
- Early stopping sulla validation loss (default 20 epoche di pazienza)
- Salvataggio automatico del miglior modello in outputs/models

### Fase 5 - Interpretabilità e Uncertainty

- **SHAP values**: analizza quali feature guidano la predizione per ogni tipo di particella (TreeExplainer per RF/XGBoost, KernelExplainer per MLP)
- **MC Dropout**: esegue 100 forward pass con dropout attivo per stimare l'incertezza del modello su ogni evento
- **Rejection curve**: mostra come migliora l'accuracy se si scartano gli eventi ad alta incertezza
- **Density vs Entropy**: mostra la distribuzione dell'incertezza per predizioni corrette ed errate
- **Uncertainty Scatter**: mostra la mappa di incertezza nel piano Bethe-Bloch (piano energia / quantità di moto)

### Fase 6 - Valutazione finale

- Tabella comparativa di tutti i modelli (accuracy, precision (macro/weighted), recall (macro/weighted), F1-score (macro/weighted) e AUC-ROC)
- Matrice di confusione per ogni modello
- Curve ROC one-vs-rest per ogni classe
- Classification report dettagliato
- Tutti i risultati vengono salvati al path: `outputs/results/`

---

## 6. Configurazione

Tutti i parametri sono configurati in **`config/config.yaml`**. È possibile modificarli senza alterare il codice. Alcune della sezioni principali:

| Sezione | Cosa controlla |
|---------|---------------|
| `paths` | Cartelle di input/output |
| `dataset` | Slug Kaggle, split train/val/test, random seed |
| `classical_models` | Iperparametri di ciascun modello ML |
| `deep_learning` | Architettura MLP, learning rate, epoche, dropout, ecc... |
| `interpretability` | Numero di campioni per SHAP |
| `uncertainty` | Numero di iterazioni MC Dropout |
| `visualization` | DPI, dimensioni figure, stile grafico, ecc... |

### Esempio: cambiare l'architettura del MLP

In `config/config.yaml`:

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

## 7. Output prodotti

Dopo un'esecuzione della papeline (completa o rapida), alcuni degli elementi generati in `outputs/` sono:

### Grafici (`outputs/figures/*/`)

| File | Descrizione |
|------|-------------|
| `range_features.png` | Tabella riassuntiva dei range calcolati sui percentili |
| `bethe_bloch.png` | Diagramma energia vs quantità di moto per le 4 particelle |
| `feature_distributions.png` | Distribuzioni di ogni feature per classe |
| `class_distribution_*.png` | Bilanciamento delle classi per i vari sets |
| `correlation_matrix.png` | Correlazioni tra features |
| `feature_importance.png` | Importanza delle feature per i modelli che la prevedono (LR, DT, RF, XGBoost) |
| `mlp_training_history.png` | Loss e accuracy (validation) ed loss (train) durante il training MLP |
| `cm_*.png` | Matrice di confusione per ogni modello |
| `roc_*.png` | Curve ROC per ogni modello |
| `shap_summary_*.png` | SHAP plot per modello |
| `SHAP_bar_*.png` | SHAP feature importance |
| `SHAP_*_class_*.png` | SHAP plot per classe |
| `uncertainty_entropy.png` | Distribuzione dell'incertezza: corretti vs errati |
| `rejection_curve.png` | Andamento dell'accuratezza vs percentuale di eventi incerti accettati |
| `uncertainty_per_class.png` | Incertezza per tipo di particella |
| `uncertainty_scatter.png` | Mappa di incertezza nel piano Bethe-Bloch (energia vs quantità di moto) |
| `model_comparison_groups.png` | Confronto complessivo tra metriche per tutti i modelli |
| `model_*_comparison.png` | Confronto per metrica tra tutti i modelli |
| `cube_separability_*.png` | Mapping tridimensionale delle features per diversi modelli |

### Risultati (`outputs/results/`)

| File | Descrizione |
|------|-------------|
| `model_comparison.csv` | CSV con i risultati ottenuti per ciascuna metrica per ogni modello |
| `report_*.txt` | Classification report dettagliato per modello |
| `cube_separability_*.txt` | Distanza intra-classe ed inter-classe tra features per tutti i modelli |
| `report_model_comparison.txt` | Tabella dei risultati finale (confronto tra modelli) |

### Logs (`outputs/logs/`)

| File | Descrizione |
|------|-------------|
| `run.log` | Log completo dell'esecuzione (con info orario) |

### Modelli (`outputs/models/`)

| File | Descrizione |
|------|-------------|
| `mlp_best.pt` | Pesi del miglior modello MLP (checkpoint PyTorch) |

---

## 8. License

**🔓 MIT License**  
Questo progetto è distribuito sotto licenza MIT, una licenza open source semplice e permissiva che consente a chiunque di utilizzare, modificare e distribuire il codice liberamente. È possibile impiegare questo software anche per scopi commerciali, a condizione che venga inclusa la nota di copyright originale. L'autore sarebbe lieto di essere citato qualora questo progetto venga riutilizzato.

---

## 9. Contatti

**👤 Giuseppe Lorenzo Di Prima**, ORCID: [Giuseppe Lorenzo Di Prima](https://orcid.org/0009-0002-9470-9370)<br>🎓 Ph.D. in Sistemi Intelligenti per l’Ingegneria<br>[🏫 Università degli Studi di Enna Kore, Italy](https://www.uke.it)<br>✉️ [giuseppelorenzo.diprima@unikorestudent.it](mailto:giuseppelorenzo.diprima@unikorestudent.it)
