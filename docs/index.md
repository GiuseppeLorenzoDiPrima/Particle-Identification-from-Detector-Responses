# Particle Identification from Detector Responses — Documentazione

Documentazione completa del progetto di Machine Learning e Deep Learning applicato alla fisica delle particelle: identificazione di 4 specie di particelle a partire dalle risposte simulate di 6 rivelatori.

---

## Indice generale

### Guida utente

| Documento | Descrizione |
|---|---|
| [Panoramica del progetto](overview.md) | Contesto fisico, obiettivi e architettura generale |
| [Installazione](installation.md) | Requisiti, dipendenze e configurazione dell'ambiente |
| [Utilizzo](usage.md) | Esecuzione della pipeline, argomenti da riga di comando |
| [Configurazione](configuration.md) | Riferimento completo del file `config/config.yaml` |
| [Pipeline](pipeline.md) | Descrizione dettagliata delle 6 fasi della pipeline |
| [Output](outputs.md) | Struttura degli output: figure, report, modelli, log |

### Riferimento API

| Modulo | Descrizione |
|---|---|
| [API — Panoramica](api/index.md) | Indice di tutti i moduli e le loro funzioni |
| [data\_classes](api/data_classes.md) | Download, caricamento e preprocessing del dataset |
| [models.baseline](api/models_baseline.md) | Classificatore a tagli (cuts-based PID) |
| [models.classical\_models](api/models_classical.md) | Modelli ML classici: LR, KNN, DT, RF, XGBoost |
| [models.deep\_learning](api/models_deep_learning.md) | MLP con PyTorch, training con early stopping |
| [utils.evaluation](api/utils_evaluation.md) | Metriche, tabella comparativa, report finale |
| [utils.interpretability](api/utils_interpretability.md) | Analisi SHAP (TreeExplainer, KernelExplainer) |
| [utils.uncertainty](api/utils_uncertainty.md) | MC Dropout per uncertainty quantification |
| [plot.visualization](api/plot_visualization.md) | Tutte le funzioni di visualizzazione |

---

## Avvio rapido

```bash
# Installazione dipendenze
pip install -r requirements.txt

# Pipeline completa
python main.py

# Solo una fase (es. fase 3 — modelli classici)
python main.py --phase 3

# Run veloce con 100k campioni
python main.py --quick
```

---

## Struttura del progetto

```
Particle-Identification-from-Detector-Responses/
├── main.py                      # Entry point della pipeline
├── config/
│   └── config.yaml              # Configurazione centralizzata
├── data_classes/
│   ├── __init__.py
│   └── data_loader.py           # Download e preprocessing dataset
├── models/
│   ├── __init__.py
│   ├── baseline.py              # Classificatore a tagli
│   ├── classical_models.py      # LR, KNN, DT, RF, XGBoost
│   └── deep_learning.py         # MLP con PyTorch
├── utils/
│   ├── __init__.py
│   ├── evaluation.py            # Metriche e report
│   ├── interpretability.py      # Analisi SHAP
│   └── uncertainty.py           # MC Dropout
├── plot/
│   ├── __init__.py
│   └── visualization.py         # Visualizzazioni matplotlib
├── docs/                        # Questa documentazione
└── outs/                        # Output generati (auto-creata)
    ├── imgs/                    # Figure e grafici
    ├── models/                  # Checkpoint modelli
    ├── results/                 # Report CSV e TXT
    └── logs/                    # File di log
```

---

## Risultati ottenuti

| Modello | Accuracy (test) | CV Accuracy |
|---|---|---|
| XGBoost | **97.10%** | 97.08% |
| Random Forest | 96.72% | 96.65% |
| MLP (PyTorch) | 96.XX% | — |
| Decision Tree | 95.XX% | 95.XX% |
| K-NN | 93.XX% | — |
| Logistic Regression | 88.XX% | — |
| Cuts-Based PID | ~70% | — |

> I valori esatti dipendono dalla versione del dataset e dalla configurazione usata.

---

## Licenza

Distribuito sotto licenza [MIT](../LICENSE).  
Autore: **Giuseppe Lorenzo Di Prima**
