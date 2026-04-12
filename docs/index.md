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
│
├── main.py                  # Entry point
├── requirements.txt         # Dipendenze Python
├── prepare.bat              # Script di setup automatico (Windows)
├── prepare.sh               # Script di setup automatico (Linux/MacOS)
├── README.md                # Questo file
├── LICENSE                  # MIT License
├── .gitignore
│
├── data_classes/
│   └── data_loader.py       # Download da Kaggle + preprocessing + suddivisione in sets
│
├── models/                  
│   ├── baseline.py          # PID tradizionale a tagli
│   ├── classical_models.py  # LR, KNN, Decision Tree, Random Forest e XGBoost
│   └── deep_learning.py     # MLP con Framework PyTorch
│
├── utils/                 
│   ├── evaluation.py        # Metriche, tabella comparativa, report, ecc...
│   ├── interpretability.py  # Analisi SHAP values
│   └── uncertainty.py       # MC Dropout e uncertainty quantification
│
├── plot/                  
│   └── visualization.py     # Grafici: Bethe-Bloch, distribuzioni, ROC, CM, ecc...
│
├── config/
│   └── config.yaml          # Configurazione centralizzata
│
├── data/                    # Dataset CSV (scaricato automaticamente)
│
├── docs/                    # Documentazione completa del progetto
│
├── .github/                 # Immagini per il README.md (questo file)
│
└── outs/
    ├── imgs/                # Tutti i grafici generati (.png)
    ├── models/              # Modelli salvati (es. mlp_best.pt)
    ├── logs/                # Log di addestramento
    └── results/             # Tabelle CSV e report testuali
```

---

## Risultati ottenuti

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

> I valori esatti dipendono dalla versione del dataset e dalla configurazione usata.

---

## Licenza


**🔓 MIT License**  
Questo progetto è distribuito sotto licenza MIT, una licenza open source semplice e permissiva che consente a chiunque di utilizzare, modificare e distribuire il codice liberamente. È possibile impiegare questo software anche per scopi commerciali, a condizione che venga inclusa la nota di copyright originale. L'autore sarebbe lieto di essere citato qualora questo progetto venga riutilizzato.

---

## Contatti

**👤 Giuseppe Lorenzo Di Prima**, ORCID: [Giuseppe Lorenzo Di Prima](https://orcid.org/0009-0002-9470-9370)<br>🎓 Ph.D. in Sistemi Intelligenti per l’Ingegneria<br>[🏫 Università degli Studi di Enna Kore, Italy](https://www.uke.it)<br>✉️ [giuseppelorenzo.diprima@unikorestudent.it](mailto:giuseppelorenzo.diprima@unikorestudent.it)
