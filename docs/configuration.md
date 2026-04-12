# Configurazione

Tutta la configurazione del progetto è centralizzata nel file `config/config.yaml`. Ogni parametro può essere modificato senza toccare il codice sorgente.

---

## Sezione `paths` — Percorsi directory

Definisce dove vengono letti i dati e dove vengono salvati tutti gli output.

```yaml
paths:
  data_dir: "data"
  output_dir: "outs"
  figures_dir: "outs/imgs"
  models_dir: "outs/models"
  results_dir: "outs/results"
  log_dir: "outs/logs"
```

| Chiave | Default | Descrizione |
|---|---|---|
| `data_dir` | `"data"` | Directory dove viene salvato il dataset scaricato da Kaggle |
| `output_dir` | `"outs"` | Directory radice di tutti gli output |
| `figures_dir` | `"outs/imgs"` | Directory radice delle figure; ogni tipo di grafico va in una sottocartella |
| `models_dir` | `"outs/models"` | Directory dove viene salvato il checkpoint della MLP (`mlp_best.pt`) |
| `results_dir` | `"outs/results"` | Directory dei report testuali e della tabella CSV comparativa |
| `log_dir` | `"outs/logs"` | Directory del file di log (`run.log`) |

Tutte le directory vengono create automaticamente (`os.makedirs(..., exist_ok=True)`) se non esistono.

---

## Sezione `dataset` — Dataset Kaggle

```yaml
dataset:
  kaggle_slug: "naharrison/particle-identification-from-detector-responses"
  filename: "pid-5M.csv"
  test_size: 0.15
  val_size: 0.2
  random_state: 42
  max_samples: null
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `kaggle_slug` | `str` | `"naharrison/..."` | Identificatore del dataset su Kaggle nel formato `utente/nome-dataset` |
| `filename` | `str` | `"pid-5M.csv"` | Nome del file CSV dentro l'archivio zip scaricato |
| `test_size` | `float` | `0.15` | Frazione del dataset totale riservata al test set (15%) |
| `val_size` | `float` | `0.2` | Frazione del training set riservata alla validation (20% del train, pari a ~17% del totale) |
| `random_state` | `int` | `42` | Seed per la riproducibilità degli split e di tutti i generatori casuali |
| `max_samples` | `int` o `null` | `null` | Se specificato, limita il dataset a questo numero di eventi (subsample casuale). `null` usa tutto il dataset |

### Nota sullo split

Lo split avviene in due passi:
1. `train+val` / `test`: dimensione test = `test_size` × N_totale
2. `train` / `val`: dimensione val = `val_size` × N_train

Con i valori di default su 5M eventi:
- **Test set:** 750.000 eventi (15%)
- **Validation set:** ~680.000 eventi (~17% del totale)
- **Training set:** ~3.570.000 eventi (~68% del totale)

Tutti gli split sono **stratificati** per mantenere la proporzione delle classi.

---

## Sezione `features` — Nomi delle variabili

```yaml
features:
  names:
    - "p"
    - "theta"
    - "beta"
    - "nphe"
    - "ein"
    - "eout"
  target: "id"
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `names` | `list[str]` | Vedi sopra | Lista delle colonne del CSV da usare come feature. Se non specificato, vengono usate tutte le colonne tranne il target |
| `target` | `str` | `"id"` | Nome della colonna target nel CSV (PDG ID numerico) |

> **Nota:** Attualmente nel codice `feature_names` viene derivata dal CSV come tutte le colonne escluso il target.

---
## Sezione `baseline_cuts` — Baseline a tagli

```yaml
baseline_cuts:
  enabled: true
  low_percentile: 10
  high_percentile: 90
  show_range: true
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Se `false`, la Fase 2 viene saltata completamente |
| `low_percentile` | `int` | `10` | Percentile inferiore usato per calcolare il limite basso dell'intervallo accettazione per ogni feature e classe |
| `high_percentile` | `int` | `90` | Percentile superiore usato per il limite alto dell'intervallo |
| `show_range` | `bool` | `true` | Se `true`, stampa a terminale la tabella dei range calcolati |

### Logica dei tagli

Per ogni classe $c$ e feature $j$, viene calcolato l'intervallo:
$$[\text{percentile}_{low}(X_j | y=c),\; \text{percentile}_{high}(X_j | y=c)]$$

Un evento viene assegnato alla classe con il maggior numero di feature che rientrano nell'intervallo. In caso di parità: distanza euclidea dal centroide.

---

## Sezione `classical_models` — Modelli ML classici

Ogni sottochiave corrisponde a un modello. Il campo `enabled: false` disabilita completamente quel modello (non viene istanziato, addestrato né valutato).

```yaml
classical_models:
  logistic_regression:
    enabled: true
    max_iter: 1500
    multi_class: "multinomial"
    solver: "lbfgs"
    class_weight: "balanced"
  knn:
    enabled: true
    n_neighbors: 7
  decision_tree:
    enabled: true
    max_depth: 15
    class_weight: "balanced"
  random_forest:
    enabled: true
    n_estimators: 300
    max_depth: 15
    n_jobs: -1
    class_weight: "balanced"
  xgboost:
    enabled: true
    n_estimators: 300
    max_depth: 15
    learning_rate: 0.1
    n_jobs: -1
    eval_metric: "mlogloss"
```

### `logistic_regression`

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Abilita/disabilita il modello |
| `max_iter` | `int` | `1500` | Numero massimo di iterazioni del solver LBFGS |
| `multi_class` | `str` | `"multinomial"` | Strategia multiclasse. `"multinomial"` minimizza la cross-entropy su tutte le classi simultaneamente |
| `solver` | `str` | `"lbfgs"` | Algoritmo di ottimizzazione. `"lbfgs"` è adatto a dataset di medie dimensioni |
| `class_weight` | `str` | `"balanced"` | Pesi inversamente proporzionali alla frequenza delle classi per gestire lo sbilanciamento |

### `knn`

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Abilita/disabilita il modello |
| `n_neighbors` | `int` | `7` | Numero di vicini da considerare per la classificazione |

### `decision_tree`

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Abilita/disabilita il modello |
| `max_depth` | `int` | `15` | Profondità massima dell'albero. Limita l'overfitting |
| `class_weight` | `str` | `"balanced"` | Pesi per gestire lo sbilanciamento delle classi |

Il `random_state` viene automaticamente letto da `dataset.random_state`.

### `random_forest`

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Abilita/disabilita il modello |
| `n_estimators` | `int` | `300` | Numero di alberi nella foresta |
| `max_depth` | `int` | `15` | Profondità massima di ogni albero |
| `n_jobs` | `int` | `-1` | Parallelismo. `-1` usa tutti i core disponibili |
| `class_weight` | `str` | `"balanced"` | Pesi per gestire lo sbilanciamento |

Il `random_state` viene automaticamente letto da `dataset.random_state`.

### `xgboost`

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Abilita/disabilita il modello |
| `n_estimators` | `int` | `300` | Numero di round di boosting |
| `max_depth` | `int` | `15` | Profondità massima di ogni albero base |
| `learning_rate` | `float` | `0.1` | Tasso di apprendimento (shrinkage). Valori più bassi richiedono più estimatori |
| `n_jobs` | `int` | `-1` | Parallelismo |
| `eval_metric` | `str` | `"mlogloss"` | Metrica di valutazione interna. `"mlogloss"` = multiclass log-loss |

Il `random_state` viene automaticamente letto da `dataset.random_state`.

---

## Sezione `cross_validation` — Cross-validation

```yaml
cross_validation:
  enabled: true
  n_folds: 5
  shuffle: true
  stratified: true
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Se `false`, salta la cross-validation e addestra direttamente sul training set completo |
| `n_folds` | `int` | `5` | Numero di fold per la k-fold stratificata |
| `shuffle` | `bool` | `true` | Se `true`, i dati vengono mescolati prima di creare i fold |
| `stratified` | `bool` | `true` | Attualmente è possibile scegliere tra `StratifiedKFold` o `KFold` in base al parametro stratified del file di configurazione (rispettivamente `true` o `false`.) |

---

## Sezione `deep_learning` — MLP con PyTorch

```yaml
deep_learning:
  hidden_layers: [64, 128, 256, 128]
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 512
  epochs: 100
  early_stopping_patience: 20
  weight_decay: 0.0001
  show_architecture: true
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `hidden_layers` | `list[int]` | `[64, 128, 256, 128]` | Lista delle dimensioni dei layer nascosti. Ogni elemento crea: Linear → BatchNorm1d → ReLU → Dropout |
| `dropout` | `float` | `0.3` | Tasso di dropout applicato dopo ogni layer nascosto. Usato anche per MC Dropout durante l'inferenza |
| `learning_rate` | `float` | `0.001` | Learning rate per l'ottimizzatore Adam |
| `batch_size` | `int` | `512` | Dimensione del mini-batch per il training e la valutazione |
| `epochs` | `int` | `100` | Numero massimo di epoche di training |
| `early_stopping_patience` | `int` | `20` | Numero di epoche anche NON consecutive senza miglioramento della validation loss prima di fermare il training |
| `weight_decay` | `float` | `0.0001` | Coefficiente di regolarizzazione L2 (weight decay) nell'ottimizzatore Adam |
| `show_architecture` | `bool` | `true` | Se `true`, stampa il summary dell'architettura MLP all'inizio del training |

### Architettura MLP con i valori di default

```
Input (6) → Linear(6,64) → BN(64) → ReLU → Dropout(0.3)
          → Linear(64,128) → BN(128) → ReLU → Dropout(0.3)
          → Linear(128,256) → BN(256) → ReLU → Dropout(0.3)
          → Linear(256,128) → BN(128) → ReLU → Dropout(0.3)
          → Linear(128,4)   [logits]
```

---

## Sezione `interpretability` — Analisi SHAP

```yaml
interpretability:
  enabled: true
  shap_samples: 1500
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Se `false`, la Fase 5a (SHAP) viene saltata completamente |
| `shap_samples` | `int` | `1500` | Numero di campioni del test set usati per il calcolo degli SHAP values. Aumentare per stime più accurate, ridurre per velocità. Identico ragionamento per il KernelExplainer della MLP, vengono usati al massimo `shap_samples_kernel_explainer` campioni indipendentemente da questo valore |

---

## Sezione `uncertainty` — MC Dropout

```yaml
uncertainty:
  enabled: true
  mc_dropout_iterations: 100
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `enabled` | `bool` | `true` | Se `false`, la Fase 5b (MC Dropout) viene saltata completamente |
| `mc_dropout_iterations` | `int` | `100` | Numero di forward pass con dropout attivo. Aumentare per stime di incertezza più stabili |

---

## Sezione `visualization` — Visualizzazioni

```yaml
visualization:
  graph: true
  dpi: 600
  figsize: [12, 8]
  style: "seaborn-v0_8-whitegrid"
  palette: "Set2"
  comparison_metrics:
    - "accuracy"
    - "precision_macro"
    - "recall_macro"
    - "f1_macro"
    - "precision_weighted"
    - "recall_weighted"
    - "f1_weighted"
  comparison_group_metrics:
    - "accuracy"
    - "precision_macro"
    - "recall_macro"
    - "f1_macro"
```

| Chiave | Tipo | Default | Descrizione |
|---|---|---|---|
| `graph` | `bool` | `true` | Se `false`, disabilita la generazione di tutti i grafici (utile per esecuzioni puramente numeriche) |
| `dpi` | `int` | `600` | Risoluzione delle figure salvate in DPI. 600 è adatto per la stampa di qualità pubblicazione |
| `figsize` | `list[int, int]` | `[12, 8]` | Dimensioni di default delle figure in pollici `[larghezza, altezza]`. Nota: alcune funzioni sovrascrivono questo valore per dimensioni più appropriate |
| `comparison_metrics` | `list[str]` | Vedi sopra | Lista di metriche per cui generare grafici a barre individuali (`plot_metrics_comparison`) |
| `comparison_group_metrics` | `list[str]` | Vedi sopra | Lista di metriche da includere nel grafico a barre raggruppate (`plot_metric_groups_comparison`) |

### Metriche disponibili per `comparison_metrics` e `comparison_group_metrics`

| Metrica | Descrizione |
|---|---|
| `accuracy` | Accuracy globale |
| `f1_macro` | F1-score macro (media non pesata per classe) |
| `f1_weighted` | F1-score pesato per la frequenza delle classi |
| `precision_macro` | Precision macro |
| `precision_weighted` | Precision pesata |
| `recall_macro` | Recall macro |
| `recall_weighted` | Recall pesata |
| `auc_roc_macro` | AUC-ROC macro (solo per modelli con `predict_proba`) |
| `auc_roc_weighted` | AUC-ROC pesata |
