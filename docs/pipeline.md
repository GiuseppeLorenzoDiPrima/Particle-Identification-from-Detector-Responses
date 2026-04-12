# Pipeline

La pipeline è composta da **6 fasi sequenziali**, orchestrate da `main.py`. Ogni fase può essere eseguita indipendentemente tramite `--phase` o `--phases`.

---

## Fase 1 — Caricamento dati e visualizzazione esplorativa

**Moduli coinvolti:** `data_classes.data_loader`, `plot.visualization`

### Operazioni

1. **Download del dataset** da Kaggle (solo se non già presente in `data/`)
2. **Caricamento CSV** in un DataFrame pandas
3. **Subsample opzionale** (se `max_samples` è specificato in config)
4. **Rimozione NaN** e validazione dell'integrità dei dati
5. **Mappatura PDG → nome fisico:** i PDG ID numerici (-11, 211, 321, 2212) vengono convertiti nei nomi delle particelle (elettrone, pione, kaone, protone)
6. **Label encoding:** i nomi delle classi vengono codificati come interi 0–3 con `sklearn.LabelEncoder`
7. **Split stratificato:** 85% train+val / 15% test, poi 80% train / 20% val
8. **Standardizzazione:** `StandardScaler` fittato **solo** sul training set, poi applicato a validation e test

#### Dizionario `data` prodotto

Tutte le fasi successive ricevono il dizionario `data` con le seguenti chiavi:

| Chiave | Tipo | Descrizione |
|---|---|---|
| `X_train` | `np.ndarray` (N_train, 6) | Feature scalate — training set |
| `X_val` | `np.ndarray` (N_val, 6) | Feature scalate — validation set |
| `X_test` | `np.ndarray` (N_test, 6) | Feature scalate — test set |
| `y_train` | `np.ndarray` (N_train,) | Label encoded — training set |
| `y_val` | `np.ndarray` (N_val,) | Label encoded — validation set |
| `y_test` | `np.ndarray` (N_test,) | Label encoded — test set |
| `X_train_raw` | `np.ndarray` (N_train, 6) | Feature **non** scalate — training set |
| `X_val_raw` | `np.ndarray` (N_val, 6) | Feature **non** scalate — validation set |
| `X_test_raw` | `np.ndarray` (N_test, 6) | Feature **non** scalate — test set |
| `feature_names` | `list[str]` | Nomi delle feature: `["p", "theta", "beta", "nphe", "ein", "eout"]` |
| `feature_for_print` | `list[str]` | Simboli Unicode per terminale: `["p", "θ", "β", "nₚₕₑ", "Eᵢₙ", "Eₒᵤₜ"]` |
| `label_encoder` | `LabelEncoder` | Encoder sklearn fittato sulle classi |
| `scaler` | `StandardScaler` | Scaler sklearn fittato sul training set |
| `df` | `pd.DataFrame` | DataFrame originale completo (con nomi particelle mappati) |
| `class_names` | `list[str]` | Nomi classi capitalizzati in ordine label encoder |

### Visualizzazioni generate

- `outs/imgs/pre-processing/bethe_bloch.png` — Scatter plot $E_{in}$ vs $p$ per classe
- `outs/imgs/pre-processing/feature_distributions.png` — 6 istogrammi delle distribuzioni feature per classe
- `outs/imgs/pre-processing/class_distribution_train.png` — Distribuzione classi nel training set
- `outs/imgs/pre-processing/class_distribution_val.png` — Distribuzione classi nel validation set
- `outs/imgs/pre-processing/class_distribution_test.png` — Distribuzione classi nel test set
- `outs/imgs/pre-processing/class_distribution_full.png` — Distribuzione classi sul dataset completo
- `outs/imgs/pre-processing/correlation_matrix.png` — Heatmap correlazioni tra feature

---

## Fase 2 — Baseline a tagli (Cuts-Based PID)

**Moduli coinvolti:** `models.baseline`, `plot.visualization`

### Operazioni

1. Istanzia `CutsBasedPID` con i parametri di config
2. **Fit** sul training set **non scalato** (`X_train_raw`, `y_train`): calcola percentili e centroidi per ogni classe
3. **Predizione e valutazione** sul test set **non scalato** (`X_test_raw`, `y_test`)
4. Opzionale: stampa tabella range a terminale (`show_range: true`)
5. Salva grafico tabellare dei range

> **Importante:** La baseline usa i dati RAW (non standardizzati) perché i tagli operano sulle grandezze fisiche originali. La standardizzazione sposterebbe i range e renderebbe i tagli privi di significato fisico.

### Algoritmo di classificazione

Per ogni evento $\mathbf{x}$:
1. Per ogni classe $c$ e feature $j$: conta se $x_j \in [p_{low}^{(c,j)},\; p_{high}^{(c,j)}]$
2. Assegna la classe con il punteggio (conteggio) massimo
3. In caso di parità: assegna la classe con centroide più vicino in distanza euclidea

### Output

- `outs/imgs/baseline/range_features.png` — Tabella matplotlib dei range
- Aggiornamento `all_results["Cuts-Based PID"]` con `y_pred`, `accuracy`, `model`

---

## Fase 3 — Modelli di Machine Learning classici

**Moduli coinvolti:** `models.classical_models`, `plot.visualization`

### Operazioni

1. **Costruzione modelli** (solo quelli `enabled: true` in config)
2. **Cross-validation stratificata** (se `enabled: true`) su `X_train` scalato
3. **Training** su `X_train` scalato completo
4. **Valutazione** su `X_test` scalato
5. Estrazione **feature importance** (dove disponibile)
6. Estrazione **probabilità di classe** (per curve ROC)

### Modelli e feature importance

| Modello | Feature importance | Metodo |
|---|---|---|
| Logistic Regression | Sì | Media valori assoluti dei coefficienti `coef_` su tutte le classi |
| K-NN | No | — |
| Decision Tree | Sì | `feature_importances_` (Gini) |
| Random Forest | Sì | `feature_importances_` (Gini medio) |
| XGBoost | Sì | `feature_importances_` (gain) |

### Struttura del risultato per ogni modello

```python
{
    "model": <fitted sklearn/xgboost model>,
    "model_name": "XGBoost",
    "cv_mean": 0.9708,         # Media accuracy cross-validation (None se CV disabilitata)
    "cv_std": 0.0003,          # Deviazione standard accuracy CV (None se CV disabilitata)
    "test_accuracy": 0.9710,
    "y_pred": np.ndarray,      # Predizioni sul test set (label encoded)
    "y_proba": np.ndarray,     # Probabilità (N_test, 4) — None per KNN senza predict_proba
    "train_time": 45.2,        # Secondi
    "feature_importance": dict # {nome_feature: valore} o None
}
```

### Output

- `outs/imgs/training/feature_importance.png` — Barplot feature importance per ogni modello
- Aggiornamento `all_results` con i risultati di ogni modello

---

## Fase 4 — Deep Learning (MLP)

**Moduli coinvolti:** `models.deep_learning`, `plot.visualization`

### Operazioni

1. **Selezione device:** CUDA se disponibile, altrimenti CPU
2. **Costruzione MLP** con architettura da config (`hidden_layers`, `dropout`)
3. **Calcolo class weights** inversamente proporzionali alla frequenza delle classi
4. **Training loop** con early stopping basato sulla validation loss
5. **Ripristino best model** (checkpoint con la validation loss minima)
6. **Valutazione** su test set (accuracy + loss)
7. **Salvataggio checkpoint** `mlp_best.pt`

### Loop di training

Per ogni epoca:
1. **Training:** forward pass → calcolo loss (CrossEntropy pesata) → backward → optimizer step
2. **Validation:** forward pass in `torch.no_grad()` → calcolo loss e accuracy
3. **Logging** ogni 5 epoche e all'epoca 0
4. **Early stopping:** se `val_loss` non migliora per `patience` epoche consecutive, il training si ferma e viene ripristinato il best state dict

### Class weights

Per classi sbilanciate, i pesi vengono calcolati come:

$$w_c = \frac{1}{n_c} \cdot \frac{N}{\sum_c \frac{1}{n_c}}$$

dove $n_c$ è il numero di campioni della classe $c$ e $N$ è il numero totale di classi. Questo assicura che classi rare ricevano più attenzione durante il training.

### Struttura del risultato

```python
{
    "model": <ParticleMLP>,
    "model_name": "MLP (PyTorch)",
    "test_accuracy": 0.9634,
    "test_loss": 0.1021,
    "y_pred": np.ndarray,      # Predizioni (N_test,)
    "y_proba": np.ndarray,     # Probabilità softmax (N_test, 4)
    "train_time": 120.5,       # Secondi
    "history": {
        "train_loss": [0.45, 0.32, ...],  # Una voce per epoca
        "val_loss": [0.41, 0.30, ...],
        "val_acc": [0.88, 0.91, ...]
    },
    "device": torch.device    # "cuda" o "cpu"
}
```

### Output

- `outs/models/mlp_best.pt` — State dict PyTorch del miglior modello
- `outs/imgs/training/mlp_training_history.png` — Grafici loss e accuracy per epoca

---

## Fase 5 — Interpretabilità e Uncertainty Quantification

### Fase 5a — Analisi SHAP

**Moduli coinvolti:** `utils.interpretability`, `plot.visualization`

#### Operazioni

1. **Subsample** di `shap_samples` eventi dal test set
2. Per ogni modello ad albero (RF, XGBoost, DT): `shap.TreeExplainer`
3. Per la MLP: `shap.KernelExplainer` con background K-Means (50 cluster, 100 campioni max)

#### TreeExplainer vs KernelExplainer

| Aspetto | TreeExplainer | KernelExplainer |
|---|---|---|
| Modelli | RF, XGBoost, DT | Qualsiasi modello |
| Complessità | $O(N \cdot D \cdot T)$ — veloce | $O(N^2 \cdot M)$ — lento |
| Campioni | 1500 | 100 (max) |
| Accuratezza | Esatta | Approssimata |

#### Normalizzazione degli SHAP values

`shap.TreeExplainer` restituisce un array 3D `(n_samples, n_features, n_classes)`.
`shap.KernelExplainer` restituisce una lista di array `(n_samples, n_features)`.
La funzione `_to_list_format` normalizza entrambi al formato lista.

#### Output per ogni modello

- `outs/imgs/SHAP/SHAP_summary_{model}.png` — Beeswarm aggregato (tutte le classi)
- `outs/imgs/SHAP/SHAP_bar_{model}.png` — Barplot importanza media feature
- `outs/imgs/SHAP/SHAP_{model}_class_{particella}.png` — Beeswarm per singola classe

### Fase 5b — MC Dropout

**Moduli coinvolti:** `utils.uncertainty`, `plot.visualization`

#### Operazioni

1. **Abilitazione dropout in inferenza** (tutti i layer `Dropout` in modalità `train()`)
2. **N forward pass** sul test set completo (N = `mc_dropout_iterations`)
3. **Calcolo statistiche:**
   - `mean_proba`: media delle probabilità su N iterazioni
   - `std_proba`: deviazione standard delle probabilità
   - `predictions`: argmax di `mean_proba`
   - `entropy`: $H = -\sum_c \bar{p}_c \log(\bar{p}_c + \epsilon)$

#### Output

- `outs/imgs/uncertainty/uncertainty_entropy.png` — Distribuzione entropia (corretti vs errati)
- `outs/imgs/uncertainty/rejection_curve.png` — Accuracy vs soglia di entropia
- `outs/imgs/uncertainty/uncertainty_per_class.png` — Box plot entropia per classe
- `outs/imgs/uncertainty/uncertainty_scatter.png` — Scatter $p$ vs $E$ colorato per classe e per entropia

---

## Fase 6 — Valutazione finale e confronto

**Moduli coinvolti:** `utils.evaluation`, `plot.visualization`

### Operazioni

1. **Calcolo metriche** per ogni modello in `all_results`
2. **Costruzione tabella comparativa** ordinata per accuracy decrescente
3. **Salvataggio CSV** e **report testuale** di confronto
4. **Classification report** per ogni modello (precision, recall, F1 per classe)
5. **Matrici di confusione** per ogni modello
6. **Curve ROC** per modelli con probabilità
7. **Grafici di confronto metriche** individuali e raggruppati

### Metriche calcolate per ogni modello

| Metrica | Descrizione |
|---|---|
| `accuracy` | Frazione di predizioni corrette |
| `f1_macro` | F1-score mediato per classe (peso uguale a ogni classe) |
| `f1_weighted` | F1-score mediato pesato per supporto di classe |
| `precision_macro` | Precision macro |
| `precision_weighted` | Precision pesata |
| `recall_macro` | Recall macro |
| `recall_weighted` | Recall pesata |
| `auc_roc_macro` | Area Under ROC curve, media macro (solo se `y_proba` disponibile) |
| `auc_roc_weighted` | Area Under ROC curve, media pesata |
| `auc_class_0..3` | AUC-ROC per singola classe |
| `CV Accuracy` | Media accuracy cross-validation (solo modelli classici) |
| `Train Time (s)` | Tempo di training in secondi |

### Output

- `outs/results/model_comparison.csv` — Tabella comparativa CSV
- `outs/results/report_model_comparison.txt` — Tabella comparativa in formato testo
- `outs/results/report_{modello}.txt` — Classification report per ogni modello
- `outs/imgs/confusion_matrix/cm_{modello}.png` — Matrice di confusione per ogni modello
- `outs/imgs/roc_curves/roc_{modello}.png` — Curve ROC per ogni modello
- `outs/imgs/model_comparison/model_{metrica}_comparison.png` — Un grafico per ogni metrica
- `outs/imgs/model_comparison/model_comparison_groups.png` — Grafico metriche raggruppate
