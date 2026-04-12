# API — `utils.evaluation`

**File sorgente:** [`utils/evaluation.py`](../../utils/evaluation.py)

Modulo per la valutazione e il confronto dei modelli. Calcola metriche di classificazione, genera la tabella comparativa, i classification report, le matrici di confusione e le curve ROC.

---

## Funzioni

### `evaluate_model`

```python
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int = 4
) -> dict
```

Calcola un set completo di metriche di classificazione per un singolo modello.

**Parametri:**

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `y_true` | `np.ndarray` (N,) | — | Label veri (label encoded, interi) |
| `y_pred` | `np.ndarray` (N,) | — | Predizioni del modello (label encoded, interi) |
| `y_proba` | `np.ndarray` (N, C) o `None` | `None` | Probabilità di classe. Se `None`, le metriche AUC-ROC non vengono calcolate |
| `n_classes` | `int` | `4` | Numero di classi (usato per la binarizzazione one-vs-rest) |

**Restituisce:** `dict[str, float]` con le seguenti chiavi:

| Chiave | Tipo | Descrizione |
|---|---|---|
| `accuracy` | `float` | Frazione di predizioni corrette: $\frac{TP}{TP + FP + TN + FN}$ |
| `f1_macro` | `float` | F1-score macro: media non pesata dell'F1 per ogni classe |
| `precision_macro` | `float` | Precision macro: media non pesata della precision per ogni classe |
| `recall_macro` | `float` | Recall macro: media non pesata della recall per ogni classe |
| `f1_weighted` | `float` | F1-score pesato per il supporto (numero di campioni) di ogni classe |
| `precision_weighted` | `float` | Precision pesata per il supporto di ogni classe |
| `recall_weighted` | `float` | Recall pesata per il supporto di ogni classe |
| `auc_roc_macro` | `float` | AUC-ROC macro one-vs-rest (solo se `y_proba` è fornito) |
| `auc_roc_weighted` | `float` | AUC-ROC pesata per supporto (solo se `y_proba` è fornito) |
| `auc_class_0` | `float` | AUC-ROC per la classe 0 (solo se `y_proba` è fornito) |
| `auc_class_1` | `float` | AUC-ROC per la classe 1 (solo se `y_proba` è fornito) |
| `auc_class_2` | `float` | AUC-ROC per la classe 2 (solo se `y_proba` è fornito) |
| `auc_class_3` | `float` | AUC-ROC per la classe 3 (solo se `y_proba` è fornito) |

**Nota su macro vs weighted:**
- **Macro:** ogni classe ha uguale peso. Penalizza i modelli che funzionano bene solo sulla classe maggioritaria.
- **Weighted:** ogni classe ha peso proporzionale al numero di campioni. Riflette le performance "medie" ponderate per la distribuzione reale.

**Calcolo AUC-ROC:** Usa `sklearn.preprocessing.label_binarize` per convertire `y_true` in formato one-hot, poi `roc_auc_score` con `multi_class="ovr"`. Se il calcolo fallisce (ad esempio per classi mancanti nel campione), logga un warning e non aggiunge le chiavi AUC al dizionario.

**Esempio:**

```python
from utils.evaluation import evaluate_model
import numpy as np

metrics = evaluate_model(y_test, y_pred, y_proba, n_classes=4)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 macro: {metrics['f1_macro']:.4f}")
print(f"AUC-ROC macro: {metrics.get('auc_roc_macro', 'N/A')}")
```

---

### `build_comparison_table`

```python
def build_comparison_table(all_results: dict, data: dict) -> pd.DataFrame
```

Costruisce la tabella comparativa di tutti i modelli.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `all_results` | `dict` | Dizionario dei risultati: `{nome_modello: {"y_pred": ..., "y_proba": ..., ...}}` |
| `data` | `dict` | Dizionario dati da `load_and_preprocess()`. Usa `y_test` e `y_train` |

**Restituisce:** `pd.DataFrame` con le seguenti colonne (nell'ordine):

| Colonna | Tipo | Descrizione |
|---|---|---|
| `Modello` | `str` | Nome del modello |
| `accuracy` | `float` | Accuracy sul test set |
| `f1_macro` | `float` | F1-score macro |
| `precision_macro` | `float` | Precision macro |
| `recall_macro` | `float` | Recall macro |
| `f1_weighted` | `float` | F1-score weighted |
| `precision_weighted` | `float` | Precision weighted |
| `recall_weighted` | `float` | Recall weighted |
| `auc_roc_macro` | `float` | AUC-ROC macro (solo se disponibile) |
| `auc_roc_weighted` | `float` | AUC-ROC weighted (solo se disponibile) |
| `CV Accuracy` | `float` | Accuracy media cross-validation (solo se disponibile nel risultato) |
| `Train Time (s)` | `float` | Tempo di training arrotondato a 1 decimale (solo se disponibile) |

Il DataFrame è **ordinato per `accuracy` decrescente** con `reset_index(drop=True)`.

**Comportamento per colonne opzionali:** Le colonne `CV Accuracy` e `Train Time (s)` vengono incluse solo se le chiavi corrispondenti (`cv_mean`, `train_time`) sono presenti nel dizionario risultato. Se alcuni modelli le hanno e altri no, la colonna esiste nel DataFrame ma i valori mancanti sono `NaN`.

---

### `generate_full_report`

```python
def generate_full_report(all_results: dict, data: dict, config: dict) -> pd.DataFrame
```

Genera il report completo della valutazione finale: tabella CSV, report testuali, matrici di confusione, curve ROC e grafici di confronto. Chiamata dalla Fase 6 in `main.py`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `all_results` | `dict` | Dizionario aggregato di tutti i risultati |
| `data` | `dict` | Dizionario dati da `load_and_preprocess()` |
| `config` | `dict` | Dizionario di configurazione |

**Restituisce:** `pd.DataFrame` — La tabella comparativa (stesso output di `build_comparison_table()`).

**Operazioni eseguite:**

1. **Tabella di confronto CSV:**
   - `outs/results/model_comparison.csv`

2. **Report testuale di confronto:**
   - `outs/results/report_model_comparison.txt`

3. **Classification report per ogni modello:**
   - `outs/results/report_{safe_name}.txt`
   - Usa `sklearn.metrics.classification_report` con `digits=4`
   - Target names: nomi delle particelle capitalizzati (da `get_particle_labels()`)

4. **Matrici di confusione** (se `visualization.graph: true`):
   - `outs/imgs/confusion_matrix/cm_{safe_name}.png`

5. **Curve ROC** per modelli con `y_proba` non-None (se `visualization.graph: true`):
   - `outs/imgs/roc_curves/roc_{safe_name}.png`

6. **Grafici di confronto** (se `visualization.graph: true`):
   - `plot_metrics_comparison()` → un file per metrica
   - `plot_metric_groups_comparison()` → grafico raggruppato

---

### `_safe_name`

```python
def _safe_name(name: str) -> str
```

Funzione privata. Converte il nome di un modello in un nome file sicuro.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `name` | `str` | Nome del modello, es. `"MLP (PyTorch)"` |

**Restituisce:** `str` — Nome trasformato:
- Minuscolo
- Spazi → `_`
- Parentesi `(` e `)` rimosse

**Esempi:**

| Input | Output |
|---|---|
| `"Logistic Regression"` | `"logistic_regression"` |
| `"K-NN"` | `"k-nn"` |
| `"Decision Tree"` | `"decision_tree"` |
| `"Random Forest"` | `"random_forest"` |
| `"XGBoost"` | `"xgboost"` |
| `"MLP (PyTorch)"` | `"mlp_pytorch"` |
| `"Cuts-Based PID"` | `"cuts-based_pid"` |
