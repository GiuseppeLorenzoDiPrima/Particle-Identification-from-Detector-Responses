# API — `models.classical_models`

**File sorgente:** [`models/classical_models.py`](../../models/classical_models.py)

Modulo per il training e la valutazione di modelli di Machine Learning classico: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest e XGBoost. Include cross-validation stratificata.

---

## Funzioni

### `_build_models`

```python
def _build_models(config: dict) -> dict
```

Funzione privata. Istanzia i modelli abilitati nel file di configurazione.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `config` | `dict` | Dizionario di configurazione. Legge la sezione `classical_models` e `dataset.random_state` |

**Restituisce:** `dict[str, estimator]` — Dizionario `{nome_modello: istanza_modello}` contenente solo i modelli con `enabled: true`.

**Modelli e iperparametri:**

#### `"Logistic Regression"` → `sklearn.linear_model.LogisticRegression`

| Parametro | Sorgente config | Descrizione |
|---|---|---|
| `max_iter` | `classical_models.logistic_regression.max_iter` | Iterazioni massime LBFGS |
| `solver` | `classical_models.logistic_regression.solver` | Algoritmo ottimizzazione |
| `class_weight` | `classical_models.logistic_regression.class_weight` | Gestione classi sbilanciate |
| `multi_class` | `classical_models.logistic_regression.multi_class` | Classificazione multi-classe |

#### `"K-NN"` → `sklearn.neighbors.KNeighborsClassifier`

| Parametro | Sorgente config | Descrizione |
|---|---|---|
| `n_neighbors` | `classical_models.knn.n_neighbors` | Numero di vicini (default: 7) |

#### `"Decision Tree"` → `sklearn.tree.DecisionTreeClassifier`

| Parametro | Sorgente config | Descrizione |
|---|---|---|
| `max_depth` | `classical_models.decision_tree.max_depth` | Profondità massima |
| `class_weight` | `classical_models.decision_tree.class_weight` | Gestione classi sbilanciate |
| `random_state` | `dataset.random_state` | Seed per riproducibilità |

#### `"Random Forest"` → `sklearn.ensemble.RandomForestClassifier`

| Parametro | Sorgente config | Descrizione |
|---|---|---|
| `n_estimators` | `classical_models.random_forest.n_estimators` | Numero di alberi |
| `max_depth` | `classical_models.random_forest.max_depth` | Profondità massima di ogni albero |
| `class_weight` | `classical_models.random_forest.class_weight` | Gestione classi sbilanciate |
| `n_jobs` | `classical_models.random_forest.n_jobs` | Parallelismo (-1 = tutti i core) |
| `random_state` | `dataset.random_state` | Seed per riproducibilità |

#### `"XGBoost"` → `xgboost.XGBClassifier`

| Parametro | Sorgente config | Descrizione |
|---|---|---|
| `n_estimators` | `classical_models.xgboost.n_estimators` | Round di boosting |
| `max_depth` | `classical_models.xgboost.max_depth` | Profondità massima alberi base |
| `learning_rate` | `classical_models.xgboost.learning_rate` | Learning rate (shrinkage) |
| `n_jobs` | `classical_models.xgboost.n_jobs` | Parallelismo |
| `eval_metric` | `classical_models.xgboost.eval_metric` | Metrica di valutazione interna |
| `random_state` | `dataset.random_state` | Seed per riproducibilità |

**Comportamento se nessun modello è abilitato:** Logga un warning e restituisce un dizionario vuoto.

---

### `run_cross_validation`

```python
def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: dict,
    config: dict
) -> dict
```

Esegue la cross-validation stratificata k-fold su tutti i modelli forniti. k viene impostato nel file di configurazione (default: 5).

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `X_train` | `np.ndarray` (N, 6) | Feature standardizzate del training set |
| `y_train` | `np.ndarray` (N,) | Label encoded del training set |
| `models` | `dict` | Dizionario `{nome: modello}` prodotto da `_build_models()` |
| `config` | `dict` | Dizionario di configurazione. Legge `cross_validation` e `dataset.random_state` |

**Restituisce:** `dict[str, dict]` con struttura:
```python
{
    "Logistic Regression": {"cv_mean": 0.88, "cv_std": 0.002, "cv_time": 45.3},
    "Random Forest":       {"cv_mean": 0.97, "cv_std": 0.001, "cv_time": 320.1},
    ...
}
```

| Chiave | Tipo | Descrizione |
|---|---|---|
| `cv_mean` | `float` | Media dell'accuracy sui k fold |
| `cv_std` | `float` | Deviazione standard dell'accuracy |
| `cv_time` | `float` | Tempo di esecuzione in secondi |

**Implementazione:**

Usa `sklearn.model_selection.StratifiedKFold` con:
- `n_splits` = `config["cross_validation"]["n_folds"]`
- `shuffle` = `config["cross_validation"]["shuffle"]`
- `random_state` = `config["dataset"]["random_state"]`

Per ogni modello: `cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")`

**Nota:** La cross-validation viene eseguita su istanze **non fittate** dei modelli. I modelli non vengono modificati da questa funzione.

---

### `train_and_evaluate`

```python
def train_and_evaluate(data: dict, config: dict) -> dict
```

Funzione principale del modulo. Esegue la pipeline completa: cross-validation (opzionale) → training sul training set completo → valutazione sul test set.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Dizionario dati da `load_and_preprocess()` |
| `config` | `dict` | Dizionario di configurazione |

**Restituisce:** `dict[str, dict]` — Un dizionario per ogni modello con struttura:

```python
{
    "XGBoost": {
        "model": <XGBClassifier fittato>,
        "model_name": "XGBoost",
        "cv_mean": 0.9708,       # None se cross_validation.enabled = false
        "cv_std": 0.0003,        # None se cross_validation.enabled = false
        "test_accuracy": 0.9710,
        "y_pred": np.ndarray,    # Shape: (N_test,)
        "y_proba": np.ndarray,   # Shape: (N_test, 4) — None se non disponibile
        "train_time": 45.2,      # Secondi
        "feature_importance": {  # None per KNN
            "p": 0.45,
            "theta": 0.12,
            "beta": 0.28,
            "nphe": 0.08,
            "ein": 0.04,
            "eout": 0.03
        }
    },
    "Random Forest": { ... },
    ...
}
```

**Pipeline interna:**

1. Chiama `_build_models(config)` per ottenere i modelli
2. Se `config["cross_validation"]["enabled"]`: chiama `run_cross_validation()`
3. Per ogni modello:
   - `model.fit(data["X_train"], data["y_train"])` — training sul set scalato completo
   - `model.predict(data["X_test"])` → `y_pred`
   - `accuracy_score(data["y_test"], y_pred)` → `test_accuracy`
   - Se il modello ha `predict_proba`: `model.predict_proba(data["X_test"])` → `y_proba`
   - Estrazione feature importance:
     - Se ha `feature_importances_` (DT, RF, XGBoost): usato direttamente
     - Se ha `coef_` (Logistic Regression): `np.abs(model.coef_).mean(axis=0)` (media dei valori assoluti su tutte le classi)

**Nota sulla feature importance di Logistic Regression:** I coefficienti `coef_` hanno shape `(n_classes, n_features)`. Facendo la media dei valori assoluti lungo l'asse delle classi si ottiene una misura aggregata dell'importanza di ogni feature nella separazione tra le classi.

---

### `plot_feature_importance`

```python
def plot_feature_importance(results: dict, feature_names: list, config: dict) -> None
```

Wrapper che delega la creazione del grafico a `plot.visualization.plot_feature_importance`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `results` | `dict` | Dizionario risultati da `train_and_evaluate()` |
| `feature_names` | `list[str]` | Lista dei nomi feature |
| `config` | `dict` | Dizionario di configurazione |

**Effetti:** Salva `outs/imgs/training/feature_importance.png`.

**Esempio completo:**

```python
from data_classes.data_loader import load_config, load_and_preprocess
from models.classical_models import train_and_evaluate, plot_feature_importance

config = load_config()
data = load_and_preprocess(config)

results = train_and_evaluate(data, config)

for name, res in results.items():
    print(f"{name}: test_acc={res['test_accuracy']:.4f}, "
          f"cv={res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

plot_feature_importance(results, data["feature_names"], config)
```
