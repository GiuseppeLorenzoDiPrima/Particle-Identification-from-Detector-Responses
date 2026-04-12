# API — `models.baseline`

**File sorgente:** [`models/baseline.py`](../../models/baseline.py)

Implementazione del classificatore tradizionale a tagli percentili (cuts-based PID), che riproduce la metodologia classica usata in fisica sperimentale per la Particle IDentification.

---

## Costanti

### `MPL_FEATURE_LABELS`

```python
MPL_FEATURE_LABELS: dict[str, str] = {
    "p":     r"$p$",
    "theta": r"$\theta$",
    "beta":  r"$\beta$",
    "nphe":  r"$n_{phe}$",
    "ein":   r"$E_{in}$",
    "eout":  r"$E_{out}$",
}
```

Mappa i nomi delle feature ai loro simboli LaTeX per la visualizzazione con matplotlib. Usata internamente in `plot_feature_ranges()`.

---

## Classe `CutsBasedPID`

```python
class CutsBasedPID:
```

Classificatore a tagli sul piano multidimensionale delle feature. Per ogni classe, definisce un intervallo di accettazione per ogni feature calcolato tramite percentili del training set. La classificazione si basa sul conteggio di quante feature di un evento rientrano nell'intervallo di ogni classe.

### Attributi di istanza

| Attributo | Tipo | Descrizione |
|---|---|---|
| `ranges` | `dict[int, dict[int, tuple[float, float]]]` | Struttura `{class_id: {feat_idx: (low, high)}}`. Popolata da `fit()` |
| `centroids` | `dict[int, np.ndarray]` | Struttura `{class_id: centroid_vector}`. Centroidi nel spazio feature per tie-breaking. Popolata da `fit()` |
| `n_classes` | `int` | Numero di classi. Impostato da `fit()` |
| `low_percentile` | `int` | Percentile inferiore per il calcolo dei range |
| `high_percentile` | `int` | Percentile superiore per il calcolo dei range |
| `show_range` | `bool` | Se stampare i range a terminale |
| `feature_names` | `list[str]` | Nomi delle feature |
| `class_names` | `list[str]` | Nomi delle classi |
| `feature_for_print` | `list[str]` | Simboli Unicode per terminale |
| `graph` | `bool` | Se generare il grafico dei range |
| `config` | `dict` | Dizionario di configurazione |

---

### `__init__`

```python
def __init__(self, config: dict, data: dict)
```

Inizializza il classificatore con i parametri di configurazione e le informazioni sui dati.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `config` | `dict` | Dizionario di configurazione. Legge `baseline_cuts.low_percentile` (default 10), `baseline_cuts.high_percentile` (default 90), `baseline_cuts.show_range`, `visualization.graph` |
| `data` | `dict` | Dizionario dati prodotto da `load_and_preprocess()`. Legge `feature_names`, `class_names`, `feature_for_print` |

**Note:**
- `ranges` e `centroids` sono inizializzati come dizionari vuoti. Vengono popolati solo dopo la chiamata a `fit()`.
- I valori di default dei percentili sono 10 e 90, configurabili attraverso il file `config.yaml` (che di default usa 10 e 90).

---

### `fit`

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> CutsBasedPID
```

Calcola gli intervalli di accettazione (percentili) e i centroidi per ogni classe a partire dai dati di training.

**Parametri:**

| Parametro | Tipo | Shape | Descrizione |
|---|---|---|---|
| `X` | `np.ndarray` | (N, n_features) | Feature matrix. **Deve essere non standardizzata** (`X_train_raw`), perché i tagli operano sulle grandezze fisiche originali |
| `y` | `np.ndarray` | (N,) | Label encoded (interi 0–3) |

**Restituisce:** `self` — per consentire il chaining `model.fit(X, y).predict(X_test)`

**Algoritmo:**

Per ogni classe $c = 0, 1, 2, 3$:
1. Seleziona $X_c = X[y == c]$
2. Calcola il centroide: $\mu_c = \text{mean}(X_c, \text{axis}=0)$
3. Per ogni feature $j = 0, \ldots, n\_features-1$:
   - $\text{low}_j^{(c)} = \text{percentile}(X_c[:, j],\; \text{low\_percentile})$
   - $\text{high}_j^{(c)} = \text{percentile}(X_c[:, j],\; \text{high\_percentile})$
   - Memorizza: `self.ranges[c][j] = (low, high)`

**Funzioni abbinate:**
- Se `show_range=True`: chiama `_print_ranges()` per stampare la tabella a terminale
- Se `graph=True`: chiama `plot_feature_ranges()` per generare il grafico

---

### `_print_ranges`

```python
def _print_ranges(self) -> None
```

Metodo privato. Stampa a terminale una tabella formattata con i range calcolati per ogni classe e feature. Usa la libreria `tabulate` con stile `"grid"`.

**Formato output:**
```
Range feature per classe:
+----------+-------------------+-------------------+---
| Classe   | p                 | theta             | ...
+----------+-------------------+-------------------+---
| Positrone | 0.234 - 4.567     | 0.123 - 1.890     | ...
...
```

Non richiede parametri (usa gli attributi di istanza).

---

### `predict`

```python
def predict(self, X: np.ndarray) -> np.ndarray
```

Classifica nuovi eventi usando gli intervalli appresi.

**Parametri:**

| Parametro | Tipo | Shape | Descrizione |
|---|---|---|---|
| `X` | `np.ndarray` | (N, n_features) | Feature matrix **non standardizzata** |

**Restituisce:** `np.ndarray` di forma `(N,)` con le predizioni come interi 0–3 (label encoded).

**Algoritmo dettagliato:**

Per ogni evento $i$:
1. Per ogni classe $c$ e feature $j$: verifica se $X[i, j] \in [\text{low}_j^{(c)}, \text{high}_j^{(c)}]$
2. Calcola il punteggio $s_{i,c} = \sum_j \mathbf{1}[X[i,j] \in \text{range}_j^{(c)}]$
3. Trova le classi con punteggio massimo
4. Se una sola classe ha punteggio massimo → assegna quella classe
5. Se più classi hanno lo stesso punteggio massimo (**tie-breaking**):
   - Calcola la distanza euclidea $d_c = \|X[i] - \mu_c\|_2$ per ogni classe in parità
   - Assegna la classe con distanza minima

**Complessità:** $O(N \cdot C \cdot F)$ dove $N$ = campioni, $C$ = classi, $F$ = feature.

**Nota:** Il confronto `X[:, j] >= low` usa `>=` e `<=` (bordi inclusi).

---

### `evaluate`

```python
def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict
```

Valuta il classificatore su un dataset e restituisce le metriche.

**Parametri:**

| Parametro | Tipo | Shape | Descrizione |
|---|---|---|---|
| `X` | `np.ndarray` | (N, n_features) | Feature matrix **non standardizzata** |
| `y` | `np.ndarray` | (N,) | Label veri (label encoded) |

**Restituisce:** `dict` con le chiavi:

| Chiave | Tipo | Descrizione |
|---|---|---|
| `y_pred` | `np.ndarray` | Predizioni del modello |
| `accuracy` | `float` | Frazione di predizioni corrette |

Internamente chiama `sklearn.metrics.accuracy_score`.

---

### `plot_feature_ranges`

```python
def plot_feature_ranges(self, feature_names: list[str] | None, config: dict) -> None
```

Genera il grafico tabellare dei range feature delegando a `plot.visualization.plot_baseline_ranges`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `feature_names` | `list[str]` o `None` | Nomi delle feature. Se `None` o se `self.ranges` è vuoto, la funzione ritorna senza fare nulla |
| `config` | `dict` | Dizionario di configurazione (passato a `plot_baseline_ranges`) |

**Effetti:** Salva `outs/imgs/baseline/range_features.png`.

---

## Funzioni

### `run_baseline`

```python
def run_baseline(data: dict, config: dict) -> dict
```

Funzione di alto livello che esegue l'intera pipeline della baseline a tagli. Chiamata dalla Fase 2 in `main.py`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Dizionario dati prodotto da `load_and_preprocess()` |
| `config` | `dict` | Dizionario di configurazione |

**Restituisce:** `dict` (vuoto se `baseline_cuts.enabled` è `false`) con le chiavi:

| Chiave | Tipo | Descrizione |
|---|---|---|
| `y_pred` | `np.ndarray` | Predizioni sul test set |
| `accuracy` | `float` | Accuracy sul test set |
| `model` | `CutsBasedPID` | Istanza del modello fittata |
| `model_name` | `str` | Stringa `"Cuts-Based PID"` |

**Comportamento:**
1. Se `config["baseline_cuts"]["enabled"]` è `False`: logga un messaggio e restituisce `{}`
2. Altrimenti:
   - Istanzia `CutsBasedPID(config=config, data=data)`
   - Chiama `model.fit(data["X_train_raw"], data["y_train"])` — usa dati NON scalati
   - Chiama `model.evaluate(data["X_test_raw"], data["y_test"])` — usa dati NON scalati
   - Aggiunge `model` e `model_name` al risultato
   - Restituisce il dizionario risultante

**Nota sull'uso dei dati raw:** La baseline usa `X_train_raw` e `X_test_raw` (non scalati) perché i tagli calcolati tramite percentili devono operare sui valori fisici originali. Usare dati standardizzati sposterebbe le medie a zero e renderebbe i percentili privi di significato fisico.

**Esempio:**

```python
from data_classes.data_loader import load_config, load_and_preprocess
from models.baseline import run_baseline

config = load_config()
data = load_and_preprocess(config)
results = run_baseline(data, config)

print(f"Accuracy baseline: {results['accuracy']:.4f}")
print(f"Predizioni: {results['y_pred'][:5]}")
```
