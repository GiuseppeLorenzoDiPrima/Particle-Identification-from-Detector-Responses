# API — `plot.visualization`

**File sorgente:** [`plot/visualization.py`](../../plot/visualization.py)

Modulo di visualizzazione per il progetto PID. Contiene tutte le funzioni matplotlib/seaborn per generare figure in stile IEEE-ready (publication quality, 600 DPI, colorblind-safe).

---

## Costanti

### `IEEE_PALETTE`

```python
IEEE_PALETTE: list[str] = [
    "#2166AC",  # blu acciaio
    "#B2182B",  # rosso scuro
    "#1B7837",  # verde foresta
    "#D6604D",  # arancione mattone
    "#762A83",  # viola scuro
    "#4D4D4D",  # grigio antracite
]
```

Palette di 6 colori progettata per essere:
- **Colorblind-safe:** distinguibile da persone con daltonismo
- **B&W-safe:** distinguibile anche in stampa in bianco e nero
- **IEEE-compatible:** adatta agli standard grafici delle pubblicazioni scientifiche IEEE

---

### `IEEE_LINESTYLES`

```python
IEEE_LINESTYLES: list = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),   # tratto-punto lungo
    (0, (5, 1)),          # trattini densi
]
```

6 stili di linea distinti per curve multi-classe (usati in `plot_roc_curves()`).

---

### `IEEE_MARKERS`

```python
IEEE_MARKERS: list[str] = ["o", "s", "D", "^", "v", "P"]
```

6 marker distinti: cerchio, quadrato, diamante, triangolo su, triangolo giù, plus riempito. Usati insieme a `IEEE_LINESTYLES` per massimizzare la distinguibilità.

---

### `FEATURE_NAMES`

```python
FEATURE_NAMES: dict[str, str] = {
    "p":     r"$p$",
    "theta": r"$\theta$",
    "beta":  r"$\beta$",
    "nphe":  r"$n_{phe}$",
    "ein":   r"$E_{in}$",
    "eout":  r"$E_{out}$",
}
```

Mappa i nomi delle colonne CSV ai simboli LaTeX per le etichette degli assi matplotlib.

---

## Funzioni di setup e utilità

### `setup_publication_style`

```python
def setup_publication_style(config: dict) -> None
```

Configura globalmente i parametri di stile matplotlib per grafici in stile IEEE.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `config` | `dict` | Dizionario di configurazione. Legge `visualization.dpi` |

**Parametri matplotlib impostati:**

| Parametro | Valore | Descrizione |
|---|---|---|
| `font.family` | `"serif"` | Font con grazie (Times-like) |
| `font.size` | `11` | Dimensione font base |
| `axes.titlesize` | `13` | Dimensione titolo assi |
| `axes.labelsize` | `11` | Dimensione etichette assi |
| `xtick.labelsize` | `10` | Dimensione tick etichette asse X |
| `ytick.labelsize` | `10` | Dimensione tick etichette asse Y |
| `legend.fontsize` | `10` | Dimensione testo legenda |
| `legend.frameon` | `True` | Box legenda visibile |
| `legend.edgecolor` | `"#d3d3d3"` | Bordo grigio chiaro per la legenda |
| `legend.fancybox` | `False` | Box legenda a angoli retti (non arrotondati) |
| `figure.dpi` | `config["visualization"]["dpi"]` | DPI per il rendering a schermo |
| `savefig.dpi` | `config["visualization"]["dpi"]` | DPI per il salvataggio su disco |
| `savefig.bbox` | `"tight"` | Ritaglia spazio vuoto al salvataggio |
| `axes.spines.top` | `False` | Rimuove il bordo superiore degli assi |
| `axes.spines.right` | `False` | Rimuove il bordo destro degli assi |
| `axes.linewidth` | `1.2` | Spessore bordi assi |
| `axes.edgecolor` | `"#333333"` | Colore bordi assi (grigio scuro) |
| `text.color` | `"#333333"` | Colore testo |
| `axes.labelcolor` | `"#333333"` | Colore etichette assi |

Usa anche `sns.set_palette(IEEE_PALETTE)` per impostare la palette seaborn.

---

### `get_particle_labels`

```python
def get_particle_labels(label_encoder) -> list[str]
```

Restituisce i nomi fisici delle particelle nell'ordine usato dal LabelEncoder.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `label_encoder` | `sklearn.LabelEncoder` | Encoder fittato sulle classi, da `data["label_encoder"]` |

**Restituisce:** `list[str]` — Nomi delle particelle in ordine label encoder. Usa `PARTICLE_NAMES` da `data_classes.data_loader` per recuperare il nome fisico da ogni classe.

**Esempio:**
```python
labels = get_particle_labels(data["label_encoder"])
# ["elettrone", "kaone", "pione", "protone"]  (ordine alfabetico LabelEncoder)
```

---

### `_find_feature_index`

```python
def _find_feature_index(feature_names: list[str], candidates: list[str]) -> int | None
```

Funzione privata. Cerca l'indice della prima feature il cui nome corrisponde a uno dei candidati (case-insensitive).

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `feature_names` | `list[str]` | Lista dei nomi feature |
| `candidates` | `list[str]` | Lista di nomi da cercare |

**Restituisce:** `int` (indice) oppure `None` se nessun candidato è trovato.

---

## Funzioni di visualizzazione dati (Fase 1)

### `plot_bethe_bloch`

```python
def plot_bethe_bloch(data: dict, config: dict) -> None
```

Genera lo scatter plot 2D dell'energia depositata vs quantità di moto — il **diagramma di Bethe-Bloch** fondamentale per la PID.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Dizionario dati. Usa `X_train_raw`, `y_train`, `feature_names`, `label_encoder` |
| `config` | `dict` | Dizionario di configurazione. Legge `paths.figures_dir`, `visualization.figsize`, `visualization.dpi` |

**Comportamento:**
1. Identifica l'indice di $p$ cercando `["p", "momentum"]` in `feature_names`
2. Identifica l'indice dell'energia cercando `["ein", "eout", "edep", "dedx", "de_dx", "energy"]`
3. Se non trovate, usa le prime due feature (fallback con warning)
4. Subsample di max 50.000 eventi dal training set per leggibilità
5. Scatter plot con scatter size `s=2`, `alpha=0.3` per densità dei punti
6. Una serie per classe con colori `IEEE_PALETTE`

**Output:** `outs/imgs/pre-processing/bethe_bloch.png`

---

### `plot_feature_distributions`

```python
def plot_feature_distributions(data: dict, config: dict) -> None
```

Griglia di istogrammi (3×2) che mostra la distribuzione di ogni feature per classe.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Usa `X_train_raw`, `y_train`, `feature_names`, `label_encoder` |
| `config` | `dict` | Legge `paths.figures_dir`, `visualization.dpi` |

**Dettagli:**
- 60 bin per istogramma
- `density=True` — distribuzione normalizzata (confronto forme, non conteggi)
- `alpha=0.55` — semi-trasparenza per sovrapposizione
- Colori `IEEE_PALETTE` per classe
- `edgecolor="white"` con `linewidth=0.4` per separazione visiva delle barre
- Override del figsize a `[14, 8]` (più ampio del default)
- Assi vuoti (se il numero di feature < numero subplot) vengono nascosti

**Output:** `outs/imgs/pre-processing/feature_distributions.png`

---

### `plot_class_distribution`

```python
def plot_class_distribution(data: dict, config: dict) -> None
```

Grafici a barre della distribuzione delle classi per ogni split del dataset.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Usa `y_train`, `y_val`, `y_test`, `label_encoder` |
| `config` | `dict` | Legge `paths.figures_dir`, `visualization.figsize`, `visualization.dpi` |

**Genera 4 file** (uno per split: training, validation, test, completo):
- Ricostruisce `y_full = np.concatenate([y_train, y_val, y_test])`
- Conteggio numerico annotato sopra ogni barra (con `ax.text()`)
- `edgecolor="#333333"` per bordi barre
- Griglia orizzontale (`axis="y"`)

**Output:**
- `outs/imgs/pre-processing/class_distribution_train.png`
- `outs/imgs/pre-processing/class_distribution_val.png`
- `outs/imgs/pre-processing/class_distribution_test.png`
- `outs/imgs/pre-processing/class_distribution_full.png`

---

### `plot_correlation_matrix`

```python
def plot_correlation_matrix(data: dict, config: dict) -> None
```

Heatmap seaborn della matrice di correlazione di Pearson tra le feature.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Usa `X_train_raw`, `feature_names` |
| `config` | `dict` | Legge `paths.figures_dir`, `visualization.figsize`, `visualization.dpi` |

**Dettagli:**
- `annot=True`, `fmt=".2f"` — annotazioni numeriche con 2 decimali
- Colormap `"coolwarm"` centrata su 0 (bianco = nessuna correlazione)
- Calcolata sui dati di training non scalati

**Output:** `outs/imgs/pre-processing/correlation_matrix.png`

---

## Funzioni metriche (Fase 6)

### `plot_confusion_matrix`

```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    config: dict,
    filename: str,
    subdir: str = "confusion_matrix"
) -> None
```

Genera e salva la matrice di confusione.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `y_true` | `np.ndarray` | Label veri (label encoded) |
| `y_pred` | `np.ndarray` | Predizioni (label encoded) |
| `labels` | `list[str]` | Nomi delle classi da mostrare sugli assi |
| `title` | `str` | Titolo del grafico |
| `config` | `dict` | Dizionario di configurazione |
| `filename` | `str` | Nome del file da salvare (es. `"cm_xgboost.png"`) |
| `subdir` | `str` | Sottocartella dentro `figures_dir`. Default: `"confusion_matrix"` |

**Implementazione:**
- `sklearn.metrics.ConfusionMatrixDisplay` con colormap `"Blues"`
- Griglia allineata ai **bordi delle celle** (`minor=True` ticks a -0.5, 0.5, ...) — non la griglia di default che cade a metà cella
- Tutti e 4 i bordi esterni visibili con `linewidth=1.5`
- Valori numerici interi nelle celle

**Output:** `outs/imgs/confusion_matrix/{filename}`

---

### `plot_roc_curves`

```python
def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    title: str,
    config: dict,
    filename: str,
    subdir: str = "roc_curves"
) -> None
```

Genera le curve ROC one-vs-rest per classificazione multiclasse.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `y_true` | `np.ndarray` (N,) | Label veri (label encoded) |
| `y_score` | `np.ndarray` (N, C) | Probabilità di classe (output di `predict_proba` o softmax) |
| `labels` | `list[str]` | Nomi delle classi |
| `title` | `str` | Titolo del grafico |
| `config` | `dict` | Dizionario di configurazione |
| `filename` | `str` | Nome del file da salvare |
| `subdir` | `str` | Sottocartella. Default: `"roc_curves"` |

**Implementazione:**
- Binarizzazione con `label_binarize(y_true, classes=range(n_classes))`
- Per ogni classe: `sklearn.metrics.roc_curve` e `sklearn.metrics.auc`
- Curva con colore `IEEE_PALETTE[i]`, linestyle `IEEE_LINESTYLES[i]`, marker `IEEE_MARKERS[i]`
- Marker ogni `max(1, len(fpr)//10)` punti per non saturare il grafico
- AUC annotato nella legenda: `"Elettrone (AUC = 0.999)"`
- Diagonale casuale in grigio `#999999` tratteggiato
- Override figsize a `[10, 8]`

**Output:** `outs/imgs/roc_curves/{filename}`

---

## Funzioni training (Fasi 3–4)

### `plot_training_history`

```python
def plot_training_history(history: dict, config: dict) -> None
```

Grafico della storia del training MLP: loss per epoca e accuracy di validation.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `history` | `dict` | Con chiavi `"train_loss"`, `"val_loss"`, `"val_acc"` (liste di float) |
| `config` | `dict` | Dizionario di configurazione |

**Genera 2 subplot affiancati:**
1. **Loss:** train loss (blu, linea continua) e val loss (rosso, tratteggiato)
2. **Accuracy:** val accuracy (verde)

Override figsize a `[14, 8]`.

**Output:** `outs/imgs/training/mlp_training_history.png`

---

### `plot_feature_importance`

```python
def plot_feature_importance(results: dict, feature_names: list, config: dict) -> None
```

Barplot orizzontali della feature importance per tutti i modelli che la supportano.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `results` | `dict` | Dizionario risultati modelli classici da `train_and_evaluate()` |
| `feature_names` | `list[str]` | Nomi delle feature |
| `config` | `dict` | Dizionario di configurazione |

**Comportamento:**
- Filtra i modelli con `feature_importance is not None`
- Se nessun modello ha feature importance: ritorna senza fare nulla
- Crea subplots affiancati (uno per modello)
- Ordine decrescente per importanza (`sorted(fi.items(), key=lambda x: x[1], reverse=True)`)
- Usa simboli LaTeX tramite `FEATURE_NAMES.get(x[0], x[0])`
- Override figsize a `[18, 8]`

**Output:** `outs/imgs/training/feature_importance.png`

---

## Funzioni uncertainty (Fase 5b)

### `plot_uncertainty_results`

```python
def plot_uncertainty_results(
    mc_results: dict,
    y_test: np.ndarray,
    data: dict,
    config: dict
) -> None
```

Genera 4 figure separate per l'analisi di incertezza MC Dropout.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `mc_results` | `dict` | Output di `mc_dropout_predict()`: contiene `entropy`, `predictions`, ecc. |
| `y_test` | `np.ndarray` (N,) | Label veri del test set |
| `data` | `dict` | Dizionario dati. Usa `label_encoder`, `feature_names`, `X_test_raw` |
| `config` | `dict` | Dizionario di configurazione |

**Grafici generati:**

#### 1. `uncertainty_entropy.png` — Distribuzione entropia

Istogrammi sovrapposti (`alpha=0.6`, 50 bin, `density=True`):
- Blu: entropia delle predizioni **corrette** (`y_pred == y_test`)
- Rosso: entropia delle predizioni **errate**

#### 2. `rejection_curve.png` — Rejection curve

100 soglie di entropia da 0 a `max(entropy)`. Per ogni soglia:
- `mask = entropy <= threshold` — eventi accettati
- Calcola accuracy sugli eventi accettati e la loro frazione

Linea tratteggiata = accuracy senza filtro (baseline).

#### 3. `uncertainty_per_class.png` — Box plot per classe

Box plot dell'entropia per ogni classe vera (`y_test == c`). Mostra la distribuzione di incertezza classe per classe.

#### 4. `uncertainty_scatter.png` — Scatter nel piano fisico

Due subplot affiancati (override figsize `[14, 8]`):
- **Sinistra:** scatter $p$ vs $E_{in}$ colorato per classe vera
- **Destra:** scatter $p$ vs $E_{in}$ colorato per entropia (colormap `"hot_r"`)

Identifica automaticamente gli indici di $p$ ed $E_{in}$ dalle `feature_names`.

---

## Funzioni SHAP (Fase 5a)

### `plot_shap_results`

```python
def plot_shap_results(
    sv_list: list,
    X_sample: np.ndarray,
    feature_names: list[str],
    labels: list[str],
    model_name: str,
    fig_dir: str,
    dpi: int,
    figsize: tuple
) -> None
```

Genera 3 tipi di figure SHAP per un singolo modello.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `sv_list` | `list` | Lista di n_classes array (N, F) di SHAP values, prodotta da `_to_list_format()` |
| `X_sample` | `np.ndarray` (N, F) | Campione del test set usato per il calcolo SHAP |
| `feature_names` | `list[str]` | Nomi delle feature (chiavi per `FEATURE_NAMES`) |
| `labels` | `list[str]` | Nomi delle classi |
| `model_name` | `str` | Nome del modello (usato in titoli e nomi file) |
| `fig_dir` | `str` | Percorso directory dove salvare le figure |
| `dpi` | `int` | Risoluzione figure |
| `figsize` | `tuple` | Dimensioni figure (larghezza, altezza) in pollici |

**Figure generate:**

#### 1. Summary aggregato — `SHAP_summary_{model}.png`

```python
shap.summary_plot(sv_list, X_sample, feature_names=feat_labels,
                  class_names=capitalized_labels, show=False)
```

Beeswarm plot SHAP aggregato su tutte le classi. Ogni punto = un campione. Asse X = SHAP value (contributo), asse Y = feature, colore = valore della feature.
Figsize override: `(6, 4)` per plot più compatti.

#### 2. Barplot importanza media — `SHAP_bar_{model}.png`

```python
mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in sv_list], axis=0)
```

Importanza media assoluta SHAP, mediata su tutte le classi. Ordinata per importanza crescente (barra più lunga in alto). Colori `IEEE_PALETTE`.

#### 3. Summary per classe — `SHAP_{model}_class_{particella}.png`

Uno per ogni classe (4 file):
```python
shap.summary_plot(sv_list[class_idx], X_sample, feature_names=feat_labels, show=False)
```

La mappatura `class_idx` → nome particella usa:
```python
CLASS_NAMES = {1: "elettrone", 2: "kaone", 3: "pione", 4: "protone"}
```

> **Nota:** `CLASS_NAMES` usa indici 1-based (il loop `for class_idx, label in enumerate(labels)` produce 0-based, quindi si accede come `CLASS_NAMES.get(class_idx + 1)`).

---

## Funzioni confronto modelli (Fase 6)

### `plot_metrics_comparison`

```python
def plot_metrics_comparison(comparison: pd.DataFrame, config: dict) -> None
```

Genera un barplot orizzontale per ogni metrica specificata in `visualization.comparison_metrics`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `comparison` | `pd.DataFrame` | Tabella comparativa da `build_comparison_table()` |
| `config` | `dict` | Dizionario di configurazione |

**Comportamento:**
- Una figura per metrica (più file in `model_comparison/`)
- Modelli ordinati per accuracy decrescente (l'ordine del DataFrame)
- Valore numerico annotato a destra di ogni barra con 4 decimali
- Asse X limitato a [0, 1.08] per lasciare spazio alle annotazioni
- Override figsize a `[10, 8]`

**Output:** `outs/imgs/model_comparison/model_{metrica}_comparison.png` per ogni metrica

---

### `plot_metric_groups_comparison`

```python
def plot_metric_groups_comparison(comparison: pd.DataFrame, config: dict) -> None
```

Genera un unico grafico a barre raggruppate: modelli sull'asse X, una barra per ogni metrica in `visualization.comparison_group_metrics`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `comparison` | `pd.DataFrame` | Tabella comparativa |
| `config` | `dict` | Dizionario di configurazione |

**Dettagli:**
- Larghezza totale barre per modello: `0.8`
- Larghezza singola barra: `0.8 / n_metrics`
- Etichette modello ruotate di 45°
- Valore numerico annotato sopra ogni barra (3 decimali)
- Asse Y limitato a [0, 1.05]
- Override figsize a `[14, 8]`

**Output:** `outs/imgs/model_comparison/model_comparison_groups.png`

---

## Funzioni baseline (Fase 2)

### `plot_baseline_ranges`

```python
def plot_baseline_ranges(
    class_names: list[str],
    ranges: dict,
    feature_names: list[str],
    config: dict
) -> None
```

Genera una tabella matplotlib con i range calcolati dalla baseline a tagli.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `class_names` | `list[str]` | Nomi delle classi (capitalizzati) |
| `ranges` | `dict` | `{class_id: {feat_idx: (low, high)}}` — attributo `ranges` di `CutsBasedPID` |
| `feature_names` | `list[str]` | Nomi delle feature (chiavi per `FEATURE_NAMES`) |
| `config` | `dict` | Dizionario di configurazione |

**Dettagli implementativi:**
- Figura dimensionata dinamicamente: `max(10, n_features * 1.8)` × `max(2.5, n_classes * 0.9)`
- `ax.axis("off")` — nessun asse, solo la tabella
- Intestazione in `IEEE_PALETTE[0]` (blu) con testo bianco grassetto
- Celle dati alternate bianche/grigie chiare (`#f7f7f7`)
- Font size 11, scaling `1.8` per altezza righe
- Valori formattati come `"{low:.3f} – {high:.3f}"`
- `bbox_inches="tight"` e `facecolor=fig.get_facecolor()` per il salvataggio

**Output:** `outs/imgs/baseline/range_features.png`
