# API — `data_classes`

**File sorgente:** [`data_classes/data_loader.py`](../../data_classes/data_loader.py)

Modulo per il download, caricamento e preprocessing del dataset "Particle Identification from Detector Responses" da [Kaggle](https://www.kaggle.com/database/naharrison/particle-identification-from-detector-responses).

---

## Costanti

### `PARTICLE_NAMES`

```python
PARTICLE_NAMES: dict[int, str] = {
    -11: "positrone",
    211: "pione",
    321: "kaone",
    2212: "protone",
}
```

Mappa dai codici PDG (Particle Data Group) numerici ai nomi fisici delle particelle. Usata per convertire la colonna `id` del CSV (che contiene interi) in stringhe leggibili prima del label encoding.

| PDG ID | Particella | Note |
|---|---|---|
| `-11` | positrone | Positrone (e⁺), antiparticella dell'elettrone |
| `211` | pione | Pione positivo (π⁺) |
| `321` | kaone | Kaone positivo (K⁺) |
| `2212` | protone | Protone (p) |

---

### `FEATURE_NAMES`

```python
FEATURE_NAMES: dict[str, str] = {
    "p":     "p",
    "theta": "θ",
    "beta":  "β",
    "nphe":  "nₚₕₑ",
    "ein":   "Eᵢₙ",
    "eout":  "Eₒᵤₜ",
}
```

Mappa i nomi delle colonne del CSV ai loro simboli Unicode per la visualizzazione nel terminale. Usata in `load_and_preprocess()` per popolare la chiave `feature_for_print` del dizionario `data`. Non usata per la visualizzazione matplotlib (che usa simboli LaTeX definiti in `plot/visualization.py`).

---

## Funzioni

### `load_config`

```python
def load_config(config_path: str = "config/config.yaml") -> dict
```

Carica un file di configurazione YAML e lo restituisce come dizionario Python.

**Parametri:**

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `config_path` | `str` | `"config/config.yaml"` | Percorso relativo o assoluto al file YAML da caricare |

**Restituisce:** `dict` — Il contenuto del file YAML come dizionario Python annidato.

**Eccezioni:**
- `FileNotFoundError` — Se il file non esiste al percorso specificato
- `yaml.YAMLError` — Se il file non è YAML valido

**Esempio:**

```python
from data_classes.data_loader import load_config

config = load_config("config/config.yaml")
print(config["dataset"]["max_samples"])  # None
```

---

### `download_dataset`

```python
def download_dataset(config: dict) -> str
```

Scarica il dataset da [Kaggle](https://www.kaggle.com/database/naharrison/particle-identification-from-detector-responses) usando l'API ufficiale (`kaggle` CLI). Se il file CSV è già presente nella directory dei dati, il download viene saltato.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `config` | `dict` | Dizionario di configurazione. Legge le chiavi `paths.data_dir`, `dataset.kaggle_slug`, `dataset.filename` |

**Restituisce:** `str` — Il percorso assoluto al file CSV scaricato.

**Comportamento dettagliato:**

1. Costruisce il percorso atteso: `{data_dir}/{filename}`
2. Se il file esiste già, logga un messaggio informativo e lo restituisce direttamente (skip download)
3. Crea `data_dir` se non esiste (`os.makedirs`)
4. Esegue il comando Kaggle via `subprocess.run`:
   ```
   kaggle datasets download -d {kaggle_slug} -p {data_dir}
   ```
   Con `check=True`, rilancia qualsiasi errore di processo come `CalledProcessError`
5. Cerca ed estrae tutti i file `.zip` trovati nella directory, poi li elimina
6. Verifica che il file CSV atteso esista dopo l'estrazione
7. Se il file atteso non esiste ma altri CSV sono presenti, li usa come fallback con un log di warning

**Eccezioni:**
- `subprocess.CalledProcessError` — Se il comando `kaggle` fallisce (API key mancante, rete, ecc.)
- `FileNotFoundError` — Se dopo download ed estrazione non viene trovato nessun CSV

**Prerequisiti:** Il file `~/.kaggle/kaggle.json` deve essere presente e valido. Vedere [Installazione](../installation.md).

**Esempio:**

```python
from data_classes.data_loader import load_config, download_dataset

config = load_config()
csv_path = download_dataset(config)
# Output: "data/pid-5M.csv" (oppure percorso relativo)
```

---

### `load_and_preprocess`

```python
def load_and_preprocess(config: dict) -> dict
```

Funzione principale del modulo. Esegue l'intera pipeline di preprocessing: download, caricamento, pulizia, encoding, split e standardizzazione.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `config` | `dict` | Dizionario di configurazione. Legge le sezioni `paths`, `dataset`, `features` |

**Restituisce:** `dict` con le seguenti chiavi:

| Chiave | Tipo | Shape | Descrizione |
|---|---|---|---|
| `X_train` | `np.ndarray` | (N_train, 6) | Feature standardizzate — training set |
| `X_val` | `np.ndarray` | (N_val, 6) | Feature standardizzate — validation set |
| `X_test` | `np.ndarray` | (N_test, 6) | Feature standardizzate — test set |
| `y_train` | `np.ndarray` | (N_train,) | Label encoded (int) — training set |
| `y_val` | `np.ndarray` | (N_val,) | Label encoded (int) — validation set |
| `y_test` | `np.ndarray` | (N_test,) | Label encoded (int) — test set |
| `X_train_raw` | `np.ndarray` | (N_train, 6) | Feature **non** standardizzate — training set |
| `X_val_raw` | `np.ndarray` | (N_val, 6) | Feature **non** standardizzate — validation set |
| `X_test_raw` | `np.ndarray` | (N_test, 6) | Feature **non** standardizzate — test set |
| `feature_names` | `list[str]` | (6,) | Nomi colonne CSV: `["p", "theta", "beta", "nphe", "ein", "eout"]` |
| `feature_for_print` | `list[str]` | (6,) | Simboli Unicode per terminale: `["p", "θ", "β", "nₚₕₑ", "Eᵢₙ", "Eₒᵤₜ"]` |
| `label_encoder` | `sklearn.LabelEncoder` | — | Encoder fittato sulle classi (usato per recuperare i nomi) |
| `scaler` | `sklearn.StandardScaler` | — | Scaler fittato **solo** sul training set |
| `df` | `pd.DataFrame` | (N, 7) | DataFrame originale con colonna `id` convertita in nomi fisici |
| `class_names` | `list[str]` | (4,) | Nomi classi capitalizzati nell'ordine del LabelEncoder |

**Pipeline interna dettagliata:**

1. **Download:** chiama `download_dataset(config)`
2. **Caricamento:** `pd.read_csv(csv_path)`
3. **Subsample:** se `config["dataset"]["max_samples"]` è non-null e inferiore alla lunghezza del DataFrame, estrae un campione casuale con `df.sample(n=max_samples, random_state=rs)`
4. **Rimozione NaN:** `df.dropna().reset_index(drop=True)`. Logga un warning con il numero di righe rimosse
5. **Mappatura PDG:** `df[target_col].map(PARTICLE_NAMES)`. Se esistono PDG ID non mappati (colonna ha NaN dopo la mappatura), solleva `ValueError`
6. **Split feature/target:** `X = df[feature_names].values`, `y_raw = df[target_col].values`
7. **Label encoding:** `le.fit_transform(y_raw)` — mappa i nomi stringa in interi 0–3
8. **Split train+val / test:** `train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)`
9. **Split train / val:** `train_test_split(X_trainval, y_trainval, test_size=val_size, random_state=rs, stratify=y_trainval)`
10. **Copia raw:** `X_train_raw = X_train.copy()` (prima della standardizzazione)
11. **Standardizzazione:** `scaler.fit_transform(X_train)`, poi `scaler.transform(X_val)` e `scaler.transform(X_test)`

**Eccezioni:**
- `ValueError` — Se esistono PDG ID nel CSV non presenti in `PARTICLE_NAMES`
- `FileNotFoundError` — Propagata da `download_dataset()` se il CSV non viene trovato

**Note importanti:**
- Lo `StandardScaler` viene fittato **esclusivamente** sul training set per prevenire il data leakage. I set di validation e test vengono solo trasformati.
- Le copie `X_*_raw` sono necessarie per la baseline cuts-based, che opera sulle grandezze fisiche originali.
- Il parametro `feature_names` è derivato dal CSV (tutte le colonne tranne `target_col`), non da `config["features"]["names"]`.

**Esempio:**

```python
from data_classes.data_loader import load_config, load_and_preprocess

config = load_config("config/config.yaml")
data = load_and_preprocess(config)

print(data["X_train"].shape)   # (3570000, 6)
print(data["class_names"])     # ["Positrone", "Kaone", "Pione", "Protone"]
print(data["feature_names"])   # ["p", "theta", "beta", "nphe", "ein", "eout"]

# Recuperare i nomi delle classi dall'encoder
import numpy as np
classes = data["label_encoder"].classes_
# array(["positrone", "kaone", "pione", "protone"])
```
