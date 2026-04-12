# API — `utils.interpretability`

**File sorgente:** [`utils/interpretability.py`](../../utils/interpretability.py)

Modulo per l'analisi dell'interpretabilità dei modelli tramite SHAP (SHapley Additive exPlanations). Supporta sia modelli ad albero (tramite `TreeExplainer`) che la MLP (tramite `KernelExplainer`).

---

## Background teorico — SHAP

Gli **SHAP values** si basano sulla teoria dei giochi cooperativi (valori di Shapley). Per ogni evento e ogni feature, il valore SHAP quantifica il **contributo marginale** di quella feature alla predizione del modello rispetto al valore atteso (baseline).

Formalmente, per il modello $f$ e l'evento $\mathbf{x}$:
$$\phi_j(f, \mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[ f(\mathbf{x}_{S \cup \{j\}}) - f(\mathbf{x}_S) \right]$$

dove $F$ è l'insieme di tutte le feature e $S$ è un sottoinsieme.

**Proprietà:**
- **Efficienza:** $\sum_j \phi_j = f(\mathbf{x}) - E[f(\mathbf{x})]$
- **Simmetria:** feature con contributi uguali ricevono SHAP values uguali
- **Monotonia:** l'aggiunta di una feature non diminuisce mai il suo SHAP value se il suo contributo marginale è sempre positivo
- **Assenza:** feature non usate dal modello hanno SHAP value 0

---

## Funzioni

### `_to_list_format`

```python
def _to_list_format(shap_values, n_classes: int) -> list
```

Funzione privata. Normalizza il formato dell'output SHAP al formato lista standard.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `shap_values` | vario | Output grezzo di `explainer.shap_values()`. Può essere array 3D o lista |
| `n_classes` | `int` | Numero di classi (usato per lo slicing dell'array 3D) |

**Restituisce:** `list` di `n_classes` array, ognuno di forma `(n_samples, n_features)`. Ogni elemento della lista corrisponde agli SHAP values per una classe.

**Conversioni gestite:**

| Input | Conversione |
|---|---|
| `list` | Restituito invariato (formato KernelExplainer) |
| `np.ndarray` di shape `(n_samples, n_features, n_classes)` | Convertito: `[shap_values[:, :, c] for c in range(n_classes)]` (formato TreeExplainer per classificazione multiclasse) |
| Qualsiasi altro | Wrappato in lista singola `[shap_values]` |

**Motivazione:** `shap.TreeExplainer.shap_values()` e `shap.KernelExplainer.shap_values()` restituiscono formati diversi. Questa funzione unifica i due formati per semplificare il codice a valle.

---

### `run_shap_analysis`

```python
def run_shap_analysis(all_results: dict, data: dict, config: dict) -> None
```

Funzione principale del modulo. Esegue l'analisi SHAP su tutti i modelli supportati e genera le visualizzazioni.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `all_results` | `dict` | Dizionario aggregato dei risultati di tutti i modelli (da `main.py`) |
| `data` | `dict` | Dizionario dati da `load_and_preprocess()` |
| `config` | `dict` | Dizionario di configurazione. Legge `interpretability.enabled`, `interpretability.shap_samples`, `paths.figures_dir`, `visualization.dpi` |

**Restituisce:** `None`. Tutti i risultati vengono salvati come file immagine.

**Comportamento se `config["interpretability"]["enabled"]` è `False`:** logga un messaggio e ritorna immediatamente.

**Pipeline interna:**

#### 1. Setup

- Crea la directory `{figures_dir}/SHAP/`
- Determina `n_samples = config["interpretability"]["shap_samples"]`
- Chiama `setup_publication_style(config)` per lo stile matplotlib IEEE
- Estrae i nomi delle classi con `get_particle_labels(data["label_encoder"])`
- Override del figsize a `[10, 8]` (dimensioni più compatte per i plot SHAP)

#### 2. Subsample del test set

```python
idx = np.random.choice(len(data["X_test"]), min(n_samples, len(data["X_test"])), replace=False)
X_sample = data["X_test"][idx]
```

Il subsample è necessario perché il calcolo SHAP è costoso. Per `TreeExplainer` vengono usati 1500 campioni; per `KernelExplainer` al massimo 100.

#### 3. SHAP per modelli tree-based

Modelli considerati (in ordine): `"Random Forest"`, `"XGBoost"`, `"Decision Tree"`. Solo quelli presenti in `all_results` vengono analizzati.

```python
explainer = shap.TreeExplainer(model)
raw_sv = explainer.shap_values(X_sample)
sv_list = _to_list_format(raw_sv, n_classes)
plot_shap_results(sv_list, X_sample, feature_names, labels, name, shap_dir, dpi, figsize)
```

**`shap.TreeExplainer`:** Algoritmo esatto che sfrutta la struttura degli alberi decisionali. Complessità $O(T \cdot D \cdot 2^D)$ dove $T$ = alberi, $D$ = profondità. Molto più veloce del KernelExplainer.

In caso di eccezione per un singolo modello, logga un warning e passa al modello successivo (non interrompe l'analisi).

#### 4. SHAP per la MLP

Il modello MLP non supporta `TreeExplainer`. Si usa invece `KernelExplainer` con una funzione predict wrapper.

```python
def mlp_predict(X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        out = torch.softmax(model(X_t), dim=1)
        return out.cpu().numpy()

n_shap = min(100, len(X_sample))       # Max 100 campioni per KernelExplainer
X_shap = X_sample[:n_shap]
background = shap.kmeans(X_shap, 50)   # 50 cluster come background
explainer = shap.KernelExplainer(mlp_predict, background)
raw_sv = explainer.shap_values(X_shap)
```

**`shap.KernelExplainer`:** Metodo model-agnostico basato su perturbazioni. La funzione predict può essere qualsiasi callable. La **distribuzione di background** (qui K-Means con 50 cluster) approssima il valore atteso $E[f(\mathbf{x})]$.

**Limitazione:** Il KernelExplainer usa al massimo 100 campioni indipendentemente da `shap_samples`, per questioni di performance. Con 100 campioni e 50 cluster di background, ogni chiamata esegue $\sim 100 \times 2^6 = 6400$ valutazioni della rete.

In caso di eccezione, logga un warning e continua.

---

### `_safe` (funzione locale)

```python
def _safe(name: str) -> str
```

Funzione locale (definita alla fine del file). Non usata internamente dal modulo — vedere `_safe_name` in `utils.evaluation` per la versione usata nei nomi file.

| Input | Output |
|---|---|
| `"MLP (PyTorch)"` | `"mlp_pytorch"` |

---

## File generati

Per ogni modello analizzato (`random_forest`, `xgboost`, `decision_tree`, `mlp`):

| File | Descrizione |
|---|---|
| `outs/imgs/SHAP/SHAP_summary_{model}.png` | Beeswarm plot aggregato su tutte le classi |
| `outs/imgs/SHAP/SHAP_bar_{model}.png` | Barplot importanza media assoluta SHAP |
| `outs/imgs/SHAP/SHAP_{model}_class_elettrone.png` | Beeswarm per la classe elettrone |
| `outs/imgs/SHAP/SHAP_{model}_class_kaone.png` | Beeswarm per la classe kaone |
| `outs/imgs/SHAP/SHAP_{model}_class_pione.png` | Beeswarm per la classe pione |
| `outs/imgs/SHAP/SHAP_{model}_class_protone.png` | Beeswarm per la classe protone |

---

## Esempio

```python
from data_classes.data_loader import load_config, load_and_preprocess
from models.classical_models import train_and_evaluate
from utils.interpretability import run_shap_analysis

config = load_config()
data = load_and_preprocess(config)
all_results = train_and_evaluate(data, config)

# Esegui SHAP solo su Random Forest e XGBoost
run_shap_analysis(all_results, data, config)
# Salva figure in outs/imgs/SHAP/
```
