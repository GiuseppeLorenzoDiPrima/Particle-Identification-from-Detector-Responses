# API — `utils.uncertainty`

**File sorgente:** [`utils/uncertainty.py`](../../utils/uncertainty.py)

Modulo per la quantificazione dell'incertezza del modello MLP tramite Monte Carlo Dropout (MC Dropout). Stima l'incertezza epistemica del modello senza modificarne l'architettura.

---

## Background teorico — MC Dropout

Il **Monte Carlo Dropout** (Gal & Ghahramani, 2016) è un metodo per approssimare l'inferenza bayesiana nelle reti neurali. L'idea chiave è che il dropout — solitamente usato solo durante il training — se lasciato attivo durante l'inferenza trasforma la rete in un **ensemble implicito di reti** con pesi diversi.

Eseguendo $N$ forward pass indipendenti con dropout attivo su un singolo input $\mathbf{x}$:
$$\hat{p}_c^{(t)}(\mathbf{x}) = \text{Softmax}(f_{\hat{\theta}^{(t)}}(\mathbf{x})), \quad t = 1, \ldots, N$$

si ottiene la distribuzione predittiva:
$$p_c(\mathbf{x}) = \frac{1}{N} \sum_{t=1}^N \hat{p}_c^{(t)}(\mathbf{x})$$

**Entropia predittiva:** misura l'incertezza totale sulla predizione:
$$H[\mathbf{p}(\mathbf{x})] = -\sum_c p_c(\mathbf{x}) \log p_c(\mathbf{x})$$

- **Alta entropia:** il modello è incerto (evento in zona di sovrapposizione tra classi)
- **Bassa entropia:** il modello è sicuro (evento ben separato)

---

## Costanti

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

Mappa i nomi delle feature ai loro simboli LaTeX per la visualizzazione con matplotlib. Usata in `plot_uncertainty_results()` per le etichette degli assi.

---

## Funzioni

### `enable_mc_dropout`

```python
def enable_mc_dropout(model: nn.Module) -> None
```

Abilita il dropout durante l'inferenza (MC Dropout). Mette il modello in modalità `eval` (congela BatchNorm, disabilita gradient accumulation) ma mantiene i layer `Dropout` in modalità `train`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `model` | `nn.Module` | Modello PyTorch. Tipicamente un'istanza di `ParticleMLP` |

**Restituisce:** `None`. Modifica il modello **in-place**.

**Implementazione:**

```python
model.eval()                                  # eval mode per tutto (BN frozen, no grad)
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.train()                        # solo Dropout rimane in train mode
```

**Perché è necessario:** Di default, `model.eval()` disabilita il dropout oltre a congelare il BatchNorm. Questa funzione ri-abilita **solo** i layer Dropout, mantenendo il BatchNorm in modalità inferenza (media e varianza fisse).

---

### `mc_dropout_predict`

```python
def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray,
    n_iterations: int,
    device: torch.device = None
) -> dict
```

Esegue $N$ forward pass con dropout attivo e calcola le statistiche predittive.

**Parametri:**

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `model` | `nn.Module` | — | Modello PyTorch (tipicamente `ParticleMLP`) |
| `X` | `np.ndarray` (N, 6) | — | Feature standardizzate del test set |
| `n_iterations` | `int` | — | Numero di forward pass da eseguire (N). Valore tipico: 100 |
| `device` | `torch.device` o `None` | `None` | Device PyTorch. Se `None`, usa CPU |

**Restituisce:** `dict` con le seguenti chiavi:

| Chiave | Tipo | Shape | Descrizione |
|---|---|---|---|
| `mean_proba` | `np.ndarray` | (N_samples, 4) | Probabilità media su N iterazioni: $\bar{p} = \frac{1}{N}\sum_t \hat{p}^{(t)}$ |
| `std_proba` | `np.ndarray` | (N_samples, 4) | Deviazione standard su N iterazioni: misura la variabilità del dropout |
| `predictions` | `np.ndarray` | (N_samples,) | Classe predetta: $\arg\max_c \bar{p}_c$ |
| `entropy` | `np.ndarray` | (N_samples,) | Entropia predittiva: $H = -\sum_c \bar{p}_c \log(\bar{p}_c + \epsilon)$ |
| `all_probas` | `np.ndarray` | (N_iter, N_samples, 4) | Tutte le probabilità di ogni iterazione (utile per analisi avanzate) |

**Implementazione:**

```python
enable_mc_dropout(model)
X_tensor = torch.FloatTensor(X).to(device)

all_probas = []
with torch.no_grad():
    for i in range(n_iterations):
        outputs = model(X_tensor)
        probas = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probas.append(probas)

all_probas = np.array(all_probas)       # (n_iter, n_samples, n_classes)
mean_proba = all_probas.mean(axis=0)
std_proba  = all_probas.std(axis=0)
predictions = mean_proba.argmax(axis=1)

epsilon = 1e-10
entropy = -np.sum(mean_proba * np.log(mean_proba + epsilon), axis=1)
```

**Nota su `epsilon`:** Il termine `epsilon = 1e-10` evita `log(0)` quando una classe ha probabilità media esattamente zero. Questo garantisce stabilità numerica senza alterare significativamente l'entropia.

**Nota su `torch.no_grad()`:** Anche con dropout attivo, non è necessario calcolare i gradienti durante l'inferenza. `no_grad()` riduce significativamente il consumo di memoria.

---

### `run_uncertainty_analysis`

```python
def run_uncertainty_analysis(
    mlp_results: dict,
    data: dict,
    config: dict
) -> dict
```

Funzione di alto livello che esegue l'analisi di incertezza completa sul modello MLP. Chiamata dalla Fase 5b in `main.py`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `mlp_results` | `dict` | Risultati del training MLP da `train_mlp()`. Deve contenere le chiavi `"model"` e opzionalmente `"device"` |
| `data` | `dict` | Dizionario dati da `load_and_preprocess()`. Usa `X_test` e `y_test` |
| `config` | `dict` | Dizionario di configurazione. Legge `uncertainty.enabled`, `uncertainty.mc_dropout_iterations` |

**Restituisce:**
- `dict` — Output di `mc_dropout_predict()` (vedi sopra), oppure
- `{}` (dizionario vuoto) se `uncertainty.enabled` è `False` o il modello MLP non è disponibile

**Comportamento:**

1. Se `config["uncertainty"]["enabled"]` è `False`: logga e ritorna `{}`
2. Se non trova il modello MLP in `mlp_results`: logga un warning e ritorna `{}`
3. Estrae il modello:
   - Prima tenta `mlp_results["model"]` (quando riceve direttamente l'output di `train_mlp()`)
   - Poi tenta `mlp_results["MLP (PyTorch)"]["model"]` (quando riceve `all_results`)
4. Chiama `mc_dropout_predict(model, data["X_test"], n_iter, device)`
5. Chiama `plot_uncertainty_results(mc_results, data["y_test"], data, config)` per generare le 4 figure
6. Restituisce `mc_results`

**Output generati:**
- `outs/imgs/uncertainty/uncertainty_entropy.png`
- `outs/imgs/uncertainty/rejection_curve.png`
- `outs/imgs/uncertainty/uncertainty_per_class.png`
- `outs/imgs/uncertainty/uncertainty_scatter.png`

---

## Esempio

```python
from data_classes.data_loader import load_config, load_and_preprocess
from models.deep_learning import train_mlp
from utils.uncertainty import run_uncertainty_analysis, mc_dropout_predict, enable_mc_dropout

config = load_config()
data = load_and_preprocess(config)
mlp_results = train_mlp(data, config)

# Analisi completa con grafici
mc_results = run_uncertainty_analysis(mlp_results, data, config)

# Analisi manuale su un subset
import numpy as np
high_unc_mask = mc_results["entropy"] > 0.5
print(f"Eventi ad alta incertezza: {high_unc_mask.sum()} / {len(mc_results['entropy'])}")
print(f"Accuracy su eventi certi: {(mc_results['predictions'][~high_unc_mask] == data['y_test'][~high_unc_mask]).mean():.4f}")

# Uso diretto di mc_dropout_predict
import torch
model = mlp_results["model"]
device = mlp_results["device"]
mc = mc_dropout_predict(model, data["X_test"][:1000], n_iterations=50, device=device)
print(f"Entropia media: {mc['entropy'].mean():.4f}")
```
