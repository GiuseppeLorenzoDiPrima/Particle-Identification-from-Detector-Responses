# API — `models.deep_learning`

**File sorgente:** [`models/deep_learning.py`](../../models/deep_learning.py)

Modulo di Deep Learning per la Particle Identification. Implementa una Multi-Layer Perceptron (MLP) con PyTorch, con Batch Normalization, Dropout, early stopping, ReLU e supporto per MC Dropout e SHAP.

---

## Classe `ParticleMLP`

```python
class ParticleMLP(nn.Module):
```

Rete neurale feed-forward (MLP) per classificazione multiclasse. L'architettura è completamente configurabile tramite i parametri del costruttore.

### Architettura

Ogni layer nascosto è composto da 4 operazioni in sequenza:
```
Linear(prev_dim → h) → BatchNorm1d(h) → ReLU() → Dropout(p)
```

L'ultimo layer è solo un `Linear(h_last → n_classes)` senza attivazione (logits grezzi, la softmax viene applicata esternamente).

Con i parametri di default (`hidden_layers=[64, 128, 256, 128]`, `input_dim=6`, `n_classes=4`):

```
Sequential(
  Linear(6, 64)   → BatchNorm1d(64)  → ReLU → Dropout(0.3)
  Linear(64, 128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
  Linear(128,256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
  Linear(256,128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
  Linear(128, 4)
)
```

**Numero di parametri con config di default:**
- Layer 1: 6×64 + 64 = 448 (weights + biases)
- BN 1: 64×2 = 128
- Layer 2: 64×128 + 128 = 8.320
- BN 2: 128×2 = 256
- Layer 3: 128×256 + 256 = 33.024
- BN 3: 256×2 = 512
- Layer 4: 256×128 + 128 = 32.896
- BN 4: 128×2 = 256
- Output: 128×4 + 4 = 516
- **Totale:** ~76.356 parametri addestrabili

---

### `__init__`

```python
def __init__(
    self,
    input_dim: int,
    n_classes: int,
    hidden_layers: list[int],
    dropout: float = 0.3
)
```

Costruisce l'architettura MLP e la memorizza in `self.network` come `nn.Sequential`.

**Parametri:**

| Parametro | Tipo | Default | Descrizione |
|---|---|---|---|
| `input_dim` | `int` | — | Numero di feature in input (6 nel progetto) |
| `n_classes` | `int` | — | Numero di classi di output (4 nel progetto) |
| `hidden_layers` | `list[int]` | — | Lista delle dimensioni dei layer nascosti. Ogni elemento genera un blocco Linear → BN → ReLU → Dropout |
| `dropout` | `float` | `0.3` | Tasso di dropout (probabilità di azzerare un neurone). Usato anche per MC Dropout durante l'inferenza |

**Attributi:**
- `self.network`: `nn.Sequential` contenente tutti i layer

---

### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Esegue il forward pass.

**Parametri:**

| Parametro | Tipo | Shape | Descrizione |
|---|---|---|---|
| `x` | `torch.Tensor` | (batch_size, input_dim) | Batch di feature standardizzate come `FloatTensor` |

**Restituisce:** `torch.Tensor` di forma `(batch_size, n_classes)` — **logits grezzi** (non normalizzati). Per ottenere probabilità usare `torch.softmax(output, dim=1)`.

---

## Funzioni

### `_prepare_loaders`

```python
def _prepare_loaders(data: dict, config: dict) -> tuple[DataLoader, DataLoader, DataLoader]
```

Funzione privata. Converte i numpy array in `DataLoader` PyTorch.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Dizionario dati da `load_and_preprocess()`. Usa `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` |
| `config` | `dict` | Dizionario di configurazione. Legge `deep_learning.batch_size` |

**Restituisce:** Tupla `(train_loader, val_loader, test_loader)` dove:
- `train_loader`: shuffle=True (i dati vengono mescolati ogni epoca)
- `val_loader`: shuffle=False
- `test_loader`: shuffle=False

**Conversioni:**
- `X` → `torch.FloatTensor`
- `y` → `torch.LongTensor` (richiesto da `CrossEntropyLoss`)
- Wrappati in `TensorDataset` poi in `DataLoader`

---

### `train_mlp`

```python
def train_mlp(data: dict, config: dict) -> dict
```

Funzione principale del modulo. Addestra la MLP con early stopping e valuta il best model sul validation set.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `data` | `dict` | Dizionario dati da `load_and_preprocess()` |
| `config` | `dict` | Dizionario di configurazione. Legge l'intera sezione `deep_learning` e `paths.models_dir` |

**Restituisce:** `dict` con le chiavi:

| Chiave | Tipo | Descrizione |
|---|---|---|
| `model` | `ParticleMLP` | Istanza del modello con i pesi del best checkpoint caricati |
| `model_name` | `str` | Stringa `"MLP (PyTorch)"` |
| `test_accuracy` | `float` | Accuracy sul test set con il best model |
| `test_loss` | `float` | Loss media sul test set con il best model |
| `y_pred` | `np.ndarray` (N_test,) | Predizioni (argmax dei logits) |
| `y_proba` | `np.ndarray` (N_test, 4) | Probabilità softmax |
| `train_time` | `float` | Tempo totale di training in secondi |
| `history` | `dict` | Storia del training: `{"train_loss": [...], "val_loss": [...], "val_acc": [...]}` — una voce per epoca |
| `device` | `torch.device` | Device usato (`"cuda"` o `"cpu"`) |

**Pipeline interna dettagliata:**

#### 1. Selezione device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### 2. Costruzione modello

```python
model = ParticleMLP(
    input_dim=n_features,           # 6
    n_classes=n_classes,            # 4
    hidden_layers=cfg["hidden_layers"],
    dropout=cfg["dropout"],
).to(device)
```

#### 3. Class weights

Per gestire classi sbilanciate, i pesi vengono calcolati come:

$$w_c = \frac{1}{n_c} \cdot \frac{C}{\sum_{c'} \frac{1}{n_{c'}}}$$

dove $n_c$ = numero campioni della classe $c$ e $C$ = numero classi. Questo normalizza i pesi in modo che la loro somma sia uguale al numero di classi.

#### 4. Loss e optimizer

```python
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"],  # L2 regularization
)
```

#### 5. Training loop

Per ogni epoca:
1. **Training:** `model.train()` → forward → `criterion(outputs, y_batch)` → `backward()` → `optimizer.step()`
2. **Validation:** `model.eval()` con `torch.no_grad()` → forward → loss e accuracy
3. **Logging:** ogni 5 epoche e alla prima (epoch 0)
4. **Early stopping:**
   - Se `val_loss < best_val_loss`: aggiorna `best_val_loss`, salva `best_state` (copia CPU dei pesi)
   - Altrimenti: incrementa `patience_counter`
   - Se `patience_counter >= early_stopping_patience`: ferma il ciclo di addestramento

#### 6. Ripristino best model

```python
model.load_state_dict(best_state)
```

Il best model è quello con la **validation loss minima**, non necessariamente quello dell'ultima epoca.

#### 7. Valutazione su test set

Con `model.eval()` e `torch.no_grad()`:
- Concatena le predizioni di tutti i batch
- Calcola `torch.softmax(outputs, dim=1)` per le probabilità

#### 8. Salvataggio checkpoint

```python
torch.save(model.state_dict(), model_path)
# Salva in: outs/models/mlp_best.pt
```

**Output generati:** `outs/models/mlp_best.pt`

---

### `plot_training_history`

```python
def plot_training_history(history: dict, config: dict) -> None
```

Wrapper che delega la creazione del grafico a `plot.visualization.plot_training_history`.

**Parametri:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `history` | `dict` | Dizionario con chiavi `"train_loss"`, `"val_loss"`, `"val_acc"` (lista di float, una per epoca) |
| `config` | `dict` | Dizionario di configurazione |

**Effetti:** Salva `outs/imgs/training/mlp_training_history.png`.

---

## Esempi

### Training e inferenza

```python
from data_classes.data_loader import load_config, load_and_preprocess
from models.deep_learning import train_mlp, plot_training_history

config = load_config()
data = load_and_preprocess(config)

results = train_mlp(data, config)
print(f"Test accuracy: {results['test_accuracy']:.4f}")
print(f"Training time: {results['train_time']:.1f}s")
print(f"Device: {results['device']}")

plot_training_history(results["history"], config)
```

### Caricamento di un modello salvato

```python
import torch
from models.deep_learning import ParticleMLP

model = ParticleMLP(
    input_dim=6,
    n_classes=4,
    hidden_layers=[64, 128, 256, 128],
    dropout=0.3
)
model.load_state_dict(torch.load("outs/models/mlp_best.pt", map_location="cpu"))
model.eval()

# Inferenza su nuovi dati
import numpy as np
X_new = torch.FloatTensor(X_new_scaled)
with torch.no_grad():
    logits = model(X_new)
    proba = torch.softmax(logits, dim=1).numpy()
    preds = proba.argmax(axis=1)
```
