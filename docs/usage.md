# Utilizzo

## Esecuzione di base

```bash
python main.py
```

Esegue la **pipeline completa** nelle 6 fasi in sequenza. Richiede che il dataset Kaggle sia scaricabile (vedere [Installazione](installation.md)).\

---

## Argomenti da riga di comando

Il punto di ingresso `main.py` accetta i seguenti argomenti:

### `--config`

**Tipo:** `str`  
**Default:** `config/config.yaml`  
**Descrizione:** Percorso al file di configurazione YAML. Consente di usare configurazioni alternative senza modificare il file di default.

```bash
python main.py --config config/config_experiment.yaml
python main.py --config /percorso/assoluto/mia_config.yaml
```

### `--phase`

**Tipo:** `int` (1–6)  
**Default:** `None` (tutte le fasi)  
**Descrizione:** Esegue **una sola fase** della pipeline. Utile per rieseguire una fase specifica senza ripetere tutto dall'inizio.

> **Nota:** Anche quando si specifica una fase specifica, i dati vengono sempre caricati (Fase 1) perché tutte le fasi successive ne dipendono.

```bash
python main.py --phase 2    # Solo baseline a tagli
python main.py --phase 3    # Solo modelli classici
python main.py --phase 4    # Solo MLP
python main.py --phase 5    # Solo SHAP + MC Dropout
python main.py --phase 6    # Solo report finale
```

### `--phases`

**Tipo:** `int` (lista, nargs="+")  
**Default:** `None` (tutte le fasi)  
**Descrizione:** Esegue **un sottoinsieme di fasi** specificate come lista separata da spazi.

```bash
python main.py --phases 3 4        # Modelli classici + MLP
python main.py --phases 1 2 3      # Fasi 1, 2 e 3
python main.py --phases 4 5 6      # Solo DL, interpretabilità e report
```

> **Nota:** `--phase` e `--phases` sono mutualmente esclusivi. Se si specifica `--phase`, `--phases` viene ignorato.

### `--quick`

**Tipo:** flag (store_true)  
**Default:** `False`  
**Descrizione:** Attiva la modalità **run veloce**. Sovrascrive automaticamente i seguenti parametri in config:

| Parametro | Valore normale | Valore quick |
|---|---|---|
| `dataset.max_samples` | `null` (tutto) | `100_000` |
| `deep_learning.epochs` | `100` | `20` |
| `interpretability.shap_samples` | `1500` | `200` |
| `uncertainty.mc_dropout_iterations` | `100` | `20` |

Utile per verificare il funzionamento del codice o fare prove rapide.

```bash
python main.py --quick
python main.py --quick --phase 4
```

---

## Esempi di utilizzo

### Pipeline completa

```bash
python main.py
```

### Test rapido dell'intera pipeline

```bash
python main.py --quick
```

### Solo preprocessing e visualizzazioni esplorative

```bash
python main.py --phase 1
```

### Solo modelli classici (senza DL)

```bash
python main.py --phases 1 3 6
```

### Solo Deep Learning e report

```bash
python main.py --phases 4 6
```

### Configurazione personalizzata

```bash
python main.py --config config/my_config.yaml --quick
```

### Rieseguire solo il report finale

Utile se si sono già addestrati i modelli e si vuole rigenerare grafici o tabelle:

```bash
python main.py --phase 6
```

---

## Output del terminale

Durante l'esecuzione, il terminale mostra solo i messaggi del **progetto** (logger `main`, `data_classes.*`, `models.*`, `utils.*`, `plot.*`), con formato compatto senza timestamp:

```
============================================================
  PARTICLE IDENTIFICATION FROM DETECTOR RESPONSES
  Pipeline di Machine Learning per Fisica delle Particelle
============================================================

=======================================================
FASE 1: Caricamento dati e visualizzazione
=======================================================
Dataset caricato: 5000000 eventi, 7 colonne
...
```

I messaggi di librerie esterne (shap, matplotlib, xgboost) vengono silenziati sul terminale ma rimangono nel file di log completo (`outs/logs/run.log`).

---

## File di log

Il file `outs/logs/run.log` contiene tutti i messaggi di logging (inclusi quelli di librerie esterne) con timestamp nel formato:

```
HH:MM:SS [LEVEL] nome.modulo: messaggio
```

Esempio:
```
14:23:01 [INFO] data_classes.data_loader: Dataset caricato: 5000000 eventi
14:23:05 [INFO] models.classical_models: Training model: XGBoost...
14:23:42 [INFO] models.classical_models:     XGBoost: test_acc=0.9710
```

Il file di log viene sovrascritto ad ogni esecuzione (`mode="w"`).

---

## Modalità fasi e dipendenze

La pipeline è progettata per essere eseguita in sequenza. La seguente tabella mostra le dipendenze tra le fasi:

| Fase | Dipende da | Output prodotto |
|---|---|---|
| 1 — Preprocessing | — | `data` dict |
| 2 — Baseline | Fase 1 | `baseline_results` |
| 3 — ML Classici | Fase 1 | `classical_results` |
| 4 — MLP | Fase 1 | `mlp_results` |
| 5 — SHAP + Uncertainty | Fase 1 + Fasi 3/4 | Figure SHAP, uncertainty |
| 6 — Report | Fase 1 + Fasi 2/3/4 | CSV, TXT, grafici finali |

> Quando si usa `--phase N` con N > 1, i dati vengono caricati automaticamente anche se la Fase 1 non è esplicitamente selezionata. Tuttavia, i risultati dei modelli precedenti non sono disponibili (es. eseguire solo `--phase 5` senza aver eseguito la fase 3 e 4 non produrrà output SHAP per i modelli classici).

Per una pipeline parziale coerente, usare sempre `--phases` con tutte le fasi necessarie.
