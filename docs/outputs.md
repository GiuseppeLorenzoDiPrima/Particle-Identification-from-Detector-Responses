# Output

Tutti gli output vengono salvati nella directory `outs/` (configurabile tramite `paths.output_dir` in `config.yaml`). La directory viene creata automaticamente alla prima esecuzione.

---

## Struttura completa degli output

```
outs/
├── imgs/                              # Tutte le figure
│   ├── pre-processing/                # Visualizzazioni esplorative (Fase 1)
│   │   ├── bethe_bloch.png
│   │   ├── feature_distributions.png
│   │   ├── class_distribution_train.png
│   │   ├── class_distribution_val.png
│   │   ├── class_distribution_test.png
│   │   └── class_distribution_full.png
│   │   └── correlation_matrix.png
│   ├── baseline/                      # Grafici baseline (Fase 2)
│   │   └── range_features.png
│   ├── training/                      # Grafici training (Fasi 3–4)
│   │   ├── feature_importance.png
│   │   └── mlp_training_history.png
│   ├── confusion_matrix/              # Matrici di confusione (Fase 6)
│   │   ├── cm_cuts-based_pid.png
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_k-nn.png
│   │   ├── cm_decision_tree.png
│   │   ├── cm_random_forest.png
│   │   ├── cm_xgboost.png
│   │   └── cm_mlp_pytorch.png
│   ├── roc_curves/                    # Curve ROC (Fase 6)
│   │   ├── roc_logistic_regression.png
│   │   ├── roc_k-nn.png
│   │   ├── roc_decision_tree.png
│   │   ├── roc_random_forest.png
│   │   ├── roc_xgboost.png
│   │   └── roc_mlp_pytorch.png
│   ├── model_comparison/              # Confronto modelli (Fase 6)
│   │   ├── model_accuracy_comparison.png
│   │   ├── model_f1_macro_comparison.png
│   │   ├── model_precision_macro_comparison.png
│   │   ├── model_recall_macro_comparison.png
│   │   ├── model_precision_weighted_comparison.png
│   │   ├── model_recall_weighted_comparison.png
│   │   ├── model_f1_weighted_comparison.png
│   │   └── model_comparison_groups.png
│   ├── SHAP/                          # Analisi SHAP (Fase 5a)
│   │   ├── SHAP_summary_random_forest.png
│   │   ├── SHAP_bar_random_forest.png
│   │   ├── SHAP_random_forest_class_elettrone.png
│   │   ├── SHAP_random_forest_class_kaone.png
│   │   ├── SHAP_random_forest_class_pione.png
│   │   ├── SHAP_random_forest_class_protone.png
│   │   ├── SHAP_summary_xgboost.png
│   │   ├── SHAP_bar_xgboost.png
│   │   ├── ... (analoghe per decision_tree e MLP)
│   └── uncertainty/                   # MC Dropout (Fase 5b)
│       ├── uncertainty_entropy.png
│       ├── rejection_curve.png
│       ├── uncertainty_per_class.png
│       └── uncertainty_scatter.png
├── models/                            # Checkpoint modelli
│   └── mlp_best.pt                    # State dict PyTorch best model
├── results/                           # Report testuali e CSV
│   ├── model_comparison.csv
│   ├── report_model_comparison.txt
│   ├── report_cuts-based_pid.txt
│   ├── report_logistic_regression.txt
│   ├── report_k-nn.txt
│   ├── report_decision_tree.txt
│   ├── report_random_forest.txt
│   ├── report_xgboost.txt
│   └── report_mlp_pytorch.txt
└── logs/
    └── run.log                        # Log completo dell'esecuzione
```

---

## Descrizione dettagliata dei file

### Figure — `outs/imgs/`

#### `pre-processing/bethe_bloch.png`

Scatter plot 2D dell'energia depositata nel calorimetro interno ($E_{in}$) in funzione del momento $p$, colorato per specie di particella. Fino a 50.000 eventi (subsample casuale) per leggibilità. Permette di visualizzare le "bande" della formula di Bethe-Bloch che separano le specie.

**Risoluzione:** 600 DPI  
**Dimensioni:** 12×8 pollici  

#### `pre-processing/feature_distributions.png`

Griglia 3×2 di istogrammi. Ogni subplot mostra la distribuzione di una feature per ogni classe di particella. Gli istogrammi sono normalizzati (density=True) per confrontare forme invece di conteggi assoluti.

**Risoluzione:** 600 DPI  
**Dimensioni:** ~21×16 pollici (override interno)

#### `pre-processing/class_distribution_{split}.png`

Grafico a barre della distribuzione delle classi per ogni split (training, validation, test, completo). Ogni barra riporta il conteggio numerico sopra la barra stessa. Disponibile in 4 varianti: `train`, `val`, `test`, `full`.

#### `pre-processing/correlation_matrix.png`

Heatmap (seaborn) della matrice di correlazione di Pearson tra le 6 feature, calcolata sui dati di training non scalati. Colormap `coolwarm` centrata su 0 (bianco = nessuna correlazione).

#### `baseline/range_features.png`

Tabella matplotlib che mostra per ogni classe e ogni feature l'intervallo `[low, high]` calcolato tramite percentili. Intestazioni in blu IEEE con testo bianco grassetto.

#### `training/feature_importance.png`

Barplot orizzontali affiancati (uno per modello) che mostrano l'importanza relativa di ogni feature. I modelli senza feature importance (KNN) non compaiono. L'ordine è decrescente per importanza.

#### `training/mlp_training_history.png`

Due subplot affiancati: (1) loss di training e validation per epoca, (2) accuracy di validation per epoca. Permette di vedere la convergenza e il punto di early stopping.

#### `confusion_matrix/cm_{modello}.png`

Matrice di confusione con scala cromatica blues. Griglia allineata alle celle. Tutti e 4 i bordi esterni visibili. Valori numerici annotati nelle celle (conteggio assoluto).

Il nome del file usa la trasformazione `_safe_name()`:
- spazi → `_`
- parentesi rimosse
- tutto minuscolo

#### `roc_curves/roc_{modello}.png`

Curve ROC one-vs-rest per ogni classe. Ogni curva ha colore, stile di linea e marker distinti (IEEE-safe). L'area AUC è riportata nella legenda. La diagonale casuale è tracciata in grigio tratteggiato.

#### `model_comparison/model_{metrica}_comparison.png`

Un file per ogni metrica specificata in `visualization.comparison_metrics`. Barplot orizzontale con i modelli ordinati per accuracy decrescente. Il valore numerico è annotato a destra di ogni barra.

#### `model_comparison/model_comparison_groups.png`

Barplot raggruppate: modelli sull'asse X, una barra per ogni metrica in `visualization.comparison_group_metrics`. Permette di confrontare più metriche simultaneamente.

#### `SHAP/SHAP_summary_{modello}.png`

Beeswarm plot SHAP aggregato (tutte le classi). Ogni punto rappresenta un campione. L'asse X è il valore SHAP (contributo alla predizione), l'asse Y è la feature. Il colore indica il valore della feature.

#### `SHAP/SHAP_bar_{modello}.png`

Barplot orizzontale dell'importanza media assoluta SHAP per feature, mediata su tutte le classi.

#### `SHAP/SHAP_{modello}_class_{particella}.png`

Beeswarm SHAP per singola classe di particella. Un file per ogni classe.

#### `uncertainty/uncertainty_entropy.png`

Istogrammi sovrapposti dell'entropia predittiva per predizioni corrette (blu) e errate (rosso). Mostra che le predizioni errate tendono ad avere entropia più alta.

#### `uncertainty/rejection_curve.png`

Curva che mostra come varia l'accuracy al variare della soglia di entropia accettata. Più si abbassa la soglia (si rifiutano più eventi incerti), più l'accuracy aumenta. La linea tratteggiata mostra l'accuracy senza filtro.

#### `uncertainty/uncertainty_per_class.png`

Box plot dell'entropia predittiva per ogni classe di particella. Mostra quali classi il modello classifica con maggiore sicurezza.

#### `uncertainty/uncertainty_scatter.png`

Due scatter plot affiancati nel piano $p$ vs $E_{in}$: (1) colorato per classe predetta, (2) colorato per entropia (colormap `hot_r`). Mostra le regioni di alta incertezza nel piano delle feature.

---

### Checkpoint — `outs/models/`

#### `mlp_best.pt`

State dict PyTorch del modello MLP con la validation loss minima durante il training. Formato: dizionario Python con tensori dei pesi. Può essere caricato con:

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
```

---

### Report — `outs/results/`

#### `model_comparison.csv`

Tabella CSV con una riga per modello e colonne per ogni metrica calcolata. Ordinata per accuracy decrescente. Include anche CV Accuracy e Train Time dove disponibili.

Esempio di intestazione:
```
Modello,accuracy,f1_macro,precision_macro,recall_macro,f1_weighted,...,CV Accuracy,Train Time (s)
```

#### `report_model_comparison.txt`

Stessa tabella di confronto in formato testo, generata con `DataFrame.to_string()`.

#### `report_{modello}.txt`

Classification report sklearn per ogni modello. Include precision, recall, F1-score e supporto per ogni classe, più le medie macro e weighted. 4 cifre decimali.

Esempio:
```
=======================================================
Classification Report - XGBoost
=======================================================
              precision    recall  f1-score   support

    Elettrone     0.9723    0.9711    0.9717    112500
       Kaone      0.9580    0.9623    0.9601    112500
       Pione      0.9788    0.9741    0.9764    112500
      Protone     0.9850    0.9878    0.9864    112500

    accuracy                         0.9710    450000
   macro avg      0.9735    0.9738    0.9737    450000
weighted avg      0.9735    0.9738    0.9737    450000
```

---

### Log — `outs/logs/`

#### `run.log`

Log completo dell'esecuzione con formato `HH:MM:SS [LEVEL] modulo: messaggio`. Include tutti i messaggi di logging (anche da librerie esterne) a livello INFO e superiore. Viene **sovrascritto** ad ogni nuova esecuzione (`mode="w"`).

Utile per:
- Riprodurre i dettagli di un'esecuzione specifica
- Debuggare problemi con librerie come shap o xgboost
- Verificare i tempi di training esatti
