# Riferimento API

Documentazione completa di ogni funzione, classe, metodo e costante del progetto.

---

## Moduli

| Modulo | File sorgente | Descrizione |
|---|---|---|
| [data\_classes](data_classes.md) | `data_classes/data_loader.py` | Download Kaggle, preprocessing, split, standardizzazione |
| [models.baseline](models_baseline.md) | `models/baseline.py` | Classificatore a tagli percentili (cuts-based PID) |
| [models.classical\_models](models_classical.md) | `models/classical_models.py` | LR, KNN, DT, RF, XGBoost con cross-validation |
| [models.deep\_learning](models_deep_learning.md) | `models/deep_learning.py` | MLP PyTorch con early stopping e MC Dropout |
| [utils.evaluation](utils_evaluation.md) | `utils/evaluation.py` | Metriche, tabella comparativa, report finale |
| [utils.interpretability](utils_interpretability.md) | `utils/interpretability.py` | Analisi SHAP con TreeExplainer e KernelExplainer |
| [utils.uncertainty](utils_uncertainty.md) | `utils/uncertainty.py` | MC Dropout, entropia predittiva, rejection curve |
| [plot.visualization](plot_visualization.md) | `plot/visualization.py` | Tutte le funzioni di visualizzazione matplotlib |

---

## Riepilogo funzioni per modulo

### `data_classes.data_loader`

| Funzione/Costante | Tipo | Descrizione |
|---|---|---|
| `PARTICLE_NAMES` | `dict` | Mappa PDG ID → nome fisico |
| `FEATURE_NAMES` | `dict` | Mappa nome feature → simbolo Unicode |
| `load_config()` | funzione | Carica file YAML e restituisce dizionario |
| `download_dataset()` | funzione | Scarica dataset da Kaggle API |
| `load_and_preprocess()` | funzione | Pipeline completa di preprocessing |

### `models.baseline`

| Funzione/Classe | Tipo | Descrizione |
|---|---|---|
| `MPL_FEATURE_LABELS` | `dict` | Mappa feature → simboli LaTeX per matplotlib |
| `CutsBasedPID` | classe | Classificatore a tagli |
| `CutsBasedPID.__init__()` | metodo | Inizializzazione con config e data |
| `CutsBasedPID.fit()` | metodo | Calcola percentili e centroidi |
| `CutsBasedPID._print_ranges()` | metodo | Stampa tabella range (privato) |
| `CutsBasedPID.predict()` | metodo | Classifica nuovi eventi |
| `CutsBasedPID.evaluate()` | metodo | Calcola accuracy sul test set |
| `CutsBasedPID.plot_feature_ranges()` | metodo | Genera grafico tabella range |
| `run_baseline()` | funzione | Pipeline completa baseline |

### `models.classical_models`

| Funzione | Tipo | Descrizione |
|---|---|---|
| `_build_models()` | funzione (privata) | Istanzia modelli da config |
| `run_cross_validation()` | funzione | Cross-validation stratificata k-fold |
| `train_and_evaluate()` | funzione | Pipeline completa training + test |
| `plot_feature_importance()` | funzione | Wrapper per grafico feature importance |

### `models.deep_learning`

| Funzione/Classe | Tipo | Descrizione |
|---|---|---|
| `ParticleMLP` | classe (`nn.Module`) | MLP configurabile |
| `ParticleMLP.__init__()` | metodo | Costruisce architettura da parametri |
| `ParticleMLP.forward()` | metodo | Forward pass |
| `_prepare_loaders()` | funzione (privata) | Crea DataLoader da numpy array |
| `train_mlp()` | funzione | Training con early stopping |
| `plot_training_history()` | funzione | Wrapper grafico storia training |

### `utils.evaluation`

| Funzione | Tipo | Descrizione |
|---|---|---|
| `evaluate_model()` | funzione | Calcola metriche complete per un modello |
| `build_comparison_table()` | funzione | Costruisce DataFrame comparativo |
| `generate_full_report()` | funzione | Report completo: CSV, TXT, figure |
| `_safe_name()` | funzione (privata) | Conversione nome → nome file sicuro |

### `utils.interpretability`

| Funzione | Tipo | Descrizione |
|---|---|---|
| `_to_list_format()` | funzione (privata) | Normalizza output SHAP |
| `run_shap_analysis()` | funzione | Analisi SHAP su tutti i modelli |

### `utils.uncertainty`

| Funzione/Costante | Tipo | Descrizione |
|---|---|---|
| `FEATURE_NAMES` | `dict` | Mappa feature → simboli LaTeX |
| `enable_mc_dropout()` | funzione | Abilita dropout in inferenza |
| `mc_dropout_predict()` | funzione | N forward pass con MC Dropout |
| `run_uncertainty_analysis()` | funzione | Pipeline completa MC Dropout |

### `plot.visualization`

| Funzione/Costante | Tipo | Descrizione |
|---|---|---|
| `IEEE_PALETTE` | `list` | 6 colori colorblind-safe |
| `IEEE_LINESTYLES` | `list` | 6 stili di linea |
| `IEEE_MARKERS` | `list` | 6 marker |
| `FEATURE_NAMES` | `dict` | Mappa feature → simboli LaTeX |
| `setup_publication_style()` | funzione | Configura rcParams IEEE-ready |
| `get_particle_labels()` | funzione | Nomi particelle dal LabelEncoder |
| `plot_bethe_bloch()` | funzione | Scatter plot Bethe-Bloch |
| `plot_feature_distributions()` | funzione | Griglia istogrammi feature |
| `plot_class_distribution()` | funzione | Barre distribuzione classi |
| `plot_correlation_matrix()` | funzione | Heatmap correlazioni |
| `plot_confusion_matrix()` | funzione | Matrice di confusione |
| `plot_roc_curves()` | funzione | Curve ROC one-vs-rest |
| `plot_training_history()` | funzione | Loss e accuracy per epoca |
| `plot_feature_importance()` | funzione | Barplot feature importance |
| `plot_uncertainty_results()` | funzione | 4 grafici MC Dropout |
| `plot_shap_results()` | funzione | Summary, bar, per-classe SHAP |
| `plot_metrics_comparison()` | funzione | Barplot confronto metriche |
| `plot_metric_groups_comparison()` | funzione | Barre raggruppate metriche |
| `plot_baseline_ranges()` | funzione | Tabella range baseline |
| `_find_feature_index()` | funzione (privata) | Cerca indice feature per nome |
