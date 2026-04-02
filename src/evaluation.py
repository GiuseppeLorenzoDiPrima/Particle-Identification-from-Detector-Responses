"""
Modulo di valutazione e confronto modelli.

Produce:
- Tabella riepilogativa con accuracy, F1 macro, AUC-ROC per classe
- Confronto visivo tra tutti i modelli
- Report testuale dei risultati
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    get_particle_labels,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=4) -> dict:
    """
    Calcola metriche complete per un modello.

    Returns:
        Dict con accuracy, f1_macro, precision_macro, recall_macro,
        e opzionalmente auc_roc_macro e auc_per_class.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            metrics["auc_roc_macro"] = roc_auc_score(
                y_bin, y_proba, multi_class="ovr", average="macro"
            )
            # AUC per singola classe
            for i in range(n_classes):
                metrics[f"auc_class_{i}"] = roc_auc_score(y_bin[:, i], y_proba[:, i])
        except ValueError:
            logger.warning("Impossibile calcolare AUC-ROC (classi mancanti?).")

    return metrics


def build_comparison_table(all_results: dict, data: dict) -> pd.DataFrame:
    """
    Costruisce la tabella di confronto tra tutti i modelli.

    Args:
        all_results: {nome_modello: {y_pred, y_proba, ...}}
        data: dizionario dati con y_test

    Returns:
        DataFrame con una riga per modello e colonne per ciascuna metrica.
    """
    y_true = data["y_test"]
    n_classes = len(np.unique(data["y_train"]))
    rows = []

    for name, res in all_results.items():
        y_pred = res["y_pred"]
        y_proba = res.get("y_proba")
        metrics = evaluate_model(y_true, y_pred, y_proba, n_classes)
        metrics["Modello"] = name

        # Aggiungi tempo di training se disponibile
        if "train_time" in res:
            metrics["Train Time (s)"] = round(res["train_time"], 1)

        # Aggiungi CV score se disponibile
        if "cv_mean" in res:
            metrics["CV Accuracy"] = res["cv_mean"]

        rows.append(metrics)

    df = pd.DataFrame(rows)

    # Riordina colonne
    col_order = ["Modello", "accuracy", "f1_macro", "precision_macro",
                 "recall_macro", "auc_roc_macro", "CV Accuracy", "Train Time (s)"]
    existing = [c for c in col_order if c in df.columns]
    other = [c for c in df.columns if c not in col_order]
    df = df[existing + other]

    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


def generate_full_report(all_results: dict, data: dict, config: dict):
    """
    Genera il report completo: tabella, matrici di confusione, curve ROC.
    """
    logger.info("=" * 50)
    logger.info("VALUTAZIONE FINALE E CONFRONTO MODELLI")
    logger.info("=" * 50)

    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    labels = get_particle_labels(data["label_encoder"])
    y_true = data["y_test"]

    # --- Tabella di confronto ---
    comparison = build_comparison_table(all_results, data)
    logger.info(f"\nTabella di confronto:\n{comparison.to_string(index=False)}")

    table_path = os.path.join(results_dir, "model_comparison.csv")
    comparison.to_csv(table_path, index=False)
    logger.info(f"Tabella salvata in {table_path}")

    # --- Classification report per ogni modello ---
    for name, res in all_results.items():
        report = classification_report(
            y_true, res["y_pred"], target_names=labels, digits=4
        )
        logger.info(f"\nClassification Report - {name}:\n{report}")

        report_path = os.path.join(results_dir, f"report_{_safe_name(name)}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report - {name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)

    # --- Matrice di confusione per ogni modello ---
    for name, res in all_results.items():
        plot_confusion_matrix(
            y_true, res["y_pred"], labels,
            title=f"Matrice di Confusione - {name}",
            config=config,
            filename=f"cm_{_safe_name(name)}.png",
        )

    # --- Curve ROC per modelli con probabilita' ---
    for name, res in all_results.items():
        if res.get("y_proba") is not None:
            plot_roc_curves(
                y_true, res["y_proba"], labels,
                title=f"Curve ROC - {name}",
                config=config,
                filename=f"roc_{_safe_name(name)}.png",
            )

    # --- Grafico di confronto accuracy ---
    _plot_accuracy_comparison(comparison, config)

    return comparison


def _plot_accuracy_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre di confronto delle accuracy."""
    import matplotlib.pyplot as plt

    fig_dir = config["paths"]["figures_dir"]
    dpi = config["visualization"]["dpi"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    models = comparison["Modello"].tolist()
    accs = comparison["accuracy"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    bars = ax.barh(models[::-1], accs[::-1], color=colors)
    for bar, acc in zip(bars, accs[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Accuracy")
    ax.set_title("Confronto Accuracy tra Modelli")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_comparison.png"))
    plt.close(fig)
    logger.info("Salvato model_comparison.png")


def _safe_name(name: str) -> str:
    """Converte un nome modello in un nome file sicuro."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
