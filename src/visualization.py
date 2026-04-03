"""
Modulo di visualizzazione per il progetto PID.

Include:
- Diagramma di Bethe-Bloch (dE/dx vs p)
- Distribuzioni delle feature per classe
- Matrici di confusione
- Curve ROC
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.data_loader import PARTICLE_NAMES

logger = logging.getLogger(__name__)


def setup_style(config: dict):
    """Imposta lo stile dei grafici dalla configurazione."""
    style = config["visualization"].get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette(config["visualization"].get("palette", "Set2"))


def get_particle_labels(label_encoder) -> list[str]:
    """Restituisce i nomi fisici delle particelle nell'ordine del LabelEncoder."""
    labels = []
    for cls in label_encoder.classes_:
        name = PARTICLE_NAMES.get(cls, str(cls))
        labels.append(name)
    return labels


def plot_bethe_bloch(data: dict, config: dict):
    """
    Plot 2D dE/dx vs momento p, colorato per tipo di particella.
    Questo e' il diagramma fondamentale per la PID in fisica sperimentale.

    Cerca le colonne piu' pertinenti tra le feature disponibili.
    """
    setup_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = data["feature_names"]
    X_raw = data["X_train_raw"]
    y = data["y_train"]
    labels = get_particle_labels(data["label_encoder"])

    # Identifica la colonna del momento (p) e una proxy per dE/dx
    p_idx = _find_feature_index(feature_names, ["p", "momentum"])
    dedx_candidates = ["ein", "eout", "edep", "dedx", "de_dx", "energy"]
    dedx_idx = _find_feature_index(feature_names, dedx_candidates)

    if p_idx is None or dedx_idx is None:
        # Fallback: usa le prime due feature
        logger.warning(
            "Non trovate colonne p/dE/dx esatte; uso le prime due feature."
        )
        p_idx, dedx_idx = 0, 1

    figsize = config["visualization"]["figsize"]
    dpi = config["visualization"]["dpi"]

    # Subsample per leggibilita'
    n_plot = min(50000, len(X_raw))
    idx = np.random.choice(len(X_raw), n_plot, replace=False)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for class_id in np.unique(y):
        mask = y[idx] == class_id
        ax.scatter(
            X_raw[idx[mask], p_idx],
            X_raw[idx[mask], dedx_idx],
            s=2, alpha=0.3,
            label=labels[class_id],
        )

    ax.set_xlabel(f"{feature_names[p_idx]} (momento)", fontsize=12)
    ax.set_ylabel(f"{feature_names[dedx_idx]} (energia depositata)", fontsize=12)
    ax.set_title("Diagramma Bethe-Bloch: energia depositata vs momento", fontsize=14)
    ax.legend(markerscale=5, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bethe_bloch.png"))
    plt.close(fig)
    logger.info("Salvato bethe_bloch.png")


def plot_feature_distributions(data: dict, config: dict):
    """Distribuzione di ogni feature, separata per classe di particella."""
    setup_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = data["feature_names"]
    X_raw = data["X_train_raw"]
    y = data["y_train"]
    labels = get_particle_labels(data["label_encoder"])
    dpi = config["visualization"]["dpi"]

    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=dpi)
    axes = axes.flatten()

    for i, fname in enumerate(feature_names):
        ax = axes[i]
        for class_id in np.unique(y):
            mask = y == class_id
            ax.hist(
                X_raw[mask, i], bins=80, alpha=0.5,
                label=labels[class_id], density=True,
            )
        ax.set_title(fname, fontsize=11)
        ax.legend(fontsize=8)

    # Nascondi assi vuoti
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuzioni feature per tipo di particella", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_distributions.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvato feature_distributions.png")


def plot_class_distribution(data: dict, config: dict):
    """Grafico a barre della distribuzione delle classi."""
    setup_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    y = data["y_train"]
    labels = get_particle_labels(data["label_encoder"])
    dpi = config["visualization"]["dpi"]

    unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    bars = ax.bar([labels[u] for u in unique], counts, color=sns.color_palette())
    ax.set_ylabel("Numero di eventi")
    ax.set_title("Distribuzione classi nel training set")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}", ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "class_distribution.png"))
    plt.close(fig)
    logger.info("Salvato class_distribution.png")


def plot_correlation_matrix(data: dict, config: dict):
    """Matrice di correlazione tra le feature."""
    setup_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = data["feature_names"]
    X_raw = data["X_train_raw"]
    dpi = config["visualization"]["dpi"]

    df = pd.DataFrame(X_raw, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matrice di correlazione delle feature")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_matrix.png"))
    plt.close(fig)
    logger.info("Salvato correlation_matrix.png")


def plot_confusion_matrix(y_true, y_pred, labels: list[str], title: str,
                          config: dict, filename: str, subdir: str = "confusion_matrix"):
    """Salva la matrice di confusione come immagine."""
    fig_dir = config["paths"]["figures_dir"]
    subdir_dir = os.path.join(fig_dir, subdir)
    os.makedirs(subdir_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(subdir_dir, filename))
    plt.close(fig)
    logger.info(f"Salvato {filename} in {subdir_dir}")


def plot_roc_curves(y_true, y_score, labels: list[str], title: str,
                    config: dict, filename: str, subdir: str = "roc"):
    """
    Curve ROC one-vs-rest per classificazione multiclasse.
    y_score: matrice (n_samples, n_classes) di probabilita'.
    """
    fig_dir = config["paths"]["figures_dir"]
    subdir_dir = os.path.join(fig_dir, subdir)
    os.makedirs(subdir_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i]) # type: ignore
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(subdir_dir, filename))
    plt.close(fig)
    logger.info(f"Salvato {filename} in {subdir_dir}")


def _find_feature_index(feature_names: list[str], candidates: list[str]):
    """Cerca l'indice della prima feature il cui nome matcha un candidato."""
    for i, name in enumerate(feature_names):
        if name.lower() in [c.lower() for c in candidates]:
            return i
    return None
