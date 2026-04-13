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

from data_classes.data_loader import PARTICLE_NAMES

logger = logging.getLogger(__name__)

# Palette IEEE-ready: colori distinti, colorblind-safe, adatti alla stampa in B&W
IEEE_PALETTE = [
    "#2166AC",  # blu
    "#B2182B",  # rosso scuro
    "#1B7837",  # verde foresta
    "#D6604D",  # arancione mattone
    "#762A83",  # viola
    "#4D4D4D",  # grigio antracite
]

# Linestyles e markers per curve multi-classe (ROC, convergenza, ecc.)
IEEE_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
IEEE_MARKERS    = ["o", "s", "D", "^", "v", "P"]

# Mappa i nomi delle feature: nome -> simbolo per visualizzazione matplotlib
FEATURE_NAMES = {
    "p": r"$p$",
    "theta": r"$\theta$",
    "beta": r"$\beta$",
    "nphe": r"$n_{phe}$",
    "ein": r"$E_{in}$",
    "eout": r"$E_{out}$"
}

def setup_publication_style(config: dict):
    """Imposta lo stile grafico per paper scientifici (IEEE-ready)."""
    sns.set_palette(IEEE_PALETTE)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.edgecolor": "#d3d3d3",
        "legend.fancybox": False,
        "figure.dpi": config["visualization"].get("dpi", 300),
        "savefig.dpi": config["visualization"].get("dpi", 300),
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#333333",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def get_particle_labels(label_encoder) -> list[str]:
    """Restituisce i nomi fisici delle particelle nell'ordine del LabelEncoder."""
    labels = []
    for cls in label_encoder.classes_:
        name = PARTICLE_NAMES.get(cls, str(cls))
        labels.append(name)
    return labels


def plot_bethe_bloch(data: dict, config: dict):
    """
    Plot 2D dE/dx vs quantità di moto p, colorato per tipo di particella.
    Questo e' il diagramma fondamentale per la PID in fisica sperimentale.

    Cerca le colonne piu' pertinenti tra le feature disponibili.
    """
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = data["feature_names"]
    X_raw = data["X_train_raw"]
    y = data["y_train"]
    labels = get_particle_labels(data["label_encoder"])

    # Identifica la colonna della quantità di moto (p) e una proxy per dE/dx
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
            label=labels[class_id].capitalize(),
        )

    feature_names = list(map(FEATURE_NAMES.get, feature_names))
    ax.set_xlabel(f"{feature_names[p_idx]} (quantità di moto)", fontsize=12)
    ax.set_ylabel(f"{feature_names[dedx_idx]} (energia depositata)", fontsize=12)
    ax.set_title("Diagramma Bethe-Bloch: energia depositata vs quantità di moto", fontsize=14)
    ax.legend(markerscale=5, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bethe_bloch.png"))
    plt.close(fig)
    logger.info("  Salvato bethe_bloch.png")


def plot_feature_distributions(data: dict, config: dict):
    """Distribuzione di ogni feature, separata per classe di particella."""
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = list(map(FEATURE_NAMES.get, data["feature_names"])) if data.get("feature_names") else []
    X_raw = data["X_train_raw"]
    y = data["y_train"]
    labels = get_particle_labels(data["label_encoder"])
    dpi = config["visualization"]["dpi"]

    # figsize = config["visualization"]["figsize"]
    figsize = [14, 8]  # Override per più spazio
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols / 2, figsize[1] * n_rows / 1.5), dpi=dpi)
    axes = axes.flatten()

    for i, fname in enumerate(feature_names):
        ax = axes[i]
        for j, class_id in enumerate(np.unique(y)):
            mask = y == class_id
            ax.hist(
                X_raw[mask, i], bins=60, alpha=0.55,
                label=labels[class_id].capitalize(), density=True,
                color=IEEE_PALETTE[j % len(IEEE_PALETTE)],
                edgecolor="white", linewidth=0.4,
            )
        ax.set_title(fname, fontsize=11, pad=8)
        ax.grid(True, linestyle=":", alpha=0.6, color="#A9A9A9")
        ax.legend(fontsize=8)

    # Nascondi assi vuoti
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuzioni feature per tipo di particella", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_distributions.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("  Salvato feature_distributions.png")


def plot_class_distribution(data: dict, config: dict):
    """Grafici a barre della distribuzione delle classi per ogni split e per il dataset completo."""
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    labels = get_particle_labels(data["label_encoder"])
    dpi = config["visualization"]["dpi"]
    figsize = config["visualization"]["figsize"]

    # Ricostruisce y_full concatenando i tre split
    y_full = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])

    splits = {
        "training":   (data["y_train"], "class_distribution_train.png"),
        "validation": (data["y_val"],   "class_distribution_val.png"),
        "test":       (data["y_test"],  "class_distribution_test.png"),
        "completo":   (y_full,          "class_distribution_full.png"),
    }

    for split_name, (y, filename) in splits.items():
        unique, counts = np.unique(y, return_counts=True)
        class_labels = [labels[u].capitalize() for u in unique]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bars = ax.bar(
            class_labels, counts,
            color=IEEE_PALETTE[:len(unique)],
            edgecolor="#333333", linewidth=0.8,
        )
        ax.set_ylabel("Numero di campioni", fontweight="bold")
        formatted = f"Distribuzione classi — {split_name} set" if split_name != "completo" else "Distribuzione classi — Dataset completo"
        ax.set_title(formatted, pad=12)
        ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="y", zorder=0)
        for bar in bars:
            bar.set_zorder(2)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, filename))
        plt.close(fig)
        logger.info(f"  Salvato {filename}")


def plot_correlation_matrix(data: dict, config: dict):
    """Matrice di correlazione tra le features."""
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = fig_path + "/pre-processing"
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = list(map(FEATURE_NAMES.get, data["feature_names"]))
    X_raw = data["X_train_raw"]
    dpi = config["visualization"]["dpi"]

    figsize = config["visualization"]["figsize"]
    df = pd.DataFrame(X_raw, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matrice di correlazione delle features")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_matrix.png"))
    plt.close(fig)
    logger.info("  Salvato correlation_matrix.png")


def plot_confusion_matrix(y_true, y_pred, labels: list[str], title: str,
                          config: dict, filename: str, subdir: str = "confusion_matrix"):
    """Salva la matrice di confusione come immagine."""
    fig_dir = config["paths"]["figures_dir"]
    subdir_dir = os.path.join(fig_dir, subdir)
    os.makedirs(subdir_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    setup_publication_style(config)
    figsize = config["visualization"]["figsize"]
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    disp = ConfusionMatrixDisplay(cm, display_labels=[label.capitalize() for label in labels])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

    # Tutti e 4 i bordi esterni visibili e marcati
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#333333")

    # Rimuovi la griglia di default (cade a metà cella) e sostituisci
    # con linee di separazione allineate ai bordi delle celle
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.6, linestyle="-")
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(subdir_dir, filename))
    plt.close(fig)
    logger.info(f"  Salvato {filename} in {str(subdir_dir).replace(os.sep, '/')}")


def plot_roc_curves(y_true, y_score, labels: list[str], title: str,
                    config: dict, filename: str, subdir: str = "roc_curves"):
    """
    Curve ROC one-vs-rest per classificazione multiclasse.
    y_score: matrice (n_samples, n_classes) di probabilita'.
    """
    fig_dir = config["paths"]["figures_dir"]
    subdir_dir = os.path.join(fig_dir, subdir)
    os.makedirs(subdir_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    setup_publication_style(config)
    # figsize = config["visualization"]["figsize"]
    figsize = [10, 8]  # Override per meno spazio
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i]) # type: ignore
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            lw=2,
            color=IEEE_PALETTE[i % len(IEEE_PALETTE)],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            marker=IEEE_MARKERS[i % len(IEEE_MARKERS)],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markevery=max(1, len(fpr) // 10),
            alpha=0.95,
            label=f"{labels[i].capitalize()} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--", alpha=0.7, zorder=1)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, pad=12)
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", zorder=0)
    ax.legend(fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(subdir_dir, filename))
    plt.close(fig)
    logger.info(f"  Salvato {filename} in {str(subdir_dir).replace(os.sep, '/')}")


def _find_feature_index(feature_names: list[str], candidates: list[str]):
    """Cerca l'indice della prima feature il cui nome matcha un candidato."""
    for i, name in enumerate(feature_names):
        if name.lower() in [c.lower() for c in candidates]:
            return i
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, config: dict):
    """Grafico loss e accuracy durante il training MLP."""
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = os.path.join(fig_path, "training")
    os.makedirs(fig_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    # figsize = config["visualization"]["figsize"]
    figsize = [14, 8]  # Override per grafici più ampi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.8,
             color=IEEE_PALETTE[0])
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.8,
             color=IEEE_PALETTE[1], linestyle="--")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training e Validation Loss", pad=12)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=1.8,
             color=IEEE_PALETTE[0])
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", linewidth=1.8,
             color=IEEE_PALETTE[1], linestyle="--")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy", fontweight="bold")
    ax2.set_title("Training e Validation Accuracy", pad=12)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "mlp_training_history.png"))
    plt.close(fig)
    logger.info("  Salvato mlp_training_history.png")


# ---------------------------------------------------------------------------
# Modelli classici
# ---------------------------------------------------------------------------

def plot_feature_importance(results: dict, feature_names: list, config: dict):
    """Grafico della feature importance per i modelli che la supportano."""
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    os.makedirs(fig_path, exist_ok=True)
    fig_dir = os.path.join(fig_path, "training")
    os.makedirs(fig_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    # figsize = config["visualization"]["figsize"]
    figsize = [18, 8]  # Override per grafici più ampi

    models_with_fi = {
        name: res["feature_importance"]
        for name, res in results.items()
        if res["feature_importance"] is not None
    }
    if not models_with_fi:
        return

    n = len(models_with_fi)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, (name, fi) in zip(axes, models_with_fi.items()):
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        feat_names = [FEATURE_NAMES.get(x[0], x[0]) for x in sorted_fi]
        values = [x[1] for x in sorted_fi]
        n_feats = len(feat_names)
        colors = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(n_feats)]
        ax.barh(feat_names[::-1], values[::-1], color=colors[::-1],
                edgecolor="#333333", linewidth=0.8, zorder=2)
        ax.set_title(f"Feature Importance — {name}", pad=12)
        ax.set_xlabel("Importanza", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="x", zorder=0)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"))
    plt.close(fig)
    logger.info("  Salvato feature_importance.png")


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------

def plot_uncertainty_results(mc_results: dict, y_test: np.ndarray,
                              data: dict, config: dict):
    """Quattro grafici di uncertainty quantification da MC Dropout."""
    setup_publication_style(config)
    fig_dir = config["paths"]["figures_dir"]
    uncertainty_dir = os.path.join(fig_dir, "uncertainty")
    os.makedirs(uncertainty_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    # figsize = config["visualization"]["figsize"]
    figsize = [10, 8]  # Override per grafici più compatti
    labels = get_particle_labels(data["label_encoder"])

    entropy = mc_results["entropy"]
    y_pred = mc_results["predictions"]

    # 1. Distribuzione dell'entropia
    correct = y_pred == y_test
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(entropy[correct], bins=50, alpha=0.6, label="Predizioni corrette",
            density=True, color=IEEE_PALETTE[0], edgecolor="white", linewidth=0.4)
    ax.hist(entropy[~correct], bins=50, alpha=0.6, label="Predizioni errate",
            density=True, color=IEEE_PALETTE[1], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Entropia", fontweight="bold")
    ax.set_ylabel("Densità", fontweight="bold")
    ax.set_title("Distribuzione incertezza: predizioni corrette vs errate", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fig.savefig(os.path.join(uncertainty_dir, "uncertainty_entropy.png"))
    plt.close(fig)
    logger.info("  Salvato uncertainty_entropy.png")

    # 2. Rejection curve
    thresholds = np.linspace(0, np.max(entropy), 100)
    accs, fractions_kept = [], []
    for thr in thresholds:
        mask = entropy <= thr
        if mask.sum() == 0:
            continue
        accs.append((y_pred[mask] == y_test[mask]).mean())
        fractions_kept.append(mask.mean())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot([k * 100 for k in fractions_kept], accs, lw=2, color=IEEE_PALETTE[0])
    ax.set_xlabel("Percentuale di eventi accettati", fontweight="bold")
    ax.set_ylabel("Accuracy sugli eventi accettati", fontweight="bold")
    ax.set_title("Rejection Curve: accuracy vs soglia di incertezza", pad=12)
    ax.axhline(y=(y_pred == y_test).mean(), color=IEEE_PALETTE[1], ls="--",
               label=f"Accuracy senza filtro: {(y_pred == y_test).mean():.4f}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fig.savefig(os.path.join(uncertainty_dir, "rejection_curve.png"))
    plt.close(fig)
    logger.info("  Salvato rejection_curve.png")

    # 3. Incertezza per classe
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    class_entropies = [entropy[y_test == c] for c in range(len(labels))]
    ax.boxplot(class_entropies, labels=[l.capitalize() for l in labels])  # type: ignore
    ax.set_ylabel("Entropia", fontweight="bold")
    ax.set_title("Distribuzione incertezza per tipo di particella", pad=12)
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9")
    fig.tight_layout()
    fig.savefig(os.path.join(uncertainty_dir, "uncertainty_per_class.png"))
    plt.close(fig)
    logger.info("  Salvato uncertainty_per_class.png")

    # 4. Scatter p vs energia colorato per classe e per entropia
    feature_names = data["feature_names"]
    X_test_raw = data["X_test_raw"]
    p_idx = next((i for i, n in enumerate(feature_names) if n.lower() == "p"), 0)
    e_idx = next((i for i, n in enumerate(feature_names)
                  if n.lower() in ("ein", "eout")), 1)

    figsize = [14, 8]  # Override per grafici più ampi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for c in range(len(labels)):
        mask = y_test == c
        ax1.scatter(X_test_raw[mask, p_idx], X_test_raw[mask, e_idx],
                    s=2, alpha=0.3, label=labels[c].capitalize(),
                    color=IEEE_PALETTE[c % len(IEEE_PALETTE)])
    ax1.set_xlabel(FEATURE_NAMES.get(feature_names[p_idx], feature_names[p_idx]),
                   fontweight="bold")
    ax1.set_ylabel(FEATURE_NAMES.get(feature_names[e_idx], feature_names[e_idx]),
                   fontweight="bold")
    ax1.set_title("Classificazione nel piano p vs energia", pad=12)
    ax1.legend(markerscale=5)

    sc = ax2.scatter(X_test_raw[:, p_idx], X_test_raw[:, e_idx],
                     c=entropy, s=2, alpha=0.3, cmap="hot_r")
    plt.colorbar(sc, ax=ax2, label="Entropia")
    ax2.set_xlabel(FEATURE_NAMES.get(feature_names[p_idx], feature_names[p_idx]),
                   fontweight="bold")
    ax2.set_ylabel(FEATURE_NAMES.get(feature_names[e_idx], feature_names[e_idx]),
                   fontweight="bold")
    ax2.set_title("Mappa di incertezza nel piano p vs energia", pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(uncertainty_dir, "uncertainty_scatter.png"))
    plt.close(fig)
    logger.info("  Salvato uncertainty_scatter.png")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def plot_shap_results(sv_list, X_sample, feature_names: list, labels: list,
                      model_name: str, fig_dir: str, dpi: int, figsize: tuple):
    """Genera summary plot, bar plot e plot per classe per un modello SHAP."""
    import shap  # type: ignore

    CLASS_NAMES = {1: "positrone", 2: "kaone", 3: "pione", 4: "protone"}

    def _safe(name: str) -> str:
        return name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    feat_labels = [FEATURE_NAMES.get(n, n) for n in feature_names]
    safe_name = _safe(model_name)
    capitalized_labels = [l.capitalize() for l in labels]

    # 1. Summary aggregato
    figsize_1 = (6, 4)  # Plot più compatti per classe
    shap.summary_plot(sv_list, X_sample, feature_names=feat_labels,
                      class_names=capitalized_labels, show=False, plot_size=None)
    fig_s = plt.gcf()
    ax_s = plt.gca()
    ax_s.set_title(f"SHAP Summary — {model_name}", fontsize=13, pad=12)
    ax_s.set_xlabel("Mean Absolute SHAP value", fontweight="bold", fontsize=11)
    ax_s.tick_params(labelsize=10)
    leg = ax_s.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(10)
    fig_s.set_size_inches(figsize_1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"SHAP_summary_{safe_name}.png"), dpi=dpi)
    plt.close("all")
    logger.info(f"    Salvato SHAP_summary_{safe_name}.png")

    # 2. Bar plot importanza media
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in sv_list], axis=0)
    sorted_idx = np.argsort(mean_abs)
    n_feats = len(sorted_idx)
    bar_colors = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(n_feats)]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.barh([feat_labels[i] for i in sorted_idx], mean_abs[sorted_idx],
            color=bar_colors, edgecolor="#333333", linewidth=0.8, zorder=2)
    ax.set_xlabel("Mean Absolute SHAP value", fontweight="bold")
    ax.set_title(f"SHAP Feature Importance — {model_name}", pad=12)
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="x", zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"SHAP_bar_{safe_name}.png"))
    plt.close(fig)
    logger.info(f"    Salvato SHAP_bar_{safe_name}.png")

    # 3. Summary per singola classe
    for class_idx, label in enumerate(labels):
        shap.summary_plot(sv_list[class_idx], X_sample, feature_names=feat_labels,
                          show=False, plot_size=None)
        fig_c = plt.gcf()
        ax_c = plt.gca()
        ax_c.set_title(f"SHAP {model_name} — {label.capitalize()}", fontsize=13, pad=12)
        ax_c.set_xlabel("Mean Absolute SHAP value", fontweight="bold", fontsize=11)
        ax_c.tick_params(labelsize=10)
        fig_c.set_size_inches(figsize_1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"SHAP_{safe_name}_class_{CLASS_NAMES.get(class_idx + 1)}.png"),
            dpi=dpi,
        )
        plt.close("all")
    logger.info(f"    Salvati plot SHAP per classe ({model_name}).")


# ---------------------------------------------------------------------------
# Confronto modelli
# ---------------------------------------------------------------------------

def plot_metrics_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre orizzontali per ogni metrica, un file per metrica."""
    setup_publication_style(config)
    metrics = config["visualization"].get("comparison_metrics", ["accuracy"])
    metrics = [m for m in metrics if m in comparison.columns]
    if not metrics:
        logger.warning("Nessuna metrica valida trovata per il confronto grafico.")
        return

    fig_dir = config["paths"]["figures_dir"]
    subdir = os.path.join(fig_dir, "model_comparison")
    os.makedirs(subdir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    # figsize = config["visualization"]["figsize"]
    figsize = [10, 8]  # Override per grafici più compatti

    models = comparison["Modello"].tolist()
    n_models = len(models)
    colors = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(n_models)]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        values = comparison[metric].tolist()
        bars = ax.barh(models[::-1], values[::-1], color=colors[::-1],
                       edgecolor="#333333", linewidth=0.8)
        for bar, value in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}", va="center", fontsize=9)
        ax.set_xlabel(metric.replace("_", " ").capitalize(), fontweight="bold")
        ax.set_title(f"Confronto {metric.replace('_', ' ').capitalize()} tra Modelli", pad=12)
        ax.set_xlim(0, 1.08)
        ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="x", zorder=0)
        for bar in bars:
            bar.set_zorder(2)
        fig.tight_layout()
        filename = f"model_{metric}_comparison.png"
        fig.savefig(os.path.join(subdir, filename))
        plt.close(fig)
        logger.info(f"  Salvato {filename} in {str(subdir).replace(os.sep, '/')}")


def plot_metric_groups_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre raggruppate per modello con multiple metriche."""
    setup_publication_style(config)
    metrics = config["visualization"].get("comparison_group_metrics", ["accuracy"])
    metrics = [m for m in metrics if m in comparison.columns]
    if not metrics:
        logger.warning("Nessuna metrica valida trovata per il confronto a gruppi.")
        return

    fig_dir = config["paths"]["figures_dir"]
    subdir = os.path.join(fig_dir, "model_comparison")
    os.makedirs(subdir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    # figsize = config["visualization"]["figsize"]
    figsize = [14, 8]  # Override per grafici più ampi

    models = comparison["Modello"].tolist()
    n_models = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    colors = [IEEE_PALETTE[i % len(IEEE_PALETTE)] for i in range(n_metrics)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, metric in enumerate(metrics):
        values = comparison[metric].tolist()
        ax.bar(x + i * width, values, width,
               label=metric.replace("_", " ").capitalize(),
               color=colors[i], edgecolor="#333333", linewidth=0.8, zorder=2)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Valore", fontweight="bold")
    ax.set_title("Confronto metriche per modello", pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metrica")
    ax.grid(True, linestyle=":", alpha=0.7, color="#A9A9A9", axis="y", zorder=0)
    for i, metric in enumerate(metrics):
        values = comparison[metric].tolist()
        for j, value in enumerate(values):
            ax.text(x[j] + i * width, value + 0.01, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    filename = "model_comparison_groups.png"
    fig.savefig(os.path.join(subdir, filename))
    plt.close(fig)
    logger.info(f"  Salvato {filename} in {str(subdir).replace(os.sep, '/')}")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def plot_baseline_ranges(class_names: list, ranges: list,
                         feature_names: list, config: dict):
    """Tabella dei range calcolati attraverso i percentili (baseline cuts)."""
    setup_publication_style(config)
    fig_path = config["paths"]["figures_dir"]
    fig_dir = os.path.join(fig_path, "baseline")
    os.makedirs(fig_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    headers = ["Classe"] + [FEATURE_NAMES.get(n, n) for n in feature_names]
    table_data = []
    for class_id, cname in enumerate(class_names):
        row = [cname]
        for j in range(len(feature_names)):
            low, high = ranges[class_id][j]
            row.append(f"{low:.3f} – {high:.3f}")
        table_data.append(row)

    fig, ax = plt.subplots(
        figsize=(max(10, len(feature_names) * 1.8), max(2.5, len(class_names) * 0.9)),
        dpi=dpi,
    )
    ax.axis("off")
    fig.patch.set_facecolor("white")

    header_color = IEEE_PALETTE[0]
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colLoc="center",
        colColours=[header_color] + ["#f7f7f7"] * len(feature_names),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_color)
        else:
            cell.set_linewidth(0.5)
            cell.set_edgecolor("#dddddd")

    ax.set_title("Tabella dei range calcolati attraverso i percentili (baseline cuts)", fontsize=16, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "range_features.png"),
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Salvato range_features.png in {str(fig_dir).replace(os.sep, '/')}")
