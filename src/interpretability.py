"""
Modulo di interpretabilità implementato attraverso SHAP values.

Analizza quali feature sono più importanti per l'identificazione
di ciascun tipo di particella, sia per i modelli classici (tree-based)
che per il MLP (deep learning).
"""

import logging
import os

import numpy as np
import shap # type: ignore
import matplotlib.pyplot as plt

from src.visualization import get_particle_labels

logger = logging.getLogger(__name__)


def _to_list_format(shap_values, n_classes):
    """
    Normalizza l'output SHAP al formato lista [array_classe_0, ..., array_classe_N].

    shap.TreeExplainer.shap_values() restituisce:
      - (n_samples, n_features, n_classes) come array 3D
    shap.KernelExplainer.shap_values() restituisce:
      - lista di array (n_samples, n_features), uno per classe

    Questa funzione converte entrambi i formati in lista.
    """
    if isinstance(shap_values, list):
        return shap_values

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) -> lista di (n_samples, n_features)
        return [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    return [shap_values]


def run_shap_analysis(all_results: dict, data: dict, config: dict):
    """
    Esegue l'analisi SHAP su modelli selezionati.

    Produce:
    - Summary plot (beeswarm) per ogni modello
    - Bar plot dell'importanza media per feature
    - Summary plot per singola classe
    """
    if not config["interpretability"]["enabled"]:
        logger.info("Analisi SHAP disabilitata in config.")
        return

    logger.info("=" * 55)
    logger.info("FASE 5a: Interpretabilità (SHAP)")
    logger.info("=" * 55)

    fig_dir = config["paths"]["figures_dir"]
    shap_dir = os.path.join(fig_dir, "SHAP")
    os.makedirs(shap_dir, exist_ok=True)
    n_samples = config["interpretability"]["shap_samples"]
    feature_names = data["feature_names"]
    labels = get_particle_labels(data["label_encoder"])
    n_classes = len(labels)
    dpi = config["visualization"]["dpi"]

    # Subsample per SHAP
    idx = np.random.choice(
        len(data["X_test"]), min(n_samples, len(data["X_test"])), replace=False
    )
    X_sample = data["X_test"][idx]

    # --- SHAP per modelli tree-based ---
    tree_models = ["Random Forest", "XGBoost", "Decision Tree"]
    logger.info(f"Analisi SHAP su {n_samples} campioni...")
    for name in tree_models:
        if name not in all_results:
            continue
        model = all_results[name]["model"]
        logger.info(f"  SHAP TreeExplainer per {name}...")

        try:
            explainer = shap.TreeExplainer(model)
            raw_sv = explainer.shap_values(X_sample)
            sv_list = _to_list_format(raw_sv, n_classes)
            _plot_shap_all(sv_list, X_sample, feature_names, labels, name, shap_dir, dpi)
        except Exception as e:
            logger.warning(f"    SHAP fallito per {name}: {e}")

    # --- SHAP per MLP (se presente) ---
    mlp_name = "MLP (PyTorch)"
    if mlp_name in all_results:
        logger.info("  SHAP KernelExplainer per MLP...")
        try:
            import torch

            model = all_results[mlp_name]["model"]
            device = all_results[mlp_name].get("device", torch.device("cpu"))

            def mlp_predict(X):
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X).to(device)
                    out = torch.softmax(model(X_t), dim=1)
                    return out.cpu().numpy()

            # KernelExplainer: usa gli stessi campioni per background e spiegazione
            n_shap = min(100, len(X_sample))
            X_shap = X_sample[:n_shap]
            background = shap.kmeans(X_shap, 50)
            explainer = shap.KernelExplainer(mlp_predict, background)
            raw_sv = explainer.shap_values(X_shap)
            sv_list = _to_list_format(raw_sv, n_classes)
            _plot_shap_all(sv_list, X_shap, feature_names, labels, "MLP", shap_dir, dpi)
        except Exception as e:
            logger.warning(f"    SHAP fallito per MLP: {e}")


def _plot_shap_all(sv_list, X_sample, feature_names, labels,
                   model_name, fig_dir, dpi):
    """
    Genera tutti i plot SHAP per un modello multiclasse.

    Args:
        sv_list: lista di array (n_classes,), ciascuno (n_samples, n_features).
    """
    # Mappa i nomi delle feature: nome -> simbolo per visualizzazione matplotlib
    FEATURE_NAMES = {
        "p": r"$p$",
        "theta": r"$\theta$",
        "beta": r"$\beta$",
        "nphe": r"$n_{phe}$",
        "ein": r"$E_{in}$",
        "eout": r"$E_{out}$"
    }
    
    safe_name = _safe(model_name)

    # --- 1. Summary plot aggregato (tutte le classi) ---
    shap.summary_plot(
        sv_list, X_sample,
        feature_names=[FEATURE_NAMES.get(name, name) for name in feature_names],
        class_names=labels,
        show=False,
    )
    plt.title(f"SHAP Summary - {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"SHAP_summary_{safe_name}.png"), dpi=dpi)
    plt.close("all")
    logger.info(f"    Salvato SHAP_summary_{safe_name}.png")

    # --- 2. Bar plot: importanza media per feature ---
    mean_abs = np.mean(
        [np.abs(sv).mean(axis=0) for sv in sv_list], axis=0
    )
    sorted_idx = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.barh(
        [FEATURE_NAMES.get(feature_names[i], feature_names[i]) for i in sorted_idx],#type: ignore
        mean_abs[sorted_idx],
    )
    ax.set_xlabel("Mean Absolute SHAP value")
    ax.set_title(f"SHAP Feature Importance - {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"SHAP_bar_{safe_name}.png"))
    plt.close(fig)
    logger.info(f"    Salvato SHAP_bar_{safe_name}.png")

    # --- 3. Summary plot per singola classe ---
    for class_idx, label in enumerate(labels):
        shap.summary_plot(
            sv_list[class_idx], X_sample,
            feature_names=[FEATURE_NAMES.get(name, name) for name in feature_names],
            show=False,
        )
        plt.title(f"SHAP {model_name} - {label}", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"SHAP_{safe_name}_class_{class_idx}.png"),
            dpi=dpi,
        )
        plt.close("all")
    logger.info(f"    Salvati plot SHAP per classe ({model_name}).")


def _safe(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
