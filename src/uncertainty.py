"""
Modulo di Uncertainty Quantification per il progetto PID.

Implementa:
- MC Dropout: esegue N forward pass con dropout attivo per stimare
  l'incertezza epistemica del modello MLP
- Analisi delle regioni di alta incertezza nel piano delle feature
"""

import logging
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.visualization import get_particle_labels

logger = logging.getLogger(__name__)


def enable_mc_dropout(model):
    """
    Abilita il dropout durante l'inferenza per MC Dropout.
    Mette in eval mode tutto tranne i layer di Dropout.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(model, X: np.ndarray, n_iterations: int,
                       device=None) -> dict:
    """
    Esegue N forward pass con dropout attivo (MC Dropout).

    Returns:
        Dict con:
        - mean_proba: probabilita' media (n_samples, n_classes)
        - std_proba: deviazione standard (n_samples, n_classes)
        - predictions: classe predetta (n_samples,)
        - entropy: entropia predittiva (n_samples,)
        - all_probas: tutte le predizioni (n_iterations, n_samples, n_classes)
    """
    if device is None:
        device = torch.device("cpu")

    enable_mc_dropout(model)
    X_tensor = torch.FloatTensor(X).to(device)

    all_probas = []
    with torch.no_grad():
        for i in range(n_iterations):
            outputs = model(X_tensor)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probas.append(probas)

    all_probas = np.array(all_probas)  # (n_iter, n_samples, n_classes)

    mean_proba = all_probas.mean(axis=0)
    std_proba = all_probas.std(axis=0)
    predictions = mean_proba.argmax(axis=1)

    # Entropia predittiva: H = -sum(p * log(p))
    epsilon = 1e-10
    entropy = -np.sum(mean_proba * np.log(mean_proba + epsilon), axis=1)

    return {
        "mean_proba": mean_proba,
        "std_proba": std_proba,
        "predictions": predictions,
        "entropy": entropy,
        "all_probas": all_probas,
    }


def run_uncertainty_analysis(mlp_results: dict, data: dict, config: dict):
    """
    Esegue l'analisi di incertezza con MC Dropout sul modello MLP.

    Produce:
    - Distribuzione dell'entropia predittiva
    - Scatter plot delle regioni di alta incertezza
    - Accuracy in funzione della soglia di incertezza (rejection curve)
    """
    if not config["uncertainty"]["enabled"]:
        logger.info("Uncertainty quantification disabilitata in config.")
        return {}

    if "MLP (PyTorch)" not in mlp_results and "model" not in mlp_results:
        logger.warning("Modello MLP non trovato, skip uncertainty analysis.")
        return {}

    logger.info("=" * 50)
    logger.info("FASE 5b: Uncertainty Quantification (MC Dropout)")
    logger.info("=" * 50)

    # Ottieni il modello MLP
    if "model" in mlp_results:
        model = mlp_results["model"]
        device = mlp_results.get("device", torch.device("cpu"))
    else:
        model = mlp_results["MLP (PyTorch)"]["model"]
        device = mlp_results["MLP (PyTorch)"].get("device", torch.device("cpu"))

    n_iter = config["uncertainty"]["mc_dropout_iterations"]
    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    labels = get_particle_labels(data["label_encoder"])

    logger.info(f"  MC Dropout con {n_iter} iterazioni...")
    mc_results = mc_dropout_predict(model, data["X_test"], n_iter, device)

    y_test = data["y_test"]
    entropy = mc_results["entropy"]
    y_pred = mc_results["predictions"]

    # --- 1. Distribuzione dell'entropia ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    correct = y_pred == y_test
    ax.hist(entropy[correct], bins=50, alpha=0.6, label="Corretti", density=True)
    ax.hist(entropy[~correct], bins=50, alpha=0.6, label="Errati", density=True)
    ax.set_xlabel("Entropia predittiva")
    ax.set_ylabel("Densita'")
    ax.set_title("Distribuzione incertezza: predizioni corrette vs errate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "uncertainty_entropy.png"))
    plt.close(fig)
    logger.info("  Salvato uncertainty_entropy.png")

    # --- 2. Rejection curve: accuracy vs % di eventi rifiutati ---
    thresholds = np.linspace(0, np.max(entropy), 100)
    accs = []
    fractions_kept = []
    for thr in thresholds:
        mask = entropy <= thr
        if mask.sum() == 0:
            continue
        accs.append((y_pred[mask] == y_test[mask]).mean())
        fractions_kept.append(mask.mean())

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.plot(fractions_kept, accs, "b-", lw=2)
    ax.set_xlabel("Frazione di eventi accettati")
    ax.set_ylabel("Accuracy sugli eventi accettati")
    ax.set_title("Rejection Curve: accuracy vs soglia di incertezza")
    ax.axhline(y=(y_pred == y_test).mean(), color="r", ls="--",
               label=f"Accuracy senza filtro: {(y_pred == y_test).mean():.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rejection_curve.png"))
    plt.close(fig)
    logger.info("  Salvato rejection_curve.png")

    # --- 3. Incertezza per classe ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    class_entropies = [entropy[y_test == c] for c in range(len(labels))]
    ax.boxplot(class_entropies, labels=labels)
    ax.set_ylabel("Entropia predittiva")
    ax.set_title("Distribuzione incertezza per tipo di particella")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "uncertainty_per_class.png"))
    plt.close(fig)
    logger.info("  Salvato uncertainty_per_class.png")

    # --- 4. Scatter delle regioni ad alta incertezza nel piano p vs dE/dx ---
    feature_names = data["feature_names"]
    X_test_raw = data["X_test_raw"]

    p_idx = next((i for i, n in enumerate(feature_names) if n.lower() == "p"), 0)
    e_idx = next((i for i, n in enumerate(feature_names) if n.lower() in ("ein", "eout")), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    # Colorato per classe
    for c in range(len(labels)):
        mask = y_test == c
        ax1.scatter(
            X_test_raw[mask, p_idx], X_test_raw[mask, e_idx],
            s=2, alpha=0.3, label=labels[c],
        )
    ax1.set_xlabel(feature_names[p_idx])
    ax1.set_ylabel(feature_names[e_idx])
    ax1.set_title("Classificazione nel piano p vs energia")
    ax1.legend(markerscale=5)

    # Colorato per incertezza
    sc = ax2.scatter(
        X_test_raw[:, p_idx], X_test_raw[:, e_idx],
        c=entropy, s=2, alpha=0.3, cmap="hot_r",
    )
    plt.colorbar(sc, ax=ax2, label="Entropia")
    ax2.set_xlabel(feature_names[p_idx])
    ax2.set_ylabel(feature_names[e_idx])
    ax2.set_title("Mappa di incertezza nel piano p vs energia")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "uncertainty_scatter.png"))
    plt.close(fig)
    logger.info("  Salvato uncertainty_scatter.png")

    return mc_results
