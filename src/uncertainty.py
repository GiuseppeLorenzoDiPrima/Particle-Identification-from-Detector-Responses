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

from src.visualization import get_particle_labels, plot_uncertainty_results

logger = logging.getLogger(__name__)

# Mappa i nomi delle feature: nome -> simbolo per visualizzazione matplotlib
FEATURE_NAMES = {
    "p": r"$p$",
    "theta": r"$\theta$",
    "beta": r"$\beta$",
    "nphe": r"$n_{phe}$",
    "ein": r"$E_{in}$",
    "eout": r"$E_{out}$"
}


def enable_mc_dropout(model):
    """
    Abilita il dropout durante l'inferenza per MC Dropout.
    Mette in eval mode tutta la rete tranne i layer di Dropout.
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
        - mean_proba: probabilità media (n_samples, n_classes)
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

    logger.info("=" * 55)
    logger.info("FASE 5b: Uncertainty Quantification (MC Dropout)")
    logger.info("=" * 55)

    # Ottieni il modello MLP
    if "model" in mlp_results:
        model = mlp_results["model"]
        device = mlp_results.get("device", torch.device("cpu"))
    else:
        model = mlp_results["MLP (PyTorch)"]["model"]
        device = mlp_results["MLP (PyTorch)"].get("device", torch.device("cpu"))

    n_iter = config["uncertainty"]["mc_dropout_iterations"]
    logger.info(f"MC Dropout con {n_iter} iterazioni...")
    mc_results = mc_dropout_predict(model, data["X_test"], n_iter, device)

    plot_uncertainty_results(mc_results, data["y_test"], data, config)
    return mc_results
