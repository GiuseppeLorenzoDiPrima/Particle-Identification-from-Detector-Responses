"""
Baseline a tagli fisici (cuts-based PID).

Implementa il metodo tradizionale di particle identification usato
in fisica sperimentale: si definiscono regioni nel piano delle
osservabili (tipicamente dE/dx vs p) per separare le specie di
particelle. Le soglie vengono apprese automaticamente dai dati
di training tramite percentili delle distribuzioni per classe.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class CutsBasedPID:
    """
    Classificatore a tagli fisici sul piano multidimensionale.

    Strategia: per ogni feature, calcola i percentili [5, 95] di ogni
    classe sul training set. Per un nuovo evento, la classe assegnata
    e' quella con il maggior numero di feature il cui valore cade
    dentro l'intervallo atteso.

    In caso di parita', si usa la distanza dal centroide della classe.
    """

    def __init__(self):
        self.ranges = {}       # {class_id: {feat_idx: (low, high)}}
        self.centroids = {}    # {class_id: centroid_vector}
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calcola intervalli e centroidi per ogni classe."""
        classes = np.unique(y)
        self.n_classes = len(classes)

        for c in classes:
            mask = y == c
            X_c = X[mask]
            self.centroids[c] = X_c.mean(axis=0)
            self.ranges[c] = {}
            for j in range(X.shape[1]):
                low = np.percentile(X_c[:, j], 5)
                high = np.percentile(X_c[:, j], 95)
                self.ranges[c][j] = (low, high)

        logger.info(f"Baseline a tagli: {self.n_classes} classi fitted.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predizione per ciascun evento."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        predictions = np.zeros(n_samples, dtype=int)

        # Per ogni evento, conta quante feature cadono nell'intervallo di ogni classe
        scores = np.zeros((n_samples, self.n_classes))
        for c in range(self.n_classes):
            for j in range(n_features):
                low, high = self.ranges[c][j]
                in_range = (X[:, j] >= low) & (X[:, j] <= high)
                scores[:, c] += in_range.astype(float)

        # Tie-breaking: in caso di parita', scegli la classe piu' vicina (centroide)
        for i in range(n_samples):
            max_score = scores[i].max()
            tied = np.where(scores[i] == max_score)[0]
            if len(tied) == 1:
                predictions[i] = tied[0]
            else:
                # Distanza euclidea dal centroide
                dists = [
                    np.linalg.norm(X[i] - self.centroids[c]) for c in tied
                ]
                predictions[i] = tied[np.argmin(dists)]

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Valuta il classificatore e restituisce le metriche."""
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        logger.info(f"Baseline cuts-based accuracy: {acc:.4f}")
        return {"y_pred": y_pred, "accuracy": acc}


def run_baseline(data: dict, config: dict) -> dict:
    """
    Esegue la pipeline della baseline a tagli.

    Usa i dati NON scalati (raw) perche' i tagli fisici operano
    sulle grandezze originali.
    """
    if not config["baseline_cuts"]["enabled"]:
        logger.info("Baseline a tagli disabilitata in config.")
        return {}

    logger.info("=" * 50)
    logger.info("FASE 2: Baseline a tagli fisici")
    logger.info("=" * 50)

    model = CutsBasedPID()
    model.fit(data["X_train_raw"], data["y_train"])

    results = model.evaluate(data["X_test_raw"], data["y_test"])
    results["model"] = model
    results["model_name"] = "Cuts-Based PID"

    return results
