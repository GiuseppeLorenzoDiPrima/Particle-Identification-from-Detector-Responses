"""
Baseline a tagli (cuts-based PID).

Implementa il metodo tradizionale di particle identification usato
in fisica sperimentale: si definiscono regioni nel piano (tipicamente
dE/dx vs p) per separare le specie di particelle. Le soglie vengono
apprese automaticamente dai dati di training tramite percentili delle
distribuzioni per classe. Un nuovo elemento viene attribuito alla classe
con il numero maggiore di feature il cui valore cade dentro l'intervallo
atteso. In caso di parità, si usa la distanza dal centroide della classe.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Mappa dei nomi feature per i grafici matplotlib
MPL_FEATURE_LABELS = {
    "p": r"$p$",
    "theta": r"$\theta$",
    "beta": r"$\beta$",
    "nphe": r"$n_{phe}$",
    "ein": r"$E_{in}$",
    "eout": r"$E_{out}$",
}


class CutsBasedPID:
    """
    Classificatore a tagli sul piano multidimensionale dE/dx vs p.

    Strategia: per ogni feature, calcola i percentili [5, 95] di ogni
    classe sul training set. Per un nuovo evento, la classe assegnata
    è quella con il maggior numero di feature il cui valore cade dentro l'intervallo atteso.

    In caso di parità, si usa la distanza dal centroide della classe.
    """

    def __init__(self, config, data):
        self.ranges = {}                        # {class_id: {feat_idx: (low, high)}}
        self.centroids = {}                     # {class_id: centroid_vector}
        self.n_classes = 0
        self.low_percentile = config["baseline_cuts"].get("low_percentile", 5)      # Percentile inferiore
        self.high_percentile = config["baseline_cuts"].get("high_percentile", 95)   # Percentile superiore
        self.show_range = config["baseline_cuts"].get("show_range", False)          # Mostra range
        self.feature_names = data["feature_names"]
        self.class_names = data["class_names"]
        self.feature_for_print = data["feature_for_print"]
        self.graph = config["visualization"].get("graph", True)
        self.config = config
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calcola intervalli e centroidi per ogni classe."""
        classes = np.unique(y)
        self.n_classes = len(classes)

        for c in classes:
            mask = y == c
            X_c = X[mask]
            self.centroids[c] = X_c.mean(axis=0)                        # Centroide della classe c
            self.ranges[c] = {}
            for j in range(X.shape[1]):
                low = np.percentile(X_c[:, j], self.low_percentile)     # Percentile inferiore
                high = np.percentile(X_c[:, j], self.high_percentile)   # Percentile superiore
                self.ranges[c][j] = (low, high)                         # Range per la feature j della classe c
                
        # Se richiesto, mostra i range in tabella
        if self.show_range:
            self._print_ranges()
            
        # Creazione grafico per i range
        if self.graph:
            self.plot_feature_ranges(self.feature_names, self.config)

        print()
        logger.info(f"Baseline a tagli: {self.n_classes} classi fitted.")
        return self
    
    def _print_ranges(self):
        """Stampa i range calcolati in formato tabellare."""
        
        headers = ["Classe"] + self.feature_names
        table = []
        
        for c in range(len(self.class_names)):
            row = [self.class_names[c]]
            for j in range(len(self.feature_names)):
                low, high = self.ranges[c][j]
                row.append(f"{low:.3f} - {high:.3f}")
            table.append(row)
        
        print()
        print("Range feature per classe:")
        print(tabulate(table, headers=headers, tablefmt="grid"))

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

        # Tie-breaking: in caso di parità, scegli la classe più vicina (centroide)
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

    def plot_feature_ranges(self, feature_names: list[str] | None, config) -> None:
        """Crea il grafico tabellare dei range feature per classe."""
        if feature_names is None or not self.ranges:
            return

        style_name = config["visualization"].get("style", "seaborn-v0_8-whitegrid")
        try:
            sns.set_style(style_name)
        except ValueError:
            plt.style.use(style_name)
        palette_name = config["visualization"].get("palette", "Set2")
        try:
            _ = sns.color_palette(palette_name, len(self.class_names))
        except Exception:
            _ = sns.color_palette("Set2", len(self.class_names))

        fig_path = config["paths"]["figures_dir"]
        fig_dir = os.path.join(fig_path, "baseline")
        os.makedirs(fig_dir, exist_ok=True)

        headers = ["Classe"] + [MPL_FEATURE_LABELS.get(name, name) for name in feature_names]
        table_data = []
        for class_id in range(len(self.class_names)):
            row = [self.class_names[class_id]]
            for j in range(len(feature_names)):
                low, high = self.ranges[class_id][j]
                row.append(f"{low:.3f} - {high:.3f}")
            table_data.append(row)

        fig, ax = plt.subplots(figsize=(12, max(2.5, len(self.class_names) * 0.9)), dpi=config["visualization"]["dpi"])
        ax.axis("off")
        fig.patch.set_facecolor("white")

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colLoc="center",
            colColours=[sns.color_palette(palette_name)[0]] + ["#f7f7f7"] * len(feature_names),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor(sns.color_palette(palette_name)[0])
            else:
                cell.set_linewidth(0.5)
                cell.set_edgecolor("#dddddd")

        ax.set_title("Range feature per classe (baseline cuts)", fontsize=16, pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "range_features.png"), bbox_inches="tight", facecolor=fig.get_facecolor())
        print()
        logger.info(f"Salvato range_features.png in {str(fig_dir).replace(os.sep, '/')}")
        plt.close(fig)

def run_baseline(data: dict, config: dict) -> dict:
    """
    Esegue la pipeline della baseline a tagli.

    Usa i dati NON scalati (raw) perché i tagli operano sulle
    grandezze originali.
    """
    if not config["baseline_cuts"]["enabled"]:
        logger.info("Baseline a tagli disabilitata in config.")
        return {}

    logger.info("=" * 55)
    logger.info("FASE 2: Baseline a tagli. Percentili usati: [{}°, {}°].".format(
        config["baseline_cuts"].get("low_percentile", 5),
        config["baseline_cuts"].get("high_percentile", 95)
    ))
    logger.info("=" * 55)

    model = CutsBasedPID(config=config, data=data)
    
    # Esecuzione e valutazione
    model.fit(data["X_train_raw"], data["y_train"])
    results = model.evaluate(data["X_test_raw"], data["y_test"])
    results["model"] = model
    results["model_name"] = "Cuts-Based PID"

    return results
