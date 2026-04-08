"""
Modulo di valutazione e confronto modelli.

Produce:
- Tabella riepilogativa con accuracy, F1 macro, AUC-ROC per classe
- Confronto visivo tra tutti i modelli
- Report testuale dei risultati
"""

import logging
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    pairwise_distances,
)
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import f_classif


from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    get_particle_labels,
)

# 3D interattivo per test (plotly opzionale)
try:
    import plotly.graph_objects as go # type: ignore
except ImportError:
    go = None

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=4) -> dict:
    """
    Calcola metriche complete per un modello.

    Returns:
        Dict con accuracy, f1_macro, precision_macro, recall_macro, f1_weighted,
        precision_weighted, recall_weighted e opzionalmente auc_roc_macro e auc_per_class.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }

    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            metrics["auc_roc_macro"] = roc_auc_score(
                y_bin, y_proba, multi_class="ovr", average="macro"
            )
            metrics["auc_roc_weighted"] = roc_auc_score(
                y_bin, y_proba, multi_class="ovr", average="weighted"
            )
            # AUC per singola classe
            for i in range(n_classes):
                metrics[f"auc_class_{i}"] = roc_auc_score(y_bin[:, i], y_proba[:, i]) # type: ignore
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
                 "recall_macro", "f1_weighted", "precision_weighted", "recall_weighted",
                 "auc_roc_macro", "auc_roc_weighted", "CV Accuracy", "Train Time (s)"]
    existing = [c for c in col_order if c in df.columns]
    other = [c for c in df.columns if c not in col_order]
    df = df[existing + other]

    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


def generate_full_report(all_results: dict, data: dict, config: dict):
    """
    Genera il report completo: tabella, matrici di confusione, curve ROC.
    """
    logger.info("=" * 55)
    logger.info("VALUTAZIONE FINALE E CONFRONTO MODELLI")
    logger.info("=" * 55)

    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    labels = get_particle_labels(data["label_encoder"])
    y_true = data["y_test"]

    # --- Tabella di confronto ---
    comparison = build_comparison_table(all_results, data)
    # Print rimossa. La tabella viene salvata come csv e mostrata solo alla fine.
    # logger.info(f"\nTabella di confronto:\n{comparison.to_string(index=False)}")

    table_path = os.path.join(results_dir, "model_comparison.csv")
    comparison.to_csv(table_path, index=False)
    logger.info(f"Tabella di confronto cvs salvata in {str(table_path).replace(os.sep, '/')}")

    # --- Model comparison report in formato text ---
    summary_path = os.path.join(results_dir, "report_model_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 17 + "\n")
        f.write("Model Comparison\n")
        f.write("=" * 17 + "\n")
        f.write(comparison.to_string(index=False))
    logger.info(f"Report di confronto testuale salvato in {str(summary_path).replace(os.sep, '/')}")

    # --- Classification report per ogni modello ---
    for name, res in all_results.items():
        report = classification_report(
            y_true, res["y_pred"], target_names=[label.capitalize() for label in labels], digits=4
        )
        logger.info(f"\nClassification Report - {name}:\n{report}")

        report_path = os.path.join(results_dir, f"report_{_safe_name(name)}.txt")
        with open(report_path, "w") as f:
            f.write("=" * 55 + "\n")
            f.write(f"Classification Report - {name}\n")
            f.write("=" * 55 + "\n")
            f.write(report) # type: ignore
            
        logger.info(f"Classification report salvato in {str(report_path).replace(os.sep, '/')}")
        
    print()
    logger.info("Salvataggio immagini e report finale...")

    # --- Matrice di confusione per ogni modello ---
    for name, res in all_results.items():
        if config["visualization"]["graph"]:
            plot_confusion_matrix(
                y_true, res["y_pred"], [label.capitalize() for label in labels],
                title=f"Matrice di Confusione - {name}",
                config=config,
                filename=f"cm_{_safe_name(name)}.png",
            )

    # --- Curve ROC per modelli con probabilita' ---
    for name, res in all_results.items():
        if res.get("y_proba") is not None:
            if config["visualization"]["graph"]:
                plot_roc_curves(
                    y_true, res["y_proba"], [label.capitalize() for label in labels],
                    title=f"Curve ROC - {name}",
                    config=config,
                    filename=f"roc_{_safe_name(name)}.png",
                )

    # --- Grafico di confronto metriche ---
    if config["visualization"]["graph"]:
        _plot_metrics_comparison(comparison, config)
        _plot_metric_groups_comparison(comparison, config)

    # --- Visualizzazione iperspazio / separabilita' inter/intra classi ---
    if config["visualization"]["graph"]:
        _plot_hypercube_separability(data, all_results, config)

    return comparison


def _plot_metrics_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre di confronto delle metriche selezionate, un file per metrica."""
    import matplotlib.pyplot as plt

    metrics = config["visualization"].get("comparison_metrics", ["accuracy"])
    metrics = [m for m in metrics if m in comparison.columns]
    if not metrics:
        logger.warning("Nessuna metrica valida trovata per il confronto grafico.")
        return

    fig_dir = config["paths"]["figures_dir"]
    subdir = os.path.join(fig_dir, "model_comparison")
    os.makedirs(subdir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    models = comparison["Modello"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models))) # type: ignore

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        values = comparison[metric].tolist()
        bars = ax.barh(models[::-1], values[::-1], color=colors)
        for bar, value in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}", va="center", fontsize=9)
        ax.set_xlabel(metric.replace("_", " ").capitalize())
        ax.set_title(f"Confronto {metric.replace('_', ' ').capitalize()} tra Modelli")
        ax.set_xlim(0, 1.05)
        fig.tight_layout()

        filename = f"model_{metric}_comparison.png"
        fig.savefig(os.path.join(subdir, filename))
        plt.close(fig)
        logger.info(f"  Salvato {filename} in {str(subdir).replace(os.sep, '/')}")


def _plot_metric_groups_comparison(comparison: pd.DataFrame, config: dict):
    """Grafico a barre raggruppate per modello con multiple metriche."""
    import matplotlib.pyplot as plt

    metrics = config["visualization"].get("comparison_group_metrics", ["accuracy"])
    metrics = [m for m in metrics if m in comparison.columns]
    if not metrics:
        logger.warning("Nessuna metrica valida trovata per il confronto a gruppi.")
        return

    fig_dir = config["paths"]["figures_dir"]
    subdir = os.path.join(fig_dir, "model_comparison")
    os.makedirs(subdir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    models = comparison["Modello"].tolist()
    n_models = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.5), 6), dpi=dpi)

    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics)) # type: ignore
    for i, metric in enumerate(metrics):
        values = comparison[metric].tolist()
        ax.bar(x + i * width, values, width, label=metric.replace("_", " ").capitalize(), color=colors[i])

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Valore")
    ax.set_title("Confronto metriche per modello")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metrica")
    for i, metric in enumerate(metrics):
        values = comparison[metric].tolist()
        for j, value in enumerate(values):
            ax.text(x[j] + i * width, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    filename = "model_comparison_groups.png"
    fig.savefig(os.path.join(subdir, filename))
    plt.close(fig)
    logger.info(f"  Salvato {filename} in {str(subdir).replace(os.sep, '/')}")


def _ellipsoid_mesh(center, cov, n=20, scale=1.5):
    """Return mesh (x,y,z) per ellissoide centrato su center con matrice cov"""
    if cov.shape != (3, 3):
        cov = np.eye(3)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-9)
    except Exception:
        eigvals = np.ones(3)
        eigvecs = np.eye(3)

    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # scalatura e rotazione
    radii = np.sqrt(eigvals) * scale
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    points = points * radii
    points = (points @ eigvecs.T) + center

    return (
        points[:, 0].reshape((n, n)),
        points[:, 1].reshape((n, n)),
        points[:, 2].reshape((n, n)),
    )


def _plot_hypercube_separability(data: dict, all_results: dict, config: dict):
    """Visualizza i modelli nello spazio 3D ridotto con PCA e separabilita'."""
    import matplotlib.pyplot as plt

    fig_dir = config["paths"]["figures_dir"]
    hypercube_dir = os.path.join(fig_dir, "hypercube_separability")
    os.makedirs(hypercube_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    results_dir = config["paths"]["results_dir"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    labels = get_particle_labels(data["label_encoder"])

    # Sottocampione rappresentativo fino a 500 sample per leggibilita'.
    n_samples = len(y_test)
    max_samples = min(500, n_samples)
    rng = np.random.default_rng(42)
    if n_samples > max_samples:
        sample_idx = rng.choice(n_samples, size=max_samples, replace=False)
        X_vis = X_test[sample_idx]
        y_vis = y_test[sample_idx]
    else:
        sample_idx = np.arange(n_samples)
        X_vis = X_test
        y_vis = y_test

    # Seleziona le 3 feature piu' descrittive tramite F-test ANOVA su tutti i test set
    if X_vis.shape[1] >= 3:
        f_vals, _ = f_classif(X_vis, y_vis)
        top3_idx = np.argsort(f_vals)[::-1][:3]
    else:
        top3_idx = np.arange(X_vis.shape[1])
    selected_features = [data["feature_names"][i] for i in top3_idx]
    X3 = X_vis[:, top3_idx]  # (n_sample, 3)

    # range identico per uniforme 'ipercubo'
    mins = X3.min(axis=0)
    maxs = X3.max(axis=0)
    span = max(maxs - mins)
    center = (mins + maxs) / 2
    cube_min = center - span / 2
    cube_max = center + span / 2

    report_lines = []

    fixed_elev = 30
    fixed_azim = 45

    for name, res in all_results.items():
        y_pred = np.asarray(res["y_pred"])
        y_pred_vis = y_pred[sample_idx]

        fig = plt.figure(figsize=(10, 8), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        intra_dists = []
        centroids = []
        class_idxs = np.unique(y_vis)

        for c in class_idxs:
            idx = np.where(y_pred_vis == c)[0]
            if len(idx) == 0:
                continue

            points = X3[idx]
            centroid = points.mean(axis=0)
            centroids.append(centroid)

            intra = 0.0
            if len(points) > 1:
                intra = np.mean(pairwise_distances(points))
            intra_dists.append(intra)

            ax.scatter3D(
                points[:, 0], points[:, 1], points[:, 2], # type: ignore
                label=f"{labels[c].capitalize()} (pred)",
                s=20, alpha=0.6, edgecolor="w",
            )

        if len(centroids) > 1:
            inter = np.mean(pairwise_distances(np.vstack(centroids)))
        else:
            inter = 0.0

        avg_intra = float(np.mean(intra_dists)) if intra_dists else 0.0

        # Mostra anche i punti veri come piccole croci per riferimento
        for c in class_idxs:
            true_idx = np.where(y_vis == c)[0]
            true_pts = X3[true_idx]
            ax.scatter3D(
                true_pts[:, 0], true_pts[:, 1], true_pts[:, 2], # type: ignore
                color="black", marker="x", s=8, alpha=0.25,
                label=None,
            )

        ax.set_title(
            f"Ipercube 3D su feature selezionate: {name.title()}\n"
            f"Inter-class dist={inter:.3f}, Intra-class dist={avg_intra:.3f}",
            fontsize=11,
        )
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1] if len(selected_features) > 1 else 'Feature-2')
        ax.set_zlabel(selected_features[2] if len(selected_features) > 2 else 'Feature-3')

        ax.set_xlim(cube_min[0], cube_max[0])
        ax.set_ylim(cube_min[1], cube_max[1])
        ax.set_zlim(cube_min[2], cube_max[2])

        ax.view_init(elev=fixed_elev, azim=fixed_azim)
        ax.legend(fontsize=9, loc="upper left")
        fig.tight_layout()

        filename = f"hypercube_separability_{_safe_name(name)}.png"
        fig.savefig(os.path.join(hypercube_dir, filename))
        plt.close(fig)

        # Plotly interattivo 3D (rotazione mouse/zoom) con ellissoidi per classe
        html_filename = os.path.join(hypercube_dir, f"hypercube_separability_{_safe_name(name)}.html")

        # CSS condiviso per HTML (no inline)
        css_name = f"hypercube_separability_{_safe_name(name)}.css"
        css_path = os.path.join(hypercube_dir, css_name)
        if not os.path.exists(css_path):
            with open(css_path, "w", encoding="utf-8") as css:
                css.write("body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }")
                css.write("\n")
                css.write("h1 { color: #333; font-size: 1.4rem; }")
                css.write("\n")
                css.write(".msg { margin: 16px 0; color: #333; }")
                css.write("\n")
                css.write("img.hypercube_img { max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15); }")
                css.write("\n")
                css.write(".plotly-wrapper { width: 100%; height: 100%; overflow: auto; }")

        if go is not None:
            figly = go.Figure()
            palette = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown"]
            for c in class_idxs:
                idx = np.where(y_pred_vis == c)[0]
                if len(idx) == 0:
                    continue
                pts = X3[idx]
                color = palette[int(c) % len(palette)]
                figly.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(size=3, opacity=0.7, color=color),
                    name=f"{labels[c].capitalize()} (pred)",
                ))

                cov = np.cov(pts, rowvar=False) if len(pts) > 2 else np.eye(3)
                cent = pts.mean(axis=0)
                ellx, elly, ellz = _ellipsoid_mesh(cent, cov, n=22, scale=1.5)
                figly.add_trace(go.Mesh3d(
                    x=ellx.flatten(), y=elly.flatten(), z=ellz.flatten(),
                    opacity=0.2,
                    color=color,
                    name=f"{labels[c].capitalize()} ellissoide",
                    showscale=False,
                ))

            figly.update_layout(
                scene=dict(
                    xaxis_title=selected_features[0],
                    yaxis_title=selected_features[1] if len(selected_features) > 1 else 'Feature-2',
                    zaxis_title=selected_features[2] if len(selected_features) > 2 else 'Feature-3',
                    aspectmode='cube',
                ),
                title=f"Ipercube 3D Interattivo: {_safe_name(name)}",
                legend=dict(font=dict(size=10)),
            )

            plotly_div = figly.to_html(full_html=False, include_plotlyjs='cdn')
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="it">\n')
                f.write('<head>\n')
                f.write('  <meta charset="utf-8" />\n')
                f.write('  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n')
                f.write(f'  <title>Ipercube 3D Interattivo - {_safe_name(name)}</title>\n')
                f.write(f'  <link rel="stylesheet" href="{css_name}" />\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write('<h1>Ipercube 3D Interattivo</h1>\n')
                f.write('<div class="plotly-wrapper">\n')
                f.write(plotly_div)
                f.write('</div>\n')
                f.write('</body>\n')
                f.write('</html>\n')
            logger.info(f"  Salvato interattivo plotly in {str(html_filename).replace(os.sep, '/')}")
        else:
            img_name = os.path.basename(filename)
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="it">\n')
                f.write('<head>\n')
                f.write('  <meta charset="utf-8" />\n')
                f.write('  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n')
                f.write(f'  <title>Ipercube 3D - {_safe_name(name)}</title>\n')
                f.write(f'  <link rel="stylesheet" href="{css_name}" />\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write('<h1>Ipercube 3D (solo immagine)</h1>\n')
                f.write('<p class="msg">Installare plotly per la versione interattiva: python -m pip install plotly</p>\n')
                f.write(f'<img class="hypercube_img" src="{img_name}" alt="hypercube" />\n')
                f.write('</body>\n')
                f.write('</html>\n')
            logger.info(f"  Plotly non disponibile. Creato fallback HTML statico: {str(html_filename).replace(os.sep, '/')}")

        report_lines.append(
            f"{name}: inter={inter:.4f}, intra={avg_intra:.4f}, n_samples={len(y_test)}"
        )

    summary_path = os.path.join(results_dir, "hypercube_separability.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Ipercube Separabilita\n")
        f.write("=" * 55 + "\n")
        f.write("\n".join(report_lines))

    logger.info(f"  Salvato hypercube separability plot/report in {str(fig_dir).replace(os.sep, '/')}")


def _safe_name(name: str) -> str:
    """Converte un nome modello in un nome file sicuro."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
