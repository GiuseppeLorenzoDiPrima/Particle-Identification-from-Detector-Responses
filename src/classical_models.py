"""
Modelli di ML classici per Particle Identification.

Confronto sistematico tra:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost

Tutti i modelli vengono valutati con cross-validation stratificata
e poi addestrati sul train completo per la valutazione finale su test.
"""

import logging
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def _build_models(config: dict) -> dict:
    """Costruisce tutti i modelli dalla configurazione."""
    cfg = config["classical_models"]

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=cfg["logistic_regression"]["max_iter"],
            solver="lbfgs",
            class_weight="balanced",
        ),
        "K-NN": KNeighborsClassifier(
            n_neighbors=cfg["knn"]["n_neighbors"],
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=cfg["decision_tree"]["max_depth"],
            class_weight="balanced",
            random_state=config["dataset"]["random_state"],
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=cfg["random_forest"]["n_estimators"],
            max_depth=cfg["random_forest"]["max_depth"],
            class_weight="balanced",
            n_jobs=cfg["random_forest"]["n_jobs"],
            random_state=config["dataset"]["random_state"],
        ),
        "XGBoost": XGBClassifier(
            n_estimators=cfg["xgboost"]["n_estimators"],
            max_depth=cfg["xgboost"]["max_depth"],
            learning_rate=cfg["xgboost"]["learning_rate"],
            n_jobs=cfg["xgboost"]["n_jobs"],
            random_state=config["dataset"]["random_state"],
            eval_metric="mlogloss",
        ),
    }
    return models


def run_cross_validation(X_train, y_train, models: dict, config: dict) -> dict:
    """
    Cross-validation stratificata per tutti i modelli.

    Returns:
        Dict {nome_modello: {"cv_mean": float, "cv_std": float}}
    """
    cv_cfg = config["cross_validation"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_folds"],
        shuffle=True,
        random_state=config["dataset"]["random_state"],
    )

    cv_results = {}
    for name, model in models.items():
        logger.info(f"  Cross-validation: {name}...")
        t0 = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        elapsed = time.time() - t0
        cv_results[name] = {
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "cv_time": elapsed,
        }
        logger.info(
            f"    {name}: {scores.mean():.4f} +/- {scores.std():.4f} "
            f"({elapsed:.1f}s)"
        )

    return cv_results


def train_and_evaluate(data: dict, config: dict) -> dict:
    """
    Pipeline completa: cross-validation + training finale + test.

    Returns:
        Dict {nome_modello: {
            "model": fitted model,
            "cv_mean", "cv_std",
            "test_accuracy",
            "y_pred", "y_proba",
            "train_time",
            "feature_importance" (se disponibile)
        }}
    """
    logger.info("=" * 50)
    logger.info("FASE 3: Modelli di ML classici")
    logger.info("=" * 50)

    models = _build_models(config)

    # Cross-validation
    logger.info("Cross-validation stratificata...")
    cv_results = run_cross_validation(data["X_train"], data["y_train"], models, config)

    # Training finale e valutazione su test
    results = {}
    for name, model in models.items():
        logger.info(f"  Training finale: {name}...")
        t0 = time.time()
        model.fit(data["X_train"], data["y_train"])
        train_time = time.time() - t0

        y_pred = model.predict(data["X_test"])
        test_acc = accuracy_score(data["y_test"], y_pred)

        # Probabilita' (per ROC)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(data["X_test"])

        # Feature importance
        feat_imp = None
        if hasattr(model, "feature_importances_"):
            feat_imp = dict(zip(data["feature_names"], model.feature_importances_))
        elif hasattr(model, "coef_"):
            # Per LR multiclasse, media dei valori assoluti dei coefficienti
            feat_imp = dict(
                zip(data["feature_names"], np.abs(model.coef_).mean(axis=0))
            )

        results[name] = {
            "model": model,
            "model_name": name,
            "cv_mean": cv_results[name]["cv_mean"],
            "cv_std": cv_results[name]["cv_std"],
            "test_accuracy": test_acc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "train_time": train_time,
            "feature_importance": feat_imp,
        }

        logger.info(
            f"    {name}: test_acc={test_acc:.4f}, "
            f"cv={cv_results[name]['cv_mean']:.4f}, "
            f"train_time={train_time:.1f}s"
        )

    return results


def plot_feature_importance(results: dict, feature_names: list, config: dict):
    """Grafico della feature importance per i modelli che la supportano."""
    import matplotlib.pyplot as plt
    import os

    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    models_with_fi = {
        name: res["feature_importance"]
        for name, res in results.items()
        if res["feature_importance"] is not None
    }

    if not models_with_fi:
        return

    n = len(models_with_fi)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, (name, fi) in zip(axes, models_with_fi.items()):
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_fi]
        values = [x[1] for x in sorted_fi]
        ax.barh(names[::-1], values[::-1])
        ax.set_title(f"Feature Importance\n{name}", fontsize=11)
        ax.set_xlabel("Importanza")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"))
    plt.close(fig)
    logger.info("Salvato feature_importance.png")
