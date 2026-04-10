"""
Modelli di ML classici per Particle Identification.

Confronto tra modelli:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost

Tutti i modelli vengono valutati con cross-validation stratificata a 
n fold sul training set. I modelli vengono successivamente addestrati
sul train completo ed infine valutati sul test set.
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
from xgboost import XGBClassifier # type: ignore

logger = logging.getLogger(__name__)


def _build_models(config: dict) -> dict:
    """Crea solo i modelli impostati nel file di configurazione."""
    cfg = config["classical_models"]
    models = {}

    if cfg.get("logistic_regression", {}).get("enabled", False):
        models["Logistic Regression"] = LogisticRegression(
            max_iter=cfg["logistic_regression"]["max_iter"],
            solver=cfg["logistic_regression"]["solver"],
            class_weight=cfg["logistic_regression"]["class_weight"],
        )

    if cfg.get("knn", {}).get("enabled", False):
        models["K-NN"] = KNeighborsClassifier(
            n_neighbors=cfg["knn"]["n_neighbors"],
        )

    if cfg.get("decision_tree", {}).get("enabled", False):
        models["Decision Tree"] = DecisionTreeClassifier(
            max_depth=cfg["decision_tree"]["max_depth"],
            class_weight=cfg["decision_tree"]["class_weight"],
            random_state=config["dataset"]["random_state"],
        )

    if cfg.get("random_forest", {}).get("enabled", False):
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=cfg["random_forest"]["n_estimators"],
            max_depth=cfg["random_forest"]["max_depth"],
            class_weight=cfg["random_forest"]["class_weight"],
            n_jobs=cfg["random_forest"]["n_jobs"],
            random_state=config["dataset"]["random_state"],
        )

    if cfg.get("xgboost", {}).get("enabled", False):
        models["XGBoost"] = XGBClassifier(
            n_estimators=cfg["xgboost"]["n_estimators"],
            max_depth=cfg["xgboost"]["max_depth"],
            learning_rate=cfg["xgboost"]["learning_rate"],
            n_jobs=cfg["xgboost"]["n_jobs"],
            random_state=config["dataset"]["random_state"],
            eval_metric=cfg["xgboost"]["eval_metric"],
        )

    if not models:
        logger.warning("Nessun modello classico abilitato in configurazione.")

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
        shuffle=cv_cfg["shuffle"],
        random_state=config["dataset"]["random_state"],
    )

    cv_results = {}
    for name, model in models.items():
        logger.info(f"  Cross-validation for model: {name}...")
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
    Pipeline completa: cross-validation + training + test.

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
    logger.info("=" * 55)
    logger.info("FASE 3: Modelli di ML classici")
    logger.info("=" * 55)

    models = _build_models(config)
    cv_results = {}

    # Cross-validation
    if config["cross_validation"].get("enabled", False):
        logger.info("Cross-validation stratificata (%d-fold)...", config["cross_validation"]["n_folds"])
        cv_results = run_cross_validation(data["X_train"], data["y_train"], models, config)

    # Training e valutazione su test
    results = {}
    print()
    logger.info("Training e valutazione su test set...")
    for name, model in models.items():
        logger.info(f"  Training model: {name}...")
        t0 = time.time()
        model.fit(data["X_train"], data["y_train"])
        train_time = time.time() - t0

        y_pred = model.predict(data["X_test"])
        test_acc = accuracy_score(data["y_test"], y_pred)

        # Probabilità (per ROC)
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
            "cv_mean": cv_results[name]["cv_mean"] if name in cv_results else None,
            "cv_std": cv_results[name]["cv_std"] if name in cv_results else None,
            "test_accuracy": test_acc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "train_time": train_time,
            "feature_importance": feat_imp,
        }

        cv_str = (
            f"cv={cv_results[name]['cv_mean']:.4f}, "
            if name in cv_results else "cv=N/A, "
        )
        logger.info(
            f"    {name}: test_acc={test_acc:.4f}, "
            f"{cv_str}"
            f"train_time={train_time:.1f}s"
        )

    return results


def plot_feature_importance(results: dict, feature_names: list, config: dict):
    """Grafico della feature importance per i modelli che la supportano."""
    from src.visualization import plot_feature_importance as _plot
    print()
    logger.info("Creazione grafico feature importance...")
    _plot(results, feature_names, config)
