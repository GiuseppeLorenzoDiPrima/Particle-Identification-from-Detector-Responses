"""
================================================
Particle Identification from Detector Responses
================================================

Progetto di ML e DL applicato alla fisica delle particelle:
Identificazione di 4 specie di particelle a partire dalle
risposte simulate con metodo Monte Carlo di 6 rivelatori, 
confrontando metodo a tagli tradizionale, modelli classici di 
Machine Learning e Deep Learning. Condotto anche uno studio
di interpretabilità dei modelli di ML e DL implementati.

Entry point del progetto. Esegue la pipeline completa:

  Fase 1: Caricamento dati e visualizzazione esplorativa
  Fase 2: Baseline a tagli (cuts-based PID)
  Fase 3: Modelli di ML classici (LR, KNN, DT, RF e XGBoost)
  Fase 4: Deep Learning (MLP con PyTorch)
  Fase 5: Interpretabilità (SHAP) e Uncertainty (MC Dropout)
  Fase 6: Valutazione finale e confronto

Uso:
    python main.py                  # Pipeline completa
    python main.py --phase 1        # Solo una fase
    python main.py --phases 1 2 3   # Fasi selezionate
    python main.py --config my.yaml # Configurazione custom
    python main.py --quick          # Run veloce (100k campioni)
"""

import argparse
import logging
import matplotlib
matplotlib.use("Agg")
import os
import sys
import time

from data_classes.data_loader import load_config, load_and_preprocess
from plot.visualization import (
    setup_publication_style,
    plot_bethe_bloch,
    plot_feature_distributions,
    plot_class_distribution,
    plot_correlation_matrix,
)
from models.baseline import run_baseline
from models.classical_models import train_and_evaluate, plot_feature_importance
from models.deep_learning import train_mlp, plot_training_history
from utils.evaluation import generate_full_report
from utils.interpretability import run_shap_analysis
from utils.uncertainty import run_uncertainty_analysis


def setup_logging(config: dict):
    """
    Configura il logging su console e file.

    Console: mostra solo i messaggi del progetto (src.* e main),
             formato compatto senza timestamp.
    File:    registra tutto (incluso shap, matplotlib, ecc...)
             con timestamp completo.
    """
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # --- File handler: tutto, verbose ---
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "run.log"), mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    root.addHandler(file_handler)

    # --- Console handler: solo progetto, compatto ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Filtra: mostra solo logger del progetto (main, __main__, data_classes.*, models.*, utils.*, plot.*)
    _PROJECT_PREFIXES = ("data_classes.", "models.", "utils.", "plot.")

    class ProjectFilter(logging.Filter):
        def filter(self, record):
            return record.name in ("main", "__main__") or any(
                record.name.startswith(p) for p in _PROJECT_PREFIXES
            )

    console_handler.addFilter(ProjectFilter())
    root.addHandler(console_handler)

    # Silenzia logger rumorosi sulla console (restano nel file)
    for noisy in ("shap", "matplotlib", "PIL", "numba", "xgboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Particle Identification - ML Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Percorso al file di configurazione YAML",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Esegui solo una fase specifica (1-6)",
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", default=None,
        help="Esegui solo le fasi specificate (es. --phases 1 2 3)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run veloce con max 100k campioni",
    )
    return parser.parse_args()


def should_run(phase: int, args) -> bool:
    """Determina se una fase deve essere eseguita."""
    if args.phase is not None:
        return phase == args.phase
    if args.phases is not None:
        return phase in args.phases
    return True


def main():
    args = parse_args()
    config = load_config(args.config)

    # Quick mode: limita i campioni
    if args.quick:
        config["dataset"]["max_samples"] = 100_000
        config["deep_learning"]["epochs"] = 20
        config["interpretability"]["shap_samples"] = 200
        config["uncertainty"]["mc_dropout_iterations"] = 20

    setup_logging(config)
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("  PARTICLE IDENTIFICATION FROM DETECTOR RESPONSES")
    logger.info("  Pipeline di Machine Learning per Fisica delle Particelle")
    logger.info("=" * 60)

    t_start = time.time()
    all_results = {}

    # ================================================================
    # FASE 1: Caricamento dati e visualizzazione esplorativa
    # ================================================================
    if should_run(1, args):
        print()
        logger.info("=" * 55)
        logger.info("FASE 1: Caricamento dati e visualizzazione")
        logger.info("=" * 55)

        data = load_and_preprocess(config)
        setup_publication_style(config)

        if config["visualization"].get("graph", True):
            print()
            logger.info("Generazione visualizzazioni esplorative...")
            plot_bethe_bloch(data, config)
            plot_feature_distributions(data, config)
            plot_class_distribution(data, config)
            plot_correlation_matrix(data, config)
        print()
        logger.info("Fase 1 completata.")
    else:
        # Carica comunque i dati se servono per le fasi successive
        data = load_and_preprocess(config)

    # ================================================================
    # FASE 2: Baseline a tagli
    # ================================================================
    if should_run(2, args):
        baseline_results = run_baseline(data, config)
        if baseline_results:
            all_results["Cuts-Based PID"] = baseline_results
        print()
        logger.info("Fase 2 completata.")

    # ================================================================
    # FASE 3: Modelli di ML classici
    # ================================================================
    if should_run(3, args):
        classical_results = train_and_evaluate(data, config)
        all_results.update(classical_results)
        plot_feature_importance(classical_results, data["feature_names"], config)
        print()
        logger.info("Fase 3 completata.")

    # ================================================================
    # FASE 4: Deep Learning (MLP)
    # ================================================================
    mlp_results = {}
    if should_run(4, args):
        mlp_results = train_mlp(data, config)
        all_results["MLP (PyTorch)"] = mlp_results
        if config["visualization"].get("graph", True):
            plot_training_history(mlp_results["history"], config)
        print()
        logger.info("Fase 4 completata.")

    # ================================================================
    # FASE 5: Interpretabilità e Uncertainty
    # ================================================================
    if should_run(5, args):
        # 5a: Analisi SHAP
        run_shap_analysis(all_results, data, config)

        # 5b: Uncertainty (MC Dropout) - solo se la MLP è stata addestrata
        if mlp_results:
            run_uncertainty_analysis(mlp_results, data, config)
        print()
        logger.info("Fase 5 completata.")

    # ================================================================
    # FASE 6: Valutazione finale e confronto
    # ================================================================
    if should_run(6, args) and all_results:
        comparison = generate_full_report(all_results, data, config)
        print()
        logger.info(f"{'=' * 55}")
        logger.info("TABELLA FINALE RIEPILOGATIVA DEI RISULTATI OTTENUTI")
        logger.info(f"{'=' * 55}")
        print()
        logger.info(f"{comparison.to_string(index=False)}")
        print()
        logger.info(f"Fase 6 completata.")
        logger.info(f"{'=' * 80}")
        logger.info("Addestramento e valutazione completati. Tutti i risultati sono stati salvati.")
        logger.info(f"{'=' * 80}")

    # ================================================================
    elapsed = time.time() - t_start
    print()
    logger.info(f"Pipeline completata in {elapsed:.1f} secondi.")
    logger.info(f"Output salvati in: {config['paths']['output_dir']}/")


if __name__ == "__main__":
    main()
