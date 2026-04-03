"""
Modulo per il download, caricamento e preprocessing del dataset
Particle Identification from Detector Responses.

Il dataset contiene risposte di 6 rivelatori per 4 specie di particelle
prodotte in scattering inelastico elettrone-protone.
"""

import os
import zipfile
import subprocess
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


# Mappa delle particelle: PDG ID -> nome fisico
# -11 = positrone (e+), 211 = pione (pi+), 321 = kaone (K+), 2212 = protone (p)
PARTICLE_NAMES = {
    -11: "elettrone",
    211: "pione",
    321: "kaone",
    2212: "protone",
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Carica la configurazione dal file YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(config: dict) -> str:
    """
    Scarica il dataset da Kaggle usando l'API ufficiale.
    Richiede che ~/.kaggle/kaggle.json sia configurato.

    Returns:
        Percorso al file CSV scaricato.
    """
    data_dir = config["paths"]["data_dir"]
    slug = config["dataset"]["kaggle_slug"]
    csv_name = config["dataset"]["filename"]
    csv_path = os.path.join(data_dir, csv_name)

    if os.path.exists(csv_path):
        logger.info(f"Dataset gia' presente in {csv_path}, skip download.")
        return csv_path

    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Download dataset da Kaggle: {slug}...")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", data_dir],
        check=True,
    )

    # Estrai lo zip scaricato
    zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
    for zf in zip_files:
        zip_path = os.path.join(data_dir, zf)
        logger.info(f"Estrazione {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        os.remove(zip_path)

    if not os.path.exists(csv_path):
        # Cerca qualsiasi CSV nella directory
        csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if csvs:
            actual_path = os.path.join(data_dir, csvs[0])
            logger.warning(
                f"File atteso '{csv_name}' non trovato. "
                f"Uso '{csvs[0]}' invece."
            )
            return actual_path
        raise FileNotFoundError(
            f"Nessun CSV trovato in {data_dir} dopo il download."
        )

    return csv_path


def load_and_preprocess(config: dict) -> dict:
    """
    Carica il dataset, applica preprocessing e split train/val/test.

    Returns:
        Dizionario con chiavi:
        - X_train, X_val, X_test: feature scalate (numpy array)
        - y_train, y_val, y_test: label encoded (numpy array)
        - X_train_raw, X_val_raw, X_test_raw: feature NON scalate
        - feature_names: lista nomi feature
        - label_encoder: LabelEncoder fitted
        - scaler: StandardScaler fitted
        - df: DataFrame completo originale
    """
    csv_path = download_dataset(config)
    logger.info(f"Caricamento dataset da {csv_path}...")

    df = pd.read_csv(csv_path)
    logger.info(f"Dataset caricato: {df.shape[0]} eventi, {df.shape[1]} colonne")
    logger.info(f"Colonne: {list(df.columns)}")

    # Subsample opzionale
    max_samples = config["dataset"].get("max_samples")
    if max_samples and max_samples < len(df):
        logger.info(f"Subsample a {max_samples} eventi...")
        df = df.sample(n=max_samples, random_state=config["dataset"]["random_state"])
        df = df.reset_index(drop=True)

    # Rimuovi righe con valori mancanti
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Rimossi {n_dropped} eventi con valori mancanti.")

    # Identifica feature e target
    target_col = config["features"]["target"]
    feature_names = [c for c in df.columns if c != target_col]
    
    # Mappa PDG ID → nome particella
    df[target_col] = df[target_col].map(PARTICLE_NAMES)
    # Controllo sicurezza
    if df[target_col].isnull().any():
        missing = df[df[target_col].isnull()]
        raise ValueError(f"Valori PDG non mappati trovati:\n{missing}")
    logger.info(f"Feature: {feature_names}")
    logger.info(f"Target: {target_col}")
    class_counts = df[target_col].value_counts().sort_index()
    logger.info("Distribuzione classi:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls.capitalize():10s}: {count}")

    X = df[feature_names].values
    y_raw = df[target_col].values

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    class_mapping = {cls.capitalize(): int(idx) for cls, idx in zip(le.classes_, le.transform(le.classes_))}
    logger.info(f"Classi codificate: {class_mapping}")

    # Split: train+val / test
    rs = config["dataset"]["random_state"]
    test_size = config["dataset"]["test_size"]
    val_size = config["dataset"]["val_size"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )

    # Split: train / val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction, random_state=rs, stratify=y_trainval
    )

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # Salva copie non scalate per visualizzazione e baseline
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    # Standardizzazione (fit solo su train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "feature_names": feature_names,
        "label_encoder": le,
        "scaler": scaler,
        "df": df,
    }
