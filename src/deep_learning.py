"""
Modulo Deep Learning per Particle Identification.

Implementa un Multi-Layer Perceptron (MLP) con PyTorch, con:
- Early stopping
- MC Dropout per uncertainty quantification
- Supporto per SHAP interpretability
"""

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class ParticleMLP(nn.Module):
    """
    MLP per classificazione di particelle.

    Architettura configurabile via config.yaml:
    - hidden_layers: lista di dimensioni dei layer nascosti
    - dropout: tasso di dropout (usato anche per MC Dropout)
    """

    def __init__(self, input_dim: int, n_classes: int, hidden_layers: list[int],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def _prepare_loaders(data: dict, config: dict):
    """Prepara i DataLoader PyTorch dai dati numpy."""
    cfg = config["deep_learning"]
    batch_size = cfg["batch_size"]

    def to_loader(X, y, shuffle=False):
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader = to_loader(data["X_val"], data["y_val"])
    test_loader = to_loader(data["X_test"], data["y_test"])

    return train_loader, val_loader, test_loader


def train_mlp(data: dict, config: dict) -> dict:
    """
    Addestra l'MLP con early stopping.

    Returns:
        Dict con modello, predizioni, metriche e storia di training.
    """
    logger.info("=" * 50)
    logger.info("FASE 4: Deep Learning (MLP)")
    logger.info("=" * 50)

    cfg = config["deep_learning"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    n_features = data["X_train"].shape[1]
    n_classes = len(np.unique(data["y_train"]))

    # Modello
    model = ParticleMLP(
        input_dim=n_features,
        n_classes=n_classes,
        hidden_layers=cfg["hidden_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    logger.info(f"Architettura MLP:\n{model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametri trainabili: {n_params:,}")

    # Ottimizzatore e loss (con pesi per classi sbilanciate)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    class_counts = np.bincount(data["y_train"])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    logger.info(f"Class weights: {dict(zip(range(n_classes), map(float, class_weights.round(4))))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # DataLoader
    train_loader, val_loader, test_loader = _prepare_loaders(data, config)

    # Training loop con early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        # --- Training ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                val_preds.append(outputs.argmax(dim=1).cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(
            np.concatenate(val_targets), np.concatenate(val_preds)
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{cfg['epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                logger.info(f"  Early stopping a epoch {epoch+1}")
                break

    train_time = time.time() - t0

    # Ripristina il miglior modello
    model.load_state_dict(best_state)
    model.eval()

    # Valutazione finale su test
    test_preds = []
    test_probas = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probas = torch.softmax(outputs, dim=1)
            test_preds.append(outputs.argmax(dim=1).cpu().numpy())
            test_probas.append(probas.cpu().numpy())

    y_pred = np.concatenate(test_preds)
    y_proba = np.concatenate(test_probas)
    test_acc = accuracy_score(data["y_test"], y_pred)

    logger.info(f"  MLP test accuracy: {test_acc:.4f} (train_time={train_time:.1f}s)")

    # Salva il modello
    model_path = os.path.join(config["paths"]["models_dir"], "mlp_best.pt")
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Modello salvato in {model_path}")

    return {
        "model": model,
        "model_name": "MLP (PyTorch)",
        "test_accuracy": test_acc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "train_time": train_time,
        "history": history,
        "device": device,
    }


def plot_training_history(history: dict, config: dict):
    """Grafico loss e accuracy durante il training."""
    import matplotlib.pyplot as plt

    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    dpi = config["visualization"]["dpi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training e Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "mlp_training_history.png"))
    plt.close(fig)
    logger.info("Salvato mlp_training_history.png")
