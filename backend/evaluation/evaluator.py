import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from backend.config import SAVED_MODELS_DIR
from backend.models.dynamic_net import count_parameters


def benchmark_inference_latency(
    model: nn.Module,
    input_dim: int,
    warmup_runs: int = 10,
    timed_runs: int = 60,
) -> float:
    """
    Measure single-sample CPU forward-pass latency in milliseconds using high-resolution perf_counter.
    """
    model.eval()
    sample = torch.randn(1, input_dim, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample)
        start = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(sample)
        elapsed = time.perf_counter() - start
    return round((elapsed / timed_runs) * 1000.0, 4)


def evaluate_candidate(
    model: nn.Module,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    exp_id: str,
    arch_id: str,
    arch_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform real validation evaluation, sklearn classification metrics, CPU latency benchmark,
    exact parameter count, and real checkpoint serialization + file size measurement.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    loss_sum = 0.0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            bs = yb.size(0)
            loss_sum += float(loss.item()) * bs
            total += bs

    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])

    accuracy = float(accuracy_score(y_true, y_pred)) if total > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0)) if total > 0 else 0.0
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0)) if total > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if total > 0 else 0.0
    val_loss = float(loss_sum / max(1, total))

    latency_ms = benchmark_inference_latency(model, input_dim)
    param_count = int(count_parameters(model))

    weights_path: Path = SAVED_MODELS_DIR / f"{exp_id}_{arch_id}.pt"
    checkpoint = {
        "architecture_id": arch_id,
        "experiment_id": exp_id,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "arch_config": arch_config,
        "metrics": {
            "accuracy": round(accuracy, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1_score": round(f1, 5),
            "val_loss": round(val_loss, 5),
            "inference_latency_ms": latency_ms,
            "parameter_count": param_count,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, str(weights_path))
    model_size_kb = round(os.path.getsize(weights_path) / 1024.0, 2)

    return {
        "accuracy": round(accuracy, 5),
        "precision_score": round(precision, 5),
        "recall_score": round(recall, 5),
        "f1_score": round(f1, 5),
        "val_loss": round(val_loss, 5),
        "inference_latency_ms": latency_ms,
        "parameter_count": param_count,
        "model_size_kb": model_size_kb,
        "model_weights_path": str(weights_path),
    }
