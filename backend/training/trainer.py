import threading
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ExperimentStopped(Exception):
    """Raised when a user safely stops an active NAS experiment."""


def _build_optimizer(model: nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    key = (name or "adam").lower()
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    if key == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _check_control(pause_event: Optional[threading.Event], stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise ExperimentStopped("Experiment stopped by user.")
    while pause_event is not None and pause_event.is_set():
        if stop_event is not None and stop_event.is_set():
            raise ExperimentStopped("Experiment stopped while paused.")
        time.sleep(0.15)


def train_candidate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    arch_config: Dict[str, Any],
    max_epochs: int = 3,
    cpu_threads: int = 4,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
    on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[nn.Module, List[Dict[str, Any]], float, float]:
    """
    Train a candidate PyTorch neural network on CPU and return:
      (trained_model, training_history, total_training_time_s, peak_memory_mb)
    """
    torch.set_num_threads(max(1, min(16, int(cpu_threads))))
    device = torch.device("cpu")
    model.to(device)

    optimizer = _build_optimizer(
        model,
        arch_config.get("optimizer", "adam"),
        float(arch_config.get("learning_rate", 0.001)),
    )
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    start_total = time.perf_counter()

    tracemalloc_started_here = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        tracemalloc_started_here = True
    else:
        tracemalloc.reset_peak()

    try:
        for epoch in range(1, max_epochs + 1):
            _check_control(pause_event, stop_event)
            epoch_start = time.perf_counter()

            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (xb, yb) in enumerate(train_loader):
                if batch_idx % 8 == 0:
                    _check_control(pause_event, stop_event)
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bs = yb.size(0)
                train_loss_sum += float(loss.item()) * bs
                preds = logits.argmax(dim=1)
                train_correct += int((preds == yb).sum().item())
                train_total += bs

            # Validation pass
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    bs = yb.size(0)
                    val_loss_sum += float(loss.item()) * bs
                    preds = logits.argmax(dim=1)
                    val_correct += int((preds == yb).sum().item())
                    val_total += bs

            epoch_time = time.perf_counter() - epoch_start
            train_loss = train_loss_sum / max(1, train_total)
            train_acc = train_correct / max(1, train_total)
            val_loss = val_loss_sum / max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            record = {
                "epoch": epoch,
                "total_epochs": max_epochs,
                "train_loss": round(train_loss, 5),
                "train_accuracy": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_accuracy": round(val_acc, 5),
                "epoch_time_s": round(epoch_time, 4),
            }
            history.append(record)
            if on_epoch_end:
                on_epoch_end(record)

        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if tracemalloc_started_here:
            tracemalloc.stop()

    total_time = time.perf_counter() - start_total

    # Combine Python traced peak allocation + PyTorch parameter & gradient + optimizer state footprint
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Gradients + Adam/AdamW 2 moment states (~4x param memory) + traced heap peak
    torch_state_bytes = param_bytes * 4
    peak_memory_mb = round((peak_bytes + torch_state_bytes) / (1024.0 * 1024.0), 3)

    return model, history, round(total_time, 4), peak_memory_mb
