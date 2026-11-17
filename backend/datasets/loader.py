import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from backend.config import CUSTOM_DATASETS_DIR, DATASETS_DIR
from backend.database.db import db_manager


def _stratified_subset(
    X: torch.Tensor, y: torch.Tensor, max_samples: int, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y
    indices = np.arange(len(y))
    y_np = y.numpy()
    try:
        sub_idx, _ = train_test_split(
            indices, train_size=max_samples, stratify=y_np, random_state=seed
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(indices, size=max_samples, replace=False)
    return X[sub_idx], y[sub_idx]


def _load_torchvision_dataset(
    name: str, max_train_samples: int, max_val_samples: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, int, int, Dict[str, Any]]:
    from torchvision import datasets

    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    try:
        train_ds = cls(root=str(DATASETS_DIR), train=True, download=True)
        val_ds = cls(root=str(DATASETS_DIR), train=False, download=True)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load or download {name.upper()} dataset ({exc}). "
            "Check your internet connection for first-time download, or select the built-in 'Digits (Offline)' dataset or upload a CSV."
        ) from exc

    # Subset on compact uint8 tensors BEFORE converting to float32 to minimize peak RAM
    X_tr_u8, y_train = _stratified_subset(train_ds.data, train_ds.targets.long(), max_train_samples, seed=42)
    X_va_u8, y_val = _stratified_subset(val_ds.data, val_ds.targets.long(), max_val_samples, seed=42)

    X_train = X_tr_u8.float().view(len(X_tr_u8), -1) / 255.0
    X_val = X_va_u8.float().view(len(X_va_u8), -1) / 255.0

    mean, std = (0.1307, 0.3081) if name == "mnist" else (0.2860, 0.3530)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=max(batch_size, 128), shuffle=False)

    meta = {
        "dataset": name,
        "display_name": "MNIST" if name == "mnist" else "Fashion-MNIST",
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "input_dim": 784,
        "num_classes": 10,
    }
    return train_loader, val_loader, 784, 10, meta


def _load_digits_dataset(
    max_train_samples: int, max_val_samples: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, int, int, Dict[str, Any]]:
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    X_train = torch.from_numpy(X_tr)
    y_train = torch.from_numpy(y_tr)
    X_val = torch.from_numpy(X_va)
    y_val = torch.from_numpy(y_va)

    X_train, y_train = _stratified_subset(X_train, y_train, max_train_samples, seed=42)
    X_val, y_val = _stratified_subset(X_val, y_val, max_val_samples, seed=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=max(batch_size, 128), shuffle=False)

    meta = {
        "dataset": "digits",
        "display_name": "UCI Handwritten Digits (8x8)",
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "input_dim": 64,
        "num_classes": 10,
    }
    return train_loader, val_loader, 64, 10, meta


def _load_csv_dataset(
    ds_id: str, max_train_samples: int, max_val_samples: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, int, int, Dict[str, Any]]:
    rec = db_manager.get_custom_dataset(ds_id)
    if not rec:
        raise ValueError(f"Custom dataset '{ds_id}' not found.")
    df = pd.read_csv(rec["file_path"])
    target_col = rec["target_column"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from CSV.")

    y_raw = df[target_col].values
    X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X_df.values.astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    num_classes = int(len(le.classes_))
    input_dim = int(X.shape[1])

    try:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    X_train, y_train = _stratified_subset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), max_train_samples, seed=42)
    X_val, y_val = _stratified_subset(torch.from_numpy(X_va), torch.from_numpy(y_va), max_val_samples, seed=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=max(batch_size, 128), shuffle=False)

    meta = {
        "dataset": f"csv:{ds_id}",
        "display_name": rec["name"],
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "input_dim": input_dim,
        "num_classes": num_classes,
    }
    return train_loader, val_loader, input_dim, num_classes, meta


def load_dataset(
    dataset_name: str,
    batch_size: int = 64,
    max_train_samples: int = 4000,
    max_val_samples: int = 1000,
) -> Tuple[DataLoader, DataLoader, int, int, Dict[str, Any]]:
    key = (dataset_name or "mnist").strip()
    if key.lower() in ("mnist", "fashion_mnist"):
        return _load_torchvision_dataset(key.lower(), max_train_samples, max_val_samples, batch_size)
    if key.lower() == "digits":
        return _load_digits_dataset(max_train_samples, max_val_samples, batch_size)
    if key.startswith("csv:"):
        return _load_csv_dataset(key.split(":", 1)[1], max_train_samples, max_val_samples, batch_size)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def register_csv_dataset(name: str, filename: str, content: bytes, target_column: str = "") -> Dict[str, Any]:
    ds_id = uuid.uuid4().hex[:10]
    safe_name = Path(filename).name.replace(" ", "_")
    dest = CUSTOM_DATASETS_DIR / f"{ds_id}_{safe_name}"
    dest.write_bytes(content)

    try:
        df = pd.read_csv(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise ValueError(f"Invalid CSV file: {exc}") from exc

    if df.shape[0] < 20 or df.shape[1] < 2:
        dest.unlink(missing_ok=True)
        raise ValueError("CSV must contain at least 20 rows and 2 columns (features + target).")

    target_col = target_column.strip() if target_column and target_column.strip() in df.columns else df.columns[-1]
    num_classes = int(df[target_col].nunique())
    if num_classes < 2 or num_classes > 100:
        dest.unlink(missing_ok=True)
        raise ValueError(f"Target column '{target_col}' must have between 2 and 100 distinct classes (found {num_classes}).")

    rec = {
        "id": ds_id,
        "name": name.strip() or Path(filename).stem,
        "filename": safe_name,
        "file_path": str(dest),
        "num_samples": int(df.shape[0]),
        "num_features": int(df.shape[1] - 1),
        "num_classes": num_classes,
        "target_column": str(target_col),
    }
    return db_manager.save_custom_dataset(rec)


def list_available_datasets() -> List[Dict[str, Any]]:
    mnist_cached = (DATASETS_DIR / "MNIST").exists()
    fashion_cached = (DATASETS_DIR / "FashionMNIST").exists()
    builtin = [
        {
            "id": "mnist",
            "name": "MNIST Handwritten Digits",
            "description": "28x28 grayscale handwritten digits (784 features, 10 classes). Default NAS benchmark.",
            "input_dim": 784,
            "num_classes": 10,
            "total_train": 60000,
            "total_val": 10000,
            "cached_locally": mnist_cached,
            "type": "builtin",
        },
        {
            "id": "fashion_mnist",
            "name": "Fashion-MNIST",
            "description": "28x28 grayscale clothing categories (784 features, 10 classes). Harder visual benchmark.",
            "input_dim": 784,
            "num_classes": 10,
            "total_train": 60000,
            "total_val": 10000,
            "cached_locally": fashion_cached,
            "type": "builtin",
        },
        {
            "id": "digits",
            "name": "UCI Handwritten Digits (8x8 Offline)",
            "description": "64-feature handwritten digits from scikit-learn (1,797 samples, 10 classes). Instant & 100% offline.",
            "input_dim": 64,
            "num_classes": 10,
            "total_train": 1347,
            "total_val": 450,
            "cached_locally": True,
            "type": "builtin",
        },
    ]
    customs = [
        {
            "id": f"csv:{c['id']}",
            "name": f"{c['name']} (CSV)",
            "description": f"Uploaded CSV ({c['filename']}), target: {c['target_column']}",
            "input_dim": c["num_features"],
            "num_classes": c["num_classes"],
            "total_train": int(c["num_samples"] * 0.8),
            "total_val": int(c["num_samples"] * 0.2),
            "cached_locally": True,
            "type": "custom",
        }
        for c in db_manager.list_custom_datasets()
    ]
    return builtin + customs
