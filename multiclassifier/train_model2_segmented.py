#!/usr/bin/env python3
"""Train Model 2 QKeras and LGN classifiers on notebook-style local CSV splits.

This script intentionally bypasses the top-level SmartPixels YAML/dataloader path.
It follows multiclassifier/train.ipynb data handling:

* QuantizedInput{Train,Test}SetLocal{N}.csv
* {Train,Test}SetLabelLocal{N}.csv
* TestSetTruePTLocal{N}.csv
* optional padded columns 14, 15, 16
* sparse integer labels with cross entropy from logits
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd


N_CLASSES = 3
LOCAL_IDS = tuple(range(12))
DEFAULT_DATA_DIR = Path("/scratch/gpfs/IOJALVO/mb7126/SmartPixels/giuData/data/ds8_only/dec6_ds8_quant")


@dataclass(frozen=True)
class QKerasSpec:
    name: str
    suffix: str
    hidden_units: int
    weight_bits: int
    activation_bits: int


@dataclass(frozen=True)
class LGNSpec:
    name: str
    size_label: str
    hidden_dims: Tuple[int, ...]
    tau: float
    n_bits: int = 100


QKERAS_SPECS: Tuple[QKerasSpec, ...] = (
    QKerasSpec("qkeras-model-2-w5a10-adc2a", "w5a10", 128, 5, 10),
    QKerasSpec("qkeras-model-2-w4a8-adc2a", "w4a8", 128, 4, 8),
)


LGN_SIZE_DIMS: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    ("s04_p434M", "lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M", (10400, 10400, 10400, 6900, 6900, 6900, 5202)),
    ("s05_currentWide_p577M", "lgn-dense-2-dense-100_128_-model2lgnFull", (12000, 12000, 12000, 8000, 8000, 8000, 6000)),
)
LGN_TAUS = (20, 40)
LGN_SPECS: Tuple[LGNSpec, ...] = tuple(
    LGNSpec(name=f"{prefix}_{tau}", size_label=size_label, hidden_dims=hidden_dims, tau=float(tau))
    for size_label, prefix, hidden_dims in LGN_SIZE_DIMS
    for tau in LGN_TAUS
)

QKERAS_BY_NAME = {spec.name: spec for spec in QKERAS_SPECS}
LGN_BY_NAME = {spec.name: spec for spec in LGN_SPECS}
ALL_MODEL_NAMES = tuple(QKERAS_BY_NAME) + tuple(LGN_BY_NAME)
STOP_REQUESTED = False


def install_signal_handlers() -> None:
    def request_stop(signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"[WARN] Received signal {signum}; stopping after the current batch.")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_tf_device(device: str) -> None:
    import tensorflow as tf

    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
    elif device == "cuda" and not tf.config.list_physical_devices("GPU"):
        print("[WARN] --device cuda requested, but TensorFlow does not see a GPU.")


def torch_device(device: str):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested, but PyTorch does not see a GPU. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def local_split_paths(data_dir: Path, local_id: int) -> Dict[str, Path]:
    return {
        "train_x": require_file(data_dir / f"QuantizedInputTrainSetLocal{local_id}.csv"),
        "train_y": require_file(data_dir / f"TrainSetLabelLocal{local_id}.csv"),
        "test_x": require_file(data_dir / f"QuantizedInputTestSetLocal{local_id}.csv"),
        "test_y": require_file(data_dir / f"TestSetLabelLocal{local_id}.csv"),
        "test_pt": require_file(data_dir / f"TestSetTruePTLocal{local_id}.csv"),
    }


def apply_notebook_padding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "14" not in df.columns:
        df["14"] = 0
    df["15"] = 0
    df["16"] = 0
    return df


def label_frame_to_vector(df: pd.DataFrame) -> np.ndarray:
    if "ptLabel" in df.columns:
        values = df["ptLabel"].to_numpy()
    elif df.shape[1] == 1:
        values = df.iloc[:, 0].to_numpy()
    else:
        values = df.to_numpy().argmax(axis=1)
    return values.astype(np.int64, copy=False).reshape(-1)


def load_local_split(
    data_dir: Path,
    local_id: int,
    *,
    pad_like_notebook: bool,
    scale: bool,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, List[str]]:
    paths = local_split_paths(data_dir, local_id)
    x_train_df = pd.read_csv(paths["train_x"])
    x_test_df = pd.read_csv(paths["test_x"])
    y_train_df = pd.read_csv(paths["train_y"])
    y_test_df = pd.read_csv(paths["test_y"])

    if pad_like_notebook:
        x_train_df = apply_notebook_padding(x_train_df)
        x_test_df = apply_notebook_padding(x_test_df)

    if list(x_train_df.columns) != list(x_test_df.columns):
        raise ValueError(
            "Train/test feature columns differ after preprocessing: "
            f"{list(x_train_df.columns)} != {list(x_test_df.columns)}"
        )

    if max_train_samples is not None:
        x_train_df = x_train_df.iloc[:max_train_samples]
        y_train_df = y_train_df.iloc[:max_train_samples]
    if max_test_samples is not None:
        x_test_df = x_test_df.iloc[:max_test_samples]
        y_test_df = y_test_df.iloc[:max_test_samples]

    x_train = x_train_df.to_numpy(dtype=np.float32, copy=True)
    x_test = x_test_df.to_numpy(dtype=np.float32, copy=True)
    y_train = label_frame_to_vector(y_train_df)
    y_test = label_frame_to_vector(y_test_df)

    if scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
        x_test = scaler.transform(x_test).astype(np.float32, copy=False)

    print(f"Local {local_id}: X_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Local {local_id}: X_test ={x_test.shape}, y_test ={y_test.shape}")
    print(f"Feature columns: {list(x_train_df.columns)}")

    return x_train, y_train, x_test, y_test, paths["test_pt"], list(x_train_df.columns)


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    validation_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if validation_split <= 0.0:
        return x, y, x[:0], y[:0]
    if not 0.0 < validation_split < 1.0:
        raise ValueError("--validation-split must be in [0, 1).")
    n_val = max(1, int(math.floor(len(x) * validation_split)))
    if n_val >= len(x):
        raise ValueError("Validation split leaves no training samples.")
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def model_prefix(local_id: int, model_kind: str, model_suffix: str, *, padded: bool, scaled: bool) -> str:
    padding = "padded_" if padded else ""
    scaling = "scaling_" if scaled else "noscaling_"
    if model_kind == "qkeras":
        return f"ds8l{local_id}_{padding}{scaling}qkeras_foldbatchnorm_d128{model_suffix}"
    return f"ds8l{local_id}_{padding}{scaling}{model_suffix}"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_results_row(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    try:
        import fcntl
    except ImportError:
        fcntl = None
    with path.open("a+", newline="", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if empty:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def true_pt_values(pt_file: Path, n_expected: int) -> np.ndarray:
    df = pd.read_csv(pt_file)
    column = "pt" if "pt" in df.columns else df.columns[0]
    values = df[column].to_numpy(dtype=np.float32, copy=False)
    if len(values) < n_expected:
        raise ValueError(f"{pt_file} has {len(values)} pt rows, expected at least {n_expected}.")
    return values[:n_expected]


def safe_fraction(mask: np.ndarray, selected: np.ndarray) -> float:
    denom = int(selected.sum())
    if denom == 0:
        return float("nan")
    return float(mask[selected].sum() / denom)


def physics_metrics(predicted: np.ndarray, pt_file: Path) -> Dict[str, float]:
    pt = true_pt_values(pt_file, len(predicted))
    high_pt_prediction = predicted == 0
    metrics = {
        "nt_gev02": safe_fraction(high_pt_prediction, np.abs(pt) > 0.2),
        "nt_gev05": safe_fraction(high_pt_prediction, np.abs(pt) > 0.5),
        "nt_gev10": safe_fraction(high_pt_prediction, np.abs(pt) > 1.0),
        "nt_gev20": safe_fraction(high_pt_prediction, np.abs(pt) > 2.0),
        "bkg_rej": safe_fraction(np.isin(predicted, (1, 2)), np.abs(pt) < 2.0),
    }
    return metrics


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    return {
        "labels": list(range(N_CLASSES)),
        "matrix": matrix.astype(int).tolist(),
    }


def count_np_model_params(model: Any) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def build_qkeras_model(input_dim: int, spec: QKerasSpec):
    import tensorflow as tf
    from qkeras import QActivation, QDense, quantized_bits

    try:
        from qkeras import QDenseBatchnorm
    except ImportError:
        QDenseBatchnorm = QDense

    inputs = tf.keras.layers.Input(shape=(input_dim,), name="input1")
    x = QDenseBatchnorm(
        spec.hidden_units,
        kernel_quantizer=quantized_bits(spec.weight_bits, 0, alpha=1),
        bias_quantizer=quantized_bits(spec.weight_bits, 0, alpha=1),
        name="dense1",
    )(inputs)
    x = QActivation(f"quantized_relu({spec.activation_bits},0)", name="relu1")(x)
    x = QDense(
        N_CLASSES,
        kernel_quantizer=quantized_bits(spec.weight_bits, 0, alpha=1),
        bias_quantizer=quantized_bits(spec.weight_bits, 0, alpha=1),
        name="dense2",
    )(x)
    outputs = tf.keras.layers.Activation("linear", name="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=spec.name.replace("-", "_"))


class StopOnSignalCallback:
    def __init__(self):
        import tensorflow as tf

        class _Callback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if STOP_REQUESTED:
                    self.model.stop_training = True

        self.callback = _Callback()


def train_qkeras(args: argparse.Namespace, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, List[str]]) -> Dict[str, Any]:
    import tensorflow as tf
    from qkeras.utils import model_save_quantized_weights

    x_train, y_train, x_test, y_test, pt_file, feature_columns = data
    spec = QKERAS_BY_NAME[args.model]

    configure_tf_device(args.device)
    tf.keras.utils.set_random_seed(args.seed)

    run_prefix = model_prefix(args.local_id, "qkeras", spec.suffix, padded=args.pad_like_notebook, scaled=args.scale)
    output_dir = args.output
    models_dir = output_dir / "models"
    csv_dir = output_dir / "csv"
    logs_dir = output_dir / "logs"
    images_dir = output_dir / "images"
    for path in (models_dir, csv_dir, logs_dir, images_dir):
        path.mkdir(parents=True, exist_ok=True)

    model = build_qkeras_model(x_train.shape[1], spec)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
    )
    model.summary()

    callbacks: List[Any] = [
        tf.keras.callbacks.CSVLogger(str(logs_dir / "training_log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir / "tensorboard")),
        StopOnSignalCallback().callback,
    ]
    if not args.disable_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
            )
        )

    class_weight = None
    if args.balance_classes:
        counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
        total = counts.sum()
        class_weight = {
            cls: float(total / (N_CLASSES * count))
            for cls, count in enumerate(counts)
            if count > 0
        }
        print(f"Class weights: {class_weight}")

    history = model.fit(
        x_train,
        y_train,
        callbacks=callbacks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        shuffle=True,
        verbose=args.verbose,
        class_weight=class_weight,
    )

    model_file = models_dir / f"{run_prefix}model.h5"
    q_weights_file = models_dir / f"{run_prefix}model_q_weights.h5"
    model_save_quantized_weights(model, str(q_weights_file))
    model.save(str(model_file))
    print(f"Save: {model_file}")
    print(f"Save: {q_weights_file}")

    save_json(logs_dir / "history.json", {k: [float(vv) for vv in v] for k, v in history.history.items()})

    score = model.evaluate(x_test, y_test, verbose=0)
    preds = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predicted = np.argmax(preds, axis=1).astype(np.int64, copy=False)

    pd.DataFrame(predicted).to_csv(csv_dir / f"{run_prefix}_predictionsFiles.csv", header=["predict"], index=False)
    pd.DataFrame(y_test).to_csv(csv_dir / f"{run_prefix}_true.csv", header=["true"], index=False)
    np.savetxt(csv_dir / "tb_input_features.dat", np.asarray(x_test, dtype=np.int32), fmt="%d")
    np.savetxt(csv_dir / "tb_output_predictions.dat", np.asarray(y_test, dtype=np.int32), fmt="%d")

    phys = physics_metrics(predicted, pt_file)
    metrics = {
        "loss": float(score[0]),
        "accuracy": float(score[1]),
        **phys,
    }
    save_json(csv_dir / f"{run_prefix}_confusion_matrix.json", confusion_matrix_dict(y_test, predicted))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "qkeras",
        "model": spec.name,
        "model_prefix": run_prefix,
        "local_id": args.local_id,
        "data_dir": str(args.data_dir),
        "pt_file": str(pt_file),
        "feature_columns": feature_columns,
        "input_dim": int(x_train.shape[1]),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "trainable_parameters": count_np_model_params(model),
        "qkeras_spec": asdict(spec),
        "training": training_metadata(args),
        "metrics": metrics,
    }
    save_json(output_dir / "metadata.json", metadata)
    save_json(output_dir / "test_metrics.json", metrics)
    append_results_row(args.results_file, result_row(metadata))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metadata


def import_torchlogix_layers():
    try:
        from torchlogix.layers import FixedBinarization, GroupSum, LogicDense
    except ImportError as exc:
        raise ImportError(
            "torchlogix is required for LGN training. Load the same environment used for "
            "the SmartPixels LGN runs before launching this script."
        ) from exc
    return FixedBinarization, GroupSum, LogicDense


class DenseOnlyLGNModel2Full:
    """Factory wrapper so the torchlogix import happens only for LGN runs."""

    def __init__(self, input_dim: int, spec: LGNSpec, device: str, thresholds: Any):
        import torch

        FixedBinarization, GroupSum, LogicDense = import_torchlogix_layers()

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_bits = spec.n_bits
                self.input_dim = input_dim
                self.bin = FixedBinarization(thresholds=thresholds)
                self.layers = torch.nn.ModuleList()
                prev_dim = input_dim * spec.n_bits
                param_kwargs = {"weight_init": "residual"}
                for hidden_dim in spec.hidden_dims:
                    self.layers.append(
                        LogicDense(
                            in_dim=prev_dim,
                            out_dim=hidden_dim,
                            device=device,
                            parametrization_kwargs=param_kwargs,
                        )
                    )
                    prev_dim = hidden_dim
                self.group_sum = GroupSum(N_CLASSES, tau=float(spec.tau), device=device)

            def forward(self, x):
                if x.shape != (x.shape[0], self.input_dim):
                    raise AssertionError(
                        f"Expected input shape (batch_size, {self.input_dim}), got {tuple(x.shape)}"
                    )
                z = self.bin(x)
                for layer in self.layers:
                    z = layer(z)
                return self.group_sum(z)

        self.model = _Model()


def derive_lgn_thresholds(x_train: np.ndarray, n_bits: int, threshold_samples: int):
    import torch
    from torchlogix.layers import Binarization

    sample = x_train[: min(len(x_train), threshold_samples)]
    tensor = torch.as_tensor(sample, dtype=torch.float32)
    return Binarization.get_initial_thresholds(
        tensor,
        num_bits=n_bits,
        one_per="feature",
        method="distributive",
    )


def class_weights_torch(y: np.ndarray, device: Any):
    import torch

    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    total = counts.sum()
    weights = np.ones(N_CLASSES, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (N_CLASSES * counts[nonzero])
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        if mask.any():
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def evaluate_torch_model(model: Any, loader: Any, criterion: Any, device: Any, *, train_mode: bool = False) -> Dict[str, float]:
    import torch

    was_training = model.training
    model.train(train_mode)
    total_loss = 0.0
    total_n = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * int(y_batch.numel())
            total_n += int(y_batch.numel())
            all_true.append(y_batch.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    model.train(was_training)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "balanced_accuracy": balanced_accuracy_np(y_true, y_pred),
    }


def predict_torch_model(model: Any, x: np.ndarray, batch_size: int, device: Any) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            logits = model(x_batch.to(device))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    model.train(was_training)
    return np.concatenate(preds).astype(np.int64, copy=False)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def make_summary_writer(path: Path):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("[WARN] No TensorBoard writer available.")
            return NullWriter()
        return SummaryWriter(log_dir=str(path))
    return SummaryWriter(logdir=str(path))


def save_torch_checkpoint(path: Path, model: Any, metadata: Dict[str, Any]) -> None:
    import torch

    state_model = getattr(model, "_orig_mod", model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "model2_segmented_lgn_state_dict",
            "metadata": metadata,
            "state_dict": state_model.state_dict(),
        },
        path,
    )


def train_lgn(args: argparse.Namespace, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, List[str]]) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    x_train_all, y_train_all, x_test, y_test, pt_file, feature_columns = data
    spec = LGN_BY_NAME[args.model]

    set_global_seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch_device(args.device)

    x_train, y_train, x_val, y_val = split_train_val(x_train_all, y_train_all, args.validation_split)
    if len(x_val) == 0:
        x_val, y_val = x_test, y_test

    thresholds = derive_lgn_thresholds(x_train, spec.n_bits, args.threshold_samples).to(device)
    wrapper = DenseOnlyLGNModel2Full(input_dim=x_train_all.shape[1], spec=spec, device=str(device), thresholds=thresholds)
    model = wrapper.model
    model.to(device)
    if args.compile_model:
        if hasattr(torch, "compile"):
            print("Compiling LGN model with torch.compile(dynamic=True)")
            model = torch.compile(model, dynamic=True)
        else:
            print("[WARN] --compile-model requested, but this PyTorch has no torch.compile.")

    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
    print(f"Number of trainable parameters: {n_params}")

    weights = class_weights_torch(y_train, device) if args.balance_classes else None
    if weights is not None:
        print(f"Class weights: {weights.detach().cpu().tolist()}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_prefix = model_prefix(args.local_id, "lgn", spec.name, padded=args.pad_like_notebook, scaled=args.scale)
    output_dir = args.output
    models_dir = output_dir / "models"
    csv_dir = output_dir / "csv"
    logs_dir = output_dir / "logs"
    for path in (models_dir, csv_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "lgn",
        "model": spec.name,
        "model_prefix": run_prefix,
        "local_id": args.local_id,
        "data_dir": str(args.data_dir),
        "pt_file": str(pt_file),
        "feature_columns": feature_columns,
        "input_dim": int(x_train_all.shape[1]),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "trainable_parameters": n_params,
        "lgn_spec": asdict(spec),
        "training": training_metadata(args),
    }

    writer = make_summary_writer(logs_dir / "tensorboard")
    best_val_loss = float("inf")
    patience_left = args.early_stopping_patience
    step = 0
    epoch = 0
    stop = False

    while not stop and not STOP_REQUESTED and (args.epochs == 0 or epoch < args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Training {spec.name} local {args.local_id} epoch {epoch}", unit="batch")
        for x_batch, y_batch in progress:
            if STOP_REQUESTED:
                stop = True
                break
            step += 1
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            if step % args.log_freq == 0:
                pred = logits.argmax(dim=1)
                acc = float((pred == y_batch).to(torch.float32).mean().item())
                progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
                writer.add_scalar("Loss/train_batch", float(loss.item()), step)
                writer.add_scalar("Accuracy/train_batch", acc, step)

            if args.eval_freq > 0 and step % args.eval_freq == 0:
                val_metrics = evaluate_torch_model(model, val_loader, criterion, device, train_mode=False)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"Validation/{name}", value, step)
                print(f"Step {step} validation: {json.dumps(val_metrics, sort_keys=True)}")
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_left = args.early_stopping_patience
                    save_torch_checkpoint(models_dir / "best_model.pth", model, {**base_metadata, "best_step": step, "best_val": val_metrics})
                else:
                    patience_left -= 1
                if not args.disable_early_stopping and patience_left <= 0:
                    print("Early stopping triggered.")
                    stop = True
                    break

            if args.max_steps is not None and step >= args.max_steps:
                print(f"Reached --max-steps={args.max_steps}.")
                stop = True
                break
        epoch += 1

    final_val_metrics = evaluate_torch_model(model, val_loader, criterion, device, train_mode=False)
    test_metrics_base = evaluate_torch_model(model, test_loader, criterion, device, train_mode=False)
    predicted = predict_torch_model(model, x_test, args.batch_size, device)
    pd.DataFrame(predicted).to_csv(csv_dir / f"{run_prefix}_predictionsFiles.csv", header=["predict"], index=False)
    pd.DataFrame(y_test).to_csv(csv_dir / f"{run_prefix}_true.csv", header=["true"], index=False)

    phys = physics_metrics(predicted, pt_file)
    metrics = {
        "loss": float(test_metrics_base["loss"]),
        "accuracy": float(test_metrics_base["accuracy"]),
        "balanced_accuracy": float(test_metrics_base["balanced_accuracy"]),
        **phys,
    }
    save_json(csv_dir / f"{run_prefix}_confusion_matrix.json", confusion_matrix_dict(y_test, predicted))

    metadata = {
        **base_metadata,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "steps": int(step),
        "epochs_completed": int(epoch),
        "stop_requested": bool(STOP_REQUESTED),
        "final_validation": final_val_metrics,
        "metrics": metrics,
    }
    save_torch_checkpoint(models_dir / "final_model.pth", model, metadata)
    save_json(output_dir / "metadata.json", metadata)
    save_json(output_dir / "test_metrics.json", metrics)
    append_results_row(args.results_file, result_row(metadata))
    writer.close()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metadata


def training_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "validation_split": args.validation_split,
        "early_stopping_patience": args.early_stopping_patience,
        "disable_early_stopping": args.disable_early_stopping,
        "balance_classes": args.balance_classes,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "eval_freq": args.eval_freq,
        "pad_like_notebook": args.pad_like_notebook,
        "scale": args.scale,
    }


def result_row(metadata: Dict[str, Any]) -> Dict[str, Any]:
    metrics = metadata["metrics"]
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "backend": metadata["backend"],
        "id": metadata["model_prefix"],
        "model": metadata["model"],
        "local_id": metadata["local_id"],
        "input_dim": metadata["input_dim"],
        "trainable_parameters": metadata["trainable_parameters"],
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "nt_gev02": metrics.get("nt_gev02"),
        "nt_gev05": metrics.get("nt_gev05"),
        "nt_gev10": metrics.get("nt_gev10"),
        "nt_gev20": metrics.get("nt_gev20"),
        "bkg_rej": metrics.get("bkg_rej"),
    }


def list_models() -> None:
    print("QKeras Model 2:")
    for spec in QKERAS_SPECS:
        print(f"  {spec.name}")
    print("LGN Model 2 full sweep:")
    for spec in LGN_SPECS:
        print(f"  {spec.name}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    model_safe = args.model.replace("/", "_")
    return args.output / model_safe / f"local_{args.local_id}"


def main(args: argparse.Namespace) -> None:
    if args.list_models:
        list_models()
        return
    install_signal_handlers()
    if args.model not in ALL_MODEL_NAMES:
        raise ValueError(f"Unknown model {args.model!r}. Use --list-models.")
    if args.local_id not in LOCAL_IDS:
        raise ValueError(f"--local-id must be one of {LOCAL_IDS}.")

    args.data_dir = args.data_dir.resolve()
    args.output = resolve_output_dir(args).resolve()
    if args.results_file is not None:
        args.results_file = args.results_file.resolve()

    set_global_seed(args.seed)
    data = load_local_split(
        args.data_dir,
        args.local_id,
        pad_like_notebook=args.pad_like_notebook,
        scale=args.scale,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    if args.model in QKERAS_BY_NAME:
        train_qkeras(args, data)
    elif args.model in LGN_BY_NAME:
        train_lgn(args, data)
    else:
        raise AssertionError(args.model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="Print supported model names and exit.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing dec6_ds8_quant CSVs.")
    parser.add_argument("--local-id", type=int, default=6, help="Local y segment id, 0 through 11.")
    parser.add_argument("--model", "-m", choices=ALL_MODEL_NAMES, default="qkeras-model-2-w4a8-adc2a")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/model2_segmented"))
    parser.add_argument("--results-file", type=Path, default=None, help="Optional aggregate CSV. Uses file locking.")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--batch-size", "-bs", type=int, default=1024)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=150, help="For LGN, 0 means train until stopped/max-steps/walltime.")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--scale", action="store_true", help="Apply StandardScaler, matching notebook's optional scale=True branch.")
    parser.add_argument("--no-pad-like-notebook", dest="pad_like_notebook", action="store_false")
    parser.set_defaults(pad_like_notebook=True)

    parser.add_argument("--weight-decay", type=float, default=0.0, help="LGN AdamW weight decay.")
    parser.add_argument("--compile-model", action="store_true", help="Compile LGN model with torch.compile.")
    parser.add_argument("--max-steps", type=int, default=None, help="LGN maximum optimizer steps.")
    parser.add_argument("--eval-freq", type=int, default=200, help="LGN validation frequency in optimizer steps.")
    parser.add_argument("--log-freq", type=int, default=100, help="LGN train logging frequency in optimizer steps.")
    parser.add_argument("--threshold-samples", type=int, default=15 * 1024, help="LGN samples used to derive binarization thresholds.")
    parser.add_argument("--num-workers", type=int, default=0, help="LGN DataLoader workers.")

    parser.add_argument("--max-train-samples", type=int, default=None, help="Debug option for short local smoke tests.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Debug option for short local smoke tests.")
    parser.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2))
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
