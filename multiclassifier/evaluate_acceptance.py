#!/usr/bin/env python3
"""Acceptance curves for segmented multiclassifier training outputs.

The Model 2 segmented SLURM jobs save predictions and labels under each run's
``csv`` directory. This script reads those CSVs directly, so it can evaluate
both QKeras and LGN runs without reloading the trained models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_CLASSES = 3
HIGH_PT_CLASS = 0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/scratch/gpfs/IOJALVO/mb7126/SmartPixels/giuData/data/ds8_only/dec6_ds8_quant")
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "OUTPUT_ROOT",
        str(
            SCRIPT_DIR
            / "results/SLURM"
            / os.environ.get("RUN_TAG", f"model2_segmented_top2_{datetime.now().strftime('%Y%m%d')}")
        ),
    )
)

EDGES = np.array(
    [
        -5,
        -4.5,
        -4,
        -3.5,
        -3,
        -2.5,
        -2,
        -1.8,
        -1.6,
        -1.4,
        -1.2,
        -1,
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        -0.15,
        0.15,
        0.2,
        0.4,
        0.6,
        0.8,
        1,
        1.2,
        1.4,
        1.6,
        1.8,
        2,
        2.5,
        3,
        3.5,
        4,
        4.5,
        5,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    backend: str
    model: str
    model_prefix: str
    local_id: Optional[int]
    predictions_file: Path
    truth_file: Path
    pt_file: Path


@dataclass
class GroupData:
    label: str
    runs: List[RunRecord]
    y_true: List[np.ndarray]
    y_pred: List[np.ndarray]
    pt: List[np.ndarray]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_vector(path: Path, preferred_columns: Iterable[str] = ()) -> np.ndarray:
    df = pd.read_csv(path)
    for column in preferred_columns:
        if column in df.columns:
            return df[column].to_numpy()
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy()
    return df.to_numpy().reshape(-1)


def find_single_csv(csv_dir: Path, suffix: str) -> Path:
    matches = sorted(csv_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one *{suffix} under {csv_dir}, found {len(matches)}")
    return matches[0]


def infer_local_id(model_prefix: str) -> Optional[int]:
    match = re.search(r"ds8l(\d+)_", model_prefix)
    return int(match.group(1)) if match else None


def infer_model_name(run_dir: Path, metadata: Dict) -> str:
    if metadata.get("model"):
        return str(metadata["model"])
    if run_dir.name.startswith("local_") and run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def infer_backend(run_dir: Path, metadata: Dict, model_prefix: str) -> str:
    if metadata.get("backend"):
        return str(metadata["backend"])
    lowered = " ".join(str(part).lower() for part in run_dir.parts)
    if "lgn" in lowered or "model2lgn" in model_prefix.lower():
        return "lgn"
    if "qkeras" in lowered or "qkeras" in model_prefix.lower():
        return "qkeras"
    return "unknown"


def resolve_pt_file(metadata: Dict, run_dir: Path, data_dir: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if metadata.get("pt_file"):
        candidates.append(Path(metadata["pt_file"]))
    if data_dir is not None and metadata.get("local_id") is not None:
        candidates.append(data_dir / f"TestSetTruePTLocal{int(metadata['local_id'])}.csv")
    if metadata.get("pt_file"):
        candidates.append(run_dir / Path(metadata["pt_file"]).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates) or "none"
    raise FileNotFoundError(f"No true-pT file found for {run_dir}. Tried: {joined}")


def build_run_record(
    run_dir: Path,
    metadata: Dict,
    data_dir: Optional[Path],
    *,
    predictions_file: Optional[Path] = None,
) -> Optional[RunRecord]:
    csv_dir = run_dir / "csv"
    if predictions_file is not None:
        csv_dir = predictions_file.parent
    if not csv_dir.is_dir():
        return None

    try:
        model_prefix = str(metadata.get("model_prefix") or "")
        if predictions_file is not None:
            model_prefix = predictions_file.name.removesuffix("_predictionsFiles.csv")
            truth_file = csv_dir / f"{model_prefix}_true.csv"
            if not truth_file.exists():
                truth_file = find_single_csv(csv_dir, "_true.csv")
        elif model_prefix:
            predictions_file = csv_dir / f"{model_prefix}_predictionsFiles.csv"
            truth_file = csv_dir / f"{model_prefix}_true.csv"
        else:
            predictions_file = find_single_csv(csv_dir, "_predictionsFiles.csv")
            truth_file = find_single_csv(csv_dir, "_true.csv")
            model_prefix = predictions_file.name.removesuffix("_predictionsFiles.csv")
    except FileNotFoundError as exc:
        print(f"[WARN] Skipping {run_dir}: {exc}")
        return None

    if not predictions_file.exists() or not truth_file.exists():
        print(f"[WARN] Skipping {run_dir}: missing prediction/true CSV.")
        return None

    local_id = metadata.get("local_id")
    if local_id is None:
        local_id = infer_local_id(model_prefix)
        metadata = {**metadata, "local_id": local_id}

    return RunRecord(
        run_dir=run_dir,
        backend=infer_backend(run_dir, metadata, model_prefix),
        model=infer_model_name(run_dir, metadata),
        model_prefix=model_prefix,
        local_id=local_id,
        predictions_file=predictions_file,
        truth_file=truth_file,
        pt_file=resolve_pt_file(metadata, run_dir, data_dir),
    )


def discover_runs(results_root: Path, data_dir: Optional[Path]) -> List[RunRecord]:
    records: List[RunRecord] = []
    seen_predictions = set()

    for metadata_file in sorted(results_root.rglob("metadata.json")):
        run_dir = metadata_file.parent
        record = build_run_record(run_dir, read_json(metadata_file), data_dir)
        if record is None:
            continue
        records.append(record)
        seen_predictions.add(record.predictions_file.resolve())

    for predictions_file in sorted(results_root.rglob("*_predictionsFiles.csv")):
        if predictions_file.resolve() in seen_predictions:
            continue
        run_dir = predictions_file.parent.parent
        metadata_file = run_dir / "metadata.json"
        metadata = read_json(metadata_file) if metadata_file.exists() else {}
        try:
            record = build_run_record(
                run_dir,
                metadata,
                data_dir,
                predictions_file=predictions_file,
            )
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping {run_dir}: {exc}")
            continue
        if record is None:
            continue
        records.append(record)
        seen_predictions.add(record.predictions_file.resolve())

    if not records:
        metadata_count = sum(1 for _ in results_root.rglob("metadata.json"))
        prediction_count = sum(1 for _ in results_root.rglob("*_predictionsFiles.csv"))
        truth_count = sum(1 for _ in results_root.rglob("*_true.csv"))
        print(
            "[WARN] Discovery found "
            f"{metadata_count} metadata.json file(s), "
            f"{prediction_count} prediction CSV(s), and "
            f"{truth_count} truth CSV(s) under {results_root}."
        )
    return records


def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        if mask.any():
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.0) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return float(center - half), float(center + half)


def acceptance_rows(
    label: str,
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
    edges: np.ndarray,
    high_pt_class: int,
    z: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i in range(len(edges) - 1):
        low = float(edges[i])
        high = float(edges[i + 1])
        selected = (pt_true_signed >= low) & (pt_true_signed < high)
        n = int(selected.sum())
        accepted = int((preds[selected] == high_pt_class).sum()) if n else 0
        acceptance = float(accepted / n) if n else float("nan")
        lo, hi = wilson_interval(accepted, n, z=z)
        rows.append(
            {
                "model": label,
                "bin_low": low,
                "bin_high": high,
                "bin_center": 0.5 * (low + high),
                "xerr": 0.5 * (high - low),
                "n": n,
                "accepted": accepted,
                "acceptance": acceptance,
                "err_low": acceptance - lo if n else float("nan"),
                "err_high": hi - acceptance if n else float("nan"),
            }
        )
    return rows


def load_group(record: RunRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pred = read_vector(record.predictions_file, ("predict",)).astype(np.int64, copy=False).reshape(-1)
    y_true = read_vector(record.truth_file, ("true", "ptLabel")).astype(np.int64, copy=False).reshape(-1)
    pt = read_vector(record.pt_file, ("pt",)).astype(np.float64, copy=False).reshape(-1)
    n = min(len(y_pred), len(y_true), len(pt))
    if len({len(y_pred), len(y_true), len(pt)}) != 1:
        print(
            f"[WARN] Length mismatch for {record.run_dir}: "
            f"pred={len(y_pred)}, true={len(y_true)}, pt={len(pt)}. Using first {n} rows."
        )
    return y_true[:n], y_pred[:n], pt[:n]


def group_label(record: RunRecord, group_by: str) -> str:
    if group_by == "run":
        return record.model_prefix
    return record.model


def make_groups(records: List[RunRecord], group_by: str) -> List[GroupData]:
    groups: Dict[str, GroupData] = {}
    for record in records:
        label = group_label(record, group_by)
        if label not in groups:
            groups[label] = GroupData(label=label, runs=[], y_true=[], y_pred=[], pt=[])
        y_true, y_pred, pt = load_group(record)
        groups[label].runs.append(record)
        groups[label].y_true.append(y_true)
        groups[label].y_pred.append(y_pred)
        groups[label].pt.append(pt)
    return [groups[label] for label in sorted(groups)]


def plot_acceptance(
    outdir: Path,
    groups: List[GroupData],
    rows_by_label: Dict[str, List[Dict[str, float]]],
    balanced_by_label: Dict[str, float],
    high_pt_class: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in groups:
        rows = rows_by_label[group.label]
        centers = np.array([row["bin_center"] for row in rows], dtype=np.float64)
        xerr = np.array([row["xerr"] for row in rows], dtype=np.float64)
        acc = np.array([row["acceptance"] for row in rows], dtype=np.float64)
        err_low = np.array([row["err_low"] for row in rows], dtype=np.float64)
        err_high = np.array([row["err_high"] for row in rows], dtype=np.float64)
        label = f"{group.label}  (bal={balanced_by_label[group.label]:.3f})"
        ax.errorbar(
            centers,
            acc,
            xerr=xerr,
            yerr=[np.maximum(err_low, 0.0), np.maximum(err_high, 0.0)],
            marker="o",
            linestyle="-",
            capsize=2.5,
            elinewidth=1.0,
            label=label,
        )

    ax.set_xlabel(r"true signed $p_T$ [GeV]")
    ax.set_ylabel(f"acceptance as high-pT (class {high_pt_class})")
    ax.set_title("Acceptance vs true signed pT")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax.grid(True, alpha=0.35)
    fig.subplots_adjust(right=0.68)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.85, borderaxespad=0.0)

    for ext in ("png", "pdf"):
        out = outdir / f"acceptance.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        "-r",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root produced by run_model2_segmented.slurm.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing TestSetTruePTLocal*.csv, used if metadata pt_file is unavailable.",
    )
    parser.add_argument("--outdir", "-o", type=Path, default=SCRIPT_DIR / "results/acceptance_model2_segmented")
    parser.add_argument("--group-by", choices=("model", "run"), default="model")
    parser.add_argument("--backend", choices=("all", "qkeras", "lgn"), default="all")
    parser.add_argument("--high-pt-class", type=int, default=HIGH_PT_CLASS)
    parser.add_argument("--z", type=float, default=1.0, help="Wilson interval z (1.0 ~ 68%%, 1.96 ~ 95%%).")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV summaries and skip acceptance image generation.")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results root does not exist: {results_root}")

    data_dir = args.data_dir.resolve() if args.data_dir is not None else None
    if data_dir is not None and not data_dir.exists():
        print(f"[WARN] Data dir does not exist: {data_dir}. Will rely on metadata pt_file paths.")
        data_dir = None

    records = discover_runs(results_root, data_dir)
    if args.backend != "all":
        records = [record for record in records if record.backend == args.backend]
    if not records:
        backend_note = "" if args.backend == "all" else f" for backend={args.backend}"
        raise SystemExit(f"No completed run metadata with prediction CSVs found{backend_note} under {results_root}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    groups = make_groups(records, args.group_by)
    print(f"Found {len(records)} runs in {len(groups)} {args.group_by} group(s).")

    all_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    rows_by_label: Dict[str, List[Dict[str, float]]] = {}
    balanced_by_label: Dict[str, float] = {}

    for group in groups:
        y_true = np.concatenate(group.y_true)
        y_pred = np.concatenate(group.y_pred)
        pt = np.concatenate(group.pt)
        balanced = balanced_accuracy_np(y_true, y_pred)
        balanced_by_label[group.label] = balanced
        rows = acceptance_rows(group.label, pt, y_pred, EDGES, args.high_pt_class, args.z)
        rows_by_label[group.label] = rows
        all_rows.extend(rows)
        summary_rows.append(
            {
                "model": group.label,
                "runs": len(group.runs),
                "samples": len(y_true),
                "balanced_accuracy": balanced,
            }
        )
        print(f"{group.label}: balanced_accuracy={balanced:.6f} runs={len(group.runs)} samples={len(y_true)}")

    pd.DataFrame(summary_rows).to_csv(args.outdir / "balanced_accuracy.csv", index=False)
    pd.DataFrame(all_rows).to_csv(args.outdir / "acceptance_bins.csv", index=False)
    print(f"Wrote {args.outdir / 'balanced_accuracy.csv'}")
    print(f"Wrote {args.outdir / 'acceptance_bins.csv'}")
    if not args.no_plot:
        plot_acceptance(args.outdir, groups, rows_by_label, balanced_by_label, args.high_pt_class)


if __name__ == "__main__":
    main()
