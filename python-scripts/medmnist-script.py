#!/usr/bin/env python3
"""
Run CV quantum model pipeline for BreastMNIST (single-run training + evaluation).

This script is focused on the continuous-variable model flow present in the
notebook `breastmnist-pennylane-cv.ipynb`. It reuses functions from ./modules.

Usage examples:
  # Basic run with defaults
  python run_cv_breastmnist.py

  # Change PCA components and CV qumodes (and optional depth)
  python run_cv_breastmnist.py --n_components 6 --n_qumodes 6 --depth 3 --epochs 40

  # Load an existing PCA and scaler
  python run_cv_breastmnist.py --pca-path path/to/pca.pkl --scaler-path path/to/scaler.pkl

  # Run a simple grid from shell (example)
  for comps in 2 4 6; do
    for q in 2 4 6; do
      python run_cv_breastmnist.py --n_components $comps --n_qumodes $q --epochs 30 --save-prefix "ablation_c${comps}_q${q}"
    done
  done
"""

import os
from tqdm import tqdm
import sys
import argparse
import time
import json
from pathlib import Path
import joblib
import inspect

import random

# make sure local modules folder is importable (matching notebook)
# sys.path.append("./modules")

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# local modules (from the notebook)
try:
    from modules.data_utils import load_data, extract_xy_from_loader, sample_batch, PCAEncoder, data_scaler, dataset_dataloaders
    from modules.model_utils import cv_qcnn, QuantumLinear, gradcam_model, ClassicalModel, dv_qcnn, DVQuantumLinear, get_cv_qcnn_qnode, get_dv_qcnn_qnode, compute_ece, brier_score, evaluate_calibration
    # training helpers are available but we will implement a simple loop here
    from modules.training_validating import run_kfold_training, evaluate_v2, noise_robustness_validation
except Exception as e:
    print("Import error:", e)
    raise

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, average_precision_score
)

from sklearn.preprocessing import label_binarize

def compute_metrics(y_true, y_pred, y_score=None, n_classes=None):
    """
    y_score: numpy array (N, C) logits or probabilities.
    If logits, we will convert to probs via softmax.
    Metrics:
    - acc, macro P/R/F1 (multiclass)
    - auroc ovr (macro) if y_score
    - auprc ovr (macro) if y_score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if n_classes is None:
        n_classes = int(np.max(y_true) + 1)

    acc = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    else:
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    auroc_ovr = None
    auprc_ovr = None

    if y_score is not None:
        y_score = np.asarray(y_score)

        row_sums = y_score.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            y_score = np.exp(y_score - np.max(y_score, axis=1, keepdims=True))
            y_score = y_score / np.sum(y_score, axis=1, keepdims=True)

        if n_classes == 2:
            #positive class prediction
            auroc_ovr = roc_auc_score(y_true, y_score[:,1])
            auprc_ovr = average_precision_score(y_true, y_score[:, 1])

        else:
            # ovr macro
            auroc_ovr = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            auprc_ovr = average_precision_score(
                label_binarize(y_true, classes=np.arange(n_classes)),
                y_score,
                average="macro"
            )

        return {
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "auroc_ovr": None if auroc_ovr is None else float(auroc_ovr),
            "auprc_ovr": None if auprc_ovr is None else float(auprc_ovr),
        }

# ------------ helper utilities ------------

def set_all_seeds(seed: int, deterministic: bool=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def try_construct_model(cls, **kwargs):
    """
    Try to construct `cls` passing only supported kwargs (useful for optional depth).
    """
    try:
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters.keys())
        # remove 'self'
        accepted.discard('self')
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return cls(**filtered)
    except Exception:
        # fallback: instantiate with only n_qumodes and n_classes if available
        fallback_kwargs = {}
        for key in ('n_qumodes', 'n_qubits', 'n_classes'):
            if key in kwargs:
                fallback_kwargs[key] = kwargs[key]
        return cls(**fallback_kwargs)


def assert_depth_effect(model, requested_depth: int):
    """
    ensure that requested depth is reflected in the model object.
    you must update these attributes names to match your implementation.
    """
    if requested_depth is None:
        return

    candidate_attrs = ["depth", "n_layers", "num_layers", "layers"]
    found = False
    for attr in candidate_attrs:
        if hasattr(model, attr):
            found = True
            break

    if not found:
        raise RuntimeError(
            "Depth ablation requested, but model exposes no depth/n_layers attribute."
            "You must implement depth control inside QuantumLinear and expose it."
        )

def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# ------------ main pipeline ------------
def main(args):
    # set_global_seed(args.seed)
    set_all_seeds(args.seed)
    
    # device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    device = "cpu"
    print(f"Device: {device}")

    # 1) Load raw dataset (as used in the notebook)
    print("Loading data...")
    train_raw_loader, train_raw_dataset, val_raw_loader, val_raw_dataset, test_raw_loader, test_raw_dataset = load_data(
        args.data_flag, batch_size=args.batch_size, train_fraction=args.train_fraction
    )

    # 2) PCA encoder (either load or fit)
    print(f"PCA n_components={args.n_components}")
    encoder = PCAEncoder(n_components=args.n_components)
    if args.pca_path and Path(args.pca_path).exists():
        print(f"Loading PCA from {args.pca_path}")
        encoder.pca = joblib.load(args.pca_path)
        encoder.fitted = True
    else:
        print("Fitting PCA encoder on training raw loader...")
        encoder.fit(train_raw_loader)
        if args.pca_path:
            joblib.dump(encoder.pca, args.pca_path)
            print(f"Saved PCA to {args.pca_path}")

    x_train_encoded, y_train_tensor = encoder.transform(train_raw_loader)
    x_val_encoded, y_val_tensor = encoder.transform(val_raw_loader)
    x_test_encoded, y_test_tensor = encoder.transform(test_raw_loader)

    # 3) Scale encoded data (try loading provided scaler, otherwise fit StandardScaler)
    scaler = None
    if args.scaler_path and Path(args.scaler_path).exists():
        print(f"Loading scaler from {args.scaler_path}")
        scaler = joblib.load(args.scaler_path)
    else:
        # try to use data_scaler from your modules if available and if user asked to use it
        if args.use_data_scaler:
            try:
                print("Using module data_scaler to scale data (it may save a scaler).")
                x_train_tensor, x_val_tensor, x_test_tensor = data_scaler(x_train_encoded, x_val_encoded, x_test_encoded, save_database_name=args.save_dbname)
                # ensure dtype
                x_train_tensor = x_train_tensor.float(); x_val_tensor = x_val_tensor.float(); x_test_tensor = x_test_tensor.float()
                # y tensors already prepared by encoder.transform
                print("Data scaled via data_scaler helper.")
                scaled_by_helper = True
            except Exception as e:
                print("data_scaler helper failed, falling back to sklearn StandardScaler. Error:", e)
                scaled_by_helper = False
        else:
            scaled_by_helper = False

        if not scaled_by_helper:
            print("Fitting StandardScaler on encoded training data")
            scaler = StandardScaler()
            scaler.fit(x_train_encoded)
            x_train_scaled = scaler.transform(x_train_encoded)
            x_val_scaled = scaler.transform(x_val_encoded)
            x_test_scaled = scaler.transform(x_test_encoded)

            x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
            x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
            x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

            if args.scaler_path:
                joblib.dump(scaler, args.scaler_path)
                print(f"Saved scaler to {args.scaler_path}")

    # Ensure we have x_*_tensor and y_*_tensor variables
    # y_train_tensor, y_val_tensor, y_test_tensor are returned by encoder.transform as tensors already
    # For safety, convert to torch tensors and correct shapes
    y_train_tensor = y_train_tensor.long().squeeze()
    y_val_tensor = y_val_tensor.long().squeeze()
    y_test_tensor = y_test_tensor.long().squeeze()

    n_classes = int(len(torch.unique(y_train_tensor)))
    print(f"n_classes: {n_classes}")
    print("Shapes:", x_train_tensor.shape, y_train_tensor.shape)


    # if args.kfold and args.kfold > 1:
    #     results = run_kfold_training(
    #         x_train_tensor, y_train_tensor,
    #         n_splits=args.kfold,
    #         epochs=args.epochs,
    #         lr = args.lr,
    #         batch_size=args.batch_size,
    #         n_qumodes = args.n_qumodes,
    #         depth = args.depth,
    #         seed=args.seed
    #     )

    # 4) Prepare PyTorch datasets/loaders using your helper if available
    
    train_dataset, train_loader = dataset_dataloaders(x_train_tensor, y_train_tensor, train=True, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_dataset, val_loader = dataset_dataloaders(x_val_tensor, y_val_tensor, train=False, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_dataset, test_loader = dataset_dataloaders(x_test_tensor, y_test_tensor, train=False, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    # 5) Construct Quantum CV model (QuantumLinear)
    print("Constructing CV model...")
    model_kwargs = {"n_qumodes": args.n_qumodes, "n_classes": n_classes}
    # include depth if provided
    if args.depth is not None:
        model_kwargs['depth'] = args.depth
        # also try common alt name
        model_kwargs['n_layers'] = args.depth

    try:
        cv_model = try_construct_model(QuantumLinear, **model_kwargs)
        
        assert_depth_effect(cv_model, args.depth)
        
    except Exception as e:
        print("Failed to instantiate QuantumLinear with provided args:", model_kwargs)
        raise

    cv_model = cv_model.to(device)
    trainable_params = count_trainable_params(cv_model)
    print(cv_model)
    print("trainable params:", trainable_params)

    # 6) Optimizer and loss
    optimizer = optim.Adam(cv_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 7) Training loop
    history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}
    best_val_f1 = -np.inf
    best_state = None
    # start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        cv_model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_scores = []

        # for xb, yb in train_loader:
        pbar = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{args.epochs}",
            leave=False,
            total=len(train_loader)
        )
        for step, (xb, yb) in enumerate(pbar, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            outputs = cv_model(xb)
            # outputs shape: (batch, n_classes) or something similar
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.detach().cpu().numpy())
            try:
                all_scores.append(outputs.detach().cpu().numpy())
            except Exception:
                pass

        train_loss = running_loss / len(train_loader.dataset)
        
        pbar.set_postfix({
            "step": f"{step}/{len(train_loader)}",
            "loss": f"{loss.item():.4f}"
        })
        
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        y_score = np.vstack(all_scores) if len(all_scores) else None
        # train_metrics = compute_metrics(y_true, y_pred, y_score)
        train_metrics = compute_metrics(y_true, y_pred, y_score, n_classes=n_classes)

        # validation
        cv_model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_scores = []
        with torch.no_grad():
            # for xb, yb in val_loader:
            pbar = tqdm(
                val_loader,
                desc="Val epoch {epoch}/{args.epochs}",
                leave=False,
                total=len(val_loader)
            )
            for step, (xb, yb) in enumerate(pbar, start=1):
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = cv_model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.append(outputs.argmax(dim=1).detach().cpu().numpy())
                val_targets.append(yb.detach().cpu().numpy())
                try:
                    val_scores.append(outputs.detach().cpu().numpy())
                except Exception:
                    pass

        val_loss = val_loss / len(val_loader.dataset)

        pbar.set_postfix({
            "step": f"{step}/{len(val_loader)}",
            "loss": f"{loss.item():.4f}"
        })
        
        val_pred = np.concatenate(val_preds, axis=0)
        val_true = np.concatenate(val_targets, axis=0)
        val_score = np.vstack(val_scores) if len(val_scores) else None
        # val_metrics = compute_metrics(val_true, val_pred, val_score)
        val_metrics = compute_metrics(val_true, val_pred, val_score, n_classes=n_classes)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        # simple progress print
        print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_f1: {train_metrics['f1']:.4f} | val_f1: {val_metrics['f1']:.4f}")

        # save best
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = cv_model.state_dict().copy()

    # elapsed = time.time() - start_time
    # print(f"Training finished in {elapsed/60:.2f} minutes. Best val F1: {best_val_f1:.4f}")
    print(f"Training finished. Best val F1: {best_val_f1:.4f}")

    # 8) Save model and history
    out_prefix = args.save_prefix or f"cv_{args.data_flag}_components{args.n_components}_qumodes{args.n_qumodes}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{out_prefix}_best.pth"
    if best_state:
        torch.save(best_state, model_path)
        print(f"Saved best model to {model_path}")

    history_path = out_dir / f"{out_prefix}_history.json"
    # convert history to JSON-serializable
    serial_history = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_metrics": history["train_metrics"],
        "val_metrics": history["val_metrics"],
        "args": vars(args),
        "trainable_params": trainable_params
    }
    with open(history_path, "w") as f:
        json.dump(serial_history, f, indent=2)
    print(f"Saved history to {history_path}")

    # 9) Final evaluation on test set (load best_state)
    if best_state:
        cv_model.load_state_dict(torch.load(model_path, map_location=device))
    cv_model.eval()
    test_preds = []
    test_targets = []
    test_scores = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = cv_model(xb)
            test_preds.append(outputs.argmax(dim=1).cpu().numpy())
            test_targets.append(yb.cpu().numpy())
            try:
                test_scores.append(outputs.cpu().numpy())
            except Exception:
                pass

    test_pred = np.concatenate(test_preds, axis=0)
    test_true = np.concatenate(test_targets, axis=0)
    test_score = np.vstack(test_scores) if len(test_scores) else None
    # test_metrics = compute_metrics(test_true, test_pred, test_score)
    test_metrics = compute_metrics(test_true, test_pred, test_score, n_classes=n_classes)
    print("Test metrics:", test_metrics)

    test_cal = evaluate_calibration(cv_model, test_loader, device)
    print("ECE score: ", test_cal["ece"])
    print("Brier score: ", test_cal["brier"])

    # Save final metrics
    final_path = out_dir / f"{out_prefix}_final_metrics.json"
    with open(final_path, "w") as f:
        # json.dump({"test_metrics": test_metrics, "val_best_f1": best_val_f1 , "ece_score": test_cal["ece"], "brier_score": test_cal["brier"], "args": vars(args)}, f, indent=2)
        json.dump({"test_metrics": test_metrics, "val_best_f1": best_val_f1 , "args": vars(args)}, f, indent=2)
    print(f"Saved final metrics to {final_path}")


    # print(f"ECE score: {test_cal["ece"]} | Brier score: {test_cal["brier"]}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run CV quantum model (breastmnist) experiment")
    
    parser.add_argument("--data-flag", type=str, default="breastmnist", help="dataset name (as used by load_data)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--n_components", type=int, default=4, help="PCA components (encoder)")
    parser.add_argument("--pca-path", type=str, default="", help="(optional) path to load/save PCA encoder (joblib)")
    parser.add_argument("--scaler-path", type=str, default="", help="(optional) path to load/save scaler (joblib)")
    parser.add_argument("--use-data-scaler", action="store_true", help="Try to use the module's data_scaler helper (if present)")
    parser.add_argument("--kfold", type=int, default=0, help="if >0, run k-fold training using run_kflod_training")
    parser.add_argument("--save-dbname", type=str, default="breast", help="database name for data_scaler if used")
    parser.add_argument("--n_qumodes", type=int, default=4, help="Number of CV qumodes (model input size)")
    parser.add_argument("--depth", type=int, default=None, help="Optional depth/n_layers to pass to model if supported")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="cv_runs")
    parser.add_argument("--save-prefix", type=str, default="")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()

    set_all_seeds(args.seed, deterministic=args.deterministic)


    if args.n_components != args.n_qumodes:
        raise ValueError(
            f"Invalid configuration: n_components ({args.n_components}) must equal"
            f"n_qumodes ({args.n_qumodes}) for 1-to-1 cv feature encoding"
        )
        

    main(args)