import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import random
import json
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

try:
    from torchvision.models import EfficientNet_B0_Weights
    HAS_WEIGHTS_API = True
except Exception:
    HAS_WEIGHTS_API = False

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
SEED = 42

DATA_ROOT = r"C:\Users\csj5\Projects\Data\Breasttumor_classi_renamed"
OUTPUT_DIR = "breast_tumor_federated_scaffold"
MODEL_NAME = "efficientnet_b0_breast_tumor_scaffold.pth"

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-3
WEIGHT_DECAY = 0
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES =["BUSBRA", "BUS", "BUSI", "UDIAT"]

USE_SCAFFOLD = True
ETA_G = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# Transforms
# =========================================================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# =========================================================
# Helpers
# =========================================================
def set_dataset_paths(root, client_names):
    paths = {}
    for client in client_names:
        paths[client] = {
            "train": os.path.join(root, client, "train"),
            "val": os.path.join(root, client, "val"),
            "test": os.path.join(root, client, "test"),
        }
    return paths


def check_class_alignment(datasets_list):
    base_classes = datasets_list[0].classes
    base_class_to_idx = datasets_list[0].class_to_idx

    for i, ds in enumerate(datasets_list[1:], start=2):
        if ds.classes != base_classes:
            raise ValueError(
                f"Class mismatch detected in dataset {i}.\n"
                f"Expected classes: {base_classes}\n"
                f"Found classes   : {ds.classes}\n"
                "All clients must have the same class folder names."
            )
        if ds.class_to_idx != base_class_to_idx:
            raise ValueError(
                f"Class-to-index mismatch detected in dataset {i}.\n"
                "All clients must use identical class folder naming."
            )

    return base_classes, base_class_to_idx


def build_client_datasets(paths_dict, split, transform):
    ds_list = []
    for client in CLIENT_NAMES:
        split_dir = paths_dict[client][split]
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        ds = datasets.ImageFolder(split_dir, transform=transform)
        ds_list.append(ds)

    classes, class_to_idx = check_class_alignment(ds_list)
    return ds_list, classes, class_to_idx


def build_combined_dataset(ds_list):
    if len(ds_list) == 1:
        return ds_list[0]
    return ConcatDataset(ds_list)


def count_samples(ds):
    if isinstance(ds, ConcatDataset):
        return sum(len(d) for d in ds.datasets)
    return len(ds)


def build_model(num_classes):
    if HAS_WEIGHTS_API:
        try:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            model = efficientnet_b0(weights=None)
    else:
        model = efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_specificity_from_cm(cm):
    num_classes = cm.shape[0]
    per_class_specificity = []

    total = cm.sum()
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        denom = tn + fp
        specificity = tn / denom if denom > 0 else 0.0
        per_class_specificity.append(specificity)

    macro_specificity = float(np.mean(per_class_specificity))
    return per_class_specificity, macro_specificity


def average_state_dicts_weighted(models, weights):
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k].detach().cpu() for i in range(len(models)))
    return avg_sd


def float_state_dict(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items() if torch.is_floating_point(v)}


def zeros_like_float_state_dict(sd):
    return {k: torch.zeros_like(v.detach().cpu()) for k, v in sd.items() if torch.is_floating_point(v)}


def state_dict_to_device(sd, device):
    return {k: v.to(device) for k, v in sd.items()}


def compute_class_weights_from_dataset(dataset, num_classes):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_targets(targets, num_classes):
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


# =========================================================
# Training / Evaluation
# =========================================================
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model = model.to(DEVICE)

    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    loop = tqdm(loader, desc="Train" if train else "Eval", leave=False)

    for images, labels in loop:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds


def train_local_scaffold(model, loader, criterion, optimizer, ci_sd_cpu, c_sd_cpu, epochs=LOCAL_EPOCHS):
    model.to(DEVICE)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    c_dev = state_dict_to_device(c_sd_cpu, DEVICE)
    ci_dev = state_dict_to_device(ci_sd_cpu, DEVICE)

    param_map = {name: p for name, p in model.named_parameters()}
    total_steps = 0

    last_loss = 0.0
    last_acc = 0.0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(loader, desc=f"SCAFFOLD Local ep{ep + 1}/{epochs}", leave=False)

        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(x)
                loss = criterion(out, y)

            if USE_AMP:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
            else:
                loss.backward()

            with torch.no_grad():
                for name, p in param_map.items():
                    if p.grad is None:
                        continue
                    p.grad.add_(c_dev[name] - ci_dev[name])

            if USE_AMP:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(y.detach().cpu().numpy())

            total_steps += 1
            pbar.set_postfix(loss=float(loss.item()))

        last_loss = running_loss / max(1, len(loader.dataset))
        last_acc = accuracy_score(all_targets, all_preds)

    return model.cpu(), total_steps, last_loss, last_acc


@torch.no_grad()
def evaluate_model(model, dataloader, criterion=None, return_per_class=False, class_names=None):
    model = model.to(DEVICE)
    model.eval()

    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0

    for x, y in tqdm(dataloader, desc="Eval", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        out = model(x)
        _, preds = out.max(1)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

        if criterion is not None:
            loss = criterion(out, y)
            total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

    if n == 0:
        return {}

    acc = accuracy_score(all_y, all_pred)
    prec_macro = precision_score(all_y, all_pred, average="macro", zero_division=0)
    rec_macro = recall_score(all_y, all_pred, average="macro", zero_division=0)
    f1_macro = f1_score(all_y, all_pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(all_y, all_pred)
    kappa = cohen_kappa_score(all_y, all_pred)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "balanced_acc": float(bal),
        "cohen_kappa": float(kappa),
    }

    if criterion is not None:
        metrics["loss"] = float(total_loss / max(1, n))

    if return_per_class:
        if class_names is None:
            raise ValueError("class_names must be provided when return_per_class=True")

        num_classes = len(class_names)
        cm = confusion_matrix(all_y, all_pred, labels=list(range(num_classes)))
        per_class_prec = precision_score(all_y, all_pred, labels=list(range(num_classes)), average=None, zero_division=0)
        per_class_rec = recall_score(all_y, all_pred, labels=list(range(num_classes)), average=None, zero_division=0)
        per_class_f1 = f1_score(all_y, all_pred, labels=list(range(num_classes)), average=None, zero_division=0)

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        per_class_spec = np.divide(tn, (tn + fp), out=np.zeros_like(tn), where=(tn + fp) != 0)
        support = cm.sum(axis=1).astype(float)
        per_class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)

        metrics.update({
            "confusion_matrix": cm,
            "per_class_precision": [float(x) for x in per_class_prec.tolist()],
            "per_class_recall": [float(x) for x in per_class_rec.tolist()],
            "per_class_f1": [float(x) for x in per_class_f1.tolist()],
            "per_class_specificity": [float(x) for x in per_class_spec.tolist()],
            "per_class_correct": [int(x) for x in tp.tolist()],
            "per_class_accuracy": [float(x) for x in per_class_acc.tolist()],
            "per_class_support": [int(x) for x in support.tolist()],
        })

    return metrics


def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR, save_cm=True):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

    cm = confusion_matrix(targets, preds)
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    print(f"\n=== {title_prefix.upper()} ===")
    print(f"Loss        : {loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision_macro:.4f}")
    print(f"Recall      : {recall_macro:.4f}")
    print(f"F1-score    : {f1_macro:.4f}")
    print(f"Kappa       : {kappa:.4f}")
    print(f"Specificity : {macro_specificity:.4f}")

    print("\nPer-class Specificity:")
    for idx, cls_name in enumerate(class_names):
        print(f"{cls_name:15s}: {per_class_specificity[idx]:.4f}")

    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(cm)

    if save_cm:
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {title_prefix}")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_{title_prefix}.png"), dpi=300)
        plt.close()

    return {
        "split": title_prefix,
        "loss": loss,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "kappa": kappa,
        "specificity_macro": macro_specificity,
        "per_class_specificity": per_class_specificity,
        "cm": cm.tolist(),
    }


def save_metrics_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "kappa",
            "specificity_macro"
        ])
        for r in results:
            writer.writerow([
                r["split"],
                f'{r["loss"]:.6f}',
                f'{r["accuracy"]:.6f}',
                f'{r["precision_macro"]:.6f}',
                f'{r["recall_macro"]:.6f}',
                f'{r["f1_macro"]:.6f}',
                f'{r["kappa"]:.6f}',
                f'{r["specificity_macro"]:.6f}'
            ])


def plot_round_curves(history, out_dir):
    rounds = np.arange(1, len(history["round"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["train_loss"], label="Train Loss")
    plt.plot(rounds, history["val_loss"], label="Val Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Federated Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_train_val_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["train_acc"], label="Train Acc")
    plt.plot(rounds, history["val_acc"], label="Val Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_train_val_acc.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["global_test_acc"], label="Global Test Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Global Test Accuracy Across Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_global_test_acc.png"), dpi=300)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

    train_ds_all = build_combined_dataset(train_datasets)
    val_ds_all = build_combined_dataset(val_datasets)
    test_ds_all = build_combined_dataset(test_datasets)

    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Train samples (all clients):", count_samples(train_ds_all))
    print("Val samples   (all clients):", count_samples(val_ds_all))
    print("Test samples  (all clients):", count_samples(test_ds_all))

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        train_loaders[client] = DataLoader(
            ds_tr,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        val_loaders[client] = DataLoader(
            ds_va,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        test_loaders[client] = DataLoader(
            ds_te,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

    global_val_loader = DataLoader(
        val_ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    global_test_loader = DataLoader(
        test_ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    global_model = build_model(num_classes).to(DEVICE)

    combined_train_targets = []
    for ds in train_datasets:
        combined_train_targets.extend([s[1] for s in ds.samples])
    combined_weights = compute_class_weights_from_targets(combined_train_targets, num_classes).to(DEVICE)
    combined_criterion = nn.CrossEntropyLoss(weight=combined_weights)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())

    history = defaultdict(list)
    round_metrics = []
    start_time = time.time()

    # SCAFFOLD only for float tensors
    initial_sd = global_model.state_dict()
    c_global = zeros_like_float_state_dict(initial_sd)
    c_clients = [zeros_like_float_state_dict(initial_sd) for _ in range(len(CLIENT_NAMES))]
    float_keys = set(c_global.keys())

    for r in range(COMM_ROUNDS):
        print(f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")

        local_models = []
        local_deltas = []
        local_delta_cs = []
        round_train_losses = []
        round_train_accs = []

        global_sd_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        for client_idx, client_name in enumerate(CLIENT_NAMES):
            print(f"\n[Client {client_name}]")

            local_model = copy.deepcopy(global_model).to(DEVICE)
            client_train_ds = train_loaders[client_name].dataset
            client_val_loader = val_loaders[client_name]
            client_test_loader = test_loaders[client_name]

            client_cw = compute_class_weights_from_dataset(client_train_ds, num_classes).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            if USE_SCAFFOLD:
                trained_local_model, K_steps, tr_loss, tr_acc = train_local_scaffold(
                    local_model,
                    train_loaders[client_name],
                    criterion,
                    optimizer,
                    ci_sd_cpu=c_clients[client_idx],
                    c_sd_cpu=c_global,
                    epochs=LOCAL_EPOCHS
                )

                print(f"  Local Train Loss: {tr_loss:.4f} | Local Train Acc: {tr_acc:.4f}")

                # validate on GPU
                local_model = trained_local_model.to(DEVICE)
                val_loss, val_acc, _, _ = run_epoch(
                    local_model,
                    client_val_loader,
                    criterion,
                    optimizer=None,
                    train=False
                )
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                local_model = local_model.cpu()

                # only float tensors participate in SCAFFOLD
                local_sd_cpu = float_state_dict(local_model.state_dict())
                K = max(1, int(K_steps))

                ci_plus = {}
                for k in float_keys:
                    xi = global_sd_cpu[k]
                    yi = local_sd_cpu[k]
                    diff = (xi - yi) / float(K * LR)
                    ci_plus[k] = (c_clients[client_idx][k].detach().cpu() - c_global[k].detach().cpu()) + diff

                delta_ci = {k: (ci_plus[k] - c_clients[client_idx][k].detach().cpu()) for k in float_keys}
                local_delta_cs.append((client_idx, delta_ci))

                delta_y = {}
                for k in float_keys:
                    delta_y[k] = local_sd_cpu[k] - global_sd_cpu[k]
                local_deltas.append(delta_y)

                for k in float_keys:
                    c_clients[client_idx][k] = ci_plus[k].detach().cpu().clone()

            else:
                logs = []
                for ep in range(LOCAL_EPOCHS):
                    tr_loss, tr_acc, _, _ = run_epoch(
                        local_model,
                        train_loaders[client_name],
                        criterion,
                        optimizer=optimizer,
                        train=True
                    )
                    logs.append((tr_loss, tr_acc))
                    print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

                tr_loss, tr_acc = logs[-1]
                print(f"  Local Train Loss: {tr_loss:.4f} | Local Train Acc: {tr_acc:.4f}")

                val_loss, val_acc, _, _ = run_epoch(
                    local_model,
                    client_val_loader,
                    criterion,
                    optimizer=None,
                    train=False
                )
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                local_model = local_model.cpu()

            round_train_losses.append(float(tr_loss))
            round_train_accs.append(float(tr_acc))
            local_models.append(local_model.cpu())

        # -------------------------------------------------
        # Aggregation
        # -------------------------------------------------
        if USE_SCAFFOLD:
            print("\nAggregating local updates (SCAFFOLD)")

            n_clients = float(len(local_deltas))
            avg_dy = {k: torch.zeros_like(v) for k, v in local_deltas[0].items()}

            for d in local_deltas:
                for k in avg_dy.keys():
                    avg_dy[k] += d[k]

            for k in avg_dy.keys():
                avg_dy[k] = avg_dy[k] / n_clients

            global_sd = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

            for k in float_keys:
                global_sd[k] = global_sd[k] + ETA_G * avg_dy[k].cpu()

            global_model.load_state_dict(global_sd)
            global_model.to(DEVICE)

            if len(local_delta_cs) > 0:
                sum_delta_c = {k: torch.zeros_like(v) for k, v in c_global.items()}

                for client_idx, dci in local_delta_cs:
                    for k in sum_delta_c.keys():
                        sum_delta_c[k] += dci[k].detach().cpu()

                for k in c_global.keys():
                    c_global[k] = (c_global[k].detach().cpu() + (sum_delta_c[k] / float(len(c_clients)))).cpu()

        else:
            total_train_size = sum(len(train_loaders[c].dataset) for c in CLIENT_NAMES)
            weights = [len(train_loaders[c].dataset) / total_train_size for c in CLIENT_NAMES]
            print("\nAggregating local models (FedAvg weighted)")
            global_model.load_state_dict(average_state_dicts_weighted(local_models, weights))
            global_model.to(DEVICE)

        # -------------------------------------------------
        # Global validation
        # -------------------------------------------------
        print("\nGlobal validation on combined val sets...")
        global_val_metrics = evaluate_model(global_model, global_val_loader, criterion=combined_criterion)

        print("Global combined val metrics:", {
            k: global_val_metrics.get(k)
            for k in ["loss", "accuracy", "f1_macro", "precision_macro", "recall_macro", "balanced_acc", "cohen_kappa"]
        })

        global_val_loss = float(global_val_metrics.get("loss", np.nan))
        global_val_acc = float(global_val_metrics.get("accuracy", np.nan))

        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
            print("\nSaved best global model.")

        # -------------------------------------------------
        # Global test
        # -------------------------------------------------
        print("\nGlobal TEST on combined test (all clients)")
        global_test_metrics = evaluate_model(
            global_model,
            global_test_loader,
            criterion=combined_criterion,
            return_per_class=True,
            class_names=class_names
        )

        print("Global combined TEST metrics summary:", {
            k: global_test_metrics.get(k)
            for k in ["accuracy", "loss", "f1_macro", "precision_macro", "recall_macro", "balanced_acc", "cohen_kappa"]
        })

        try:
            cm = global_test_metrics.get("confusion_matrix", None)
            per_prec = global_test_metrics.get("per_class_precision", [])
            per_rec = global_test_metrics.get("per_class_recall", [])
            per_f1 = global_test_metrics.get("per_class_f1", [])
            per_spec = global_test_metrics.get("per_class_specificity", [])
            per_acc = global_test_metrics.get("per_class_accuracy", [])
            per_support = global_test_metrics.get("per_class_support", [])
            per_correct = global_test_metrics.get("per_class_correct", [])

            print("\nCombined TEST per-class metrics (order = {}):".format(class_names))
            header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
            print("{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))
            for ci, cname in enumerate(class_names):
                s = int(per_support[ci]) if ci < len(per_support) else 0
                ccount = int(per_correct[ci]) if ci < len(per_correct) else 0
                acc_val = float(per_acc[ci]) if ci < len(per_acc) else np.nan
                pval = float(per_prec[ci]) if ci < len(per_prec) else np.nan
                rval = float(per_rec[ci]) if ci < len(per_rec) else np.nan
                fval = float(per_f1[ci]) if ci < len(per_f1) else np.nan
                sval = float(per_spec[ci]) if ci < len(per_spec) else np.nan
                print("{:12s} {:8d} {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(
                    cname, s, ccount, acc_val, pval, rval, fval, sval
                ))

            combined_rows = []
            if cm is not None:
                for ci, cname in enumerate(class_names):
                    combined_rows.append({
                        "class": cname,
                        "support": int(per_support[ci]) if ci < len(per_support) else 0,
                        "correct": int(per_correct[ci]) if ci < len(per_correct) else 0,
                        "acc": float(per_acc[ci]) if ci < len(per_acc) else np.nan,
                        "prec": float(per_prec[ci]) if ci < len(per_prec) else np.nan,
                        "rec": float(per_rec[ci]) if ci < len(per_rec) else np.nan,
                        "f1": float(per_f1[ci]) if ci < len(per_f1) else np.nan,
                        "spec": float(per_spec[ci]) if ci < len(per_spec) else np.nan
                    })

            macro_row = {
                "class": "macro",
                "support": int(cm.sum()) if cm is not None else 0,
                "correct": int(np.sum([r["correct"] for r in combined_rows])) if combined_rows else 0,
                "acc": float(global_test_metrics.get("balanced_acc", np.nan)),
                "prec": float(global_test_metrics.get("precision_macro", np.nan)),
                "rec": float(global_test_metrics.get("recall_macro", np.nan)),
                "f1": float(global_test_metrics.get("f1_macro", np.nan)),
                "spec": float(np.mean(per_spec)) if len(per_spec) > 0 else float(np.nan)
            }
            combined_rows.append(macro_row)
            df_combined = pd.DataFrame(combined_rows)
            combined_csv = os.path.join(OUTPUT_DIR, "combined_test_metrics.csv")
            df_combined.to_csv(combined_csv, index=False)
            if cm is not None:
                pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
                    os.path.join(OUTPUT_DIR, "combined_confusion_matrix.csv")
                )
            print(f"Saved combined per-class test metrics CSV to: {combined_csv}")
        except Exception as e:
            print("Warning saving/printing combined per-class metrics:", e)

        # -------------------------------------------------
        # Per-client test
        # -------------------------------------------------
        per_client_test_metrics = []
        print("\nGlobal TEST on each client test set")
        for i, client_name in enumerate(CLIENT_NAMES):
            client_train_ds = train_loaders[client_name].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds, num_classes).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)

            cl_metrics = evaluate_model(
                global_model,
                test_loaders[client_name],
                criterion=client_criterion,
                return_per_class=True,
                class_names=class_names
            )

            acc = cl_metrics.get("accuracy", np.nan)
            prec = cl_metrics.get("precision_macro", np.nan)
            rec = cl_metrics.get("recall_macro", np.nan)
            f1 = cl_metrics.get("f1_macro", np.nan)
            kappa = cl_metrics.get("cohen_kappa", np.nan)
            mean_spec = float(np.mean(cl_metrics.get("per_class_specificity", [np.nan])))

            print(f"\n[CLIENT {i}] {client_name}")
            print(f"  Accuracy        : {acc:.4f}")
            print(f"  Precision (mac) : {prec:.4f}")
            print(f"  Recall (mac)    : {rec:.4f}")
            print(f"  F1 (mac)        : {f1:.4f}")
            print(f"  Mean Specificity: {mean_spec:.4f}")
            print(f"  Cohen's kappa   : {kappa:.4f}")

            try:
                cmc = cl_metrics.get("confusion_matrix", None)
                rows = []
                if cmc is not None:
                    per_support = cl_metrics.get("per_class_support", [])
                    per_correct = cl_metrics.get("per_class_correct", [])
                    per_accs = cl_metrics.get("per_class_accuracy", [])
                    per_precs = cl_metrics.get("per_class_precision", [])
                    per_recs = cl_metrics.get("per_class_recall", [])
                    per_f1s = cl_metrics.get("per_class_f1", [])
                    per_specs = cl_metrics.get("per_class_specificity", [])

                    print(f"\n  Per-class metrics (order = {class_names}):")
                    header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
                    print("    " + "{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))

                    for ci, cname in enumerate(class_names):
                        s = int(per_support[ci]) if ci < len(per_support) else 0
                        ccount = int(per_correct[ci]) if ci < len(per_correct) else 0
                        acc_val = float(per_accs[ci]) if ci < len(per_accs) else np.nan
                        pval = float(per_precs[ci]) if ci < len(per_precs) else np.nan
                        rval = float(per_recs[ci]) if ci < len(per_recs) else np.nan
                        fval = float(per_f1s[ci]) if ci < len(per_f1s) else np.nan
                        sval = float(per_specs[ci]) if ci < len(per_specs) else np.nan
                        print("    {:12s} {:8d} {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(
                            cname, int(s), int(ccount), float(acc_val), float(pval), float(rval), float(fval), float(sval)
                        ))

                        rows.append({
                            "class": cname,
                            "support": s,
                            "correct": ccount,
                            "acc": acc_val,
                            "prec": pval,
                            "rec": rval,
                            "f1": fval,
                            "spec": sval
                        })

                if rows:
                    rows.append({
                        "class": "macro",
                        "support": int(cmc.sum()) if cmc is not None else 0,
                        "correct": int(np.sum([r["correct"] for r in rows])) if rows else 0,
                        "acc": cl_metrics.get("balanced_acc", np.nan),
                        "prec": cl_metrics.get("precision_macro", np.nan),
                        "rec": cl_metrics.get("recall_macro", np.nan),
                        "f1": cl_metrics.get("f1_macro", np.nan),
                        "spec": float(np.mean(cl_metrics.get("per_class_specificity", [np.nan])))
                    })

                    dfc = pd.DataFrame(rows)
                    safe_name = client_name.replace(" ", "_")
                    client_csv = os.path.join(OUTPUT_DIR, f"{safe_name}_test_metrics_round{r + 1}.csv")
                    dfc.to_csv(client_csv, index=False)
                    if cmc is not None:
                        pd.DataFrame(cmc, index=class_names, columns=class_names).to_csv(
                            os.path.join(OUTPUT_DIR, f"{safe_name}_confusion_matrix_round{r + 1}.csv")
                        )
                    print(f"Saved per-client metrics CSV to {client_csv}")

            except Exception as e:
                print("Warning saving per-client metrics CSV:", e)

            per_client_test_metrics.append(cl_metrics)

        # -------------------------------------------------
        # Save checkpoint
        # -------------------------------------------------
        ckpt = {
            "round": r + 1,
            "model_state": global_model.state_dict(),
            "global_val_metrics": global_val_metrics,
            "global_test_metrics": global_test_metrics,
            "per_client_test_metrics": per_client_test_metrics,
            "client_names": CLIENT_NAMES,
            "class_names": class_names,
            "use_scaffold": USE_SCAFFOLD,
            "c_global": c_global,
            "c_clients": c_clients,
        }
        ckpt_path = os.path.join(OUTPUT_DIR, f"global_round_{r + 1}.pth")
        torch.save(ckpt, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # -------------------------------------------------
        # Round logs
        # -------------------------------------------------
        rm = {
            "round": r + 1,
            "train_loss": float(np.mean(round_train_losses)),
            "train_acc": float(np.mean(round_train_accs)),
            "val_loss": global_val_loss,
            "val_acc": global_val_acc,
            "global_test_loss": float(global_test_metrics.get("loss", np.nan)),
            "global_test_acc": float(global_test_metrics.get("accuracy", np.nan)),
            "global_test_precision": float(global_test_metrics.get("precision_macro", np.nan)),
            "global_test_recall": float(global_test_metrics.get("recall_macro", np.nan)),
            "global_test_f1": float(global_test_metrics.get("f1_macro", np.nan)),
            "global_test_kappa": float(global_test_metrics.get("cohen_kappa", np.nan)),
            "global_test_specificity": float(np.mean(global_test_metrics.get("per_class_specificity", [np.nan]))),
        }

        for i, client_name in enumerate(CLIENT_NAMES):
            cl_metrics = per_client_test_metrics[i]
            safe = client_name.replace(" ", "_")
            rm[f"{safe}_test_loss"] = float(cl_metrics.get("loss", np.nan))
            rm[f"{safe}_test_acc"] = float(cl_metrics.get("accuracy", np.nan))
            rm[f"{safe}_test_precision"] = float(cl_metrics.get("precision_macro", np.nan))
            rm[f"{safe}_test_recall"] = float(cl_metrics.get("recall_macro", np.nan))
            rm[f"{safe}_test_f1"] = float(cl_metrics.get("f1_macro", np.nan))
            rm[f"{safe}_test_kappa"] = float(cl_metrics.get("cohen_kappa", np.nan))
            rm[f"{safe}_test_specificity"] = float(np.mean(cl_metrics.get("per_class_specificity", [np.nan])))

        round_metrics.append(rm)

        history["round"].append(r + 1)
        history["train_loss"].append(rm["train_loss"])
        history["train_acc"].append(rm["train_acc"])
        history["val_loss"].append(rm["val_loss"])
        history["val_acc"].append(rm["val_acc"])
        history["global_test_acc"].append(rm["global_test_acc"])

        print(
            f"\n[ROUND {r + 1}] "
            f"Train Loss: {rm['train_loss']:.4f} | Train Acc: {rm['train_acc']:.4f} | "
            f"Val Loss: {rm['val_loss']:.4f} | Val Acc: {rm['val_acc']:.4f} | "
            f"Global Test Acc: {rm['global_test_acc']:.4f}"
        )

        plot_round_curves(history, OUTPUT_DIR)

        df = pd.DataFrame(round_metrics)
        csv_path = os.path.join(OUTPUT_DIR, "fl_round_results.csv")
        df.to_csv(csv_path, index=False)
        print("Saved per-round summary CSV to", csv_path)

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    global_model.load_state_dict(best_model_wts)

    final_results = []

    print("\n==============================")
    print("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print("==============================")
    result_all = evaluate_loader(
        global_model,
        global_test_loader,
        combined_criterion,
        class_names,
        title_prefix="all_clients_final",
        save_dir=OUTPUT_DIR,
        save_cm=True
    )
    final_results.append(result_all)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name in CLIENT_NAMES:
        client_train_ds = train_loaders[client_name].dataset
        client_cw = compute_class_weights_from_dataset(client_train_ds, num_classes).to(DEVICE)
        client_criterion = nn.CrossEntropyLoss(weight=client_cw)

        result_client = evaluate_loader(
            global_model,
            test_loaders[client_name],
            client_criterion,
            class_names,
            title_prefix=f"{client_name}_final",
            save_dir=OUTPUT_DIR,
            save_cm=True
        )
        final_results.append(result_client)

    with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"), "w") as f:
        json.dump(round_metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(round_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(round_metrics)

    save_metrics_csv(final_results, os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    with open(os.path.join(OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    torch.save({
        "model_state": global_model.state_dict(),
        "class_names": class_names,
        "use_scaffold": USE_SCAFFOLD,
        "c_global": c_global,
        "c_clients": c_clients,
    }, os.path.join(OUTPUT_DIR, "global_final.pth"))

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "global_final.pth"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))


if __name__ == "__main__":
    main()