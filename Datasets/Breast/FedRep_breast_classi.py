import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import random
import json
import csv
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt

try:
    from torchvision.models import EfficientNet_B0_Weights
    USE_WEIGHTS_API = True
except Exception:
    USE_WEIGHTS_API = False

# =========================
# Config
# =========================
SEED = 42
DATA_ROOT = r"xxxx\Data\Breasttumor_classi_renamed"
OUTPUT_DIR = "breast_tumor_fedrep"
MODEL_NAME = "efficientnet_b0_breast_tumor_fedrep.pth"

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-4
WEIGHT_DECAY = 0
NUM_WORKERS = 0
IMG_SIZE = 224
DROPOUT_P = 0.2

CLIENT_NAMES =["BUSBRA", "BUS", "BUSI", "UDIAT"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Transforms
# =========================
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

# =========================
# Helpers
# =========================
def set_dataset_paths(root, client_names):
    """
    Expected structure:
    DATA_ROOT/
        Sartajbhuvaji/train, val, test
        rm1000/train, val, test
        thomasdubail/train, val, test
        figshare/train, val, test
    Each split folder must contain class subfolders.
    """
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
                "All clients must have identical class folder names."
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
    if USE_WEIGHTS_API:
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        base = efficientnet_b0(pretrained=True)

    in_features = base.classifier[1].in_features

    class FedRepEfficientNet(nn.Module):
        def __init__(self, backbone, num_classes, dropout_p=0.2):
            super().__init__()
            self.representation = nn.Sequential(
                backbone.features,
                backbone.avgpool,
                nn.Flatten()
            )
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )

        def forward(self, x):
            x = self.representation(x)
            x = self.head(x)
            return x

    return FedRepEfficientNet(base, num_classes, dropout_p=DROPOUT_P)

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

def average_representation_state_dict(models, weights):
    """
    Average only shared representation parameters.
    Heads remain local and are never averaged.
    """
    base_sd = models[0].state_dict()
    rep_keys = [
        k for k in base_sd.keys()
        if k.startswith("representation.") and base_sd[k].dtype.is_floating_point
    ]

    if len(rep_keys) == 0:
        raise RuntimeError("No representation keys found to average.")

    avg_sd = {k: v.detach().cpu().clone() for k, v in base_sd.items()}

    for k in rep_keys:
        acc = torch.zeros_like(base_sd[k], dtype=torch.float32)
        for w, m in zip(weights, models):
            sd = m.state_dict()
            acc += float(w) * sd[k].detach().cpu().to(torch.float32)
        avg_sd[k] = acc.to(base_sd[k].dtype)

    return avg_sd

def load_representation(model, rep_state):
    model_sd = model.state_dict()
    for k, v in rep_state.items():
        if k.startswith("representation."):
            model_sd[k] = v.clone()
    model.load_state_dict(model_sd, strict=False)

def load_head_state(model, head_state):
    model_sd = model.state_dict()
    for k, v in head_state.items():
        if k.startswith("head."):
            model_sd[k] = v.clone()
    model.load_state_dict(model_sd, strict=False)

def compute_class_weights_from_dataset(dataset, num_classes):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()

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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
    epoch_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
    return epoch_loss, epoch_acc, all_targets, all_preds

def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR, save_cm=True):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    print(f"\n=== {title_prefix.upper()} ===")
    print(f"Loss        : {loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision_macro:.4f}")
    print(f"Recall      : {recall_macro:.4f}")
    print(f"F1-score    : {f1_macro:.4f}")
    print(f"Kappa       : {kappa:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Specificity : {macro_specificity:.4f}")

    print("\nPer-class Specificity:")
    for idx, cls_name in enumerate(class_names):
        print(f"{cls_name:15s}: {per_class_specificity[idx]:.4f}")

    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))

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
        "balanced_acc": bal_acc,
        "specificity_macro": macro_specificity,
        "per_class_specificity": per_class_specificity,
        "cm": cm.tolist(),
        "targets": targets,
        "preds": preds
    }

def evaluate_fedrep_across_clients(client_models, loaders_dict, class_names, criterion):
    """
    Evaluate personalized client models on their own loaders and aggregate
    all predictions into one overall metric set.
    """
    all_targets = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0
    per_client_results = {}

    for client_name in CLIENT_NAMES:
        model = client_models[client_name]
        loader = loaders_dict[client_name]

        # Make sure the model is on the correct device before evaluation.
        model = model.to(DEVICE)

        loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

        per_client_results[client_name] = {
            "loss": loss,
            "accuracy": acc
        }

        all_targets.extend(targets)
        all_preds.extend(preds)
        total_loss += loss * len(loader.dataset)
        total_samples += len(loader.dataset)

        # Keep the updated device placement in case the caller reuses the model
        client_models[client_name] = model

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    precision_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)
    avg_loss = total_loss / max(1, total_samples)
    acc = accuracy_score(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "kappa": kappa,
        "balanced_acc": bal_acc,
        "specificity_macro": macro_specificity,
        "per_class_specificity": per_class_specificity,
        "cm": cm,
        "per_client": per_client_results,
        "targets": all_targets,
        "preds": all_preds,
    }

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

# =========================
# Main
# =========================
def main():
    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    # Per-client datasets
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

    # Loaders
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

    # Model
    global_model = build_model(num_classes).to(DEVICE)
    print(f"Global model created with {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,} trainable params")

    # One personalized head per client
    client_head_states = {
        client: copy.deepcopy(global_model.head.state_dict())
        for client in CLIENT_NAMES
    }

    round_results = []
    history = defaultdict(list)
    best_val_loss = float("inf")
    best_rep_state = None

    start_time = time.time()

    # Class weights for global evaluation across all clients
    combined_targets = []
    for ds in train_datasets:
        combined_targets.extend([s[1] for s in ds.samples])

    combined_counts = np.bincount(combined_targets, minlength=num_classes).astype(np.float32)
    combined_counts[combined_counts == 0] = 1.0
    combined_weights = combined_counts.sum() / combined_counts
    combined_weights = combined_weights / np.mean(combined_weights)
    combined_weights = torch.tensor(combined_weights, dtype=torch.float32).to(DEVICE)
    global_eval_criterion = nn.CrossEntropyLoss(weight=combined_weights)

    for r in range(COMM_ROUNDS):
        print("\n" + "=" * 60)
        print(f"COMM ROUND {r + 1}/{COMM_ROUNDS}")
        print("=" * 60)

        local_models = {}
        local_weights = []
        round_train_losses = []
        round_train_accs = []
        round_summary = {"round": r + 1}

        # -------- Local training --------
        for client_name in CLIENT_NAMES:
            print(f"\n[Client {client_name}]")

            local_model = copy.deepcopy(global_model).to(DEVICE)
            load_head_state(local_model, client_head_states[client_name])

            client_train_ds = train_loaders[client_name].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds, num_classes).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            client_epoch_losses = []
            client_epoch_accs = []

            for ep in range(LOCAL_EPOCHS):
                tr_loss, tr_acc, _, _ = run_epoch(
                    local_model,
                    train_loaders[client_name],
                    criterion,
                    optimizer=optimizer,
                    train=True
                )
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(
                local_model,
                val_loaders[client_name],
                criterion,
                optimizer=None,
                train=False
            )
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models[client_name] = local_model
            client_head_states[client_name] = copy.deepcopy(local_model.head.state_dict())

            local_weights.append(len(client_train_ds))
            round_train_losses.append(float(np.mean(client_epoch_losses)))
            round_train_accs.append(float(np.mean(client_epoch_accs)))

            round_summary[f"{client_name}_train_loss"] = float(np.mean(client_epoch_losses))
            round_summary[f"{client_name}_train_acc"] = float(np.mean(client_epoch_accs))
            round_summary[f"{client_name}_localval_loss"] = float(val_loss)
            round_summary[f"{client_name}_localval_acc"] = float(val_acc)

        total_train_size = sum(local_weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        norm_weights = [w / total_train_size for w in local_weights]

        # -------- FedRep aggregation --------
        print("\nAggregating shared representation (FedRep)...")

        # Use CPU copies only for averaging so we do not disturb the GPU models.
        cpu_models_for_avg = [copy.deepcopy(local_models[c]).cpu() for c in CLIENT_NAMES]
        avg_state = average_representation_state_dict(cpu_models_for_avg, norm_weights)

        # Update global model representation on DEVICE
        load_representation(global_model, avg_state)
        global_model = global_model.to(DEVICE)

        if best_rep_state is None:
            best_rep_state = copy.deepcopy(global_model.representation.state_dict())

        # Broadcast averaged representation back to the live client models on DEVICE
        for client_name in CLIENT_NAMES:
            load_representation(local_models[client_name], avg_state)
            local_models[client_name] = local_models[client_name].to(DEVICE)

        # -------- Global validation using personalized client models --------
        print("\nGlobal validation on combined val sets using personalized heads...")
        global_val_result = evaluate_fedrep_across_clients(
            local_models,
            val_loaders,
            class_names,
            global_eval_criterion
        )
        print(
            f"Global combined val | Loss: {global_val_result['loss']:.4f} | "
            f"Acc: {global_val_result['accuracy']:.4f} | "
            f"F1: {global_val_result['f1_macro']:.4f}"
        )

        if global_val_result["loss"] < best_val_loss:
            best_val_loss = global_val_result["loss"]
            best_rep_state = copy.deepcopy(global_model.representation.state_dict())
            torch.save(
                {
                    "representation_state": best_rep_state,
                    "client_head_states": client_head_states,
                    "class_names": class_names,
                    "client_names": CLIENT_NAMES
                },
                os.path.join(OUTPUT_DIR, MODEL_NAME)
            )
            print("Saved best FedRep model.")

        # -------- Global test --------
        print("\nGlobal TEST on combined test (personalized heads)")
        global_test_result = evaluate_fedrep_across_clients(
            local_models,
            test_loaders,
            class_names,
            global_eval_criterion
        )

        print(
            f"Global combined test | Loss: {global_test_result['loss']:.4f} | "
            f"Acc: {global_test_result['accuracy']:.4f} | "
            f"F1: {global_test_result['f1_macro']:.4f} | "
            f"Kappa: {global_test_result['kappa']:.4f}"
        )

        round_summary["global_val_loss"] = float(global_val_result["loss"])
        round_summary["global_val_acc"] = float(global_val_result["accuracy"])
        round_summary["global_test_loss"] = float(global_test_result["loss"])
        round_summary["global_test_acc"] = float(global_test_result["accuracy"])
        round_summary["global_test_f1"] = float(global_test_result["f1_macro"])
        round_summary["global_test_kappa"] = float(global_test_result["kappa"])

        # Save combined per-class test metrics
        cm = global_test_result["cm"]
        per_prec = precision_score(
            global_test_result["targets"],
            global_test_result["preds"],
            average=None,
            labels=list(range(num_classes)),
            zero_division=0
        )
        per_rec = recall_score(
            global_test_result["targets"],
            global_test_result["preds"],
            average=None,
            labels=list(range(num_classes)),
            zero_division=0
        )
        per_f1 = f1_score(
            global_test_result["targets"],
            global_test_result["preds"],
            average=None,
            labels=list(range(num_classes)),
            zero_division=0
        )
        tp = np.diag(cm).astype(float)
        support = cm.sum(axis=1).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        per_spec = np.divide(tn, (tn + fp), out=np.zeros_like(tn), where=(tn + fp) != 0)
        per_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)

        combined_rows = []
        for ci, cname in enumerate(class_names):
            combined_rows.append({
                "class": cname,
                "support": int(support[ci]),
                "correct": int(tp[ci]),
                "acc": float(per_acc[ci]),
                "prec": float(per_prec[ci]),
                "rec": float(per_rec[ci]),
                "f1": float(per_f1[ci]),
                "spec": float(per_spec[ci]),
            })

        combined_rows.append({
            "class": "macro",
            "support": int(cm.sum()),
            "correct": int(tp.sum()),
            "acc": float(global_test_result["balanced_acc"]),
            "prec": float(global_test_result["precision_macro"]),
            "rec": float(global_test_result["recall_macro"]),
            "f1": float(global_test_result["f1_macro"]),
            "spec": float(np.mean(per_spec)),
        })

        pd.DataFrame(combined_rows).to_csv(
            os.path.join(OUTPUT_DIR, f"combined_test_metrics_round{r + 1}.csv"),
            index=False
        )
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            os.path.join(OUTPUT_DIR, f"combined_confusion_matrix_round{r + 1}.csv")
        )

        # -------- Per-client test evaluation --------
        print("\nIndividual client tests after this round")
        for client_name in CLIENT_NAMES:
            client_result = evaluate_loader(
                local_models[client_name],
                test_loaders[client_name],
                global_eval_criterion,
                class_names,
                title_prefix=f"{client_name}_round_{r + 1}",
                save_dir=OUTPUT_DIR,
                save_cm=True
            )

            round_summary[f"{client_name}_test_loss"] = float(client_result["loss"])
            round_summary[f"{client_name}_test_acc"] = float(client_result["accuracy"])
            round_summary[f"{client_name}_test_f1"] = float(client_result["f1_macro"])
            round_summary[f"{client_name}_test_kappa"] = float(client_result["kappa"])

        # -------- Save checkpoint --------
        ckpt = {
            "round": r + 1,
            "representation_state": copy.deepcopy(global_model.representation.state_dict()),
            "client_head_states": copy.deepcopy(client_head_states),
            "global_val_result": global_val_result,
            "global_test_result": global_test_result,
            "client_names": CLIENT_NAMES,
            "class_names": class_names,
        }
        ckpt_path = os.path.join(OUTPUT_DIR, f"fedrep_round_{r + 1}.pth")
        torch.save(ckpt, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # -------- Save round summary --------
        round_results.append(round_summary)
        pd.DataFrame(round_results).to_csv(os.path.join(OUTPUT_DIR, "fedrep_round_results.csv"), index=False)
        with open(os.path.join(OUTPUT_DIR, "fedrep_round_results.json"), "w") as f:
            json.dump(round_results, f, indent=2)

        # -------- Update plots --------
        history["round"].append(r + 1)
        history["train_loss"].append(float(np.mean(round_train_losses)))
        history["train_acc"].append(float(np.mean(round_train_accs)))
        history["val_loss"].append(float(global_val_result["loss"]))
        history["val_acc"].append(float(global_val_result["accuracy"]))
        history["global_test_acc"].append(float(global_test_result["accuracy"]))

        plot_round_curves(history, OUTPUT_DIR)

        print(
            f"\n[ROUND {r + 1}] "
            f"Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | "
            f"Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f} | "
            f"Global Test Acc: {history['global_test_acc'][-1]:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    if best_rep_state is not None:
        load_representation(global_model, best_rep_state)

    final_model_path = os.path.join(OUTPUT_DIR, "global_final_fedrep.pth")
    torch.save(
        {
            "representation_state": global_model.representation.state_dict(),
            "client_head_states": client_head_states,
            "class_names": class_names,
            "client_names": CLIENT_NAMES
        },
        final_model_path
    )

    print("\nSaved outputs to:")
    print(final_model_path)
    print(os.path.join(OUTPUT_DIR, "fedrep_round_results.csv"))
    print(os.path.join(OUTPUT_DIR, "fedrep_round_results.json"))

if __name__ == "__main__":
    main()