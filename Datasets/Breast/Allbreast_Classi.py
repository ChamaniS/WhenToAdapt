import os
import time
import copy
import random
import json
import csv
import numpy as np
from collections import defaultdict

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
    cohen_kappa_score
)
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
SEED = 42
DATA_ROOT = r"C:\Users\csj5\Projects\Data\Breasttumor_classi_renamed"
OUTPUT_DIR = "breast_tumor_client"
MODEL_NAME = "efficientnet_b0_breast_tumor.pth"

BATCH_SIZE = 4
NUM_EPOCHS = 1
LR = 1e-4
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES =["BUSBRA", "BUS", "BUSI", "UDIAT"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    Make sure all ImageFolder datasets have the same class order.
    This is important when merging multiple clients.
    """
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
    """
    Build a list of ImageFolder datasets, one per client, for the requested split.
    """
    ds_list = []
    valid_clients = []

    for client in CLIENT_NAMES:
        split_dir = paths_dict[client][split]
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        ds = datasets.ImageFolder(split_dir, transform=transform)
        ds_list.append(ds)
        valid_clients.append(client)

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
    model = efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def compute_specificity_from_cm(cm):
    """
    Multiclass specificity for each class using one-vs-rest:
    specificity_i = TN_i / (TN_i + FP_i)
    """
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

def run_epoch(model, loader, criterion, optimizer=None, train=True):
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
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
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

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds

def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

    cm = confusion_matrix(targets, preds)
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    print(f"\n=== {title_prefix.upper()} Results ===")
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

    # Save confusion matrix plot
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

# =========================
# Data
# =========================
paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

# Build per-client datasets for each split
train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

# Combined datasets
train_ds = build_combined_dataset(train_datasets)
val_ds = build_combined_dataset(val_datasets)
test_ds_all = build_combined_dataset(test_datasets)

num_classes = len(class_names)

print("Classes:", class_names)
print("Train samples (all clients):", count_samples(train_ds))
print("Val samples   (all clients):", count_samples(val_ds))
print("Test samples   (all clients):", count_samples(test_ds_all))

for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
    print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

test_loader_all = DataLoader(
    test_ds_all,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

# Separate test loaders for each client
test_loaders_client = {}
for client, ds in zip(CLIENT_NAMES, test_datasets):
    test_loaders_client[client] = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

# =========================
# Model
# =========================
model = build_model(num_classes).to(DEVICE)

# =========================
# Loss / Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# =========================
# Training
# =========================
best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
history = defaultdict(list)
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 40)

    train_loss, train_acc, _, _ = run_epoch(
        model, train_loader, criterion, optimizer=optimizer, train=True
    )

    val_loss, val_acc, _, _ = run_epoch(
        model, val_loader, criterion, optimizer=None, train=False
    )

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
        print("Saved best model.")

elapsed = time.time() - start_time
print(f"\nTraining finished in {elapsed / 60:.2f} minutes")

# Load best weights
model.load_state_dict(best_model_wts)

# =========================
# Plot curves
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=300)
plt.close()

# =========================
# Testing
# =========================
all_results = []

# 1) Test on all clients together
result_all = evaluate_loader(
    model,
    test_loader_all,
    criterion,
    class_names,
    title_prefix="all_clients_test",
    save_dir=OUTPUT_DIR
)
all_results.append(result_all)

# 2) Test each client separately
for client in CLIENT_NAMES:
    result_client = evaluate_loader(
        model,
        test_loaders_client[client],
        criterion,
        class_names,
        title_prefix=f"{client}_test",
        save_dir=OUTPUT_DIR
    )
    all_results.append(result_client)

# Save metrics to CSV and JSON
save_metrics_csv(all_results, os.path.join(OUTPUT_DIR, "test_metrics.csv"))
with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved metrics to:")
print(os.path.join(OUTPUT_DIR, "test_metrics.csv"))
print(os.path.join(OUTPUT_DIR, "test_metrics.json"))