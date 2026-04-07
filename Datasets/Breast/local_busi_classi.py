import os
import time
import copy
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
DATA_ROOT = r"C:\Users\csj5\Projects\Data\Breasttumor_classi\BUSI"
OUTPUT_DIR = "breast_tumor_client"
MODEL_NAME = "efficientnet_b0_breast_tumor.pth"

BATCH_SIZE = 4
NUM_EPOCHS = 120
LR = 1e-4
NUM_WORKERS = 0
IMG_SIZE = 224

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
# Data
# =========================
train_dir = os.path.join(DATA_ROOT, "train")
val_dir = os.path.join(DATA_ROOT, "val")
test_dir = os.path.join(DATA_ROOT, "test")

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

class_names = train_ds.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
print("Test samples:", len(test_ds))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================
# Model
# =========================
def build_model(num_classes):
    model = efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

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
# Helpers
# =========================
def compute_specificity_from_cm(cm):
    """
    Multiclass specificity for each class using one-vs-rest:
    specificity_i = TN_i / (TN_i + FP_i)
    Returns:
        per_class_specificity: list
        macro_specificity: float
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
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

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

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds

# =========================
# Training
# =========================
best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())

history = defaultdict(list)
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
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
print(f"\nTraining finished in {elapsed/60:.2f} minutes")

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
# plt.show()

# =========================
# Test evaluation
# =========================
test_loss, test_acc, test_targets, test_preds = run_epoch(
    model, test_loader, criterion, optimizer=None, train=False
)

cm = confusion_matrix(test_targets, test_preds)

test_precision_macro = precision_score(test_targets, test_preds, average="macro", zero_division=0)
test_recall_macro = recall_score(test_targets, test_preds, average="macro", zero_division=0)
test_f1_macro = f1_score(test_targets, test_preds, average="macro", zero_division=0)
test_kappa = cohen_kappa_score(test_targets, test_preds)

per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

print("\n=== Test Results ===")
print(f"Test Loss       : {test_loss:.4f}")
print(f"Test Accuracy   : {test_acc:.4f}")
print(f"Test Precision  : {test_precision_macro:.4f}")
print(f"Test Recall     : {test_recall_macro:.4f}")
print(f"Test F1-score   : {test_f1_macro:.4f}")
print(f"Test Kappa      : {test_kappa:.4f}")
print(f"Test Specificity: {macro_specificity:.4f}")

print("\nPer-class Specificity:")
for idx, cls_name in enumerate(class_names):
    print(f"{cls_name:12s}: {per_class_specificity[idx]:.4f}")

print("\nClassification Report:")
print(classification_report(test_targets, test_preds, target_names=class_names, digits=4, zero_division=0))

print("Confusion Matrix:")
print(cm)

# Save confusion matrix as image
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
# plt.show()