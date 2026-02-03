# train.py (top portion — replace your current header + CONFIG with this)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, balanced_accuracy_score
)
import random
from typing import List
import argparse

# import the helpers from your dataset_utils.py
from dataset_utils import make_dataloaders, compute_class_weights_from_dataset, make_weighted_sampler_from_dataset

# ----------------------------
#                CONFIG
# ----------------------------
# IMPORTANT: DATA_ROOT must point to the folder that contains train/, val/, test/
# Example Windows absolute path (raw string) OR use forward slashes:
#   r"xxxxx\Projects\Data\Tuberculosis_Data\Shenzhen_split"
#
# Replace the path below with your actual split root (not the inner archive images folder).
DATA_ROOT = r"xxxxx\Projects\Data\Tuberculosis_Data\Shenzhen"   # <- set to your split-root containing train/val/test
OUTPUT_DIR = r"./outputs_tb"
ARCH = "densenet121"                   # e.g. "densenet121", "densenet169", "swin_base_patch4_window7_224", ...
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_FOCAL = True
FOCAL_GAMMA = 2.0
FOCAL_USE_CLASS_WEIGHTS = True
USE_AMP = True
WORKERS = 4
PRETRAINED = True
DROPOUT_P = 0.5
SEED = 42
PIN_MEMORY = True
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ----------------------------
#           HELPERS
# ----------------------------
def add_dropout_head(model, num_classes, p=0.5):
    """
    Replace/reset the classifier head with: Dropout(p) -> Linear(in_features, num_classes)
    Works for many timm models and torchvision models.
    """
    # timm common
    if hasattr(model, "get_classifier"):
        # timm models: use get_classifier() to inspect, model.reset_classifier()
        try:
            in_ch = model.get_classifier().in_features
            model.reset_classifier(0)
            model.classifier = nn.Sequential(nn.Dropout(p), nn.Linear(in_ch, num_classes))
            return model
        except Exception:
            pass
    # torchvision DenseNet uses .classifier
    if hasattr(model, "classifier"):
        try:
            in_ch = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p), nn.Linear(in_ch, num_classes))
            return model
        except Exception:
            pass
    # common fc
    if hasattr(model, "fc"):
        try:
            in_ch = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p), nn.Linear(in_ch, num_classes))
            return model
        except Exception:
            pass
    # head fallback
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        try:
            in_ch = model.head.in_features
            model.head = nn.Sequential(nn.Dropout(p), nn.Linear(in_ch, num_classes))
            return model
        except Exception:
            pass
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
#           Focal loss
# ----------------------------
def focal_loss(outputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    """
    outputs: raw logits (N, C)
    targets: long (N,)
    alpha: tensor/list of per-class scalars or None
    """
    ce = nn.CrossEntropyLoss(reduction='none')
    loss_ce = ce(outputs, targets)                  # shape (N,)
    # compute pt (probability of true class)
    with torch.no_grad():
        probs = torch.softmax(outputs, dim=1)
        pt = probs[range(len(targets)), targets]    # shape (N,)
    loss = ((1 - pt) ** gamma) * loss_ce
    if alpha is not None:
        a = alpha[targets].to(outputs.device)
        loss = a * loss
    return loss.mean() if reduction == 'mean' else loss.sum()

# ----------------------------
#      Metric helpers
# ----------------------------
def compute_epoch_metrics(all_labels: List[int], all_preds: List[int], class_names: List[str]):
    labels_arr = np.array(all_labels, dtype=int)
    preds_arr = np.array(all_preds, dtype=int)
    num_classes = len(class_names)
    if labels_arr.size == 0:
        # return zeros if nothing
        return {
            'confusion_matrix': np.zeros((num_classes, num_classes), dtype=int),
            'per_class_acc': np.zeros(num_classes),
            'per_class_spec': np.zeros(num_classes),
            'per_class_prec': np.zeros(num_classes),
            'per_class_rec': np.zeros(num_classes),
            'per_class_f1': np.zeros(num_classes),
            'macro_prec': 0.0,
            'macro_rec': 0.0,
            'macro_f1': 0.0,
            'balanced_acc': 0.0,
            'cohen_kappa': 0.0,
            'per_class_correct_counts': np.zeros(num_classes, dtype=int),
            'per_class_totals': np.zeros(num_classes, dtype=int),
            'per_class_correct_frac': np.zeros(num_classes, dtype=float)
        }
    cm = confusion_matrix(labels_arr, preds_arr, labels=list(range(num_classes)))
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - (tp + fp + fn)
    per_class_acc = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
    per_class_spec = np.divide(tn, (tn + fp), out=np.zeros_like(tn), where=(tn + fp) != 0)
    per_class_prec = precision_score(labels_arr, preds_arr, labels=list(range(num_classes)), average=None, zero_division=0)
    per_class_rec = recall_score(labels_arr, preds_arr, labels=list(range(num_classes)), average=None, zero_division=0)
    per_class_f1 = f1_score(labels_arr, preds_arr, labels=list(range(num_classes)), average=None, zero_division=0)
    macro_prec = precision_score(labels_arr, preds_arr, average='macro', zero_division=0)
    macro_rec = recall_score(labels_arr, preds_arr, average='macro', zero_division=0)
    macro_f1 = f1_score(labels_arr, preds_arr, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(labels_arr, preds_arr)
    kappa = cohen_kappa_score(labels_arr, preds_arr)
    per_class_correct_counts = tp.astype(int)
    per_class_totals = cm.sum(axis=1).astype(int)
    per_class_correct_frac = np.divide(per_class_correct_counts, per_class_totals, out=np.zeros_like(per_class_correct_counts, dtype=float), where=per_class_totals != 0)
    return {
        'confusion_matrix': cm,
        'per_class_acc': per_class_acc,
        'per_class_spec': per_class_spec,
        'per_class_prec': per_class_prec,
        'per_class_rec': per_class_rec,
        'per_class_f1': per_class_f1,
        'macro_prec': macro_prec,
        'macro_rec': macro_rec,
        'macro_f1': macro_f1,
        'balanced_acc': balanced_acc,
        'cohen_kappa': kappa,
        'per_class_correct_counts': per_class_correct_counts,
        'per_class_totals': per_class_totals,
        'per_class_correct_frac': per_class_correct_frac
    }

def pretty_print_full_metrics(metrics, class_names, prefix=""):
    num_classes = len(class_names)
    print(f"\n{prefix} Per-class correctly-classified (correct / total -> frac):")
    correct_counts = metrics.get('per_class_correct_counts', np.zeros(num_classes, dtype=int))
    totals = metrics.get('per_class_totals', np.zeros(num_classes, dtype=int))
    fracs = metrics.get('per_class_correct_frac', np.zeros(num_classes, dtype=float))
    for i, cname in enumerate(class_names):
        print(f"  {cname:15s} | {correct_counts[i]:4d} / {totals[i]:4d} -> {fracs[i]*100:6.2f}%")
    print(f"\n{prefix} Per-class metrics:")
    for i, cname in enumerate(class_names):
        acc = metrics['per_class_acc'][i] if metrics['per_class_acc'].size else 0.0
        prec = metrics['per_class_prec'][i] if metrics['per_class_prec'].size else 0.0
        rec = metrics['per_class_rec'][i] if metrics['per_class_rec'].size else 0.0
        f1 = metrics['per_class_f1'][i] if metrics['per_class_f1'].size else 0.0
        spec = metrics['per_class_spec'][i] if metrics['per_class_spec'].size else 0.0
        print(f"  {cname:15s} | acc: {acc*100:6.2f}% | prec: {prec*100:6.2f}% | rec: {rec*100:6.2f}% | f1: {f1*100:6.2f}% | spec: {spec*100:6.2f}%")
    print(f"\n{prefix} Macro / mean metrics:")
    print(f"  Mean class accuracy (balanced acc): {metrics['balanced_acc']*100:.2f}%")
    print(f"  Macro precision: {metrics['macro_prec']*100:.2f}%")
    print(f"  Macro recall   : {metrics['macro_rec']*100:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']*100:.2f}%")
    mean_spec = np.mean(metrics['per_class_spec']) if metrics['per_class_spec'].size else 0.0
    print(f"  Mean specificity: {mean_spec*100:.2f}%")
    print(f"  Cohen's kappa  : {metrics['cohen_kappa']:.4f}\n")

# ----------------------------
#        Data loaders
# ----------------------------
def make_dataloaders(root_dir, batch_size=16, image_size=224, workers=4, pin_memory=True):
    """
    Expects root_dir/train, root_dir/val, root_dir/test with class subfolders.
    Returns dict of DataLoaders, sizes dict, class_names list, and train_dataset object.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        normalize
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    data_paths = {}
    for split in ('train','val','test'):
        p = os.path.join(root_dir, split)
        if not os.path.isdir(p):
            raise ValueError(f"Expected directory for split {split}: {p}")
        data_paths[split] = p

    train_ds = datasets.ImageFolder(data_paths['train'], transform=train_tf)
    val_ds = datasets.ImageFolder(data_paths['val'], transform=val_tf)
    test_ds = datasets.ImageFolder(data_paths['test'], transform=val_tf)

    class_names = train_ds.classes

    # dataloaders (train loader created later if using sampler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=pin_memory)

    sizes = {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders, sizes, class_names, train_ds

def compute_class_weights_from_dataset(dataset):
    """
    dataset: torchvision.datasets.ImageFolder
    returns torch.Tensor of shape (num_classes,)
    weight per class = total_count / (num_classes * count[class])  (inverse frequency scaled)
    """
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float32)
    total = counts.sum()
    # avoid zero division
    weights = total / (counts + 1e-8)
    # normalize to mean 1
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

# ----------------------------
#   train / eval / test loops
# ----------------------------
def train_one_epoch(model, loader, criterion, opt, device, scaler=None,
                    class_names=None, use_focal=False, focal_alpha=None, focal_gamma=2.0):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(x)
            loss = focal_loss(out, y, alpha=focal_alpha, gamma=focal_gamma) if use_focal else criterion(out, y)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        _, pred = out.max(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        running_loss += loss.item() * y.size(0)
        all_preds.extend(pred.cpu().numpy()); all_labels.extend(y.cpu().numpy())
        pbar.set_postfix(loss=running_loss / total, acc=correct/total)
    metrics = compute_epoch_metrics(all_labels, all_preds, class_names)
    overall_loss = (running_loss/total) if total>0 else 0.0
    overall_acc = (correct/total) if total>0 else 0.0
    return overall_loss, overall_acc, metrics

def eval_model(model, loader, criterion, device, class_names=None,
               use_focal=False, focal_alpha=None, focal_gamma=2.0):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = focal_loss(out, y, alpha=focal_alpha, gamma=focal_gamma) if use_focal else criterion(out, y)
            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            running_loss += loss.item() * y.size(0)
            all_preds.extend(pred.cpu().numpy()); all_labels.extend(y.cpu().numpy())
            pbar.set_postfix(loss=running_loss / total, acc=correct/total)
    metrics = compute_epoch_metrics(all_labels, all_preds, class_names)
    overall_loss = (running_loss/total) if total>0 else 0.0
    overall_acc = (correct/total) if total>0 else 0.0
    return overall_loss, overall_acc, metrics

# ----------------------------
#            MAIN
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataloaders, sizes, class_names, train_ds = make_dataloaders(
        root_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        workers=WORKERS,
        pin_memory=PIN_MEMORY and (device.type == "cuda")
    )

    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Dataset sizes:", sizes)

    class_weights = compute_class_weights_from_dataset(train_ds).to(device)
    print("Class weights (scaled to mean=1):", class_weights.tolist())

    # Balanced sampler alternative: WeightedRandomSampler on train
    sampler = make_weighted_sampler_from_dataset(train_ds)
    dataloaders['train'] = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                      num_workers=WORKERS, pin_memory=PIN_MEMORY and (device.type == "cuda"))

    # Model creation
    if ARCH.startswith("densenet"):
        print(f"Creating torchvision model {ARCH} (pretrained={PRETRAINED})")
        model = getattr(torchvision.models, ARCH)(pretrained=PRETRAINED)
        # replace classifier head
        if hasattr(model, "classifier"):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(num_ftrs, num_classes))
        elif hasattr(model, "fc"):
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(num_ftrs, num_classes))
        model = model.to(device)
    else:
        print(f"Creating timm model {ARCH} (pretrained={PRETRAINED})")
        model = timm.create_model(ARCH, pretrained=PRETRAINED, num_classes=num_classes)
        model = add_dropout_head(model, num_classes, p=DROPOUT_P)
        model = model.to(device)

    print(f"Model {ARCH} created with {count_parameters(model):,} trainable parameters")

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    focal_alpha = class_weights if (USE_FOCAL and FOCAL_USE_CLASS_WEIGHTS) else None
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and USE_AMP) else None

    log = []
    best_f1, best_state = -1.0, None
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc, train_metrics = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer,
            device, scaler, class_names, USE_FOCAL, focal_alpha, FOCAL_GAMMA
        )
        pretty_print_full_metrics(train_metrics, class_names, prefix="Train")
        val_loss, val_acc, val_metrics = eval_model(
            model, dataloaders['val'], criterion, device, class_names, USE_FOCAL, focal_alpha, FOCAL_GAMMA
        )
        pretty_print_full_metrics(val_metrics, class_names, prefix="Val")

        val_f1 = val_metrics['macro_f1']
        scheduler.step(val_loss)

        print(f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | Val loss={val_loss:.4f}, acc={val_acc:.4f}, val_macroF1={val_f1:.4f}")

        log.append([
            epoch+1,
            train_loss,
            train_acc,
            float(np.mean(train_metrics.get('per_class_correct_frac', np.zeros(num_classes)))),
            val_loss,
            val_acc,
            float(np.mean(val_metrics.get('per_class_correct_frac', np.zeros(num_classes)))),
            val_f1
        ])

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(OUTPUT_DIR, "best_model.pth"))
            print("✅ Saved new best model (macro-F1 improved)")

        # save training log
        csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
        pd.DataFrame(log, columns=[
            "epoch",
            "train_loss",
            "train_acc",
            "train_mean_per_class_correct_frac",
            "val_loss",
            "val_acc",
            "val_mean_per_class_correct_frac",
            "val_macro_f1"
        ]).to_csv(csv_path, index=False)
        print(f"Metrics logged to {csv_path}")

        # Plot curves
        df = pd.read_csv(csv_path)
        epochs = df["epoch"]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, df["train_loss"], label="Train Loss")
        plt.plot(epochs, df["val_loss"], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, df["train_acc"] * 100, label="Train Acc")
        plt.plot(epochs, df["val_acc"] * 100, label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curve")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "acc_curve.png"), dpi=200)
        plt.close()

    total_minutes = (time.time() - start_time) / 60.0
    print(f"\nTraining complete in {total_minutes:.1f} min. Best val macro-F1={best_f1:.4f}")

    if "train_mean_per_class_correct_frac" in df.columns and "val_mean_per_class_correct_frac" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, df["train_mean_per_class_correct_frac"] * 100, label="Train mean per-class correct (%)")
        plt.plot(epochs, df["val_mean_per_class_correct_frac"] * 100, label="Val mean per-class correct (%)")
        plt.xlabel("Epoch"); plt.ylabel("Mean per-class correct (%)"); plt.title("Mean Per-Class Correct Fraction")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "mean_per_class_correct_frac_curve.png"), dpi=200)
        plt.close()

    # Final test evaluation (load best model if exists)
    print("\nRunning final evaluation on TEST set...")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_metrics = eval_model(
        model, dataloaders['test'], criterion, device, class_names, USE_FOCAL, focal_alpha, FOCAL_GAMMA
    )
    print(f"\nTest loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    pretty_print_full_metrics(test_metrics, class_names, prefix="Test")

    # Save test CSV
    test_csv_path = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    cm = test_metrics['confusion_matrix']
    per_class_rows = []
    for i, cname in enumerate(class_names):
        per_class_rows.append({
            "class": cname,
            "count": int(cm[i].sum()),
            "correct": int(test_metrics['per_class_correct_counts'][i]),
            "total": int(test_metrics['per_class_totals'][i]),
            "frac_correct": float(test_metrics['per_class_correct_frac'][i]),
            "acc": float(test_metrics['per_class_acc'][i]),
            "prec": float(test_metrics['per_class_prec'][i]),
            "rec": float(test_metrics['per_class_rec'][i]),
            "f1": float(test_metrics['per_class_f1'][i]),
            "spec": float(test_metrics['per_class_spec'][i])
        })
    df_test = pd.DataFrame(per_class_rows)
    macro_row = {
        "class": "macro",
        "count": int(cm.sum()),
        "correct": int(np.sum(test_metrics['per_class_correct_counts'])),
        "total": int(np.sum(test_metrics['per_class_totals'])),
        "frac_correct": float(np.sum(test_metrics['per_class_correct_counts']) / np.sum(test_metrics['per_class_totals'])) if np.sum(test_metrics['per_class_totals'])>0 else 0.0,
        "acc": test_metrics['balanced_acc'],
        "prec": test_metrics['macro_prec'],
        "rec": test_metrics['macro_rec'],
        "f1": test_metrics['macro_f1'],
        "spec": float(np.mean(test_metrics['per_class_spec']))
    }
    df_test = pd.concat([df_test, pd.DataFrame([macro_row])], ignore_index=True)
    df_test.to_csv(test_csv_path, index=False)
    print(f"Test metrics saved to {test_csv_path}")

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Saved final model to", final_model_path)

    print("✅ Training + plotting + testing finished. All outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
