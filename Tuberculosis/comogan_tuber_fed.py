# fl_train.py
"""
Federated averaging (FedAvg) across 4 clients for TB CXR classification,
with per-client/per-class test printing and visualization (comparison grids).
Adapted to allow inserting CoMoGAN harmonization per-client validation sets.

Fixes applied:
 - recursive image discovery (works when val/ contains class subfolders)
 - robust harmonize_client_validation stub now produces visibly different "harmonized"
   images (autocontrast + contrast/brightness tweak) so comparison grids show differences.
 - make_comparison_grid is debug-friendly and will create a fallback grid from originals if harmonized outputs missing
 - uses non-interactive matplotlib backend and protected save calls
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ensure matplotlib doesn't require an X display
import matplotlib
matplotlib.use("Agg")

# avoid repeated torchvision C++ image extension warnings on Windows
from torchvision import io as tv_io
try:
    tv_io.set_image_backend('PIL')
except Exception:
    pass

import copy
import time
import random
import shutil
from glob import glob
from typing import List, Tuple
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import timm
import torchvision
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG - set these paths
# ----------------------------
CLIENT_ROOTS = [
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Shenzhen",     # <-- update these 4 paths to your local paths
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Montgomery",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\TBX11K",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES = ["Shenzhen", "Montgomery", "TBX11K", "Pakistan"]  # used only for display
OUTPUT_DIR = r"./fl_outputs_comogan"
ARCH = "densenet169"   # or other timm / torchvision model name
PRETRAINED = True
IMG_SIZE = 224
BATCH_SIZE = 4          # local batch size (per client)
WORKERS = 1
LOCAL_EPOCHS = 12      # local epochs per communication round
COMM_ROUNDS = 10        # number of federated rounds
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = False         # set True to use amp when CUDA available
PIN_MEMORY = True
DROPOUT_P = 0.5
SEED = 42
CLASS_NAMES = ["normal", "positive"]  # canonical ordering
# ----------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Utility helpers for file IO + visuals
# ----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def image_list_in_dir(d):
    """
    Recursively collect image file paths under directory d.
    Returns sorted list of full paths. Works when validation folder has class subfolders.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(d):
        return []
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return sorted(out)

def select_val_pairs_for_comparison(original_val_dir, hm_val_dir, n_samples=7):
    """
    Pair original vs harmonized images by basename when possible.
    If basenames don't match, pair by index as fallback.
    """
    orig_list = image_list_in_dir(original_val_dir)
    hm_list = image_list_in_dir(hm_val_dir)
    if len(orig_list) == 0 or len(hm_list) == 0:
        return []
    # map basenames to paths (keep first occurrence)
    orig_map = {}
    for p in orig_list:
        b = os.path.basename(p)
        if b not in orig_map:
            orig_map[b] = p
    hm_map = {}
    for p in hm_list:
        b = os.path.basename(p)
        if b not in hm_map:
            hm_map[b] = p
    common = sorted(set(orig_map.keys()) & set(hm_map.keys()))
    pairs = []
    if len(common) > 0:
        sel = common[:n_samples]
        for fn in sel:
            pairs.append((orig_map[fn], hm_map[fn], fn))
    else:
        # fallback: pair by index
        n = min(len(orig_list), len(hm_list), n_samples)
        for i in range(n):
            fn = os.path.basename(orig_list[i])
            pairs.append((orig_list[i], hm_list[i], fn))
    return pairs

def make_comparison_grid(original_val_dir, hm_val_dir, client_name, out_base, n_samples=7):
    """
    Debug-friendly comparison grid:
      - prints discovered counts
      - if no matched pairs, falls back to original images only (orig vs orig)
      - saves to out_base/ComparisonGrid/comparison_<client_name>.png
    """
    base_dest = ensure_dir(os.path.join(out_base, "ComparisonGrid"))
    print(f"[VIS] orig_val_dir: {original_val_dir}")
    print(f"[VIS] hm_val_dir  : {hm_val_dir}")
    orig_list = image_list_in_dir(original_val_dir)
    hm_list = image_list_in_dir(hm_val_dir)
    print(f"[VIS] found {len(orig_list)} original images, {len(hm_list)} harmonized images")

    fn_pairs = select_val_pairs_for_comparison(original_val_dir, hm_val_dir, n_samples=n_samples)
    print(f"[VIS] matched pairs count: {len(fn_pairs)}")

    # fallback: if no matched pairs but we have original images, create orig vs orig pairs
    if len(fn_pairs) == 0 and len(orig_list) > 0:
        print("[VIS] No matched pairs found; creating fallback pairs using original images only")
        fn_pairs = []
        n = min(len(orig_list), n_samples)
        for i in range(n):
            p = orig_list[i]
            fn = os.path.basename(p)
            fn_pairs.append((p, p, fn))

    if len(fn_pairs) == 0:
        print(f"[VIS] no images available for comparison for {client_name} - nothing will be saved")
        return

    top_imgs = []
    mid_imgs = []
    diff_imgs = []
    titles = []
    for orig_p, hm_p, fn in fn_pairs:
        try:
            orig = np.array(Image.open(orig_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
            hm = np.array(Image.open(hm_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
            if orig.dtype != np.uint8:
                orig = np.clip(orig, 0, 255).astype(np.uint8)
            if hm.dtype != np.uint8:
                hm = np.clip(hm, 0, 255).astype(np.uint8)
            top_imgs.append(orig)
            mid_imgs.append(hm)

            diff = np.abs(hm.astype(np.float32) - orig.astype(np.float32))
            amplified = diff * 3.0
            amplified = np.clip(amplified, 0, 255).astype(np.uint8)
            if amplified.max() < 8:
                if diff.max() > 0:
                    amplified = (diff / (diff.max() + 1e-8) * 255.0).astype(np.uint8)
                else:
                    amplified = np.zeros_like(amplified, dtype=np.uint8)
            diff_imgs.append(amplified)
            titles.append(fn)
        except Exception as e:
            print(f"[VIS] skipping pair {fn} due to read error: {e}")

    n = len(top_imgs)
    if n == 0:
        print(f"[VIS] no readable pairs for {client_name}")
        return

    fig, axs = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    if n == 1:
        # normalize axs shape to (3,1)
        axs = np.array([[axs[0]],[axs[1]],[axs[2]]]) if hasattr(axs, '__len__') else np.array([[axs]])
        axs = axs.reshape(3, 1)

    col_titles = [t[:24] for t in titles]
    for i in range(n):
        axs[0, i].imshow(top_imgs[i]); axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel("Original", fontsize=10)
        axs[0, i].set_title(col_titles[i], fontsize=9)

        axs[1, i].imshow(mid_imgs[i]); axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel("Harmonized", fontsize=10)

        axs[2, i].imshow(diff_imgs[i]); axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_ylabel("Amplified diff", fontsize=10)

    fig.suptitle(f"Harmonized vs Original: {client_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(base_dest, f"comparison_{client_name}.png")
    try:
        plt.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[VIS] Saved comparison grid for {client_name} at {out_png}")
    except Exception as e:
        print("[VIS] failed saving comparison grid:", e)
        try:
            plt.close(fig)
        except Exception:
            pass

# A helper that writes a per-round copy so you can keep historic grids if desired
def save_comparison_grid_round_copy(client_name, out_base, round_idx):
    base_dest = ensure_dir(os.path.join(out_base, "ComparisonGrid"))
    src = os.path.join(base_dest, f"comparison_{client_name}.png")
    if os.path.exists(src):
        dst = os.path.join(base_dest, f"comparison_{client_name}_round{round_idx}.png")
        try:
            shutil.copyfile(src, dst)
            print(f"[VIS] Saved round copy {dst}")
        except Exception as e:
            print("[VIS] failed to copy comparison grid round:", e)

# ----------------------------
# Dataset utilities (PathListDataset + multi-client gather)
# ----------------------------
class PathListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None, loader=default_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

def gather_samples_from_client_split(client_root: str, split: str, class_names: List[str]):
    split_dir = os.path.join(client_root, split)
    if not os.path.isdir(split_dir):
        raise ValueError(f"Missing {split} in {client_root}")
    samples = []
    canon_map = {c.lower(): i for i,c in enumerate(class_names)}
    for cls_folder in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls_folder)
        if not os.path.isdir(cls_path): continue
        key = cls_folder.lower()
        if key not in canon_map:
            print(f"Warning: unknown class folder '{cls_folder}' in {split_dir}; skipping")
            continue
        label = canon_map[key]
        for fn in os.listdir(cls_path):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                samples.append((os.path.join(cls_path, fn), label))
    return samples

def make_multi_client_dataloaders(client_roots, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
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

    train_samples_all, val_samples_all, test_samples_all = [], [], []
    per_client_dataloaders = []
    per_client_test_dsets = {}

    for client_root in client_roots:
        tr = gather_samples_from_client_split(client_root, "train", CLASS_NAMES)
        va = gather_samples_from_client_split(client_root, "val", CLASS_NAMES)
        te = gather_samples_from_client_split(client_root, "test", CLASS_NAMES)
        print(f"[DATA] client {client_root} -> train:{len(tr)} val:{len(va)} test:{len(te)}")
        train_samples_all.extend(tr); val_samples_all.extend(va); test_samples_all.extend(te)
        # per-client dataloaders
        train_ds = PathListDataset(tr, transform=train_tf)
        val_ds = PathListDataset(va, transform=val_tf)
        test_ds = PathListDataset(te, transform=val_tf)
        per_client_dataloaders.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "train_ds": train_ds,
            "client_root": client_root
        })
        per_client_test_dsets[client_root] = test_ds

    combined_train_ds = PathListDataset(train_samples_all, transform=train_tf)
    combined_val_ds = PathListDataset(val_samples_all, transform=val_tf)
    combined_test_ds = PathListDataset(test_samples_all, transform=val_tf)

    dataloaders_combined = {
        "train": DataLoader(combined_train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
        "val": DataLoader(combined_val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
        "test": DataLoader(combined_test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    }
    sizes = {"train": len(combined_train_ds), "val": len(combined_val_ds), "test": len(combined_test_ds)}
    return dataloaders_combined, sizes, CLASS_NAMES, combined_train_ds, per_client_dataloaders, per_client_test_dsets

def compute_class_weights_from_dataset(dataset):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

def make_weighted_sampler_from_dataset(dataset):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    inv_freq = 1.0 / np.maximum(counts, 1.0)
    weights = [float(inv_freq[t]) for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ----------------------------
# Model + training utilities
# ----------------------------
def create_model(num_classes, arch=ARCH, pretrained=PRETRAINED):
    if arch.startswith("densenet") and hasattr(torchvision.models, arch):
        model = getattr(torchvision.models, arch)(pretrained=pretrained)
        # replace classifier
        if hasattr(model, "classifier"):
            in_ch = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_ch, num_classes))
        elif hasattr(model, "fc"):
            in_ch = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_ch, num_classes))
        return model
    else:
        # use timm
        model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
        return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def average_models_weighted(models: List[torch.nn.Module], weights: List[float]):
    if len(models) == 0:
        raise ValueError("No models to average")
    if len(models) != len(weights):
        raise ValueError("models and weights must have same length")
    sum_w = float(sum(weights))
    if sum_w == 0.0:
        raise ValueError("Sum of weights is zero")
    norm_weights = [w / sum_w for w in weights]
    base_sd = models[0].state_dict()
    avg_sd = {}
    with torch.no_grad():
        for k, v0 in base_sd.items():
            acc = torch.zeros_like(v0, dtype=torch.float32, device="cpu")
            for m, w in zip(models, norm_weights):
                vm = m.state_dict()[k].cpu().to(dtype=torch.float32)
                acc += float(w) * vm
            try:
                acc = acc.to(dtype=v0.dtype)
            except Exception:
                acc = acc
            avg_sd[k] = acc
    return avg_sd

# ----------------------------
# Training / evaluation loops
# ----------------------------
def train_local(model, dataloader, criterion, optimizer, device, epochs=LOCAL_EPOCHS, use_amp=False):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type=="cuda") else None
    logs = []
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"LocalTrain ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = model(x)
                loss = criterion(out, y)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss = running_loss / total if total>0 else 0.0, acc = correct/total if total>0 else 0.0)
        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        logs.append((epoch_loss, epoch_acc))
    return logs

@torch.no_grad()
def evaluate_model(model, dataloader, device, criterion=None, return_per_class=False, class_names=None):
    model.eval()
    all_y = []; all_pred = []
    total_loss = 0.0; n = 0
    for x, y in tqdm(dataloader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
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
        "cohen_kappa": float(kappa)
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
            "per_class_support": [int(x) for x in support.tolist()]
        })
    return metrics

# ----------------------------
# Minimal harmonization hook (replace with CoMoGAN call)
# ----------------------------
def harmonize_client_validation(client_root, client_name, out_base):
    """
    This function should run CoMoGAN (or any harmonizer) on the client validation images
    and save results into: out_base/harmonized/{client_name}/val/*.png

    Current robust stub:
      - ensures dst folder exists
      - recursively finds images under client_root/val (including class subfolders)
      - writes "harmonized" images that are visually different (autocontrast + contrast/brightness tweak)
      - overwrites existing files so comparison grids always find outputs

    Replace the body with a call to your CoMoGAN pipeline; return the folder path where harmonized images are saved.
    """
    src_val = os.path.join(client_root, "val")
    dst_base = ensure_dir(os.path.join(out_base, "harmonized", client_name, "val"))
    if not os.path.isdir(src_val):
        print(f"[HARM] no val dir for {client_name} at {src_val}")
        return dst_base

    images = image_list_in_dir(src_val)
    print(f"[HARM] found {len(images)} images to (stub) harmonize for {client_name}")

    for p in images:
        try:
            # load
            img = Image.open(p).convert("RGB")

            # simple deterministic transformation to simulate harmonization:
            # 1) autocontrast to stretch intensities
            img = ImageOps.autocontrast(img, cutoff=0)

            # 2) tweak contrast and brightness slightly (deterministic per-filename)
            seed = sum(bytearray(os.path.basename(p).encode("utf-8"))) % 1000
            # deterministic pseudo-random parameters
            ctr = 0.9 + (seed % 11) * 0.01   # 0.90 - 1.00
            bri = 0.95 + ((seed // 11) % 11) * 0.005  # 0.95 - ~1.0
            # apply
            img = ImageEnhance.Contrast(img).enhance(ctr)
            img = ImageEnhance.Brightness(img).enhance(bri)

            # optional slight color shift (small)
            img_arr = np.array(img).astype(np.float32)
            # add small deterministic channel offsets
            r_off = ((seed % 7) - 3) * 0.6
            g_off = (((seed//7) % 5) - 2) * 0.4
            b_off = (((seed//35) % 3) - 1) * 0.6
            img_arr[..., 0] = np.clip(img_arr[..., 0] + r_off, 0, 255)
            img_arr[..., 1] = np.clip(img_arr[..., 1] + g_off, 0, 255)
            img_arr[..., 2] = np.clip(img_arr[..., 2] + b_off, 0, 255)
            img_out = Image.fromarray(img_arr.astype(np.uint8))

            # save to dst (basename preserved)
            dst = os.path.join(dst_base, os.path.basename(p))
            img_out.save(dst)
        except Exception as e:
            print("[HARM] processing error for", p, ":", e)

    print(f"[HARM] (stub) harmonized validation images for {client_name} -> {dst_base}")
    return dst_base

# ----------------------------
# Federated training main
# ----------------------------
def main():
    print("DEVICE:", DEVICE)
    combined_loaders, combined_sizes, class_names, combined_train_ds, per_client_dataloaders, per_client_test_dsets = make_multi_client_dataloaders(
        CLIENT_ROOTS, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda")
    )
    num_classes = len(class_names)
    print("class names:", class_names)
    client_train_sizes = [len(per_client_dataloaders[i]['train'].dataset) for i in range(len(per_client_dataloaders))]
    total_train = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
    print("client train sizes:", client_train_sizes)

    global_model = create_model(num_classes=num_classes, arch=ARCH, pretrained=PRETRAINED).to(DEVICE)
    print(f"Global model {ARCH} created with {count_parameters(global_model):,} trainable params")

    # Prepare logging
    round_results = []
    global_test_acc_fname = os.path.join(OUTPUT_DIR, "global_test_accuracy_rounds.png")
    global_test_loss_fname = os.path.join(OUTPUT_DIR, "global_test_loss_rounds.png")
    per_client_acc_history = {i: [] for i in range(len(per_client_dataloaders))}
    per_client_loss_history = {i: [] for i in range(len(per_client_dataloaders))}

    for r in range(COMM_ROUNDS):
        print("\n" + "="*60)
        print(f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print("="*60)
        local_models = []
        weights = []
        round_summary = {"round": r+1}
        # Each client trains locally starting from global weights
        for i, client in enumerate(per_client_dataloaders):
            print(f"\n[CLIENT {i}] {CLIENT_NAMES[i]}: local training")
            local_model = copy.deepcopy(global_model)
            train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(train_ds).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            logs = train_local(local_model, client['train'], criterion, optimizer, DEVICE, epochs=LOCAL_EPOCHS, use_amp=USE_AMP)
            last_train_loss, last_train_acc = logs[-1]
            print(f"[CLIENT {i}] last local epoch loss={last_train_loss:.4f}, acc={last_train_acc:.4f}")
            round_summary[f"client{i}_train_loss"] = float(last_train_loss)
            round_summary[f"client{i}_train_acc"] = float(last_train_acc)
            # local validation
            print(f"[CLIENT {i}] local validation")
            local_val_metrics = evaluate_model(local_model, client['val'], DEVICE, criterion=criterion)
            print(f"[CLIENT {i}] local val acc={local_val_metrics.get('accuracy', np.nan):.4f}, loss={local_val_metrics.get('loss', np.nan):.4f}")
            round_summary[f"client{i}_localval_loss"] = float(local_val_metrics.get("loss", np.nan))
            round_summary[f"client{i}_localval_acc"] = float(local_val_metrics.get("accuracy", np.nan))
            # move local model to CPU for aggregation to save GPU memory
            local_models.append(local_model.cpu())
            w = float(client_train_sizes[i]) / float(total_train)
            weights.append(w)
            print(f"[CLIENT {i}] aggregation weight: {w:.4f}")

        # Aggregate via weighted FedAvg
        print("\nAggregating local models (FedAvg weighted)")
        avg_state = average_models_weighted(local_models, weights)
        avg_state_on_device = {k: v.to(DEVICE) for k, v in avg_state.items()}
        global_model.load_state_dict(avg_state_on_device)
        global_model.to(DEVICE)

        # Global validation on combined val (concatenate client val datasets)
        print("\nGlobal validation on combined val sets...")
        combined_val_dsets = [per_client_dataloaders[i]['val'].dataset for i in range(len(per_client_dataloaders))]
        combined_val = ConcatDataset(combined_val_dsets)
        combined_val_loader = DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda"))

        combined_train_targets = []
        for i in range(len(per_client_dataloaders)):
            combined_train_targets.extend([s[1] for s in per_client_dataloaders[i]['train'].dataset.samples])
        counts = np.bincount(combined_train_targets, minlength=num_classes).astype(np.float32)
        counts[counts==0] = 1.0
        weights_arr = 1.0 / counts
        weights_arr = weights_arr * (len(weights_arr) / weights_arr.sum())
        combined_class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
        combined_criterion = nn.CrossEntropyLoss(weight=combined_class_weights)

        global_val_metrics = evaluate_model(global_model, combined_val_loader, DEVICE, criterion=combined_criterion)
        print("Global combined val metrics:", global_val_metrics)
        round_summary["global_val_loss"] = float(global_val_metrics.get("loss", np.nan))
        round_summary["global_val_acc"] = float(global_val_metrics.get("accuracy", np.nan))

        # Global test on combined test set
        print("\nGlobal TEST on combined test (all clients)")
        combined_test_dsets = [per_client_dataloaders[i]['test'].dataset for i in range(len(per_client_dataloaders))]
        combined_test = ConcatDataset(combined_test_dsets)
        combined_test_loader = DataLoader(combined_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda"))
        global_test_metrics = evaluate_model(global_model, combined_test_loader, DEVICE, criterion=combined_criterion, return_per_class=True, class_names=class_names)
        print("Global combined TEST metrics summary:", {k: global_test_metrics.get(k) for k in ["accuracy","loss","f1_macro","precision_macro","recall_macro","balanced_acc","cohen_kappa"]})
        round_summary["global_test_loss"] = float(global_test_metrics.get("loss", np.nan))
        round_summary["global_test_acc"] = float(global_test_metrics.get("accuracy", np.nan))

        # Print and save combined per-class CSV (unchanged)
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
                "support": int(cm.sum()) if cm is not None else sum([r["support"] for r in combined_rows]) if combined_rows else 0,
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
                pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(OUTPUT_DIR, "combined_confusion_matrix.csv"))
            print(f"Saved combined per-class test metrics CSV to: {combined_csv}")
        except Exception as e:
            print("Warning saving/printing combined per-class metrics:", e)

        # --- per-client global TEST and per-class printing + visuals ---
        per_client_test_metrics = []
        for i, client in enumerate(per_client_dataloaders):
            print(f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set")
            client_train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)
            cl_metrics = evaluate_model(global_model, client['test'], DEVICE, criterion=client_criterion, return_per_class=True, class_names=class_names)

            # compute mean specificity across classes (if per-class specificity present)
            mean_spec = None
            if "per_class_specificity" in cl_metrics:
                specs = cl_metrics.get("per_class_specificity", [])
                if len(specs) > 0:
                    mean_spec = float(np.mean(specs))

            acc = cl_metrics.get("accuracy", np.nan)
            prec = cl_metrics.get("precision_macro", np.nan)
            rec = cl_metrics.get("recall_macro", np.nan)
            f1 = cl_metrics.get("f1_macro", np.nan)
            kappa = cl_metrics.get("cohen_kappa", np.nan)
            if mean_spec is None and "per_class_specificity" in cl_metrics:
                mean_spec = float(np.mean(cl_metrics.get("per_class_specificity", [np.nan])))
            print(f"[CLIENT {i}] Summary metrics:")
            print(f"  Accuracy       : {acc:.4f}")
            print(f"  Precision (mac): {prec:.4f}")
            print(f"  Recall (mac)   : {rec:.4f}")
            print(f"  F1 (mac)       : {f1:.4f}")
            if mean_spec is not None:
                print(f"  Mean Specificity: {mean_spec:.4f}")
            else:
                print(f"  Mean Specificity: n/a")
            print(f"  Cohen's kappa  : {kappa:.4f}")

            # Per-class table
            if "per_class_precision" in cl_metrics:
                print(f"\n  Per-class metrics (order = {class_names}):")
                header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
                print("    " + "{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))
                cm = cl_metrics.get("confusion_matrix", None)
                tp_counts = np.diag(cm).astype(int) if cm is not None else [0]*len(class_names)
                supports = cm.sum(axis=1).astype(int) if cm is not None else [0]*len(class_names)
                precisions = cl_metrics.get("per_class_precision", [])
                recalls = cl_metrics.get("per_class_recall", [])
                f1s = cl_metrics.get("per_class_f1", [])
                specs = cl_metrics.get("per_class_specificity", [])
                accs = cl_metrics.get("per_class_accuracy", [])
                for ci, cname in enumerate(class_names):
                    s = supports[ci] if ci < len(supports) else 0
                    ccount = tp_counts[ci] if ci < len(tp_counts) else 0
                    acc_val = accs[ci] if ci < len(accs) else np.nan
                    pval = precisions[ci] if ci < len(precisions) else np.nan
                    rval = recalls[ci] if ci < len(recalls) else np.nan
                    fval = f1s[ci] if ci < len(f1s) else np.nan
                    sval = specs[ci] if ci < len(specs) else np.nan
                    print(f"    {cname:12s} {s:8d} {ccount:8d} {acc_val:8.4f} {pval:8.4f} {rval:8.4f} {fval:8.4f} {sval:8.4f}")

            # Save per-client metrics CSV (match format used for combined)
            try:
                cmc = cl_metrics.get("confusion_matrix", None)
                rows=[]
                if cmc is not None:
                    per_support = cl_metrics.get("per_class_support", [])
                    per_correct = cl_metrics.get("per_class_correct", [])
                    per_accs = cl_metrics.get("per_class_accuracy", [])
                    per_precs = cl_metrics.get("per_class_precision", [])
                    per_recs = cl_metrics.get("per_class_recall", [])
                    per_f1s = cl_metrics.get("per_class_f1", [])
                    per_specs = cl_metrics.get("per_class_specificity", [])
                    for i_c, cname in enumerate(class_names):
                        rows.append({
                            "class": cname,
                            "support": int(per_support[i_c]) if i_c < len(per_support) else int(cmc[i_c].sum()) if cmc is not None else 0,
                            "correct": int(per_correct[i_c]) if i_c < len(per_correct) else int(np.diag(cmc)[i_c]) if cmc is not None else 0,
                            "acc": float(per_accs[i_c]) if i_c < len(per_accs) else np.nan,
                            "prec": float(per_precs[i_c]) if i_c < len(per_precs) else np.nan,
                            "rec": float(per_recs[i_c]) if i_c < len(per_recs) else np.nan,
                            "f1": float(per_f1s[i_c]) if i_c < len(per_f1s) else np.nan,
                            "spec": float(per_specs[i_c]) if i_c < len(per_specs) else np.nan
                        })
                macro = {
                    "class": "macro",
                    "support": int(cmc.sum()) if cmc is not None else 0,
                    "correct": int(np.sum([r["correct"] for r in rows])) if rows else 0,
                    "acc": cl_metrics.get("balanced_acc", np.nan),
                    "prec": cl_metrics.get("precision_macro", np.nan),
                    "rec": cl_metrics.get("recall_macro", np.nan),
                    "f1": cl_metrics.get("f1_macro", np.nan),
                    "spec": float(np.mean(cl_metrics.get("per_class_specificity", [np.nan]))) if "per_class_specificity" in cl_metrics else np.nan
                }
                if rows:
                    rows.append(macro)
                    dfc = pd.DataFrame(rows)
                    safe_name = CLIENT_NAMES[i].replace(" ", "_")
                    client_csv = os.path.join(OUTPUT_DIR, f"{safe_name}_test_metrics_round{r+1}.csv")
                    dfc.to_csv(client_csv, index=False)
                    if cmc is not None:
                        pd.DataFrame(cmc, index=class_names, columns=class_names).to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_confusion_matrix_round{r+1}.csv"))
                    print(f"Saved per-client metrics CSV to {client_csv}")
            except Exception as e:
                print("Warning saving per-client metrics CSV:", e)

            per_client_test_metrics.append(cl_metrics)
            round_summary[f"client{i}_test_loss"] = float(cl_metrics.get("loss", np.nan))
            round_summary[f"client{i}_test_acc"] = float(cl_metrics.get("accuracy", np.nan))
            per_client_acc_history[i].append(float(cl_metrics.get("accuracy", np.nan)))
            per_client_loss_history[i].append(float(cl_metrics.get("loss", np.nan)))

            # --- VISUALS: run harmonization (CoMoGAN) on client val and save comparison grid ---
            client_root = client.get("client_root", None)
            if client_root is not None:
                hm_dir = harmonize_client_validation(client_root, CLIENT_NAMES[i], OUTPUT_DIR)
                orig_val_dir = os.path.join(client_root, "val")
                # make/overwrite comparison grid (no round suffix)
                make_comparison_grid(orig_val_dir, hm_dir, CLIENT_NAMES[i], OUTPUT_DIR, n_samples=7)
                # save round copy for archival
                save_comparison_grid_round_copy(CLIENT_NAMES[i], OUTPUT_DIR, r+1)

        # Save checkpoint and per-round summary
        ckpt = {
            "round": r+1,
            "model_state": global_model.state_dict(),
            "global_val_metrics": global_val_metrics,
            "global_test_metrics": global_test_metrics,
            "per_client_test_metrics": per_client_test_metrics,
            "client_names": CLIENT_NAMES,
            "class_names": class_names
        }
        ckpt_path = os.path.join(OUTPUT_DIR, f"global_round_{r+1}.pth")
        try:
            torch.save(ckpt, ckpt_path)
            print("Saved checkpoint:", ckpt_path)
        except Exception as e:
            print("Warning saving checkpoint:", e)

        # Append summary and save CSV
        round_results.append(round_summary)
        df = pd.DataFrame(round_results)
        csv_path = os.path.join(OUTPUT_DIR, "fl_round_results.csv")
        try:
            df.to_csv(csv_path, index=False)
            print("Saved per-round summary CSV to", csv_path)
        except Exception as e:
            print("Warning saving per-round summary CSV:", e)

        # Update plots for global test accuracy & loss (overwrite to replace after each round)
        rounds = list(range(1, len(round_results)+1))
        gtest_acc = [rr.get("global_test_acc", 0.0) for rr in round_results]
        gtest_loss = [rr.get("global_test_loss", 0.0) for rr in round_results]

        try:
            plt.figure(figsize=(6,4))
            plt.plot(rounds, gtest_acc, marker='o')
            plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Global Test Accuracy")
            plt.grid(True)
            plt.savefig(global_test_acc_fname); plt.close()
        except Exception as e:
            print("Warning saving global test acc plot:", e)

        try:
            plt.figure(figsize=(6,4))
            plt.plot(rounds, gtest_loss, marker='o')
            plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Global Test Loss")
            plt.grid(True)
            plt.savefig(global_test_loss_fname); plt.close()
        except Exception as e:
            print("Warning saving global test loss plot:", e)

        # Also save round-stamped versions (historical)
        try:
            plt.figure(figsize=(6,4))
            plt.plot(rounds, gtest_acc, marker='o')
            plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Global Test Accuracy")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"global_test_accuracy_rounds_up_to_{r+1}.png")); plt.close()
        except Exception:
            pass

        try:
            plt.figure(figsize=(6,4))
            plt.plot(rounds, gtest_loss, marker='o')
            plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Global Test Loss")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"global_test_loss_rounds_up_to_{r+1}.png")); plt.close()
        except Exception:
            pass

        # per-client accuracy plot (overwrite)
        per_client_acc_fname = os.path.join(OUTPUT_DIR, "per_client_test_accuracy_rounds.png")
        try:
            plt.figure(figsize=(8,5))
            for i, name in enumerate(CLIENT_NAMES):
                plt.plot(range(1, len(per_client_acc_history[i])+1), per_client_acc_history[i], label=name, marker='o')
            plt.xlabel("Global Round"); plt.ylabel("Test Accuracy");plt.title("Per-client Test Accuracy"); plt.legend(); plt.grid(True)
            plt.savefig(per_client_acc_fname); plt.close()
        except Exception as e:
            print("Warning saving per-client acc plot:", e)

        per_client_loss_fname = os.path.join(OUTPUT_DIR, "per_client_test_loss_rounds.png")
        try:
            plt.figure(figsize=(8,5))
            for i, name in enumerate(CLIENT_NAMES):
                plt.plot(range(1, len(per_client_loss_history[i])+1), per_client_loss_history[i], label=name, marker='o')
            plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Per-client Test Loss"); plt.legend(); plt.grid(True)
            plt.savefig(per_client_loss_fname); plt.close()
        except Exception as e:
            print("Warning saving per-client loss plot:", e)

        # also save round-stamped copies of per-client plots
        try:
            shutil.copyfile(per_client_acc_fname, os.path.join(OUTPUT_DIR, f"per_client_test_accuracy_rounds_up_to_{r+1}.png"))
            shutil.copyfile(per_client_loss_fname, os.path.join(OUTPUT_DIR, f"per_client_test_loss_rounds_up_to_{r+1}.png"))
        except Exception:
            pass

    # End of rounds
    final_model_path = os.path.join(OUTPUT_DIR, "global_final.pth")
    try:
        torch.save({"model_state": global_model.state_dict(), "class_names": class_names}, final_model_path)
        print("Federated training finished. Final global model saved to:", final_model_path)
    except Exception as e:
        print("Warning saving final model:", e)

if __name__ == "__main__":
    main()
