# fl_train_adain_with_grid.py
"""
Federated averaging (FedAvg) with AdaIN-style harmonization saved per communication round,
plus comparison-grid visualization between original vs harmonized validation images.

Usage: adjust CLIENT_ROOTS and run:
    python fl_train_adain_with_grid.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torchvision import io as tv_io
try:
    tv_io.set_image_backend('PIL')
except Exception:
    pass

import copy
import time
import random
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
from typing import List, Tuple
from PIL import Image
import shutil

# ----------------------------
# CONFIG - set these paths
# ----------------------------
CLIENT_ROOTS = [
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Shenzhen",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Montgomery",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\TBX11K",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES = ["Shenzhen", "Montgomery", "TBX11K", "Pakistan"]
OUTPUT_DIR = r"./fl_outputs_AdaIN_tubo"
ARCH = "densenet169"
PRETRAINED = True
IMG_SIZE = 224
BATCH_SIZE = 4
WORKERS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = False
PIN_MEMORY = True
DROPOUT_P = 0.5
SEED = 42
CLASS_NAMES = ["normal", "positive"]

# Harmonization options
HARMONIZE = True
STYLE_SOURCE = "first_client"  # "first_client" | "per_client" | "random_global"
HARMONIZE_VAL_TEST = True

# Visualization options
GRID_SAMPLES = 7  # number of sample columns in comparison grid per client

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
# Utilities
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ----------------------------
# Dataset utilities
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
        return []
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
        train_ds = PathListDataset(tr, transform=train_tf)
        val_ds = PathListDataset(va, transform=val_tf)
        test_ds = PathListDataset(te, transform=val_tf)
        per_client_dataloaders.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "train_ds": train_ds
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

# ----------------------------
# Simple pixel-space AdaIN helpers
# ----------------------------
to_tensor_no_norm = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor()])
to_pil_from_tensor = transforms.ToPILImage()

def image_load_as_tensor(path):
    img = default_loader(path)
    t = to_tensor_no_norm(img)
    return t

def compute_channel_mean_std(tensor):
    c = tensor.view(tensor.shape[0], -1)
    mu = c.mean(dim=1)
    std = c.std(dim=1, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return mu, std

def adain_transfer(content_t, style_t):
    mu_c, std_c = compute_channel_mean_std(content_t)
    mu_s, std_s = compute_channel_mean_std(style_t)
    mu_c = mu_c[:, None, None]; std_c = std_c[:, None, None]
    mu_s = mu_s[:, None, None]; std_s = std_s[:, None, None]
    normalized = (content_t - mu_c) / std_c
    out = normalized * std_s + mu_s
    out = torch.clamp(out, 0.0, 1.0)
    return out

# ----------------------------
# Harmonization orchestration
# ----------------------------
def pick_style_image_paths(client_roots):
    rng = random.Random(SEED)
    client_train_samples = {}
    for cr in client_roots:
        tr = gather_samples_from_client_split(cr, "train", CLASS_NAMES)
        client_train_samples[cr] = [p for p,l in tr]
    style_map = {}
    if STYLE_SOURCE == "first_client":
        ref_cr = client_roots[0]
        refs = client_train_samples.get(ref_cr, [])
        if len(refs) == 0:
            raise RuntimeError(f"No train images in reference client {ref_cr}")
        ref_path = refs[len(refs)//2]
        for cr in client_roots:
            style_map[cr] = ref_path
    elif STYLE_SOURCE == "per_client":
        for cr in client_roots:
            refs = client_train_samples.get(cr, [])
            style_map[cr] = (refs[len(refs)//2] if len(refs)>0 else None)
    elif STYLE_SOURCE == "random_global":
        all_paths = []
        for v in client_train_samples.values():
            all_paths.extend(v)
        if len(all_paths) == 0:
            raise RuntimeError("No train images found in any client")
        ref = rng.choice(all_paths)
        for cr in client_roots:
            style_map[cr] = ref
    else:
        raise ValueError("Unknown STYLE_SOURCE")
    return style_map

def create_harmonized_client_folders(round_idx, client_roots):
    style_map = pick_style_image_paths(client_roots)
    harmonized_round_dir = os.path.join(OUTPUT_DIR, f"harmonized_round_{round_idx}")
    if os.path.exists(harmonized_round_dir):
        shutil.rmtree(harmonized_round_dir)
    os.makedirs(harmonized_round_dir, exist_ok=True)
    harmonized_roots = []
    for cr, cname in zip(client_roots, CLIENT_NAMES):
        dest_root = os.path.join(harmonized_round_dir, cname)
        os.makedirs(dest_root, exist_ok=True)
        for split in ["train", "val", "test"]:
            src_split_dir = os.path.join(cr, split)
            dest_split_dir = os.path.join(dest_root, split)
            if not os.path.exists(src_split_dir):
                os.makedirs(dest_split_dir, exist_ok=True)
                continue
            for cls_folder in os.listdir(src_split_dir):
                src_cls_dir = os.path.join(src_split_dir, cls_folder)
                if not os.path.isdir(src_cls_dir):
                    continue
                dest_cls_dir = os.path.join(dest_split_dir, cls_folder)
                os.makedirs(dest_cls_dir, exist_ok=True)
                for fn in os.listdir(src_cls_dir):
                    if not fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                        continue
                    src_fp = os.path.join(src_cls_dir, fn)
                    dest_fp = os.path.join(dest_cls_dir, fn)
                    if split == "train" or (HARMONIZE_VAL_TEST and split in ["val","test"]):
                        style_path = style_map.get(cr, None)
                        if style_path is None:
                            shutil.copy2(src_fp, dest_fp)
                        else:
                            try:
                                content_t = image_load_as_tensor(src_fp)
                                style_t = image_load_as_tensor(style_path)
                                harmonized_t = adain_transfer(content_t, style_t)
                                pil = to_pil_from_tensor(harmonized_t)
                                pil.save(dest_fp)
                            except Exception as e:
                                print(f"[HARM] fallback copy for {src_fp}: {e}")
                                shutil.copy2(src_fp, dest_fp)
                    else:
                        shutil.copy2(src_fp, dest_fp)
        harmonized_roots.append(dest_root)
    return harmonized_roots

# ----------------------------
# Comparison grid helpers
# ----------------------------
def select_val_pairs_for_comparison(orig_val_dir, hm_val_dir, n_samples=7):
    """
    Find matching image files between orig_val_dir and hm_val_dir by scanning class subfolders.
    Returns list of tuples (orig_path, hm_path, filename) up to n_samples (or fewer).
    """
    pairs = []
    # iterate classes in orig_val_dir
    for cls in os.listdir(orig_val_dir):
        orig_cls_dir = os.path.join(orig_val_dir, cls)
        hm_cls_dir = os.path.join(hm_val_dir, cls)
        if not os.path.isdir(orig_cls_dir) or not os.path.isdir(hm_cls_dir):
            continue
        orig_files = [f for f in os.listdir(orig_cls_dir) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]
        for fn in orig_files:
            orig_fp = os.path.join(orig_cls_dir, fn)
            hm_fp = os.path.join(hm_cls_dir, fn)
            if os.path.exists(hm_fp):
                pairs.append((orig_fp, hm_fp, fn))
    if len(pairs) == 0:
        return []
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    return pairs[:min(n_samples, len(pairs))]

def make_comparison_grid(original_val_dir, hm_val_dir, client_name, out_base, n_samples=7):
    base_dest = os.path.join(out_base, "ComparisonGrid")
    ensure_dir(base_dest)
    fn_pairs = select_val_pairs_for_comparison(original_val_dir, hm_val_dir, n_samples=n_samples)
    if len(fn_pairs) == 0:
        print(f"[VIS] no matching pairs for comparison for {client_name}")
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
        plt.savefig(out_png); plt.close(fig)
        print(f"[VIS] Saved comparison grid for {client_name} at {out_png}")
    except Exception as e:
        print("[VIS] failed saving comparison grid:", e)
        plt.close(fig)

# ----------------------------
# Model + training utilities (unchanged from your original)
# ----------------------------
def create_model(num_classes, arch=ARCH, pretrained=PRETRAINED):
    if arch.startswith("densenet") and hasattr(torchvision.models, arch):
        model = getattr(torchvision.models, arch)(pretrained=pretrained)
        if hasattr(model, "classifier"):
            in_ch = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_ch, num_classes))
        elif hasattr(model, "fc"):
            in_ch = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_ch, num_classes))
        return model
    else:
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
# Federated training main with harmonization + grid generation
# ----------------------------
def main():
    print("DEVICE:", DEVICE)
    num_clients = len(CLIENT_ROOTS)
    num_classes = len(CLASS_NAMES)
    global_model = create_model(num_classes=num_classes, arch=ARCH, pretrained=PRETRAINED).to(DEVICE)
    print(f"Global model {ARCH} created with {count_parameters(global_model):,} trainable params")

    round_results = []
    global_test_acc_fname = os.path.join(OUTPUT_DIR, "global_test_accuracy_rounds.png")
    global_test_loss_fname = os.path.join(OUTPUT_DIR, "global_test_loss_rounds.png")
    per_client_acc_history = {i: [] for i in range(num_clients)}
    per_client_loss_history = {i: [] for i in range(num_clients)}

    for r in range(COMM_ROUNDS):
        print("\n" + "="*60)
        print(f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print("="*60)

        if HARMONIZE:
            print("Creating harmonized datasets for round", r+1)
            harmonized_roots = create_harmonized_client_folders(r+1, CLIENT_ROOTS)
            used_client_roots = harmonized_roots
            # Create comparison grids for each client: original val vs harmonized val
            for orig_root, hm_root, cname in zip(CLIENT_ROOTS, harmonized_roots, CLIENT_NAMES):
                orig_val = os.path.join(orig_root, "val")
                hm_val = os.path.join(hm_root, "val")
                if os.path.isdir(orig_val) and os.path.isdir(hm_val):
                    make_comparison_grid(orig_val, hm_val, cname, os.path.join(OUTPUT_DIR, f"harmonized_round_{r+1}"), n_samples=GRID_SAMPLES)
                else:
                    print(f"[VIS] missing val folders for {cname}: orig={orig_val}, hm={hm_val}")
        else:
            used_client_roots = CLIENT_ROOTS

        combined_loaders, combined_sizes, class_names, combined_train_ds, per_client_dataloaders, per_client_test_dsets = make_multi_client_dataloaders(
            used_client_roots, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda")
        )

        client_train_sizes = [len(per_client_dataloaders[i]['train'].dataset) for i in range(len(per_client_dataloaders))]
        total_train = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
        print("client train sizes:", client_train_sizes)

        local_models = []
        weights = []
        round_summary = {"round": r+1}

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
            print(f"[CLIENT {i}] local validation")
            local_val_metrics = evaluate_model(local_model, client['val'], DEVICE, criterion=criterion)
            print(f"[CLIENT {i}] local val acc={local_val_metrics.get('accuracy', np.nan):.4f}, loss={local_val_metrics.get('loss', np.nan):.4f}")
            round_summary[f"client{i}_localval_loss"] = float(local_val_metrics.get("loss", np.nan))
            round_summary[f"client{i}_localval_acc"] = float(local_val_metrics.get("accuracy", np.nan))
            local_models.append(local_model.cpu())
            w = float(client_train_sizes[i]) / float(total_train)
            weights.append(w)
            print(f"[CLIENT {i}] aggregation weight: {w:.4f}")

        print("\nAggregating local models (FedAvg weighted)")
        avg_state = average_models_weighted(local_models, weights)
        avg_state_on_device = {k: v.to(DEVICE) for k, v in avg_state.items()}
        global_model.load_state_dict(avg_state_on_device)
        global_model.to(DEVICE)

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

        print("\nGlobal TEST on combined test (all clients)")
        combined_test_dsets = [per_client_dataloaders[i]['test'].dataset for i in range(len(per_client_dataloaders))]
        combined_test = ConcatDataset(combined_test_dsets)
        combined_test_loader = DataLoader(combined_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda"))
        global_test_metrics = evaluate_model(global_model, combined_test_loader, DEVICE, criterion=combined_criterion, return_per_class=True, class_names=class_names)
        print("Global combined TEST metrics summary:", {k: global_test_metrics.get(k) for k in ["accuracy","loss","f1_macro","precision_macro","recall_macro","balanced_acc","cohen_kappa"]})
        round_summary["global_test_loss"] = float(global_test_metrics.get("loss", np.nan))
        round_summary["global_test_acc"] = float(global_test_metrics.get("accuracy", np.nan))

        # Combined per-class CSV saving
        try:
            cm = global_test_metrics.get("confusion_matrix", None)
            per_prec = global_test_metrics.get("per_class_precision", [])
            per_rec = global_test_metrics.get("per_class_recall", [])
            per_f1 = global_test_metrics.get("per_class_f1", [])
            per_spec = global_test_metrics.get("per_class_specificity", [])
            per_acc = global_test_metrics.get("per_class_accuracy", [])
            per_support = global_test_metrics.get("per_class_support", [])
            per_correct = global_test_metrics.get("per_class_correct", [])

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
            combined_csv = os.path.join(OUTPUT_DIR, f"combined_test_metrics_round{r+1}.csv")
            df_combined.to_csv(combined_csv, index=False)
            if cm is not None:
                pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(OUTPUT_DIR, f"combined_confusion_matrix_round{r+1}.csv"))
            print(f"Saved combined per-class test metrics CSV to: {combined_csv}")
        except Exception as e:
            print("Warning saving/printing combined per-class metrics:", e)

        # Per-client global test & CSVs
        per_client_test_metrics = []
        for i, client in enumerate(per_client_dataloaders):
            print(f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set")
            client_train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)
            cl_metrics = evaluate_model(global_model, client['test'], DEVICE, criterion=client_criterion, return_per_class=True, class_names=class_names)
            mean_spec = None
            if "per_class_specificity" in cl_metrics:
                specs = cl_metrics.get("per_class_specificity", [])
                if len(specs) > 0:
                    mean_spec = float(np.mean(specs))
            print(f"[CLIENT {i}] Summary metrics:")
            acc = cl_metrics.get("accuracy", np.nan)
            prec = cl_metrics.get("precision_macro", np.nan)
            rec = cl_metrics.get("recall_macro", np.nan)
            f1 = cl_metrics.get("f1_macro", np.nan)
            kappa = cl_metrics.get("cohen_kappa", np.nan)
            print(f"  Accuracy       : {acc:.4f}")
            print(f"  Precision (mac): {prec:.4f}")
            print(f"  Recall (mac)   : {rec:.4f}")
            print(f"  F1 (mac)       : {f1:.4f}")
            if mean_spec is not None:
                print(f"  Mean Specificity: {mean_spec:.4f}")
            else:
                print(f"  Mean Specificity: n/a")
            print(f"  Cohen's kappa  : {kappa:.4f}")

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
        torch.save(ckpt, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        round_results.append(round_summary)
        df = pd.DataFrame(round_results)
        csv_path = os.path.join(OUTPUT_DIR, "fl_round_results.csv")
        df.to_csv(csv_path, index=False)
        print("Saved per-round summary CSV to", csv_path)

        rounds = list(range(1, len(round_results)+1))
        gtest_acc = [rr.get("global_test_acc", 0.0) for rr in round_results]
        gtest_loss = [rr.get("global_test_loss", 0.0) for rr in round_results]

        plt.figure(figsize=(6,4))
        plt.plot(rounds, gtest_acc)
        plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Global Test Accuracy")
        plt.savefig(global_test_acc_fname); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(rounds, gtest_loss)
        plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Global Test Loss")
        plt.savefig(global_test_loss_fname); plt.close()

        per_client_acc_fname = os.path.join(OUTPUT_DIR, "per_client_test_accuracy_rounds.png")
        plt.figure(figsize=(8,5))
        for i, name in enumerate(CLIENT_NAMES):
            plt.plot(range(1, len(per_client_acc_history[i])+1), per_client_acc_history[i], label=name)
        plt.xlabel("Global Round"); plt.ylabel("Test Accuracy");plt.title("Per-client Test Accuracy"); plt.legend()
        plt.savefig(per_client_acc_fname); plt.close()

        per_client_loss_fname = os.path.join(OUTPUT_DIR, "per_client_test_loss_rounds.png")
        plt.figure(figsize=(8,5))
        for i, name in enumerate(CLIENT_NAMES):
            plt.plot(range(1, len(per_client_loss_history[i])+1), per_client_loss_history[i], label=name)
        plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Per-client Test Loss"); plt.legend()
        plt.savefig(per_client_loss_fname); plt.close()

    final_model_path = os.path.join(OUTPUT_DIR, "global_final.pth")
    torch.save({"model_state": global_model.state_dict(), "class_names": class_names}, final_model_path)
    print("Federated training finished. Final global model saved to:", final_model_path)

if __name__ == "__main__":
    main()
