import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import csv
import json
import random
import shutil
import time
from collections import defaultdict
from typing import Dict, List
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False

try:
    from torchvision.models import EfficientNet_B0_Weights
    HAS_TORCHVISION_WEIGHTS = True
except Exception:
    HAS_TORCHVISION_WEIGHTS = False

# =========================================================
# Config
# =========================================================
SEED = 42
DATA_ROOT = r"/lustre06/project/6008975/csj5/Breasttumor_classi_renamed/"
OUTPUT_DIR = "breast_classi_federated_comogan"
MODEL_NAME = "efficientnet_b0_breast_tumor_fedavg_comogan.pth"
WEIGHTS_PATH = r"/lustre06/project/6008975/csj5/narvalenv/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"

output_file = r"/lustre06/project/6008975/csj5/narvalenv/comogan_breast_classi.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

CLIENT_NAMES = ["BUSBRA", "BUS", "BUSI", "UDIAT"]

# Only these exact images will be used for the comparison grids
REFERENCE_IMAGES = {
    "BUSBRA": "0001-r.png",
    "BUS": "00104.png",
    "BUSI": "101.png",
    "UDIAT": "000007.png",
}

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-3
WEIGHT_DECAY =0
NUM_WORKERS = 0
IMG_SIZE = 224
USE_AMP = True
HARMONIZE_ALL_SPLITS = True
SAVE_COMPARISON_GRIDS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
SCALER = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda" and USE_AMP))

# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = SEED):
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
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# =========================================================
# Path helpers
# =========================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def client_paths(root: str, client_names: List[str]) -> Dict[str, Dict[str, str]]:
    paths = {}
    for client in client_names:
        paths[client] = {
            "train": os.path.join(root, client, "train"),
            "val": os.path.join(root, client, "val"),
            "test": os.path.join(root, client, "test"),
        }
    return paths

def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))

def is_valid_image(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def list_images_recursive(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if is_image_file(fn):
                out.append(os.path.join(root, fn))
    return sorted(out)

# =========================================================
# Safe dataset
# =========================================================
class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that:
    1) uses a real loader (default_loader) so __getitem__ works,
    2) filters corrupted files before training starts.
    """
    def __init__(self, root, transform=None, target_transform=None, loader=None, is_valid_file=None):
        if loader is None:
            loader = default_loader
        if is_valid_file is None:
            is_valid_file = is_valid_image

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Extra safety pass in case anything slipped through
        filtered_samples = []
        bad_count = 0
        for path, target in self.samples:
            if is_valid_image(path):
                filtered_samples.append((path, target))
            else:
                bad_count += 1
                print(f"[SKIP] Corrupted image removed from dataset: {path}")

        self.samples = filtered_samples
        self.imgs = filtered_samples
        self.targets = [s[1] for s in filtered_samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in: {root}")

        if bad_count > 0:
            print(f"[INFO] Filtered {bad_count} invalid image(s) from {root}")

# =========================================================
# Data harmonization
# =========================================================
def comogan_harmonize_image(img: Image.Image, client_name: str, split: str, class_name: str, filename: str) -> Image.Image:
    """
    Placeholder harmonizer.

    Replace this with your actual CoMoGAN generator inference.
    """
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=0)

    seed_val = sum(bytearray(f"{client_name}_{split}_{class_name}_{filename}".encode("utf-8"))) % 1000
    contrast = 0.90 + (seed_val % 11) * 0.01
    brightness = 0.95 + ((seed_val // 11) % 11) * 0.005
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    arr = np.array(img).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] + ((seed_val % 7) - 3) * 0.6, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] + (((seed_val // 7) % 5) - 2) * 0.4, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] + (((seed_val // 35) % 3) - 1) * 0.6, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def harmonize_split(client_root: str, client_name: str, split: str, out_base: str) -> str:
    src_split = os.path.join(client_root, split)
    dst_split = ensure_dir(os.path.join(out_base, "harmonized", client_name, split))

    if not os.path.isdir(src_split):
        raise FileNotFoundError(f"Missing split directory: {src_split}")

    class_folders = [d for d in sorted(os.listdir(src_split)) if os.path.isdir(os.path.join(src_split, d))]
    if len(class_folders) == 0:
        raise ValueError(f"No class subfolders found in {src_split}")

    for cls in class_folders:
        src_cls = os.path.join(src_split, cls)
        dst_cls = ensure_dir(os.path.join(dst_split, cls))

        for fn in sorted(os.listdir(src_cls)):
            if not is_image_file(fn):
                continue

            src_path = os.path.join(src_cls, fn)
            dst_path = os.path.join(dst_cls, fn)

            try:
                if not is_valid_image(src_path):
                    print(f"[HARM-SKIP] Invalid source image: {src_path}")
                    continue

                if os.path.exists(dst_path):
                    if is_valid_image(dst_path):
                        continue
                    try:
                        os.remove(dst_path)
                    except Exception:
                        pass

                img = Image.open(src_path).convert("RGB")
                out = comogan_harmonize_image(img, client_name, split, cls, fn)

                # keep a real image extension
                base, ext = os.path.splitext(dst_path)
                tmp_path = base + ".tmp" + ext   # example: image.tmp.jpg
                out.save(tmp_path)
                os.replace(tmp_path, dst_path)

            except Exception as e:
                print(f"[HARM] skipping {src_path}: {e}")
                try:
                    base, ext = os.path.splitext(dst_path)
                    tmp_path = base + ".tmp" + ext
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

    return dst_split

def harmonize_client_all_splits(client_root: str, client_name: str, out_base: str, splits=("train", "val", "test")) -> Dict[str, str]:
    out = {}
    for sp in splits:
        out[sp] = harmonize_split(client_root, client_name, sp, out_base)
    return out

# =========================================================
# Comparison grid
# =========================================================
def make_comparison_grid(
    original_dirs: List[str],
    harmonized_dirs: List[str],
    client_name: str,
    out_base: str,
    reference_images: Dict[str, str],
):
    grid_dir = ensure_dir(os.path.join(out_base, "ComparisonGrid"))

    ref_fn = reference_images.get(client_name, None)
    if ref_fn is None:
        print(f"[VIS] No reference image specified for {client_name}")
        return

    orig_match = None
    harm_match = None

    # Search all original splits
    for d in original_dirs:
        for p in list_images_recursive(d):
            if os.path.basename(p) == ref_fn and is_valid_image(p):
                orig_match = p
                break
        if orig_match is not None:
            break

    # Search all harmonized splits
    for d in harmonized_dirs:
        for p in list_images_recursive(d):
            if os.path.basename(p) == ref_fn and is_valid_image(p):
                harm_match = p
                break
        if harm_match is not None:
            break

    if orig_match is None:
        print(f"[VIS] reference image not found in any original split for {client_name}: {ref_fn}")
        return
    if harm_match is None:
        print(f"[VIS] reference image not found in any harmonized split for {client_name}: {ref_fn}")
        return

    try:
        orig = np.array(Image.open(orig_match).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        harm = np.array(Image.open(harm_match).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        diff = np.abs(harm.astype(np.float32) - orig.astype(np.float32)) * 3.0
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        fig, axs = plt.subplots(3, 1, figsize=(4.5, 10))
        axs[0].imshow(orig)
        axs[0].axis("off")
        axs[0].set_ylabel("Original", fontsize=10)

        axs[1].imshow(harm)
        axs[1].axis("off")
        axs[1].set_ylabel("Harmonized", fontsize=10)

        axs[2].imshow(diff)
        axs[2].axis("off")
        axs[2].set_ylabel("Abs amplified diff", fontsize=10)

        fig.suptitle(f"CoMoGAN-style Harmonization: {client_name} | {ref_fn}", fontsize=13)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        out_png = os.path.join(grid_dir, f"comparison_{client_name}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[VIS] saved comparison grid to {out_png}")

    except Exception as e:
        print(f"[VIS] failed to build comparison grid for {client_name}: {e}")

# =========================================================
# Dataset helpers
# =========================================================
def check_class_alignment(datasets_list):
    base_classes = datasets_list[0].classes
    base_class_to_idx = datasets_list[0].class_to_idx
    for i, ds in enumerate(datasets_list[1:], start=2):
        if ds.classes != base_classes:
            raise ValueError(
                f"Class mismatch detected in dataset {i}.\n"
                f"Expected classes: {base_classes}\n"
                f"Found classes   : {ds.classes}"
            )
        if ds.class_to_idx != base_class_to_idx:
            raise ValueError("Class-to-index mismatch detected. All clients must use identical class folders.")
    return base_classes, base_class_to_idx

def build_client_datasets(paths_dict, split, transform):
    ds_list = []
    for client in CLIENT_NAMES:
        split_dir = paths_dict[client][split]
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        ds_list.append(SafeImageFolder(split_dir, transform=transform))
    classes, class_to_idx = check_class_alignment(ds_list)
    return ds_list, classes, class_to_idx

def build_combined_dataset(ds_list):
    return ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)

def count_samples(ds):
    if isinstance(ds, ConcatDataset):
        return sum(len(d) for d in ds.datasets)
    return len(ds)

def extract_targets(dataset):
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "samples"):
        return [s[1] for s in dataset.samples]
    if isinstance(dataset, ConcatDataset):
        out = []
        for ds in dataset.datasets:
            out.extend(extract_targets(ds))
        return out
    raise ValueError("Cannot extract targets from dataset")

def compute_class_weights(dataset, num_classes):
    targets = np.array(extract_targets(dataset), dtype=np.int64)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

# =========================================================
# Model helpers
# =========================================================
def build_model(num_classes: int):
    try:
        model = efficientnet_b0(weights=None)
    except TypeError:
        model = efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_local_weights(model: nn.Module, weights_path: str):
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[WARN] Local weights not found: {weights_path}")
        return model

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        kk = k.replace("module.", "")
        if kk.startswith("model."):
            kk = kk[len("model."):]
        cleaned[kk] = v

    model_sd = model.state_dict()
    matched = {}
    for k, v in cleaned.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            matched[k] = v

    model_sd.update(matched)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"[WEIGHTS] loaded {len(matched)} tensors from {weights_path}")
    if len(missing) > 0:
        print(f"[WEIGHTS] missing keys (expected for classifier): {len(missing)}")
    if len(unexpected) > 0:
        print(f"[WEIGHTS] unexpected keys: {len(unexpected)}")
    return model

def average_state_dicts_weighted(models, weights):
    if len(models) == 0:
        raise ValueError("No models to average")
    if len(models) != len(weights):
        raise ValueError("models and weights length mismatch")
    sum_w = float(sum(weights))
    if sum_w == 0:
        raise ValueError("Sum of weights is zero")

    norm_w = [w / sum_w for w in weights]
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(norm_w[i] * models[i].state_dict()[k].detach().cpu() for i in range(len(models)))
    return avg_sd

# =========================================================
# Training / evaluation
# =========================================================
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    loop = tqdm(loader, desc="Train" if train else "Eval", leave=False)
    for images, labels in loop:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and USE_AMP)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if DEVICE.type == "cuda" and USE_AMP:
                SCALER.scale(loss).backward()
                SCALER.step(optimizer)
                SCALER.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda" and USE_AMP)):
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

def compute_specificity_from_cm(cm):
    num_classes = cm.shape[0]
    vals = []
    total = cm.sum()
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        vals.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return vals, float(np.mean(vals)) if len(vals) > 0 else 0.0

@torch.no_grad()
def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR, save_cm=True):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
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
        "loss": float(loss),
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "kappa": float(kappa),
        "specificity_macro": float(macro_specificity),
        "per_class_specificity": per_class_specificity,
        "cm": cm.tolist(),
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

def save_metrics_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa", "specificity_macro"
        ])
        for r in results:
            writer.writerow([
                r["split"], f'{r["loss"]:.6f}', f'{r["accuracy"]:.6f}', f'{r["precision_macro"]:.6f}',
                f'{r["recall_macro"]:.6f}', f'{r["f1_macro"]:.6f}', f'{r["kappa"]:.6f}', f'{r["specificity_macro"]:.6f}'
            ])

# =========================================================
# Main
# =========================================================
def main():
    print("DEVICE:", DEVICE)
    paths = client_paths(DATA_ROOT, CLIENT_NAMES)

    if HARMONIZE_ALL_SPLITS:
        print("\n[STEP] Harmonizing all client splits with CoMoGAN hook...")
        harmonized_paths = {}
        for client in CLIENT_NAMES:
            client_root = os.path.join(DATA_ROOT, client)
            harmonized_paths[client] = harmonize_client_all_splits(client_root, client, OUTPUT_DIR, splits=("train", "val", "test"))
            if SAVE_COMPARISON_GRIDS:
                make_comparison_grid(
    original_dirs=[paths[client]["train"], paths[client]["val"], paths[client]["test"]],
    harmonized_dirs=[harmonized_paths[client]["train"], harmonized_paths[client]["val"], harmonized_paths[client]["test"]],
    client_name=client,
    out_base=OUTPUT_DIR,
    reference_images=REFERENCE_IMAGES,
)        
        train_root_map = {c: harmonized_paths[c]["train"] for c in CLIENT_NAMES}
        val_root_map = {c: harmonized_paths[c]["val"] for c in CLIENT_NAMES}
        test_root_map = {c: harmonized_paths[c]["test"] for c in CLIENT_NAMES}
    else:
        train_root_map = {c: paths[c]["train"] for c in CLIENT_NAMES}
        val_root_map = {c: paths[c]["val"] for c in CLIENT_NAMES}
        test_root_map = {c: paths[c]["test"] for c in CLIENT_NAMES}

    train_datasets = [SafeImageFolder(train_root_map[c], transform=train_tfms) for c in CLIENT_NAMES]
    val_datasets = [SafeImageFolder(val_root_map[c], transform=eval_tfms) for c in CLIENT_NAMES]
    test_datasets = [SafeImageFolder(test_root_map[c], transform=eval_tfms) for c in CLIENT_NAMES]

    class_names, class_to_idx = check_class_alignment(train_datasets)
    check_class_alignment(val_datasets)
    check_class_alignment(test_datasets)
    num_classes = len(class_names)

    print("\nClasses:", class_names)
    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

    train_ds_all = build_combined_dataset(train_datasets)
    val_ds_all = build_combined_dataset(val_datasets)
    test_ds_all = build_combined_dataset(test_datasets)

    print("Train samples (all clients):", count_samples(train_ds_all))
    print("Val samples   (all clients):", count_samples(val_ds_all))
    print("Test samples  (all clients):", count_samples(test_ds_all))

    train_loaders, val_loaders, test_loaders = {}, {}, {}
    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        train_loaders[client] = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loaders[client] = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        test_loaders[client] = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    global_val_loader = DataLoader(val_ds_all, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    global_test_loader = DataLoader(test_ds_all, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    global_model = build_model(num_classes).to(DEVICE)
    global_model = load_local_weights(global_model, WEIGHTS_PATH).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())

    history = defaultdict(list)
    round_metrics = []
    start_time = time.time()

    for r in range(COMM_ROUNDS):
        print(f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")
        local_models, local_weights = [], []
        round_train_losses, round_train_accs = [], []

        for client_name in CLIENT_NAMES:
            print(f"\n[Client {client_name}]")
            local_model = copy.deepcopy(global_model).to(DEVICE)
            train_loader = train_loaders[client_name]
            val_loader = val_loaders[client_name]

            client_cw = compute_class_weights(train_loader.dataset, num_classes).to(DEVICE)
            local_criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            client_epoch_losses, client_epoch_accs = [], []
            for ep in range(LOCAL_EPOCHS):
                tr_loss, tr_acc, _, _ = run_epoch(local_model, train_loader, local_criterion, optimizer=optimizer, train=True)
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(local_model, val_loader, criterion, optimizer=None, train=False)
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models.append(copy.deepcopy(local_model).cpu())
            local_weights.append(len(train_loader.dataset))
            round_train_losses.append(float(np.mean(client_epoch_losses)))
            round_train_accs.append(float(np.mean(client_epoch_accs)))

        total_train_size = sum(local_weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        norm_weights = [w / total_train_size for w in local_weights]
        global_model.load_state_dict(average_state_dicts_weighted(local_models, norm_weights))
        global_model.to(DEVICE)

        global_val_loss, global_val_acc, _, _ = run_epoch(global_model, global_val_loader, criterion, optimizer=None, train=False)
        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
            print("\nSaved best global model.")

        rm = {
            "round": r + 1,
            "train_loss": float(np.mean(round_train_losses)),
            "train_acc": float(np.mean(round_train_accs)),
            "val_loss": float(global_val_loss),
            "val_acc": float(global_val_acc),
        }

        print("\n" + "=" * 30)
        print(f"GLOBAL TEST AFTER ROUND {r + 1} (ALL CLIENTS TOGETHER)")
        print("=" * 30)
        global_test_result = evaluate_loader(global_model, global_test_loader, criterion, class_names, title_prefix=f"global_round_{r + 1}", save_dir=OUTPUT_DIR, save_cm=True)
        rm["global_test_loss"] = global_test_result["loss"]
        rm["global_test_acc"] = global_test_result["accuracy"]
        rm["global_test_precision"] = global_test_result["precision_macro"]
        rm["global_test_recall"] = global_test_result["recall_macro"]
        rm["global_test_f1"] = global_test_result["f1_macro"]
        rm["global_test_kappa"] = global_test_result["kappa"]
        rm["global_test_specificity"] = global_test_result["specificity_macro"]

        print("\n" + "=" * 30)
        print(f"INDIVIDUAL CLIENT TESTS AFTER ROUND {r + 1}")
        print("=" * 30)
        for client_name in CLIENT_NAMES:
            client_result = evaluate_loader(global_model, test_loaders[client_name], criterion, class_names, title_prefix=f"{client_name}_round_{r + 1}", save_dir=OUTPUT_DIR, save_cm=True)
            rm[f"{client_name}_test_loss"] = client_result["loss"]
            rm[f"{client_name}_test_acc"] = client_result["accuracy"]
            rm[f"{client_name}_test_precision"] = client_result["precision_macro"]
            rm[f"{client_name}_test_recall"] = client_result["recall_macro"]
            rm[f"{client_name}_test_f1"] = client_result["f1_macro"]
            rm[f"{client_name}_test_kappa"] = client_result["kappa"]
            rm[f"{client_name}_test_specificity"] = client_result["specificity_macro"]

        round_metrics.append(rm)
        history["round"].append(r + 1)
        history["train_loss"].append(rm["train_loss"])
        history["train_acc"].append(rm["train_acc"])
        history["val_loss"].append(rm["val_loss"])
        history["val_acc"].append(rm["val_acc"])
        history["global_test_acc"].append(rm["global_test_acc"])

        print(
            f"\n[ROUND {r + 1}] Train Loss: {rm['train_loss']:.4f} | Train Acc: {rm['train_acc']:.4f} | "
            f"Val Loss: {rm['val_loss']:.4f} | Val Acc: {rm['val_acc']:.4f} | Global Test Acc: {rm['global_test_acc']:.4f}"
        )
        plot_round_curves(history, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    global_model.load_state_dict(best_model_wts)

    final_results = []
    print("\n==============================")
    print("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print("==============================")
    result_all = evaluate_loader(global_model, global_test_loader, criterion, class_names, title_prefix="all_clients_final", save_dir=OUTPUT_DIR, save_cm=True)
    final_results.append(result_all)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name in CLIENT_NAMES:
        result_client = evaluate_loader(global_model, test_loaders[client_name], criterion, class_names, title_prefix=f"{client_name}_final", save_dir=OUTPUT_DIR, save_cm=True)
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

    torch.save({"model_state": global_model.state_dict(), "class_names": class_names}, os.path.join(OUTPUT_DIR, "global_final.pth"))

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "global_final.pth"))

if __name__ == "__main__":
    main()