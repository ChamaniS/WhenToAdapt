import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import json
import csv
import random
from collections import defaultdict
from pathlib import Path
import sys
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
)
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# Config
# =========================================================
SEED = 42

DATA_ROOT = r"/lustre06/project/6008975/csj5/Breasttumor_classi_renamed/"
OUTPUT_DIR = "breast_class_federated_fda"
MODEL_NAME = "efficientnet_b0_breast_tumor_fedavg_fda.pth"

# This file will be loaded into EfficientNet-B0
WEIGHTS_PATH = r"/lustre06/project/6008975/csj5/narvalenv/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"

output_file = r"/lustre06/project/6008975/csj5/narvalenv/FDA_breast.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-3
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES = ["BUSBRA", "BUS", "BUSI", "UDIAT"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# FDA settings
USE_FDA = True
FDA_L = 0.05

# One reference image per client, used to build the FDA harmonized version
REFERENCE_IMAGES = {
    "BUSBRA": "0001-r.png",
    "BUS": "00104.png",
    "BUSI": "101.png",
    "UDIAT": "000007.png",
}

FDA_OUTPUT_SUBDIR = "FDA_Harmonized"
GRID_OUTPUT_SUBDIR = "ComparisonGrids"

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
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch.lower() for ch in stem if ch.isalnum())

def is_image_file(path):
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

def list_images_recursive(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if is_image_file(full):
                files.append(full)
    return sorted(files)

def find_image_by_basename(root_dirs, target_filename):
    """
    Search recursively across a list of directories for a file whose basename
    matches the target filename after normalization.
    """
    target_key = normalize_name(target_filename)
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for path in list_images_recursive(root):
            if normalize_name(os.path.basename(path)) == target_key:
                return path
    return None

def get_split_paths(root, client_names):
    """
    Expected structure:
      DATA_ROOT/
        clientA/train/<class folders>
        clientA/val/<class folders>
        clientA/test/<class folders>
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
                "All clients must have the same class folder names."
            )
        if ds.class_to_idx != base_class_to_idx:
            raise ValueError(
                f"Class-to-index mismatch detected in dataset {i}.\n"
                "All clients must use identical class folder naming."
            )

    return base_classes, base_class_to_idx

def build_combined_dataset(ds_list):
    if len(ds_list) == 1:
        return ds_list[0]
    return ConcatDataset(ds_list)

def count_samples(ds):
    if isinstance(ds, ConcatDataset):
        return sum(len(d) for d in ds.datasets)
    return len(ds)

def load_state_dict_flexibly(weights_path):
    """
    Loads a checkpoint from several common formats:
    - raw state_dict
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    Also strips 'module.' prefixes if present.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"WEIGHTS_PATH not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        clean_state_dict[key] = v

    return clean_state_dict

def build_model(num_classes, weights_path=None):
    """
    EfficientNet-B0 with optional local pretrained weights.
    The classifier head is replaced for num_classes, and the loaded checkpoint
    skips mismatched classifier weights.
    """
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if weights_path is not None and os.path.isfile(weights_path):
        state_dict = load_state_dict_flexibly(weights_path)

        model_state = model.state_dict()
        filtered_state = {}

        skipped = []
        loaded = 0

        for k, v in state_dict.items():
            # Skip classifier head because num_classes differs
            if k.startswith("classifier.1."):
                skipped.append(k)
                continue

            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
                loaded += 1
            else:
                skipped.append(k)

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"[Weights] Loaded checkpoint: {weights_path}")
        print(f"[Weights] Loaded tensors: {loaded}")
        print(f"[Weights] Missing keys after load: {len(missing)}")
        print(f"[Weights] Unexpected keys after load: {len(unexpected)}")
    else:
        print("[Weights] No valid WEIGHTS_PATH provided. Training from scratch.")

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
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd

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
            "specificity_macro",
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
                f'{r["specificity_macro"]:.6f}',
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
# FDA: Fourier Domain Adaptation
# =========================================================
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

def load_rgb_uint8(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_rgb_uint8(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def fda_swap_amplitude(src_img, ref_img, L=0.05):
    """
    FDA amplitude swapping for RGB uint8 images.
    src_img, ref_img: HxWx3 uint8 arrays
    """
    src = src_img.astype(np.float32)
    ref = ref_img.astype(np.float32)

    h, w, _ = src.shape

    if ref.shape[0] != h or ref.shape[1] != w:
        ref = np.array(
            Image.fromarray(ref.astype(np.uint8)).resize((w, h), resample=RESAMPLE_BILINEAR)
        ).astype(np.float32)

    b_h = max(1, int(np.floor(h * L)))
    b_w = max(1, int(np.floor(w * L)))

    c_h = h // 2
    c_w = w // 2

    out = np.zeros_like(src, dtype=np.uint8)

    for ch in range(3):
        src_f = np.fft.fft2(src[:, :, ch])
        src_fshift = np.fft.fftshift(src_f)
        src_amp = np.abs(src_fshift)
        src_pha = np.angle(src_fshift)

        ref_f = np.fft.fft2(ref[:, :, ch])
        ref_fshift = np.fft.fftshift(ref_f)
        ref_amp = np.abs(ref_fshift)

        h1 = max(0, c_h - b_h)
        h2 = min(h, c_h + b_h)
        w1 = max(0, c_w - b_w)
        w2 = min(w, c_w + b_w)

        src_amp[h1:h2, w1:w2] = ref_amp[h1:h2, w1:w2]

        combined = src_amp * np.exp(1j * src_pha)
        combined_ishift = np.fft.ifftshift(combined)
        rec = np.fft.ifft2(combined_ishift)
        rec = np.real(rec)
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        out[:, :, ch] = rec

    return out

def build_reference_image_for_client(original_client_root, ref_filename):
    """
    Search the original client's train/val/test folders for the requested filename.
    """
    search_dirs = [
        os.path.join(original_client_root, "train"),
        os.path.join(original_client_root, "val"),
        os.path.join(original_client_root, "test"),
    ]
    ref_path = find_image_by_basename(search_dirs, ref_filename)
    if ref_path is None:
        # Fallback: first image from train/val/test
        for d in search_dirs:
            imgs = list_images_recursive(d)
            if imgs:
                ref_path = imgs[0]
                print(f"[FDA] Warning: reference '{ref_filename}' not found for {original_client_root}.")
                print(f"[FDA] Falling back to first available image: {ref_path}")
                break

    if ref_path is None:
        raise ValueError(f"No images found for client root: {original_client_root}")

    ref_img = load_rgb_uint8(ref_path)
    return ref_path, ref_img

def create_fda_split(src_split_dir, dst_split_dir, ref_img, L=0.05):
    """
    Copy the ImageFolder structure from src_split_dir to dst_split_dir,
    but replace each source image with its FDA-harmonized version.
    """
    ensure_dir(dst_split_dir)

    ds = datasets.ImageFolder(src_split_dir, transform=None)
    if len(ds) == 0:
        raise ValueError(f"No images found in {src_split_dir}")

    for img_path, class_idx in ds.samples:
        rel_path = os.path.relpath(img_path, src_split_dir)
        dst_path = os.path.join(dst_split_dir, rel_path)
        ensure_dir(os.path.dirname(dst_path))

        src_img = load_rgb_uint8(img_path)
        harmonized = fda_swap_amplitude(src_img, ref_img, L=L)
        save_rgb_uint8(harmonized, dst_path)

def create_fda_datasets(paths_dict, client_names, out_base, reference_images, L=0.05):
    """
    Create harmonized train/val/test copies for every client.
    """
    base = os.path.join(out_base, FDA_OUTPUT_SUBDIR)
    ensure_dir(base)

    used_paths = {}

    for client in client_names:
        client_root = os.path.join(DATA_ROOT, client)
        ref_filename = reference_images.get(client, None)
        if ref_filename is None:
            raise ValueError(f"Missing reference image filename for client: {client}")

        ref_path, ref_img = build_reference_image_for_client(client_root, ref_filename)
        print(f"[FDA] Client: {client}")
        print(f"[FDA] Reference image: {ref_path}")

        client_base = os.path.join(base, client)
        train_dst = os.path.join(client_base, "train")
        val_dst = os.path.join(client_base, "val")
        test_dst = os.path.join(client_base, "test")

        print(f"[FDA] Harmonizing train/val/test for {client} ...")
        create_fda_split(paths_dict[client]["train"], train_dst, ref_img, L=L)
        create_fda_split(paths_dict[client]["val"], val_dst, ref_img, L=L)
        create_fda_split(paths_dict[client]["test"], test_dst, ref_img, L=L)

        used_paths[client] = {
            "train": train_dst,
            "val": val_dst,
            "test": test_dst,
        }

    print("[FDA] Harmonization finished.")
    return used_paths

DIFF_AMPLIFICATION = 6.0   
DIFF_CLIP_MAX = 255

def save_comparison_grid(original_img_path, harmonized_img_path, save_path, title=None):
    orig = load_rgb_uint8(original_img_path)
    harm = load_rgb_uint8(harmonized_img_path)

    if orig.shape != harm.shape:
        harm = np.array(
            Image.fromarray(harm).resize((orig.shape[1], orig.shape[0]), resample=RESAMPLE_BILINEAR),
            dtype=np.uint8
        )

    diff = np.abs(orig.astype(np.int16) - harm.astype(np.int16)).astype(np.float32)
    diff_vis = np.clip(diff * DIFF_AMPLIFICATION, 0, DIFF_CLIP_MAX).astype(np.uint8)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    if title is None:
        title = os.path.basename(original_img_path)

    axs[0].imshow(orig)
    axs[0].axis("off")
    axs[0].set_title("Original", fontsize=12)

    axs[1].imshow(harm)
    axs[1].axis("off")
    axs[1].set_title("FDA Harmonized", fontsize=12)

    axs[2].imshow(diff_vis)
    axs[2].axis("off")
    axs[2].set_title(f"Amplified Absolute Difference (x{DIFF_AMPLIFICATION})", fontsize=12)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=180)
    plt.close(fig)

def save_selected_comparison_grids(original_paths_dict, harmonized_paths_dict, client_names, out_base, reference_images):
    grids_base = os.path.join(out_base, GRID_OUTPUT_SUBDIR)
    ensure_dir(grids_base)

    for client in client_names:
        ref_filename = reference_images.get(client)
        if ref_filename is None:
            print(f"[Grid] No reference image specified for {client}. Skipping.")
            continue

        original_dirs = [
            original_paths_dict[client]["train"],
            original_paths_dict[client]["val"],
            original_paths_dict[client]["test"],
        ]
        harmonized_dirs = [
            harmonized_paths_dict[client]["train"],
            harmonized_paths_dict[client]["val"],
            harmonized_paths_dict[client]["test"],
        ]

        orig_path = find_image_by_basename(original_dirs, ref_filename)
        harm_path = find_image_by_basename(harmonized_dirs, ref_filename)

        if orig_path is None:
            print(f"[Grid] Could not find original image '{ref_filename}' for client '{client}'")
            continue
        if harm_path is None:
            print(f"[Grid] Could not find harmonized image '{ref_filename}' for client '{client}'")
            continue

        client_out = os.path.join(grids_base, client)
        ensure_dir(client_out)

        safe_name = "".join(ch if ch.isalnum() else "_" for ch in os.path.splitext(ref_filename)[0]).strip("_")
        save_path = os.path.join(client_out, f"comparison_grid_{safe_name}.png")

        save_comparison_grid(
            orig_path,
            harm_path,
            save_path,
            title=f"{client} - {ref_filename}",
        )
        print(f"[Grid] Saved: {save_path}")

# =========================================================
# Main
# =========================================================
def main():
    paths = get_split_paths(DATA_ROOT, CLIENT_NAMES)

    # Check original dataset structure
    for client in CLIENT_NAMES:
        for split in ["train", "val", "test"]:
            if not os.path.isdir(paths[client][split]):
                raise FileNotFoundError(f"Missing directory: {paths[client][split]}")

    print("Original dataset paths:")
    for client in CLIENT_NAMES:
        print(f"\nClient: {client}")
        print(f"  train: {paths[client]['train']}")
        print(f"  val  : {paths[client]['val']}")
        print(f"  test : {paths[client]['test']}")

    # FDA harmonization
    if USE_FDA:
        used_paths = create_fda_datasets(
            paths_dict=paths,
            client_names=CLIENT_NAMES,
            out_base=OUTPUT_DIR,
            reference_images=REFERENCE_IMAGES,
            L=FDA_L,
        )
        save_selected_comparison_grids(
            original_paths_dict=paths,
            harmonized_paths_dict=used_paths,
            client_names=CLIENT_NAMES,
            out_base=OUTPUT_DIR,
            reference_images=REFERENCE_IMAGES,
        )
    else:
        used_paths = paths

    # Build per-client datasets from either original or FDA-harmonized data
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for client in CLIENT_NAMES:
        train_ds = datasets.ImageFolder(used_paths[client]["train"], transform=train_tfms)
        val_ds = datasets.ImageFolder(used_paths[client]["val"], transform=eval_tfms)
        test_ds = datasets.ImageFolder(used_paths[client]["test"], transform=eval_tfms)

        if len(train_ds) == 0:
            raise ValueError(f"No training images found for {client}: {used_paths[client]['train']}")
        if len(val_ds) == 0:
            raise ValueError(f"No validation images found for {client}: {used_paths[client]['val']}")
        if len(test_ds) == 0:
            raise ValueError(f"No test images found for {client}: {used_paths[client]['test']}")

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)

    # Validate class alignment
    class_names, class_to_idx = check_class_alignment(train_datasets)
    num_classes = len(class_names)

    print("\nClasses:", class_names)
    print("Class to idx:", class_to_idx)

    train_ds_all = build_combined_dataset(train_datasets)
    val_ds_all = build_combined_dataset(val_datasets)
    test_ds_all = build_combined_dataset(test_datasets)

    print("\nDataset sizes:")
    print("Train samples (all clients):", count_samples(train_ds_all))
    print("Val samples   (all clients):", count_samples(val_ds_all))
    print("Test samples  (all clients):", count_samples(test_ds_all))

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

    # DataLoaders
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        train_loaders[client] = DataLoader(
            ds_tr,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        val_loaders[client] = DataLoader(
            ds_va,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        test_loaders[client] = DataLoader(
            ds_te,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    global_val_loader = DataLoader(
        val_ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    global_test_loader = DataLoader(
        test_ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Model / loss
    global_model = build_model(num_classes, weights_path=WEIGHTS_PATH).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())

    history = defaultdict(list)
    round_metrics = []
    start_time = time.time()

    for r in range(COMM_ROUNDS):
        print(f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")

        local_models = []
        local_weights = []

        round_train_losses = []
        round_train_accs = []

        for client_name in CLIENT_NAMES:
            print(f"\n[Client {client_name}]")

            local_model = copy.deepcopy(global_model).to(DEVICE)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=1e-4)

            train_loader = train_loaders[client_name]
            val_loader = val_loaders[client_name]

            client_epoch_losses = []
            client_epoch_accs = []

            for ep in range(LOCAL_EPOCHS):
                tr_loss, tr_acc, _, _ = run_epoch(
                    local_model,
                    train_loader,
                    criterion,
                    optimizer=optimizer,
                    train=True,
                )
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(
                local_model,
                val_loader,
                criterion,
                optimizer=None,
                train=False,
            )
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models.append(local_model)
            local_weights.append(len(train_loader.dataset))

            round_train_losses.append(float(np.mean(client_epoch_losses)))
            round_train_accs.append(float(np.mean(client_epoch_accs)))

        total_train_size = sum(local_weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        # FedAvg
        norm_weights = [w / total_train_size for w in local_weights]
        global_model.load_state_dict(average_state_dicts_weighted(local_models, norm_weights))

        # Global validation
        global_val_loss, global_val_acc, _, _ = run_epoch(
            global_model,
            global_val_loader,
            criterion,
            optimizer=None,
            train=False,
        )

        # Save best model
        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
            print("\nSaved best global model.")

        rm = {
            "round": r + 1,
            "train_loss": float(np.mean(round_train_losses)),
            "train_acc": float(np.mean(round_train_accs)),
            "val_loss": global_val_loss,
            "val_acc": global_val_acc,
        }

        # Global test after this round
        print("\n" + "=" * 30)
        print(f"GLOBAL TEST AFTER ROUND {r + 1} (ALL CLIENTS TOGETHER)")
        print("=" * 30)

        global_test_result = evaluate_loader(
            global_model,
            global_test_loader,
            criterion,
            class_names,
            title_prefix=f"global_round_{r + 1}",
            save_dir=OUTPUT_DIR,
            save_cm=True,
        )

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
            client_result = evaluate_loader(
                global_model,
                test_loaders[client_name],
                criterion,
                class_names,
                title_prefix=f"{client_name}_round_{r + 1}",
                save_dir=OUTPUT_DIR,
                save_cm=True,
            )

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
            f"\n[ROUND {r + 1}] "
            f"Train Loss: {rm['train_loss']:.4f} | Train Acc: {rm['train_acc']:.4f} | "
            f"Val Loss: {rm['val_loss']:.4f} | Val Acc: {rm['val_acc']:.4f} | "
            f"Global Test Acc: {rm['global_test_acc']:.4f}"
        )

        plot_round_curves(history, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    # Load best weights
    global_model.load_state_dict(best_model_wts)

    # Final evaluation
    final_results = []

    print("\n==============================")
    print("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print("==============================")
    result_all = evaluate_loader(
        global_model,
        global_test_loader,
        criterion,
        class_names,
        title_prefix="all_clients_final",
        save_dir=OUTPUT_DIR,
        save_cm=True,
    )
    final_results.append(result_all)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name in CLIENT_NAMES:
        result_client = evaluate_loader(
            global_model,
            test_loaders[client_name],
            criterion,
            class_names,
            title_prefix=f"{client_name}_final",
            save_dir=OUTPUT_DIR,
            save_cm=True,
        )
        final_results.append(result_client)

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"), "w") as f:
        json.dump(round_metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(round_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(round_metrics)

    save_metrics_csv(final_results, os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    with open(os.path.join(OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))
    print(os.path.join(OUTPUT_DIR, FDA_OUTPUT_SUBDIR))
    print(os.path.join(OUTPUT_DIR, GRID_OUTPUT_SUBDIR))

if __name__ == "__main__":
    main()