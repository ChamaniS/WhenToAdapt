import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import random
import json
import csv
from collections import defaultdict

import numpy as np
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, RandomSampler
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


# =========================================================
# Config
# =========================================================
SEED = 42

DATA_ROOT =  r"/xxxxBreasttumor_classi_renamed/"
OUTPUT_DIR = "breast_tumor_federated_cut"
HARMONIZED_ROOT = os.path.join(OUTPUT_DIR, "CUT_Harmonized")
CUT_DIR = os.path.join(OUTPUT_DIR, "CUT")
MODEL_NAME = "efficientnet_b0_brain_tumor_fedavg_cut.pth"
WEIGHTS_PATH = r"/xxxxxxxx/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"
output_file = r"/xxxxxxxx/CUT_breast_classi.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

BATCH_SIZE = 8
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-3
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES = ["BUSBRA", "BUS", "BUSI", "UDIAT"]
REFERENCE_CLIENT = "BUSBRA"
REFERENCE_INDEX = CLIENT_NAMES.index(REFERENCE_CLIENT)

# These are optional. If a filename is missing, the code falls back to the first
# available image in that client split.
REFERENCE_IMAGES = {
    "BUSBRA": "0001-r.png",
    "BUS": "00104.png",
    "BUSI": "101.png",
    "UDIAT": "000007.png",
}

# CUT settings
CUT_EPOCHS =10
BATCH_CUT = 8
LAMBDA_NCE = 1.0
NCE_NUM_PATCHES = 256
NCE_LAYERS = [1, 2, 3, 4]
LR_G = 1e-3
LR_D = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HARMONIZED_ROOT, exist_ok=True)
os.makedirs(CUT_DIR, exist_ok=True)


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

cut_img_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================================================
# Utilities
# =========================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(path):
    return path.lower().endswith(IMG_EXTS)


def list_images_recursive(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if is_image_file(fn):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_tree_force(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    import shutil
    shutil.copytree(src, dst)


def copy_tree_as_jpg(src_dir, dst_dir):
    """
    Copy a directory tree while converting every image to .jpg.
    Preserves folder structure.
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source not found: {src_dir}")

    if os.path.exists(dst_dir):
        import shutil
        shutil.rmtree(dst_dir)

    for dirpath, _, filenames in os.walk(src_dir):
        for fname in filenames:
            if not is_image_file(fname):
                continue

            src_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(src_path, src_dir)
            rel_base, _ = os.path.splitext(rel_path)
            dst_path = os.path.join(dst_dir, rel_base + ".jpg")

            ensure_dir(os.path.dirname(dst_path))
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path, quality=95)


def find_image_by_basename(root_dir, basename):
    """
    Search train/val/test recursively for the first matching basename.
    Returns (path, split_name) or (None, None).
    """
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for dirpath, _, filenames in os.walk(split_dir):
            for fn in filenames:
                if fn == basename and is_image_file(fn):
                    return os.path.join(dirpath, fn), split
    return None, None


def find_sample_image_for_client(root_dir, client_name, preferred_basename=None):
    """
    Try preferred basename first. If missing, fall back to the first available
    image under train/val/test for that client.
    """
    client_root = os.path.join(root_dir, client_name)

    if preferred_basename:
        path, split = find_image_by_basename(client_root, preferred_basename)
        if path is not None:
            return path, split

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(client_root, split)
        if not os.path.isdir(split_dir):
            continue
        files = list_images_recursive(split_dir)
        if len(files) > 0:
            return files[0], split

    return None, None


def average_state_dicts_weighted(state_dicts, weights):
    avg_sd = {}
    for k in state_dicts[0].keys():
        avg_sd[k] = sum(weights[i] * state_dicts[i][k] for i in range(len(state_dicts)))
    return avg_sd


def count_samples(ds):
    if isinstance(ds, ConcatDataset):
        return sum(len(d) for d in ds.datasets)
    return len(ds)


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


def load_efficientnet_weights(model, weights_path):
    """
    Loads EfficientNet-B0 weights from a local .pth file.
    Handles:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - DataParallel keys with "module."
    Skips keys with mismatched shapes.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"WEIGHTS_PATH not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "The file at WEIGHTS_PATH does not look like a PyTorch state_dict "
            "or a checkpoint containing one."
        )

    model_state = model.state_dict()
    filtered_state = {}

    for k, v in checkpoint.items():
        key = k.replace("module.", "")
        if key not in model_state:
            continue
        if model_state[key].shape != v.shape:
            continue
        filtered_state[key] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"[Weights] Loaded from: {weights_path}")
    print(f"[Weights] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return model


def build_model(num_classes):
    model = efficientnet_b0(weights=None)
    model = load_efficientnet_weights(model, WEIGHTS_PATH)
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


def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR, save_cm=True):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

    cm = confusion_matrix(targets, preds)
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    if np.isnan(kappa):
        kappa = 0.0
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
# CUT: dataset for unpaired harmonization
# =========================================================
class ImageTreeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else cut_img_tfms
        self.files = list_images_recursive(root_dir)
        if len(self.files) == 0:
            raise RuntimeError(f"No image files found under: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, path


class CUTGenerator(nn.Module):
    def __init__(self, in_ch=3, ngf=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, ngf, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        self.enc2 = self._enc_block(ngf, ngf * 2)
        self.enc3 = self._enc_block(ngf * 2, ngf * 4)
        self.enc4 = self._enc_block(ngf * 4, ngf * 8)
        self.enc5 = self._enc_block(ngf * 8, ngf * 8)

        self.dec5 = self._dec_block(ngf * 8, ngf * 8)
        self.dec4 = self._dec_block(ngf * 16, ngf * 4)
        self.dec3 = self._dec_block(ngf * 8, ngf * 2)
        self.dec2 = self._dec_block(ngf * 4, ngf)
        self.final = nn.Sequential(
            nn.Conv2d(ngf * 2, in_ch, 7, 1, 3),
            nn.Tanh(),
        )

    def _enc_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        )

    def _dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        )

    def encode(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return [f1, f2, f3, f4, f5]

    def decode(self, feats):
        f1, f2, f3, f4, f5 = feats
        d5 = self.dec5(f5)
        d5_cat = torch.cat([d5, f4], dim=1)
        d4 = self.dec4(d5_cat)
        d4_cat = torch.cat([d4, f3], dim=1)
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, f2], dim=1)
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, f1], dim=1)
        out = self.final(d2_cat)
        return out

    def forward(self, x):
        feats = self.encode(x)
        out = self.decode(feats)
        return out, feats


class PatchSampleF(nn.Module):
    def __init__(self, nc, inner_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, inner_dim, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 1, 1, 0, bias=True),
        )

    def forward(self, feat):
        return self.net(feat)


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.matmul(q, k.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return self.cross_entropy_loss(logits, labels)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def train_cut(domainA_dir, domainB_dir, save_dir, epochs=CUT_EPOCHS, device=DEVICE,
              lambda_nce=LAMBDA_NCE, nce_layers=NCE_LAYERS, n_patches=NCE_NUM_PATCHES,
              lr_g=LR_G, lr_d=LR_D, batch_size=BATCH_CUT):
    """
    Train CUT generator A -> B using unpaired images.
    domainA_dir: source client split root
    domainB_dir: reference client split root
    """
    ensure_dir(save_dir)

    dsA = ImageTreeDataset(domainA_dir, transform=cut_img_tfms)
    dsB = ImageTreeDataset(domainB_dir, transform=cut_img_tfms)

    max_samples = max(len(dsA), len(dsB))
    samplerA = RandomSampler(dsA, replacement=True, num_samples=max_samples)
    samplerB = RandomSampler(dsB, replacement=True, num_samples=max_samples)

    loaderA = DataLoader(dsA, batch_size=batch_size, sampler=samplerA, drop_last=True, num_workers=2, pin_memory=PIN_MEMORY)
    loaderB = DataLoader(dsB, batch_size=batch_size, sampler=samplerB, drop_last=True, num_workers=2, pin_memory=PIN_MEMORY)

    G = CUTGenerator().to(device)
    D = PatchDiscriminator().to(device)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    ngf = 64
    layer_nc = {1: ngf, 2: ngf * 2, 3: ngf * 4, 4: ngf * 8, 5: ngf * 8}
    nce_mlps = {}
    for l in nce_layers:
        nc = layer_nc.get(l, ngf * 4)
        nce_mlps[str(l)] = PatchSampleF(nc, inner_dim=256).to(device)

    nce_loss = PatchNCELoss().to(device)
    criterion_GAN = nn.MSELoss().to(device)

    mlp_params = []
    for m in nce_mlps.values():
        mlp_params += list(m.parameters())

    opt_G = optim.Adam(list(G.parameters()) + mlp_params, lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    real_label = 0.9
    fake_label = 0.0

    print(f"[CUT] Training {os.path.basename(domainA_dir)} -> {os.path.basename(domainB_dir)} for {epochs} epochs")
    iterB = iter(loaderB)

    for epoch in range(epochs):
        G.train()
        D.train()

        loop = tqdm(loaderA, desc=f"CUT Epoch {epoch + 1}/{epochs}")
        for real_A, _ in loop:
            try:
                real_B, _ = next(iterB)
            except StopIteration:
                iterB = iter(loaderB)
                real_B, _ = next(iterB)

            real_A = real_A.to(device, non_blocking=True)
            real_B = real_B.to(device, non_blocking=True)

            real_A_in = real_A * 2.0 - 1.0
            real_B_in = real_B * 2.0 - 1.0

            fake_B, feats_A = G(real_A_in)
            feats_fake = G.encode(fake_B)

            # -----------------------
            # Train Discriminator
            # -----------------------
            opt_D.zero_grad(set_to_none=True)
            pred_real = D(real_B_in)
            valid_tensor = torch.full_like(pred_real, real_label, device=device)
            loss_D_real = criterion_GAN(pred_real, valid_tensor)

            pred_fake = D(fake_B.detach())
            fake_tensor = torch.full_like(pred_fake, fake_label, device=device)
            loss_D_fake = criterion_GAN(pred_fake, fake_tensor)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            # -----------------------
            # Train Generator
            # -----------------------
            opt_G.zero_grad(set_to_none=True)

            pred_fake_for_g = D(fake_B)
            valid_tensor = torch.full_like(pred_fake_for_g, real_label, device=device)
            loss_G_GAN = criterion_GAN(pred_fake_for_g, valid_tensor)

            loss_NCE = 0.0
            for l in nce_layers:
                idx = l - 1
                if idx < 0 or idx >= len(feats_A):
                    continue

                feat_q = feats_A[idx]
                feat_k = feats_fake[idx]

                proj = nce_mlps[str(l)]
                q_proj = proj(feat_q)
                k_proj = proj(feat_k)

                Bn, Cn, Hn, Wn = q_proj.shape
                total_positions = Hn * Wn
                num_patches = min(n_patches, total_positions)

                patch_idx = torch.randperm(total_positions, device=device)[:num_patches]

                q_patches = q_proj.view(Bn, Cn, -1)[:, :, patch_idx]
                k_patches = k_proj.view(Bn, Cn, -1)[:, :, patch_idx]

                q_flat = q_patches.permute(0, 2, 1).reshape(-1, Cn)
                k_flat = k_patches.permute(0, 2, 1).reshape(-1, Cn)

                loss_NCE = loss_NCE + nce_loss(q_flat, k_flat)

            loss_NCE = loss_NCE * lambda_nce
            loss_G = loss_G_GAN + loss_NCE
            loss_G.backward()
            opt_G.step()

            loop.set_postfix({
                "loss_G": float(loss_G.item()),
                "loss_D": float(loss_D.item()),
                "loss_NCE": float(loss_NCE.item()) if torch.is_tensor(loss_NCE) else float(loss_NCE),
            })

        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "mlps": {k: v.state_dict() for k, v in nce_mlps.items()},
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        }, os.path.join(save_dir, f"cut_epoch_{epoch + 1}.pth"))

    torch.save({
        "G": G.state_dict(),
        "mlps": {k: v.state_dict() for k, v in nce_mlps.items()},
    }, os.path.join(save_dir, "cut_final.pth"))

    print(f"[CUT] Finished and saved to {save_dir}")
    return G.cpu()


def load_cut_generator_from_ckpt(ckpt_path, device=DEVICE):
    ck = torch.load(ckpt_path, map_location="cpu")
    G = CUTGenerator()
    if isinstance(ck, dict) and "G" in ck:
        G.load_state_dict(ck["G"])
    else:
        G.load_state_dict(ck)
    return G.to(device)


def harmonize_folder_with_generator(generator, src_dir, dst_dir, device=DEVICE, size=(224, 224)):
    """
    Preserve folder structure under src_dir.
    Save all harmonized images as .jpg.
    """
    ensure_dir(dst_dir)
    generator = generator.to(device)
    generator.eval()

    with torch.no_grad():
        for dirpath, _, filenames in os.walk(src_dir):
            for fname in filenames:
                if not is_image_file(fname):
                    continue

                src_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(src_path, src_dir)
                rel_base, _ = os.path.splitext(rel_path)
                dst_path = os.path.join(dst_dir, rel_base + ".jpg")
                ensure_dir(os.path.dirname(dst_path))

                img = Image.open(src_path).convert("RGB")
                inp = cut_img_tfms(img).unsqueeze(0).to(device) * 2.0 - 1.0
                out, _ = generator(inp)
                out = (out.squeeze(0).detach().cpu().clamp(-1, 1) + 1.0) / 2.0
                out_img = transforms.ToPILImage()(out)
                out_img = out_img.convert("RGB")
                out_img.save(dst_path, quality=95)


def make_reference_comparison_grid(original_root, harmonized_root, reference_images, save_path, size=(224, 224)):
    """
    One figure:
    row 1 = original images
    row 2 = harmonized images
    row 3 = amplified abs difference
    columns = the four clients
    If the requested basename is missing, falls back to the first available image
    for that client.
    """
    client_order = list(reference_images.keys())
    n = len(client_order)

    originals = []
    harmonized = []
    diffs = []
    titles = []

    for client in client_order:
        preferred_basename = reference_images.get(client, None)
        orig_path, orig_split = find_sample_image_for_client(original_root, client, preferred_basename)
        hm_path, hm_split = find_sample_image_for_client(harmonized_root, client, preferred_basename)

        if orig_path is None:
            raise FileNotFoundError(f"No image found for client {client} under original root: {original_root}")
        if hm_path is None:
            raise FileNotFoundError(f"No image found for client {client} under harmonized root: {harmonized_root}")

        if preferred_basename is not None and os.path.basename(orig_path) != preferred_basename:
            print(f"[Grid] Warning: {client} original file '{preferred_basename}' not found. Using '{os.path.basename(orig_path)}' instead.")
        if preferred_basename is not None and os.path.basename(hm_path) != preferred_basename:
            print(f"[Grid] Warning: {client} harmonized file '{preferred_basename}' not found. Using '{os.path.basename(hm_path)}' instead.")

        orig = np.array(Image.open(orig_path).convert("RGB").resize(size))
        hm = np.array(Image.open(hm_path).convert("RGB").resize(size))
        diff = np.clip(np.abs(orig.astype(np.int16) - hm.astype(np.int16)) * 4, 0, 255).astype(np.uint8)

        originals.append(orig)
        harmonized.append(hm)
        diffs.append(diff)
        titles.append(f"{client}\n{orig_split}")

    fig, axs = plt.subplots(3, n, figsize=(4.5 * n, 11))
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    for i in range(n):
        axs[0, i].imshow(originals[i])
        axs[0, i].axis("off")
        axs[0, i].set_title(titles[i], fontsize=10)

        axs[1, i].imshow(harmonized[i])
        axs[1, i].axis("off")

        axs[2, i].imshow(diffs[i])
        axs[2, i].axis("off")

    fig.text(0.01, 0.83, "Original images", rotation=90, va="center", fontsize=12)
    fig.text(0.01, 0.50, "Harmonized images", rotation=90, va="center", fontsize=12)
    fig.text(0.01, 0.18, "Difference images", rotation=90, va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison grid to: {save_path}")


# =========================================================
# Harmonization pipeline
# =========================================================
def harmonize_all_clients_with_cut():
    """
    1) Train CUT A->reference for each non-reference client
    2) Apply generator to train/val/test
    3) Keep class folder structure intact
    4) Save the comparison grid
    """
    ensure_dir(HARMONIZED_ROOT)
    ensure_dir(CUT_DIR)

    cut_models = {}

    # Train or load CUT generators
    for client in CLIENT_NAMES:
        if client == REFERENCE_CLIENT:
            continue

        save_dir = os.path.join(CUT_DIR, f"{client}_to_{REFERENCE_CLIENT}")
        final_ckpt = os.path.join(save_dir, "cut_final.pth")

        if os.path.exists(final_ckpt):
            print(f"[CUT] Loading existing generator for {client} -> {REFERENCE_CLIENT}")
            cut_models[client] = load_cut_generator_from_ckpt(final_ckpt, device=DEVICE).cpu()
        else:
            src_root = os.path.join(DATA_ROOT, client, "train")
            ref_root = os.path.join(DATA_ROOT, REFERENCE_CLIENT, "train")
            G = train_cut(
                domainA_dir=src_root,
                domainB_dir=ref_root,
                save_dir=save_dir,
                epochs=CUT_EPOCHS,
                device=DEVICE,
                lambda_nce=LAMBDA_NCE,
                nce_layers=NCE_LAYERS,
                n_patches=NCE_NUM_PATCHES,
                lr_g=LR_G,
                lr_d=LR_D,
                batch_size=BATCH_CUT,
            )
            cut_models[client] = G.cpu()

    # Harmonize all splits
    for client in CLIENT_NAMES:
        print(f"\n[HARM] Processing client: {client}")
        for split in ["train", "val", "test"]:
            src_split = os.path.join(DATA_ROOT, client, split)
            dst_split = os.path.join(HARMONIZED_ROOT, client, split)

            if client == REFERENCE_CLIENT:
                copy_tree_as_jpg(src_split, dst_split)
                print(f"  Copied reference split as JPG: {dst_split}")
            else:
                if os.path.exists(dst_split):
                    import shutil
                    shutil.rmtree(dst_split)
                ensure_dir(dst_split)
                print(f"  Harmonizing {split}...")
                harmonize_folder_with_generator(
                    generator=cut_models[client],
                    src_dir=src_split,
                    dst_dir=dst_split,
                    device=DEVICE,
                    size=(IMG_SIZE, IMG_SIZE),
                )

    grid_path = os.path.join(OUTPUT_DIR, "comparison_grid_original_vs_harmonized.png")
    make_reference_comparison_grid(
        original_root=DATA_ROOT,
        harmonized_root=HARMONIZED_ROOT,
        reference_images=REFERENCE_IMAGES,
        save_path=grid_path,
        size=(IMG_SIZE, IMG_SIZE),
    )


# =========================================================
# Main Federated Classification on Harmonized Data
# =========================================================
def main():
    harmonize_all_clients_with_cut()

    paths = set_dataset_paths(HARMONIZED_ROOT, CLIENT_NAMES)

    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

    train_ds_all = build_combined_dataset(train_datasets)
    val_ds_all = build_combined_dataset(val_datasets)
    test_ds_all = build_combined_dataset(test_datasets)

    num_classes = len(class_names)

    print("\nClasses:", class_names)
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

    global_model = build_model(num_classes).to(DEVICE)
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

        norm_weights = [w / total_train_size for w in local_weights]
        global_model.load_state_dict(average_state_dicts_weighted([m.state_dict() for m in local_models], norm_weights))

        global_val_loss, global_val_acc, _, _ = run_epoch(
            global_model,
            global_val_loader,
            criterion,
            optimizer=None,
            train=False,
        )

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

    global_model.load_state_dict(best_model_wts)

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
    print(os.path.join(OUTPUT_DIR, "comparison_grid_original_vs_harmonized.png"))


if __name__ == "__main__":
    main()