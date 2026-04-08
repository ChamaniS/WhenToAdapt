import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import random
import json
import csv
import shutil
import stat
from glob import glob
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets.folder import default_loader
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

DATA_ROOT = r"/xxxxBreasttumor_classi_renamed/"
OUTPUT_DIR = "breast_classi_federated_cyclegan"
MODEL_NAME = "efficientnet_b0_breast_tumor_fedavg_cyclegan.pth"

WEIGHTS_PATH = r"/xxxxxxxx/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"
output_file = r"/xxxxxxxx/cycleGAN_breast_classi.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

CLIENT_NAMES =  ["BUSBRA", "BUS", "BUSI", "UDIAT"]
REFERENCE_IDX = 0  

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
NUM_WORKERS = 0
IMG_SIZE = 224

# CycleGAN settings
CYCLEGAN_EPOCHS = 20
BATCH_CYCLEGAN = 1
CYCLEGAN_LR = 1e-3
CYCLEGAN_BETA1 = 0.5
CYCLEGAN_BETA2 = 0.999
CYCLEGAN_POOL_SIZE = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# Will be set after reading dataset classes
CLASS_NAMES = None

# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
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
# File / folder helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def copy_tree_force(src: str, dst: str):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst, onerror=_on_rm_error)
    shutil.copytree(src, dst)

def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

def find_file_by_name(root: str, filename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == filename:
                return os.path.join(dirpath, fn)
    return None

def get_relative_path_of_file(root: str, filename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == filename:
                return os.path.relpath(os.path.join(dirpath, fn), root)
    return None

def read_rgb_image(path: str, size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img)

def save_three_row_reference_grid(original_path: str, harmonized_path: str, out_path: str, title: str):
    orig = read_rgb_image(original_path, (IMG_SIZE, IMG_SIZE))
    harm = read_rgb_image(harmonized_path, (IMG_SIZE, IMG_SIZE))
    diff = np.abs(orig.astype(np.int16) - harm.astype(np.int16)).astype(np.uint8)

    fig, axs = plt.subplots(3, 1, figsize=(6, 14))
    axs[0].imshow(orig)
    axs[0].set_title(f"{title} - Original")
    axs[0].axis("off")

    axs[1].imshow(harm)
    axs[1].set_title(f"{title} - Harmonized")
    axs[1].axis("off")

    axs[2].imshow(diff)
    axs[2].set_title(f"{title} - Absolute Difference")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# Dataset helpers
# =========================================================
def set_dataset_paths(root: str, client_names: List[str]) -> Dict[str, Dict[str, str]]:
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

# =========================================================
# CycleGAN datasets
# =========================================================
class ImageFolderSimple(Dataset):
    def __init__(self, folder=None, files_list=None, size=(224, 224), augment=False, validate=True):
        if files_list is not None:
            cand = [
                p for p in files_list
                if os.path.isfile(p) and is_image_file(p)
            ]
        elif folder is not None:
            cand = sorted([
                p for p in glob(os.path.join(folder, "**", "*"), recursive=True)
                if os.path.isfile(p) and is_image_file(p)
            ])
        else:
            cand = []

        good = []
        if validate:
            for p in cand:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    good.append(p)
                except Exception:
                    print(f"[ImageFolderSimple] skipping unreadable file: {p}")
        else:
            good = cand

        if len(good) == 0:
            raise RuntimeError("No valid images found in folder/list provided")

        self.files = good
        self.size = size
        self.augment = augment
        self.base_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])
        self.aug_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])

    def __len__(self):
        return max(1, len(self.files))

    def __getitem__(self, idx):
        p = self.files[idx % len(self.files)]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            for _ in range(5):
                rp = random.choice(self.files)
                try:
                    img = Image.open(rp).convert("RGB")
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Failed to open images in folder/list; last tried: {p}")

        if self.augment:
            img = self.aug_trans(img)
        t = self.base_trans(img)
        return t

class ImageListDataset(Dataset):
    def __init__(self, files: List[str], size=(224, 224), augment=False, validate=True):
        self.ds = ImageFolderSimple(folder=None, files_list=files, size=size, augment=augment, validate=validate)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

# =========================================================
# CycleGAN modules
# =========================================================
class ResnetBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, nblocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        mult = 1
        for _ in range(n_down):
            mult_prev = mult
            mult *= 2
            model += [
                nn.Conv2d(ngf * mult_prev, ngf * mult, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(ngf * mult),
                nn.ReLU(True),
            ]
        for _ in range(nblocks):
            model += [ResnetBlock(ngf * mult)]
        for _ in range(n_down):
            mult_prev = mult
            mult //= 2
            model += [
                nn.ConvTranspose2d(ngf * mult_prev, ngf * mult, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult),
                nn.ReLU(True),
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7, 1, 0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_ch, ndf, kw, 2, padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*sequence)

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
            try:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            except Exception:
                pass
        if hasattr(m, "bias") and m.bias is not None:
            try:
                nn.init.constant_(m.bias.data, 0.0)
            except Exception:
                pass

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)

# =========================================================
# CycleGAN training
# =========================================================
def collect_all_image_paths_for_client(client_root: str) -> List[str]:
    paths = []
    for split in ("train", "val", "test"):
        sp = os.path.join(client_root, split)
        if not os.path.isdir(sp):
            continue
        for dirpath, _, filenames in os.walk(sp):
            for fn in filenames:
                if is_image_file(fn):
                    paths.append(os.path.join(dirpath, fn))
    return sorted(paths)

def train_cyclegan_from_lists(listA, listB, epochs=CYCLEGAN_EPOCHS, device=DEVICE):
    dsA = ImageListDataset(listA, size=(IMG_SIZE, IMG_SIZE), augment=True)
    dsB = ImageListDataset(listB, size=(IMG_SIZE, IMG_SIZE), augment=True)

    loaderA = DataLoader(dsA, batch_size=BATCH_CYCLEGAN, shuffle=True, drop_last=True, num_workers=0)
    loaderB = DataLoader(dsB, batch_size=BATCH_CYCLEGAN, shuffle=True, drop_last=True, num_workers=0)

    G_A2B = ResnetGenerator().to(device)
    G_B2A = ResnetGenerator().to(device)
    D_A = NLayerDiscriminator().to(device)
    D_B = NLayerDiscriminator().to(device)

    for net in [G_A2B, G_B2A, D_A, D_B]:
        net.apply(weights_init_normal)

    criterion_GAN = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    optimizer_G = optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=CYCLEGAN_LR,
        betas=(CYCLEGAN_BETA1, CYCLEGAN_BETA2),
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=CYCLEGAN_LR, betas=(CYCLEGAN_BETA1, CYCLEGAN_BETA2))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=CYCLEGAN_LR, betas=(CYCLEGAN_BETA1, CYCLEGAN_BETA2))

    fake_A_pool = ImagePool(CYCLEGAN_POOL_SIZE)
    fake_B_pool = ImagePool(CYCLEGAN_POOL_SIZE)

    real_label = 1.0
    fake_label = 0.0

    print(f"[CycleGAN] Train in-memory lists A({len(listA)}) <-> B({len(listB)}) for {epochs} epochs")
    iterB = iter(loaderB)

    for epoch in range(epochs):
        loop = tqdm(loaderA, desc=f"CycleGAN {epoch+1}/{epochs}", leave=False)
        for real_A in loop:
            try:
                real_B = next(iterB)
            except StopIteration:
                iterB = iter(loaderB)
                real_B = next(iterB)

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # ----------------------
            # Train Generators
            # ----------------------
            optimizer_G.zero_grad()

            same_B = G_A2B(real_B)
            loss_id_B = criterion_identity(same_B, real_B) * 10.0

            same_A = G_B2A(real_A)
            loss_id_A = criterion_identity(same_A, real_A) * 10.0

            fake_B = G_A2B(real_A)
            pred_fake_B = D_B(fake_B)
            valid_B = torch.full_like(pred_fake_B, real_label, device=device)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, valid_B)

            fake_A = G_B2A(real_B)
            pred_fake_A = D_A(fake_A)
            valid_A = torch.full_like(pred_fake_A, real_label, device=device)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, valid_A)

            rec_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * 5.0

            rec_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B) * 5.0

            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()

            # ----------------------
            # Train D_A
            # ----------------------
            optimizer_D_A.zero_grad()
            pred_real = D_A(real_A)
            valid = torch.full_like(pred_real, real_label, device=device)
            loss_D_real = criterion_GAN(pred_real, valid)

            fake_A_detached = fake_A_pool.query(fake_A.detach())
            pred_fake = D_A(fake_A_detached)
            fake = torch.full_like(pred_fake, fake_label, device=device)
            loss_D_fake = criterion_GAN(pred_fake, fake)

            loss_D_A = 0.5 * (loss_D_real + loss_D_fake)
            loss_D_A.backward()
            optimizer_D_A.step()

            # ----------------------
            # Train D_B
            # ----------------------
            optimizer_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            valid_B2 = torch.full_like(pred_real_B, real_label, device=device)
            loss_D_real_B = criterion_GAN(pred_real_B, valid_B2)

            fake_B_detached = fake_B_pool.query(fake_B.detach())
            pred_fake_B = D_B(fake_B_detached)
            fake_B_lbl = torch.full_like(pred_fake_B, fake_label, device=device)
            loss_D_fake_B = criterion_GAN(pred_fake_B, fake_B_lbl)

            loss_D_B = 0.5 * (loss_D_real_B + loss_D_fake_B)
            loss_D_B.backward()
            optimizer_D_B.step()

            loop.set_postfix({
                "loss_G": float(loss_G.item()),
                "loss_D_A": float(loss_D_A.item()),
                "loss_D_B": float(loss_D_B.item()),
            })

    return G_A2B.cpu(), G_B2A.cpu()

def harmonize_client_with_generator(generator, client_root, out_base, device=DEVICE, size=(IMG_SIZE, IMG_SIZE)):
    if generator is None:
        print(f"[HARM] No generator provided for {client_root} - skipping harmonization")
        return

    try:
        generator = generator.to(device)
    except Exception:
        pass

    generator.eval()
    splits = ["train", "val", "test"]
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    for split in splits:
        split_src = os.path.join(client_root, split)
        if not os.path.isdir(split_src):
            print(f"[HARM] Missing split '{split}' in {client_root}, skipping")
            continue

        for cls in os.listdir(split_src):
            cls_src = os.path.join(split_src, cls)
            if not os.path.isdir(cls_src):
                continue

            dst_dir = os.path.join(out_base, split, cls)
            ensure_dir(dst_dir)

            for fname in sorted(os.listdir(cls_src)):
                if not is_image_file(fname):
                    continue

                src_path = os.path.join(cls_src, fname)
                dst_path = os.path.join(dst_dir, fname)

                if os.path.exists(dst_path):
                    continue

                try:
                    with Image.open(src_path) as pil:
                        img = pil.convert("RGB")

                    inp = tf(img).unsqueeze(0).to(device)
                    inp = inp * 2.0 - 1.0

                    with torch.no_grad():
                        out = generator(inp)

                    out = out.squeeze(0).detach().cpu().clamp(-1.0, 1.0)
                    out = (out + 1.0) / 2.0
                    out_img = transforms.ToPILImage()(out)
                    out_img.save(dst_path)
                except Exception as e:
                    print(f"[HARM] failed harmonizing '{src_path}' -> '{dst_path}': {e}")

def create_reference_comparison_grids(original_root: str, harmonized_root: str, client_name: str, out_dir: str):
    ensure_dir(out_dir)
    ref_fname = REFERENCE_IMAGES.get(client_name, None)
    if ref_fname is None:
        print(f"[VIS] No reference image specified for {client_name}; skipping.")
        return

    orig_rel = get_relative_path_of_file(original_root, ref_fname)
    harm_rel = get_relative_path_of_file(harmonized_root, ref_fname)

    if orig_rel is None:
        print(f"[VIS] Original reference image not found for {client_name}: {ref_fname}")
        return
    if harm_rel is None:
        print(f"[VIS] Harmonized reference image not found for {client_name}: {ref_fname}")
        return

    original_path = os.path.join(original_root, orig_rel)
    harmonized_path = os.path.join(harmonized_root, harm_rel)

    out_path = os.path.join(out_dir, f"reference_comparison_{client_name}.png")
    save_three_row_reference_grid(
        original_path=original_path,
        harmonized_path=harmonized_path,
        out_path=out_path,
        title=client_name,
    )
    print(f"[VIS] Saved reference comparison grid for {client_name} -> {out_path}")

# =========================================================
# Classification dataset helpers
# =========================================================
class PathListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, loader=default_loader):
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
    canon_map = {c.lower(): i for i, c in enumerate(class_names)}

    for cls_folder in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls_folder)
        if not os.path.isdir(cls_path):
            continue

        key = cls_folder.lower()
        if key not in canon_map:
            print(f"Warning: unknown class folder '{cls_folder}' in {split_dir}; skipping")
            continue

        label = canon_map[key]
        for fn in os.listdir(cls_path):
            if is_image_file(fn):
                samples.append((os.path.join(cls_path, fn), label))
    return samples

def make_multi_client_dataloaders_from_roots(
    client_roots,
    class_names,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    per_client_dataloaders = []
    train_samples_all, val_samples_all, test_samples_all = [], [], []

    for client_root in client_roots:
        tr = gather_samples_from_client_split(client_root, "train", class_names)
        va = gather_samples_from_client_split(client_root, "val", class_names)
        te = gather_samples_from_client_split(client_root, "test", class_names)

        print(f"[DATA] client {client_root} -> train:{len(tr)} val:{len(va)} test:{len(te)}")

        train_samples_all.extend(tr)
        val_samples_all.extend(va)
        test_samples_all.extend(te)

        train_ds = PathListDataset(tr, transform=train_tf)
        val_ds = PathListDataset(va, transform=val_tf)
        test_ds = PathListDataset(te, transform=val_tf)

        per_client_dataloaders.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
        })

    combined_train_ds = PathListDataset(train_samples_all, transform=train_tf)
    combined_val_ds = PathListDataset(val_samples_all, transform=val_tf)
    combined_test_ds = PathListDataset(test_samples_all, transform=val_tf)

    dataloaders_combined = {
        "train": DataLoader(combined_train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
        "val": DataLoader(combined_val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
        "test": DataLoader(combined_test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
    }

    sizes = {
        "train": len(combined_train_ds),
        "val": len(combined_val_ds),
        "test": len(combined_test_ds),
    }

    return dataloaders_combined, sizes, class_names, combined_train_ds, per_client_dataloaders

def compute_class_weights_from_dataset(dataset):
    if not hasattr(dataset, "samples") or len(dataset.samples) == 0:
        if CLASS_NAMES is not None:
            return torch.ones(len(CLASS_NAMES), dtype=torch.float32)
        return torch.ones(1, dtype=torch.float32)

    targets = [s[1] for s in dataset.samples]
    num_classes = max(targets) + 1 if len(targets) > 0 else (len(CLASS_NAMES) if CLASS_NAMES is not None else 1)

    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

# =========================================================
# Classification model helpers
# =========================================================
def smart_load_weights_into_efficientnet(model: nn.Module, weights_path: str):
    if not os.path.isfile(weights_path):
        print(f"[WEIGHTS] File not found: {weights_path}")
        return

    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
    except Exception as e:
        print(f"[WEIGHTS] Failed to read checkpoint '{weights_path}': {e}")
        return

    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "ema_state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        print(f"[WEIGHTS] Unsupported checkpoint format in {weights_path}")
        return

    ckpt_sd = {}
    for k, v in checkpoint.items():
        if isinstance(v, torch.Tensor):
            kk = k.replace("module.", "")
            ckpt_sd[kk] = v

    model_sd = model.state_dict()
    filtered = {}
    loaded = 0

    for k, v in ckpt_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            if k.startswith("classifier.1"):
                continue
            filtered[k] = v
            loaded += 1

    if loaded == 0:
        print(
            "[WEIGHTS] No matching parameters were loaded from the local checkpoint.\n"
            "          This usually means the file is from a different EfficientNet implementation."
        )
        return

    model_sd.update(filtered)
    model.load_state_dict(model_sd, strict=False)
    print(f"[WEIGHTS] Loaded {loaded} matching parameters from {weights_path}")

def build_model(num_classes):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    smart_load_weights_into_efficientnet(model, WEIGHTS_PATH)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def average_state_dicts_weighted(models, weights):
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd

# =========================================================
# Training / evaluation
# =========================================================
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

    epoch_loss = running_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    epoch_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
    return epoch_loss, epoch_acc, all_targets, all_preds

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

    macro_specificity = float(np.mean(per_class_specificity)) if len(per_class_specificity) > 0 else 0.0
    return per_class_specificity, macro_specificity

def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR, save_cm=True):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(targets, preds, labels=labels)
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
    print(classification_report(targets, preds, labels=labels, target_names=class_names, digits=4, zero_division=0))

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
    if len(results) == 0:
        return
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
    global CLASS_NAMES

    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    # Build per-client datasets
    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

    CLASS_NAMES = class_names

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

    # =====================================================
    # Step 1: Train CycleGANs and harmonize 4 clients
    # =====================================================
    print("\n==============================")
    print("STEP 1: CycleGAN harmonization")
    print("==============================")

    client_paths = []
    for i, client_root in enumerate([os.path.join(DATA_ROOT, c) for c in CLIENT_NAMES]):
        pths = collect_all_image_paths_for_client(client_root)
        print(f"[DATA] client {CLIENT_NAMES[i]} has {len(pths)} images (train/val/test combined)")
        client_paths.append(pths)

    cyclegan_generators = {}
    reference_client_root = os.path.join(DATA_ROOT, CLIENT_NAMES[REFERENCE_IDX])

    for i, client_root in enumerate([os.path.join(DATA_ROOT, c) for c in CLIENT_NAMES]):
        if i == REFERENCE_IDX:
            print(f"[HARM] skipping CycleGAN for reference client {CLIENT_NAMES[i]}")
            cyclegan_generators[i] = None
            continue

        listA = client_paths[i]
        listB = client_paths[REFERENCE_IDX]

        if len(listA) == 0 or len(listB) == 0:
            print(f"[WARN] empty lists for CycleGAN A({len(listA)}) B({len(listB)}). Skipping generator training.")
            cyclegan_generators[i] = None
            continue

        G_A2B, G_B2A = train_cyclegan_from_lists(listA, listB, epochs=CYCLEGAN_EPOCHS, device=DEVICE)
        cyclegan_generators[i] = G_A2B.cpu()

    harmonized_base = os.path.join(OUTPUT_DIR, "CycleGAN_Harmonized")
    ensure_dir(harmonized_base)

    harmonized_roots = []
    original_client_roots = [os.path.join(DATA_ROOT, c) for c in CLIENT_NAMES]

    for i, client_root in enumerate(original_client_roots):
        cname = CLIENT_NAMES[i]
        out_client_root = os.path.join(harmonized_base, cname)
        ensure_dir(out_client_root)

        if i == REFERENCE_IDX:
            for sp in ["train", "val", "test"]:
                src_sp = os.path.join(client_root, sp)
                dst_sp = os.path.join(out_client_root, sp)
                if os.path.exists(dst_sp):
                    shutil.rmtree(dst_sp, onerror=_on_rm_error)
                if os.path.exists(src_sp):
                    copy_tree_force(src_sp, dst_sp)
        else:
            G = cyclegan_generators.get(i, None)
            if G is None:
                print(f"[WARN] no generator for client {cname}, copying as-is")
                for sp in ["train", "val", "test"]:
                    src_sp = os.path.join(client_root, sp)
                    dst_sp = os.path.join(out_client_root, sp)
                    if os.path.exists(dst_sp):
                        shutil.rmtree(dst_sp, onerror=_on_rm_error)
                    if os.path.exists(src_sp):
                        copy_tree_force(src_sp, dst_sp)
            else:
                print(f"[HARM] Harmonizing client {cname} -> {CLIENT_NAMES[REFERENCE_IDX]}")
                harmonize_client_with_generator(G, client_root, out_client_root, device=DEVICE, size=(IMG_SIZE, IMG_SIZE))

        harmonized_roots.append(out_client_root)

    print("[HARM] Harmonization finished. Harmonized datasets at:", harmonized_base)

    ref_grid_dir = os.path.join(OUTPUT_DIR, "Reference_Comparison_Grids")
    ensure_dir(ref_grid_dir)
    for i, cname in enumerate(CLIENT_NAMES):
        create_reference_comparison_grids(
            original_root=original_client_roots[i],
            harmonized_root=harmonized_roots[i],
            client_name=cname,
            out_dir=ref_grid_dir
        )

    # =====================================================
    # Step 2: Build dataloaders from harmonized roots
    # =====================================================
    combined_loaders, combined_sizes, class_names, combined_train_ds, per_client_dataloaders = \
        make_multi_client_dataloaders_from_roots(
            harmonized_roots,
            class_names,
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY and (DEVICE.type == "cuda")
        )

    client_train_sizes = [len(per_client_dataloaders[i]["train"].dataset) for i in range(len(per_client_dataloaders))]
    total_train = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
    print("client train sizes:", client_train_sizes)

    # =====================================================
    # Step 3: Initialize classification model
    # =====================================================
    global_model = build_model(num_classes=num_classes).to(DEVICE)
    print(f"Global model EfficientNet-B0 with {count_parameters(global_model):,} trainable params")

    round_metrics = []
    history = defaultdict(list)
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())
    start_time = time.time()

    per_client_test_acc_history = {i: [] for i in range(len(per_client_dataloaders))}
    per_client_perclass_history = {i: [] for i in range(len(per_client_dataloaders))}

    # =====================================================
    # Federated rounds
    # =====================================================
    for r in range(COMM_ROUNDS):
        print("\n" + "=" * 40)
        print(f"COMM ROUND {r + 1}/{COMM_ROUNDS}")
        print("=" * 40)

        local_models = []
        weights = []
        round_summary = {"round": r + 1}

        for i, client in enumerate(per_client_dataloaders):
            print(f"[CLIENT {i}] {CLIENT_NAMES[i]}: local training")

            local_model = copy.deepcopy(global_model).to(DEVICE)
            train_ds = client["train"].dataset

            client_cw = compute_class_weights_from_dataset(train_ds).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=1e-4)

            client_epoch_losses = []
            client_epoch_accs = []

            for ep in range(LOCAL_EPOCHS):
                tr_loss, tr_acc, _, _ = run_epoch(
                    local_model,
                    client["train"],
                    criterion,
                    optimizer=optimizer,
                    train=True,
                )
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(
                local_model,
                client["val"],
                criterion,
                optimizer=None,
                train=False,
            )
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models.append(local_model.cpu())
            local_weights = len(train_ds)
            weights.append(local_weights)

            round_summary[f"client{i}_train_loss"] = float(np.mean(client_epoch_losses))
            round_summary[f"client{i}_train_acc"] = float(np.mean(client_epoch_accs))
            round_summary[f"client{i}_localval_loss"] = float(val_loss)
            round_summary[f"client{i}_localval_acc"] = float(val_acc)

        total_train_size = sum(weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        norm_weights = [w / total_train_size for w in weights]
        global_model.load_state_dict(average_state_dicts_weighted(local_models, norm_weights))
        global_model.to(DEVICE)

        global_val_loss, global_val_acc, _, _ = run_epoch(
            global_model,
            global_val_loader,
            nn.CrossEntropyLoss(),
            optimizer=None,
            train=False,
        )

        round_summary["global_val_loss"] = float(global_val_loss)
        round_summary["global_val_acc"] = float(global_val_acc)

        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
            print("\nSaved best global model.")

        print("\n" + "=" * 30)
        print(f"GLOBAL TEST AFTER ROUND {r + 1} (ALL CLIENTS TOGETHER)")
        print("=" * 30)

        global_test_result = evaluate_loader(
            global_model,
            global_test_loader,
            nn.CrossEntropyLoss(),
            class_names,
            title_prefix=f"global_round_{r + 1}",
            save_dir=OUTPUT_DIR,
            save_cm=True,
        )

        round_summary["global_test_loss"] = global_test_result["loss"]
        round_summary["global_test_acc"] = global_test_result["accuracy"]
        round_summary["global_test_precision"] = global_test_result["precision_macro"]
        round_summary["global_test_recall"] = global_test_result["recall_macro"]
        round_summary["global_test_f1"] = global_test_result["f1_macro"]
        round_summary["global_test_kappa"] = global_test_result["kappa"]
        round_summary["global_test_specificity"] = global_test_result["specificity_macro"]

        print("\n" + "=" * 30)
        print(f"INDIVIDUAL CLIENT TESTS AFTER ROUND {r + 1}")
        print("=" * 30)

        for i, client in enumerate(per_client_dataloaders):
            client_result = evaluate_loader(
                global_model,
                client["test"],
                nn.CrossEntropyLoss(),
                class_names,
                title_prefix=f"{CLIENT_NAMES[i]}_round_{r + 1}",
                save_dir=OUTPUT_DIR,
                save_cm=True,
            )

            round_summary[f"client{i}_test_loss"] = client_result["loss"]
            round_summary[f"client{i}_test_acc"] = client_result["accuracy"]
            round_summary[f"client{i}_test_precision"] = client_result["precision_macro"]
            round_summary[f"client{i}_test_recall"] = client_result["recall_macro"]
            round_summary[f"client{i}_test_f1"] = client_result["f1_macro"]
            round_summary[f"client{i}_test_kappa"] = client_result["kappa"]
            round_summary[f"client{i}_test_specificity"] = client_result["specificity_macro"]

            per_client_test_acc_history[i].append(float(client_result["accuracy"]))
            per_client_perclass_history[i].append(client_result)

        round_metrics.append(round_summary)

        history["round"].append(r + 1)
        history["train_loss"].append(float(np.mean([round_summary[f"client{i}_train_loss"] for i in range(len(CLIENT_NAMES))])))
        history["train_acc"].append(float(np.mean([round_summary[f"client{i}_train_acc"] for i in range(len(CLIENT_NAMES))])))
        history["val_loss"].append(round_summary["global_val_loss"])
        history["val_acc"].append(round_summary["global_val_acc"])
        history["global_test_acc"].append(round_summary["global_test_acc"])

        print(
            f"\n[ROUND {r + 1}] "
            f"Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | "
            f"Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f} | "
            f"Global Test Acc: {history['global_test_acc'][-1]:.4f}"
        )

        plot_round_curves(history, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    global_model.load_state_dict(best_model_wts)

    # =====================================================
    # Final evaluation
    # =====================================================
    final_results = []

    print("\n==============================")
    print("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print("==============================")
    result_all = evaluate_loader(
        global_model,
        global_test_loader,
        nn.CrossEntropyLoss(),
        class_names,
        title_prefix="all_clients_final",
        save_dir=OUTPUT_DIR,
        save_cm=True,
    )
    final_results.append(result_all)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name, client in zip(CLIENT_NAMES, per_client_dataloaders):
        result_client = evaluate_loader(
            global_model,
            client["test"],
            nn.CrossEntropyLoss(),
            class_names,
            title_prefix=f"{client_name}_final",
            save_dir=OUTPUT_DIR,
            save_cm=True,
        )
        final_results.append(result_client)

    if len(round_metrics) > 0:
        with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"), "w") as f:
            json.dump(round_metrics, f, indent=2)

        with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(round_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(round_metrics)

    save_metrics_csv(final_results, os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    with open(os.path.join(OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    try:
        hist_path = os.path.join(OUTPUT_DIR, "per_client_perclass_history.txt")
        with open(hist_path, "w") as f:
            for i in range(len(per_client_perclass_history)):
                f.write(f"== Client {i} {CLIENT_NAMES[i]} ==\n")
                for rr_idx, cm in enumerate(per_client_perclass_history[i]):
                    f.write(f"Round {rr_idx+1}: accuracy={cm.get('accuracy', 'n/a')}, kappa={cm.get('kappa', 'n/a')}\n")
                    pcs = cm.get("per_class_specificity", [])
                    f.write(f"  specificity: {pcs}\n")
                f.write("\n")
        print("[HIST] Saved per-client per-class textual history to:", hist_path)
    except Exception as e:
        print("[HIST] failed writing per-class history:", e)

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "Reference_Comparison_Grids"))
    print(os.path.join(OUTPUT_DIR, "CycleGAN_Harmonized"))

if __name__ == "__main__":
    main()