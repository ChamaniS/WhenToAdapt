import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
import math
import random
import shutil
import stat
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torchvision.transforms as T
import sys
from unet import UNET

output_file = r"/lustre06/project/6008975/csj5/narvalenv/cyclegan_breast.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

# ============================================================
# Settings
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client_names = ["BUSBRA","BUS", "BUSI", "UDIAT"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 12
COMM_ROUNDS = 10

# CycleGAN settings
TRAIN_CYCLEGAN = True
CYCLEGAN_EPOCHS = 60
BATCH_CYCLEGAN = 4
CYCLEGAN_IMAGE_SIZE = 224
reference_idx = 0

# Segmentation settings
SEG_BATCH_SIZE = 4
IMG_SIZE = 224

start_time = time.time()
out_dir = "Outputs_cycleGAN"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Path to pre-prepared splits root
# Must contain:
# <splits_root>/<CLIENT>/{train,val,test}/{images,masks}
# ============================================================
splits_root = r"/lustre06/project/6008975/csj5/Breasttumor_seg/"

# ============================================================
# Per-client expected extensions
# ============================================================
client_ext_map = {
    "BUS": ((".png",), (".png",)),
    "BUSBRA": ((".png",), (".png",)),
    "BUSI": ((".png",), (".png",)),
    "UDIAT": ((".png",), (".png",)),
}

# ============================================================
# Build split directories
# ============================================================
train_img_dirs = []
train_mask_dirs = []
val_img_dirs = []
val_mask_dirs = []
test_img_dirs = []
test_mask_dirs = []

required_subpaths = [
    ("train", "images"), ("train", "masks"),
    ("val", "images"), ("val", "masks"),
    ("test", "images"), ("test", "masks"),
]

for cname in client_names:
    base = os.path.join(splits_root, cname)
    missing = []
    for split, sub in required_subpaths:
        p = os.path.join(base, split, sub)
        if not os.path.isdir(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            f"Missing required split folders for client '{cname}':\n" + "\n".join(missing)
        )

    train_img_dirs.append(os.path.join(base, "train", "images"))
    train_mask_dirs.append(os.path.join(base, "train", "masks"))
    val_img_dirs.append(os.path.join(base, "val", "images"))
    val_mask_dirs.append(os.path.join(base, "val", "masks"))
    test_img_dirs.append(os.path.join(base, "test", "images"))
    test_mask_dirs.append(os.path.join(base, "test", "masks"))

print("Using these dataset splits:")
for i, name in enumerate(client_names):
    print(f"Client {i}: {name}")
    print(f"  train imgs: {train_img_dirs[i]}  masks: {train_mask_dirs[i]}")
    print(f"  val   imgs: {val_img_dirs[i]}    masks: {val_mask_dirs[i]}")
    print(f"  test  imgs: {test_img_dirs[i]}   masks: {test_mask_dirs[i]}")

# ============================================================
# Utilities
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def copy_tree_force(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst, onerror=_on_rm_error)
    shutil.copytree(src, dst)

def save_image(arr, path):
    Image.fromarray(arr).save(path)

def _unnormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr

def _mask_to_uint8(mask_tensor):
    m = mask_tensor.detach().cpu().numpy()
    if m.ndim == 3:
        m = np.squeeze(m, axis=0)
    m = (m > 0.5).astype(np.uint8) * 255
    return m

def make_unet():
    try:
        return UNET(in_channels=3, num_classes=1)
    except TypeError:
        return UNET(in_channels=3, out_channels=1)

# ============================================================
# Dataset for segmentation
# ============================================================
class SkinPairDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, img_exts=None, mask_exts=None, return_filename=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_filename = return_filename

        if img_exts is None:
            img_exts = (".png")
        if mask_exts is None:
            mask_exts = (".png")

        self.img_exts = tuple(e.lower() for e in img_exts)
        self.mask_exts = tuple(e.lower() for e in mask_exts)

        files = []
        for ext in self.img_exts:
            files.extend(glob.glob(os.path.join(self.img_dir, f"*{ext}")))
        files = sorted(files)

        pairs = []
        missing_masks = 0

        for img_path in files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None

            for mext in self.mask_exts:
                candidate = os.path.join(self.mask_dir, stem + mext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                alt_candidates = (
                    [os.path.join(self.mask_dir, stem + "_mask" + mext) for mext in self.mask_exts] +
                    [os.path.join(self.mask_dir, stem + "-mask" + mext) for mext in self.mask_exts] +
                    [os.path.join(self.mask_dir, stem.replace("_lesion", "") + mext) for mext in self.mask_exts]
                )
                for c in alt_candidates:
                    if os.path.exists(c):
                        mask_path = c
                        break

            if mask_path is None:
                missing_masks += 1
                continue

            pairs.append((img_path, mask_path))

        if len(pairs) == 0:
            raise ValueError(f"No image-mask pairs found in {img_dir} / {mask_dir}. Missing masks: {missing_masks}")

        self.pairs = pairs
        if missing_masks > 0:
            print(f"Warning: {missing_masks} images in {img_dir} had no matching masks and were skipped.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = (np.asarray(mask) > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            mask = np.expand_dims(mask.astype(np.float32), 0)

        if self.return_filename:
            return img, mask, os.path.basename(img_path)
        return img, mask

# ============================================================
# Image-only dataset for CycleGAN training
# ============================================================
class ImageFolderSimple(Dataset):
    def __init__(self, folder, size=(224, 224), augment=False):
        self.files = sorted([
            p for p in glob.glob(os.path.join(folder, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        if len(self.files) == 0:
            raise ValueError(f"No image files found in {folder}")

        self.size = size
        self.augment = augment
        self.base_trans = T.Compose([
            T.Resize(self.size),
            T.CenterCrop(self.size),
            T.ToTensor()
        ])
        self.aug_trans = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10)
        ])

    def __len__(self):
        return max(1, len(self.files))

    def __getitem__(self, idx):
        p = self.files[idx % len(self.files)]
        img = Image.open(p).convert("RGB")
        if self.augment:
            img = self.aug_trans(img)
        t = self.base_trans(img)  # [0,1]
        return t

# ============================================================
# CycleGAN modules
# ============================================================
def conv_block(in_ch, out_ch, k=3, stride=1, padding=1, norm=True, relu=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=False))
    if relu:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

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

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7, 1, 0), nn.Tanh()]
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
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("InstanceNorm2d") != -1 or classname.find("BatchNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

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

# ============================================================
# CycleGAN training
# ============================================================
def train_cyclegan(domainA_dir, domainB_dir, save_dir, epochs=CYCLEGAN_EPOCHS, device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)

    dsA = ImageFolderSimple(domainA_dir, size=(CYCLEGAN_IMAGE_SIZE, CYCLEGAN_IMAGE_SIZE), augment=True)
    dsB = ImageFolderSimple(domainB_dir, size=(CYCLEGAN_IMAGE_SIZE, CYCLEGAN_IMAGE_SIZE), augment=True)

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

    optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=1e-4, betas=(0.5, 0.999))

    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    real_label = 1.0
    fake_label = 0.0

    print(f"[CycleGAN] Train {domainA_dir} <-> {domainB_dir} for {epochs} epochs")

    iterB = iter(loaderB)
    for epoch in range(epochs):
        loop = tqdm(loaderA, desc=f"CycleGAN Epoch {epoch + 1}/{epochs}")
        for real_A in loop:
            try:
                real_B = next(iterB)
            except StopIteration:
                iterB = iter(loaderB)
                real_B = next(iterB)

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # -----------------------
            # Train Generators
            # -----------------------
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

            # -----------------------
            # Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            pred_real_A = D_A(real_A)
            valid_real_A = torch.full_like(pred_real_A, real_label, device=device)
            loss_D_real_A = criterion_GAN(pred_real_A, valid_real_A)

            fake_A_detached = fake_A_pool.query(fake_A.detach())
            pred_fake_A = D_A(fake_A_detached)
            fake_A_label = torch.full_like(pred_fake_A, fake_label, device=device)
            loss_D_fake_A = criterion_GAN(pred_fake_A, fake_A_label)

            loss_D_A = 0.5 * (loss_D_real_A + loss_D_fake_A)
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            # Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            valid_real_B = torch.full_like(pred_real_B, real_label, device=device)
            loss_D_real_B = criterion_GAN(pred_real_B, valid_real_B)

            fake_B_detached = fake_B_pool.query(fake_B.detach())
            pred_fake_B = D_B(fake_B_detached)
            fake_B_label = torch.full_like(pred_fake_B, fake_label, device=device)
            loss_D_fake_B = criterion_GAN(pred_fake_B, fake_B_label)

            loss_D_B = 0.5 * (loss_D_real_B + loss_D_fake_B)
            loss_D_B.backward()
            optimizer_D_B.step()

            loop.set_postfix({
                "loss_G": float(loss_G.item()),
                "loss_D_A": float(loss_D_A.item()),
                "loss_D_B": float(loss_D_B.item())
            })

        torch.save({
            "G_A2B": G_A2B.state_dict(),
            "G_B2A": G_B2A.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "opt_G": optimizer_G.state_dict(),
        }, os.path.join(save_dir, f"cyclegan_epoch_{epoch + 1}.pth"))

    torch.save({
        "G_A2B": G_A2B.state_dict(),
        "G_B2A": G_B2A.state_dict(),
    }, os.path.join(save_dir, "cyclegan_final.pth"))

    print(f"[CycleGAN] finished and saved to {save_dir}")
    return G_A2B

# ============================================================
# Harmonization
# ============================================================
def harmonize_folder_with_generator(generator, src_dir, dst_dir, mask_src_dir=None, mask_dst_dir=None, device=DEVICE, size=(224, 224)):
    os.makedirs(dst_dir, exist_ok=True)
    if mask_src_dir and mask_dst_dir:
        os.makedirs(mask_dst_dir, exist_ok=True)

    tf = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor()
    ])

    generator = generator.to(device)
    generator.eval()

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

    with torch.no_grad():
        for p in sorted([f for f in glob.glob(os.path.join(src_dir, "*")) if f.lower().endswith(valid_exts)]):
            img = Image.open(p).convert("RGB")
            inp = tf(img).unsqueeze(0).to(device) * 2.0 - 1.0  # [0,1] -> [-1,1]
            out = generator(inp)
            out = (out.squeeze(0).detach().cpu().clamp(-1, 1) + 1.0) / 2.0
            out_img = T.ToPILImage()(out)

            basename = os.path.basename(p)
            out_img.save(os.path.join(dst_dir, basename))

            if mask_src_dir and mask_dst_dir:
                base, _ = os.path.splitext(basename)
                copied = False

                direct = os.path.join(mask_src_dir, basename)
                if os.path.exists(direct):
                    shutil.copy(direct, os.path.join(mask_dst_dir, basename))
                    copied = True

                if not copied:
                    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                        alt = os.path.join(mask_src_dir, base + ext)
                        if os.path.exists(alt):
                            shutil.copy(alt, os.path.join(mask_dst_dir, base + ext))
                            copied = True
                            break

                if not copied:
                    print(f"Warning: no mask found for {basename} in {mask_src_dir}")

# ============================================================
# Segmentation transforms
# ============================================================
tr_tf = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

visual_val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE)
])

# ============================================================
# Loader helpers
# ============================================================
def get_loader(img_dir, mask_dir, transform, client_name=None, batch_size=SEG_BATCH_SIZE, shuffle=True, return_filename=False):
    if client_name is not None and client_name in client_ext_map:
        img_exts, mask_exts = client_ext_map[client_name]
    else:
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    ds = SkinPairDataset(
        img_dir,
        mask_dir,
        transform=transform,
        img_exts=img_exts,
        mask_exts=mask_exts,
        return_filename=return_filename
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def get_global_test_loader(transform, batch_size=SEG_BATCH_SIZE):
    datasets = []
    for i, cname in enumerate(client_names):
        if cname in client_ext_map:
            img_exts, mask_exts = client_ext_map[cname]
        else:
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        ds = SkinPairDataset(
            test_img_dirs[i],
            test_mask_dirs[i],
            transform=transform,
            img_exts=img_exts,
            mask_exts=mask_exts,
            return_filename=False
        )
        datasets.append(ds)

    global_test_ds = ConcatDataset(datasets)
    return DataLoader(global_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

# ============================================================
# Metrics / loss
# ============================================================
def compute_metrics(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    TP = (pred * target).sum().item()
    TN = ((1 - pred) * (1 - target)).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()

    dice_with_bg = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou_with_bg = (TP + smooth) / (TP + FP + FN + smooth)
    acc = (TP + TN) / (TP + TN + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)

    if target.sum() == 0:
        dice_no_bg = 1.0 if pred.sum() == 0 else 0.0
        iou_no_bg = 1.0 if pred.sum() == 0 else 0.0
    else:
        intersection = (pred * target).sum().item()
        dice_no_bg = (2 * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
        iou_no_bg = (intersection + smooth) / (pred.sum().item() + target.sum().item() - intersection + smooth)

    return dict(
        dice_with_bg=dice_with_bg,
        dice_no_bg=dice_no_bg,
        iou_with_bg=iou_with_bg,
        iou_no_bg=iou_no_bg,
        accuracy=acc,
        precision=precision,
        recall=recall,
        specificity=specificity,
    )

def average_metrics(metrics_list):
    if len(metrics_list) == 0:
        return {}
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_loss_fn(device):
    return smp.losses.DiceLoss(mode="binary", from_logits=True).to(device)

def average_models_weighted(models, weights):
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd

# ============================================================
# Training / Evaluation
# ============================================================
def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss = 0.0
    metrics = []

    for _ in range(LOCAL_EPOCHS):
        for batch in tqdm(loader, leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, target = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch format in train_local")

            if target.dim() == 3:
                target = target.unsqueeze(1)
            target = target.float()

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            preds = model(data)
            loss = loss_fn(preds, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            metrics.append(compute_metrics(preds.detach(), target))

    avg_metrics = average_metrics(metrics)
    print("Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return total_loss / max(1, len(loader)), avg_metrics

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss = 0.0
    metrics = []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
        else:
            raise RuntimeError("Unexpected batch format in evaluate")

        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        preds = model(data)
        loss = loss_fn(preds, target)

        total_loss += loss.item()
        metrics.append(compute_metrics(preds, target))

    avg_metrics = average_metrics(metrics)
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return total_loss / max(1, len(loader)), avg_metrics

# ============================================================
# Visualizations
# ============================================================
def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))

    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Per-client Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_cyclefed.png"))
    plt.close()

    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Per-client IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_cyclefed.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Test Dice Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg_cyclefed.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Test IoU Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg_cyclefed.png"))
    plt.close()

def save_transformed_samples(img_dir, mask_dir, transform, client_name, out_base, n_samples=8, prefix="harmonized", client_for_ext=None):
    if client_for_ext is None:
        client_for_ext = client_name

    if client_for_ext in client_ext_map:
        img_exts, mask_exts = client_ext_map[client_for_ext]
    else:
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    ds = SkinPairDataset(
        img_dir,
        mask_dir,
        transform=transform,
        img_exts=img_exts,
        mask_exts=mask_exts,
        return_filename=False
    )

    dest = os.path.join(out_base, f"{client_name}", prefix)
    ensure_dir(dest)

    num = min(n_samples, len(ds))
    for i in range(num):
        img_t, mask_t = ds[i]
        if isinstance(img_t, np.ndarray):
            img_arr = img_t.astype(np.uint8)
        else:
            img_arr = _unnormalize_image(img_t)
        save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))

        if isinstance(mask_t, np.ndarray):
            m_arr = (mask_t > 0.5).astype(np.uint8) * 255
        else:
            m_arr = _mask_to_uint8(mask_t)
        save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))

def make_comparison_grid_and_histograms_updated_original_vs_hm(original_dir, hm_dir, client_name, out_base, n_samples=7):
    base_dest = os.path.join(out_base, "ComparisonGrid", client_name)
    ensure_dir(base_dest)
    diffs_dest = os.path.join(base_dest, "diffs")
    ensure_dir(diffs_dest)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    fnames = sorted([f for f in os.listdir(original_dir) if f.lower().endswith(valid_exts)])[:n_samples]
    if len(fnames) == 0:
        return

    top_imgs = []
    mid_imgs = []
    diff_imgs = []
    short_names = []

    for fname in fnames:
        orig_p = os.path.join(original_dir, fname)
        hm_p = os.path.join(hm_dir, fname)
        if not os.path.exists(hm_p):
            continue

        orig = np.array(Image.open(orig_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        hm = np.array(Image.open(hm_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        diff = np.clip(np.abs(orig.astype(np.int16) - hm.astype(np.int16)) * 4, 0, 255).astype(np.uint8)

        top_imgs.append(orig)
        mid_imgs.append(hm)
        diff_imgs.append(diff)
        short_names.append(fname)

        save_image(orig, os.path.join(base_dest, f"orig_{fname}"))
        save_image(hm, os.path.join(base_dest, f"hm_{fname}"))
        save_image(diff, os.path.join(diffs_dest, f"diff_{fname}"))

    n = len(top_imgs)
    if n == 0:
        return

    fig, axs = plt.subplots(3, n, figsize=(3 * n, 6))
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    for i in range(n):
        axs[0, i].imshow(top_imgs[i])
        axs[0, i].axis("off")
        axs[0, i].set_title(short_names[i][:12])

        axs[1, i].imshow(mid_imgs[i])
        axs[1, i].axis("off")

        axs[2, i].imshow(diff_imgs[i])
        axs[2, i].axis("off")

    fig.suptitle(f"Harmonized (CycleGAN) vs. Original: {client_name}", fontsize=16, y=0.98)
    fig.text(0.01, 0.82, "Original\nimages", fontsize=12, va="center", rotation="vertical")
    fig.text(0.01, 0.50, "Harmonized\nimages", fontsize=12, va="center", rotation="vertical")
    fig.text(0.01, 0.18, "Amplified\nDifference", fontsize=12, va="center", rotation="vertical")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dest, f"comparison_{client_name}.png"))
    plt.close()
    print(f"Saved comparison grid for {client_name} at {base_dest}")

@torch.no_grad()
def save_test_predictions(global_model, test_loader, client_name, out_base=None, max_to_save=16, device_arg=None):
    if out_base is None:
        out_base = out_dir
    device = DEVICE if device_arg is None else device_arg

    global_model.eval()
    latest_dir = os.path.join(out_base, "TestPreds", client_name, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir, onerror=_on_rm_error)
    ensure_dir(latest_dir)

    saved = 0
    for idx, batch in enumerate(test_loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, target, fnames = batch
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
            fnames = None
        else:
            raise RuntimeError("Unexpected batch format from test_loader")

        data = data.to(device)
        preds = global_model(data)
        probs = torch.sigmoid(preds)
        bin_mask = (probs > 0.5).float()

        bsz = data.size(0)
        for b in range(bsz):
            mask_arr = _mask_to_uint8(bin_mask[b])

            if fnames is not None:
                try:
                    base, _ = os.path.splitext(str(fnames[b]))
                    fname = f"{base}_pred.png"
                except Exception:
                    fname = f"{client_name}_pred_mask_{idx}_{b}.png"
            else:
                fname = f"{client_name}_pred_mask_{idx}_{b}.png"

            save_image(mask_arr, os.path.join(latest_dir, fname))
            saved += 1
            if saved >= max_to_save:
                break

        if saved >= max_to_save:
            break

    print(f"Saved {saved} prediction masks for {client_name} in {latest_dir}")

# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Train/load CycleGAN for each non-reference client
    # --------------------------------------------------------
    cyclegan_models = {}

    for i in range(NUM_CLIENTS):
        if i == reference_idx:
            print(f"[CycleGAN] Skipping reference client {client_names[i]}")
            continue

        a_dir = train_img_dirs[i]
        b_dir = train_img_dirs[reference_idx]
        save_dir = os.path.join(out_dir, "CycleGAN", f"{client_names[i]}_to_{client_names[reference_idx]}")
        os.makedirs(save_dir, exist_ok=True)

        final_ckpt = os.path.join(save_dir, "cyclegan_final.pth")
        if os.path.exists(final_ckpt):
            ck = torch.load(final_ckpt, map_location="cpu")
            G_A2B = ResnetGenerator()
            G_A2B.load_state_dict(ck["G_A2B"])
            cyclegan_models[i] = G_A2B.cpu()
            print(f"[CycleGAN] Loaded existing generator for {client_names[i]} -> {client_names[reference_idx]}")
        else:
            if not TRAIN_CYCLEGAN:
                raise FileNotFoundError(f"CycleGAN checkpoint not found: {final_ckpt}")
            G_A2B = train_cyclegan(a_dir, b_dir, save_dir, epochs=CYCLEGAN_EPOCHS, device=DEVICE)
            cyclegan_models[i] = G_A2B.cpu()

    # --------------------------------------------------------
    # 2) Harmonize all splits (train/val/test) into new folders
    # --------------------------------------------------------
    hist_base = os.path.join(out_dir, "CycleGAN_Harmonized")

    hm_train_dirs = []
    hm_train_mask_dirs = []
    hm_val_dirs = []
    hm_val_mask_dirs = []
    hm_test_dirs = []
    hm_test_mask_dirs = []

    for i in range(NUM_CLIENTS):
        cname = client_names[i]

        dst_train = os.path.join(hist_base, cname, "train_images")
        dst_val = os.path.join(hist_base, cname, "val_images")
        dst_test = os.path.join(hist_base, cname, "test_images")
        dst_train_mask = os.path.join(hist_base, cname, "train_masks")
        dst_val_mask = os.path.join(hist_base, cname, "val_masks")
        dst_test_mask = os.path.join(hist_base, cname, "test_masks")

        if i == reference_idx:
            print(f"[HARM] Copying reference client unchanged: {cname}")
            copy_tree_force(train_img_dirs[i], dst_train)
            copy_tree_force(val_img_dirs[i], dst_val)
            copy_tree_force(test_img_dirs[i], dst_test)
            copy_tree_force(train_mask_dirs[i], dst_train_mask)
            copy_tree_force(val_mask_dirs[i], dst_val_mask)
            copy_tree_force(test_mask_dirs[i], dst_test_mask)
        else:
            print(f"[HARM] Harmonizing {cname} -> {client_names[reference_idx]}")
            ensure_dir(dst_train)
            ensure_dir(dst_val)
            ensure_dir(dst_test)
            ensure_dir(dst_train_mask)
            ensure_dir(dst_val_mask)
            ensure_dir(dst_test_mask)

            G = cyclegan_models[i]
            harmonize_folder_with_generator(
                G, train_img_dirs[i], dst_train,
                mask_src_dir=train_mask_dirs[i], mask_dst_dir=dst_train_mask,
                device=DEVICE, size=(CYCLEGAN_IMAGE_SIZE, CYCLEGAN_IMAGE_SIZE)
            )
            harmonize_folder_with_generator(
                G, val_img_dirs[i], dst_val,
                mask_src_dir=val_mask_dirs[i], mask_dst_dir=dst_val_mask,
                device=DEVICE, size=(CYCLEGAN_IMAGE_SIZE, CYCLEGAN_IMAGE_SIZE)
            )
            harmonize_folder_with_generator(
                G, test_img_dirs[i], dst_test,
                mask_src_dir=test_mask_dirs[i], mask_dst_dir=dst_test_mask,
                device=DEVICE, size=(CYCLEGAN_IMAGE_SIZE, CYCLEGAN_IMAGE_SIZE)
            )

        hm_train_dirs.append(dst_train)
        hm_train_mask_dirs.append(dst_train_mask)
        hm_val_dirs.append(dst_val)
        hm_val_mask_dirs.append(dst_val_mask)
        hm_test_dirs.append(dst_test)
        hm_test_mask_dirs.append(dst_test_mask)

    print("[HARM] Harmonization complete. Harmonized datasets written under:", hist_base)

    # --------------------------------------------------------
    # 3) Save a few harmonized samples and comparison grids
    # --------------------------------------------------------
    visuals_base = os.path.join(out_dir, "HarmonizedSamples_CycleGAN")
    for i in range(NUM_CLIENTS):
        cname = client_names[i]
        save_transformed_samples(
            hm_val_dirs[i], hm_val_mask_dirs[i], val_tf,
            cname, visuals_base, n_samples=7, prefix="harmonized",
            client_for_ext=cname
        )
        save_transformed_samples(
            hm_train_dirs[i], hm_train_mask_dirs[i], tr_tf,
            cname, visuals_base, n_samples=7, prefix="augmented",
            client_for_ext=cname
        )
        make_comparison_grid_and_histograms_updated_original_vs_hm(
            val_img_dirs[i], hm_val_dirs[i], cname, visuals_base
        )

    # --------------------------------------------------------
    # 4) Federated segmentation training on harmonized data
    # --------------------------------------------------------
    global_model = make_unet().to(DEVICE)
    global_test_loader = get_global_test_loader(val_tf, batch_size=SEG_BATCH_SIZE)

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r + 1}/{COMM_ROUNDS}]")
        local_models = []
        weights = []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(
                hm_train_dirs[i],
                hm_train_mask_dirs[i],
                tr_tf,
                client_name=client_names[i],
                batch_size=SEG_BATCH_SIZE,
                shuffle=True,
                return_filename=False
            )

            val_loader = get_loader(
                hm_val_dirs[i],
                hm_val_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
                batch_size=SEG_BATCH_SIZE,
                shuffle=False,
                return_filename=False
            )

            print(f"[Client {client_names[i]}] Local training")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset)
            weights.append(sz)
            total_sz += sz

        if total_sz == 0:
            raise RuntimeError("Total training size across clients is 0. Check your split folders.")

        norm_weights = [w / total_sz for w in weights]
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

        # Global test on all clients combined
        global_test_loss, global_test_metrics = evaluate(
            global_test_loader,
            global_model,
            get_loss_fn(DEVICE),
            split="Global Test"
        )

        rm = {
            "global_test_loss": global_test_loss,
            "global_dice_no_bg": global_test_metrics.get("dice_no_bg", 0),
            "global_iou_no_bg": global_test_metrics.get("iou_no_bg", 0),
            "global_accuracy": global_test_metrics.get("accuracy", 0),
            "global_precision": global_test_metrics.get("precision", 0),
            "global_recall": global_test_metrics.get("recall", 0),
            "global_specificity": global_test_metrics.get("specificity", 0),
        }

        # Per-client test metrics
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(
                hm_test_dirs[i],
                hm_test_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
                batch_size=SEG_BATCH_SIZE,
                shuffle=False,
                return_filename=False
            )
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)

            save_test_predictions(
                global_model,
                test_loader,
                client_names[i],
                out_base=out_dir,
                max_to_save=int(len(test_loader.dataset)),
                device_arg=DEVICE
            )

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

        print(
            f"[GLOBAL TEST AFTER ROUND {r + 1}] "
            f"Dice(no bg): {rm['global_dice_no_bg']:.4f} | "
            f"IoU(no bg): {rm['global_iou_no_bg']:.4f} | "
            f"Acc: {rm['global_accuracy']:.4f} | "
            f"Prec: {rm['global_precision']:.4f} | "
            f"Recall: {rm['global_recall']:.4f} | "
            f"Spec: {rm['global_specificity']:.4f}"
        )

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()