# cycle_gan_harmonize_and_fedavg_tb_with_visuals_nosave.py
# No .pth/.ckpt/.csv saved. Comparison grids saved only as combined PNGs.
# Saves: global train/test acc/loss plots + one plot showing per-client TEST accuracy across rounds.
# Prints detailed per-client test metrics (including per-class table) after each global round.

import os
os.environ["KMP_DUPLICATE_OK"] = "TRUE"
import copy, random, shutil, stat
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as T
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torchvision
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, accuracy_score
)

# -------------------------
# Basic FS helpers
# -------------------------
def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def copy_tree_force(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst, onerror=_on_rm_error)
    shutil.copytree(src, dst)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Config - update paths and hyperparams
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TORCH = torch.device(DEVICE)
NUM_CLIENTS = 4
CYCLEGAN_EPOCHS = 30           # set low for quick runs; increase for real training
BATCH_CYCLEGAN = 1
COMM_ROUNDS = 10
LOCAL_EPOCHS = 6
OUT_DIR = "./Outputs_TB_cycleGAN"
ensure_dir(OUT_DIR)

CLIENT_ROOTS = [
    r"xxxxx\Projects\Data\Tuberculosis_Data\Shenzhen",
    r"xxxxx\Projects\Data\Tuberculosis_Data\Montgomery",
    r"xxxxx\Projects\Data\Tuberculosis_Data\TBX11K",
    r"xxxxx\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES = ["Shenzhen", "Montgomery", "TBX11K", "Pakistan"]
reference_idx = 3  # harmonize others to Pakistan

# classification config
ARCH = "densenet121"
PRETRAINED = True
IMG_SIZE = 224
BATCH_SIZE = 1
WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = False
PIN_MEMORY = True
DROPOUT_P = 0.5
SEED = 42
CLASS_NAMES = ["normal", "positive"]
NUM_CLASSES = len(CLASS_NAMES)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# -------------------------
# Plot filenames (only global + single per-client test accuracy plot)
# -------------------------
GLOBAL_TEST_ACC_FN  = os.path.join(OUT_DIR, "global_test_accuracy_rounds.png")
GLOBAL_TEST_LOSS_FN = os.path.join(OUT_DIR, "global_test_loss_rounds.png")
GLOBAL_TRAIN_ACC_FN = os.path.join(OUT_DIR, "global_train_accuracy_rounds.png")
GLOBAL_TRAIN_LOSS_FN= os.path.join(OUT_DIR, "global_train_loss_rounds.png")

PER_CLIENT_TEST_ACC_FN  = os.path.join(OUT_DIR, "per_client_test_accuracy_over_rounds.png")

# -------------------------
# Safe ImageFolderSimple: pre-validate files to avoid PIL worker crashes
# -------------------------
class ImageFolderSimple(Dataset):
    def __init__(self, folder=None, files_list=None, size=(224,224), augment=False, validate=True):
        if files_list is not None:
            cand = [p for p in files_list if os.path.isfile(p) and p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        elif folder is not None:
            cand = sorted([p for p in glob(os.path.join(folder, "**", "*"), recursive=True)
                       if p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')) and os.path.isfile(p)])
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
            raise RuntimeError(f"No valid images found in folder/list provided")
        self.files = good
        self.size = size
        self.augment = augment
        self.base_trans = T.Compose([T.Resize(self.size), T.CenterCrop(self.size), T.ToTensor()])
        self.aug_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10)])
    def __len__(self): return max(1, len(self.files))
    def __getitem__(self, idx):
        p = self.files[idx % len(self.files)]
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            for _ in range(5):
                rp = random.choice(self.files)
                try:
                    img = Image.open(rp).convert('RGB')
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Failed to open images in folder/list; last tried: {p}")
        if self.augment:
            img = self.aug_trans(img)
        t = self.base_trans(img)
        return t

# -------------------------
# Image-list dataset for CycleGAN (no temp folders)
# -------------------------
class ImageListDataset(Dataset):
    def __init__(self, files: List[str], size=(224,224), augment=False, validate=True):
        self.ds = ImageFolderSimple(folder=None, files_list=files, size=size, augment=augment, validate=validate)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx): return self.ds[idx]

# -------------------------
# CycleGAN minimal modules
# -------------------------
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
    def forward(self, x): return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, nblocks=6):
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_ch, ngf, 7, 1, 0, bias=False),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        n_down = 2
        mult = 1
        for i in range(n_down):
            mult_prev = mult
            mult *= 2
            model += [nn.Conv2d(ngf * mult_prev, ngf * mult, 3, 2, 1, bias=False),
                      nn.InstanceNorm2d(ngf * mult),
                      nn.ReLU(True)]
        for i in range(nblocks):
            model += [ResnetBlock(ngf * mult)]
        for i in range(n_down):
            mult_prev = mult
            mult //= 2
            model += [nn.ConvTranspose2d(ngf * mult_prev, ngf * mult, 3, 2, 1, output_padding=1, bias=False),
                      nn.InstanceNorm2d(ngf * mult),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7, 1, 0), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4; padw = 1
        sequence = [nn.Conv2d(in_ch, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, bias=False),
                         nn.InstanceNorm2d(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw, bias=False),
                     nn.InstanceNorm2d(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*sequence)
    def forward(self, x): return self.model(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            try:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            except Exception: pass
        if hasattr(m, 'bias') and m.bias is not None:
            try:
                nn.init.constant_(m.bias.data, 0.0)
            except Exception: pass

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
    def query(self, images):
        if self.pool_size == 0: return images
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

# -------------------------
# Train CycleGAN between two *lists* of image paths (unpaired)
# -------------------------
def train_cyclegan_from_lists(listA, listB, epochs=CYCLEGAN_EPOCHS, device=DEVICE_TORCH):
    dsA = ImageListDataset(listA, size=(IMG_SIZE,IMG_SIZE), augment=True)
    dsB = ImageListDataset(listB, size=(IMG_SIZE,IMG_SIZE), augment=True)
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

    fake_A_pool = ImagePool(50); fake_B_pool = ImagePool(50)
    real_label = 1.0; fake_label = 0.0

    print(f"[CycleGAN] Train in-memory lists A({len(listA)}) <-> B({len(listB)}) for {epochs} epochs")
    iterB = iter(loaderB)
    for epoch in range(epochs):
        loop = tqdm(loaderA, desc=f"Epoch {epoch+1}/{epochs}")
        for real_A in loop:
            try:
                real_B = next(iterB)
            except StopIteration:
                iterB = iter(loaderB)
                real_B = next(iterB)
            real_A = real_A.to(device); real_B = real_B.to(device)

            optimizer_G.zero_grad()
            same_B = G_A2B(real_B); loss_id_B = criterion_identity(same_B, real_B) * 10.0
            same_A = G_B2A(real_A); loss_id_A = criterion_identity(same_A, real_A) * 10.0

            fake_B = G_A2B(real_A)
            pred_fake_B = D_B(fake_B); valid = torch.full_like(pred_fake_B, real_label, device=device)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, valid)

            fake_A = G_B2A(real_B)
            pred_fake_A = D_A(fake_A); validA = torch.full_like(pred_fake_A, real_label, device=device)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, validA)

            rec_A = G_B2A(fake_B); loss_cycle_A = criterion_cycle(rec_A, real_A) * 5.0
            rec_B = G_A2B(fake_A); loss_cycle_B = criterion_cycle(rec_B, real_B) * 5.0

            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward(); optimizer_G.step()

            optimizer_D_A.zero_grad()
            pred_real = D_A(real_A); valid_label = torch.full_like(pred_real, real_label, device=device)
            loss_D_real = criterion_GAN(pred_real, valid_label)
            fake_A_detached = fake_A_pool.query(fake_A.detach()); pred_fake = D_A(fake_A_detached)
            fake_label_tensor = torch.full_like(pred_fake, fake_label, device=device)
            loss_D_fake = criterion_GAN(pred_fake, fake_label_tensor)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5; loss_D_A.backward(); optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            pred_real_B = D_B(real_B); valid_label_B = torch.full_like(pred_real_B, real_label, device=device)
            loss_D_real_B = criterion_GAN(pred_real_B, valid_label_B)
            fake_B_detached = fake_B_pool.query(fake_B.detach()); pred_fake_B = D_B(fake_B_detached)
            fake_label_tensorB = torch.full_like(pred_fake_B, fake_label, device=device)
            loss_D_fake_B = criterion_GAN(pred_fake_B, fake_label_tensorB)
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5; loss_D_B.backward(); optimizer_D_B.step()

            loop.set_postfix({"loss_G": loss_G.item(), "loss_D_A": loss_D_A.item(), "loss_D_B": loss_D_B.item()})

    # return CPU generators (no saving to disk)
    return G_A2B.cpu(), G_B2A.cpu()

# -------------------------
# Harmonize images for classification folder layout
# client_root/<split>/<class> -> out_base/<client_name>/<split>/<class>
# -------------------------
def harmonize_client_with_generator(generator, client_root, out_base, device=DEVICE_TORCH, size=(IMG_SIZE,IMG_SIZE)):
    if generator is None:
        print(f"[HARM] No generator provided for {client_root} - skipping harmonization")
        return
    try:
        generator = generator.to(device)
    except Exception:
        pass
    generator.eval()
    splits = ["train", "val", "test"]
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    for split in splits:
        split_src = os.path.join(client_root, split)
        if not os.path.isdir(split_src):
            print(f"[HARM] Missing split '{split}' in {client_root}, skipping split")
            continue
        for cls in os.listdir(split_src):
            cls_src = os.path.join(split_src, cls)
            if not os.path.isdir(cls_src):
                continue
            dst_dir = os.path.join(out_base, split, cls)
            ensure_dir(dst_dir)
            for fname in sorted(os.listdir(cls_src)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
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
                    out_img = T.ToPILImage()(out)
                    out_img.save(dst_path)
                except Exception as e:
                    print(f"[HARM] failed harmonizing '{src_path}' -> '{dst_path}': {e}")
                    continue

# -------------------------
# Classification dataset & dataloader helpers
# -------------------------
class PathListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None, loader=default_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader
    def __len__(self): return len(self.samples)
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

def make_multi_client_dataloaders_from_roots(client_roots, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY):
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

    per_client_dataloaders = []
    per_client_test_dsets = {}
    train_samples_all, val_samples_all, test_samples_all = [], [], []

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
    if not hasattr(dataset, 'samples'):
        return torch.ones(NUM_CLASSES, dtype=torch.float32)
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

# -------------------------
# Classification model & training helpers
# -------------------------
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
    if len(models) == 0: raise ValueError("No models to average")
    if len(models) != len(weights): raise ValueError("models and weights must have same length")
    sum_w = float(sum(weights))
    if sum_w == 0.0: raise ValueError("Sum of weights is zero")
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
                pass
            avg_sd[k] = acc
    return avg_sd

def train_local(model, dataloader, criterion, optimizer, device, epochs=LOCAL_EPOCHS, use_amp=False):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type=="cuda") else None
    logs = []
    for ep in range(epochs):
        model.train()
        running_loss = 0.0; correct = 0; total = 0
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
        epoch_acc = correct / max(1, total) if total>0 else 0.0
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

# -------------------------
# Comparison grid: original vs harmonized (sample up to n_samples)
# -------------------------
def select_val_pairs_for_comparison(original_val_dir, hm_val_dir, n_samples=7):
    orig_files = {}
    hm_files = set()
    if os.path.isdir(original_val_dir):
        for cls in os.listdir(original_val_dir):
            cls_path = os.path.join(original_val_dir, cls)
            if not os.path.isdir(cls_path): continue
            for fn in os.listdir(cls_path):
                if fn.lower().endswith(('.png','.jpg','.jpeg')):
                    orig_files[fn] = os.path.join(cls_path, fn)
    if os.path.isdir(hm_val_dir):
        for cls in os.listdir(hm_val_dir):
            cls_path = os.path.join(hm_val_dir, cls)
            if not os.path.isdir(cls_path): continue
            for fn in os.listdir(cls_path):
                if fn.lower().endswith(('.png','.jpg','.jpeg')):
                    hm_files.add(fn)
    common = [fn for fn in sorted(orig_files.keys()) if fn in hm_files]
    if len(common) == 0:
        common = sorted(list(orig_files.keys()))
    chosen = common[:n_samples]
    pairs = []
    for fn in chosen:
        orig_p = orig_files.get(fn)
        hm_p = None
        if os.path.isdir(hm_val_dir):
            for cls in os.listdir(hm_val_dir):
                cls_path = os.path.join(hm_val_dir, cls)
                if not os.path.isdir(cls_path): continue
                candidate = os.path.join(cls_path, fn)
                if os.path.exists(candidate):
                    hm_p = candidate
                    break
        if orig_p and hm_p:
            pairs.append((orig_p, hm_p, fn))
    return pairs

def make_comparison_grid(original_val_dir, hm_val_dir, client_name, out_base, n_samples=7):
    base_dest = os.path.join(out_base, "ComparisonGrid")
    ensure_dir(base_dest)
    fn_pairs = select_val_pairs_for_comparison(original_val_dir, hm_val_dir, n_samples=n_samples)
    if len(fn_pairs) == 0:
        print(f"[VIS] no matching pairs for comparison for {client_name}")
        return
    top_imgs = []
    mid_imgs = []
    titles = []
    for orig_p, hm_p, fn in fn_pairs:
        try:
            orig = np.array(Image.open(orig_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
            hm = np.array(Image.open(hm_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
            top_imgs.append(orig)
            mid_imgs.append(hm)
            titles.append(fn)
        except Exception as e:
            print(f"[VIS] skipping pair {fn} due to read error: {e}")
    n = len(top_imgs)
    if n == 0:
        print(f"[VIS] no readable pairs for {client_name}")
        return
    fig, axs = plt.subplots(2, n, figsize=(2.5*n, 5))
    if n == 1:
        axs = np.array([[axs[0]],[axs[1]]])
    for i in range(n):
        axs[0, i].imshow(top_imgs[i]); axs[0, i].axis('off'); axs[0, i].set_title(titles[i][:12])
        axs[1, i].imshow(mid_imgs[i]); axs[1, i].axis('off')
    fig.suptitle(f"Harmonized (CycleGAN) vs Original: {client_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(base_dest, f"comparison_{client_name}.png")
    try:
        plt.savefig(out_png); plt.close(fig)
        print(f"[VIS] Saved comparison grid for {client_name} at {out_png}")
    except Exception as e:
        print("[VIS] failed saving comparison grid:", e)
        plt.close(fig)

# -------------------------
# Utility: collect all image paths for a client (train/val/test)
# -------------------------
def collect_all_image_paths_for_client(client_root):
    paths = []
    for split in ("train","val","test"):
        sp = os.path.join(client_root, split)
        if not os.path.isdir(sp): continue
        for cls in os.listdir(sp):
            cls_path = os.path.join(sp, cls)
            if not os.path.isdir(cls_path): continue
            for fn in os.listdir(cls_path):
                if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    paths.append(os.path.join(cls_path, fn))
    return paths

# -------------------------
# Plot helpers (TRAIN + TEST)
# Only save global plots + single per-client test accuracy over rounds plot
# -------------------------
def save_round_series_plots(
    round_results,
    per_client_test_acc_history,
    out_dir=OUT_DIR
):
    rounds = list(range(1, len(round_results)+1))
    # global metrics
    gtest_acc  = [rr.get("global_test_acc", 0.0) for rr in round_results]
    gtest_loss = [rr.get("global_test_loss", 0.0) for rr in round_results]
    gtrain_acc  = [rr.get("global_train_acc", 0.0) for rr in round_results]
    gtrain_loss = [rr.get("global_train_loss", 0.0) for rr in round_results]

    # Global TEST Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(rounds, gtest_acc, marker='o')
    plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Global Test Accuracy")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(GLOBAL_TEST_ACC_FN); plt.close()

    # Global TEST Loss
    plt.figure(figsize=(6,4))
    plt.plot(rounds, gtest_loss, marker='o')
    plt.xlabel("Global Round"); plt.ylabel("Test Loss"); plt.title("Global Test Loss")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(GLOBAL_TEST_LOSS_FN); plt.close()

    # Global TRAIN Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(rounds, gtrain_acc, marker='o')
    plt.xlabel("Global Round"); plt.ylabel("Train Accuracy"); plt.title("Global Train Accuracy")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(GLOBAL_TRAIN_ACC_FN); plt.close()

    # Global TRAIN Loss
    plt.figure(figsize=(6,4))
    plt.plot(rounds, gtrain_loss, marker='o')
    plt.xlabel("Global Round"); plt.ylabel("Train Loss"); plt.title("Global Train Loss")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(GLOBAL_TRAIN_LOSS_FN); plt.close()

    # Single plot: per-client TEST accuracy across rounds (one PNG)
    plt.figure(figsize=(8,5))
    for i, name in enumerate(CLIENT_NAMES):
        vals = per_client_test_acc_history.get(i, [])
        plt.plot(range(1, len(vals)+1), vals, marker='o', label=name)
    plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Per-client Test Accuracy (per round)"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    try:
        plt.savefig(PER_CLIENT_TEST_ACC_FN); plt.close()
        print(f"[PLOT] Saved per-client test accuracy across rounds -> {PER_CLIENT_TEST_ACC_FN}")
    except Exception as e:
        print("[PLOT] failed saving per-client test accuracy plot:", e)
        plt.close()

# -------------------------
# Top-level flow
# -------------------------
def main():
    print("DEVICE:", DEVICE)
    cyclegan_models = {}

    # 1) Build lists for each client; train CycleGAN in-memory between lists (no temp dirs, no saving)
    client_paths = []
    for i, client_root in enumerate(CLIENT_ROOTS):
        pths = collect_all_image_paths_for_client(client_root)
        print(f"[DATA] client {CLIENT_NAMES[i]} has {len(pths)} images (train/val/test combined)")
        client_paths.append(pths)

    for i, client_root in enumerate(CLIENT_ROOTS):
        if i == reference_idx:
            print(f"[HARM] skipping CycleGAN for reference client {CLIENT_NAMES[i]}")
            cyclegan_models[i] = None
            continue
        listA = client_paths[i]
        listB = client_paths[reference_idx]
        if len(listA) == 0 or len(listB) == 0:
            print(f"[WARN] empty lists for CycleGAN A({len(listA)}) B({len(listB)}). Skipping generator training; will copy images as-is.")
            cyclegan_models[i] = None
            continue
        # train in memory
        G_A2B, G_B2A = train_cyclegan_from_lists(listA, listB, epochs=CYCLEGAN_EPOCHS, device=DEVICE_TORCH)
        cyclegan_models[i] = G_A2B.cpu()
        # do not save generators to disk

    # 2) Harmonize each client (non-reference) using its generator; reference client copied as-is
    harmonized_base = os.path.join(OUT_DIR, "CycleGAN_Harmonized")
    ensure_dir(harmonized_base)
    harmonized_roots = []
    for i, client_root in enumerate(CLIENT_ROOTS):
        cname = CLIENT_NAMES[i]
        out_client_root = os.path.join(harmonized_base, cname)
        ensure_dir(out_client_root)
        if i == reference_idx:
            # copy entire client structure (train/val/test class subfolders) as-is
            for sp in ["train","val","test"]:
                src_sp = os.path.join(client_root, sp)
                dst_sp = os.path.join(out_client_root, sp)
                if os.path.exists(dst_sp): shutil.rmtree(dst_sp, onerror=_on_rm_error)
                if os.path.exists(src_sp):
                    try:
                        copy_tree_force(src_sp, dst_sp)
                    except Exception:
                        shutil.copytree(src_sp, dst_sp)
        else:
            G = cyclegan_models.get(i, None)
            if G is None:
                print(f"[WARN] no generator for client {cname}, copying as-is")
                for sp in ["train","val","test"]:
                    src_sp = os.path.join(client_root, sp)
                    dst_sp = os.path.join(out_client_root, sp)
                    if os.path.exists(dst_sp): shutil.rmtree(dst_sp, onerror=_on_rm_error)
                    if os.path.exists(src_sp):
                        try:
                            copy_tree_force(src_sp, dst_sp)
                        except Exception:
                            shutil.copytree(src_sp, dst_sp)
            else:
                print(f"[HARM] Harmonizing client {cname} -> {CLIENT_NAMES[reference_idx]}")
                harmonize_client_with_generator(G, client_root, out_client_root, device=DEVICE_TORCH, size=(IMG_SIZE,IMG_SIZE))
        harmonized_roots.append(out_client_root)
    print("[HARM] Harmonization finished. Harmonized datasets at:", harmonized_base)

    # 3) Build dataloaders from harmonized roots and run FedAvg
    combined_loaders, combined_sizes, class_names, combined_train_ds, per_client_dataloaders, per_client_test_dsets = \
        make_multi_client_dataloaders_from_roots(harmonized_roots, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE=="cuda"))
    client_train_sizes = [len(per_client_dataloaders[i]['train'].dataset) for i in range(len(per_client_dataloaders))]
    total_train = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
    print("client train sizes:", client_train_sizes)

    global_model = create_model(num_classes=NUM_CLASSES, arch=ARCH, pretrained=PRETRAINED).to(DEVICE_TORCH)
    print(f"Global model {ARCH} with {count_parameters(global_model):,} params")

    round_results = []
    # histories
    per_client_test_acc_history  = {i: [] for i in range(len(per_client_dataloaders))}
    # keep textual per-class history (no per-class PNGs)
    per_client_perclass_history = {i: [] for i in range(len(per_client_dataloaders))}

    # 3a) create comparison grids
    visuals_out = os.path.join(OUT_DIR, "CycleGAN_Harmonized_Visuals")
    ensure_dir(visuals_out)
    for i, client_root in enumerate(CLIENT_ROOTS):
        original_val_dir = os.path.join(client_root, "val")
        hm_val_dir = os.path.join(harmonized_roots[i], "val")
        make_comparison_grid(original_val_dir, hm_val_dir, CLIENT_NAMES[i], visuals_out, n_samples=7)

    # Federated rounds
    for r in range(COMM_ROUNDS):
        print("\n" + "="*40)
        print(f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print("="*40)
        local_models = []; weights = []; round_summary = {"round": r+1}
        # Per-client local training
        for i, client in enumerate(per_client_dataloaders):
            print(f"[CLIENT {i}] {CLIENT_NAMES[i]}: local training")
            local_model = copy.deepcopy(global_model)
            train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(train_ds).to(DEVICE_TORCH)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            logs = train_local(local_model, client['train'], criterion, optimizer, device=DEVICE_TORCH, epochs=LOCAL_EPOCHS, use_amp=USE_AMP)
            last_train_loss, last_train_acc = logs[-1]
            round_summary[f"client{i}_train_loss"] = float(last_train_loss)
            round_summary[f"client{i}_train_acc"] = float(last_train_acc)

            print(f"[CLIENT {i}] last local epoch loss={last_train_loss:.4f}, acc={last_train_acc:.4f}")
            # validation
            print(f"[CLIENT {i}] local validation")
            local_val_metrics = evaluate_model(local_model, client['val'], DEVICE_TORCH, criterion=criterion)
            round_summary[f"client{i}_localval_loss"] = float(local_val_metrics.get("loss", float('nan')))
            round_summary[f"client{i}_localval_acc"] = float(local_val_metrics.get("accuracy", float('nan')))
            local_models.append(local_model.cpu())
            w = float(client_train_sizes[i]) / float(total_train)
            weights.append(w)
            print(f"[CLIENT {i}] aggregation weight: {w:.4f}")

        # Aggregate (FedAvg)
        print("\nAggregating local models (FedAvg weighted)")
        avg_state = average_models_weighted(local_models, weights)
        avg_state_on_device = {k: v.to(DEVICE_TORCH) for k, v in avg_state.items()}
        global_model.load_state_dict(avg_state_on_device)
        global_model.to(DEVICE_TORCH)

        # Build combined loaders for eval
        combined_val_dsets = [per_client_dataloaders[i]['val'].dataset for i in range(len(per_client_dataloaders))]
        combined_val = ConcatDataset(combined_val_dsets)
        combined_val_loader = DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE=="cuda"))

        combined_train_dsets = [per_client_dataloaders[i]['train'].dataset for i in range(len(per_client_dataloaders))]
        combined_train = ConcatDataset(combined_train_dsets)
        combined_train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE=="cuda"))

        # Compute class weights for combined (for loss reporting)
        combined_train_targets = []
        for i in range(len(per_client_dataloaders)):
            combined_train_targets.extend([s[1] for s in per_client_dataloaders[i]['train'].dataset.samples])
        counts = np.bincount(combined_train_targets, minlength=NUM_CLASSES).astype(np.float32)
        counts[counts==0] = 1.0
        weights_arr = 1.0 / counts
        weights_arr = weights_arr * (len(weights_arr) / weights_arr.sum())
        combined_class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE_TORCH)
        combined_criterion = nn.CrossEntropyLoss(weight=combined_class_weights)

        # Global TRAIN (evaluate aggregated model on combined train set)
        global_train_metrics = evaluate_model(global_model, combined_train_loader, DEVICE_TORCH, criterion=combined_criterion)
        round_summary["global_train_loss"] = float(global_train_metrics.get("loss", float('nan')))
        round_summary["global_train_acc"]  = float(global_train_metrics.get("accuracy", float('nan')))

        # Global VALIDATION (optional summary)
        global_val_metrics = evaluate_model(global_model, combined_val_loader, DEVICE_TORCH, criterion=combined_criterion)
        round_summary["global_val_loss"] = float(global_val_metrics.get("loss", float('nan')))
        round_summary["global_val_acc"] = float(global_val_metrics.get("accuracy", float('nan')))

        # Global TEST on combined test (per-class too)
        combined_test_dsets = [per_client_dataloaders[i]['test'].dataset for i in range(len(per_client_dataloaders))]
        combined_test = ConcatDataset(combined_test_dsets)
        combined_test_loader = DataLoader(combined_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE=="cuda"))
        global_test_metrics = evaluate_model(global_model, combined_test_loader, DEVICE_TORCH, criterion=combined_criterion, return_per_class=True, class_names=CLASS_NAMES)

        # Print global combined test metrics clearly after evaluation
        print("\n[GLOBAL TEST - COMBINED DATASET] Round {:d}".format(r+1))
        print("  Accuracy       : {:.4f}".format(global_test_metrics.get("accuracy", float('nan'))))
        print("  Loss           : {:.4f}".format(global_test_metrics.get("loss", float('nan'))))
        print("  Precision (mac): {:.4f}".format(global_test_metrics.get("precision_macro", float('nan'))))
        print("  Recall (mac)   : {:.4f}".format(global_test_metrics.get("recall_macro", float('nan'))))
        print("  F1 (mac)       : {:.4f}".format(global_test_metrics.get("f1_macro", float('nan'))))
        print("  Balanced Acc   : {:.4f}".format(global_test_metrics.get("balanced_acc", float('nan'))))
        print("  Cohen's kappa  : {:.4f}".format(global_test_metrics.get("cohen_kappa", float('nan'))))

        round_summary["global_test_loss"] = float(global_test_metrics.get("loss", float('nan')))
        round_summary["global_test_acc"] = float(global_test_metrics.get("accuracy", float('nan')))

        # Global TEST per client: print detailed metrics (including per-class table) and record per-client test accuracy for the single plot
        for i, client in enumerate(per_client_dataloaders):
            print(f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set")
            # use class weights computed from that client's train set
            client_train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE_TORCH)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)
            cl_metrics = evaluate_model(global_model, client['test'], DEVICE_TORCH, criterion=client_criterion, return_per_class=True, class_names=CLASS_NAMES)

            # compute mean specificity across classes (if per-class specificity present)
            mean_spec = None
            if "per_class_specificity" in cl_metrics:
                specs = cl_metrics.get("per_class_specificity", [])
                if len(specs) > 0:
                    mean_spec = float(np.mean(specs))
            else:
                # fallback: derive specificity from confusion matrix for binary case (not used unless needed)
                cm = cl_metrics.get("confusion_matrix", None)
                if cm is not None and cm.shape[0] == 2:
                    mean_spec = None

            # Print requested summary metrics for this client
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

            # Also print per-class table (optional, more detailed)
            if "per_class_precision" in cl_metrics:
                print(f"\n  Per-class metrics (order = {CLASS_NAMES}):")
                header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
                print("    " + "{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))
                cm = cl_metrics.get("confusion_matrix", None)
                tp_counts = np.diag(cm).astype(int) if cm is not None else [0]*len(CLASS_NAMES)
                supports = cm.sum(axis=1).astype(int) if cm is not None else [0]*len(CLASS_NAMES)
                precisions = cl_metrics.get("per_class_precision", [])
                recalls = cl_metrics.get("per_class_recall", [])
                f1s = cl_metrics.get("per_class_f1", [])
                specs = cl_metrics.get("per_class_specificity", [])
                accs = cl_metrics.get("per_class_accuracy", [])
                for ci, cname in enumerate(CLASS_NAMES):
                    s = int(supports[ci]) if ci < len(supports) else 0
                    ccount = int(tp_counts[ci]) if ci < len(tp_counts) else 0
                    acc_val = accs[ci] if ci < len(accs) else np.nan
                    pval = precisions[ci] if ci < len(precisions) else np.nan
                    rval = recalls[ci] if ci < len(recalls) else np.nan
                    fval = f1s[ci] if ci < len(f1s) else np.nan
                    sval = specs[ci] if ci < len(specs) else np.nan
                    print(f"    {cname:12s} {s:8d} {ccount:8d} {acc_val:8.4f} {pval:8.4f} {rval:8.4f} {fval:8.4f} {sval:8.4f}")

            # record per-client test accuracy for the single per-client plot
            acc_val = float(cl_metrics.get("accuracy", float('nan')))
            per_client_test_acc_history[i].append(acc_val)

            # keep textual history (no PNG/LaTeX)
            per_client_perclass_history[i].append(cl_metrics)

            round_summary[f"client{i}_test_acc"] = acc_val
            round_summary[f"client{i}_test_loss"] = float(cl_metrics.get("loss", float('nan')))

        # Append per-round summary
        round_results.append(round_summary)

        # Save global + per-client summary plots (only the chosen ones)
        try:
            save_round_series_plots(round_results=round_results, per_client_test_acc_history=per_client_test_acc_history, out_dir=OUT_DIR)
        except Exception as e:
            print("[PLOT] plotting failed with:", e)

    # After all rounds, save full per-client per-class history as a textual file for inspection
    try:
        hist_path = os.path.join(OUT_DIR, "per_client_perclass_history.txt")
        with open(hist_path, "w") as f:
            for i in range(len(per_client_perclass_history)):
                f.write(f"== Client {i} {CLIENT_NAMES[i]} ==\n")
                for rr_idx, cm in enumerate(per_client_perclass_history[i]):
                    f.write(f"Round {rr_idx+1}: accuracy={cm.get('accuracy', 'n/a')}, cohen_kappa={cm.get('cohen_kappa', 'n/a')}\n")
                    pcs = cm.get("per_class_precision", [])
                    prs = cm.get("per_class_recall", [])
                    pfs = cm.get("per_class_f1", [])
                    pss = cm.get("per_class_specificity", [])
                    supp = cm.get("per_class_support", [])
                    f.write(f"  precision: {pcs}\n  recall: {prs}\n  f1: {pfs}\n  specificity: {pss}\n  support: {supp}\n")
                f.write("\n")
        print("[HIST] Saved per-client per-class textual history to:", hist_path)
    except Exception as e:
        print("[HIST] failed writing per-class history:", e)

    print("Federated training finished. (No intermediate model files or .csv files were saved.)")
    print(f"Harmonized datasets stored under: {os.path.join(OUT_DIR, 'CycleGAN_Harmonized')}")
    print(f"Comparison grids stored under: {os.path.join(OUT_DIR, 'CycleGAN_Harmonized_Visuals', 'ComparisonGrid')}")
    print("Global plots:")
    print(f" - {GLOBAL_TRAIN_ACC_FN}")
    print(f" - {GLOBAL_TRAIN_LOSS_FN}")
    print(f" - {GLOBAL_TEST_ACC_FN}")
    print(f" - {GLOBAL_TEST_LOSS_FN}")
    print("Per-client test accuracy (single PNG):")
    print(f" - {PER_CLIENT_TEST_ACC_FN}")

if __name__ == "__main__":
    main()
