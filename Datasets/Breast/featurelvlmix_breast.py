import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
import shutil
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
from PIL import Image

from models.unet_mixstyle import UNET
# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client_names =  ["BUSBRA","BUS", "BUSI", "UDIAT"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()

out_dir = "Outputs_featurelvl_breast"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Path to pre-prepared splits root
# It must contain per-client subfolders as:
# <splits_root>/<CLIENT>/{train,val,test}/{images,masks}
# -------------------------
splits_root =r"xxxx\Data\Breasttumor_seg"

# -------------------------
# Per-client expected extensions
# -------------------------
client_ext_map = {
    "BUSBRA": ((".png",), (".png",)),
    "BUS": ((".png",), (".png",)),
    "BUSI": ((".png",), (".png",)),
    "UDIAT": ((".png",), (".png",)),
}

# -------------------------
# Helper dataset that pairs images + masks by stem
# -------------------------
class SkinPairDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, img_exts=None, mask_exts=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if img_exts is None:
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        if mask_exts is None:
            mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

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

            # direct stem match
            for mext in self.mask_exts:
                candidate = os.path.join(self.mask_dir, stem + mext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            # alternate naming patterns
            if mask_path is None:
                alt_candidates = []
                for mext in self.mask_exts:
                    alt_candidates.extend([
                        os.path.join(self.mask_dir, stem + "_mask" + mext),
                        os.path.join(self.mask_dir, stem + "-mask" + mext),
                        os.path.join(self.mask_dir, stem.replace("_lesion", "") + mext),
                    ])
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
            raise RuntimeError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask {mask_path}")
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

        return img, mask

# -------------------------
# Build train/val/test directory lists
# -------------------------
train_img_dirs, train_mask_dirs = [], []
val_img_dirs, val_mask_dirs = [], []
test_img_dirs, test_mask_dirs = [], []

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
    print(f"  val   imgs: {val_img_dirs[i]}  masks: {val_mask_dirs[i]}")
    print(f"  test  imgs: {test_img_dirs[i]}  masks: {test_mask_dirs[i]}")

# -------------------------
# Loader helpers
# -------------------------
def get_loader(img_dir, mask_dir, transform, client_name=None, batch_size=4, shuffle=True):
    if client_name is not None and client_name in client_ext_map:
        img_exts, mask_exts = client_ext_map[client_name]
    else:
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    ds = SkinPairDataset(img_dir, mask_dir, transform=transform, img_exts=img_exts, mask_exts=mask_exts)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_global_test_loader(transform, batch_size=4):
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
        )
        datasets.append(ds)

    global_test_ds = ConcatDataset(datasets)
    return DataLoader(global_test_ds, batch_size=batch_size, shuffle=False)

# -------------------------
# Metrics
# -------------------------
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

# -------------------------
# Visualization helpers
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _unnormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr

def _tensor_to_uint8_rgb(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr

def _rgb_abs_diff_uint8(orig_u8, mix_u8, amp=3.0):
    diff = np.abs(mix_u8.astype(np.float32) - orig_u8.astype(np.float32))
    diff = np.clip(diff * amp, 0, 255).astype(np.uint8)
    return diff

def _feature_to_heatmap(feat_tensor):
    if feat_tensor.dim() == 4:
        feat_tensor = feat_tensor[0]
    fmap = feat_tensor.detach().cpu().float().mean(dim=0).numpy()
    fmap = fmap - fmap.min()
    if fmap.max() > 0:
        fmap = fmap / fmap.max()
    fmap = (fmap * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat

def _freeze_batchnorm_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

def find_image_and_mask_for_client(client_name, target_filename):
    base = os.path.join(splits_root, client_name)
    target_lower = target_filename.lower()

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base, split, "images")
        mask_dir = os.path.join(base, split, "masks")

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        for fname in os.listdir(img_dir):
            if fname.lower() == target_lower:
                img_path = os.path.join(img_dir, fname)
                stem = os.path.splitext(fname)[0]

                if client_name in client_ext_map:
                    _, mask_exts = client_ext_map[client_name]
                else:
                    mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

                mask_path = None
                for mext in mask_exts:
                    candidate = os.path.join(mask_dir, stem + mext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break

                if mask_path is None:
                    for mext in mask_exts:
                        for cand in [
                            os.path.join(mask_dir, stem + "_mask" + mext),
                            os.path.join(mask_dir, stem + "-mask" + mext),
                            os.path.join(mask_dir, stem.replace("_lesion", "") + mext),
                        ]:
                            if os.path.exists(cand):
                                mask_path = cand
                                break
                        if mask_path is not None:
                            break

                if mask_path is None:
                    raise FileNotFoundError(f"Found image {img_path} but could not find a matching mask in {mask_dir}")

                return img_path, mask_path, split

    raise FileNotFoundError(f"Could not find {target_filename} under client {client_name}")

# -------------------------
# Visualization-only input MixStyle for the grid
# This is used only to create the harmonized image for row 2,
# and row 3 is the amplified absolute difference between original and harmonized.
# -------------------------
def input_mixstyle_for_visualization(images, p=1.0, alpha=0.2, eps=1e-6, force_pair=True, fixed_lam=0.5):
    if not torch.is_tensor(images):
        images = torch.tensor(images)

    if not images.is_floating_point():
        images = images.float()

    B, C, H, W = images.shape
    if B <= 1:
        return images

    if np.random.rand() > p:
        return images

    x = images.view(B, C, -1)
    mu = x.mean(dim=2).view(B, C, 1, 1)
    var = x.var(dim=2, unbiased=False).view(B, C, 1, 1)
    sigma = torch.sqrt(var + eps)

    x_norm = (images - mu) / sigma

    lam = torch.full((B, 1, 1, 1), float(fixed_lam), device=images.device, dtype=images.dtype)

    if force_pair:
        perm = torch.roll(torch.arange(B, device=images.device), shifts=1)
    else:
        perm = torch.randperm(B, device=images.device)

    mu2 = mu[perm]
    sigma2 = sigma[perm]

    mu_mix = mu * lam + mu2 * (1.0 - lam)
    sigma_mix = sigma * lam + sigma2 * (1.0 - lam)

    return x_norm * sigma_mix + mu_mix

def save_original_harmonized_diff_grid(selected_items, transform, out_base, grid_name="feature_mixstyle_four_dataset_comparison.png", diff_amp=3.0):
    """
    Saves:
      row 1: original image
      row 2: harmonized image (visualization-only input MixStyle)
      row 3: amplified absolute RGB difference between original and harmonized
    """
    ensure_dir(out_base)

    imgs = []
    titles = []

    for item in selected_items:
        client_name = item["client_name"]
        filename = item["filename"]

        img_path, mask_path, split_name = find_image_and_mask_for_client(client_name, filename)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)

        out = transform(image=img_rgb, mask=mask)
        imgs.append(out["image"])
        titles.append(f"{client_name}\n{filename}\n({split_name})")

    batch = torch.stack(imgs, dim=0).to(DEVICE).float()

    harmonized = input_mixstyle_for_visualization(
        batch,
        p=1.0,
        alpha=0.2,
        force_pair=True,
        fixed_lam=0.5
    ).clamp(0.0, 1.0)

    originals = batch.detach().cpu()
    harmonized_cpu = harmonized.detach().cpu()

    n = len(selected_items)
    fig, axs = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    fig.suptitle("Original vs Harmonized Comparison Across 4 Skin Lesion Datasets", fontsize=16, y=0.98)
    row_labels = ["Original", "Harmonized", "Amplified |Difference|"]
    for r in range(3):
        fig.text(0.01, 0.82 - r * 0.31, row_labels[r], fontsize=12, va='center', rotation='vertical')

    for i in range(n):
        orig_img = _tensor_to_uint8_rgb(originals[i])
        harm_img = _tensor_to_uint8_rgb(harmonized_cpu[i])
        diff_img = _rgb_abs_diff_uint8(orig_img, harm_img, amp=diff_amp)

        axs[0, i].imshow(orig_img)
        axs[0, i].axis("off")
        axs[0, i].set_title(titles[i], fontsize=9)

        axs[1, i].imshow(harm_img)
        axs[1, i].axis("off")

        axs[2, i].imshow(diff_img)
        axs[2, i].axis("off")

    plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])
    out_path = os.path.join(out_base, grid_name)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved original/harmonized comparison grid to: {out_path}")

# -------------------------
# Training / Eval
# -------------------------
def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss, metrics = 0.0, []
    n_batches = 0

    for _ in range(LOCAL_EPOCHS):
        for batch in tqdm(loader, leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, target = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch format in train_local")

            if target.dim() == 3:
                target = target.unsqueeze(1).float()
            elif target.dim() == 4:
                target = target.float()
            else:
                raise RuntimeError(f"Unexpected target dims: {target.shape}")

            data = data.to(DEVICE).float()
            target = target.to(DEVICE)

            preds = model(data)
            loss = loss_fn(preds, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            metrics.append(compute_metrics(preds.detach(), target))
            n_batches += 1

    avg_metrics = average_metrics(metrics)
    avg_loss = total_loss / max(n_batches, 1)
    print("Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in (avg_metrics or {}).items()]))
    return avg_loss, avg_metrics

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss, metrics = 0.0, []
    n_batches = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
        else:
            raise RuntimeError("Unexpected batch format in evaluate")

        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        elif target.dim() == 4:
            target = target.float()
        else:
            raise RuntimeError(f"Unexpected target dims: {target.shape}")

        data = data.to(DEVICE).float()
        target = target.to(DEVICE)

        preds = model(data)
        loss = loss_fn(preds, target)

        total_loss += loss.item()
        metrics.append(compute_metrics(preds, target))
        n_batches += 1

    avg_metrics = average_metrics(metrics)
    avg_loss = total_loss / max(n_batches, 1)
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k, v in (avg_metrics or {}).items()]))
    return avg_loss, avg_metrics

# -------------------------
# Plotting
# -------------------------
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
    plt.savefig(os.path.join(out_dir, "dice_no_bg_featuremixstyle.png"))
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
    plt.savefig(os.path.join(out_dir, "iou_no_bg_featuremixstyle.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Test Dice Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg_featuremixstyle.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Test IoU Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg_featuremixstyle.png"))
    plt.close()

# -------------------------
# Main FedAvg
# -------------------------
def main():
    tr_tf = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    selected_items = [
        {"client_name": "HAM10K",   "filename": "ISIC_0032968.JPG"},
        {"client_name": "PH2",      "filename": "IMD405.bmp"},
        {"client_name": "ISIC2017", "filename": "ISIC_0014191.jpg"},
        {"client_name": "ISIC2018", "filename": "ISIC_0014715.jpg"},
    ]

    # Feature-level MixStyle model for training
    global_model = UNET(
        in_channels=3,
        out_channels=1,
        init_features=32,
        mixstyle=True,
        mix_p=0.5,
        mix_alpha=0.1
    ).to(DEVICE)

    # Visualization-only grid: original / harmonized / amplified difference
    save_original_harmonized_diff_grid(
        selected_items=selected_items,
        transform=val_tf,
        out_base=os.path.join(out_dir, "MixStyleSamples"),
        grid_name="feature_mixstyle_four_dataset_comparison.png",
        diff_amp=3.0
    )

    global_test_loader = get_global_test_loader(val_tf, batch_size=4)
    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models, weights = [], []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(
                train_img_dirs[i],
                train_mask_dirs[i],
                tr_tf,
                client_name=client_names[i],
                batch_size=4,
                shuffle=True
            )
            val_loader = get_loader(
                val_img_dirs[i],
                val_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
                batch_size=4,
                shuffle=False
            )

            print(f"[Client {client_names[i]}]")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset)
            weights.append(sz)
            total_sz += sz

        if total_sz == 0:
            raise RuntimeError("Total training size across clients is 0. Check your split folders and masks.")

        norm_weights = [w / total_sz for w in weights]
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

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

        for i in range(NUM_CLIENTS):
            test_loader = get_loader(
                test_img_dirs[i],
                test_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
                batch_size=4,
                shuffle=False
            )
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

        print(
            f"[GLOBAL TEST AFTER ROUND {r+1}] "
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