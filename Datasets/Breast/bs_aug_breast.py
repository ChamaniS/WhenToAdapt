import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.UNET import UNET
# from models.DuckNet import DuckNet

# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# client_names should match folder names inside dataset_splits
client_names = ["BUSBRA","BUS", "BUSI", "UDIAT"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()
out_dir = "Outputs_aug"
os.makedirs(out_dir, exist_ok=True)

# Only these images will be used for comparison grids
# Search is now across train/val/test image folders, not only train.
selected_grid_targets = {
    "BUSBRA": ["0001-r.png"],
    "BUS": ["00104.png"],
    "BUSI": ["101.png"],
    "UDIAT": ["000007.png"],
}

# -------------------------
# Path to pre-prepared splits root
# It must contain per-client subfolders as:
# <splits_root>/<CLIENT>/{train,val,test}/{images,masks}
# -------------------------
splits_root = r"xxxx\Data\Breasttumor_seg"

# -------------------------
# Per-client expected extensions
# Format: client_name -> (image_exts_tuple, mask_exts_tuple)
# NOTE: extensions are lowercase and include the dot, e.g. ".jpg"
# -------------------------
client_ext_map = {
    "BUS": ((".png",), (".png",)),
    "BUSBRA": ((".png",), (".png",)),
    "BUSI": ((".png",), (".png",)),
    "UDIAT": ((".png",), (".png",)),
}

# -------------------------
# Albumentations transforms
# -------------------------
train_tf = A.Compose([
    # spatial augmentations
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    # photometric augmentations
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.3
    ),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    # final resize + normalization + to tensor
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Preview-only transform for comparison grids
# Same visual changes, but no normalization or tensor conversion
preview_tf = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.3
    ),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.Resize(224, 224),
])

# -------------------------
# Helper dataset that pairs images + masks by stem
# -------------------------
class SkinPairDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, img_exts=None, mask_exts=None):
        """
        img_dir: directory with image files
        mask_dir: directory with mask files (may have different extension)
        img_exts, mask_exts: tuples of allowed extensions (like ('.jpg', '.png'))
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if img_exts is None:
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        if mask_exts is None:
            mask_exts = (".png", ".jpg", ".bmp", ".tif", ".tiff")

        self.img_exts = tuple(e.lower() for e in img_exts)
        self.mask_exts = tuple(e.lower() for e in mask_exts)

        # Case-insensitive file collection
        all_img_files = []
        for p in Path(self.img_dir).iterdir():
            if p.is_file() and p.suffix.lower() in self.img_exts:
                all_img_files.append(str(p))
        all_img_files = sorted(all_img_files)

        pairs = []
        missing_masks = 0
        for img_path in all_img_files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None

            # direct stem match
            for mext in self.mask_exts:
                candidate = os.path.join(self.mask_dir, stem + mext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            # try case-insensitive direct match on basename
            if mask_path is None:
                stem_lower = stem.lower()
                for p in Path(self.mask_dir).iterdir():
                    if p.is_file():
                        p_stem = p.stem.lower()
                        if p_stem == stem_lower:
                            if p.suffix.lower() in self.mask_exts:
                                mask_path = str(p)
                                break

            # try common alternate naming patterns
            if mask_path is None:
                alt_candidates = [
                    os.path.join(self.mask_dir, stem + "_mask" + mext) for mext in self.mask_exts
                ] + [
                    os.path.join(self.mask_dir, stem + "-mask" + mext) for mext in self.mask_exts
                ] + [
                    os.path.join(self.mask_dir, stem.replace("_lesion", "") + mext) for mext in self.mask_exts
                ]
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

        mask = np.asarray(mask)
        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            mask = np.expand_dims(mask.astype(np.float32), 0)

        return img, mask

# -------------------------
# Build train/val/test image & mask directory lists from splits_root
# -------------------------
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
        mask_exts = (".png", ".jpg", ".bmp", ".tif", ".tiff")

    ds = SkinPairDataset(img_dir, mask_dir, transform=transform, img_exts=img_exts, mask_exts=mask_exts)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_global_test_loader(transform, batch_size=4):
    """
    Combines the TEST splits of all clients into one global test loader.
    This is the loader used to evaluate the global model after each communication round.
    """
    datasets = []
    for i, cname in enumerate(client_names):
        if cname in client_ext_map:
            img_exts, mask_exts = client_ext_map[cname]
        else:
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            mask_exts = (".png", ".jpg", ".bmp", ".tif", ".tiff")

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
# Comparison grid helpers
# -------------------------
def ensure_uint8(img):
    """
    Converts an image array to uint8 safely for visualization.
    """
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = np.nan_to_num(img)
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def find_reference_image(client_name, filename):
    """
    Searches for the filename case-insensitively across train/val/test image folders
    for the given client.
    """
    client_index = client_names.index(client_name)
    candidate_dirs = [
        train_img_dirs[client_index],
        val_img_dirs[client_index],
        test_img_dirs[client_index],
    ]

    target = filename.lower()
    for d in candidate_dirs:
        for p in Path(d).rglob("*"):
            if p.is_file() and p.name.lower() == target:
                return str(p)

    raise FileNotFoundError(
        f"Could not find '{filename}' under any of train/val/test image folders for {client_name}"
    )

def load_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_comparison_grid_for_client(client_name, image_path, save_path):
    original = load_rgb_image(image_path)
    harmonized = preview_tf(image=original)["image"]

    original = ensure_uint8(original)
    harmonized = ensure_uint8(harmonized)

    # match shapes safely
    if original.shape != harmonized.shape:
        harmonized = cv2.resize(harmonized, (original.shape[1], original.shape[0]))

    diff = np.abs(original.astype(np.int16) - harmonized.astype(np.int16))
    diff = np.clip(diff * 4, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    axes[0].imshow(original)
    axes[0].set_title(f"{client_name} | Original")
    axes[0].axis("off")

    axes[1].imshow(harmonized)
    axes[1].set_title(f"{client_name} | Harmonized")
    axes[1].axis("off")

    axes[2].imshow(diff)
    axes[2].set_title(f"{client_name} | Amplified Absolute Difference")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_selected_grids():
    """
    Saves comparison grids only for the selected images.
    Looks in train/val/test image folders.
    """
    for cname, filenames in selected_grid_targets.items():
        for fname in filenames:
            img_path = find_reference_image(cname, fname)
            safe_name = os.path.splitext(os.path.basename(fname))[0].replace(" ", "_")
            save_path = os.path.join(out_dir, f"{cname}_{safe_name}_harmonization_grid.png")
            save_comparison_grid_for_client(cname, img_path, save_path)

# -------------------------
# Metric / helper functions
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
# Training / Eval functions
# -------------------------
def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss, metrics = 0.0, []
    for _ in range(LOCAL_EPOCHS):
        for data, target in tqdm(loader, leave=False):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(DEVICE)
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).to(DEVICE)

            data = data.to(DEVICE)
            target = target.to(DEVICE).unsqueeze(1).float()

            preds = model(data)
            loss = loss_fn(preds, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            metrics.append(compute_metrics(preds.detach(), target))

    avg_metrics = average_metrics(metrics)
    print("Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return (total_loss / len(loader.dataset)) if len(loader.dataset) > 0 else 0.0, avg_metrics

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss, metrics = 0.0, []
    for data, target in loader:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(DEVICE)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(DEVICE)

        data = data.to(DEVICE)
        target = target.to(DEVICE).unsqueeze(1).float()

        preds = model(data)
        loss = loss_fn(preds, target)

        total_loss += loss.item()
        metrics.append(compute_metrics(preds, target))

    avg_metrics = average_metrics(metrics)
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return (total_loss / len(loader.dataset)) if len(loader.dataset) > 0 else 0.0, avg_metrics

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
    plt.savefig(os.path.join(out_dir, "dice_no_bg_ducknetbs4.png"))
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
    plt.savefig(os.path.join(out_dir, "iou_no_bg_ducknetbs4.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Test Dice Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg_ducknetbs4.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Test IoU Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg_ducknetbs4.png"))
    plt.close()

# -------------------------
# Main FedAvg (uses existing split dirs)
# -------------------------
def main():
    # Save grids only for the selected images
    save_selected_grids()

    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
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
                train_tf,
                client_name=client_names[i]
            )
            val_loader = get_loader(
                val_img_dirs[i],
                val_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
                shuffle=False
            )

            print(f"[Client {client_names[i]}]")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset) if hasattr(train_loader.dataset, "__len__") else 0
            weights.append(sz)
            total_sz += sz

        if total_sz == 0:
            raise RuntimeError("Total training size across clients is 0. Check your split folders and manifests.")

        # FedAvg
        norm_weights = [w / total_sz for w in weights]
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

        # ---- Global testing on ALL clients combined ----
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

        # Keep per-client test metrics too, if you still want them
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(
                test_img_dirs[i],
                test_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
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