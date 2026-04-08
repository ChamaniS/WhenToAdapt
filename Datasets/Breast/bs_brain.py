import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import sys
from unet import UNET

output_file = r"/xxxxxxxx/BS_breast.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# client_names should match folder names inside dataset_splits
client_names = ["BUSBRA","BUS", "BUSI", "UDIAT"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 1
COMM_ROUNDS = 1
start_time = time.time()
out_dir = "Outputs_BSbreast"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Path to pre-prepared splits root
# It must contain per-client subfolders as:
# <splits_root>/<CLIENT>/{train,val,test}/{images,masks}
# -------------------------
splits_root = r"/xxxxBreasttumor_seg/"

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
    plt.savefig(os.path.join(out_dir, "dice_no_bg.png"))
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
    plt.savefig(os.path.join(out_dir, "iou_no_bg.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Test Dice Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Test IoU Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg.png"))
    plt.close()

# -------------------------
# Main FedAvg (uses existing split dirs)
# -------------------------
def main():
    tr_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0] * 3, std=[1] * 3),
        ToTensorV2()
    ])
    val_tf = tr_tf

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
                tr_tf,
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
    main()s