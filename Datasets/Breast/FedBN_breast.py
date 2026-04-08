import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
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

from models.UNET import UNET

# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client_names = ["BUSBRA","BUS", "BUSI", "UDIAT"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()

out_dir = "Outputs_breastsBN"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Path to pre-prepared splits root
# It must contain per-client subfolders as:
# <splits_root>/<CLIENT>/{train,val,test}/{images,masks}
# -------------------------
splits_root = r"xxxx\Data\Breasttumor_seg"

# -------------------------
# Per-client expected extensions
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
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

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

            # direct stem match
            for mext in self.mask_exts:
                candidate = os.path.join(self.mask_dir, stem + mext)
                if os.path.exists(candidate):
                    mask_path = candidate
            pairs.append((img_path, mask_path))
        if len(pairs) == 0:
            raise ValueError(
                f"No image-mask pairs found in {img_dir} / {mask_dir}. Missing masks: {missing_masks}"
            )

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

# -------------------------
# Metrics / helper functions
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

def get_bn_state_keys(model):
    """
    Returns all BatchNorm-related state_dict keys:
    weights, bias, running_mean, running_var, num_batches_tracked.
    These are kept local in FedBN and must not be averaged.
    """
    bn_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            prefix = module_name + "." if module_name else ""
            for pname, _ in module.named_parameters(recurse=False):
                bn_keys.add(prefix + pname)
            for bname, _ in module.named_buffers(recurse=False):
                bn_keys.add(prefix + bname)
    return bn_keys

def fedbn_average(models, weights):
    """
    Average only non-BN parameters across clients.
    BN parameters and BN running statistics stay local.
    """
    reference_state = models[0].state_dict()
    bn_keys = get_bn_state_keys(models[0])

    avg_state = copy.deepcopy(reference_state)

    for key in reference_state.keys():
        if key in bn_keys:
            continue
        avg_state[key] = sum(weights[i] * models[i].state_dict()[key] for i in range(len(models)))

    return avg_state

# -------------------------
# Training / Eval functions
# -------------------------
def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss, metrics = 0.0, []

    for _ in range(LOCAL_EPOCHS):
        for data, target in tqdm(loader, leave=False):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)

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
            data = torch.from_numpy(data)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

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
    plt.savefig(os.path.join(out_dir, "dice_no_bg_fedbn.png"))
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
    plt.savefig(os.path.join(out_dir, "iou_no_bg_fedbn.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global average test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Average Test Dice Across Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg_fedbn.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global average test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Average Test IoU Across Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg_fedbn.png"))
    plt.close()

# -------------------------
# Main FedBN
# -------------------------
def main():
    tr_tf = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=[0] * 3, std=[1] * 3),
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0] * 3, std=[1] * 3),
        ToTensorV2()
    ])

    # If your UNET uses out_channels instead of num_classes, change this line.
    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")

        local_models = []
        weights = []
        total_sz = 0

        # Local training
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
            raise RuntimeError("Total training size across clients is 0. Check your split folders.")

        norm_weights = [w / total_sz for w in weights]

        # FedBN aggregation: average only non-BN parameters
        shared_state = fedbn_average(local_models, norm_weights)

        # Update global shared weights, keep its BN as-is
        global_model.load_state_dict(shared_state, strict=False)

        # Sync only shared weights back to clients; BN stays client-specific
        for lm in local_models:
            lm.load_state_dict(shared_state, strict=False)

        # Test each client model on its own test split
        rm = {}
        global_test_metrics_sum = None
        global_test_total = 0

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
            _, test_metrics = evaluate(test_loader, local_models[i], get_loss_fn(DEVICE), split="Test")

            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)

            # Weighted global average across client test sets
            n = len(test_loader.dataset)
            global_test_total += n
            if global_test_metrics_sum is None:
                global_test_metrics_sum = {k: v * n for k, v in test_metrics.items()}
            else:
                for k, v in test_metrics.items():
                    global_test_metrics_sum[k] += v * n

        global_test_metrics = {k: v / global_test_total for k, v in global_test_metrics_sum.items()}

        rm["global_dice_no_bg"] = global_test_metrics.get("dice_no_bg", 0)
        rm["global_iou_no_bg"] = global_test_metrics.get("iou_no_bg", 0)
        rm["global_accuracy"] = global_test_metrics.get("accuracy", 0)
        rm["global_precision"] = global_test_metrics.get("precision", 0)
        rm["global_recall"] = global_test_metrics.get("recall", 0)
        rm["global_specificity"] = global_test_metrics.get("specificity", 0)

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
    main()ss