import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
import shutil
import random
import sys

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
from PIL import Image
from models.UNET import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client_names = ["BUSBRA", "BUS", "BUSI", "UDIAT"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
IMG_SIZE = 224
BATCH_SIZE = 4
SEED = 42

start_time = time.time()
out_dir = "Outputs_AdaIN_breast"
os.makedirs(out_dir, exist_ok=True)

splits_root = r"xxxx\Data\Breasttumor_seg"

client_ext_map = {
    "BUSBRA": ((".png",), (".png",)),
    "BUS": ((".png",), (".png",)),
    "BUSI": ((".png",), (".png",)),
    "UDIAT": ((".png",), (".png",)),
}

HARMONIZE = True
HARMONIZE_VAL_TEST = True
STYLE_SOURCE = "first_client"

selected_grid_targets = {
    "BUSBRA": ["0001-r.png"],
    "BUS": ["00104.png"],
    "BUSI": ["101.png"],
    "UDIAT": ["000007.png"],
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def is_image_file(fn):
    return fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def list_images_in_dir(d):
    if not os.path.isdir(d):
        return []
    return sorted([f for f in os.listdir(d) if is_image_file(f)])


def safe_copy(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def pil_load_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return img


def pil_to_tensor_01(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def tensor_01_to_pil(tensor):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def compute_channel_mean_std(tensor):
    c = tensor.view(tensor.shape[0], -1)
    mu = c.mean(dim=1)
    std = c.std(dim=1, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return mu, std


def adain_transfer(content_t, style_t):
    mu_c, std_c = compute_channel_mean_std(content_t)
    mu_s, std_s = compute_channel_mean_std(style_t)

    mu_c = mu_c[:, None, None]
    std_c = std_c[:, None, None]
    mu_s = mu_s[:, None, None]
    std_s = std_s[:, None, None]

    normalized = (content_t - mu_c) / std_c
    out = normalized * std_s + mu_s
    out = torch.clamp(out, 0.0, 1.0)
    return out


def get_all_image_paths(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, fn) for fn in list_images_in_dir(folder)]


def find_matching_file(folder, target_name):
    if not os.path.isdir(folder):
        return None

    target_lower = target_name.lower()
    target_stem = os.path.splitext(target_lower)[0]

    files = list_images_in_dir(folder)
    for fn in files:
        if fn.lower() == target_lower:
            return os.path.join(folder, fn)

    for fn in files:
        if os.path.splitext(fn.lower())[0] == target_stem:
            return os.path.join(folder, fn)

    return None


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

            for mext in self.mask_exts:
                candidate = os.path.join(self.mask_dir, stem + mext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                alt_candidates = []
                for mext in self.mask_exts:
                    alt_candidates.append(os.path.join(self.mask_dir, stem + "_mask" + mext))
                    alt_candidates.append(os.path.join(self.mask_dir, stem + "-mask" + mext))
                    alt_candidates.append(os.path.join(self.mask_dir, stem.replace("_lesion", "") + mext))
                    alt_candidates.append(os.path.join(self.mask_dir, stem.replace("-image", "") + mext))

                for c in alt_candidates:
                    if os.path.exists(c):
                        mask_path = c
                        break

            if mask_path is None:
                missing_masks += 1
                continue

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

        mask = (np.asarray(mask) > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            mask = np.expand_dims(mask.astype(np.float32), 0)

        return img, mask


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


def get_client_exts(client_name):
    if client_name in client_ext_map:
        return client_ext_map[client_name]
    return (
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    )


def get_loader(img_dir, mask_dir, transform, client_name=None, batch_size=BATCH_SIZE, shuffle=True):
    img_exts, mask_exts = get_client_exts(client_name) if client_name is not None else (
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    )
    ds = SkinPairDataset(img_dir, mask_dir, transform=transform, img_exts=img_exts, mask_exts=mask_exts)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def get_global_test_loader(transform, batch_size=BATCH_SIZE):
    datasets = []
    for i, cname in enumerate(client_names):
        img_exts, mask_exts = get_client_exts(cname)
        ds = SkinPairDataset(
            test_img_dirs[i],
            test_mask_dirs[i],
            transform=transform,
            img_exts=img_exts,
            mask_exts=mask_exts,
        )
        datasets.append(ds)

    global_test_ds = ConcatDataset(datasets)
    return DataLoader(
        global_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def choose_style_image_paths(base_img_dirs):
    rng = random.Random(SEED)

    client_train_images = {}
    for i, d in enumerate(base_img_dirs):
        client_train_images[i] = get_all_image_paths(d)

    style_map = {}

    if STYLE_SOURCE == "first_client":
        ref_imgs = client_train_images.get(0, [])
        if len(ref_imgs) == 0:
            raise RuntimeError("No images found in first client for AdaIN style reference.")
        ref = ref_imgs[len(ref_imgs) // 2]
        for i in range(len(base_img_dirs)):
            style_map[i] = ref

    elif STYLE_SOURCE == "per_client":
        for i in range(len(base_img_dirs)):
            imgs = client_train_images.get(i, [])
            style_map[i] = imgs[len(imgs) // 2] if len(imgs) > 0 else None

    elif STYLE_SOURCE == "random_global":
        all_imgs = []
        for lst in client_train_images.values():
            all_imgs.extend(lst)
        if len(all_imgs) == 0:
            raise RuntimeError("No images found for random_global style.")
        ref = rng.choice(all_imgs)
        for i in range(len(base_img_dirs)):
            style_map[i] = ref

    else:
        raise ValueError(f"Unknown STYLE_SOURCE = {STYLE_SOURCE}")

    return style_map


def harmonize_image_file(src_img_fp, dst_img_fp, style_path):
    try:
        content = pil_load_rgb(src_img_fp, size=IMG_SIZE)
        content_t = pil_to_tensor_01(content)

        if style_path is None:
            safe_copy(src_img_fp, dst_img_fp)
            return

        style = pil_load_rgb(style_path, size=IMG_SIZE)
        style_t = pil_to_tensor_01(style)

        hm_t = adain_transfer(content_t, style_t)
        hm_pil = tensor_01_to_pil(hm_t)
        ensure_dir(os.path.dirname(dst_img_fp))
        hm_pil.save(dst_img_fp)
    except Exception as e:
        print(f"[AdaIN] Failed on {src_img_fp}: {e}")
        safe_copy(src_img_fp, dst_img_fp)


def create_harmonized_round(round_idx, src_img_dirs, src_mask_dirs, out_base_dir=out_dir):
    style_map = choose_style_image_paths(src_img_dirs)

    hr_base = os.path.join(out_base_dir, f"harmonized_round_{round_idx}")
    if os.path.exists(hr_base):
        shutil.rmtree(hr_base)
    ensure_dir(hr_base)

    harmonized_img_dirs = []
    harmonized_mask_dirs = []

    for i in range(len(src_img_dirs)):
        client_name = client_names[i]
        client_root = os.path.join(hr_base, client_name)
        img_out = os.path.join(client_root, "train", "images")
        mask_out = os.path.join(client_root, "train", "masks")
        ensure_dir(img_out)
        ensure_dir(mask_out)

        if os.path.isdir(src_mask_dirs[i]):
            for fn in list_images_in_dir(src_mask_dirs[i]):
                safe_copy(os.path.join(src_mask_dirs[i], fn), os.path.join(mask_out, fn))

        style_path = style_map.get(i, None)
        if os.path.isdir(src_img_dirs[i]):
            for fn in list_images_in_dir(src_img_dirs[i]):
                src_fp = os.path.join(src_img_dirs[i], fn)
                dst_fp = os.path.join(img_out, fn)
                harmonize_image_file(src_fp, dst_fp, style_path)

        harmonized_img_dirs.append(img_out)
        harmonized_mask_dirs.append(mask_out)

    return harmonized_img_dirs, harmonized_mask_dirs, style_map


def create_copied_split(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, maybe_style_path=None, harmonize=False):
    ensure_dir(dst_img_dir)
    ensure_dir(dst_mask_dir)

    if os.path.isdir(src_mask_dir):
        for fn in list_images_in_dir(src_mask_dir):
            safe_copy(os.path.join(src_mask_dir, fn), os.path.join(dst_mask_dir, fn))

    if os.path.isdir(src_img_dir):
        for fn in list_images_in_dir(src_img_dir):
            src_fp = os.path.join(src_img_dir, fn)
            dst_fp = os.path.join(dst_img_dir, fn)
            if harmonize and maybe_style_path is not None:
                harmonize_image_file(src_fp, dst_fp, maybe_style_path)
            else:
                safe_copy(src_fp, dst_fp)


def create_round_eval_dirs(round_idx, style_map, out_base_dir=out_dir):
    round_root = os.path.join(out_base_dir, f"harmonized_round_{round_idx}")
    ensure_dir(round_root)

    used_val_img_dirs, used_val_mask_dirs = [], []
    used_test_img_dirs, used_test_mask_dirs = [], []

    for i in range(NUM_CLIENTS):
        client_name = client_names[i]
        client_root = os.path.join(round_root, client_name)

        val_img_out = os.path.join(client_root, "val", "images")
        val_mask_out = os.path.join(client_root, "val", "masks")
        test_img_out = os.path.join(client_root, "test", "images")
        test_mask_out = os.path.join(client_root, "test", "masks")

        style_path = style_map.get(i, None)

        create_copied_split(
            val_img_dirs[i], val_mask_dirs[i],
            val_img_out, val_mask_out,
            maybe_style_path=style_path,
            harmonize=HARMONIZE_VAL_TEST
        )
        create_copied_split(
            test_img_dirs[i], test_mask_dirs[i],
            test_img_out, test_mask_out,
            maybe_style_path=style_path,
            harmonize=HARMONIZE_VAL_TEST
        )

        used_val_img_dirs.append(val_img_out)
        used_val_mask_dirs.append(val_mask_out)
        used_test_img_dirs.append(test_img_out)
        used_test_mask_dirs.append(test_mask_out)

    return used_val_img_dirs, used_val_mask_dirs, used_test_img_dirs, used_test_mask_dirs


def save_comparison_grid(original_paths, harmonized_paths, client_name, round_idx, out_base_dir=out_dir):
    pairs = []
    for orig_p in original_paths:
        fn = os.path.basename(orig_p)
        hm_p = harmonized_paths.get(fn.lower(), None)
        if hm_p is not None:
            pairs.append((orig_p, hm_p, fn))

    if len(pairs) == 0:
        print(f"[GRID] No valid pairs found for {client_name} in round {round_idx}")
        return

    n = len(pairs)
    fig, axs = plt.subplots(3, n, figsize=(3.2 * n, 8.5))
    if n == 1:
        axs = np.array(axs).reshape(3, 1)

    for i, (orig_p, hm_p, fn) in enumerate(pairs):
        try:
            orig = np.array(Image.open(orig_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
            hm = np.array(Image.open(hm_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

            diff = np.abs(hm.astype(np.float32) - orig.astype(np.float32))
            amplified = diff * 3.0
            amplified = np.clip(amplified, 0, 255).astype(np.uint8)

            if amplified.max() < 8:
                if diff.max() > 0:
                    amplified = (diff / (diff.max() + 1e-8) * 255.0).astype(np.uint8)
                else:
                    amplified = np.zeros_like(amplified, dtype=np.uint8)

            axs[0, i].imshow(orig)
            axs[0, i].axis("off")
            axs[0, i].set_title(fn, fontsize=9)
            if i == 0:
                axs[0, i].set_ylabel("Original", fontsize=10)

            axs[1, i].imshow(hm)
            axs[1, i].axis("off")
            if i == 0:
                axs[1, i].set_ylabel("Harmonized", fontsize=10)

            axs[2, i].imshow(amplified)
            axs[2, i].axis("off")
            if i == 0:
                axs[2, i].set_ylabel("Amplified diff", fontsize=10)

        except Exception as e:
            print(f"[GRID] Skipping {fn} for {client_name}: {e}")

    fig.suptitle(f"AdaIN Comparison Grid - {client_name} - Round {round_idx}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    grid_dir = os.path.join(out_base_dir, "ComparisonGrid", f"round_{round_idx}")
    ensure_dir(grid_dir)
    out_png = os.path.join(grid_dir, f"comparison_{client_name}.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[GRID] Saved {out_png}")


def create_selected_comparison_grids(round_idx, original_img_dirs, harmonized_img_dirs, selected_targets, out_base_dir=out_dir):
    for i, cname in enumerate(client_names):
        targets = selected_targets.get(cname, [])
        if len(targets) == 0:
            continue

        orig_dir = original_img_dirs[i]
        hm_dir = harmonized_img_dirs[i]

        harmonized_map = {}
        for fn in list_images_in_dir(hm_dir):
            harmonized_map[fn.lower()] = os.path.join(hm_dir, fn)

        original_paths = []
        for target_name in targets:
            orig_fp = find_matching_file(orig_dir, target_name)
            if orig_fp is None:
                print(f"[GRID] Could not find original file '{target_name}' in {orig_dir}")
                continue
            original_paths.append(orig_fp)

        if len(original_paths) == 0:
            print(f"[GRID] No originals found for {cname}, round {round_idx}")
            continue

        save_comparison_grid(original_paths, harmonized_map, cname, round_idx, out_base_dir=out_base_dir)


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


def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss = 0.0
    metrics = []
    num_batches = 0

    for _ in range(LOCAL_EPOCHS):
        for data, target in tqdm(loader, leave=False):
            data = data.to(DEVICE)
            target = target.to(DEVICE).unsqueeze(1).float()

            preds = model(data)
            loss = loss_fn(preds, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1
            metrics.append(compute_metrics(preds.detach(), target))

    avg_metrics = average_metrics(metrics)
    if avg_metrics:
        print("Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return total_loss / max(1, num_batches), avg_metrics


@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss = 0.0
    metrics = []
    num_batches = 0

    for data, target in loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE).unsqueeze(1).float()

        preds = model(data)
        loss = loss_fn(preds, target)

        total_loss += loss.item()
        num_batches += 1
        metrics.append(compute_metrics(preds, target))

    avg_metrics = average_metrics(metrics)
    if avg_metrics:
        print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return total_loss / max(1, num_batches), avg_metrics


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
    plt.savefig(os.path.join(out_dir, "dice_no_bg_adain.png"))
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
    plt.savefig(os.path.join(out_dir, "iou_no_bg_adain.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Test Dice Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg_adain.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Test IoU Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg_adain.png"))
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tr_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2()
    ])
    val_tf = tr_tf
    test_tf = tr_tf

    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")

        if HARMONIZE:
            print(f"[AdaIN] Creating harmonized TRAIN folders for round {r+1}")
            used_train_img_dirs, used_train_mask_dirs, style_map = create_harmonized_round(
                r + 1, train_img_dirs, train_mask_dirs, out_base_dir=out_dir
            )
        else:
            used_train_img_dirs = train_img_dirs
            used_train_mask_dirs = train_mask_dirs
            style_map = choose_style_image_paths(train_img_dirs)

        if HARMONIZE_VAL_TEST:
            used_val_img_dirs, used_val_mask_dirs, used_test_img_dirs, used_test_mask_dirs = create_round_eval_dirs(
                r + 1, style_map, out_base_dir=out_dir
            )
        else:
            used_val_img_dirs = val_img_dirs
            used_val_mask_dirs = val_mask_dirs
            used_test_img_dirs = test_img_dirs
            used_test_mask_dirs = test_mask_dirs

        create_selected_comparison_grids(
            r + 1,
            val_img_dirs,
            used_val_img_dirs,
            selected_grid_targets,
            out_base_dir=out_dir
        )

        local_models = []
        weights = []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(
                used_train_img_dirs[i],
                used_train_mask_dirs[i],
                tr_tf,
                client_name=client_names[i],
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            val_loader = get_loader(
                used_val_img_dirs[i],
                used_val_mask_dirs[i],
                val_tf,
                client_name=client_names[i],
                batch_size=BATCH_SIZE,
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
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

        global_test_loader = get_global_test_loader(test_tf, batch_size=BATCH_SIZE)
        global_test_loss, global_test_metrics = evaluate(
            global_test_loader,
            global_model,
            get_loss_fn(DEVICE),
            split="Global Test"
        )

        rm = {
            "global_test_loss": global_test_loss,
            "global_dice_no_bg": global_test_metrics.get("dice_no_bg", 0.0),
            "global_iou_no_bg": global_test_metrics.get("iou_no_bg", 0.0),
            "global_accuracy": global_test_metrics.get("accuracy", 0.0),
            "global_precision": global_test_metrics.get("precision", 0.0),
            "global_recall": global_test_metrics.get("recall", 0.0),
            "global_specificity": global_test_metrics.get("specificity", 0.0),
        }

        for i in range(NUM_CLIENTS):
            test_loader = get_loader(
                used_test_img_dirs[i],
                used_test_mask_dirs[i],
                test_tf,
                client_name=client_names[i],
                batch_size=BATCH_SIZE,
                shuffle=False
            )
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0.0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0.0)

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