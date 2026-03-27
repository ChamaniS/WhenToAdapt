import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import glob
import shutil
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
from PIL import Image
from skimage import exposure

from unet import UNET

# ============================================================
# Output redirection
# ============================================================
output_file = "/lustre09/project/6008975/csj5/causalenv/skin_harm_hismat_avg.txt"
sys.stdout = open(output_file, "w")

# ============================================================
# Settings
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client_names = ["HAM10K", "PH2", "ISIC2017", "ISIC2018"]
NUM_CLIENTS = len(client_names)

LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()

out_dir = "Outputs_avg"
os.makedirs(out_dir, exist_ok=True)

# Use one client to build the average reference image
REFERENCE_CLIENT_IDX = 0
REFERENCE_IMAGE_PATH = None   # optional path to a custom image; leave None to compute average
IMG_SIZE = 224
N_REF_SAMPLES = 64            # number of images used to compute the average reference image
USE_ALL_REF_IMAGES = False    # set True to average all images in reference client's train folder

# ============================================================
# Dataset root
# ============================================================
splits_root = r"/lustre09/project/6008975/csj5/skinlesions/"

# ============================================================
# Per-client expected extensions
# ============================================================
client_ext_map = {
    "HAM10K": ((".jpg",), (".png",)),
    "ISIC2017": ((".jpg",), (".png",)),
    "ISIC2018": ((".jpg",), (".png",)),
    "PH2": ((".jpg", ".png", ".bmp", ".jpeg"), (".png", ".bmp", ".jpg", ".jpeg")),
}

# ============================================================
# Helper functions
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def image_list_in_dir(img_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    if not os.path.isdir(img_dir):
        return files
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(files)

def dir_has_files(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0

def find_mask_path(mask_dir, stem, mask_exts):
    for mext in mask_exts:
        candidate = os.path.join(mask_dir, stem + mext)
        if os.path.exists(candidate):
            return candidate

    alt_candidates = []
    for mext in mask_exts:
        alt_candidates.extend([
            os.path.join(mask_dir, stem + "_mask" + mext),
            os.path.join(mask_dir, stem + "-mask" + mext),
            os.path.join(mask_dir, stem.replace("_lesion", "") + mext),
        ])

    for c in alt_candidates:
        if os.path.exists(c):
            return c

    return None

def match_histogram_rgb(src_rgb, ref_rgb):
    try:
        matched = exposure.match_histograms(src_rgb, ref_rgb, channel_axis=-1)
    except TypeError:
        matched = exposure.match_histograms(src_rgb, ref_rgb, multichannel=True)
    return np.clip(matched, 0, 255).astype(np.uint8)

def save_image(arr, path):
    Image.fromarray(arr).save(path)

# ============================================================
# Dataset
# ============================================================
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
            mask_path = find_mask_path(self.mask_dir, stem, self.mask_exts)

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

# ============================================================
# DataLoader helpers
# ============================================================
def get_loader(img_dir, mask_dir, transform, client_name=None, batch_size=4, shuffle=True):
    if client_name is not None and client_name in client_ext_map:
        img_exts, mask_exts = client_ext_map[client_name]
    else:
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    ds = SkinPairDataset(img_dir, mask_dir, transform=transform, img_exts=img_exts, mask_exts=mask_exts)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_global_test_loader(test_img_dirs_used, test_mask_dirs_used, transform, batch_size=4):
    datasets = []
    for i, cname in enumerate(client_names):
        if cname in client_ext_map:
            img_exts, mask_exts = client_ext_map[cname]
        else:
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        ds = SkinPairDataset(
            test_img_dirs_used[i],
            test_mask_dirs_used[i],
            transform=transform,
            img_exts=img_exts,
            mask_exts=mask_exts,
        )
        datasets.append(ds)

    global_test_ds = ConcatDataset(datasets)
    return DataLoader(global_test_ds, batch_size=batch_size, shuffle=False)

# ============================================================
# Metrics / loss / aggregation
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
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k].to(avg_sd[k].device) for i in range(len(models)))
    return avg_sd

# ============================================================
# Average reference image helpers
# ============================================================
def compute_average_reference_image(ref_img_dir, n_samples=64, resize_to=(224, 224), use_all=False):
    """
    Build an average representative image from multiple images.
    - If use_all=True, uses all images in ref_img_dir.
    - Otherwise uses the first n_samples images in sorted order.
    Returns a uint8 RGB image.
    """
    files = image_list_in_dir(ref_img_dir)
    if len(files) == 0:
        raise ValueError(f"No images found in reference dir: {ref_img_dir}")

    if not use_all:
        files = files[:min(len(files), n_samples)]

    acc = None
    count = 0

    for p in files:
        img = Image.open(p).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32)

        if acc is None:
            acc = arr
        else:
            acc += arr
        count += 1

    avg = (acc / count).astype(np.uint8)
    return avg

def load_or_compute_reference_image(reference_client_idx=0, reference_image_path=None, resize_to=(224, 224), n_samples=64, use_all=False):
    """
    If reference_image_path exists, use that.
    Otherwise compute average reference image from the chosen reference client's train images.
    """
    if reference_image_path is not None and os.path.exists(reference_image_path):
        img = Image.open(reference_image_path).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, resample=Image.BILINEAR)
        return np.array(img).astype(np.uint8)

    ref_img_dir = train_img_dirs[reference_client_idx]
    return compute_average_reference_image(
        ref_img_dir,
        n_samples=n_samples,
        resize_to=resize_to,
        use_all=use_all,
    )

def copy_histmatched_split(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, reference_image, mask_exts):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    img_files = image_list_in_dir(src_img_dir)
    if len(img_files) == 0:
        raise ValueError(f"No images found in {src_img_dir}")

    for img_path in img_files:
        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        matched_rgb = match_histogram_rgb(img_rgb, reference_image)

        out_img_path = os.path.join(dst_img_dir, fname)
        cv2.imwrite(out_img_path, cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR))

        src_mask_path = find_mask_path(src_mask_dir, stem, mask_exts)
        if src_mask_path is None:
            continue

        mask_fname = os.path.basename(src_mask_path)
        out_mask_path = os.path.join(dst_mask_dir, mask_fname)
        shutil.copy2(src_mask_path, out_mask_path)

def histmatched_split_ready(img_dir, mask_dir):
    return dir_has_files(img_dir) and dir_has_files(mask_dir)

def prepare_histmatched_datasets(train_img_dirs, train_mask_dirs,
                                 val_img_dirs, val_mask_dirs,
                                 test_img_dirs, test_mask_dirs,
                                 client_names, out_base,
                                 reference_client_idx=0,
                                 reference_image_path=None,
                                 resize_ref=(224, 224),
                                 n_ref_samples=64,
                                 use_all_ref=False):
    """
    Reuses existing hist-matched folders if present.
    If a split folder is missing or empty, it is created.
    Uses an average representative image as reference.
    """
    base = os.path.join(out_base, "HistMatch")
    os.makedirs(base, exist_ok=True)

    ref_img = load_or_compute_reference_image(
        reference_client_idx=reference_client_idx,
        reference_image_path=reference_image_path,
        resize_to=resize_ref,
        n_samples=n_ref_samples,
        use_all=use_all_ref,
    )

    ref_save = os.path.join(base, "reference_image_avg.png")
    if not os.path.exists(ref_save):
        Image.fromarray(ref_img).save(ref_save)
    print(f"[HistMatch] Average reference image saved to: {ref_save}")

    train_img_hm_dirs, train_mask_hm_dirs = [], []
    val_img_hm_dirs, val_mask_hm_dirs = [], []
    test_img_hm_dirs, test_mask_hm_dirs = [], []

    for i, cname in enumerate(client_names):
        client_base = os.path.join(base, cname)
        train_dst_img = os.path.join(client_base, "train_images")
        train_dst_mask = os.path.join(client_base, "train_masks")
        val_dst_img = os.path.join(client_base, "val_images")
        val_dst_mask = os.path.join(client_base, "val_masks")
        test_dst_img = os.path.join(client_base, "test_images")
        test_dst_mask = os.path.join(client_base, "test_masks")

        _, mask_exts = client_ext_map.get(cname, ((".png",), (".png",)))

        print(f"[HistMatch] Checking client {cname} ...")

        if histmatched_split_ready(train_dst_img, train_dst_mask) and \
           histmatched_split_ready(val_dst_img, val_dst_mask) and \
           histmatched_split_ready(test_dst_img, test_dst_mask):
            print(f"[HistMatch] Using existing harmonized folders for {cname}.")
        else:
            print(f"[HistMatch] Creating harmonized folders for {cname} ...")
            copy_histmatched_split(train_img_dirs[i], train_mask_dirs[i], train_dst_img, train_dst_mask, ref_img, mask_exts)
            copy_histmatched_split(val_img_dirs[i], val_mask_dirs[i], val_dst_img, val_dst_mask, ref_img, mask_exts)
            copy_histmatched_split(test_img_dirs[i], test_mask_dirs[i], test_dst_img, test_dst_mask, ref_img, mask_exts)

        train_img_hm_dirs.append(train_dst_img)
        train_mask_hm_dirs.append(train_dst_mask)
        val_img_hm_dirs.append(val_dst_img)
        val_mask_hm_dirs.append(val_dst_mask)
        test_img_hm_dirs.append(test_dst_img)
        test_mask_hm_dirs.append(test_dst_mask)

    print("[HistMatch] Dataset preparation complete.")
    return train_img_hm_dirs, train_mask_hm_dirs, val_img_hm_dirs, val_mask_hm_dirs, test_img_hm_dirs, test_mask_hm_dirs

# ============================================================
# Comparison grid
# ============================================================
def select_val_pairs_for_comparison(orig_dir, hm_dir, n_samples=1):
    orig = image_list_in_dir(orig_dir)
    hm = image_list_in_dir(hm_dir)
    if len(orig) == 0 or len(hm) == 0:
        return []

    orig_map = {}
    for p in orig:
        b = os.path.basename(p)
        if b not in orig_map:
            orig_map[b] = p

    hm_map = {}
    for p in hm:
        b = os.path.basename(p)
        if b not in hm_map:
            hm_map[b] = p

    common = sorted(set(orig_map.keys()) & set(hm_map.keys()))
    pairs = []

    if len(common) > 0:
        sel = common[:n_samples]
        for b in sel:
            pairs.append((orig_map[b], hm_map[b], b))
    else:
        n = min(len(orig), len(hm), n_samples)
        for i in range(n):
            pairs.append((orig[i], hm[i], os.path.basename(orig[i])))

    return pairs

def make_comparison_grid_debug(orig_dir, hm_dir, client_name, out_base, n_samples=1):
    base = ensure_dir(os.path.join(out_base, "ComparisonGrid"))
    orig_count = len(image_list_in_dir(orig_dir))
    hm_count = len(image_list_in_dir(hm_dir))
    print(f"[VIS] client {client_name}: orig images {orig_count}, harmonized {hm_count}")

    pairs = select_val_pairs_for_comparison(orig_dir, hm_dir, n_samples=n_samples)
    if len(pairs) == 0:
        print(f"[VIS] No pairs found for {client_name} - skipping grid.")
        return

    top_imgs, mid_imgs, diff_imgs, titles = [], [], [], []
    rows = []

    for orig_p, hm_p, fn in pairs:
        try:
            o = Image.open(orig_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            h = Image.open(hm_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            A0 = np.array(o).astype(np.int32)
            B0 = np.array(h).astype(np.int32)
            D = np.abs(B0 - A0)

            top_imgs.append(A0.astype(np.uint8))
            mid_imgs.append(B0.astype(np.uint8))

            amplified = np.clip(D * 3.0, 0, 255).astype(np.uint8)
            if amplified.max() < 8 and D.max() > 0:
                amplified = (D / (D.max() + 1e-8) * 255.0).astype(np.uint8)
            diff_imgs.append(amplified)

            titles.append(fn[:24])

            rows.append({
                "file": fn,
                "max_diff": int(D.max()),
                "mean_diff": float(D.mean()),
                "nonzero_pixels": int((D.sum(axis=2) > 0).sum()),
                "orig_path": os.path.abspath(orig_p),
                "hm_path": os.path.abspath(hm_p),
            })
        except Exception as e:
            print("[VIS] skipping pair", fn, ":", e)

    n = len(top_imgs)
    if n == 0:
        print(f"[VIS] no readable images for {client_name}")
        return

    fig, axs = plt.subplots(3, n, figsize=(3.5 * n, 8))
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]]).reshape(3, 1)

    for i in range(n):
        axs[0, i].imshow(top_imgs[i]); axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel("Original", fontsize=10)
        axs[0, i].set_title(titles[i], fontsize=9)

        axs[1, i].imshow(mid_imgs[i]); axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel("Harmonized", fontsize=10)

        axs[2, i].imshow(diff_imgs[i]); axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_ylabel("Amplified diff", fontsize=10)

    fig.suptitle(f"Harmonized vs Original: {client_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(base, f"comparison_{client_name}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[VIS] saved", out_png)

    all_heat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(n):
        all_heat += diff_imgs[i].sum(axis=2).astype(np.float32)

    fig3 = plt.figure(figsize=(6, 4))
    plt.imshow(all_heat, cmap='magma', vmax=max(1, all_heat.max()))
    plt.title(f"Combined diff heatmap: {client_name}")
    plt.axis('off')
    out_heat = os.path.join(base, f"comparison_{client_name}_diffheat.png")
    plt.savefig(out_heat, bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print("[VIS] saved", out_heat)

    for r in rows:
        print(f"[VIS-stats] {r['file']}: max={r['max_diff']} mean={r['mean_diff']:.3f} nonzero_px={r['nonzero_pixels']}")

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

(
    train_img_dirs_used, train_mask_dirs_used,
    val_img_dirs_used, val_mask_dirs_used,
    test_img_dirs_used, test_mask_dirs_used
) = prepare_histmatched_datasets(
    train_img_dirs, train_mask_dirs,
    val_img_dirs, val_mask_dirs,
    test_img_dirs, test_mask_dirs,
    client_names, out_dir,
    reference_client_idx=REFERENCE_CLIENT_IDX,
    reference_image_path=REFERENCE_IMAGE_PATH,
    resize_ref=(IMG_SIZE, IMG_SIZE),
    n_ref_samples=N_REF_SAMPLES,
    use_all_ref=USE_ALL_REF_IMAGES
)

for i, cname in enumerate(client_names):
    make_comparison_grid_debug(
        val_img_dirs[i],
        val_img_dirs_used[i],
        cname,
        out_dir,
        n_samples=1
    )


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
    plt.savefig(os.path.join(out_dir, "dice_no_bg_histmatch_ducknetbs4_avg.png"))
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
    plt.savefig(os.path.join(out_dir, "iou_no_bg_histmatch_ducknetbs4_avg.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_dice_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test Dice")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Global Test Dice Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_dice_no_bg_histmatch_ducknetbs4_avg.png"))
    plt.close()

    plt.figure()
    vals = [rm.get("global_iou_no_bg", 0) for rm in round_metrics]
    plt.plot(rounds, vals, label="Global test IoU")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Global Test IoU Across All Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_iou_no_bg_histmatch_ducknetbs4_avg.png"))
    plt.close()


def main():
    tr_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0] * 3, std=[1] * 3),
        ToTensorV2()
    ])
    val_tf = tr_tf

    global_model=UNET(in_channels=3, out_channels=1).to(DEVICE)
    global_test_loader = get_global_test_loader(test_img_dirs_used, test_mask_dirs_used, val_tf, batch_size=4)

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
                train_img_dirs_used[i],
                train_mask_dirs_used[i],
                tr_tf,
                client_name=client_names[i]
            )
            val_loader = get_loader(
                val_img_dirs_used[i],
                val_mask_dirs_used[i],
                val_tf,
                client_name=client_names[i],
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
                test_img_dirs_used[i],
                test_mask_dirs_used[i],
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