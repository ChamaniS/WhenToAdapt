"""my-awesome-app: Polyp Segmentation with U-Net for Flower FL (with Histogram Matching harmonization)."""

import os, cv2, torch, numpy as np, shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
from skimage import exposure  # <-- histogram matching
from my_awesome_app.unet import UNET
from pathlib import Path
import shutil

# -----------------------
# General setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where to store harmonized copies (cached between rounds/runs)
DEFAULT_OUT_BASE = "xxxxx/flower-tutorial/my-awesome-app/my_awesome_app/Outputs"
os.makedirs(DEFAULT_OUT_BASE, exist_ok=True)


# -----------------------
# Dataset
# -----------------------
class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir)
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.masks = sorted([f for f in os.listdir(mask_dir)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        # keep only pairs that exist in both
        both = sorted(list(set(self.images) & set(self.masks)))
        if both:
            self.images, self.masks = both, both
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224))

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.transform:
            image = self.transform(image)

        return image, mask


# -----------------------
# Metrics
# -----------------------
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

    return {
        "dice_with_bg": dice_with_bg,
        "dice_no_bg": dice_no_bg,
        "iou_with_bg": iou_with_bg,
        "iou_no_bg": iou_no_bg,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


# -----------------------
# Histogram Matching utils
# -----------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _avg_reference_image(ref_img_dir: str, n_samples: int = 64, size=(224, 224)) -> np.ndarray:
    files = sorted([f for f in os.listdir(ref_img_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])[:n_samples]
    if not files:
        raise ValueError(f"No images found in {ref_img_dir} for reference.")
    acc = None
    for fn in files:
        arr = np.array(Image.open(os.path.join(ref_img_dir, fn)).convert("RGB").resize(size, Image.BILINEAR),
                       dtype=np.float32)
        acc = arr if acc is None else (acc + arr)
    avg = (acc / len(files)).clip(0, 255).astype(np.uint8)
    return avg

def _match_dir_to_reference(src_img_dir: str, src_mask_dir: str, dst_img_dir: str, dst_mask_dir: str, reference: np.ndarray):
    _ensure_dir(dst_img_dir); _ensure_dir(dst_mask_dir)
    img_files = sorted([f for f in os.listdir(src_img_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for fn in img_files:
        src_img = np.array(Image.open(os.path.join(src_img_dir, fn)).convert("RGB"))
        # try newer skimage API first
        try:
            matched = exposure.match_histograms(src_img, reference, channel_axis=-1)
        except TypeError:
            matched = exposure.match_histograms(src_img, reference, multichannel=True)
        matched = matched.clip(0, 255).astype(np.uint8)
        Image.fromarray(matched).save(os.path.join(dst_img_dir, fn))

        # copy mask (if exists)
        src_m = os.path.join(src_mask_dir, fn)
        if os.path.exists(src_m):
            Image.open(src_m).save(os.path.join(dst_mask_dir, fn))

def _client_name(partition_id: int) -> str:
    return ["Kvasir", "ETIS", "CVC-Colon", "CVC-Clinic"][partition_id]

def save_image(arr, path):
    """
    Save a numpy uint8 array (HxW or HxWxC) to path.
    """
    if isinstance(path, Path):
        path = str(path)
    # If array is float in [0,1], convert to 0-255
    if arr.dtype != np.uint8:
        try:
            # cover float in [0,1] or [0,255]
            if arr.max() <= 1.5:
                arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)

def _maybe_create_histmatched_split(
    base_path_map: dict,
    out_base: str,
    ref_client_idx: int = 0,
    n_ref_samples: int = 64,
) -> dict:
    """
    Creates (or reuses if already created) histogram-matched copies for each client's
    train/val/test under:
      {out_base}/HistMatch/<Client>/{train/images,train/masks,val/images,...}
    Returns a dict with per-client img/mask roots for each split.
    """
    cache_root = os.path.join(out_base, "HistMatch")
    _ensure_dir(cache_root)

    # Reference image file cache
    ref_img_path = os.path.join(cache_root, "reference_image.png")
    if os.path.exists(ref_img_path):
        reference = np.array(Image.open(ref_img_path).convert("RGB"))
    else:
        ref_train_images = os.path.join(base_path_map[ref_client_idx], "train", "images")
        reference = _avg_reference_image(ref_train_images, n_samples=n_ref_samples, size=(224, 224))
        Image.fromarray(reference).save(ref_img_path)

    # Create matched dirs for all clients/splits
    out_map = {}
    for pid in range(4):
        cname = _client_name(pid)
        src_root = base_path_map[pid]
        dst_root = os.path.join(cache_root, cname)

        # dir layout
        pairs = [
            ("train/images", "train/masks"),
            ("val/images",   "val/masks"),
            ("test/images",  "test/masks"),
        ]
        for split_img_rel, split_mask_rel in pairs:
            src_img_dir = os.path.join(src_root, split_img_rel)
            src_msk_dir = os.path.join(src_root, split_mask_rel)

            dst_img_dir = os.path.join(dst_root, split_img_rel)
            dst_msk_dir = os.path.join(dst_root, split_mask_rel)

            # Only create if missing (cache)
            if not (os.path.isdir(dst_img_dir) and os.listdir(dst_img_dir)):
                _match_dir_to_reference(src_img_dir, src_msk_dir, dst_img_dir, dst_msk_dir, reference)

        out_map[pid] = {
            "train_img": os.path.join(dst_root, "train/images"),
            "train_msk": os.path.join(dst_root, "train/masks"),
            "val_img":   os.path.join(dst_root, "val/images"),
            "val_msk":   os.path.join(dst_root, "val/masks"),
            "test_img":  os.path.join(dst_root, "test/images"),
            "test_msk":  os.path.join(dst_root, "test/masks"),
        }
    return out_map


# -----------------------
# Data loading
# -----------------------
def load_data(
    partition_id: int,
    num_partitions: int,
    use_hist_match: bool = True,
    reference_client_idx: int = 0,
    n_ref_samples: int = 64,
    out_base: str = DEFAULT_OUT_BASE,
):
    """
    Loads train/val/test loaders for a given client.
    If use_hist_match=True, creates/uses histogram-matched copies (cached under out_base).
    """

    base_path = "xxxxx/flower-tutorial/my-awesome-app/my_awesome_app/Polyps/"
    dataset_map = {
        0: os.path.join(base_path, "Kvasir/Kvasir"),
        1: os.path.join(base_path, "ETIS/rearranged"),
        2: os.path.join(base_path, "CVC-Colon/rearranged"),
        3: os.path.join(base_path, "CVC-Clinic/rearranged"),
    }

    # Normalization only; we already resize inside dataset to 224x224
    transform_img = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])

    # Build paths for this client
    if use_hist_match:
        # Create/reuse matched copies for ALL clients, then pick this client's dirs
        matched_map = _maybe_create_histmatched_split(
            dataset_map, out_base, ref_client_idx=reference_client_idx, n_ref_samples=n_ref_samples
        )
        d = matched_map[partition_id]
        train_img_dir, train_msk_dir = d["train_img"], d["train_msk"]
        val_img_dir,   val_msk_dir   = d["val_img"],   d["val_msk"]
        test_img_dir,  test_msk_dir  = d["test_img"],  d["test_msk"]
    else:
        root = dataset_map[partition_id]
        train_img_dir, train_msk_dir = os.path.join(root, "train/images"), os.path.join(root, "train/masks")
        val_img_dir,   val_msk_dir   = os.path.join(root, "val/images"),   os.path.join(root, "val/masks")
        test_img_dir,  test_msk_dir  = os.path.join(root, "test/images"),  os.path.join(root, "test/masks")

    train_set = PolypDataset(train_img_dir, train_msk_dir, transform=transform_img)
    val_set   = PolypDataset(val_img_dir,   val_msk_dir,   transform=transform_img)
    test_set  = PolypDataset(test_img_dir,  test_msk_dir,  transform=transform_img)

    trainloader = DataLoader(train_set, batch_size=4, shuffle=True,  num_workers=0)
    valloader   = DataLoader(val_set,   batch_size=4, shuffle=False, num_workers=0)
    testloader  = DataLoader(test_set,  batch_size=4, shuffle=False, num_workers=0)

    print(f"[DEBUG] Client {partition_id}/{num_partitions} ({_client_name(partition_id)}): "
          f"HM={'ON' if use_hist_match else 'OFF'} "
          f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}",
          flush=True)

    return trainloader, valloader, testloader


# -----------------------
# Loss
# -----------------------
def get_loss_fn(net, device):
    return nn.BCEWithLogitsLoss().to(device)


# -----------------------
# Train / Eval
# -----------------------
def train(net, trainloader, epochs, device):
    net.to(device).train()
    criterion = get_loss_fn(net, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    for epoch in range(epochs):
        running_loss, metrics_accum, n_batches = 0.0, {}, 0
        for images, masks in trainloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            batch_metrics = compute_metrics(outputs, masks)
            running_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            n_batches += 1
        avg_loss = running_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in metrics_accum.items()}
        print(f"[Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, " +
              ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return avg_loss, avg_metrics


def evaluate_val(net, valloader, device):
    net.to(device).eval()
    criterion = get_loss_fn(net, device)
    loss_sum, metrics_accum, n = 0.0, {}, 0
    with torch.no_grad():
        for images, masks in valloader:
            images, masks = images.to(device), masks.to(device)
            outputs = net(images)
            loss_sum += criterion(outputs, masks).item()
            batch_metrics = compute_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            n += 1
    avg_loss = loss_sum / n
    avg_metrics = {k: v / n for k, v in metrics_accum.items()}
    print(f"[Val] Loss: {avg_loss:.4f}, " +
          ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return avg_loss, avg_metrics


def test_final(net, testloader, device, partition_id=None):
    net.to(device).eval()
    criterion = get_loss_fn(net, device)
    loss_sum, metrics_accum, n = 0.0, {}, 0
    with torch.no_grad():
        for images, masks in testloader:
            images, masks = images.to(device), masks.to(device)
            outputs = net(images)
            loss_sum += criterion(outputs, masks).item()
            batch_metrics = compute_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            n += 1
    avg_loss = loss_sum / n
    avg_metrics = {k: v / n for k, v in metrics_accum.items()}
    tag = f"[Client {partition_id}] " if partition_id is not None else ""
    print(f"{tag}Test Loss: {avg_loss:.4f}, " +
          ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return avg_loss, avg_metrics


# -----------------------
# Save test predictions (latest only)
# -----------------------
def _mask_to_uint8(mask_tensor):
    """Convert single-channel tensor (1,H,W) or (H,W) float [0,1] to uint8 0/255."""
    m = mask_tensor.cpu().numpy()
    if m.ndim == 3:
        m = np.squeeze(m, axis=0)
    m = (m > 0.5).astype(np.uint8) * 255
    return m

def _safe_basename_from_dataset(dataset, global_index):
    """
    Try to obtain the filename (with extension) for the sample at global_index from common dataset attributes.
    Returns None if it cannot be determined.
    """
    try:
        # PolypDataset exposes .images as list of filenames (with extensions) in sorted order
        if hasattr(dataset, "images") and isinstance(dataset.images, list):
            if 0 <= global_index < len(dataset.images):
                return dataset.images[global_index]
    except Exception:
        pass
    return None

def save_test_predictions(global_model,
                          test_loader,
                          client_name,
                          out_base: str = None,
                          round_num: int = None,
                          max_to_save: int = 16,
                          device_arg: torch.device = None):
    """
    Save predicted masks for `test_loader` into:
        {out_base}/TestPreds/{client_name}/latest

    If out_base is None, this will fall back to DEFAULT_OUT_BASE.
    This folder is cleared and replaced every round (keeps only 'latest').

    This version attempts to preserve the original test filenames by reading
    `test_loader.dataset.images` (as available in PolypDataset). If filenames
    cannot be determined, it falls back to a generated name.
    """
    # fallback to DEFAULT_OUT_BASE if caller passed None
    if out_base is None:
        out_base = DEFAULT_OUT_BASE

    if device_arg is None:
        device = device if "device" in globals() else torch.device("cpu")
    else:
        device = device_arg

    global_model.eval()
    latest_dir = Path(out_base) / "TestPreds" / client_name / "latest"

    # clear and recreate
    if latest_dir.exists():
        import shutil
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    # try to fetch dataset filenames if possible
    dataset_filenames = None
    try:
        ds = test_loader.dataset
        if hasattr(ds, "images") and isinstance(ds.images, list):
            dataset_filenames = ds.images  # list of filenames (with extension)
    except Exception:
        dataset_filenames = None

    # === OVERRIDE: save all samples in the test set (don't cap to 16) ===
    try:
        max_to_save = len(test_loader.dataset)
    except Exception:
        # fallback: keep provided max_to_save if dataset length not available
        pass
    # ===================================================================

    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # batch expected as (images, masks)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, masks = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch format from test_loader - expected (images, masks)")

            images = images.to(device)
            preds = global_model(images)
            probs = torch.sigmoid(preds)
            bin_mask = (probs > 0.5).float()

            bsz = images.size(0)
            for b in range(bsz):
                mask_t = bin_mask[b].cpu()
                mask_arr = _mask_to_uint8(mask_t)

                # choose filename:
                fname = None
                if dataset_filenames is not None:
                    global_idx = batch_idx * test_loader.batch_size + b
                    if global_idx < len(dataset_filenames):
                        orig_name = dataset_filenames[global_idx]
                        base, _ext = os.path.splitext(orig_name)
                        fname = f"{base}_pred.png"
                if fname is None:
                    fname = f"{client_name}_pred_mask_{round_num if round_num is not None else 'r0'}_{batch_idx}_{b}.png"

                save_image(mask_arr, str(latest_dir / fname))
                saved += 1
                if saved >= max_to_save:
                    break
            if saved >= max_to_save:
                break

    print(f"Saved {saved} prediction masks for {client_name} in {latest_dir}")


# -----------------------
# Weights (default)
# -----------------------
def get_weights(net):
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict((k, torch.tensor(v)) for k, v in params_dict)
    net.load_state_dict(state_dict, strict=True)
