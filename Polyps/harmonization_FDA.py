import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os, copy, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from models.UNET import UNET
from dataset import CVCDataset   # must support (img_dir, mask_dir, transform) and optionally return_filename=True

# -------------------------
# Settings
# -------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()

out_dir = "Outputs"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Client dataset directories
# -------------------------
train_img_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
]
train_mask_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",
]
val_img_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_imgs",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\images",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\images",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\images"
]
val_mask_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_masks",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\masks",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\masks",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\masks"
]
test_img_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_imgs",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\images",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\images",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images"
]
test_mask_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\masks",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks"
]
client_names = ["Kvasir", "ETIS", "CVC-Colon","CVC-Clinic"]

# -------------------------
# Additional imports & helpers for saving visualizations
# -------------------------
import numpy as np
from PIL import Image
import shutil
from skimage import exposure  # kept if you still want histogram matching elsewhere

def _unnormalize_image(tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """
    tensor: torch tensor CxHxW normalized by mean/std
    returns uint8 HxWx3 numpy array
    """
    arr = tensor.cpu().numpy()
    if arr.ndim == 3:
        c, h, w = arr.shape
        arr = arr.transpose(1,2,0)
    else:
        # unexpected shape
        arr = arr
    arr = arr * np.array(std).reshape(1,1,3) + np.array(mean).reshape(1,1,3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr

def _mask_to_uint8(mask_tensor):
    """
    mask_tensor: torch tensor (1,H,W) or (H,W)
    returns uint8 HxW
    """
    m = mask_tensor.cpu().numpy()
    if m.ndim == 3:
        m = np.squeeze(m, axis=0)
    m = (m > 0.5).astype(np.uint8) * 255
    return m

def save_image(arr, path):
    Image.fromarray(arr).save(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Utils
# -------------------------
def get_loader(img_dir, mask_dir, transform, batch_size=8, shuffle=True, return_filename=True):
    """
    Builds a DataLoader. If your CVCDataset supports return_filename parameter,
    we request filenames to enable saving predictions with the original filenames.
    """
    # If CVCDataset doesn't accept return_filename kwarg, this will raise; if so set return_filename=False
    try:
        ds = CVCDataset(img_dir, mask_dir, transform=transform, return_filename=return_filename)
    except TypeError:
        ds = CVCDataset(img_dir, mask_dir, transform=transform)
        return_filename = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

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
    return dict(dice_with_bg=dice_with_bg, dice_no_bg=dice_no_bg,
                iou_with_bg=iou_with_bg, iou_no_bg=iou_no_bg,
                accuracy=acc, precision=precision, recall=recall, specificity=specificity)

def average_metrics(metrics_list):
    if not metrics_list:
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
# -------------------------
# FOURIER DOMAIN ADAPTATION (FDA) FUNCTIONS
# -------------------------
# -------------------------

def _get_first_kvasir_reference(ref_dir, resize_to=None):
    """
    Return the first image (sorted) from ref_dir as uint8 HxWx3 numpy array.
    This enforces "first training sample from Kvasir" requirement.
    """
    files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if not files:
        raise ValueError(f"No images found in Kvasir reference dir: {ref_dir}")
    first = files[0]
    p = os.path.join(ref_dir, first)
    ref = Image.open(p).convert("RGB")
    if resize_to is not None:
        ref = ref.resize(resize_to, resample=Image.BILINEAR)
    return np.array(ref).astype(np.uint8)

def fda_swap_amplitude(src_img, ref_img, L=0.1):
    """
    Fourier Domain Adaptation amplitude swapping:
    - src_img, ref_img: HxWx3 uint8 arrays (RGB)
    - L: proportion of low-frequency region to swap (0.0-0.5). Typical ~0.01-0.1
    Returns reconstructed uint8 HxWx3 image.
    """
    # convert to float
    src = src_img.astype(np.float32)
    ref = ref_img.astype(np.float32)

    h, w, _ = src.shape

    # Resize reference to source size if necessary
    if (ref.shape[0] != h) or (ref.shape[1] != w):
        ref = np.array(Image.fromarray(ref.astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)).astype(np.float32)

    # radius for low-frequency square
    b_h = int(np.floor(h * L))
    b_w = int(np.floor(w * L))
    if b_h < 1: b_h = 1
    if b_w < 1: b_w = 1

    # Center coordinates
    c_h = h // 2
    c_w = w // 2

    out = np.zeros_like(src, dtype=np.uint8)

    for ch in range(3):
        # FFT
        src_f = np.fft.fft2(src[:, :, ch])
        src_fshift = np.fft.fftshift(src_f)
        src_amp, src_pha = np.abs(src_fshift), np.angle(src_fshift)

        ref_f = np.fft.fft2(ref[:, :, ch])
        ref_fshift = np.fft.fftshift(ref_f)
        ref_amp = np.abs(ref_fshift)

        # swap low-frequency amplitude (centered square)
        h1 = c_h - b_h
        h2 = c_h + b_h
        w1 = c_w - b_w
        w2 = c_w + b_w

        # Clip indices
        h1 = max(0, h1); h2 = min(h, h2)
        w1 = max(0, w1); w2 = min(w, w2)

        src_amp[h1:h2, w1:w2] = ref_amp[h1:h2, w1:w2]

        # recombine
        combined = src_amp * np.exp(1j * src_pha)
        # inverse shifts and inverse fft
        combined_ishift = np.fft.ifftshift(combined)
        rec = np.fft.ifft2(combined_ishift)
        rec = np.real(rec)

        # clip and store
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        out[:, :, ch] = rec

    return out

def create_fda_datasets(train_img_dirs, train_mask_dirs,
                        val_img_dirs, val_mask_dirs,
                        test_img_dirs, test_mask_dirs,
                        client_names, out_base, reference_client_idx=0, L=0.05):
    """
    Create Fourier-domain-adapted copies for each client's train/val/test images
    using the first image of the reference client's training folder.
    Saves into out_base/FDA/<client>/<split>_images and corresponding masks (copied).
    Returns lists of new directories analogous to create_histogram_matched_datasets.
    """
    base = os.path.join(out_base, "FDA")
    ensure_dir(base)

    # get the FIRST training image of the reference client as requested
    ref_dir = train_img_dirs[reference_client_idx]
    print(f"[FDA] Loading reference image (first sample) from {ref_dir} ...")
    ref_img_first = _get_first_kvasir_reference(ref_dir)  # uint8 array

    # create per-client fda copies
    train_img_fda_dirs = []
    train_mask_fda_dirs = []
    val_img_fda_dirs = []
    val_mask_fda_dirs = []
    test_img_fda_dirs = []
    test_mask_fda_dirs = []

    for i, cname in enumerate(client_names):
        client_base = os.path.join(base, cname)
        train_dst_img = os.path.join(client_base, "train_images")
        train_dst_mask = os.path.join(client_base, "train_masks")
        val_dst_img = os.path.join(client_base, "val_images")
        val_dst_mask = os.path.join(client_base, "val_masks")
        test_dst_img = os.path.join(client_base, "test_images")
        test_dst_mask = os.path.join(client_base, "test_masks")
        print(f"[FDA] Harmonizing (FDA) client {cname} train/val/test ...")

        # helper to process a split
        def _process_split(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
            ensure_dir(dst_img_dir); ensure_dir(dst_mask_dir)
            img_files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            for fn in img_files:
                src_img_p = os.path.join(src_img_dir, fn)
                dst_img_p = os.path.join(dst_img_dir, fn)
                try:
                    src_img = np.array(Image.open(src_img_p).convert("RGB")).astype(np.uint8)
                except Exception as e:
                    print(f"[FDA] Warning: failed to load {src_img_p}: {e}")
                    continue
                # perform FDA with ref image resized inside function
                matched = fda_swap_amplitude(src_img, ref_img_first, L=L)
                Image.fromarray(matched).save(dst_img_p)

                # copy mask if exists (same filename)
                src_mask_p = os.path.join(src_mask_dir, fn)
                if os.path.exists(src_mask_p):
                    try:
                        shutil.copy(src_mask_p, os.path.join(dst_mask_dir, fn))
                    except Exception:
                        # try reading/writing
                        try:
                            m = Image.open(src_mask_p)
                            m.save(os.path.join(dst_mask_dir, fn))
                        except Exception:
                            pass
                else:
                    # try alternative extensions
                    base, _ = os.path.splitext(fn)
                    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                        alt = os.path.join(src_mask_dir, base + ext)
                        if os.path.exists(alt):
                            shutil.copy(alt, os.path.join(dst_mask_dir, base + ext))
                            break

        # process train/val/test
        _process_split(train_img_dirs[i], train_mask_dirs[i], train_dst_img, train_dst_mask)
        _process_split(val_img_dirs[i], val_mask_dirs[i], val_dst_img, val_dst_mask)
        _process_split(test_img_dirs[i], test_mask_dirs[i], test_dst_img, test_dst_mask)

        train_img_fda_dirs.append(train_dst_img); train_mask_fda_dirs.append(train_dst_mask)
        val_img_fda_dirs.append(val_dst_img); val_mask_fda_dirs.append(val_dst_mask)
        test_img_fda_dirs.append(test_dst_img); test_mask_fda_dirs.append(test_dst_mask)

    print("[FDA] Done creating FDA-harmonized datasets.")
    return train_img_fda_dirs, train_mask_fda_dirs, val_img_fda_dirs, val_mask_fda_dirs, test_img_fda_dirs, test_mask_fda_dirs

# -------------------------
# Existing sample/grid/prediction functions (kept unchanged)
# -------------------------
def save_transformed_samples(img_dir, mask_dir, transform, client_name, out_base, n_samples=8, prefix="harmonized"):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
    dest = os.path.join(out_base, f"{client_name}", prefix)
    ensure_dir(dest)
    num = min(n_samples, len(ds))
    for i in range(num):
        try:
            item = ds[i]
            # support either (img,mask) or (img,mask,filename)
            if isinstance(item, tuple) and len(item) >= 2:
                img_t, mask_t = item[0], item[1]
            else:
                raise ValueError("Unexpected dataset __getitem__ return")
        except Exception as e:
            raise

        # If img_t is an albumentations->ToTensorV2 result (tensor)
        if isinstance(img_t, np.ndarray):
            img_arr = img_t.astype(np.uint8)
            save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))
        else:
            img_arr = _unnormalize_image(img_t)
            save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))

        # mask
        if isinstance(mask_t, np.ndarray):
            m_arr = (mask_t > 0.5).astype(np.uint8) * 255
            save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))
        else:
            m_arr = _mask_to_uint8(mask_t)
            save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))

def save_test_predictions(global_model, test_loader, client_name, out_base=None, round_num=None, max_to_save=16, device_arg=None):
    """
    Save predicted masks (as .png) for test images for a client.
    - If the DataLoader/dataset returns filenames (third element in batch), those filenames will be used (keeps original order).
    - If filenames are not available, fallback names are used.
    """
    # fallback out_base
    if out_base is None:
        out_base = out_dir

    if device_arg is None:
        device = DEVICE if "DEVICE" in globals() else torch.device("cpu")
    else:
        device = device_arg

    global_model.eval()
    latest_dir = os.path.join(out_base, "TestPreds", client_name, "latest")

    # Clear latest dir then recreate
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    ensure_dir(latest_dir)

    saved = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # batch can be (data, target) or (data, target, filenames)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, target, fnames = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, target = batch
                fnames = None
            else:
                # unexpected structure
                raise RuntimeError("Unexpected batch format from test_loader")

            # ensure proper mask/channel dims in target if used anywhere
            if target.dim() == 3:
                # B x H x W -> B x 1 x H x W
                target = target.unsqueeze(1)
            # otherwise assume B x 1 x H x W

            data = data.to(device)
            preds = global_model(data)
            probs = torch.sigmoid(preds)
            bin_mask = (probs > 0.5).float()

            # Save each prediction in the batch
            b_sz = data.size(0)
            for b in range(b_sz):
                mask_t = bin_mask[b].cpu()
                mask_arr = _mask_to_uint8(mask_t)   # convert to 0/255 uint8

                # Use original filename if available to preserve ordering
                if fnames is not None:
                    try:
                        # fnames may be list of strings or tuple; handle accordingly
                        orig_name = fnames[b]
                        base, ext = os.path.splitext(orig_name)
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

def make_comparison_grid_and_histograms_updated(img_dir, mask_dir, val_transform, visual_transform,
                                                client_name, out_base, n_samples=7, diff_amp=4.0):
    base_dest = os.path.join(out_base, "HarmonizedSamples", client_name)
    diffs_dest = os.path.join(base_dest, "diffs")
    hist_dest = os.path.join(base_dest, "histograms")
    ensure_dir(base_dest); ensure_dir(diffs_dest); ensure_dir(hist_dest)

    fnames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])[:n_samples]
    if len(fnames) == 0:
        print(f"No images found in {img_dir} for {client_name}")
        return

    top_imgs, mid_imgs, diff_color_imgs, short_names = [], [], [], []

    for fname in fnames:
        img_path = os.path.join(img_dir, fname)
        orig_pil = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig_pil)  # HxWx3 uint8

        # visual (original resized)
        vis_out = visual_transform(image=orig_np)
        vis_img = vis_out['image']
        top = vis_img.astype(np.uint8) if isinstance(vis_img, np.ndarray) else np.array(vis_img).astype(np.uint8)

        # harmonized via val_transform (Resize + Normalize + ToTensorV2 typically)
        vt_out = val_transform(image=orig_np)
        img_t = vt_out['image']
        if isinstance(img_t, torch.Tensor):
            mid = _unnormalize_image(img_t)
        elif isinstance(img_t, np.ndarray):
            hnp = img_t
            if hnp.dtype != np.uint8:
                if hnp.max() <= 1.5:
                    hnp = (np.clip(hnp,0,1)*255).astype(np.uint8)
                else:
                    hnp = np.clip(hnp,0,255).astype(np.uint8)
            mid = hnp.astype(np.uint8)
        else:
            mid = top.copy()

        # Ensure consistent size (use mid size)
        h_h, h_w = mid.shape[0], mid.shape[1]
        orig_resized = np.array(orig_pil.resize((h_w, h_h), resample=Image.BILINEAR)).astype(np.uint8)
        if top.shape[0] != h_h or top.shape[1] != h_w:
            top = np.array(Image.fromarray(top).resize((h_w, h_h), resample=Image.BILINEAR)).astype(np.uint8)

        # compute color absolute diff per channel (uint8)
        diff_color = np.abs(orig_resized.astype(int) - mid.astype(int)).astype(np.uint8)  # HxWx3

        # Amplified diff for visibility
        amp = np.clip((diff_color.astype(float) * diff_amp), 0, 255).astype(np.uint8)

        # Save color diff and amplified diff to diffs_dest
        diff_fname = os.path.join(diffs_dest, f"diff_color_{fname}")
        amp_fname  = os.path.join(diffs_dest, f"diff_color_amp_{fname}")
        save_image(diff_color, diff_fname)
        save_image(amp, amp_fname)

        # Save RGB histograms (orig_resized vs mid) into hist_dest (ONLY histograms there)
        fig, ax = plt.subplots(1,3, figsize=(12,3))
        colors = ['r','g','b']
        for ch in range(3):
            ax[ch].hist(orig_resized[:,:,ch].ravel(), bins=256, alpha=0.5, label='orig', color=colors[ch])
            ax[ch].hist(mid[:,:,ch].ravel(), bins=256, alpha=0.5, label='harm', color=colors[ch], histtype='step')
            ax[ch].legend(fontsize=6)
            ax[ch].set_title(['R','G','B'][ch])
        plt.tight_layout()
        hist_path = os.path.join(hist_dest, f"hist_{fname}.png")
        plt.savefig(hist_path)
        plt.close(fig)

        # Collect for large grid
        top_imgs.append(orig_resized)
        mid_imgs.append(mid)
        diff_color_imgs.append(amp)  # show amplified diff in the grid for visibility
        short_names.append(fname if len(fname) <= 20 else fname[:17] + "...")

    # Build 3-row grid (Original | Harmonized | Amplified Color Diff)
    n = len(top_imgs)
    fig_w = max(3 * n, 8)
    fig_h = 6
    fig, axs = plt.subplots(3, n, figsize=(fig_w, fig_h))
    if n == 1:
        axs = np.array([[axs[0]],[axs[1]],[axs[2]]])

    fig.suptitle(f"Harmonized (Basic preprocessing + Normalization + Consistent augmentation.) vs. Original: {client_name}", fontsize=16, y=0.98)
    fig.text(0.01, 0.82, "Original\nimages", fontsize=12, va='center', rotation='vertical')
    fig.text(0.01, 0.50, "Harmonized\nimages", fontsize=12, va='center', rotation='vertical')
    fig.text(0.01, 0.18, "Amplified\nDifference", fontsize=12, va='center', rotation='vertical')

    for i in range(n):
        axs[0, i].imshow(top_imgs[i]); axs[0, i].axis('off'); axs[0, i].set_title(short_names[i], fontsize=8)
        axs[1, i].imshow(mid_imgs[i]); axs[1, i].axis('off')
        axs[2, i].imshow(diff_color_imgs[i]); axs[2, i].axis('off')

    plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.94])
    plt.subplots_adjust(wspace=0.01, hspace=0.02)
    grid_path = os.path.join(base_dest, f"comparison_grid_{client_name}.png")
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison grid: {grid_path}")
    print(f"Saved color diffs in: {diffs_dest} (raw + amplified).")
    print(f"Saved RGB histograms in: {hist_dest} (histograms only).")

# -------------------------
# Training / Eval functions (kept the same)
# -------------------------
def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss, metrics = 0.0, []
    for _ in range(LOCAL_EPOCHS):
        for batch in tqdm(loader, leave=False):
            # Support (data,target) or (data,target,filenames)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, target = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch format in train_local")

            # Ensure target has shape Bx1xHxW
            if target.dim() == 3:
                target = target.unsqueeze(1).float()
            elif target.dim() == 4:
                target = target.float()
            else:
                raise RuntimeError(f"Unexpected target dims: {target.shape}")

            data, target = data.to(DEVICE), target.to(DEVICE)
            preds = model(data)
            loss = loss_fn(preds, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            metrics.append(compute_metrics(preds.detach(), target))
    avg_metrics = average_metrics(metrics)
    if len(loader.dataset) > 0:
        avg_loss = total_loss / len(loader.dataset)
    else:
        avg_loss = total_loss
    print("Train: " + " | ".join([f"{k}: {v:.4f}" for k,v in (avg_metrics or {}).items()]))
    return avg_loss, avg_metrics

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss, metrics = 0.0, []
    n_items = 0
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

        data, target = data.to(DEVICE), target.to(DEVICE)
        preds = model(data)
        loss = loss_fn(preds, target)
        total_loss += loss.item()
        metrics.append(compute_metrics(preds, target))
        n_items += 1
    avg_metrics = average_metrics(metrics) if metrics else {}
    avg_loss = (total_loss / n_items) if n_items > 0 else 0.0
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k,v in (avg_metrics or {}).items()]))
    return avg_loss, avg_metrics

# -------------------------
# Plotting (kept the same)
# -------------------------
def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))

    # Dice_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Per-client Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_harm1.png"))
    plt.close()

    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Per-client IoU ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_harm1.png"))
    plt.close()

# -------------------------
# Main FedAvg
# -------------------------
def main():
    # -------------------------
    # Basic preprocessing + normalization + consistent augmentation
    # -------------------------
    tr_tf = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        # photometric augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Validation / test: only resize + normalize
    global val_transform  # used by comparison grid function
    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = val_tf  # alias for the comparison function

    # VISUAL transforms used to create original-resized images for comparison (no Normalize/ToTensor)
    visual_val_tf = A.Compose([
        A.Resize(224, 224),
    ])

    # -------------------------
    # Optionally create FDA copies and use those paths (or default original dirs)
    # -------------------------
    USE_FDA = True   # <-- set to False to use original directories instead
    FDA_L = 0.05     # proportion of low-frequency region to swap (tune e.g. 0.02 - 0.1)

    if USE_FDA:
        (train_img_dirs_used, train_mask_dirs_used,
         val_img_dirs_used, val_mask_dirs_used,
         test_img_dirs_used, test_mask_dirs_used) = create_fda_datasets(
            train_img_dirs, train_mask_dirs,
            val_img_dirs, val_mask_dirs,
            test_img_dirs, test_mask_dirs,
            client_names, out_dir,
            reference_client_idx=0,
            L=FDA_L
        )
    else:
        train_img_dirs_used = train_img_dirs; train_mask_dirs_used = train_mask_dirs
        val_img_dirs_used = val_img_dirs; val_mask_dirs_used = val_mask_dirs
        test_img_dirs_used = test_img_dirs; test_mask_dirs_used = test_mask_dirs

    # -------------------------
    # Save harmonized/augmented samples and comparison grids BEFORE training
    # -------------------------
    print("Saving harmonized samples (val-style) and augmented samples (train-style) for each client...")
    visuals_base = os.path.join(out_dir, "HarmonizedSamples")
    for i in range(NUM_CLIENTS):
        cname = client_names[i]
        # harmonized (deterministic - val transform) - save transformed images and masks (first 7)
        save_transformed_samples(val_img_dirs_used[i], val_mask_dirs_used[i], val_tf, cname, visuals_base, n_samples=7, prefix="harmonized")
        # augmented (train transform - random) - also save first 7 augmented examples if you want
        save_transformed_samples(train_img_dirs_used[i], train_mask_dirs_used[i], tr_tf, cname, visuals_base, n_samples=7, prefix="augmented")

        # Create and save the comparison grid (top: original resized, bottom: harmonized)
        # Note: this uses val_transform (Resize+Normalize->ToTensor) to produce "harmonized" images for display
        make_comparison_grid_and_histograms_updated(val_img_dirs_used[i], val_mask_dirs_used[i], val_tf, visual_val_tf, cname, visuals_base, n_samples=7)
    print("Saved harmonized/augmented samples and comparison grids to Outputs/HarmonizedSamples/")

    # -------------------------
    # Federated training loop (FedAvg)
    # -------------------------
    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models, weights = [], []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(train_img_dirs_used[i], train_mask_dirs_used[i], tr_tf, batch_size=8, shuffle=True, return_filename=True)
            val_loader = get_loader(val_img_dirs_used[i], val_mask_dirs_used[i], val_tf, batch_size=8, shuffle=False, return_filename=True)

            print(f"[Client {client_names[i]}]")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz

        # FedAvg
        norm_weights = [w/total_sz for w in weights]
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

        # ---- Testing ----
        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs_used[i], test_mask_dirs_used[i], val_tf, batch_size=8, shuffle=False, return_filename=True)
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)

            save_test_predictions(global_model, test_loader, client_names[i], out_base=out_dir, round_num=(r+1), max_to_save=int(len(test_loader.dataset)), device_arg=DEVICE)

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()
