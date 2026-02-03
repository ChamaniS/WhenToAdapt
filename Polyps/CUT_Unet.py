# cycle_gan_harmonize_and_fedavg.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os, copy, time, random, math, shutil, stat
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T
import torchvision.models as tv_models
from torch.nn.utils import spectral_norm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from models.unet import UNET
from dataset import CVCDataset

def _on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def copy_tree_force(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst, onerror=_on_rm_error)
    shutil.copytree(src, dst)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10

CUT_EPOCHS =60
BATCH_CUT = 4
LAMBDA_NCE = 1.0
NCE_NUM_PATCHES = 256
NCE_LAYERS = [1, 2, 3, 4]
LR_G = 2e-4
LR_D = 2e-4
OUT_DIR = "Outputs"
os.makedirs(OUT_DIR, exist_ok=True)


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
reference_idx = 0  # Kvasir(0) as canonical reference


REF_SEGMENTER_PATH = os.path.join(OUT_DIR, "ref_segmenter.pth")


LAMBDA_PERCEPTUAL = 0.05
LAMBDA_SEG = 5.0

class ImageFolderSimple(Dataset):
    def __init__(self, folder, size=(224,224), augment=False):
        self.files = sorted([p for p in glob(os.path.join(folder, "*")) if p.lower().endswith(('.png','.jpg','.jpeg'))])
        self.size = size
        self.augment = augment
        self.base_trans = T.Compose([T.Resize(self.size), T.CenterCrop(self.size), T.ToTensor()])  # returns [0,1]
        # pixel range [0,1]
        self.aug_trans = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10)
        ])
    def __len__(self): return max(1, len(self.files))
    def __getitem__(self, idx):
        p = self.files[idx % len(self.files)]
        img = Image.open(p).convert('RGB')
        if self.augment:
            img = self.aug_trans(img)
        t = self.base_trans(img)  # tensor CxHxW [0..1]
        return t

def enc_block(in_ch, out_ch, down=True):
    if down:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True)
        )

class CUTGenerator(nn.Module):
    def __init__(self, in_ch=3, ngf=64, n_down=4):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 7, 1, 3, bias=False), nn.InstanceNorm2d(ngf), nn.ReLU(True))
        self.enc2 = enc_block(ngf, ngf*2, down=True)   # /2
        self.enc3 = enc_block(ngf*2, ngf*4, down=True) # /4
        self.enc4 = enc_block(ngf*4, ngf*8, down=True) # /8
        self.enc5 = enc_block(ngf*8, ngf*8, down=True) # /16
        # decoder (mirror)
        self.dec5 = enc_block(ngf*8, ngf*8, down=False)
        self.dec4 = enc_block(ngf*16, ngf*4, down=False)
        self.dec3 = enc_block(ngf*8, ngf*2, down=False)
        self.dec2 = enc_block(ngf*4, ngf, down=False)
        self.final = nn.Sequential(nn.Conv2d(ngf*2, in_ch, 7, 1, 3), nn.Tanh())

    def encode(self, x):
        f1 = self.enc1(x)   # full res
        f2 = self.enc2(f1)  # /2
        f3 = self.enc3(f2)  # /4
        f4 = self.enc4(f3)  # /8
        f5 = self.enc5(f4)  # /16
        return [f1, f2, f3, f4, f5]

    def decode(self, feats):
        f1, f2, f3, f4, f5 = feats
        d5 = self.dec5(f5)
        d5_cat = torch.cat([d5, f4], dim=1)
        d4 = self.dec4(d5_cat)
        d4_cat = torch.cat([d4, f3], dim=1)
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, f2], dim=1)
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, f1], dim=1)
        out = self.final(d2_cat)
        return out

    def forward(self, x):
        feats = self.encode(x)
        out = self.decode(feats)
        return out, feats  # return output and encoder features


class PatchSampleF(nn.Module):
    def __init__(self, nc, inner_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, inner_dim, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 1, 1, 0, bias=True)
        )

    def forward(self, feat):  # feat: [B,C,H,W]
        return self.net(feat)  # [B,inner_dim,H,W]

class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.matmul(q, k.T) / self.temperature  # [N,N]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        return loss


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, 1, 4, 1, 1)
        )
    def forward(self, x): return self.model(x)

def make_vgg16_features(device):
    vgg = tv_models.vgg16(pretrained=True).features.to(device).eval()
    for p in vgg.parameters(): p.requires_grad = False
    return vgg

def vgg_feature_list(vgg, x, layers=(3,8,15)):
    feats = []
    out = x
    for i, layer in enumerate(vgg):
        out = layer(out)
        if i in layers:
            feats.append(out)
    return feats


def train_cut(domainA_dir, domainB_dir, save_dir, epochs=CUT_EPOCHS, device=DEVICE,
              lambda_nce=LAMBDA_NCE, nce_layers=NCE_LAYERS, n_patches=NCE_NUM_PATCHES,
              lr_g=LR_G, lr_d=LR_D, batch_size=BATCH_CUT, ref_segmenter=None):
    os.makedirs(save_dir, exist_ok=True)
    dsA = ImageFolderSimple(domainA_dir, size=(224,224), augment=True)
    dsB = ImageFolderSimple(domainB_dir, size=(224,224), augment=True)

    max_samples = max(len(dsA), len(dsB))
    samplerA = RandomSampler(dsA, replacement=True, num_samples=max_samples)
    samplerB = RandomSampler(dsB, replacement=True, num_samples=max_samples)
    loaderA = DataLoader(dsA, batch_size=batch_size, sampler=samplerA, drop_last=True, num_workers=2)
    loaderB = DataLoader(dsB, batch_size=batch_size, sampler=samplerB, drop_last=True, num_workers=2)

    G = CUTGenerator().to(device)
    D = PatchDiscriminator().to(device)
    G.apply(weights_init_normal) if 'weights_init_normal' in globals() else None
    D.apply(weights_init_normal) if 'weights_init_normal' in globals() else None

    ngf = 64
    layer_nc = {1: ngf, 2: ngf*2, 3: ngf*4, 4: ngf*8, 5: ngf*8}
    nce_mlps = {}
    for l in nce_layers:
        nc = layer_nc.get(l, ngf*4)
        nce_mlps[str(l)] = PatchSampleF(nc, inner_dim=256).to(device)

    nce_loss = PatchNCELoss().to(device)
    criterion_GAN = nn.MSELoss().to(device)

    mlp_params = []
    for m in nce_mlps.values():
        mlp_params += list(m.parameters())

    opt_G = optim.Adam(list(G.parameters()) + mlp_params, lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5,0.999))
    vgg = make_vgg16_features(device) if (LAMBDA_PERCEPTUAL > 0 and ref_segmenter is None) else None

    real_label = 0.9
    fake_label = 0.0

    print(f"[CUT] Train {domainA_dir} -> {domainB_dir} for {epochs} epochs")
    iterB = iter(loaderB)
    for epoch in range(epochs):
        loop = tqdm(loaderA, desc=f"Epoch {epoch+1}/{epochs}")
        for real_A in loop:
            try:
                real_B = next(iterB)
            except StopIteration:
                iterB = iter(loaderB)
                real_B = next(iterB)
            real_A = real_A.to(device)  # [0,1]
            real_B = real_B.to(device)

            fake_B, feats_A = G((real_A * 2.0) - 1.0)
            fake_B_for_enc = (fake_B + 1.0) / 2.0

            feats_fake = G.encode(fake_B_for_enc)

            opt_D.zero_grad()
            pred_real = D(real_B * 2.0 - 1.0)
            valid_tensor = torch.full_like(pred_real, real_label, device=device)
            loss_D_real = criterion_GAN(pred_real, valid_tensor)
            pred_fake = D(fake_B.detach())
            fake_tensor = torch.full_like(pred_fake, fake_label, device=device)
            loss_D_fake = criterion_GAN(pred_fake, fake_tensor)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad()
            pred_fake_for_g = D(fake_B)
            valid_tensor = torch.full_like(pred_fake_for_g, real_label, device=device)
            loss_G_GAN = criterion_GAN(pred_fake_for_g, valid_tensor)

            loss_NCE = 0.0
            for l in nce_layers:
                idx = l - 1
                if idx >= len(feats_A) or idx < 0: continue
                feat_q = feats_A[idx]
                feat_k = feats_fake[idx]
                proj = nce_mlps[str(l)]
                q_proj = proj(feat_q)  # [B,dim,H,W]
                k_proj = proj(feat_k)
                Bn, Cn, Hn, Wn = q_proj.shape
                num_patches = min(n_patches, Hn*Wn)
                # sample random spatial positions
                idxs = torch.randperm(Hn*Wn, device=device)[:num_patches]
                q_patches = q_proj.view(Bn, Cn, -1)[:,:,idxs]  # [B, C, P]
                k_patches = k_proj.view(Bn, Cn, -1)[:,:,idxs]
                # reshape to [B*P, C]
                q_flat = q_patches.permute(0,2,1).reshape(-1, Cn)
                k_flat = k_patches.permute(0,2,1).reshape(-1, Cn)
                loss_layer = nce_loss(q_flat, k_flat)
                loss_NCE += loss_layer
            loss_NCE = loss_NCE * lambda_nce

            # optional perceptual loss (if vgg used)
            loss_perc = 0.0
            if vgg is not None and LAMBDA_PERCEPTUAL > 0:
                fake_norm = (fake_B + 1.0) / 2.0
                realA_norm = real_A
                mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
                std = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
                f_fake = (fake_norm - mean) / std
                f_real = (realA_norm - mean) / std
                feats_fake_v = vgg_feature_list(vgg, f_fake)
                feats_real_v = vgg_feature_list(vgg, f_real)
                for fa, fr in zip(feats_fake_v, feats_real_v):
                    loss_perc += nn.L1Loss()(fa, fr)
                loss_perc = loss_perc * LAMBDA_PERCEPTUAL

            # total G loss
            loss_G = loss_G_GAN + loss_NCE
            if isinstance(loss_perc, torch.Tensor):
                loss_G = loss_G + loss_perc
            loss_G.backward()
            opt_G.step()

            loop.set_postfix({
                "loss_G": loss_G.item(),
                "loss_D": loss_D.item(),
                "loss_NCE": loss_NCE.item() if isinstance(loss_NCE, torch.Tensor) else float(loss_NCE)
            })

        # save checkpoint per epoch
        torch.save({
            'G': G.state_dict(),
            'D': D.state_dict(),
            'mlps': {k: v.state_dict() for k, v in nce_mlps.items()},
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict()
        }, os.path.join(save_dir, f"cut_epoch_{epoch+1}.pth"))

    # final save
    torch.save({'G': G.state_dict(), 'mlps': {k: v.state_dict() for k, v in nce_mlps.items()}},
               os.path.join(save_dir, "cut_final.pth"))
    print(f"[CUT] finished and saved to {save_dir}")
    return G.cpu()


def harmonize_folder_with_generator(generator, src_dir, dst_dir, mask_src_dir=None, mask_dst_dir=None, device=DEVICE, size=(224,224)):
    os.makedirs(dst_dir, exist_ok=True)
    if mask_src_dir and mask_dst_dir:
        os.makedirs(mask_dst_dir, exist_ok=True)
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    generator = generator.to(device)
    generator.eval()
    with torch.no_grad():
        for p in sorted([f for f in glob(os.path.join(src_dir, "*")) if f.lower().endswith(('.png','.jpg','.jpeg'))]):
            img = Image.open(p).convert('RGB')
            inp = tf(img).unsqueeze(0).to(device) * 2.0 - 1.0
            out, _ = generator(inp)
            out = (out.squeeze(0).detach().cpu().clamp(-1,1) + 1.0) / 2.0  # [0..1]
            out_img = T.ToPILImage()(out)
            basename = os.path.basename(p)
            out_img.save(os.path.join(dst_dir, basename))
            # copy mask
            if mask_src_dir and mask_dst_dir:
                src_mask_p = os.path.join(mask_src_dir, basename)
                if os.path.exists(src_mask_p):
                    shutil.copy(src_mask_p, os.path.join(mask_dst_dir, basename))
                else:
                    # try alternative extensions
                    base, _ = os.path.splitext(basename)
                    for ext in ('.png','.jpg','.jpeg','.bmp'):
                        alt = os.path.join(mask_src_dir, base + ext)
                        if os.path.exists(alt):
                            shutil.copy(alt, os.path.join(mask_dst_dir, base + ext))
                            break


tr_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0] * 3, std=[1] * 3),
    ToTensorV2()
])
val_tf = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0] * 3, std=[1] * 3), ToTensorV2()])
visual_val_tf = A.Compose([A.Resize(224, 224)])


def _unnormalize_image(tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    arr = tensor.cpu().numpy()
    if arr.ndim == 3:
        c,h,w = arr.shape
        arr = arr.transpose(1,2,0)
    arr = arr * np.array(std).reshape(1,1,3) + np.array(mean).reshape(1,1,3)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
def _mask_to_uint8(mask_tensor):
    m = mask_tensor.cpu().numpy()
    if m.ndim == 3:
        m = np.squeeze(m, axis=0)
    m = (m > 0.5).astype(np.uint8) * 255
    return m
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def save_image(arr, path): Image.fromarray(arr).save(path)

def save_transformed_samples(img_dir, mask_dir, transform, client_name, out_base, n_samples=8, prefix="harmonized"):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
    dest = os.path.join(out_base, f"{client_name}", prefix)
    ensure_dir(dest)
    num = min(n_samples, len(ds))
    for i in range(num):
        try:
            item = ds[i]
            if isinstance(item, tuple) and len(item) >= 2:
                img_t, mask_t = item[0], item[1]
            else:
                raise ValueError("Unexpected dataset __getitem__ return")
        except Exception as e:
            raise
        if isinstance(img_t, np.ndarray):
            img_arr = img_t.astype(np.uint8)
            save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))
        else:
            img_arr = _unnormalize_image(img_t)
            save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))
        if isinstance(mask_t, np.ndarray):
            m_arr = (mask_t > 0.5).astype(np.uint8) * 255
            save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))
        else:
            m_arr = _mask_to_uint8(mask_t)
            save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))

def save_test_predictions(global_model, test_loader, client_name, out_base=None, round_num=None, max_to_save=16, device_arg=None):
    if out_base is None: out_base = OUT_DIR
    device = DEVICE if device_arg is None else device_arg
    global_model.eval()
    latest_dir = os.path.join(out_base, "TestPreds", client_name, "latest")
    if os.path.exists(latest_dir): shutil.rmtree(latest_dir)
    ensure_dir(latest_dir)
    saved = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, target, fnames = batch
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, target = batch[0], batch[1]
                fnames = None
            else:
                raise RuntimeError("Unexpected batch format from test_loader")
            if target.dim() == 3:
                target = target.unsqueeze(1)
            data = data.to(device)
            preds = global_model(data)
            probs = torch.sigmoid(preds)
            bin_mask = (probs > 0.5).float()
            bsz = data.size(0)
            for b in range(bsz):
                mask_t = bin_mask[b].cpu()
                mask_arr = _mask_to_uint8(mask_t)
                if fnames is not None:
                    try:
                        orig_name = fnames[b]
                        base, _ = os.path.splitext(orig_name)
                        fname = f"{base}_pred.png"
                    except Exception:
                        fname = f"{client_name}_pred_mask_{idx}_{b}.png"
                else:
                    fname = f"{client_name}_pred_mask_{idx}_{b}.png"
                save_image(mask_arr, os.path.join(latest_dir, fname))
                saved += 1
                if saved >= max_to_save: break
            if saved >= max_to_save: break
    print(f"Saved {saved} prediction masks for {client_name} in {latest_dir}")

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
    dice_no_bg = max(0.0, min(1.0, dice_no_bg))
    iou_no_bg = max(0.0, min(1.0, iou_no_bg))
    dice_with_bg = max(0.0, min(1.0, dice_with_bg))
    iou_with_bg = max(0.0, min(1.0, iou_with_bg))
    return dict(dice_with_bg=dice_with_bg, dice_no_bg=dice_no_bg,
                iou_with_bg=iou_with_bg, iou_no_bg=iou_no_bg,
                accuracy=acc, precision=precision, recall=recall, specificity=specificity)

def average_metrics(metrics_list):
    if not metrics_list: return {}
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_loss_fn(device): return smp.losses.DiceLoss(mode="binary", from_logits=True).to(device)
def average_models_weighted(models, weights):
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd

def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss, metrics = 0.0, []
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
            data, target = data.to(DEVICE), target.to(DEVICE)
            preds = model(data)
            loss = loss_fn(preds, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            metrics.append(compute_metrics(preds.detach(), target))
    avg_metrics = average_metrics(metrics)
    avg_loss = total_loss / max(1, len(loader.dataset))
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

def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("Dice"); plt.title("Per-client Dice"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_cut.png")); plt.close()
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("IoU"); plt.title("Per-client IoU"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_cut.png")); plt.close()

def main():
    ref_segmenter = None
    if os.path.exists(REF_SEGMENTER_PATH):
        print("[HARM] Loading reference segmenter for seg-consistency loss from:", REF_SEGMENTER_PATH)
        ref_segmenter = UNET(in_channels=3, out_channels=1).to(DEVICE)
        ck = torch.load(REF_SEGMENTER_PATH, map_location=DEVICE)
        ref_segmenter.load_state_dict(ck)
        ref_segmenter.eval()
        for p in ref_segmenter.parameters():
            p.requires_grad = False
    else:
        print("[HARM] No reference segmenter found at REF_SEGMENTER_PATH; seg-consistency loss will be skipped.")

    cut_models = {}
    for i in range(NUM_CLIENTS):
        if i == reference_idx:
            print(f"[HARM] Skipping CUT for reference client {client_names[i]}")
            continue
        a_dir = train_img_dirs[i]
        b_dir = train_img_dirs[reference_idx]
        save_dir = os.path.join(OUT_DIR, "CUT", f"{client_names[i]}_to_{client_names[reference_idx]}")
        os.makedirs(save_dir, exist_ok=True)
        final_ckpt = os.path.join(save_dir, "cut_final.pth")
        if os.path.exists(final_ckpt):
            G = CUTGenerator()
            ck = torch.load(final_ckpt, map_location='cpu')
            G.load_state_dict(ck['G'] if 'G' in ck else ck['G_state_dict'] if 'G_state_dict' in ck else ck['G'])
            G.to(DEVICE)
            cut_models[i] = G.cpu()
            print(f"[HARM] Loaded existing CUT generator for {client_names[i]} -> {client_names[reference_idx]}")
        else:
            G = train_cut(a_dir, b_dir, save_dir, epochs=CUT_EPOCHS, device=DEVICE, lambda_nce=LAMBDA_NCE,
                          nce_layers=NCE_LAYERS, n_patches=NCE_NUM_PATCHES, lr_g=LR_G, lr_d=LR_D, batch_size=BATCH_CUT,
                          ref_segmenter=ref_segmenter)
            cut_models[i] = G.cpu()

    hist_base = os.path.join(OUT_DIR, "CUT_Harmonized")
    hm_train_dirs = []
    hm_train_mask_dirs = []
    hm_val_dirs = []
    hm_val_mask_dirs = []
    hm_test_dirs = []
    hm_test_mask_dirs = []
    for i in range(NUM_CLIENTS):
        cname = client_names[i]
        if i == reference_idx:
            dst_train = os.path.join(hist_base, cname, "train_images")
            dst_val = os.path.join(hist_base, cname, "val_images")
            dst_test = os.path.join(hist_base, cname, "test_images")
            dst_train_mask = os.path.join(hist_base, cname, "train_masks")
            dst_val_mask = os.path.join(hist_base, cname, "val_masks")
            dst_test_mask = os.path.join(hist_base, cname, "test_masks")
            copy_tree_force(train_img_dirs[i], dst_train)
            copy_tree_force(val_img_dirs[i], dst_val)
            copy_tree_force(test_img_dirs[i], dst_test)
            copy_tree_force(train_mask_dirs[i], dst_train_mask)
            copy_tree_force(val_mask_dirs[i], dst_val_mask)
            copy_tree_force(test_mask_dirs[i], dst_test_mask)
        else:
            G = cut_models[i]
            dst_train = os.path.join(hist_base, cname, "train_images"); dst_val = os.path.join(hist_base, cname, "val_images"); dst_test = os.path.join(hist_base, cname, "test_images")
            dst_train_mask = os.path.join(hist_base, cname, "train_masks"); dst_val_mask = os.path.join(hist_base, cname, "val_masks"); dst_test_mask = os.path.join(hist_base, cname, "test_masks")
            print(f"[HARM] Harmonizing {cname} -> {client_names[reference_idx]} (train/val/test) using CUT model")
            ensure_dir(dst_train); ensure_dir(dst_val); ensure_dir(dst_test)
            ensure_dir(dst_train_mask); ensure_dir(dst_val_mask); ensure_dir(dst_test_mask)
            # harmonize each split
            harmonize_folder_with_generator(G, train_img_dirs[i], dst_train, mask_src_dir=train_mask_dirs[i], mask_dst_dir=dst_train_mask, device=DEVICE, size=(224,224))
            harmonize_folder_with_generator(G, val_img_dirs[i], dst_val, mask_src_dir=val_mask_dirs[i], mask_dst_dir=dst_val_mask, device=DEVICE, size=(224,224))
            harmonize_folder_with_generator(G, test_img_dirs[i], dst_test, mask_src_dir=test_mask_dirs[i], mask_dst_dir=dst_test_mask, device=DEVICE, size=(224,224))
        hm_train_dirs.append(dst_train); hm_train_mask_dirs.append(dst_train_mask)
        hm_val_dirs.append(dst_val); hm_val_mask_dirs.append(dst_val_mask)
        hm_test_dirs.append(dst_test); hm_test_mask_dirs.append(dst_test_mask)

    print("[HARM] Harmonization complete. Harmonized datasets written under:", hist_base)

    visuals_base = os.path.join(OUT_DIR, "HarmonizedSamples_CUT")
    for i in range(NUM_CLIENTS):
        cname = client_names[i]
        save_transformed_samples(hm_val_dirs[i], hm_val_mask_dirs[i], val_tf, cname, visuals_base, n_samples=7, prefix="harmonized")
        save_transformed_samples(hm_train_dirs[i], hm_train_mask_dirs[i], tr_tf, cname, visuals_base, n_samples=7, prefix="augmented")
        try:
            make_comparison_grid_and_histograms_updated_original_vs_hm(val_img_dirs[i], hm_val_dirs[i], cname, visuals_base)
        except Exception as e:
            print(f"[WARN] Could not make comparison grid for {cname}: {e}")

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
            train_loader = get_loader(hm_train_dirs[i], hm_train_mask_dirs[i], tr_tf, batch_size=8, shuffle=True, return_filename=True)
            val_loader = get_loader(hm_val_dirs[i], hm_val_mask_dirs[i], val_tf, batch_size=8, shuffle=False, return_filename=True)
            print(f"[Client {client_names[i]}] Local training")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")
            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz
        norm_weights = [w/total_sz for w in weights]
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))
        # evaluate global on each client's harmonized test set
        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(hm_test_dirs[i], hm_test_mask_dirs[i], val_tf, batch_size=8, shuffle=False, return_filename=True)
            print(f"[Client {client_names[i]}] Global Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)
            save_test_predictions(global_model, test_loader, client_names[i], out_base=OUT_DIR, round_num=(r+1), max_to_save=int(len(test_loader.dataset)), device_arg=DEVICE)
        round_metrics.append(rm)
        plot_metrics(round_metrics, OUT_DIR)
    print("Finished FedAvg on harmonized data.")

def make_comparison_grid_and_histograms_updated_original_vs_hm(original_dir, hm_dir, client_name, out_base, n_samples=7):
    base_dest = os.path.join(out_base, "ComparisonGrid", client_name)
    ensure_dir(base_dest); diffs_dest = os.path.join(base_dest, "diffs"); ensure_dir(diffs_dest)
    fnames = sorted([f for f in os.listdir(original_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])[:n_samples]
    if len(fnames) == 0: return
    top_imgs=[]; mid_imgs=[]; diff_imgs=[]; short_names=[]
    for fname in fnames:
        orig_p = os.path.join(original_dir, fname)
        hm_p = os.path.join(hm_dir, fname)
        if not os.path.exists(hm_p): continue
        orig = np.array(Image.open(orig_p).convert('RGB').resize((224,224)))
        hm = np.array(Image.open(hm_p).convert('RGB').resize((224,224)))
        diff = np.clip((np.abs(orig.astype(int) - hm.astype(int))*4),0,255).astype(np.uint8)
        top_imgs.append(orig); mid_imgs.append(hm); diff_imgs.append(diff); short_names.append(fname)
        save_image(orig, os.path.join(base_dest, f"orig_{fname}"))
        save_image(hm, os.path.join(base_dest, f"hm_{fname}"))
        save_image(diff, os.path.join(diffs_dest, f"diff_{fname}"))
    n=len(top_imgs)
    fig, axs = plt.subplots(3, n, figsize=(3*n,6))
    if n==1: axs=np.array([[axs[0]],[axs[1]],[axs[2]]])
    for i in range(n):
        axs[0,i].imshow(top_imgs[i]); axs[0,i].axis('off'); axs[0,i].set_title(short_names[i][:12])
        axs[1,i].imshow(mid_imgs[i]); axs[1,i].axis('off')
        axs[2,i].imshow(diff_imgs[i]); axs[2,i].axis('off')

    fig.suptitle(f"Harmonized (CUT) vs. Original: {client_name}", fontsize=16, y=0.98)
    fig.text(0.01, 0.82, "Original\nimages", fontsize=12, va='center', rotation='vertical')
    fig.text(0.01, 0.50, "Harmonized\nimages", fontsize=12, va='center', rotation='vertical')
    fig.text(0.01, 0.18, "Amplified\nDifference", fontsize=12, va='center', rotation='vertical')


    plt.tight_layout(); plt.savefig(os.path.join(base_dest, f"comparison_{client_name}.png")); plt.close()
    print(f"Saved comparison grid for {client_name} at {base_dest}")


def get_loader(img_dir, mask_dir, transform, batch_size=8, shuffle=True, return_filename=True):
    try:
        ds = CVCDataset(img_dir, mask_dir, transform=transform, return_filename=return_filename)
    except TypeError:
        ds = CVCDataset(img_dir, mask_dir, transform=transform)
        return_filename = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    main()
