# cycle_gan_harmonize_and_fedavg_tb_with_visuals_nosave_cut.py
# CUT-based harmonization replacement for your CycleGAN flow.
# Based on user's original script, fixed sampling and comparison-grid format.
# Requirements: torch, torchvision, timm, PIL, numpy, matplotlib, sklearn, tqdm

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
# Basic FS helpers (unchanged)
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

# CUT-specific / training
CUT_EPOCHS = 30           # number of epochs for CUT training (lower for quick runs)
BATCH_CUT = 1
CUT_LAYERS_SAMPLE = 5     # number of layers from encoder to use (we extract up to 5)
PATCHNCE_SAMPLES = 256    # number of sampled patches per layer (S). Reduce if OOM (64-128)
PATCHNCE_TAU = 0.07
CUT_IDENTITY = True       # use identity loss on target domain (LPatchNCE)
CUT_ID_WEIGHT = 1.0       # lambdaY
CUT_X_WEIGHT = 1.0 if CUT_IDENTITY else 10.0  # lambdaX
CUT_LR = 2e-4
CUT_BETA = (0.5, 0.999)
CUT_WEIGHT_DECAY = 0.0

# other existing config (kept as in your original)
COMM_ROUNDS = 10
LOCAL_EPOCHS = 6
OUT_DIR = "./Outputs_TB_cut"
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
ARCH = "densenet169"
PRETRAINED = True
IMG_SIZE = 224
BATCH_SIZE = 8
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
# Image-list dataset for CUT (no temp folders)
# -------------------------
class ImageListDataset(Dataset):
    def __init__(self, files: List[str], size=(224,224), augment=False, validate=True):
        self.ds = ImageFolderSimple(folder=None, files_list=files, size=size, augment=augment, validate=validate)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx): return self.ds[idx]

# -------------------------
# Reusable small blocks (ResnetBlock kept from your cycle code)
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

# -------------------------
# Generator that exposes encoder features (split encoder/decoder)
# -------------------------
class CutResnetGenerator(nn.Module):
    """
    Resnet-style generator split into an encoder and decoder so we can extract
    multilayer patch features (as in CUT paper). Encoder returns feature maps
    at several depths.
    """
    def __init__(self, in_ch=3, out_ch=3, ngf=64, nblocks=6):
        super().__init__()
        # build encoder layers (we keep layers so we can pull intermediates)
        enc = []
        enc += [nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, ngf, 7, 1, 0, bias=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(True)]
        # two downsampling convs
        mult = 1
        for i in range(2):
            enc += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1, bias=False),
                    nn.InstanceNorm2d(ngf * mult * 2),
                    nn.ReLU(True)]
            mult *= 2
        # residual blocks
        res_blocks = []
        for i in range(nblocks):
            res_blocks += [ResnetBlock(ngf * mult)]
        # decoder (upsample)
        dec = []
        for i in range(2):
            dec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, 2, 1, output_padding=1, bias=False),
                    nn.InstanceNorm2d(int(ngf * mult / 2)),
                    nn.ReLU(True)]
            mult = int(mult / 2)
        dec += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7, 1, 0), nn.Tanh()]

        # register modules
        self.encoder_initial = nn.Sequential(*enc)       # initial conv + downsample
        self.res_blocks = nn.Sequential(*res_blocks)    # residual mid blocks
        self.decoder = nn.Sequential(*dec)              # upsample + final conv

        # initialize weights like cycle code
        self.apply(self._weights_init_normal)

    def _weights_init_normal(self, m):
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

    def forward(self, x):
        # convenience full forward
        feats = self.encode(x)
        last = feats[-1]
        out = self.decode(last)
        return out

    def encode(self, x):
        """
        Return a list of feature maps at different depths (from shallow->deep).
        We'll return up to 5 feature maps:
          feats[0] = input resolution features (we use the input itself)
          feats[1] = after initial conv + downsample
          feats[2] = after first resblock output
          feats[3] = after last resblock output (mid)
          feats[4] = the deepest feature (same as 3 here but kept for indexing)
        Each element is a tensor (B, C, H_l, W_l)
        """
        feats = []
        # pixel-level feature: include raw input (normalized to [-1,1] expected by decode)
        feats.append(x)  # pixel-level

        out = self.encoder_initial(x)
        feats.append(out)

        # run all res blocks but capture first and final outputs
        if len(self.res_blocks) > 0:
            # first
            first_block = self.res_blocks[0]
            out_first = first_block(out)
            # rest
            if len(self.res_blocks) > 1:
                rest_blocks = nn.Sequential(*list(self.res_blocks.children())[1:])
                out_rest = rest_blocks(out_first)
            else:
                out_rest = out_first
            feats.append(out_first)
            feats.append(out_rest)
        else:
            feats.append(out)
            feats.append(out)

        # ensure length == 5
        if len(feats) < 5:
            feats = (feats + [feats[-1]] * 5)[:5]
        return feats

    def decode(self, feat):
        return self.decoder(feat)

# -------------------------
# Discriminator (PatchGAN same as before - keep)
# -------------------------
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

# -------------------------
# PatchNCE loss implementation
# -------------------------
cross_entropy_loss = torch.nn.CrossEntropyLoss()

def PatchNCELoss(f_q, f_k, tau=PATCHNCE_TAU):
    """
    f_q: (B, D, S) query features
    f_k: (B, D, S) positive features (matching locations)
    Returns scalar loss averaged over batch.
    """
    B, D, S = f_q.shape
    # l_pos: (B, S)
    l_pos = (f_q * f_k).sum(dim=1)  # B x S
    # l_neg: compute dot between queries and all keys per sample
    # create B x S x S
    l_neg = torch.bmm(f_q.transpose(1,2), f_k)  # B x S x S
    # mask diagonal so matching positions are not negatives
    eye = torch.eye(S, device=f_q.device, dtype=torch.bool)[None, :, :]
    l_neg.masked_fill_(eye, -1e9)
    # logits: concat pos (as index 0) and negs -> (B, S, S+1)
    l_pos_unsq = l_pos.unsqueeze(2)  # B x S x 1
    logits = torch.cat([l_pos_unsq, l_neg], dim=2)  # B x S x (S+1)
    logits = logits / tau
    # flatten for cross entropy
    logits_flat = logits.reshape(-1, logits.shape[-1])  # (B*S) x (S+1)
    targets = torch.zeros(logits_flat.shape[0], dtype=torch.long, device=f_q.device)
    loss = cross_entropy_loss(logits_flat, targets)
    return loss

# small MLP projector used per layer (paper uses 2-layer MLP)
class ProjectorMLP(nn.Module):
    def __init__(self, in_ch, hidden=256, out_dim=256):
        super().__init__()
        # Use 1x1 convs as per CUT paper
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, kernel_size=1, bias=True)
        )
    def forward(self, x):
        # x: B x C x H x W -> out: B x D x H x W
        return self.net(x)

# -------------------------
# Utility: sample S patch locations and gather features into (B, D, S)
# Improved: can return indices, and also can gather using given indices so positives and queries match.
# -------------------------
def flatten_feat(feat):
    # input: B x C x H x W or B x C x HW
    if feat.dim() == 4:
        B,C,H,W = feat.shape
        feat_flat = feat.view(B, C, H*W)
        spatial = H*W
    elif feat.dim() == 3:
        B,C,spatial = feat.shape
        feat_flat = feat
    else:
        raise ValueError(f"Unexpected feat.dim()={feat.dim()}")
    return feat_flat, spatial

def sample_patch_indices(spatial, S, device):
    # returns a single list of indices for each batch element (list length B)
    # We'll sample independently for each batch element to match CUT paper randomness
    return [torch.randperm(spatial, device=device)[:S] for _ in range(1)]  # return list-of-1 then reuse for broadcast

def gather_patches_by_indices(feat, indices):
    """
    feat: B x C x spatial  (tensor)
    indices: list of length B where each element is 1D LongTensor of size S
    returns: B x C x S
    """
    B, C, spatial = feat.shape
    S = indices[0].shape[0]
    out_list = []
    for b in range(B):
        idx = indices[b]
        out_list.append(feat[b:b+1, :, idx])  # 1 x C x S
    out = torch.cat(out_list, dim=0)
    return out

def sample_patches_from_feature_map(feat, n_patches=PATCHNCE_SAMPLES, reuse_indices=None):
    """
    Accepts feature maps either as (B, C, H, W) or (B, C, spatial).
    Returns:
      - out: (B, D, S)
      - indices: list of length B with LongTensors of spatial indices
    If reuse_indices is provided (list of B index tensors), we use them instead of sampling new ones.
    """
    feat_flat, spatial = flatten_feat(feat)
    B = feat_flat.shape[0]
    device = feat_flat.device
    S = min(n_patches, spatial)
    # sample indices per sample
    if reuse_indices is None:
        indices = [torch.randperm(spatial, device=device)[:S] for _ in range(B)]
    else:
        indices = reuse_indices
    # gather
    out_list = []
    for b in range(B):
        idx = indices[b]
        out_list.append(feat_flat[b:b+1, :, idx])  # 1 x C x S
    out = torch.cat(out_list, dim=0)
    return out, indices

# -------------------------
# CUT trainer: train G (A->B) with PatchNCE and GAN
# -------------------------
def train_cut_from_lists(listA, listB, epochs=CUT_EPOCHS, device=DEVICE_TORCH):
    """
    Train CUT mapping listA -> listB (unpaired). Returns generator (CPU) trained
    to map images from domain A to domain B.
    """
    dsA = ImageListDataset(listA, size=(IMG_SIZE,IMG_SIZE), augment=True)
    dsB = ImageListDataset(listB, size=(IMG_SIZE,IMG_SIZE), augment=True)
    loaderA = DataLoader(dsA, batch_size=BATCH_CUT, shuffle=True, drop_last=True, num_workers=0)
    loaderB = DataLoader(dsB, batch_size=BATCH_CUT, shuffle=True, drop_last=True, num_workers=0)

    G_A2B = CutResnetGenerator().to(device)
    D_B = NLayerDiscriminator().to(device)

    projectors = {}  # layer_idx -> ProjectorMLP

    opt_G = optim.Adam(G_A2B.parameters(), lr=CUT_LR, betas=CUT_BETA, weight_decay=CUT_WEIGHT_DECAY)
    opt_D = optim.Adam(D_B.parameters(), lr=CUT_LR, betas=CUT_BETA, weight_decay=CUT_WEIGHT_DECAY)

    criterion_gan = nn.MSELoss().to(device)
    real_label = 1.0; fake_label = 0.0

    print(f"[CUT] Train in-memory lists A({len(listA)}) -> B({len(listB)}) for {epochs} epochs (one-sided CUT)")
    iterB = iter(loaderB)
    for epoch in range(epochs):
        loop = tqdm(loaderA, desc=f"CUT Epoch {epoch+1}/{epochs}")
        for real_A in loop:
            try:
                real_B = next(iterB)
            except StopIteration:
                iterB = iter(loaderB)
                real_B = next(iterB)
            real_A = real_A.to(device); real_B = real_B.to(device)

            # Train Discriminator D_B
            opt_D.zero_grad()
            pred_real = D_B(real_B)
            valid = torch.full_like(pred_real, real_label, device=device)
            loss_D_real = criterion_gan(pred_real, valid)
            with torch.no_grad():
                fake_B = G_A2B(real_A)
            pred_fake = D_B(fake_B.detach())
            fake_label_tensor = torch.full_like(pred_fake, fake_label, device=device)
            loss_D_fake = criterion_gan(pred_fake, fake_label_tensor)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # Train Generator G_A2B
            opt_G.zero_grad()
            fake_B = G_A2B(real_A)
            pred_fake_for_G = D_B(fake_B)
            valid_G = torch.full_like(pred_fake_for_G, real_label, device=device)
            loss_G_GAN = criterion_gan(pred_fake_for_G, valid_G)

            # Identity loss (LPatchNCE on real_B -> G(real_B))
            loss_id = 0.0
            if CUT_IDENTITY:
                with torch.no_grad():
                    fake_from_B = G_A2B(real_B)
                feats_B = G_A2B.encode(real_B)
                feats_GB = G_A2B.encode(fake_from_B)
                for l in range(CUT_LAYERS_SAMPLE):
                    if l >= len(feats_B): break
                    fB = feats_B[l]
                    fGB = feats_GB[l]
                    key = f"l{l}"
                    if key not in projectors:
                        projectors[key] = ProjectorMLP(in_ch=fB.shape[1]).to(device)
                    H = projectors[key]
                    pB = H(fB)
                    pGB = H(fGB)
                    # flatten spatial and normalize
                    pB_flat = pB.view(pB.shape[0], pB.shape[1], -1)
                    pGB_flat = pGB.view(pGB.shape[0], pGB.shape[1], -1)
                    pB_flat = nn.functional.normalize(pB_flat, dim=1)
                    pGB_flat = nn.functional.normalize(pGB_flat, dim=1)
                    # Sample indices once from pGB and use same indices for pB
                    fq, idxs = sample_patches_from_feature_map(pGB_flat, n_patches=PATCHNCE_SAMPLES)
                    # gather positives from pB_flat using same indices
                    # build positives by indexing pB_flat per-sample
                    Bbs, Cc, Sp = pB_flat.shape
                    # convert pB_flat to B x C x spatial if necessary (it already is)
                    # gather positives
                    fk_list = []
                    for b in range(Bbs):
                        fk_list.append(pB_flat[b:b+1,:, idxs[b]])
                    fk = torch.cat(fk_list, dim=0)
                    loss_id += PatchNCELoss(fq, fk, tau=PATCHNCE_TAU)
                loss_id = loss_id * CUT_ID_WEIGHT

            # PatchNCE between real_A and fake_B (main CUT loss)
            feats_A = G_A2B.encode(real_A)
            feats_fakeB = G_A2B.encode(fake_B)
            loss_patchnce = 0.0
            for l in range(CUT_LAYERS_SAMPLE):
                if l >= len(feats_A): break
                fA = feats_A[l]
                fFake = feats_fakeB[l]
                key = f"l{l}"
                if key not in projectors:
                    projectors[key] = ProjectorMLP(in_ch=fA.shape[1]).to(device)
                H = projectors[key]
                pA = H(fA)
                pFake = H(fFake)
                pA_flat = pA.view(pA.shape[0], pA.shape[1], -1)
                pFake_flat = pFake.view(pFake.shape[0], pFake.shape[1], -1)
                pA_flat = nn.functional.normalize(pA_flat, dim=1)
                pFake_flat = nn.functional.normalize(pFake_flat, dim=1)
                # sample indices from pFake_flat and use same indices on pA_flat
                fq, idxs = sample_patches_from_feature_map(pFake_flat, n_patches=PATCHNCE_SAMPLES)
                fk_list = []
                for b in range(fq.shape[0]):
                    fk_list.append(pA_flat[b:b+1,:, idxs[b]])
                fk = torch.cat(fk_list, dim=0)
                loss_patchnce += PatchNCELoss(fq, fk, tau=PATCHNCE_TAU)
            loss_patchnce = loss_patchnce * CUT_X_WEIGHT

            loss_G = loss_G_GAN + loss_patchnce + loss_id
            loss_G.backward()
            opt_G.step()

            loop.set_postfix({"loss_D": float(loss_D), "loss_G_GAN": float(loss_G_GAN), "loss_patchnce": float(loss_patchnce), "loss_id": float(loss_id)})

    return G_A2B.cpu()

# -------------------------
# Harmonize images for classification folder layout (unchanged, but uses single generator)
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
                    # generator expects input in [-1,1]
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
# Classification dataset & dataloader helpers (unchanged)
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
# Classification model & training helpers (unchanged)
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
# Comparison grid: original vs harmonized vs amplified difference (color) (improved)
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

def make_comparison_grid(original_val_dir, hm_val_dir, client_name, out_base, n_samples=7, amplification=8.0):
    """
    Create a grid with three rows:
      1) Original images (top)
      2) Harmonized images (middle)
      3) Amplified colorized difference (bottom)
    amplification: scalar multiplier applied to (hm - orig) to make differences visible.
    """
    base_dest = os.path.join(out_base, "ComparisonGrid")
    ensure_dir(base_dest)
    fn_pairs = select_val_pairs_for_comparison(original_val_dir, hm_val_dir, n_samples=n_samples)
    if len(fn_pairs) == 0:
        print(f"[VIS] no matching pairs for comparison for {client_name}")
        return
    orig_imgs = []
    hm_imgs = []
    titles = []
    for orig_p, hm_p, fn in fn_pairs:
        try:
            orig = np.array(Image.open(orig_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)/255.0
            hm = np.array(Image.open(hm_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)/255.0
            orig_imgs.append(orig)
            hm_imgs.append(hm)
            titles.append(fn)
        except Exception as e:
            print(f"[VIS] skipping pair {fn} due to read error: {e}")
    n = len(orig_imgs)
    if n == 0:
        print(f"[VIS] no readable pairs for {client_name}")
        return

    # create figure: 3 rows x n columns
    fig, axs = plt.subplots(3, n, figsize=(2.8*n, 3*3), constrained_layout=False)
    if n == 1:
        axs = np.array([[axs[0]],[axs[1]],[axs[2]]])

    for i in range(n):
        o = orig_imgs[i]
        h = hm_imgs[i]
        # original
        axs[0, i].imshow(o); axs[0, i].axis('off'); axs[0, i].set_title(titles[i][:15])
        # harmonized
        axs[1, i].imshow(h); axs[1, i].axis('off')
        # amplified difference: colorize
        diff = (h - o) * amplification
        # convert to a single-channel magnitude for color mapping: signed values preserved by mapping via diverging cmap
        # compute min/max for normalization (fix small epsilon)
        vmin = np.min(diff)
        vmax = np.max(diff)
        norm = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(-0.1, 0.1)
        # use diverging colormap to highlight positive/negative changes
        cmap = plt.cm.seismic
        colored = cmap(norm(diff[...,0]))  # use red-blue on first channel surrogate
        # to give multi-channel colorization we can combine channels: compute intensity as mean across RGB channels of diff
        # but for visual clarity we colorize the mean difference with the diverging cmap
        mean_diff = diff.mean(axis=2)
        colored = cmap(norm(mean_diff))
        # colored is RGBA; convert to RGB
        colored_rgb = (colored[..., :3] * 255).astype(np.uint8)
        axs[2, i].imshow(colored_rgb); axs[2, i].axis('off')

    fig.suptitle(f"Harmonized (CUT) vs Original: {client_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(base_dest, f"comparison_{client_name}.png")
    try:
        plt.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[VIS] Saved comparison grid for {client_name} at {out_png}")
    except Exception as e:
        print("[VIS] failed saving comparison grid:", e)
        plt.close(fig)

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
# Plot helpers (unchanged)
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
    plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Per-client Test Accuracy "); plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    try:
        plt.savefig(PER_CLIENT_TEST_ACC_FN); plt.close()
        print(f"[PLOT] Saved per-client test accuracy across rounds -> {PER_CLIENT_TEST_ACC_FN}")
    except Exception as e:
        print("[PLOT] failed saving per-client test accuracy plot:", e)
        plt.close()

# -------------------------
# Top-level flow (main) - mirrors your original script but calls CUT trainer
# -------------------------
def main():
    print("DEVICE:", DEVICE)
    cut_models = {}

    # 1) Build lists for each client; train CUT in-memory between lists (no temp dirs, no saving)
    client_paths = []
    for i, client_root in enumerate(CLIENT_ROOTS):
        pths = collect_all_image_paths_for_client(client_root)
        print(f"[DATA] client {CLIENT_NAMES[i]} has {len(pths)} images (train/val/test combined)")
        client_paths.append(pths)

    for i, client_root in enumerate(CLIENT_ROOTS):
        if i == reference_idx:
            print(f"[HARM] skipping CUT for reference client {CLIENT_NAMES[i]}")
            cut_models[i] = None
            continue
        listA = client_paths[i]
        listB = client_paths[reference_idx]
        if len(listA) == 0 or len(listB) == 0:
            print(f"[WARN] empty lists for CUT A({len(listA)}) B({len(listB)}). Skipping generator training; will copy images as-is.")
            cut_models[i] = None
            continue
        # train in memory
        G_A2B = train_cut_from_lists(listA, listB, epochs=CUT_EPOCHS, device=DEVICE_TORCH)
        cut_models[i] = G_A2B.cpu()
        # do not save generators to disk

    # 2) Harmonize each client (non-reference) using its generator; reference client copied as-is
    harmonized_base = os.path.join(OUT_DIR, "CUT_Harmonized")
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
            G = cut_models.get(i, None)
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
                print(f"[HARM] Harmonizing client {cname} -> {CLIENT_NAMES[reference_idx]} (CUT)")
                harmonize_client_with_generator(G, client_root, out_client_root, device=DEVICE_TORCH, size=(IMG_SIZE,IMG_SIZE))
        harmonized_roots.append(out_client_root)
    print("[HARM] Harmonization finished. Harmonized datasets at:", harmonized_base)

    # 3) Build dataloaders from harmonized roots and run FedAvg (unchanged)
    combined_loaders, combined_sizes, class_names, combined_train_ds, per_client_dataloaders, per_client_test_dsets = \
        make_multi_client_dataloaders_from_roots(harmonized_roots, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE=="cuda"))
    client_train_sizes = [len(per_client_dataloaders[i]['train'].dataset) for i in range(len(per_client_dataloaders)) if 'train' in per_client_dataloaders[i]]
    if sum(client_train_sizes) == 0: client_train_sizes = [1 for _ in range(len(per_client_dataloaders))]
    total_train = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
    print("client train sizes:", client_train_sizes)

    global_model = create_model(num_classes=NUM_CLASSES, arch=ARCH, pretrained=PRETRAINED).to(DEVICE_TORCH)
    print(f"Global model {ARCH} with {count_parameters(global_model):,} params")

    round_results = []
    # histories
    per_client_test_acc_history  = {i: [] for i in range(len(per_client_dataloaders))}
    per_client_perclass_history = {i: [] for i in range(len(per_client_dataloaders))}

    # 3a) create comparison grids using improved visuals
    visuals_out = os.path.join(OUT_DIR, "CUT_Harmonized_Visuals")
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

            acc_val = float(cl_metrics.get("accuracy", float('nan')))
            per_client_test_acc_history[i].append(acc_val)
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
    print(f"Harmonized datasets stored under: {os.path.join(OUT_DIR, 'CUT_Harmonized')}")
    print(f"Comparison grids stored under: {os.path.join(OUT_DIR, 'CUT_Harmonized_Visuals', 'ComparisonGrid')}")
    print("Global plots:")
    print(f" - {GLOBAL_TRAIN_ACC_FN}")
    print(f" - {GLOBAL_TRAIN_LOSS_FN}")
    print(f" - {GLOBAL_TEST_ACC_FN}")
    print(f" - {GLOBAL_TEST_LOSS_FN}")
    print("Per-client test accuracy (single PNG):")
    print(f" - {PER_CLIENT_TEST_ACC_FN}")

if __name__ == "__main__":
    main()
