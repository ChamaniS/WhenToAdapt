import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import glob

'''
class CVCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Accept all common image formats
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        # get base name (without extension)
        base_name = os.path.splitext(img_name)[0]

        # find corresponding mask with any extension
        mask_candidates = glob.glob(os.path.join(self.mask_dir, base_name + ".*"))
        if len(mask_candidates) == 0:
            raise FileNotFoundError(f"No mask found for {img_name} in {self.mask_dir}")
        mask_path = mask_candidates[0]  # take first match

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Preprocess mask (binary segmentation)
        mask = np.array(mask)
        mask[mask < 100] = 0
        mask[mask >= 100] = 1
        mask = Image.fromarray(mask.astype(np.uint8))

        # Apply transforms separately
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask
'''
class CVCDataset(Dataset):
    """
    Robust dataset for CVC / polyp-style datasets.

    - Matches image and mask files by basename (intersection), so different extensions / extra files are tolerated.
    - If `transform` is an albumentations Compose with ToTensorV2, it will be applied to BOTH image and mask.
    - If `transform` is None, dataset will resize using PIL:
        - images: bilinear resize -> converted to float tensor in [0,1] CxHxW
        - masks: nearest resize -> binary (0/1) float tensor shape 1xHxW

    Parameters:
        image_dir, mask_dir: directories
        transform: albumentations.Compose or torchvision transform or None
        target_size: (width, height) target if transform is None
        mask_threshold: grayscale threshold to binarize masks (0..255)
        return_filename: if True, __getitem__ returns (image, mask, filename)
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        target_size: Tuple[int, int] = (224, 224),
        mask_threshold: int = 100,
        return_filename: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size  # (W, H) for PIL resize
        self.mask_threshold = mask_threshold
        self.return_filename = return_filename

        # collect candidate files (sorted for deterministic ordering)
        imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
        masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])

        # map basenames -> filenames (keep first occurrence if multiple extensions)
        img_map = {os.path.splitext(f)[0]: f for f in imgs}
        mask_map = {os.path.splitext(f)[0]: f for f in masks}

        # intersection of basenames -> ensure pairs (sorted to be deterministic)
        common = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
        if len(common) == 0:
            # fallback: use all image basenames and warn if masks are missing
            common = sorted(list(img_map.keys()))
            missing = [k for k in common if k not in mask_map]
            if missing:
                print(
                    f"[CVCDataset WARNING] {len(missing)} images appear missing masks "
                    f"(first missing: {missing[0]}). Dataset may be misaligned."
                )

        # final lists (sorted by common basenames)
        self.common_basenames = common
        self.images = [img_map[k] for k in self.common_basenames]
        # mask may be None if not present; we keep None but handle later
        self.masks = [mask_map.get(k, None) for k in self.common_basenames]

    def __len__(self):
        return len(self.images)

    def _load_image(self, path: str) -> np.ndarray:
        # return HxWx3 uint8 RGB
        pil = Image.open(path).convert("RGB")
        return np.array(pil)

    def _load_mask(self, path: str) -> np.ndarray:
        # return HxW uint8 grayscale
        pil = Image.open(path).convert("L")
        return np.array(pil)

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        # thresholding: returns 0 or 1 uint8
        m = np.array(mask, copy=True)
        m[m < self.mask_threshold] = 0
        m[m >= self.mask_threshold] = 1
        return m.astype(np.uint8)

    def __getitem__(self, index: int):
        img_name = self.images[index]
        mask_name = self.masks[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name) if mask_name is not None else None

        # load raw arrays
        image = self._load_image(img_path)  # HxWx3 uint8
        if mask_path is not None and os.path.exists(mask_path):
            mask = self._load_mask(mask_path)  # HxW uint8
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # binarize mask to 0/1
        mask = self._binarize_mask(mask)

        # apply transform
        if self.transform is not None:
            # If transform is albumentations (expects numpy and returns dict)
            try:
                augmented = self.transform(image=image, mask=mask)
                image_t = augmented["image"]
                mask_t = augmented["mask"]
            except Exception:
                # Try torchvision-style transform: accept PIL Image and return Tensor
                pil_img = Image.fromarray(image)
                pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
                image_t = self.transform(pil_img)  # user-provided transform should convert to tensor
                # convert mask to tensor manually
                mask_t = TF.to_tensor(pil_mask).squeeze(0)  # HxW in [0,1]
                mask_t = (mask_t > 0.5).float()
        else:
            # deterministic fallback: resize with PIL and convert to tensors
            pil_img = Image.fromarray(image).resize(self.target_size, resample=Image.BILINEAR)
            image_t = TF.to_tensor(pil_img).float()  # CxHxW in [0,1]

            pil_mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(self.target_size, resample=Image.NEAREST)
            mask_t = TF.to_tensor(pil_mask).squeeze(0)  # HxW in [0,1]
            mask_t = (mask_t > 0.5).float()

        # If albumentations ToTensorV2 was used, image_t/mask_t are already torch tensors.
        # Normalize shapes/types:
        if isinstance(mask_t, np.ndarray):
            mask_t = torch.from_numpy(mask_t).float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)  # 1xHxW
        elif mask_t.ndim == 3 and mask_t.shape[0] != 1:
            mask_t = mask_t[0:1, ...].float()

        if isinstance(image_t, np.ndarray):
            # numpy HxWxC -> CxHxW and scale to [0,1]
            image_t = torch.from_numpy(image_t.transpose(2, 0, 1)).float() / 255.0

        image_t = image_t.float()
        mask_t = (mask_t > 0.5).float()

        if self.return_filename:
            return image_t, mask_t, img_name
        else:
            return image_t, mask_t