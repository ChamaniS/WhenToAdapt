import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class EmbryoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index].replace(".jpg",".BMP")) #load 1 channel image
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg",".BMP")) #load 1 channel mask
        image = np.array(Image.open(img_path).convert("RGB")) #make 3 channel image
        mask = np.array(Image.open(mask_path).convert("L")) #converting masks in to grayscale

        #preprocessing for the masks
        mask[mask == 255] = 4  #ICM
        mask[mask == 192] = 3  #Blastocoel
        mask[mask == 242] = 3 # Blastocoel
        mask[mask == 226] = 3 # Blastocoel
        mask[mask == 128] = 2  #TE
        mask[mask == 64] = 1    #ZP
        mask[mask == 105] = 1 # ZP
        mask[mask == 10] = 0    #Background
        mask[mask == 0] = 0    #Background


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask



class HAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) #load 1 channel image
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg",".png")) #load 1 channel mask
        image = np.array(Image.open(img_path).convert("RGB")) #make 3 channel image
        mask = np.array(Image.open(mask_path).convert("L")) #converting masks in to grayscale

        #preprocessing for the masks
        #mask[mask == 0] = 0   #Background
        mask[mask < 100] = 0  # Background
        mask[mask > 100] = 1  # tumor



        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


class CVCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) #load 1 channel image
        mask_path = os.path.join(self.mask_dir, self.images[index]) #load 1 channel mask
        image = np.array(Image.open(img_path).convert("RGB")) #make 3 channel image
        mask = np.array(Image.open(mask_path).convert("L")) #converting masks in to grayscale

        #preprocessing for the masks
        #mask[mask == 0] = 0   #Background
        mask[mask < 100] = 0  # Background
        mask[mask > 100] = 1  # tumor
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

class CVCINDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]  # Only image files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)  # .jpg
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))  # .png

        image = np.array(Image.open(img_path).convert("RGB"))  # Convert image to 3 channels
        mask = np.array(Image.open(mask_path).convert("L"))  # Convert mask to grayscale (1 channel)

        # Preprocessing for the mask
        mask[mask < 100] = 0   # Background
        mask[mask >= 100] = 1  # Instrument or tumor

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class covidCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]  # Image filename
        img_path = os.path.join(self.image_dir, self.images[index]) #load 1 channel image
        mask_filename = img_filename.replace("images", "label") # Modify the mask name to match the new format
        mask_path = os.path.join(self.mask_dir, mask_filename)
        image = np.array(Image.open(img_path).convert("RGB")) #make 3 channel image
        mask = np.array(Image.open(mask_path).convert("L")) #converting masks in to grayscale
        #preprocessing for the masks
        #mask[mask == 0] = 0   #Background
        mask[mask < 100] = 0  # Background
        mask[mask > 100] = 1  # tumor
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


class FHPsAOPMSBDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_filename = img_filename.replace("images", "label")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Remapping mask values
        mask[mask == 0] = 0
        mask[mask == 127] = 1
        mask[mask == 255] = 2

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations["image"], augmentations["mask"]

        return image, mask
