import cv2
import numpy as np
import os
import json
from PIL import Image
import torch
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam


class WeddingGownDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".json")])
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Larger resolution for better segmentation
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        with open(mask_path) as f:
            mask_data = json.load(f)
        mask = self.decode_rle(
            mask_data["annotations"][0]["segmentation"]
        )  # Adjust as needed

        # Resize mask to match image size
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=(256, 256)
        ).squeeze()

        return img, mask

    def decode_rle(self, rle):
        """Decode RLE to a binary mask."""
        from pycocotools import mask as mask_utils

        binary_mask = mask_utils.decode(rle)
        return binary_mask


def blend_images(bride_image, gown_mask, gown_image):
    """
    Overlay the gown mask onto the bride image using the gown image.
    - bride_image: numpy array of the bride's image (H, W, 3)
    - gown_mask: binary numpy array of the gown segmentation mask (H, W)
    - gown_image: numpy array of the gown image (H, W, 3)
    """
    # Ensure mask is binary
    gown_mask = (gown_mask > 0.5).astype(np.uint8)

    # Resize gown image to match bride image
    gown_image = cv2.resize(gown_image, (bride_image.shape[1], bride_image.shape[0]))

    # Blend images
    blended = np.where(gown_mask[:, :, None], gown_image, bride_image)

    return blended
