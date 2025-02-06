"""
Utilities for VGTO (Virtual Gown Try-On)
Refer to white paper for guidance.
"""

from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import json
import cv2
import os


class UNet(nn.Module):
    """U-Net Convolutional Neural Network"""

    """
        While encoding U-Net reduces the size of spacial dimensioning while still extracting
        high level features. Then runs a convolution over those features before decoding them.
        This model implements 'skip-connection' this allows the model to match the spacial data
        with the high level features from the decoder. This implementation is what makes this a
        true U-Net model.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.bottleneck(in_channels, 64)
        self.enc2 = self.bottleneck(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution
        self.conv_btlnk = self.bottleneck(128, 256)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.bottleneck(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.bottleneck(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()  # Sigmoid for pixel-wise probabilities

    def bottleneck(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))

        # Convolution
        bottleneck = self.conv_btlnk(self.pool(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final_conv(dec1)


class SegmentationDataset(Dataset):
    """
    Prepare data for segmentation.
    Dataset contains:
    - images/ (JPG files)
    - masks/ (JSON annotations)
    Returns:
    - image: torch.Tensor
    - mask: torch.Tensor
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
        self.masks = sorted(
            [f for f in os.listdir(self.img_dir) if f.endswith(".json")]
        )
        # Ensure matching images and masks
        if len(self.imgs) != len(self.masks):
            raise ValueError("Mismatch between images and masks in dataset!")
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        mask_name = self.masks[idx]
        mask_path = os.path.join(self.img_dir, mask_name)
        with open(mask_path, "r") as f:
            mask = self._create_mask(json.load(f), image.size)

        # Apply image transformations
        image = self.transform(image)

        # Ensure mask is a NumPy array with correct dtype
        mask = np.array(mask, dtype=np.uint8)

        # Resize the mask correctly
        w, h = image.shape[1], image.shape[2]  # Image tensor is (C, H, W)
        mask = TF.resize(Image.fromarray(mask), (h, w))  # Resize with (H, W)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

    def _create_mask(self, data, size):
        """Creates a binary mask from JSON annotations."""
        w, h = size  # Correct the shape order
        mask = np.zeros((h, w), dtype=np.uint8)
        if not data.get("annotations"):
            print(f"Warning: No annotations found in the JSON file.")
            return mask
        for annotation in data["annotations"]:
            bbox = annotation.get("bbox", None)
            segmentation = annotation.get("segmentation", None)
            predicted_iou = annotation.get("predicted_iou", None)
            if bbox:
                print(f"Bounding Box: {bbox}")
            if segmentation:
                print(
                    f"Segmentation RLE: {segmentation['counts'][:50]}..."
                )  # Partial view
            if predicted_iou:
                print(f"Predicted IoU: {predicted_iou}")
        return mask
