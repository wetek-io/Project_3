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

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class UNet(nn.Module):
    """U-Net Convolutional Neural Network"""

    """
        While encoding U-Net reduces the size of spacial dimensioning while still extracting
        high level features. Then runs a convolution over those features before decoding them.
        This model implements 'skip-connection' this allows the model to match the spacial data
        with the high level features from the decoder. This implementation is what makes this a
        true U-Net model.
    """

    def __init__(self, in_channels=3, out_channels=1, debug=False):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()  # Sigmoid for pixel-wise probabilities

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Input validation
        if self.debug:
            print(
                f"Expected 4D input (batch, channels, height, width) but got {x.shape}"
            )
            print(f"Expected 3 channels for input images but got {x.size(1)}")

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))

        # Convolution
        bottleneck = self.conv_block(self.pool(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Output
        output = self.activation(self.final_conv(dec1))

        # Output validation
        if self.debug:
            print(f"Expected output shape {x.shape} but got {output.shape}")

        return output


class SegmentationDataset(Dataset):
    """
    Prepare data for segmentation
    Dataset: (images/, masks/)
    Return: image: tensor[], mask: tensor[]
    """

    def __init__(self, img_dir, transform=TRANSFORM, debug=False):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.masks = [f for f in os.listdir(self.img_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        mask_name = self.masks[idx]
        mask_path = os.path.join(self.img_dir, mask_name)
        with open(mask_path, "r") as f:
            mask_data = json.load(f)
        mask = self._create_mask(mask_data, image.size)

        if self.transform:
            image = self.transform(image)
            mask = TF.resize(Image.fromarray(mask), image.size)
            mask = torch.from_numpy(mask).float()

        # Debugging checks
        if self.debug:
            print(f"Image and mask sizes do not match: {image.shape} vs {mask.shape}")
            print(f"Expected image dtype torch.float32 but got {image.dtype}")
            print(f"Expected mask dtype torch.float32 but got {mask.dtype}")

        # Turn image into tensor data
        return image, mask

    def _create_mask(self, data, size):
        w, h = size
        mask = np.zeros((w, h), dtype=np.uint8)

        if "annotations" not in data or not data["annotations"]:
            print(f"Warning: No annotations found in data: {data}")
            return mask

        for annotation in data["annotations"]:
            if "segmentation" in annotation:
                segmentation = annotation
                if isinstance(segmentation, list):
                    for poly in segmentation:
                        poly = np.array(poly).reshape(-1, 2)
                        mask = cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
