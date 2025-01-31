"""
Utilities for VGTO (Virtual Gown Try-On)
Refer to white paper for guidance.
"""

from typing import Sequence
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tensorflow_datasets as tfds
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import json
import os

FeaturesDict = tfds.features.FeaturesDict
Scalar = tfds.features.Scalar
BBoxFeature = tfds.features.BBoxFeature
Tensor = tfds.features.Tensor
Image = tfds.features.Image

import tensorflow_datasets as tfds
import numpy as np

FeaturesDict = tfds.features.FeaturesDict(
    {
        "annotations": tfds.features.Sequence(
            {
                "area": tfds.features.Scalar(
                    dtype=np.uint64, description="The area in pixels of the mask."
                ),
                "bbox": tfds.features.BBoxFeature(
                    shape=(4,),
                    dtype=np.float32,
                    description="The box around the mask, in TFDS format.",
                ),
                "crop_box": tfds.features.BBoxFeature(
                    shape=(4,),
                    dtype=np.float32,
                    description="The crop of the image used to generate the mask, in TFDS format.",
                ),
                "id": tfds.features.Scalar(
                    dtype=np.uint64, description="Identifier for the annotation."
                ),
                "point_coords": tfds.features.Tensor(
                    shape=(1, 2),
                    dtype=np.float64,
                    description="The point coordinates input to the model to generate the mask.",
                ),
                "predicted_iou": tfds.features.Scalar(
                    dtype=np.float64,
                    description="The model's own prediction of the mask's quality.",
                ),
                "segmentation": tfds.features.FeaturesDict(
                    {
                        "counts": tfds.features.Text(),  # Corrected as a string field
                        "size": tfds.features.Tensor(shape=(2,), dtype=np.uint64),
                    }
                ),
                "stability_score": tfds.features.Scalar(
                    dtype=np.float64, description="A measure of the mask's quality."
                ),
            }
        ),
        "image": tfds.features.FeaturesDict(
            {
                "content": tfds.features.Image(
                    shape=(None, None, 3),
                    dtype=np.uint8,
                    description="Content of the image.",
                ),
                "file_name": tfds.features.Text(),  # Corrected as a string field
                "height": tfds.features.Scalar(
                    dtype=np.uint64, description="Height of the image."
                ),
                "image_id": tfds.features.Scalar(
                    dtype=np.uint64, description="Unique identifier for the image."
                ),
                "width": tfds.features.Scalar(
                    dtype=np.uint64, description="Width of the image."
                ),
            }
        ),
    }
)


class UNet(nn.Module):
    """U-Net Convolutional Neural Network"""

    def __init__(self, in_channels=3, out_channels=1):
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
        self.activation = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
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
        bottleneck = self.bottleneck(self.pool(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Output
        output = self.activation(self.final_conv(dec1))

        return output


class SegmentationDataset(Dataset):
    """
    Prepare data for segmentation
    Dataset: (images/, masks/)
    Return: image: tensor[], mask: tensor[]
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.masks = [f for f in os.listdir(self.img_dir) if f.endswith(".json")]
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

        original_size = image.size  # (width, height)
        image = self.transform(image)
        mask = TF.resize(Image.fromarray(mask), (original_size[1], original_size[0]))
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
