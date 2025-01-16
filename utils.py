"""
Utilities for VGTO (Virtual Gown Try-On)
Refer to white paper for guidance.
"""

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import os


def resize(image: Image.Image, dimensions: tuple[int, int]) -> Image.Image:
    return image.resize(dimensions)


class CNN(nn.Module):
    def __init__(self):
        """
        Convolution Neural Network
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, 3, 1
        )  # 1 input channel, 32 conv_features, 1 kernel size 3
        self.conv2 = nn.Conv2d(
            32, 64, 3, 1
        )  # 32 input layers, 64 conv_features, 1 kernel size 3

        # Ensure adjacent pixels are 0 or active
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Full connect layer 1
        self.fc1 = nn.Linear(9216, 128)

        # Full connect layer 2 with 10 output labels
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: data
        x = self.conv1(x)

        # rectified-linear function activation
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # Max pool over x
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Flatten x to 1 dim
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class UNet(nn.Module):
    """U-Net Convolutional Neural Network"""

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
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

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Output
        return self.activation(self.final_conv(dec1))


class SegmentationDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_len = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_len)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, self.img_len[idx])
        mask_path = os.path.join(self.mask_dir, self.img_len[idx])

        image = resize(Image.open(img_path).convert("RGB"), (156, 56))
        mask = resize(Image.open(mask_path).convert("L"), (156, 56))
        image.save(img_path)
        mask.save(mask_path)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

        return image, mask
