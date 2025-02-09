import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Import U-Net and Dataset
from utils import UNet, SegmentationDataset

# Configuration
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "./dataset"
SAVE_PATH = "./checkpoints/unet.pth"

# Data Augmentation & Normalization
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load Dataset
train_dataset = SegmentationDataset(DATASET_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Model
model = UNet(in_channels=3, out_channels=2).to(
    DEVICE
)  # 2 output channels (user & gown)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

    # Save Model Checkpoint
    if (epoch + 1) % 5 == 0:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Checkpoint saved at {SAVE_PATH}")

print("Training Complete!")
