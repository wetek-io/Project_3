from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
import time
import os

# Import U-Net and Dataset
from utils import unet

# Configuration
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "./SA1B_Meta_AI_Segmentation_Dataset"
SAVE_PATH = "./checkpoints/"

# Data Augmentation & Normalization
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load Dataset
train_dataset = unet.SegmentationDataset(DATASET_PATH)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
)

# Validation Dataset and Loader
val_dataset = unet.SegmentationDataset(
    DATASET_PATH.replace("train", "val")
)  # Adjust path if needed
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True
)

# Initiate Scaler
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Initiate Write
writer = SummaryWriter(log_dir="./logs")

# Initialize Model
model = unet.UNet(in_channels=3, out_channels=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Validation Function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# Training Loop with Validation and Checkpoint Naming
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_loss:.4f}")

    # Validation Loop
    val_loss = validate(model, val_loader, criterion)
    writer.add_scalar("Loss/val", val_loss, epoch)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save Checkpoint
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(SAVE_PATH, f"unet_epoch_{epoch + 1}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Update Learning Rate
    scheduler.step()

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

writer.close()
print("Training Complete!")
