import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import datetime
import sys
import time
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
# ----------------------
# 1. Dataset
# ----------------------
class CellDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, float(label)))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir,"all", rel_path)

        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)

        return img, label



def validation():
    model.eval()
    val_loss, total = 0, 0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # concat predictions and labels
    all_outputs = torch.cat(all_outputs).squeeze().numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    # average loss
    val_loss /= total
     # correlation metrics
    pearson_corr, _ = pearsonr(all_outputs, all_labels)
    r2 = r2_score(all_labels, all_outputs)

    return val_loss, pearson_corr, r2
# ----------------------
# 2. Transforms
# ----------------------
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # normalize grayscale
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # expand to 3 channels
])

val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

# ----------------------
# 3. Dataloaders
# ----------------------
root_dir = "/media/huifang/data/sennet/codex/cell_images"  # contains snc/ and non-snc/
train_dataset = CellDataset(os.path.join(root_dir, "regression_train.txt"), root_dir, train_transform)
val_dataset = CellDataset(os.path.join(root_dir, "regression_val.txt"), root_dir, val_transform)


train_loader = DataLoader(train_dataset, batch_size=320, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=320, shuffle=False, num_workers=16, pin_memory=True)

# ----------------------
# 1. Model: Swin-Tiny for regression
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "swin_tiny_patch4_window7_224",  # pretrained on ImageNet
    pretrained=True,
    num_classes=1   # regression head
)
model = model.to(device)

# ----------------------
# 2. Loss & Optimizer
# ----------------------
criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
logger = SummaryWriter(root_dir + "/logs_swin")

# ----------------------
# 3. Training loop (same as ResNet18)
# ----------------------
def train_model(num_epochs=10):
    prev_time = time.time()
    for epoch in range(num_epochs+1):
        model.train()
        train_loss, total = 0, 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            # progress logging
            batches_done = epoch * len(train_loader) + i
            batches_left = num_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\r" + "---[Epoch %d/%d] [Batch %d/%d] [Train loss: %f]   ETA: %s" %
                (epoch, num_epochs, i, len(train_loader), loss.item(),  time_left))
            # # --------------tensor board--------------------------------#
            if batches_done % 20 == 0:
                info = {'loss': loss.item(), 'train_loss': loss.item()}
                for tag, value in info.items():
                    logger.add_scalar(tag, value, batches_done)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), root_dir + '/regression_models/cell_regressor_%d.pth' % (epoch))
            val_loss, pearson_corr, r2 = validation()
            print("\nValidation Results:")
            print(f" Loss (MAE): {val_loss:.4f}")
            print(f" Pearson Correlation: {pearson_corr:.4f}")
            print(f" R² Score: {r2:.4f}")

# ----------------------
# Run training
# ----------------------
train_model(num_epochs=100)
