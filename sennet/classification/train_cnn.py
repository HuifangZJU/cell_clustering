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
# ----------------------
# 1. Dataset
# ----------------------
class CellDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)

        return img, label

def validation():
    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    return val_acc
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
root_dir = "/media/huifang/data1/sennet/codex/cell_images"  # contains snc/ and non-snc/
train_dataset = CellDataset(os.path.join(root_dir, "train.txt"), root_dir, train_transform)
val_dataset = CellDataset(os.path.join(root_dir, "val.txt"), root_dir, val_transform)


train_loader = DataLoader(train_dataset, batch_size=320, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=320, shuffle=False, num_workers=16, pin_memory=True)

# ----------------------
# 4. Model
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: ResNet18 (modern backbones can be swapped in)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification
model = model.to(device)

# ----------------------
# 5. Loss & Optimizer
# ----------------------
criterion = nn.BCEWithLogitsLoss()  # since output is single logit
optimizer = optim.Adam(model.parameters(), lr=1e-4)
logger = SummaryWriter(root_dir+"/logs")
# ----------------------
# 6. Training loop
# ----------------------
def train_model(num_epochs=10):
    prev_time = time.time()
    for epoch in range(num_epochs+1):
        # ---- Training ----
        model.train()
        train_loss, correct, total = 0, 0, 0
        for i, (imgs, labels) in enumerate(train_loader):

            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

            train_acc = correct / total
            train_loss /= total

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
                info = {'loss': loss.item(), 'train_loss': loss.item(),'train_acc': train_acc}
                for tag, value in info.items():
                    logger.add_scalar(tag, value, batches_done)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), root_dir + '/models/cell_classifier_%d.pth' % (epoch))
            val_acc = validation()
            print('\nvalidation accuracy : ' )
            print(val_acc)

# ----------------------
# Run training
# ----------------------
train_model(num_epochs=100)
