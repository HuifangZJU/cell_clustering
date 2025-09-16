import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import models
import pandas as pd


samples = [
"1720-L",
"1881-L",
"1881-R",
"2049-L",
"BRPC-23-268-L",
"1995-R",
"2196-R",
"BRPC-23-495-R",
"1812-L",
"1851-L",
"1851-R",
"1921-L",
"1889-R",
"1913-L",
"1913-R",
"BRPC-23-378-R",
"1850-R",
"1885-L",
"1885-R",
"2016-L",
"BRPC-23-563-L",
"1850-L",
"1904-L",
"1904-R",
"2021-R",
"2022-L",
"BRPC-24-001-R",
"2022-R",
"2115-L"
]

# ----------------------
# 1. Dataset for test
# ----------------------
class DSCellDataset(Dataset):
    def __init__(self, list_file, root_dir, transform=None):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                filename, ds_str = line.strip().split()
                ds_score = float(ds_str)
                self.samples.append((filename, ds_score))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, ds_score = self.samples[idx]
        img_path = os.path.join(self.root_dir, filename)
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        cellid = filename.split('.')[0]
        return img, ds_score, cellid
# ----------------------
# 3. Evaluation
# ----------------------
def evaluate_model_on_dslist(model, list_file, sample_dir, device):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # grayscale → 3 channel
    ])

    dataset = DSCellDataset(list_file, sample_dir, transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    all_cellids, all_preds, all_ds_scores = [], [], []

    sample_ds_scores = np.array([s[1] for s in dataset.samples])
    if len(sample_ds_scores)==0:
        return [],[]

    with torch.no_grad():
        for imgs, ds_scores, cellids in tqdm(loader, desc="Evaluating", total=len(loader)):
            imgs = imgs.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(imgs))

            all_preds.extend(outputs.squeeze().tolist())
            all_cellids.extend(cellids)
            all_ds_scores.extend(ds_scores.tolist())
    return all_preds,all_cellids,all_ds_scores



def run_single_sample(sample_id):
    # Evaluate vs DS-threshold labels
    data_root_dir = "/media/huifang/data/sennet/xenium/deepscence/cell_images/all"
    sample_dir = os.path.join(data_root_dir, sample_id)
    list_file = os.path.join(sample_dir, "image_lists.txt")
    preds,cells,dscores = evaluate_model_on_dslist(model, list_file, sample_dir, device)
    return preds,cells,dscores

# ----------------------
# 4. Usage
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_root = "/media/huifang/data/sennet/codex/cell_images/models/resnet"
model_name='top1-models-equal-neg'
# Load trained model

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
state_dict = torch.load(os.path.join(model_root, model_name, "cell_classifier_100.pth"), map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)


out_file = f"/media/huifang/data/sennet/codex/cell_images/results/{model_name}_predictions.csv"
# Always overwrite at the start (create new file with header)
pd.DataFrame(columns=["sample_id", "cellid","ds","pred"]).to_csv(out_file, index=False)

all_preds_global = []
all_cells_global = []
all_ds_global=[]

for sample_id in samples:
    print(sample_id)
    preds,cells,dscores = run_single_sample(sample_id)

    # Store global arrays (for later overall metrics)
    all_preds_global.extend(preds)
    all_cells_global.extend(cells)
    all_ds_global.append(dscores)
    # Save sample-wise results immediately
    df = pd.DataFrame({
        "sample_id": [sample_id] * len(preds),
        "cellids": cells,
        "ds": dscores,
        "pred": preds
    })
    df.to_csv(out_file, mode="a", header=False, index=False)   # append to file




