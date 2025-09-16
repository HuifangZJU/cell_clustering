import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import models
import pandas as pd
from scipy.stats import pearsonr, spearmanr
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
        return img, ds_score, filename


# ----------------------
# 2. Compute ds labels
# ----------------------
def compute_ds_labels(ds_scores, topk_percent=10):
    """
    Convert ds scores to binary labels:
    1 = snc (top-k% highest ds scores)
    0 = non-snc (others)
    """
    threshold = np.percentile(ds_scores, 100 - topk_percent)
    return (ds_scores >= threshold).astype(int), threshold

def plot_cm_matrix(acc,cm,suffix):
    # Normalized confusion matrix (row-wise)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Build custom annotation strings
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_normalized[i, j] * 100:.1f}%\n({cm[i, j]})"

    print(f"\nAgreement between CODEX-p16 predictions and DeepScence labels (Sample {suffix}):")
    print(f"Overall Accuracy: {acc:.4f}")


    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Pred snc (CODEX-p16)", "Pred non-snc (CODEX-p16)"],
                yticklabels=["True snc (DeepScence)", "True non-snc (DeepScence)"],
                cbar_kws={"label": "Proportion (row-normalized)"})

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.title(f"Confusion Matrix: CODEX-p16 vs DeepScence (Sample {suffix})\nAccuracy={acc:.4f}")
    plt.xlabel("CODEX-p16 Prediction")
    plt.ylabel("DeepScence Label")
    plt.tight_layout()
    outdir = "/media/huifang/data1/sennet/codex/cell_images/results"
    out_path = os.path.join(outdir, f"{suffix}_confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ----------------------
# 3. Evaluation
# ----------------------
def evaluate_model_on_dslist(model, list_file, sample_dir, device,sample_id, topk_percent=1):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # grayscale → 3 channel
    ])

    dataset = DSCellDataset(list_file, sample_dir, transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    all_preds, all_labels, all_filenames = [], [], []

    sample_ds_scores = np.array([s[1] for s in dataset.samples])
    if len(sample_ds_scores)==0:
        return [],[]
    print('computing ds labels')
    ds_labels, ds_threshold = compute_ds_labels(sample_ds_scores, topk_percent)
    print('done')
    with torch.no_grad():
        for imgs, ds_scores, filenames in tqdm(loader, desc="Evaluating", total=len(loader)):
            imgs = imgs.to(device, non_blocking=True)
            outputs = torch.sigmoid(model(imgs))
            preds = (outputs> 0.9).int().cpu().numpy()
            labels = (ds_scores.numpy() >= ds_threshold).astype(int)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.squeeze().tolist())
            all_filenames.extend(filenames)



    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds, labels=[1, 0])

    plot_cm_matrix(acc,cm,sample_id)

    return np.array(all_preds),np.array(all_labels),acc



def run_single_sample(sample_id):
    # Evaluate vs DS-threshold labels
    sample_dir = os.path.join(xenium_root_dir, sample_id)
    list_file = os.path.join(sample_dir, "image_lists.txt")
    preds,labels,acc = evaluate_model_on_dslist(model, list_file, sample_dir, device, sample_id, topk_percent=1)
    return preds,labels,acc

# ----------------------
# 4. Usage
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

codex_root_dir = "/media/huifang/data1/sennet/codex/cell_images"
xenium_root_dir= "/media/huifang/data1/sennet/xenium/deepscence/cell_images/all"
# Load trained model

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
state_dict = torch.load(os.path.join(codex_root_dir, "models/top1", "cell_classifier_60.pth"), map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
samples = [
# "BRPC-24-08-7", have no ds reads?
"1720-L",
# "1881-L",
# "1881-R",
# "2049-L",
# "BRPC-23-268-L",
# "1995-R",
# "2196-R",
# "BRPC-23-495-R",
# "1812-L",
# "1851-L",
# "1851-R",
# "1921-L",
# "1889-R",
# "1913-L",
# "1913-R",
# "BRPC-23-378-R",
# "1850-R",
# "1885-L",
# "1885-R",
# "2016-L",
# "BRPC-23-563-L",
# "1850-L",
# "1904-L",
# "1904-R",
# "2021-R",
# "2022-L",
# "BRPC-24-001-R",
# "2022-R",
# "2115-L"
]

out_file = "/media/huifang/data1/sennet/codex/cell_images/results/sample_predictions.csv"


# 🔹 Always overwrite at the start (create new file with header)
pd.DataFrame(columns=["sample_id", "pred", "label"]).to_csv(out_file, index=False)

all_preds_global = []
all_labels_global = []
all_acc=[]

for sample_id in samples:
    preds, labels,acc = run_single_sample(sample_id)

    # Store global arrays (for later overall metrics)
    all_preds_global.extend(preds.tolist())
    all_labels_global.extend(labels.tolist())
    all_acc.append(acc)

    # Save sample-wise results immediately
    df = pd.DataFrame({
        "sample_id": [sample_id] * len(preds),
        "pred": preds.tolist(),
        "label": labels.tolist()
    })
    df.to_csv(out_file, mode="a", header=False, index=False)   # append to file

all_preds_global = np.array(all_preds_global)
all_labels_global = np.array(all_labels_global)
all_acc = np.array(all_acc)
print(all_acc)

overall_acc = (all_preds_global == all_labels_global).mean()
overall_cm = confusion_matrix(all_labels_global, all_preds_global, labels=[1, 0])

plot_cm_matrix(overall_acc,overall_cm,'overeall')

print("\n=== Overall Results ===")
print(f"Overall Accuracy: {overall_acc:.4f}")
print("Overall Confusion Matrix:\n", overall_cm)
print("\nDetailed Report:\n",
      classification_report(all_labels_global, all_preds_global, labels=[1, 0],
                            target_names=["snc (DS)", "non-snc (DS)"]))


