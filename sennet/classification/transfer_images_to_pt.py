import os
import torch
import numpy as np
from torchvision.io import read_image
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor, as_completed

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
"BRPC-24-08-7",
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
"2115-L"]

def build_pt_dataset(sample_dir, list_file, output_file, image_size=96):
    """
    Convert cropped cell PNGs into a .pt dataset with precomputed DeepScence labels.
    Saves a tuple: (images, labels, filenames, ds_scores)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5,), (0.5,)),  # normalize grayscale
    ])

    samples, filenames, ds_scores = [], [], []

    # 1. Read image list
    with open(list_file, "r") as f:
        for line in f:
            filename, ds_str = line.strip().split()
            ds_score = float(ds_str)

            img_path = os.path.join(sample_dir, filename)
            img = read_image(img_path)[:1, :, :]  # grayscale
            img = transform(img)

            samples.append(img)
            filenames.append(filename)
            ds_scores.append(ds_score)

    # 2. Stack into tensors
    images = torch.stack(samples)  # (N, 1, H, W)
    ds_scores = np.array(ds_scores)

    # 3. Save
    torch.save((images, filenames, ds_scores), output_file)

    return sample_dir, images.shape


# ---- Parallel execution ----
xenium_dir = "/media/huifang/data/sennet/xenium/deepscence/cell_images/all"

def process_sample(sample_id):
    sample_dir = os.path.join(xenium_dir, sample_id)
    list_file = os.path.join(sample_dir, "image_lists.txt")
    output_file = os.path.join(sample_dir, "dataset.pt")
    return build_pt_dataset(sample_dir, list_file, output_file, image_size=96)

# run with up to 16 workers
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(process_sample, sid): sid for sid in samples}
    for future in as_completed(futures):
        sample_dir, shape = future.result()
        print(f"Finished {sample_dir} -> {shape}")