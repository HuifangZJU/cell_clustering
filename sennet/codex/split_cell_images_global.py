import os
import pickle
import numpy as np
from skimage.measure import regionprops
from skimage.io import imsave
from matplotlib import pyplot as plt
from skimage import exposure
import imageio.v2 as imageio
import scanpy as sc
import os, pickle, imageio, math, random
import scanpy as sc
from skimage import exposure
from collections import defaultdict




import math
def get_cell_meta_data(props):
    # compute sizes (area in pixels) for each cell
    cell_sizes = [p.area for p in props]

    # summary stats
    avg_size = np.mean(cell_sizes)
    median_size = np.median(cell_sizes)
    min_size = np.min(cell_sizes)
    max_size = np.max(cell_sizes)

    print(f"Number of cells: {len(cell_sizes)}")
    print(f"Average cell size (px): {avg_size:.2f}")
    print(f"Median cell size (px): {median_size:.2f}")
    print(f"Min cell size (px): {min_size}")

def crop_square_masked(image, mask, cell_id):
    """Crop a square patch around a cell (centered, mask applied, black background)."""
    coords = np.argwhere(mask == cell_id)
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)

    h = maxr - minr + 1
    w = maxc - minc + 1
    size = max(h, w)  # square size

    # center of the cell
    center_r = (minr + maxr) // 2
    center_c = (minc + maxc) // 2
    half = size // 2

    # square boundaries
    start_r = max(center_r - half, 0)
    end_r   = min(center_r + half + 1, image.shape[0])
    start_c = max(center_c - half, 0)
    end_c   = min(center_c + half + 1, image.shape[1])

    crop_img = image[start_r:end_r, start_c:end_c]
    crop_mask = (mask[start_r:end_r, start_c:end_c] == cell_id)

    # apply mask (black background)
    masked_crop = np.zeros_like(crop_img)
    masked_crop[crop_mask] = crop_img[crop_mask]

    return crop_img


def select_top_bottom_cells(cell_ids, marker_read, k=10):

    assert len(cell_ids) == len(marker_read), "cell_ids and marker_read must have the same length"
    n = len(cell_ids)
    n_select = math.ceil(n * k / 100)

    # Pair and sort by marker value
    paired = list(zip(cell_ids, marker_read))
    paired_sorted = sorted(paired, key=lambda x: x[1])

    # Bottom k% (negative)
    negative_cells = [cid for cid, _ in paired_sorted[:n_select]]
    # Top k% (positive)
    positive_cells = [cid for cid, _ in paired_sorted[-n_select:]]

    return positive_cells, negative_cells

import random
def select_top_and_random_negatives_global(global_cells, k=1, neg_ratio=3, seed=None):
    if seed is not None:
        random.seed(seed)

    n = len(global_cells)
    n_select = math.ceil(n * k / 100)

    # sort by marker value
    sorted_cells = sorted(global_cells, key=lambda x: x[3])

    # positives = top k%
    pos_cells = sorted_cells[-n_select:]

    # pool = everything else
    remaining = sorted_cells[:-n_select]

    # negatives = random sample with ratio
    n_neg = min(len(remaining), n_select * neg_ratio)
    neg_cells = random.sample(remaining, n_neg) if len(remaining) >= n_neg else remaining

    return pos_cells, neg_cells


def get_global_pos_neg_list():
    # ---------- Step 1: Collect global cells ----------
    global_cells = []  # (sample, region, cell_id, marker_val)

    for sample_id, regions in data_groups.items():
        for region_id in regions:
            adata = sc.read_h5ad(os.path.join(root_dir, "regional_data", f"{sample_id}_{region_id}.h5ad"))
            for cid, val in zip(adata.obs['label'].tolist(), adata.obs['p16'].tolist()):
                global_cells.append((sample_id, region_id, cid, val))
            del adata

    # ---------- Step 2: Select global pos/neg ----------
    pos_cells, neg_cells = select_top_and_random_negatives_global(global_cells, k=1, seed=42)
    # ---------- Step 3: Group by (sample_id, region_id) ----------
    pos_dict = defaultdict(list)
    neg_dict = defaultdict(list)

    for s, r, cid, _ in pos_cells:
        pos_dict[(s, r)].append(cid)

    for s, r, cid, _ in neg_cells:
        neg_dict[(s, r)].append(cid)

    return pos_dict,neg_dict




root_dir = "/media/huifang/data/sennet/codex/"  # adjust this path if needed
pos_out_dir ="/media/huifang/data/sennet/codex/cell_images/snc/"
neg_out_dir="/media/huifang/data/sennet/codex/cell_images/non-snc/"

# --- Channel to crop ---
channel = "DAPI"
topk=1
# sample_visualization()
data_groups={
    'S1':['reg005','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010'],
'S2':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010','reg0011','reg0012'],
'S3':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010','reg0011','reg0012','reg0013','reg0014'],
'S4':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008'],
'marchsample_S1':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010'],
'marchsample_S2':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010','reg0011','reg0012'],
             }

pos_dict, neg_dict = get_global_pos_neg_list()


# ---------- Step 4: Iterate per sample/region ----------
for sample_id, regions in data_groups.items():
    for region_id in regions:
        with open(os.path.join(root_dir, sample_id, f"{region_id}.pickle"), "rb") as fh:
            data = pickle.load(fh)

        masks = data["masks"]
        img = data["image_dict"][channel]

        print(f"Begin saving images of {sample_id}_{region_id}...")

        # positives
        for cid in pos_dict.get((sample_id, region_id), []):
            crop = crop_square_masked(img, masks, cid)
            crop_adj = exposure.rescale_intensity(crop, out_range=(0, 255)).astype(np.uint8)
            imageio.imwrite(os.path.join(pos_out_dir, f"{sample_id}_{region_id}_{cid}.png"), crop_adj)

        # negatives
        for cid in neg_dict.get((sample_id, region_id), []):
            crop = crop_square_masked(img, masks, cid)
            crop_adj = exposure.rescale_intensity(crop, out_range=(0, 255)).astype(np.uint8)
            imageio.imwrite(os.path.join(neg_out_dir, f"{sample_id}_{region_id}_{cid}.png"), crop_adj)

        print(f" -> Saved {len(pos_dict.get((sample_id, region_id), [])) + len(neg_dict.get((sample_id, region_id), []))} cells\n")






