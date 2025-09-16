import os
import pickle
import numpy as np
from skimage.measure import regionprops
from skimage.io import imsave
from matplotlib import pyplot as plt
from skimage import exposure
import imageio.v2 as imageio
import scanpy as sc


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


def select_top_and_random_negatives(cell_ids, marker_read, k=10, seed=None):
    import random
    assert len(cell_ids) == len(marker_read), "cell_ids and marker_read must have the same length"

    if seed is not None:
        random.seed(seed)

    n = len(cell_ids)
    n_select = math.ceil(n * k / 100)

    # Pair and sort by marker value
    paired = list(zip(cell_ids, marker_read))
    paired_sorted = sorted(paired, key=lambda x: x[1])

    # Top k% as positive
    positive_cells = [cid for cid, _ in paired_sorted[-n_select:]]

    # Remaining pool for negatives
    remaining_cells = [cid for cid, _ in paired_sorted[:-n_select]]

    # Randomly sample negatives
    negative_cells = random.sample(remaining_cells, n_select) if len(remaining_cells) >= n_select else remaining_cells

    return positive_cells, negative_cells

def sample_visualization():
    # Replace with your file path
    file_path = "/media/huifang/data1/sennet/codex/20250314_Yang_SenNet_S1/seg_output_sennet_20250412_S1_reg0010.pickle"
    adata = sc.read_h5ad("/media/huifang/data1/sennet/codex/regional_data/S1_reg0010.h5ad")

    cell_ids = adata.obs['label'].tolist()
    marker_read = adata.obs['p16'].tolist()
    pos_cells, neg_cells = select_top_bottom_cells(cell_ids,marker_read,1)
    print(len(pos_cells))

    with open(file_path, "rb") as f:  # "rb" = read in binary mode
        data = pickle.load(f)


    masks = data["masks"]  # segmentation mask
    dapi_img = data["image_dict"]["DAPI"]  # DAPI channel
    # p16_channel = data["image_dict"]["p16"]


    # props = regionprops(masks)
    # props = neg_cells
    props = pos_cells
    # --- Visualize a batch of cells ---
    batch_size = 12
    for start in range(0, len(props), batch_size):
        end = min(start + batch_size, len(props))
        batch = props[start:end]

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()

        for i, prop in enumerate(batch):
            # cell_id = prop.label
            cell_id = prop
            crop = crop_square_masked(dapi_img, masks, cell_id)

            axes[i].imshow(crop, cmap="gray")
            axes[i].set_title(f"Cell {start + i + 1}")
            axes[i].axis("off")

        # hide empty subplots
        for j in range(len(batch), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

root_dir = "/media/huifang/data1/sennet/codex/"  # adjust this path if needed
pos_out_dir ="/media/huifang/data1/sennet/codex/cell_images/snc/"
neg_out_dir="/media/huifang/data1/sennet/codex/cell_images/non-snc/"

# --- Channel to crop ---
channel = "DAPI"
topk=1
# sample_visualization()
data_groups={
    'S1':['reg005','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010'],
'S2':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010','reg0011','reg0012'],
'S3':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010','reg0011','reg0012','reg0013','reg0014'],
'S4':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg008'],
'marchsample_S1':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010'],
'marchsample_S2':['reg001','reg002','reg003','reg004','reg005','reg006','reg007','reg008','reg009','reg0010','reg0011','reg0012'],
             }
# --- Iterate through all pickle files ---
for sample_id, regions in data_groups.items():
    for region_id in regions:
        # Load h5ad file

        adata = sc.read_h5ad(os.path.join(root_dir, "regional_data", f"{sample_id}_{region_id}.h5ad"))
        cell_ids = adata.obs['label'].tolist()
        marker_read = adata.obs['p16'].tolist()
        # pos_cells, neg_cells = select_top_bottom_cells(cell_ids, marker_read, topk)

        pos_cells,neg_cells = select_top_and_random_negatives(cell_ids, marker_read, topk)
        del adata

        # Load pickle file
        with open(os.path.join(root_dir, sample_id, f"{region_id}.pickle"), "rb") as fh:
            data = pickle.load(fh)

        masks = data["masks"]
        img = data["image_dict"][channel]

        print(f"Begin saving images of {sample_id}_{region_id}...")

        # Save positive cells
        for cell_id in pos_cells:
            crop = crop_square_masked(img, masks, cell_id)
            crop_adj = exposure.rescale_intensity(crop, out_range=(0, 255)).astype(np.uint8)

            outpath = os.path.join(pos_out_dir, f"{sample_id}_{region_id}_{cell_id}.png")
            imageio.imwrite(outpath, crop_adj)

        # Save negative cells
        for cell_id in neg_cells:
            crop = crop_square_masked(img, masks, cell_id)
            crop_adj = exposure.rescale_intensity(crop, out_range=(0, 255)).astype(np.uint8)

            outpath = os.path.join(neg_out_dir, f"{sample_id}_{region_id}_{cell_id}.png")
            imageio.imwrite(outpath, crop_adj)

        print(f" -> Saved {len(pos_cells) + len(neg_cells)} cells\n")






