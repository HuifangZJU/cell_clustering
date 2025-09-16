import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import tifffile as tiff
import os
from PIL import Image
import time

def visualize_image_with_polygons(image_data, cell_boundaries):
    img_np = image_data[:,:]

    fig, ax = plt.subplots(figsize=(10, 10))
    # Optional: set figure size
    ax.imshow(img_np, origin='upper')  # origin='upper' matches (0,0) at top-left

    # Plot the boundaries
    for _, group in cell_boundaries.groupby("cell_id"):
        coords = group[["vertex_x", "vertex_y"]].values
        polygon = Polygon(coords, closed=True, fill=False, edgecolor='red', linewidth=0.5)
        ax.add_patch(polygon)

    # Formatting
    ax.set_title("Cell Boundaries Overlayed on Tissue Image", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def select_top_bottom_cells(df, k=1):
    # # Compute thresholds
    lower_thresh = df["ds"].quantile(k / 100.0)
    upper_thresh = df["ds"].quantile(1 - k / 100.0)

    # Select bottom k% (ds <= lower_thresh)
    bottom_cells = df[df["ds"] <= lower_thresh]["cell_id"].tolist()

    # Select top k% (ds >= upper_thresh)
    top_cells = df[df["ds"] >= upper_thresh]["cell_id"].tolist()

    return top_cells, bottom_cells

def normalize_to_save(img):
    # Check the min and max of the original image
    img_min = img.min()
    img_max = img.max()
    # print(f"Original Image Min: {img_min}, Max: {img_max}")

    # Normalize the image to [0, 255]
    # Formula: (pixel_value - min) / (max - min) * 255
    # Since we know min=0 and max=4469, this simplifies to:
    img_normalized = ((img.astype(np.float32)-img_min) / (img_max-img_min)) * 255.0

    # Convert back to an unsigned 8-bit integer type
    img_normalized = img_normalized.astype(np.uint8)
    return img_normalized

def get_centroid(cell_centers, cell_id):
    row = cell_centers.loc[cell_centers['cell_id'] == cell_id]
    if row.empty:
        return None  # or raise an error
    return tuple(row[['x_centroid', 'y_centroid']].values[0])

def crop_and_save_snc_images(img_np, cell_boundaries, cell_centers,top_cells, bottom_cells,output_dir,sample_id):


    # Group boundaries by cell_id
    boundary_group = cell_boundaries.groupby("cell_id")
    # Loop through each cell
    for cell_id in top_cells:
        x_center, y_center = get_centroid(cell_centers,cell_id)
        boundary = boundary_group.get_group(cell_id)

        dx = np.abs(boundary["vertex_x"] - x_center).max()
        dy = np.abs(boundary["vertex_y"] - y_center).max()
        radius = int(np.ceil(max(dx, dy)))

        left = int(max(0, x_center - radius))
        right = int(min(img_np.shape[1], x_center + radius))
        top = int(max(0, y_center - radius))
        bottom = int(min(img_np.shape[0], y_center + radius))

        # Save each channel separately

        crop = img_np[top:bottom, left:right]
        save_path = os.path.join(output_dir+'snc', f"{sample_id}_{cell_id}.png")
        Image.fromarray(normalize_to_save(crop)).save(save_path)

    for cell_id in bottom_cells:
        x_center, y_center = get_centroid(cell_centers, cell_id)
        boundary = boundary_group.get_group(cell_id)

        dx = np.abs(boundary["vertex_x"] - x_center).max()
        dy = np.abs(boundary["vertex_y"] - y_center).max()
        radius = int(np.ceil(max(dx, dy)))

        left = int(max(0, x_center - radius))
        right = int(min(img_np.shape[1], x_center + radius))
        top = int(max(0, y_center - radius))
        bottom = int(min(img_np.shape[0], y_center + radius))

        # Save each channel separately

        crop = img_np[top:bottom, left:right]
        save_path = os.path.join(output_dir + 'non-snc', f"{sample_id}_{cell_id}.png")
        Image.fromarray(normalize_to_save(crop)).save(save_path)


def crop_and_save_images(img_np, cell_boundaries, cell_centers, subset,output_dir,sample_id):
    centers_scores = pd.merge(cell_centers, subset[['cell_id', 'ds']], on="cell_id", how="inner")
    # Group boundaries by cell_id
    boundary_group = cell_boundaries.groupby("cell_id")

    # Create subfolder for this sample_id
    sample_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # Image list file inside sample_id folder
    list_file = os.path.join(sample_dir, "image_lists.txt")
    with open(list_file, "w") as f:  # overwrite each run for this sample_id
        for _, row in centers_scores.iterrows():
            cell_id = row["cell_id"]
            x_center, y_center, ds_score = row['x_centroid'], row['y_centroid'], row['ds']
            boundary = boundary_group.get_group(cell_id)

            dx = np.abs(boundary["vertex_x"] - x_center).max()
            dy = np.abs(boundary["vertex_y"] - y_center).max()
            radius = int(np.ceil(max(dx, dy)))

            left = int(max(0, x_center - radius))
            right = int(min(img_np.shape[1], x_center + radius))
            top = int(max(0, y_center - radius))
            bottom = int(min(img_np.shape[0], y_center + radius))

            # Save each channel separately

            crop = img_np[top:bottom, left:right]
            filename = f"{cell_id}.png"  # exclude sample_id
            save_path = os.path.join(sample_dir, filename)
            Image.fromarray(normalize_to_save(crop)).save(save_path)

            # Record filename + ds score (relative path within sample folder)
            f.write(f"{filename} {ds_score:.8f}\n")



def change2xenium_sampleid(s):
    return s[:-1]+'-'+s[-1]

root_path = "/media/huifang/data1/sennet/xenium/"
# output_dir ="/media/huifang/data1/sennet/xenium/deepscence/cell_images/"
output_dir ="/media/huifang/data1/sennet/xenium/deepscence/cell_images/all/"
deepscence_file = "/media/huifang/data1/sennet/xenium/deepscence/xall_scored_meta.csv"
ds_score = pd.read_csv(deepscence_file)
# Split the "Unnamed: 0" column into sampleid and cell_id
ds_score[['sampleid', 'cell_id']] = ds_score['Unnamed: 0'].str.split('_', n=1, expand=True)
# Keep only needed columns
result = ds_score[['sampleid', 'cell_id', 'ds']]


dataset = open(root_path+'data_list.txt')
lines = dataset.readlines()
for i in range(len(lines)):
    line = lines[i]
    line = line.rstrip().split(' ')
    subfolder = os.path.join(root_path, line[0])
    if os.path.exists(os.path.join(subfolder, 'outs')):
        subfolder = os.path.join(subfolder, 'outs')
    sample_id = change2xenium_sampleid( line[1])

    subset = result[result['sampleid'] == sample_id]



    cell_path = subfolder +'/cells.csv.gz'
    cell_boundary_path = subfolder + '/cell_boundaries.csv.gz'
    ome_tiff_path = subfolder + "/morphology_focus/morphology_focus_0000.ome.tif"
    para_path = subfolder +'/experiment.xenium'

    cells = pd.read_csv(cell_path)

    # Load parameters
    with open(para_path, "r") as file:
        experiment_data = json.load(file)
    pixel_size = experiment_data.get("pixel_size")

    # Load cell boundaries
    cell_boundaries = pd.read_csv(cell_boundary_path)
    cell_boundaries["vertex_x"] /= pixel_size
    cell_boundaries["vertex_y"] /= pixel_size

    # Load cell centers (safe copy)
    cell_centers = cells[['cell_id', 'x_centroid', 'y_centroid']].copy()
    cell_centers.loc[:, 'x_centroid'] /= pixel_size
    cell_centers.loc[:, 'y_centroid'] /= pixel_size

    # Load image (first plane only to avoid series warnings)
    image_data = tiff.imread(ome_tiff_path, key=0)

    # visualize_image_with_polygons(image_data,cell_boundaries)

    # # Crop and save the images
    # top_cells, bottom_cells = select_top_bottom_cells(subset, 1)
    # crop_and_save_snc_images(image_data, cell_boundaries, cell_centers, top_cells, bottom_cells, output_dir,sample_id)
    crop_and_save_images(image_data, cell_boundaries, cell_centers, subset, output_dir, sample_id)
    #




