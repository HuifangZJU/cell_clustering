import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import tifffile as tiff
import os
from PIL import Image
import dask.array as da
import time

def visualize_image_with_polygons(image_data, cell_boundaries):
    img_np = image_data[0,:,:]

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


def load_ome_tiff_efficiently(ome_tiff_path, use_dask=False):
    """
    Efficiently loads an OME-TIFF file, either as a memory-mapped array or using Dask for lazy loading.

    Parameters:
        ome_tiff_path (str): Path to the OME-TIFF file.
        use_dask (bool): Whether to use Dask for lazy loading. Defaults to False.

    Returns:
        array-like: The loaded image data (NumPy or Dask array).
    """
    try:
        with tiff.TiffFile(ome_tiff_path) as ome_tiff:
            # Inspect metadata
            print("OME-TIFF Metadata:")
            print(ome_tiff.series[0].shape)  # Example of inspecting the shape of the first series

            # Access the desired series (e.g., the first one)
            series = ome_tiff.series[0]

            if use_dask:
                # Load as Dask array for lazy processing
                image_data = da.from_array(series.asarray(), chunks="auto")
            else:
                # Load as memory-mapped NumPy array for immediate access
                image_data = series.asarray(out="memmap")

            # Select the first slice if ndim > 2
            #if image_data.ndim > 2:
                #image_data = image_data[0, 0]

            return image_data
    except Exception as e:
        print(f"Error loading OME-TIFF: {e}")
        return None


def crop_and_save_images(img_np, cell_boundaries, cell_centers,output_dir):


    # Group boundaries by cell_id
    boundary_group = cell_boundaries.groupby("cell_id")

    # Prepare output subfolders for channels 0–3
    for ch in range(4):
        os.makedirs(os.path.join(output_dir, str(ch)), exist_ok=True)

    # Loop through each cell
    for _, row in cell_centers.iterrows():
        cell_id = row["cell_id"]
        print('saving '+ str(cell_id) + " ...")
        x_center, y_center = row["x_centroid"], row["y_centroid"]

        if cell_id not in boundary_group.groups:
            continue

        boundary = boundary_group.get_group(cell_id)

        dx = np.abs(boundary["vertex_x"] - x_center).max()
        dy = np.abs(boundary["vertex_y"] - y_center).max()
        radius = int(np.ceil(max(dx, dy)))



        left = int(max(0, x_center - radius))
        right = int(min(img_np.shape[2], x_center + radius))
        top = int(max(0, y_center - radius))
        bottom = int(min(img_np.shape[1], y_center + radius))

        # Save each channel separately
        for ch in range(4):
            crop = img_np[ch, top:bottom, left:right]
            save_path = os.path.join(output_dir, str(ch), f"{cell_id}.png")

            if isinstance(crop, da.Array):
                # Convert Dask array to NumPy array
                crop = crop.compute()

            Image.fromarray(normalize_to_save(crop)).save(save_path)

            # Image.fromarray(crop.astype(np.uint8)).save(save_path)
            # print('saved')
            # test = input()


def analyze_image(img):
    # Step 1: Truncate values greater than 1000
    max_val = 3000
    img_clipped = np.clip(img, 0, max_val)

    # Step 2: Normalize to [0, 255] after truncation
    img_normalized = (img_clipped.astype(np.float32) / max_val) * 255.0
    img_normalized = img_normalized.astype(np.uint8)
    plt.imshow(img_normalized)
    plt.show()

    # Show histograms to visualize the distribution before and after truncation/normalization
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(img.ravel(), bins=256, range=(0, img.max()), color='blue', alpha=0.7)
    plt.title("Original Image Value Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(img_normalized.ravel(), bins=256, range=(0, 255), color='green', alpha=0.7)
    plt.title("Truncated & Normalized Image Distribution")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()



def run_sigle_data(para_path, ome_tiff_path,cell_boundary_path,cell_path,cell_types_file_path,output_dir):

    # cell_types = pd.read_csv(cell_types_file_path)
    # df = pd.read_csv(cell_boundary_path)
    # cells = pd.read_csv(cell_path)
    # cell_centers = cells[['cell_id', 'x_centroid', 'y_centroid']]
    #
    # merged = cell_centers.merge(cell_types, left_on='cell_id', right_on='Barcode')
    #
    # # --- Plot ---
    # plt.figure(figsize=(12, 12))
    #
    # scatter = plt.scatter(
    #     merged['x_centroid'],
    #     merged['y_centroid'],
    #     c=merged['Cluster'],
    #     cmap='tab20',  # use a categorical colormap
    #     s=10,
    #     alpha=0.8
    # )
    #
    # plt.gca().invert_yaxis()  # match image coordinates
    # plt.axis('equal')
    # plt.axis('off')
    # plt.title('Cell Type Spatial Distribution')
    # # Define ticks as unique sorted integer cluster values
    # clusters = sorted(merged['Cluster'].unique())
    # cbar = plt.colorbar(scatter, ticks=clusters)
    # cbar.set_label('Cell Type Cluster')
    # plt.tight_layout()
    # plt.show()
    # test = input()


    # Load experiment parameters
    with open(para_path, "r") as file:
        experiment_data = json.load(file)
    pixel_size = experiment_data.get("pixel_size")


    cell_types = pd.read_csv(cell_types_file_path)
    # Load the cell boundary data
    cell_boundaries = pd.read_csv(cell_boundary_path)
    cell_boundaries["vertex_x"] = cell_boundaries["vertex_x"] / pixel_size
    cell_boundaries["vertex_y"] = cell_boundaries["vertex_y"] / pixel_size
    cells = pd.read_csv(cell_path)
    cell_centers = cells[['cell_id', 'x_centroid', 'y_centroid']]
    cell_centers['x_centroid'] = cell_centers['x_centroid']/pixel_size
    cell_centers['y_centroid'] = cell_centers['y_centroid'] / pixel_size

    # img_np = tiff.imread(ome_tiff_path)
    # print(cell_centers)
    # print(cell_boundaries)



    # Load the OME-TIFF file
    image_data = load_ome_tiff_efficiently(ome_tiff_path, use_dask=True)
    # image_data = tiff.imread(ome_tiff_path)



    # # Crop and save the images
    crop_and_save_images(image_data, cell_boundaries, cell_centers, output_dir)


root_path = "/media/huifang/data/Xenium/sennet_data/"
if __name__ == "__main__":
    dataset = open(root_path+'data_list.txt')
    lines = dataset.readlines()
    for i in range(len(lines)):
        line = lines[i]
        line = line.rstrip()
        print(i)
        print(line)


        cell_path = root_path+line+'/cells.csv.gz'
        cell_boundary_path = root_path + line + '/cell_boundaries.csv.gz'
        # cell_boundary_path = root_path + line + '/nucleus_boundaries.csv.gz'
        ome_tiff_path = root_path + line + "/morphology_focus/morphology_focus_0000.ome.tif"
        para_path = root_path+line+'/experiment.xenium'
        cell_types_file_path = root_path+line+'/gene_cluster_result/hvg/clusters_all.csv'

        output_dir = root_path+line+'/preprocessing'
        start_time = time.time()
        run_sigle_data(para_path,ome_tiff_path,cell_boundary_path,cell_path,cell_types_file_path,output_dir)
        elapsed_time = time.time() - start_time
        print(f"Finished dataset {i}: {line} in {elapsed_time:.2f}s")





