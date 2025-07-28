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

def visualize_image_with_polygons(image_data, cell_boundaries, pixel_size):
    """
    Visualize the image with polygons and centroids overlaid.

    :param image_data: The image data array (2D).
    :param cell_boundaries: The DataFrame containing cell boundary information.
    :param pixel_size: The scaling factor for converting coordinates.
    """
    # Plot the image with polygons overlayed
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image in its original size
    ax.imshow(image_data, cmap='gray', origin='upper')

    # Loop through each cell_id and plot its polygon
    patches = []
    centroids = []
    cell_ids = cell_boundaries['cell_id'].unique()
    for cell_id in cell_ids:
        # Get the vertices for the current cell
        cell_data = cell_boundaries[cell_boundaries['cell_id'] == cell_id]
        vertices = cell_data[['vertex_x', 'vertex_y']].values
        vertices = vertices / pixel_size

        # Create a polygon for the cell
        polygon = Polygon(vertices, closed=True, edgecolor='blue', facecolor='none', linewidth=0.5)
        patches.append(polygon)

        # Calculate the centroid of the polygon
        centroid_x = np.mean(vertices[:, 0])
        centroid_y = np.mean(vertices[:, 1])
        centroids.append((centroid_x, centroid_y))

    # Add all patches to the axes
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)

    # Plot the centroids
    centroids = np.array(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], color='red', s=10, label='Centroids')

    # Set axis limits based on the image dimensions
    ax.set_xlim(0, image_data.shape[1])
    ax.set_ylim(image_data.shape[0], 0)  # Flip the y-axis to match the image coordinates
    ax.set_aspect('equal')
    ax.set_title("Cell Boundary Polygons Overlaid on Image")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.legend()
    plt.show()


def normalize_to_save(img):
    # Check the min and max of the original image
    img_min = img.min()
    img_max = img.max()
    # print(f"Original Image Min: {img_min}, Max: {img_max}")

    # Normalize the image to [0, 255]
    # Formula: (pixel_value - min) / (max - min) * 255
    # Since we know min=0 and max=4469, this simplifies to:
    img_normalized = ((img.astype(np.float32)-img_min) / img_max) * 255.0

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
            if image_data.ndim > 2:
                image_data = image_data[0, 0]

            return image_data
    except Exception as e:
        print(f"Error loading OME-TIFF: {e}")
        return None


def crop_and_save_images_with_types(image_data, cell_boundaries, cell_types, output_dir, pixel_size):
    # Merge cell types with cell boundaries
    # Rename the key in cell_types to match cell_boundaries
    cell_types = cell_types.rename(columns={'cellid': 'cell_id'})

    # Merge cell types with cell boundaries
    cell_data = cell_boundaries.merge(cell_types, on='cell_id')
    # cell_data = cell_data.sample(n=50, random_state=42)


    # Ensure the output directories exist
    fixed_ratio_dir = os.path.join(output_dir, "fixed_ratio")
    # fixed_size_dir = os.path.join(output_dir, "fixed_size")
    os.makedirs(fixed_ratio_dir, exist_ok=True)
    # os.makedirs(fixed_size_dir, exist_ok=True)

    # # Calculate the maximum bounding box size for 'fixed_size' cropping
    # max_size = 0
    # for cell_id in cell_data['cell_id'].unique():
    #     vertices = cell_boundaries[cell_boundaries['cell_id'] == cell_id][['vertex_x', 'vertex_y']].values
    #     vertices = vertices / pixel_size
    #     minx, miny = vertices.min(axis=0)
    #     maxx, maxy = vertices.max(axis=0)
    #     max_size = max(max_size, max(maxx - minx, maxy - miny))
    # max_size = int(np.ceil(max_size))

    # Process each cell
    cell_info = []  # will hold tuples of (cell_id, centroid_x, centroid_y)
    fixed_ratio_entries = []  # will hold lines for fixed_ratio images: (cell_id, path, cell_type)
    fixed_size_entries = []    # will hold lines for fixed_size images: (cell_id, path, cell_type)
    for cell_id in cell_data['cell_id'].unique():
        # Get cell type
        cell_type = cell_data[cell_data['cell_id'] == cell_id]['celltype'].iloc[0]
        # Prepare subsubfolders
        fixed_ratio_type_dir = os.path.join(fixed_ratio_dir, cell_type)
        # fixed_size_type_dir = os.path.join(fixed_size_dir, cell_type)
        os.makedirs(fixed_ratio_type_dir, exist_ok=True)
        # os.makedirs(fixed_size_type_dir, exist_ok=True)

        # Get cell vertices
        vertices = cell_boundaries[cell_boundaries['cell_id'] == cell_id][['vertex_x', 'vertex_y']].values
        vertices = vertices / pixel_size
        minx, miny = vertices.min(axis=0)
        maxx, maxy = vertices.max(axis=0)
        # Determine the side length of the square (the larger dimension)
        side = int(max(maxx - minx, maxy - miny))
        # Compute the center of the bounding box
        center_x = int((minx + maxx) / 2.0)
        center_y = int((miny + maxy) / 2.0)

        # Add cell info to the list (using integer or float as needed)
        cell_info.append((cell_id, center_x, center_y, cell_type))

        # Half the side length to find the square boundaries
        half_side = side // 2
        # Ensure boundaries don't go outside the image range if needed (optional)
        square_minx = max(center_x - half_side, 0)
        square_miny = max(center_y - half_side, 0)
        square_maxx = min(center_x + half_side, image_data.shape[1])
        square_maxy = min(center_y + half_side, image_data.shape[0])
        # Crop the image to the square region
        fixed_ratio_crop = image_data[square_miny:square_maxy, square_minx:square_maxx]
        # Before saving the image
        if isinstance(fixed_ratio_crop, da.Array):
            # Convert Dask array to NumPy array
            fixed_ratio_crop = fixed_ratio_crop.compute()
        # Save the cropped image
        fixed_ratio_path = os.path.join(fixed_ratio_type_dir, f"cell_{cell_id}.png")

        Image.fromarray(normalize_to_save(fixed_ratio_crop)).save(fixed_ratio_path)

        # # Crop 'fixed_size'
        # center_x, center_y = int((minx + maxx) / 2), int((miny + maxy) / 2)
        # half_size = max_size // 2
        # fixed_size_crop = image_data[
        #     max(0, center_y - half_size):min(image_data.shape[0], center_y + half_size),
        #     max(0, center_x - half_size):min(image_data.shape[1], center_x + half_size)
        # ]
        # if isinstance(fixed_size_crop, da.Array):
        #     # Convert Dask array to NumPy array
        #     fixed_size_crop = fixed_size_crop.compute()
        #
        # fixed_size_path = os.path.join(fixed_size_type_dir, f"cell_{cell_id}.png")
        # Image.fromarray(normalize_to_save(fixed_size_crop)).save(fixed_size_path)

        fixed_ratio_entries.append((cell_id, fixed_ratio_path, cell_type))
        # fixed_size_entries.append((cell_id, fixed_size_path, cell_type))
    # Create a DataFrame from collected info
    print('begin saveing outputs..')
    cell_info_df = pd.DataFrame(cell_info, columns=['cell_id', 'centroid_x', 'centroid_y', 'cell_type'])

    # Save cell centroids and types to CSV
    cell_info_csv_path = os.path.join(output_dir, "cell_centroids.csv")
    cell_info_df.to_csv(cell_info_csv_path, index=False)

    # Map each cell_type to an integer class ID
    unique_types = cell_info_df['cell_type'].unique()
    type_to_id = {ct: i for i, ct in enumerate(unique_types)}

    # Write the image_list files
    # Format: cell_id image_path class_id
    fixed_ratio_list_path = os.path.join(output_dir, "fixed_ratio_image_list.txt")
    with open(fixed_ratio_list_path, 'w') as f:
        for cid, img_path, ctype in fixed_ratio_entries:
            f.write(f"{cid} {img_path} {type_to_id[ctype]}\n")

    # fixed_size_list_path = os.path.join(output_dir, "fixed_size_image_list.txt")
    # with open(fixed_size_list_path, 'w') as f:
    #     for cid, img_path, ctype in fixed_size_entries:
    #         f.write(f"{cid} {img_path} {type_to_id[ctype]}\n")

    # Plot the distribution of cell types based on centroid positions
    fig, ax = plt.subplots()

    for ct in cell_info_df['cell_type'].unique():
        subset = cell_info_df[cell_info_df['cell_type'] == ct]
        ax.scatter(subset['centroid_x'], subset['centroid_y'], label=ct, alpha=0.7,s=2, edgecolors='none')

    ax.set_xlabel("Centroid X")
    ax.set_ylabel("Centroid Y")
    ax.set_title("Cell Type Distribution by Centroid Location")
    ax.legend()

    # Save the figure without displaying
    fig_path = os.path.join(output_dir, "cell_type_distribution.png",)
    plt.savefig(fig_path, dpi=600)
    plt.close(fig)

    # Plot the distribution of cell numbers based on cell types
    # Count how many cells of each type we have
    counts = cell_info_df['cell_type'].value_counts()

    # We rely on the order in which type_to_id was constructed, typically from unique_types
    unique_types_in_order = list(type_to_id.keys())

    labels = []
    counts_in_order = []
    for ct in unique_types_in_order:
        class_id = type_to_id[ct]
        labels.append(f"{class_id}: {ct}")
        # Get the count for this cell type (0 if not present)
        counts_in_order.append(counts.get(ct, 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(unique_types_in_order)), counts_in_order, color='skyblue')

    ax.set_xticks(range(len(unique_types_in_order)))
    ax.set_xticklabels(labels, rotation=90)

    ax.set_xlabel("Class ID : Cell Type")
    ax.set_ylabel("Number of Cells")
    ax.set_title("Distribution of Cell Types")

    # Save the figure as a high-resolution PNG without showing it
    output_path = os.path.join(output_dir, "cell_type_distribution_counts.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

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



def run_sigle_data(para_path, ome_tiff_path,cell_boundary_path,cell_types_file_path,output_dir):
    # Load experiment parameters
    with open(para_path, "r") as file:
        experiment_data = json.load(file)
    pixel_size = experiment_data.get("pixel_size")

    cell_types = pd.read_csv(cell_types_file_path)
    # Load the cell boundary data
    cell_boundaries = pd.read_parquet(cell_boundary_path)

    if not pd.api.types.is_integer_dtype(cell_boundaries['cell_id']):
        print("The dtype of 'cell_id' is not an integer type. Decoding to string_id")
        # Decode the bytes objects to strings
        cell_boundaries['cell_id'] = cell_boundaries['cell_id'].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # Remove the `b'` and trailing `'` from the strings
        cell_boundaries['cell_id'] = cell_boundaries['cell_id'].str.replace(r"^b'|'$", '', regex=True)


    # Load the OME-TIFF file
    image_data = load_ome_tiff_efficiently(ome_tiff_path, use_dask=True)

    # ome_tiff = tiff.TiffFile(ome_tiff_path)
    # image_data = ome_tiff.asarray()
    # if image_data.ndim > 2:
    #     image_data = image_data[0, 0]
    # analyze_image(image_data)
    # Visualize the image with polygons
    # visualize_image_with_polygons(image_data, cell_boundaries, pixel_size)



    # # Crop and save the images
    crop_and_save_images_with_types(image_data, cell_boundaries, cell_types, output_dir, pixel_size)


root_path = "/media/huifang/data/Xenium/xenium_data/"
if __name__ == "__main__":
    dataset = open('data_list.txt')
    lines = dataset.readlines()
    for i in range(15,len(lines)):
        line = lines[i]
        line = line.rstrip()
        print(i)
        print(line)
        para_path = root_path+line+'/experiment.xenium'
        ome_tiff_path = root_path + line+"/morphology_focus.ome.tif"
        cell_boundary_path = root_path + line+ "/cell_boundaries.parquet"
        cell_types_file_path = "/media/huifang/data/Xenium/celltype/"+line+'.csv'
        output_dir = root_path+line+'/preprocessing'
        start_time = time.time()
        run_sigle_data(para_path,ome_tiff_path,cell_boundary_path,cell_types_file_path,output_dir)
        elapsed_time = time.time() - start_time
        print(f"Finished dataset {i}: {line} in {elapsed_time:.2f}s")





