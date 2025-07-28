import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

train_feature = np.load('/media/huifang/data/experiment/scan/vizgen/scan/train_features.npy')
f = open('/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_train.txt')

# Assume `cells` is already defined and `train_feature` is a numpy array
cells = f.readlines()
num_cells = len(cells)

# Initialize lists or arrays to store cellids and features
cellids = []
features = []

# Loop through cells
for i in range(num_cells):
    cell = cells[i]
    cell = cell.split('.')[1]
    cell = cell.split('_')
    cellid = int(cell[-1])

    feature = train_feature[i, :]  # Extract feature for the current cell

    # Append to the lists
    cellids.append(cellid)
    features.append(feature)

# Convert to numpy arrays
cellids = np.array(cellids)
features = np.array(features)

# # Save to npy files
# np.save('./changxin/cellids.npy', cellids)
# np.save('./changxin/features.npy', features)
#
#
# # Load data
# cellids = np.load('cellids.npy')  # Shape (9321,)
# img_features = np.load('features.npy')  # Shape (9321, 512)

meta_cell = pd.read_csv('cell_metadata.csv', index_col=0)  # [713121 rows x 8 columns]
cell_by_gene = pd.read_csv('cell_by_gene.csv', index_col=0)  # [713121 rows x 550 columns]

# Ensure cellids match the index type of DataFrames
# Convert DataFrame indices to integers if needed
meta_cell.index = meta_cell.index.astype(int)
cell_by_gene.index = cell_by_gene.index.astype(int)

# Filter the rows based on cellids
filtered_meta_cell = meta_cell.loc[cellids]
filtered_cell_by_gene = cell_by_gene.loc[cellids]

# Verify shapes
print(f"Filtered meta_cell: {filtered_meta_cell.shape}")
print(f"Filtered cell_by_gene: {filtered_cell_by_gene.shape}")

# Save filtered data if needed
filtered_meta_cell.to_csv('filtered_meta_cell.csv')
filtered_cell_by_gene.to_csv('filtered_cell_by_gene.csv')

print("Filtered data saved to 'filtered_meta_cell.csv' and 'filtered_cell_by_gene.csv'")
