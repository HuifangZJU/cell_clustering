import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the centroids and cell type data
centroids_file = "/media/huifang/data/Xenium/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs/preprocessing/cell_centroids.csv"
cell_types_file = "/media/huifang/data/Xenium/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs/preprocessing/Xenium_V1_FF_Mouse_Brain_Coronal_Subset.csv"


centroids_data = pd.read_csv(centroids_file)
cell_types_data = pd.read_csv(cell_types_file)

# Merge the centroids data with cell types
merged_data = pd.merge(centroids_data, cell_types_data, left_on="cell_id", right_on="cellid")

# Plot the cell type distribution spatially
plt.figure(figsize=(12, 12))
sns.scatterplot(
    x=merged_data["centroid_x"],
    y=merged_data["centroid_y"],
    hue=merged_data["celltype"],
    palette="tab20",  # Adjust palette as needed
    s=10,  # Adjust marker size
    alpha=0.8  # Adjust transparency
)

plt.title("Spatial Distribution of Cell Types")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.gca().invert_yaxis()  # Invert y-axis for correct spatial representation
plt.tight_layout()
plt.show()
