
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pathlib
import matplotlib.colors as mcolors
import os
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import networkx as nx
def plot_clustering_layout():
    clustering_dir = root_path + line + '/clustering/'

    for method_name in os.listdir(clustering_dir):
        method_path = os.path.join(clustering_dir, method_name)
        cell_types_file_path = os.path.join(method_path, "clusters.csv")

        cell_types = pd.read_csv(cell_types_file_path)
        merged = cell_centers.merge(cell_types, left_on='cell_id', right_on='Barcode')
        # Sorted unique cluster IDs

        # Prepare discrete colormap
        cluster_ids = sorted(merged['Cluster'].unique())
        cmap = plt.get_cmap('tab20', len(cluster_ids))
        norm = mcolors.BoundaryNorm(
            boundaries=np.arange(min(cluster_ids) - 0.5, max(cluster_ids) + 1.5),
            ncolors=len(cluster_ids)
        )

        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            merged['x_centroid'],
            merged['y_centroid'],
            c=merged['Cluster'],
            cmap=cmap,
            norm=norm,
            s=1,
            alpha=0.8
        )
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Cell Type Spatial Distribution')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, ticks=cluster_ids, fraction=0.03, pad=0.01)
        cbar.set_label('Cell Type Cluster')

        # Save to file
        plt.tight_layout()
        plt.savefig(os.path.join(clustering_dir, method_name + ".png"), dpi=300, bbox_inches='tight')
        plt.close()
        print('saved')

def visualize_cell_distrubutions():
    # Get x/y ranges from cell centers
    x_min, x_max = cell_centers['x_centroid'].min(), cell_centers['x_centroid'].max()
    y_min, y_max = cell_centers['y_centroid'].min(), cell_centers['y_centroid'].max()
    width = x_max - x_min
    height = y_max - y_min

    # Set base size and compute aspect-matching figsize
    base_size = 25  # scale this as needed
    aspect_ratio = height / width
    figsize = (base_size, base_size * aspect_ratio)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        cell_centers['x_centroid'],
        cell_centers['y_centroid'],
        # color='black',
        color='#CBA6F7',
        s=50,
        alpha=0.9
    )
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title('Cell Type Spatial Distribution')
    plt.show()

root_path = "/media/huifang/data/sennet/xenium/"
if __name__ == "__main__":
    dataset = open(root_path+'data_list.txt')
    lines = dataset.readlines()
    for i in range(5,len(lines)):
        line = lines[i]
        line = line.rstrip()
        print(i)
        print(line)


        cell_path = root_path+line.split(' ')[0]+'/outs/cells.csv.gz'
        cells = pd.read_csv(cell_path)
        cell_centers = cells[['cell_id', 'x_centroid', 'y_centroid']]

        # visualize_cell_distrubutions()
        # Select a region of interest (ROI) by bounding box (center 500,300, width/height 200)
        x_center, y_center, region_size = 5700, 600, 150
        roi = cell_centers[
            (cell_centers['x_centroid'] > x_center - region_size / 2) &
            (cell_centers['x_centroid'] < x_center + region_size / 2) &
            (cell_centers['y_centroid'] > y_center - region_size / 2) &
            (cell_centers['y_centroid'] < y_center + region_size / 2)
            ].reset_index(drop=True)

        points = roi[['x_centroid', 'y_centroid']].values
        print("Number of cells in region of interest:", len(points))


        # Determine a safe number of neighbors
        k_neighbors = min(100, len(points) - 1)  # exclude the query point itself

        # Recompute center index
        center_point = np.array([[x_center, y_center]])
        dists_to_center = np.linalg.norm(points - center_point, axis=1)
        center_idx = np.argmin(dists_to_center)

        # Build KNN model and query neighbors for the center cell
        knn_model = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(points)  # +1 includes the query point
        _, knn_indices = knn_model.kneighbors([points[center_idx]])
        knn_edges_center = [(center_idx, int(i)) for i in knn_indices[0] if i != center_idx]

        # Compute pairwise distances and MST
        dist_matrix = distance_matrix(points, points)
        mst_sparse = minimum_spanning_tree(dist_matrix)
        mst_coo = mst_sparse.tocoo()
        greedy_edges = list(zip(mst_coo.row, mst_coo.col))

        # Build NetworkX graph from MST to find shortest paths
        G = nx.Graph()
        G.add_edges_from(greedy_edges)

        # Compute shortest path lengths (number of steps) from query (center_idx)
        path_lengths = nx.single_source_shortest_path_length(G, center_idx)

        # Normalize path lengths for transparency (higher steps → more transparent)
        max_step = max(path_lengths.values())
        # Add a parameter to control exponential decay speed
        decay_rate = 0.08  # adjust this to make decay slower (< 0.4) or faster (> 0.4)

        # Recompute alpha with configurable exponential decay
        edge_alphas = []
        for i, j in greedy_edges:
            step_i = path_lengths.get(i, max_step)
            step_j = path_lengths.get(j, max_step)
            step = min(step_i, step_j)
            decay = np.exp(-decay_rate * step)
            alpha = 0.9 * decay  # scale for visibility
            edge_alphas.append(alpha)

        # Visualization comparing MST and KNN graph
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # axs[0].scatter(points[:, 0], points[:, 1], s=50, alpha=0.9, color='#CBA6F7')
        axs[0].scatter(points[:, 0], points[:, 1], s=50, alpha=0.9, color='black')

        for (i, j), alpha in zip(greedy_edges, edge_alphas):
            axs[0].plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]],
                        color='green', linewidth=5, alpha=alpha, zorder=1)

        # # Highlight center/query point
        axs[0].scatter(points[center_idx, 0], points[center_idx, 1], color='red', s=80,
                    edgecolor='black', linewidth=1, label='Query Cell', zorder=3)

        # axs[0].set_title("Greedy Connectivity (MST) with Query Distance Transparency")
        axs[0].invert_xaxis()
        axs[0].set_aspect('equal')
        axs[0].axis('off')
        axs[0].legend(loc='lower right')




        # KNN graph plot
        # axs[1].scatter(points[:, 0], points[:, 1], s=50, alpha=0.9, color='#CBA6F7')
        axs[1].scatter(points[:, 0], points[:, 1], s=50, alpha=0.9, color='black')

        for i, j in knn_edges_center:
            axs[1].plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='green', linewidth=2)
        plt.scatter(points[center_idx, 0], points[center_idx, 1], color='red', s=80, edgecolor='black',
                    linewidth=1, label='Query Cell', zorder=3)
        # axs[1].scatter(points[center_idx, 0], points[center_idx, 1], color='red', s=100, label='Query Cell')
        # axs[1].set_title("KNN Graph for One Cell")
        axs[1].invert_xaxis()
        axs[1].set_aspect('equal')
        axs[1].axis('off')
        axs[1].legend(loc='lower right')

        plt.tight_layout()
        fig.canvas.draw()
        out_dir = pathlib.Path("/home/huifang/workspace/grant/k99/resubmission/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, ax in enumerate(fig.axes, start=1):
            # tight bounding box of *this* axes in figure coordinates
            bbox = ax.get_tightbbox(fig.canvas.get_renderer()) \
                .transformed(fig.dpi_scale_trans.inverted())

            # build filename subplot_1.png, subplot_2.png, ...
            fname = out_dir / f"subplot_{idx}.png"

            # save only the region inside bbox
            fig.savefig(fname, dpi=300, bbox_inches=bbox)
            print(f"Saved {fname}")

        # plt.close(fig)  # optional: free memory
        plt.show()











