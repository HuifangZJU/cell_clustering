import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_image_path(imagelist,query):
    query_image_path = imagelist[query]
    query_image_path = query_image_path.split(' ')
    query_path = query_image_path[0]
    query_label = int(query_image_path[1])
    return query_path,query_label

def get_image_path_all(imagelist):
    paths=[]
    targets=[]
    for i in range(0,len(imagelist)):
        query_image_path = imagelist[i]
        query_image_path = query_image_path.split(' ')
        query_path = query_image_path[0]
        query_label = int(query_image_path[1])
        paths.append(query_path)
        targets.append(query_label)
    return paths,np.asarray(targets)

img_features = np.load('features.npy') # <class 'numpy.ndarray'>, shape (9321, 512)
meta_cell = pd.read_csv('filtered_meta_cell.csv', index_col=0)  #<class 'pandas.core.frame.DataFrame'>, [9321 rows x 8 columns]
neighbors = np.load('topk-train-neighbors.npy')
positions_x = meta_cell['center_x']
positions_y = meta_cell['center_y']
positions = np.column_stack((positions_x.values, positions_y.values))

imagelist = '/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_train_with_position.txt'
imagelist = open(imagelist,'r')
imagelist = imagelist.readlines()
image_paths,targets = get_image_path_all(imagelist)


# # cell_by_gene = pd.read_csv('filtered_cell_by_gene.csv', index_col=0)  #<class 'pandas.core.frame.DataFrame'>, [9321 rows x 550 columns]
# for i in range(neighbors.shape[0]):
#     query = neighbors[i, 0]
#     query_neighbor = neighbors[i, 1:]
#     query_image_path = image_paths[query]
#     query_label = targets[query]
#     print('query label is ' + query_label)
#     print('neighbor labels are: ')
#     query_image = plt.imread(query_image_path)
#
#     # plt.imshow(query_image)
#     f, a = plt.subplots(2, 3)
#     a[0, 0].imshow(query_image, cmap='gray')
#     for i in range(2):
#         for j in range(3):
#             id = i * 3 + j - 1
#             if id < 0:
#                 continue
#             temp_neighbor_image_path = image_paths[query_neighbor[id]]
#             knn_label = targets[query_neighbor[id]]
#             print(knn_label + ' ')
#             temp_neighbor_image = plt.imread(temp_neighbor_image_path)
#             a[i][j].imshow(temp_neighbor_image, cmap='gray')
#
#     plt.show()

for i in range(neighbors.shape[0]):
    query = neighbors[i, 0]
    query_neighbors = neighbors[i, 1:]

    query_image_path = image_paths[query]
    query_label = targets[query]
    query_position = positions[query]

    print('Query label:', query_label)
    print('Neighbor labels:')

    query_image = plt.imread(query_image_path)

    # Side-by-side visualization
    f, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Query image
    axes[0, 0].imshow(query_image, cmap='gray')
    axes[0, 0].set_title(f"Query: {query_label}")
    axes[0, 0].axis('off')

    # Neighbor images
    for idx, ax in enumerate(axes.flat[1:], start=1):
        if idx - 1 < len(query_neighbors):
            neighbor = query_neighbors[idx - 1]
            neighbor_image_path = image_paths[neighbor]
            neighbor_label = targets[neighbor]
            neighbor_image = plt.imread(neighbor_image_path)

            ax.imshow(neighbor_image, cmap='gray')
            ax.set_title(f"Neighbor: {neighbor_label}")
            ax.axis('off')
        else:
            ax.axis('off')

    # Show spatial distribution with colormap
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=np.arange(positions.shape[0]), cmap='viridis', alpha=0.5
    )
    ax.scatter(query_position[0], query_position[1], color='red', label='Query', edgecolor='black', s=100)

    for n_idx in query_neighbors:
        neighbor_position = positions[n_idx]
        ax.scatter(neighbor_position[0], neighbor_position[1], color='blue', label='Neighbor', s=100)

    ax.legend(loc='lower left')
    ax.set_title('Spatial Distribution')
    plt.colorbar(scatter, ax=ax, label='Point Index')

    plt.tight_layout()
    plt.show()