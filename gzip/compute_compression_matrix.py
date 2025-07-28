"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import gzip
import cv2
from matplotlib import pyplot as plt
from PIL import Image



import numpy as np
import gzip
import time
import cv2
import multiprocessing
import torch
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from pyclustering.cluster.kmedoids import kmedoids
import shutil
import tarfile
import os
import gzip
from sklearn.cluster import KMeans
C_LEVEL=4
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import numpy as np
from sklearn.manifold import MDS

def compute_cx12_with_tar(images,ids):
    # Create the target folder
    folder_name = './temp/'+'-'.join([f'id{idx}' for idx in ids])
    tar_file_name = './temp/'+'-'.join([f'id{idx}' for idx in ids])+'.tar'

    os.makedirs(folder_name)

    # Move images to the folder
    for image_path in images:
        shutil.copy(image_path, os.path.join(folder_name, os.path.basename(image_path)))

    # Create the tar file
    with tarfile.open(tar_file_name, 'w') as tar:
        tar.add(folder_name)

    # Compress the tar file with gzip
    with open(tar_file_name, 'rb') as f_in:
        compressed_data = gzip.compress(f_in.read(),compresslevel=C_LEVEL)

    for root, dirs, files in os.walk(folder_name):
        for file in files:
            os.remove(os.path.join(root, file))

    # Delete the temporary folder
    shutil.rmtree(folder_name)
    # Delete the temporary tar file and the folder
    os.remove(tar_file_name)
    return len(compressed_data)

def compute_tar_distance(cx1,path1,id1,cx2,path2,id2):
    cx12 = compute_cx12_with_tar([path1,path2],[id1,id2])
    ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)
    return ncd


def compute_minumum_distance(img1_origin, cx1, img2_origin, cx2):
    img2s = [img2_origin, np.fliplr(img2_origin), np.flipud(img2_origin), np.fliplr(np.flipud(img2_origin))]
    img1s = [img1_origin, np.fliplr(img1_origin), np.flipud(img1_origin), np.fliplr(np.flipud(img1_origin))]
    ncds = []
    for img1 in img1s:
        for img2 in img2s:
            img12 = np.concatenate((img1, img2), axis=1)
            cx12 = len(gzip.compress(img12.tobytes(), compresslevel=C_LEVEL))
            ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)
            ncds.append(ncd)
            # img21 = np.concatenate((img2, img1), axis=1)
            # cx21 = len(gzip.compress(img21.tobytes(), compresslevel=C_LEVEL))
            # ncd = (cx21 - min(cx1, cx2)) / max(cx1, cx2)
            # ncds.append(ncd)
    return min(ncds)

def compute_distance(img1, cx1, img2, cx2):
    img12 = np.concatenate((img1, img2), axis=0)
    cx12 = len(gzip.compress(img12.tobytes(), compresslevel=C_LEVEL))
    ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)
    return ncd


def load_images_and_labels(data):
    paths = [line.strip().split(' ')[0] for line in data]
    imgs = [cv2.resize(cv2.imread(line.strip().split(' ')[0], cv2.IMREAD_GRAYSCALE), (96, 96)) for line in data]
    # imgs = [cv2.imread(line.strip().split(' ')[0], cv2.IMREAD_GRAYSCALE) for line in data]
    labels = [line.strip().split(' ')[1] for line in data]
    sizes = [float(line.strip().split(' ')[2]) for line in data]

    return paths,imgs, labels,sizes

def process_test_image(test_image_data):
    test_img, test_label, test_cx, images, labels, cxs = test_image_data
    distance_from_i = [compute_minumum_distance(test_img, test_cx, img2, cx2)
        for img2, label2, cx2 in zip(images, labels, cxs)
    ]
    distance_from_i = np.array(distance_from_i)
    return distance_from_i

def process_test_image_tar(test_image_data):
    test_label, test_cx, test_path, test_id, labels, cxs, paths, ids = test_image_data
    distance_from_i = [compute_tar_distance(test_cx, test_path,test_id, cx2, path2,id2)
        for label2, cx2, path2, id2 in zip(labels, cxs, paths,ids)
    ]
    distance_from_i = np.array(distance_from_i)
    return distance_from_i

def find_nearest_neighbors(distances, k):
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    return neighbors

def comput_ncd(images,labels,sizes):
    num_cpus = multiprocessing.cpu_count()
    cxs = np.array([len(gzip.compress(img.tobytes(), compresslevel=C_LEVEL)) for img in images])

    test_image_data_list = zip(images, labels, cxs, [images] * len(images), [labels] * len(images), [cxs] * len(images))

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        distances = pool.map(process_test_image, test_image_data_list)
    distances = np.array(distances)
    np.save('ncd_minimum_distances_downsample_4.npy',distances)
    # np.save('labels.npy', labels)
    # size_distance = compute_size_abstract_distance(sizes)
    # np.save('size_distances.npy',size_distance)


def compute_tar_ncd(paths,ids,labels):
    num_cpus = multiprocessing.cpu_count()
    cxs = [compute_cx12_with_tar([img],[id]) for img,id in zip(paths,ids)]
    # cxs = [len(gzip.compress(img.tobytes(), compresslevel=C_LEVEL)) for img in images]

    test_image_data_list = zip(labels, cxs, paths,ids, [labels] * len(labels),
                               [cxs] * len(labels), [paths]*len(labels),[ids]*len(labels))

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        distances = pool.map(process_test_image_tar, test_image_data_list)
    distances = np.array(distances)
    np.save('ncd_tar_distances.npy', distances)

def calculate_dataset_matrix():
    with open('/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_all_with_size.txt',
              'r') as f:
        data = f.readlines()
        # data = data[::4]
        paths, images, labels, sizes = load_images_and_labels(data)
        # List of unique class labels
        ids = np.arange(len(images))
    total_samples = len(labels)
    labels = np.array(labels)
    print("Total Number of Test Samples:", total_samples)
    # compute_tar_ncd(paths,ids,labels)
    comput_ncd(images,labels,sizes)


def compute_size_abstract_distance(size_vector):
    size_matrix = np.abs(np.subtract.outer(size_vector, size_vector))
    return size_matrix


import networkx as nx
import leidenalg as la
from igraph import Graph

def build_knn_network(distances, k):
    n_points = distances.shape[0]
    knn_network = nx.Graph()

    # Find k nearest neighbors for each data point
    k_nearest_neighbors = np.argsort(distances, axis=1)[:, 1:k+1]

    # Add edges to the k-NN network
    for i in range(n_points):
        for neighbor in k_nearest_neighbors[i]:
            knn_network.add_edge(i, neighbor)

    return knn_network


def leiden_clustering(knn_network):
    # Convert the NetworkX graph to a numpy array
    adjacency_matrix = nx.to_numpy_array(knn_network)


    # Create an igraph object from the adjacency matrix
    igraph_network = Graph.Adjacency(adjacency_matrix.tolist(), mode='DIRECTED')
    n_comms = 3

    partition = la.CPMVertexPartition(igraph_network,
                                      initial_membership=np.random.choice(n_comms, len(knn_network.nodes)),
                                      resolution_parameter=0.5)
    opt = la.Optimiser()
    opt.consider_empty_community = False
    opt.optimise_partition(partition,n_iterations=-1)
    # # Use the Leiden clustering algorithm with the specified resolution parameter
    # Get the cluster assignments for each data point
    cluster_indices = np.array(partition.membership)


    return cluster_indices

def plot_bin(distances):
    flattened_distances = distances.flatten()
    # Plot the histogram using Matplotlib
    plt.hist(flattened_distances, bins=100, density=True, alpha=0.7, color='b')
    plt.xlabel('Distances')
    plt.ylabel('Frequency')
    plt.title('Value Distribution of Distances')
    plt.grid(True)
    plt.show()

def plot_intra_inter_class_distances(distances, labels):

    # Get the unique classes from the labels
    unique_classes = np.unique(labels)

    # Initialize lists to store intra-class and inter-class distances
    intra_distances_all = []
    inter_distances_all = []

    # Compute distances within the same class (intra-class)
    for class_label in unique_classes:
        indices_same_class = np.where(labels == class_label)[0]
        indices_combinations = np.array(np.meshgrid(indices_same_class, indices_same_class)).T.reshape(-1, 2)
        intra_distances_all.append(distances[indices_combinations[:, 0], indices_combinations[:, 1]])

    # Compute distances between different classes (inter-class)
    for i in range(len(unique_classes)):
        inter_distances_temp=[]
        for j in range(len(unique_classes)):
            if i==j:
                continue
            indices_class_i = np.where(labels == unique_classes[i])[0]
            indices_class_j = np.where(labels == unique_classes[j])[0]
            indices_combinations = np.array(np.meshgrid(indices_class_i, indices_class_j)).T.reshape(-1, 2)
            inter_distances_temp.extend(distances[indices_combinations[:, 0], indices_combinations[:, 1]])
        inter_distances_all.append(inter_distances_temp)

    # Create separate histograms for each class
    unique_classes = np.unique(labels)

    # Plot both distributions in one figure with different colors
    for intra_distances,inter_distances in zip(intra_distances_all,inter_distances_all):
        # Determine the common x-axis range for both distributions
        min_range = min(np.min(intra_distances), np.min(inter_distances))
        max_range = max(np.max(intra_distances), np.max(inter_distances))

        # Plot both distributions with the same x-axis scale
        plt.figure(figsize=(8, 5))

        plt.hist(inter_distances, bins=100, range=(min_range, max_range), color='salmon', alpha=0.7,
                 label='Inter-Class Distances')
        plt.hist(intra_distances, bins=100, range=(min_range, max_range), color='skyblue', alpha=0.7,
                 label='Intra-Class Distances')


        plt.xlabel('Distances')
        plt.ylabel('Frequency')
        plt.title('Intra-Class and Inter-Class Distance Distributions')
        plt.legend()

        plt.tight_layout()
        plt.show()

def plot_classwise_value_distribution(cxs, labels):
    # Get the unique classes from the labels
    unique_classes = np.unique(labels)

    # Create a separate histogram for each class
    plt.figure(figsize=(10, 6))
    for class_label in unique_classes:
        values_class = cxs[labels == class_label]
        plt.hist(values_class, bins=30, alpha=0.7, label=f'Class {class_label}')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Class-wise Value Distribution')
    plt.legend()
    plt.show()


# Define the cost function (Stress function)
def stress_function(coordinates):
    pairwise_distances = pdist(coordinates)
    return np.sum((pairwise_distances - distances.flatten()) ** 2)

def create_virtual_points(distances):
    dimensionality = 8192  # You can adjust this based on your requirements
    mds = MDS(n_components=dimensionality, dissimilarity='precomputed', random_state=4200,normalized_stress='auto')

    # Generate virtual points in the high-dimensional space
    virtual_points = mds.fit_transform(distances)
    return virtual_points




ds_rates = [4,8,16,32,64,128,256,512]
if __name__ == "__main__":
    # calculate_dataset_matrix()
    # print('done')
    # test = input()

    cxs = np.load('vizgen_cxs.npy')
    cxs = cxs.astype(float)
    sizes_all = np.load('vizgen_sizes.npy')
    sizes_all = np.array(sizes_all.astype(float))
    ncd_distances_all = np.load('ncd_minimum_distances_downsample_4.npy')
    # ncd_distances_all = np.load('vizgen_ncd.npy')
    ncd_distances_all = ncd_distances_all.astype(float)

    # plot_bin(ncd_distances_all)
    labels_all = np.load('vizgen_labels.npy')
    labels_all = labels_all.astype(int)
    size_distance_all = np.load('vizgen_size_distances.npy')
    size_distance_all = size_distance_all.astype(float)



    for ds_rate in ds_rates[::-1]:
        print("Down-sample rate is:", ds_rate)
        ds_temp = int(ds_rate/4)
        ncd_distances = ncd_distances_all[::ds_temp,::ds_temp]

        row_min = np.min(ncd_distances, axis=1, keepdims=True)
        row_max = np.max(ncd_distances, axis=1, keepdims=True)

        # Normalize each row independently
        ncd_distances = (ncd_distances - row_min) / (row_max - row_min)
        # ncd_distances = 10000*(ncd_distances - row_min)

        for i in range(ncd_distances.shape[0]):
            for j in range(ncd_distances.shape[1]):
                if ncd_distances[i,j] > ncd_distances[j,i]:
                    ncd_distances[i,j] = ncd_distances[j,i]


        # plot_bin(ncd_distances)
        labels = labels_all[::ds_rate]
        size_distance = size_distance_all[::ds_rate,::ds_rate]
        sizes = sizes_all[::ds_rate]
        sizes = sizes[:,np.newaxis]
        # plot_bin(size_distance)

        distances = ncd_distances+0.00002*size_distance

        # #
        virtual_points = create_virtual_points(distances)
        distances_reconstructed = squareform(pdist(virtual_points))

        #
        #
        #
        #
        # topk=5
        # indices = find_nearest_neighbors(distances, topk)
        # # print(indices)
        # neighbor_targets = np.take(labels, np.argsort(distances, axis=1)[:, 1:topk+1], axis=0)  # Exclude sample itself for eval
        # anchor_targets = np.repeat(labels.reshape(-1, 1), topk, axis=1)
        # accuracy = np.mean(neighbor_targets == anchor_targets)
        # print(f"Total Accuracy: {accuracy:.4f}")

        start_time = time.time()

        # Total number of clusters you want to form
        n_clusters = 3
        # # # # Step 1: Build the k-NN network
        # knn_network = build_knn_network(distances, 10)
        # # Step 2: Perform Leiden clustering and get the cluster indices
        # cluster_indices = leiden_clustering(knn_network)


        # Create a KMeans instance with the desired number of clusters
        kmeans = KMeans(n_clusters=n_clusters,n_init='auto')
        # Fit the model to your data and obtain cluster assignments
        cluster_indices = kmeans.fit_predict(virtual_points)


        # Calculate the cost matrix for the Hungarian algorithm
        n_clusters_pred = len(np.unique(cluster_indices))
        cost_matrix = np.zeros((n_clusters, n_clusters_pred))
        for i in range(n_clusters):
            mask_i = (labels == i)
            for j in range(n_clusters_pred):
                mask_j = (cluster_indices == j)
                cost_matrix[i, j] = np.logical_and(mask_i, mask_j).sum()

        # Solve the linear sum assignment problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        # Calculate the average per-class accuracy
        total_matched_samples = cost_matrix[row_ind, col_ind].sum()
        total_samples = len(labels)
        average_per_class_accuracy = total_matched_samples / total_samples

        print("Clustering Accuracy:", average_per_class_accuracy)



        #



      # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        # test = input()




