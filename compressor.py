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
import time
import cv2
import multiprocessing
import shutil
import tarfile
import os
import gzip
C_LEVEL=4


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



def compute_distance(img1_origin, cx1, img2_origin, cx2):

    # size1 = img1_origin.shape[0]
    # size2 = img2_origin.shape[0]
    # # if size1 <size2:
    # #     new_image = np.zeros([size2,size2])
    # #     new_image[:size1,:size1] = img1_origin
    # #     img1_origin = new_image
    # # else:
    # #     new_image = np.zeros([size1,size1])
    # #     new_image[:size2, :size2] = img2_origin
    # #     img2_origin = new_image
    #
    # #
    # new_size = int(np.sqrt(size1*size1+size2*size2))
    # img1_origin = cv2.resize(img1_origin,(new_size,new_size))
    # img2_origin = cv2.resize(img2_origin, (new_size, new_size))
    # # print(cx1)
    # # print(cx2)
    # # print(len(gzip.compress(img1_origin.tobytes(), compresslevel=C_LEVEL)))
    # # print(len(gzip.compress(img2_origin.tobytes(), compresslevel=C_LEVEL)))
    # # test = input()

    img2s = [img2_origin,np.fliplr(img2_origin),np.flipud(img2_origin),np.fliplr(np.flipud(img2_origin))]
    img1s = [img1_origin, np.fliplr(img1_origin), np.flipud(img1_origin), np.fliplr(np.flipud(img1_origin))]

    ncds=[]
    for img1 in img1s:
        for img2 in img2s:
            img12 = np.concatenate((img1, img2), axis=1)
            cx12 = len(gzip.compress(img12.tobytes(), compresslevel=C_LEVEL))
            ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)
            ncds.append(ncd)
            img21 = np.concatenate((img2, img1), axis=1)
            cx21 = len(gzip.compress(img21.tobytes(), compresslevel=C_LEVEL))
            ncd = (cx21 - min(cx1, cx2)) / max(cx1, cx2)
            ncds.append(ncd)
    return min(ncds)


def compute_tar_distance(img1, cx1,path1,id1, img2, cx2,path2,id2):
    cx12 = compute_cx12_with_tar([path1,path2],[id1,id2])
    ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)
    return ncd


def compute_combined_distance(img1, cx1,size1, img2, cx2,size2):
    img12 = np.concatenate((img1, img2), axis=0)
    cx12 = len(gzip.compress(img12.tobytes(), compresslevel=C_LEVEL))
    ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)
    size_dis = np.abs(size2-size1)
    return ncd+0.0002*size_dis

def calculate_accuracy(top_k_class, label1):
    return np.sum(top_k_class == label1) / len(top_k_class)

def load_images_and_labels(data):
    paths = [line.strip().split(' ')[0] for line in data]
    imgs = [cv2.resize(cv2.imread(line.strip().split(' ')[0], cv2.IMREAD_GRAYSCALE), (96, 96)) for line in data]
    # imgs = [cv2.imread(line.strip().split(' ')[0], cv2.IMREAD_GRAYSCALE) for line in data]
    labels = [line.strip().split(' ')[1] for line in data]
    sizes = [float(line.strip().split(' ')[2]) for line in data]

    return paths,imgs, labels,sizes

def process_test_image(test_image_data):
    test_img, test_label, test_cx, images, labels, cxs = test_image_data
    distance_from_i = [
        [compute_distance(test_img, test_cx, img2, cx2), label2]
        for img2, label2, cx2 in zip(images, labels, cxs)
    ]
    distance_from_i = np.array(distance_from_i)
    top_k_class = distance_from_i[np.argsort(distance_from_i[:, 0])][1:6, 1]
    acc = calculate_accuracy(top_k_class, test_label)
    return acc

def process_test_image_tar(test_image_data):
    test_img, test_label, test_cx, test_path, test_id, images, labels, cxs, paths, ids = test_image_data
    distance_from_i = [
        [compute_tar_distance(test_img, test_cx, test_path,test_id, img2, cx2, path2,id2), label2]
        for img2, label2, cx2, path2, id2 in zip(images, labels, cxs, paths,ids)
    ]
    distance_from_i = np.array(distance_from_i)
    top_k_class = distance_from_i[np.argsort(distance_from_i[:, 0])][1:6, 1]
    acc = calculate_accuracy(top_k_class, test_label)
    return acc

def process_test_image_ncd_and_size(test_image_data):
    test_img, test_label, test_cx, test_size, images, labels, cxs,sizes = test_image_data
    distance_from_i = [
        [compute_combined_distance(test_img, test_cx, test_size, img2, cx2, size2), label2]
        for img2, label2, cx2,size2 in zip(images, labels, cxs,sizes)
    ]

    distance_from_i = np.array(distance_from_i)
    top_k_class = distance_from_i[np.argsort(distance_from_i[:, 0])][:5, 1]
    acc = calculate_accuracy(top_k_class, test_label)
    return acc

def process_test_image_size(test_image_data):
    test_img, test_label, test_size, images, labels, sizes = test_image_data
    distance_from_i = [
        [np.abs(size2-test_size), label2]
        for img2, label2, size2 in zip(images, labels, sizes)
    ]
    distance_from_i = np.array(distance_from_i)
    top_k_class = distance_from_i[np.argsort(distance_from_i[:, 0])][:5, 1]
    acc = calculate_accuracy(top_k_class, test_label)
    return acc

def plot_classwise_value_distribution(cxs, labels):
    # Get the unique classes from the labels
    unique_classes = np.unique(labels)

    # Create a separate histogram for each class
    plt.figure(figsize=(10, 6))
    for class_label in unique_classes:
        values_class = cxs[labels == class_label]
        plt.hist(values_class, bins=100, alpha=0.7, label=f'Class {class_label}')

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Class-wise Value Distribution')
    plt.legend()
    plt.show()
def compute_ncd():
    num_cpus = multiprocessing.cpu_count()
    cxs = [len(gzip.compress(img.tobytes(), compresslevel=C_LEVEL)) for img in images]

    # plot_classwise_value_distribution(np.array(cxs), np.array(labels))

    test_image_data_list = zip(images, labels, cxs, [images] * len(images), [labels] * len(images), [cxs] * len(images))

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        accuracies = pool.map(process_test_image, test_image_data_list)
    return accuracies

def compute_tar_ncd():
    num_cpus = multiprocessing.cpu_count()
    cxs = [compute_cx12_with_tar([img],[id]) for img,id in zip(paths,ids)]
    # cxs = [len(gzip.compress(img.tobytes(), compresslevel=C_LEVEL)) for img in images]

    test_image_data_list = zip(images, labels, cxs, paths,ids, [images] * len(images), [labels] * len(images),
                               [cxs] * len(images), [paths]*len(images),[ids]*len(images))

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        accuracies = pool.map(process_test_image_tar, test_image_data_list)
    return accuracies

def compute_size_distance():
    num_cpus = multiprocessing.cpu_count()
    test_image_data_list = zip(images, labels, sizes, [images] * len(images), [labels] * len(images), [sizes] * len(images))

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        accuracies = pool.map(process_test_image_size, test_image_data_list)
    return accuracies

def compute_ncd_and_size():
    num_cpus = multiprocessing.cpu_count()
    cxs = [len(gzip.compress(img.tobytes(), compresslevel=C_LEVEL)) for img in images]

    test_image_data_list = zip(images, labels, cxs, sizes, [images] * len(images), [labels] * len(images),
                               [cxs] * len(images),[sizes] * len(images))

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        accuracies = pool.map(process_test_image_ncd_and_size, test_image_data_list)
    return accuracies




ds_rates = [512,256,128,64,32,16,8,4,2,1]
if __name__ == "__main__":
    for ds_rate in ds_rates:
        print("Down-sample rate is:", ds_rate)
        start_time = time.time()
        with open('/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_all_with_size.txt',
                  'r') as f:
            data = f.readlines()
            data = data[::ds_rate]
            paths, images, labels, sizes = load_images_and_labels(data)
            # List of unique class labels
            ids = np.arange(len(images))

        class_labels = np.unique(labels)
        total_samples = len(labels)
        # print("Total Number of Test Samples:", total_samples)
        # for label in class_labels:
        #     num_samples_in_class = np.sum(np.array(labels) == label)
        #     percentage_in_class = num_samples_in_class / total_samples * 100
        #     print(f"Class {label} Number of Samples: {num_samples_in_class}")

        accuracies = compute_ncd()
        # accuracies = compute_tar_ncd()
        # accuracies = compute_ncd_and_size()
        # accuracies = compute_size_distance()

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")

        total_acc = np.sum(accuracies) / len(data)
        print(f"Total Accuracy: {total_acc:.4f}")
        test = input()
        # class_accs=[]
        # for label in class_labels:
        #     index = np.array(labels) == label
        #     acc = np.mean(np.array(accuracies)[index])
        #     class_accs.append(acc)
        #     print(f"Class {label} Accuracy: {acc:.4f}")
        # print(f"Total MAP is: {np.mean(np.array(class_accs)):.4f}")


