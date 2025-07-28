"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import gzip
import cv2
from scipy.stats import mode
import multiprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


COM_LEVEL = 5
def compute_distance(img1, cx1, img2, cx2):
    img12 = np.concatenate((img1, img2), axis=0)
    cx12 = len(gzip.compress(img12.tobytes(), compresslevel=COM_LEVEL))
    ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)

    return ncd

def calculate_accuracy(top_k_class, label1):
    return np.sum(top_k_class == label1) / len(top_k_class)

def load_images_and_labels(data):
    imgs = [cv2.resize(cv2.imread(line.strip().split(' ')[0], cv2.IMREAD_GRAYSCALE), (96, 96)) for line in data]
    # imgs = [cv2.imread(line.strip().split(' ')[0], cv2.IMREAD_GRAYSCALE) for line in data]
    labels = [line.strip().split(' ')[1] for line in data]
    return imgs, labels

def load_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
        print(len(data))
        # data = data[::1000] 0.717948717948718  8-class:0.4588744588744589  8-class-bbox:0.4957805907172996
        # data = data[::100] 0.7481767481767482   8-class:0.48406676783004554  8-class-bbox:0.5012674271229405
        data = data[::1000]
    images,labels = load_images_and_labels(data)
    cxs = [len(gzip.compress(img.tobytes(), compresslevel=COM_LEVEL)) for img in images]
    return images,labels,cxs


def process_test_image(test_image_data):
    test_img, test_label, test_cx, train_images, train_labels, train_cxs = test_image_data
    distance_from_i = [
        [compute_distance(test_img, test_cx, train_img, train_cx), train_label]
        for train_img, train_label, train_cx in zip(train_images, train_labels, train_cxs)
    ]
    distance_from_i = np.array(distance_from_i)
    top_k_class = distance_from_i[np.argsort(distance_from_i[:, 0])][:5, 1]

    # Convert labels to unique integers to make the array numeric
    label_to_int = {label: i for i, label in enumerate(np.unique(top_k_class))}
    top_k_class_int = np.array([label_to_int[label] for label in top_k_class])

    # Handle NaN values in top_k_class_int using numpy.nan_to_num
    top_k_class_no_nan = np.nan_to_num(top_k_class_int, nan=len(label_to_int))

    # Compute the mode (most frequent value) using set and .count
    mode_counts = [(val, np.count_nonzero(top_k_class_no_nan == val)) for val in np.unique(top_k_class_no_nan)]
    predict_class = max(mode_counts, key=lambda x: x[1])[0]

    # Convert back the predicted integer label to the original label
    predict_class_label = [label for label, i in label_to_int.items() if i == predict_class][0]

    return predict_class_label, test_label


def calculate_per_class_accuracy(results, class_labels):
    # Create dictionaries to store counts of correct predictions and total instances for each class
    correct_counts = {label: 0 for label in class_labels}
    total_counts = {label: 0 for label in class_labels}

    # Update counts based on the results
    for predict_class, test_label in results:
        total_counts[test_label] += 1
        if predict_class == test_label:
            correct_counts[test_label] += 1

    # Calculate per-class accuracy
    per_class_accuracy = {
        label: correct_counts[label] / total_counts[label] if total_counts[label] > 0 else 0.0
        for label in class_labels
    }

    return per_class_accuracy

def plot_confusion_matrix(conf_matrix, class_labels, output_dir):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix = conf_matrix / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def save_results(output_dir, total_samples, per_class_accuracy, class_labels, right, test_images, test_labels, conf_matrix,elapsed_time):
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'output.txt')
    # Plot confusion matrix
    class_names = ['a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'shsy5y', 'skbr3', 'skov3']
    # class_names = ['bv2', 'huh7','shsy5y', 'skbr3']
    plot_confusion_matrix(conf_matrix, class_names,output_dir)

    with open(output_file, 'w') as f:
        f.write(f"Total Number of Test Samples: {total_samples}\n\n")

        for label in class_labels:
            num_samples_in_class = np.sum(np.array(test_labels) == label)
            percentage_in_class = num_samples_in_class / total_samples * 100

            f.write(f"Class {label} Accuracy: {per_class_accuracy[label]:.4f}\n")
            f.write(f"Class {label} Number of Samples: {num_samples_in_class}\n")
            f.write(f"Class {label} Percentage in Test Sample: {percentage_in_class:.2f}%\n\n")

    # Print total accuracy
    total_acc = right / len(test_images)
    with open(output_file, 'a') as f:
        f.write(f"Total Accuracy: {total_acc}\n")
        f.write(f"Total running time: {elapsed_time:.2f} seconds\n")


base_dir = '/media/huifang/data/livecell/LIVECell_dataset_2021/single_cell_bbox/'
def main():
    start_time = time.time()

    test_images, test_labels, test_cxs = load_data(base_dir+'test.txt')
    train_images, train_labels, train_cxs = load_data(base_dir+'train.txt')

    test_image_data_list = zip(test_images, test_labels, test_cxs, [train_images] * len(test_images), [train_labels] * len(test_images), [train_cxs] * len(test_images))

    # Number of CPU cores to use (can be adjusted based on your system)
    num_cpus = multiprocessing.cpu_count()

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Process the test images using multiple CPUs in parallel
        results = pool.map(process_test_image, test_image_data_list)
    # List of unique class labels
    class_labels = np.unique(test_labels)

    # Calculate per-class accuracy
    per_class_accuracy = calculate_per_class_accuracy(results, class_labels)

    total_samples = len(test_labels)
    print("Total Number of Test Samples:", total_samples)
    total_samples_train = len(train_labels)
    print("Total Number of Train Samples:", total_samples_train)

    # Print per-class accuracy
    for label in class_labels:
        num_samples_in_class = np.sum(np.array(test_labels) == label)
        percentage_in_class = num_samples_in_class / total_samples * 100

        print(f"Class {label} Accuracy: {per_class_accuracy[label]:.4f}")
        print(f"Class {label} Number of Samples: {num_samples_in_class}")
        print(f"Class {label} Percentage in Test Sample: {percentage_in_class:.2f}%")
    right = sum(1 for predict_class, test_label in results if predict_class == test_label)

    # Print total accuracy
    total_acc = right / len(test_images)
    print("Total Accuracy:", total_acc)

    # Compute confusion matrix
    true_labels = [test_label for predict_class, test_label in results]
    predicted_labels = [predict_class for predict_class, test_label in results]
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")

    save_results(base_dir+'result1000-8class-compression5/', total_samples, per_class_accuracy, class_labels, right, test_images, test_labels,conf_matrix,elapsed_time)

if __name__ == "__main__":
    main()



