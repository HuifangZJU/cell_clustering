"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
import gzip
import cv2
from PIL import  Image
import io

def compute_distance(img1, cx1, img2, cx2):
    img12 = Image.new('L', (img1.width + img2.width, max(img1.height, img2.height)))
    img12.paste(img1, (0, 0))
    img12.paste(img2, (img1.width, 0))
    cx12 = compute_single_image_compressed_length(img12)
    ncd = (cx12 - min(cx1, cx2)) / max(cx1, cx2)

    return ncd

def calculate_accuracy(top_k_class, label1):
    return np.sum(top_k_class == label1) / len(top_k_class)

def load_images_and_labels(data):
    imgs = [Image.open(line.strip().split(' ')[0]).resize((96, 96)) for line in data]
    labels = [line.strip().split(' ')[1] for line in data]
    return imgs, labels

def compute_single_image_compressed_length(image):
    # Compress the image and get the compressed data
    compressed_data = io.BytesIO()

    compression_level = 50
    image.save(compressed_data, format='JPEG', optimize=True, quality=compression_level)

    # Get the compressed data size directly without closing the BytesIO object
    compressed_bytes = compressed_data.getbuffer()
    return len(compressed_bytes)

def get_image_cxs(images):
    cxs = []
    for image in images:
        compressed_bytes = compute_single_image_compressed_length(image)
        cxs.append(compressed_bytes)
    return cxs


with open('/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_train_with_position.txt', 'r') as f:
    data = f.readlines()
    data = data[::512]
images,labels = load_images_and_labels(data)

cxs =get_image_cxs(images)
total_acc=0
for i, img1 in enumerate(images):
    print(i)
    label1 = labels[i]
    cx1 = cxs[i]

    distance_from_i = [
        [compute_distance(img1, cx1, img2, cxs[j]), labels[j]]
        for j, img2 in enumerate(images) if j != i
    ]
    distance_from_i = np.array(distance_from_i)
    top_k_class = distance_from_i[np.argsort(distance_from_i[:, 0])][:5, 1]
    acc = calculate_accuracy(top_k_class, label1)
    total_acc += acc

total_acc = total_acc/len(data)
print(total_acc)
