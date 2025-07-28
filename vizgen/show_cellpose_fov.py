import numpy as np
from matplotlib import pyplot as plt
import tifffile
import os
import cv2


def normImg(img):
    vmax = 30000
    img[img > vmax] = vmax
    img = img * (254 / vmax)
    img[img > 254] = 254
    return img

image_path = "/media/huifang/data/vizgen/HumanBreastCancerPatient1/images/z0_fov_images/DAPI/"
outline_path="/media/huifang/data/vizgen/HumanBreastCancerPatient1/images/z0_fov_images/DAPI/txt_outlines/"

outline_names = os.listdir(outline_path)
for outline_name in outline_names:
    # outline_name="z0_fov27_cell0_neighbor_cp_outlines.txt"
    local_image_path = image_path + outline_name[:-16] +'.tif'
    print(local_image_path)
    local_image = plt.imread(local_image_path)
    local_image = normImg(local_image)
    # local_range = outline_name.split("_")
    # v_start_pixel = int(local_range[1])
    # u_start_pixel = int(local_range[2])
    # v_end_pixel = int(local_range[3])
    # u_end_pixel = int(local_range[4])
    # local_image = tiff_image[v_start_pixel:v_end_pixel,u_start_pixel:u_end_pixel]
    
    #f,a = plt.subplots(1,2)
    #a[0].imshow(local_image)
    #a[1].imshow(local_image)
    
    plt.figure(figsize=(16, 12)) 
    plt.imshow(local_image,cmap='gray')
    outline_file = open(outline_path+outline_name,'r')
    outlines = outline_file.readlines()
    for line in outlines:
        location = line.split(',')
        location = np.array(location).astype(int)
        # v_pixels = location[::2]-v_start_pixel
        # u_pixels = location[1::2]-u_start_pixel
        v_pixels = location[::4]
        u_pixels = location[1::4]
        #a[1].scatter(v_pixels,u_pixels,s=1.5,c='w')
        #a[1].scatter(v_pixels,u_pixels,s=1.5)
        plt.scatter(v_pixels,u_pixels,s=10)
    plt.show()

