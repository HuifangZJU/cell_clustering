import os

import tifffile
import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from math import cos, sin
from PIL import Image, ImageDraw

vizgen_path = '/media/huifang/data/vizgen/'

marker_genes_dict = {
    'Cancer3': ['FOXM1', 'BIRC5', 'PLK1', 'MYBL2'],
    'Cancer4': ['CDCA7', 'E2F1', 'FOXM1', 'EZH2', 'LRP6'],
    'Cancer1': ['CDH1', 'VCAM1', 'LAMB3', 'TAPBP', 'FZD7'],
    'Cancer2': ['LRP6', 'EPHB3', 'ERBB3', 'FGFR2', 'DKK1'],
    'Cancer5': ['PKM', 'SOD2', 'SPP1', 'CEBPB', 'AKT3'],
    'Dendritic': ['CD207', 'FCER1A', 'HLA-DRA', 'CSF1R', 'HLA-DRB1'],
    'Endothelial': ['COL4A1', 'PLVAP', 'PECAM1', 'ETS1', 'VWF'],
    'Fibroblast1': ['COL1A1', 'FN1', 'COL11A1', 'COL5A1', 'ACTA2'],
    'Fibroblast2': ['COL5A1', 'FN1', 'COL1A1', 'COL11A1'],
    'B/PlasmaB': ['CD79A', 'MZB1', 'POU2AF1', 'XBP1', 'FCRL5'],
    'TAM-C1Q': ['FCGR3A', 'CD14', 'CYBB', 'C1QC', 'CSF1R'],
    'TAM-SPP1': ['SPP1', 'TGFBI', 'MMP9', 'FCGR3A'],
    'T/NK': ['LYZ', 'TRAC', 'CD3E', 'CD2', 'PTPRC', 'ZAP70'],
    'Mast': ['KIT', 'BST2', 'CTSG', 'CD22', 'TGFB1'],
}


def get_pixel_limit(center_x, center_y, crop_size, img_height, img_width):
    x_min = center_x - crop_size
    y_min = center_y - crop_size
    x_max = center_x + crop_size
    y_max = center_y + crop_size
    if x_min < 0:
        x_min = 0
        x_max = 2 * crop_size
    if y_min < 0:
        y_min = 0
        y_max = 2 * crop_size
    if x_max > img_height:
        x_max = img_height
        x_min = img_height - 2 * crop_size
    if y_max > img_width:
        y_max = img_width
        y_min = img_width - 2 * crop_size

    return [x_min, x_max, y_min, y_max]


def normImg(img):
    # avoid zeros
    vmax = img.max()
    if vmax == 0:
        return img
    else:
        vmax = np.min([vmax, 30000])
        img[img > vmax] = vmax
        img = img * (254 / vmax)
        img[img > 254] = 254
        img = img.astype(np.uint8)
        return img


def get_pixel_location(boundary, tmpImage):
    bb = []
    for i in range(boundary.shape[1]):
        bb.append((boundary[0, i], boundary[1, i]))
    ImageDraw.Draw(tmpImage).polygon(bb, outline=255, fill=255)
    tmpImage = np.asarray(tmpImage)
    index = np.where(tmpImage == 255)
    # center_x = int(np.mean(index[0]))
    # center_y = int(np.mean(index[1]))
    center_x = int((np.max(index[0]) + np.min(index[0])) / 2)
    center_y = int((np.max(index[1]) + np.min(index[1])) / 2)
    return center_x, center_y, index


def get_square_crop_size(center_x, center_y, indexes):
    xy_range = [np.max(indexes[0]) - center_x, center_x - np.min(indexes[0]), np.max(indexes[1]) - center_y,
                center_y - np.min(indexes[1])]
    return np.max(xy_range)


def get_cp_location(cellpose_mask, center_x, center_y):
    # cellcnt = np.unique(cellpose_mask[center_x - 5:center_x  + 5, center_y - 5:center_y + 5])
    cellcnt = cellpose_mask[center_x, center_y]
    if cellcnt == 0:
        return [], [], []
    else:
        cellpixels = np.where(cellpose_mask == cellcnt)
        center_x = int(np.mean(cellpixels[0]))
        center_y = int(np.mean(cellpixels[1]))
    return center_x, center_y, cellpixels


def get_cell_image(center_x,center_y,indexes,img_height,img_width,fov_image):
    crop_size = get_square_crop_size(center_x, center_y, indexes)
    coordinates = get_pixel_limit(center_x, center_y, crop_size + 2, img_height, img_width)
    cell_image = fov_image[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
    return cell_image


def save_type_to_png(path_prefix,fov, inst_cell, centerx, centery, image):
    des_path = path_prefix + '/'
    image_name = 'fov' + str(fov) + '_cell' + str(inst_cell) + '_x' + str(centerx) + '_y' + str(centery) + '.png'
    os.makedirs(des_path, exist_ok=True)
    savepng(image, des_path + image_name)




def savepng(img, des_file):
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.convert('L')
    img.save(des_file)


def drawpolygon(boundary, tmpImage, color):
    bb = []
    for i in range(boundary.shape[1]):
        bb.append((boundary[0, i], boundary[1, i]))
    ImageDraw.Draw(tmpImage).polygon(bb, outline=color)


def get_fov_cell_boundaries(dataset_name, z_index_number, fov):
    base_path = vizgen_path + dataset_name + '/'
    cellBoundaries = h5py.File(base_path + 'cell_boundaries/feature_data_' + str(fov) + '.hdf5')
    if len(cellBoundaries.keys()) == 0:
        return [], []
    cellid = [id for id in cellBoundaries['featuredata']]
    z_index = 'zIndex_' + str(z_index_number)

    transformation_matrix = pd.read_csv(base_path + 'images/micron_to_mosaic_pixel_transform.csv',
                                        header=None, sep=' ').values
    currentCells = []
    for inst_cell in cellid:
        inst_cell = str(inst_cell)
        z_index = str(z_index)
        temp = cellBoundaries['featuredata'][inst_cell][z_index]['p_0']['coordinates'][0]
        boundaryPolygon = np.ones((temp.shape[0], temp.shape[1] + 1))
        boundaryPolygon[:, :-1] = temp
        transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1]
        currentCells.append(transformedBoundary)
    cellid = np.asarray(cellid)
    return currentCells, cellid


def calculate_elongation(m):
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
    return (x + y ** 0.5) / (x - y ** 0.5)

def extract_cell_info_from_line(line):
    line = line.split(' ')
    line = line[0].split('/')
    cellname = line[-1]
    fov, cell_inst_id, x, y = extract_cell_info_from_cellname(cellname)
    return fov,cell_inst_id,x,y

def extract_cell_info_from_cellname(cellname):
    cellname = cellname.split('_')
    print(cellname)
    fov = cellname[0][3:]
    cell_inst_id = cellname[1][4:]
    x = cellname[2][1:]
    y = cellname[3].split('.')[0][1:]
    return fov,cell_inst_id,x,y