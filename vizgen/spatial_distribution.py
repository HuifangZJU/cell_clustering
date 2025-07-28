import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from vizgen_util import *
from matplotlib.path import Path
from matplotlib.patches import Rectangle


def extract_cell_info_from_line(line):
    line = line.split(' ')
    line = line[0].split('/')
    cellname = line[-1]
    cellname = cellname.split('_')
    fov = cellname[0][3:]
    cell_inst_id = cellname[2].split('.')[0]
    return fov,cell_inst_id

def get_cell_ids(imagelist):
    f = open(imagelist)
    lines = f.readlines()
    cells = []
    for line in lines:
        fov, cell_inst_id = extract_cell_info_from_line(line)
        cells.append(cell_inst_id)
    return cells

def attend_coordinates_to_images(coordinate, lines):
    # Check if the lengths match
    if coordinate.shape[1] != len(lines):
        raise ValueError("Length mismatch between coordinate and lines.")

    new_lines = []
    for coord, line in zip(coordinate.T, lines):
        line = line.rstrip()
        new_line = line + ' ' + ' '.join(str(coord_val) for coord_val in coord)
        new_lines.append(new_line)
    return new_lines

def update_cluster_ids(ad_viz,order,labels):
    original_clust = ad_viz.obs['leiden']
    label_gt = np.ones(original_clust.shape)
    for c, l in zip(order, labels):
        temp = original_clust.isin([c])
        label_gt[temp] = l
    label_gt = pd.Categorical(label_gt)
    ad_viz.obs['leiden'] = label_gt
    return ad_viz

# -----------------  Select Data  ----------------- #
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 0
# -----------------  Paths to Data  ----------------- #
base_path = '/media/huifang/data/vizgen/' + dataset_name + '/'
z_index = 'zIndex_' + str(z_index_number)

# -----------------  Read cluster result  ----------------- #
ad_viz = sc.read_h5ad('/media/huifang/data/vizgen/HumanBreastCancerPatient1/gene_cluster_image/z0_all_data_res0.1_cluster_mata.hdf5')
order = ['0','3','4','2']
labels = ['0','1','2','2']
theta = np.deg2rad(-90)
rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
ad_viz.obs[['center_x', 'center_y']] = ad_viz.obs[['center_x', 'center_y']].dot(rot)

ad_viz = ad_viz[ad_viz.obs['leiden'].isin(order),:]
ad_viz = update_cluster_ids(ad_viz,order,labels)
coordinate = np.asarray([ad_viz.obs['center_y'],ad_viz.obs['center_x']])
ad_viz.obsm["spatial"] = coordinate.transpose()
sc.pl.embedding(ad_viz, basis="spatial", color="leiden",size=20)
# -----------------  Read image list and select cells  ----------------- #
imagelist='/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1.txt'

# prediction =pd.Categorical(np.load('/home/huifang/workspace/experiment/scan/vizgen/original/scan/train_predictions.npy'))
# prediction =pd.Categorical(np.load('/home/huifang/workspace/code/TokenCut/ncut.npy'))

cells = get_cell_ids(imagelist)
ad_viz = ad_viz[cells]
gt =ad_viz.obs['leiden']


# #----------------- save coordinate --------------------#
# f = open(imagelist)
# lines = f.readlines()
# new_list = attend_coordinates_to_images(coordinate,lines)
# np.savetxt('/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_train_with_position.txt', new_list, fmt='%s', delimiter='')

# #----------------- calculate the accuracy --------------------#
# matches = sum(elem1 == elem2 for elem1, elem2 in zip(gt, prediction))
# percentage = matches / len(gt) * 100
# print(percentage)
# print('3...')
# #----------------- show the spatial plot --------------------#
sc.pl.embedding(ad_viz, basis="spatial", color="leiden",size=30)
# ad_viz.obs['leiden'] = prediction
# sc.pl.embedding(ad_viz, basis="spatial", color="leiden",size=20)
