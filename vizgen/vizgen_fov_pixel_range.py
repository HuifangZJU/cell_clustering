import h5py
import numpy as np
import pandas as pd
import os

# Select data
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 0

# Paths to Data
base_path = '/media/huifang/data/vizgen/' + dataset_name +'/'
transformation_matrix = pd.read_csv(base_path + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values
# Prepare to Save
fov_range=[]
fov_files = os.listdir(base_path + 'cell_boundaries')
print("Totally "+ str(len(fov_files))+" fovs.")
for i in range(0,len(fov_files)):
    if np.mod(i,10) == 0:
        print("fov " + str(i) + "...")
    try:
        cellBoundaries = h5py.File(base_path + 'cell_boundaries/feature_data_' + str(i)+'.hdf5')
    except FileNotFoundError:
        print('loss fov '+str(i) + '!')
        continue
    if len(cellBoundaries.keys())==0:
        continue
    cellid = [id for id in cellBoundaries['featuredata']]
    z_index = 'zIndex_' + str(z_index_number)
    # collect boundaries in fov
    currentCells = []
    for inst_cell in cellid:
        # try:
        inst_cell = str(inst_cell)
        z_index = str(z_index)
        temp = cellBoundaries['featuredata'][inst_cell][z_index]['p_0']['coordinates'][0]
        boundaryPolygon = np.ones((temp.shape[0], temp.shape[1]+1))
        boundaryPolygon[:, :-1] = temp
        transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1]
        currentCells.append(transformedBoundary)
    if len(currentCells)==0:
        continue
    minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
    maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)
    fov_range.append([i, minCoord[1],maxCoord[1],minCoord[0],maxCoord[0]])
# assert len(fov_files) == len(fov_range)
np.save(base_path +'fov_range.npy', np.array(fov_range))
print('done')