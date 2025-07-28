import tifffile
import cv2
import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from math import cos, sin
from PIL import Image, ImageDraw


def drawpolygon(boundary, tmpImage):
    bb=[]
    for i in range(boundary.shape[1]):
        bb.append((boundary[0,i],boundary[1,i]))
    ImageDraw.Draw(tmpImage).polygon(bb,outline=255)

def normImg(img):
    vmax = 30000
    img[img > vmax] = vmax
    img = img * (254 / vmax)
    img[img > 254] = 254
    return img

# Paths to Data
base_path = '/media/huifang/data/vizgen/'
dataset_name = 'HumanBreastCancerPatient1/'
z_index_number = 0
fov = 27
dataset_suffix = 'z' + str(z_index_number) + '_fov' + str(fov)
tiffimagepath = base_path + dataset_name + 'images/'

# Load Cell Boundaries

# load transformation matrix
# This is used to transfer from global coordinates (the coordinates of transcripts
# and cells across the sample) to mosaic coordinates (used for the large TIF image files).
filename = base_path + dataset_name + 'images/micron_to_mosaic_pixel_transform.csv'
transformation_matrix = pd.read_csv(filename, header=None, sep=' ').values


filename = base_path + dataset_name + 'cell_boundaries/feature_data_' + str(fov) + '.hdf5'

#cellboundary only has one key:featuredata
cellBoundaries = h5py.File(filename)
meta_cell = pd.read_csv(base_path + dataset_name + 'cell_metadata' + '.csv', index_col=0)
meta_cell = meta_cell[meta_cell.fov == fov]
z_index = 'zIndex_' + str(z_index_number)

# collect boundaries in fov
currentCells = []
for inst_cell in meta_cell.index.tolist():
    # try:
    inst_cell = str(inst_cell)
    z_index = str(z_index)
    temp = cellBoundaries['featuredata'][inst_cell][z_index]['p_0']['coordinates'][0]
    boundaryPolygon = np.ones((temp.shape[0], temp.shape[1]+1))
    boundaryPolygon[:, :-1] = temp
    transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1]
    currentCells.append(transformedBoundary)
minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)
print(minCoord)
print(maxCoord)

# image stuff
# image stuff
fov_image = tifffile.imread(base_path + dataset_name + 'images/z' + str(z_index_number) + '/DAPI/mosaic_DAPI_' + dataset_suffix + '.tif')
print(base_path + dataset_name + 'images/z' + str(z_index_number) + '/DAPI/mosaic_DAPI_' + dataset_suffix + '.tif')
fov_image = normImg(fov_image)
fov_image = Image.fromarray(fov_image)
# plt.imshow(fov_image,cmap='gray')
print(len(currentCells))
for boundary in currentCells:
    boundary[0] = boundary[0] - minCoord[0]
    boundary[1]= boundary[1] - minCoord[1]
    drawpolygon(boundary,fov_image)

plt.imshow(fov_image)
plt.show()
test = input()

background_image = tifffile.imread(base_path + dataset_name + 'images/' + str(z_index_number) + '/Cellbound2/mosaic_Cellbound2_' + dataset_suffix + '.tif')
background_image = normImg(background_image)

# load transcript data - this takes about 25min
transcripts = pd.read_csv(base_path + dataset_name + 'detected_transcripts/' + str(z_index_number) + '/' + dataset_suffix + '.txt', header=None, index_col=0)
# 2,3,4: globalx, globaly,globalz
# 5,6:localx,localy
# 7: fov
# 8:gene
temp = transcripts[[2,3]].values
transcript_positions = np.ones((temp.shape[0], temp.shape[1]+1))
transcript_positions[:, :-1] = temp
# Transform coordinates to mosaic pixel coordinates
transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1]
transcripts.loc[:, 5] = transformed_positions[0, :]
transcripts.loc[:,6] = transformed_positions[1, :]

# plt.imshow(background_image)
gene_image = np.zeros(fov_image.shape)
gene_maps = transcripts.groupby(by=8)
for gene,data in gene_maps:
    local_x = np.asarray(data[5] - minCoord[0]).astype(int)
    local_y = np.asarray(data[6] - minCoord[1]).astype(int)
    gene_image[(local_x,local_y)] = gene_image[(local_x,local_y)]+1
    # plt.plot(local_x, local_y, 'r.')
    # plt.axis('equal')
    # plt.show()
gene_image = normImg(gene_image)



cell_seg_image = np.zeros(fov_image.shape)
######### show single cell images
crop_size = 128
for boundary in currentCells:
    tmpImage = fov_image.copy()
    tmbImage_bg = background_image.copy()
    tmpImage_gene = np.zeros(fov_image.shape)
    tmpImage = Image.fromarray(tmpImage)

    boundary = boundary.astype(int)
    boundary[0, :] = boundary[0, :] - minCoord[0]
    boundary[1, :] = boundary[1, :] - minCoord[1]
    boundary_min = np.min(boundary, axis=1)
    boundary_max = np.max(boundary, axis=1)

    bb=[]
    for i in range(boundary.shape[1]):
        bb.append((boundary[0,i],boundary[1,i]))
    ImageDraw.Draw(tmpImage).polygon(bb,outline=255,fill=255)
    tmpImage = np.asarray(tmpImage)
    index = np.where(tmpImage ==255)
    boundary_min_x = np.min(index[0])
    boundary_min_y = np.min(index[1])

    boundary_max_x = np.max(index[0])
    boundary_max_y = np.max(index[1])

    center_x = int((boundary_max_x + boundary_min_x)/2)
    center_y = int((boundary_max_y + boundary_min_y)/2)
    x_min = center_x - crop_size
    y_min = center_y - crop_size
    x_max = center_x + crop_size
    y_max = center_y + crop_size
    if x_min<0:
        x_off_set = -x_min
        x_min = 0
        x_max = x_max + x_off_set
    if y_min<0:
        y_off_set = -y_min
        y_min = 0
        y_max = y_max + y_off_set

    cell_seg_image[index] = fov_image[index]
    #show single cells
    tmbImage_bg[index] = fov_image[index]
    tmpImage_gene[index] = gene_image[index]
    crop_img1 = fov_image[x_min:x_max,y_min:y_max]
    crop_img2 = tmbImage_bg[x_min:x_max, y_min:y_max]
    crop_img3 = tmpImage_gene[x_min:x_max, y_min:y_max]
    f,a = plt.subplots(1,3)
    a[0].imshow(crop_img1,vmin=0,vmax=255,cmap='gray')
    a[1].imshow(crop_img2,vmin=0,vmax=255,cmap='gray')
    a[2].imshow(crop_img3,cmap='gray')
    plt.show()

f,a = plt.subplots(2,2)
a[0,0].imshow(fov_image)
a[0,1].imshow(background_image)
a[1,0].imshow(gene_image)
a[1,1].imshow(cell_seg_image)
plt.show()

# segmentation data
#########################
# polygon_data = []
# for inst_index in range(len(currentCells)):
#   inst_cell = currentCells[inst_index]
#   df_poly_z = pd.DataFrame(inst_cell).transpose()
#   df_poly_z.columns.tolist()
#   inst_name = meta_cell.iloc[inst_index].name
#   inst_poly = {'coordinates': df_poly_z.values.tolist(), 'name': inst_name}
#   polygon_data.append(inst_poly)
#
# df_obs = transcripts[['local_x', 'local_y', 'global_z', 'gene']]
# df_obs.columns = ['x', 'y', 'z', 'name']
# scatter_data = df_obs.to_dict('records')
# test = input()

# load field of view transcript data
#0: cell id?
# 2,3,4: globalx, globaly,globalz
# 5,6:localx,localy
# 7: fov
# 8:gene
# transcripts = pd.read_csv(base_path + dataset_name + 'detected_transcripts/' + str(z_index_number) + '/' + dataset_suffix + '.txt', header=None, index_col=0)


# transpose to current field of view coordinate
# temp = transcripts[[2,3]].values
# transcript_positions = np.ones((temp.shape[0], temp.shape[1]+1))
# transcript_positions[:, :-1] = temp
# # Transform coordinates to mosaic pixel coordinates
# transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1]
# transcripts.loc[:, 5] = transformed_positions[0, :]
# transcripts.loc[:,6] = transformed_positions[1, :]
