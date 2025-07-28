
from vizgen_util import *
import os


# Select Data
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 0
# single cell crop size
crop_size = 128

# Paths to Data
base_path = '/media/huifang/data/vizgen/' + dataset_name +'/'
transformation_matrix = pd.read_csv(base_path + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values
tiffimagepath = base_path + 'images/z'+ str(z_index_number) + '_fov_images/'
cellimagepath = base_path + 'cropped_cells/'
os.makedirs(cellimagepath, exist_ok=True)

total_cell = 0
fov_cnt = len(os.listdir(base_path + 'cell_boundaries'))
for fov in range(0,fov_cnt):
    cellBoundaries = h5py.File(base_path + 'cell_boundaries/feature_data_' + str(fov) + '.hdf5')
    cellid = [id for id in cellBoundaries['featuredata']]
    z_index = 'zIndex_' + str(z_index_number)
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
    #get fov crop image pixel offset
    minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
    maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)

    # image stuff
    dataset_suffix = 'z' + str(z_index_number) + '_fov' + str(fov)
    fov_image = tifffile.imread(tiffimagepath+'/DAPI/mosaic_DAPI_' + dataset_suffix + '.tif')
    img_height, img_width = fov_image.shape
    fov_image = normImg(fov_image)
    ######### show single cell images####
    cell_id = 0
    for boundary in currentCells:
        tmpImage = fov_image.copy()
        tmpImage = Image.fromarray(tmpImage)

        boundary = boundary.astype(int)
        boundary[0, :] = boundary[0, :] - minCoord[0]
        boundary[1, :] = boundary[1, :] - minCoord[1]
        boundary_min = np.min(boundary, axis=1)
        boundary_max = np.max(boundary, axis=1)

        center_x, center_y,index = get_pixel_location(boundary, tmpImage)
        x_min, x_max, y_min, y_max = get_pixel_limit(center_x,center_y,crop_size,img_height,img_width)

        # show single cells
        cell_image = np.zeros(fov_image.shape)
        cell_image[index] = fov_image[index]
        neighbor_img = fov_image[x_min:x_max,y_min:y_max]
        cell_image = cell_image[x_min:x_max, y_min:y_max]

        # Save to file
        # des_file = cellimagepath +'z'+ str(z_index_number)+ '_fov' + str(fov) + '_cell' + str(cell_id) + '_neighbor'+ '.png'
        # savepng(neighbor_img,des_file)
        # des_file = cellimagepath + 'z'+ str(z_index_number)+ '_fov' + str(fov) + '_cell'+ str(cell_id)+ '_image'  + '.png'
        # savepng(cell_image,des_file)
        # cell_id += 1
        # total_cell +=1

        # Visualization
        f,a = plt.subplots(1,2)
        a[0].imshow(cell_image,vmin=0,vmax=255)
        a[1].imshow(neighbor_img,vmin=0,vmax=255)
        plt.show()
    print('fov ' + str(fov) + ' is done, ' + str(cell_id)+' cells.')
print('all done, totally ' + str(total_cell) + ' cells.')
