from vizgen_util import *

# Select Data
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 5

# Paths to Data
base_path = '/media/huifang/data/vizgen/' + dataset_name + '/'
tiffimagepath = base_path + 'images/'
z_index = 'zIndex_' + str(z_index_number)
transformation_matrix = pd.read_csv(base_path + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values

fov_cnt = len(os.listdir(base_path + 'cell_boundaries'))
for fov in range(20,fov_cnt,10):
    print(fov)
    cellBoundaries = h5py.File(base_path + 'cell_boundaries/feature_data_' + str(fov) + '.hdf5')
    if len(cellBoundaries.keys()) == 0:
        continue
    cellid = [id for id in cellBoundaries['featuredata']]
    z_index = 'zIndex_' + str(z_index_number)
    # collect boundaries in fov
    currentCells = []
    for inst_cell in cellid:
        inst_cell = str(inst_cell)
        z_index = str(z_index)
        temp = cellBoundaries['featuredata'][inst_cell][z_index]['p_0']['coordinates'][0]
        boundaryPolygon = np.ones((temp.shape[0], temp.shape[1]+1))
        boundaryPolygon[:, :-1] = temp
        transformedBoundary = np.matmul(transformation_matrix, np.transpose(boundaryPolygon))[:-1]
        currentCells.append(transformedBoundary)
    minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
    maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)

    # image stuff
    dataset_suffix = 'z' + str(z_index_number) + '_fov' + str(fov)
    fov_image = tifffile.imread(base_path + 'images/z' + str(z_index_number)+ '_fov_images/DAPI/mosaic_DAPI_' + dataset_suffix + '.tif')
    # fov_image = tifffile.imread(base_path + 'images/z' + str(z_index_number)+ '_fov_images/Cellbound1/mosaic_Cellbound1_'+ dataset_suffix + '.tif')
    fov_image = normImg(fov_image)
    fov_image = Image.fromarray(fov_image)
    for boundary in currentCells:
        boundary[0] = boundary[0] - minCoord[0]
        boundary[1]= boundary[1] - minCoord[1]
        drawpolygon(boundary, fov_image,color=255)

    # # Save to File
    # des_dir = tiffimagepath + 'z' + str(z_index_number) + '/seg/'
    # os.makedirs(des_dir, exist_ok=True)
    # des_file = des_dir + 'seg_z' + str(z_index_number) + '_fov' + str(fov) + '.png'
    # fov_image.save(des_file)
    # print('done')

    # Visualization
    plt.imshow(fov_image)
    plt.show()

