from vizgen_util import *

# Select data
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 5 
image_type = 'DAPI'
# Paths to Data
base_path = '/media/huifang/data/vizgen/' + dataset_name +'/'
print('Reading tiffimage...')
tiffimagepath = '/media/huifang/data/vizgen/'+ dataset_name +'/images/mosaic_' + image_type + '_z'+ str(z_index_number) +'.tif'
tiff_image = tifffile.imread(tiffimagepath)
fov_range = np.load(base_path +'fov_range.npy')
for i in range(0,fov_range.shape[0]):
    Coord = fov_range[i,:]
    if len(Coord) == 4:
        fov = i
        tmpImage = tiff_image[Coord[0]:Coord[1],Coord[2]:Coord[3]]
    else:
        fov = Coord[0]
        tmpImage = tiff_image[Coord[1]:Coord[2], Coord[3]:Coord[4]]


    des_dir = base_path + 'images/z'+ str(z_index_number) +'_fov_images/'+ image_type +'/'
    os.makedirs(des_dir, exist_ok=True)
    des_file = des_dir + 'mosaic_'+ image_type+'_z' + str(z_index_number) + '_fov' + str(fov) + '.tif'
    tifffile.imwrite(des_file,tmpImage)
    print('fov ' + str(fov) +' done')

