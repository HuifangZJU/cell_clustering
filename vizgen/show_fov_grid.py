from vizgen_util import *
from matplotlib.patches import Rectangle
# Select data
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 0
image_type = 'DAPI'
# Paths to Data
base_path = '/media/huifang/data/vizgen/' + dataset_name +'/'
tissue_image_path = '/home/huifang/Pictures/experiment/pipeline/breast.png'
tissue_image = plt.imread(tissue_image_path)
scalex = tissue_image.shape[0]/94805
scaley = tissue_image.shape[1]/110485
f, a = plt.subplots()
a.imshow(tissue_image)

fov_range = np.load(base_path +'fov_range.npy')
for i in range(0,fov_range.shape[0]):
    Coord = fov_range[i,1:]
    a.add_patch(
            Rectangle((Coord[2]*scaley,Coord[0]*scalex), (Coord[3]-Coord[2])*scaley,(Coord[1]-Coord[0])*scalex,
                      edgecolor='gray', fill=False, linewidth=0.5))
plt.show()