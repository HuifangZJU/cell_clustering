import time
import scanpy as sc
from vizgen_util import *
from vizgen_gene_utils import visualize_dotplot,visualize_spatial_distribution
import concurrent.futures
import os

cellposeid=[]

def get_cellpose_data(ad_viz):
    cellposeid = np.load(base_path+'cellposeid.npy')
    # fovs = cellposeid[:0]
    cellid = cellposeid[:,1]
    ad_viz = ad_viz[cellid]
    return ad_viz

def get_patch_data():
    xmin = min(centerx) + i * x_patch_size
    xmax = min(centerx) + (i + 1) * x_patch_size
    ymin = min(centery) + j * y_patch_size
    ymax = min(centery) + (j + 1) * y_patch_size

    xid = (centerx <= xmax) & (centerx >= xmin)
    yid = (centery <= ymax) & (centery >= ymin)
    xyid = xid & yid

    ad_viz_temp = ad_viz[xyid]
    ad_viz_temp = ad_viz_temp[ad_viz_temp.obs['leiden'].isin(clusts), :]
    return ad_viz_temp



def save_patch_image_list(ad_viz_temp,save_file,labels,image_type):
    temp_clusts = ad_viz_temp.obs['leiden']
    coordinate_y = ad_viz_temp.obs['center_y']
    coordinate_x =ad_viz_temp.obs['center_x']


    image_path_template = cluster_image_path + cluster_prefix + '/' + image_type + '/cluster_{}/'

    f=open(save_file,'w')
    for clust,label in zip(clusts,labels):
        image_path = image_path_template.format(clust)
        files = os.listdir(image_path)
        temp_clust_id = temp_clusts.str.contains(clust)
        temp_clust_id = temp_clust_id.index[temp_clust_id]
        for cellid in temp_clust_id:
            image_name = [file for file in files if cellid in file]
            image_name = image_name[0]
            x = coordinate_x[cellid]
            y = coordinate_y[cellid]
            if os.path.exists(image_path + image_name):
                f.write(image_path + image_name + ' ' + str(label) + ' ' + str(int(x)) + ' ' + str(int(y)) + '\n')
            else:
                print(image_path + image_name)
                return


def process_fov(fov):
    currentCells, cellid = get_fov_cell_boundaries(dataset_name, z_index_number, fov)
    if not currentCells:
        print('fov ' + str(fov) + ' has no cells! ')
        return
    fov_suffix = 'z' + str(z_index_number) + '_fov' + str(fov)
    fov_image = tifffile.imread(tiffimagepath+'/DAPI/mosaic_DAPI_' + fov_suffix + '.tif')
    cellpose_mask = plt.imread(tiffimagepath+'/DAPI/masks//mosaic_DAPI_' + fov_suffix + '_cp_masks.png')
    fov_image = normImg(fov_image)
    minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)

    for clust in ad_viz.obs['leiden'].cat.categories:
        cluster_meta_cells = ad_viz[ad_viz.obs['leiden'].isin([clust]), :]
        cluster_fov_meta_cells = cluster_meta_cells[cluster_meta_cells.obs['fov'].isin([fov]), :]
        cluster_fov_meta_cells = cluster_fov_meta_cells.obs

        for inst_cell in cluster_fov_meta_cells.index.tolist():
            id = np.where(cellid == inst_cell)
            boundary = currentCells[id[0][0]]
            boundary = boundary.astype(int)
            boundary[0, :] = boundary[0, :] - minCoord[0]
            boundary[1, :] = boundary[1, :] - minCoord[1]

            vg_center_x, vg_center_y, vg_indexes = get_pixel_location(boundary, Image.fromarray(fov_image))

            center_x, center_y, indexes = get_cp_location(cellpose_mask, int(np.mean(vg_indexes[0])), int(np.mean(vg_indexes[1])))
            if indexes:
                cellposeid.append([fov,inst_cell,center_x,center_y])


    print('fov ' + str(fov) + ' done')

def run():
    # Set the number of concurrent workers (threads)
    num_workers = 16
    start_time = time.time()
    # Use ThreadPoolExecutor to parallelize the loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        fov_indices = range(fov_cnt)
        executor.map(process_fov, fov_indices)

    end_time = time.time()
    print(end_time - start_time)
    np.save(base_path + 'cellposeid.npy', np.array(cellposeid))





# -----------------  Select Data  ----------------- #
dataset_name = 'HumanColonCancerPatient1'
z_index_number = 0
# -----------------  Paths to Data  ----------------- #
base_path = '/media/huifang/data/vizgen/' + dataset_name + '/'
z_index = 'zIndex_' + str(z_index_number)
# image stuff
tiffimagepath = base_path + 'images/z' + str(z_index_number)+'_fov_images'+'/'
path_to_save = base_path + 'gene_cluster_image/'
cluster_image_path = "/media/huifang/data/vizgen/HumanColonCancerPatient1/gene_clustered_images/"

# -----------------  Read cluster result  ----------------- #
cluster_prefix = 'z' + str(z_index_number) + '_ds1_res0.1'
print('Reading cluster data...')
# ad_viz = sc.read_h5ad('/media/huifang/data/vizgen/HumanBreastCancerPatient1/gene_cluster_result/z0_all_data_res0.1_cluster_mata.hdf5')
ad_viz = sc.read_h5ad('/media/huifang/data/vizgen/HumanColonCancerPatient1/gene_cluster_result/z0_ds1_res0.1/cluster_mata.hdf5')
# visualize_dotplot(ad_viz)

# -----------------  get cellpose ids ----------------- #
fov_cnt = len(os.listdir(base_path + 'cell_boundaries'))
print('totally ' + str(fov_cnt)+ ' fovs.')
# run()
# -----------------  select and visualization patches ----------------- #
clusts = ['0','1','2']
labels=  ['0','1','2']
# visualize_dotplot(ad_viz)
# visualize_dotplot(ad_viz,order)
# visualize_spatial_distribution(ad_viz,order,10)

imagetype='cellpose'
if imagetype == 'cellpose':
    ad_viz = get_cellpose_data(ad_viz)


centery=ad_viz.obs['center_y']
centerx=ad_viz.obs['center_x']

xstep = 10
ystep = 8
x_range = max(centerx) - min(centerx)
y_range = max(centery) - min(centery)

x_patch_size = x_range / xstep
y_patch_size = y_range / ystep

xmins = [min(centerx) + i * x_patch_size for i in range(xstep)]
xmaxs = [min(centerx) + (i + 1) * x_patch_size for i in range(xstep)]
ymins = [min(centery) + j * y_patch_size for j in range(ystep)]
ymaxs = [min(centery) + (j + 1) * y_patch_size for j in range(ystep)]

for i, (xmin, xmax) in enumerate(zip(xmins, xmaxs)):
    for j, (ymin, ymax) in enumerate(zip(ymins, ymaxs)):
        print(f"Total Patches: {xstep * ystep} --- Current Patch: {i * ystep + j + 1}")

        xid = (centerx <= xmax) & (centerx >= xmin)
        yid = (centery <= ymax) & (centery >= ymin)
        xyid = xid & yid

        ad_viz_temp = ad_viz[xyid]
        ad_viz_temp = ad_viz_temp[ad_viz_temp.obs['leiden'].isin(clusts), :]
        gt = ad_viz_temp.obs['leiden']
        counts = gt.value_counts()
        if len(counts) == 0:
            print('No cells!')
            continue
        # if max(counts)/min(counts)>30:
        #     continue
        if sum(counts)<100:
            continue
        print(counts)
        print(sum(counts))
        # visualize_spatial_distribution(ad_viz_temp, clusts, 30)

        # visualize_spatial_distribution(ad_viz_temp,clusts,30,saveflag=True, savename='colon_'+imagetype+'_'+str(i * ystep + j + 1))

        save_txt ='/home/huifang/workspace/data/imagelists/vizgen_patches/'+imagetype+ '/'+ imagetype +'_colon_patch'+str(i * ystep + j + 1)+'.txt'
        start_time = time.time()
        save_patch_image_list(ad_viz_temp,save_txt,labels,imagetype)

        end_time = time.time()
        # print(end_time-start_time)
        print('done')




