import scanpy as sc
from vizgen_util import *
import time
import concurrent.futures


def process_fov(fov):
    currentCells, cellid = get_fov_cell_boundaries(dataset_name, z_index_number, fov)
    if not currentCells:
        print('fov ' + str(fov) + ' has no cells! ')
        return

    fov_suffix = 'z' + str(z_index_number) + '_fov' + str(fov)
    fov_image = tifffile.imread(tiffimagepath + '/DAPI/mosaic_DAPI_' + fov_suffix + '.tif')
    cellpose_mask = plt.imread(tiffimagepath + '/DAPI/masks//mosaic_DAPI_' + fov_suffix + '_cp_masks.png')
    # seg_image = plt.imread(tiffimagepath + '/seg/' + 'seg_' + fov_suffix + '.png')
    fov_image = normImg(fov_image)
    img_height, img_width = fov_image.shape
    minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
    for clust in ad_viz.obs['leiden'].cat.categories:
        # for clust in order:
        # print('saving clust '+ str(clust) +'...')
        cluster_meta_cells = ad_viz[ad_viz.obs['leiden'].isin([clust]), :]
        cluster_fov_meta_cells = cluster_meta_cells[cluster_meta_cells.obs['fov'].isin([fov]), :]
        cluster_fov_meta_cells = cluster_fov_meta_cells.obs

        for inst_cell in cluster_fov_meta_cells.index.tolist():
            CP_FLAG = True
            id = np.where(cellid == inst_cell)
            boundary = currentCells[id[0][0]]
            boundary = boundary.astype(int)
            boundary[0, :] = boundary[0, :] - minCoord[0]
            boundary[1, :] = boundary[1, :] - minCoord[1]

            vg_center_x, vg_center_y, vg_indexes = get_pixel_location(boundary, Image.fromarray(fov_image))
            cp_center_x, cp_center_y, cp_indexes = get_cp_location(cellpose_mask, int(np.mean(vg_indexes[0])),
                                                                   int(np.mean(vg_indexes[1])))
            if not cp_indexes:
                CP_FLAG = False

            cell_image = get_cell_image(vg_center_x, vg_center_y, vg_indexes, img_height, img_width, fov_image)
            path_prefix = base_path + 'gene_clustered_images/' + cluster_prefix + '/vizgen/cluster_' + str(clust)
            save_type_to_png(path_prefix, fov, inst_cell, vg_center_x, vg_center_y, cell_image)
            if CP_FLAG:
                path_prefix = base_path + 'gene_clustered_images/' + cluster_prefix + '/cellpose/cluster_' + str(clust)
                save_type_to_png(path_prefix, fov, inst_cell, cp_center_x, cp_center_y, cell_image)

    print('fov ' + str(fov) + ' done')



# -----------------  Select Data  ----------------- #
dataset_name = 'HumanColonCancerPatient1'
z_index_number = 0

# -----------------  Paths to Data  ----------------- #
base_path = '/media/huifang/data/vizgen/' + dataset_name + '/'
z_index = 'zIndex_' + str(z_index_number)
# image stuff
tiffimagepath = base_path + 'images/z' + str(z_index_number)+'_fov_images'+'/'
path_to_save = base_path + 'gene_cluster_image/'
# -----------------  Read cluster result  ----------------- #
cluster_prefix = 'z' + str(z_index_number) + '_ds1_res0.1'
print('Reading cluster data...')
# ad_viz = sc.read_h5ad(base_path + 'gene_cluster_result/'+ cluster_prefix+'/cluster_mata.hdf5')
ad_viz = sc.read_h5ad('/media/huifang/data/vizgen/HumanColonCancerPatient1/gene_cluster_result/z0_ds1_res0.1/cluster_mata.hdf5')
# visualize_dotplot(ad_viz)
# -----------------  select and visualization clusters ----------------- #
# order = ['0','3','4','2']
# visualize_dotplot(ad_viz,order)
# visualize_spatial_distribution(ad_viz)

# -----------------  save corresponding images to file ----------------- #


# get cell boundaries
fov_cnt = len(os.listdir(base_path + 'cell_boundaries'))
print('totally ' + str(fov_cnt)+ ' fovs.')
num_workers = 16
start_time = time.time()
# Use ThreadPoolExecutor to parallelize the loop
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    fov_indices = range(fov_cnt)
    executor.map(process_fov, fov_indices)

end_time = time.time()
print(end_time - start_time)
print('all done')

