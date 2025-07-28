import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from vizgen_util import *
from matplotlib.path import Path
from matplotlib.patches import Rectangle
import cv2
# -----------------  Select Data  ----------------- #
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 0

# -----------------  Paths to Data  ----------------- #
base_path = '/media/huifang/data/vizgen/' + dataset_name + '/'
z_index = 'zIndex_' + str(z_index_number)
# image stuff
tiffimagepath = base_path + 'images/z' + str(z_index_number)+'_fov_images'+'/'
cluster_image_path = "/media/huifang/data/vizgen/HumanBreastCancerPatient1/gene_cluster_image/"


# -----------------  Read cluster result  ----------------- #
cluster_prefix = 'z' + str(z_index_number) + '_all_data'
# cluster_prefix = 'z' + str(z_index_number) + '_all'
print('Reading cluster data...')
ad_viz = sc.read_h5ad(cluster_image_path + cluster_prefix+ '_cluster_mata.hdf5')

# -----------------  Visualization  ----------------- #

order = ['0','3','4','2']
# order = ['4','2']
# order = ['5','6','0','11','3','8','4','10','9','14','1','2','13','16','7','12']
# order =['5','8','13','3','9','14','11','18','1','15','6','2','10','17']
# ad_viz = ad_viz[ad_viz.obs['leiden'].isin(order),:]
# # sc.pl.dotplot(ad_viz,marker_genes_dict,'leiden',categories_order=order,dendrogram=False)
# print('Dot plotting..')
# sc.pl.dotplot(ad_viz,marker_genes_dict,'leiden',dendrogram=True)

# Calculate Leiden Signatures
#########################################
ser_counts = ad_viz.obs['leiden'].value_counts()
ser_counts.name = 'cell counts'
meta_leiden = pd.DataFrame(ser_counts)
cat_name = 'leiden'
sig_leiden = pd.DataFrame(columns=ad_viz.var_names, index=ad_viz.obs[cat_name].cat.categories)
for clust in ad_viz.obs[cat_name].cat.categories:
    sig_leiden.loc[clust] = ad_viz[ad_viz.obs[cat_name].isin([clust]),:].X.mean(0)
sig_leiden = sig_leiden.transpose()
leiden_clusters = ['Leiden-' + str(x) for x in sig_leiden.columns.tolist()]
sig_leiden.columns = leiden_clusters
meta_leiden.index = sig_leiden.columns.tolist()
meta_leiden['leiden'] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())


# save corresponding images to file
crop_size = 256
# get cell boundaries
fov_cnt = len(os.listdir(base_path + 'cell_boundaries'))
cell_center_by_cluster=[]
colors = [(1,0,0),(0,0,1),(0,1,0),(0,1,0)]
# colors = [(0,1,0),(0,1,0)]

# 159
for fov in range(311,fov_cnt):
    cell_center_by_cluster = []
    perimeter_by_cluster=[]
    area_by_cluster=[]
    elongation_by_cluster=[]

    currentCells, cellid = get_fov_cell_boundaries(dataset_name, z_index_number, fov)
    minCoord = np.min([np.min(x, axis=1) for x in currentCells], axis=0).astype(int)
    maxCoord = np.max([np.max(x, axis=1) for x in currentCells], axis=0).astype(int)

    fov_suffix = 'z' + str(z_index_number) + '_fov' + str(fov)
    fov_image = tifffile.imread(tiffimagepath+'/DAPI/mosaic_DAPI_' + fov_suffix + '.tif')
    bound_image = tifffile.imread(tiffimagepath + '/Cellbound1/mosaic_Cellbound1_' + fov_suffix + '.tif')
    cellpose_mask = plt.imread(tiffimagepath+'/DAPI/masks//mosaic_DAPI_' + fov_suffix + '_cp_masks.png')

    seg_image = plt.imread(tiffimagepath + '/seg/' + 'seg_' + fov_suffix + '.png')
    fov_image = normImg(fov_image)
    img_height, img_width = fov_image.shape
    fig, a = plt.subplots(nrows=1, ncols=1)
    a.imshow(fov_image)
    # for clust in ad_viz.obs[cat_name].cat.categories:
    for i in range(0,len(order)):
        clust = order[i]
        # print('saving clust '+ str(clust) +'...')
        cluster_meta_cells = ad_viz[ad_viz.obs[cat_name].isin([clust]),:]
        cluster_fov_meta_cells = cluster_meta_cells[cluster_meta_cells.obs['fov'].isin([fov]),:]
        cluster_fov_meta_cells = cluster_fov_meta_cells.obs
        centers=[]
        perimeters=[]
        areas=[]
        elongations=[]
        for inst_cell in cluster_fov_meta_cells.index.tolist():
            id = np.where(cellid==inst_cell)
            boundary = currentCells[id[0][0]]
            boundary = boundary.astype(int)
            boundary[0, :] = boundary[0, :] - minCoord[0]
            boundary[1, :] = boundary[1, :] - minCoord[1]

            center_x, center_y, index = get_pixel_location(boundary, Image.fromarray(fov_image))
            nuclei_center_x, nuclei_center_y, nuclei_index = get_cp_location(cellpose_mask, int(np.mean(index[0])), int(np.mean(index[1])))



            if not nuclei_index:
                continue
            tmp = np.zeros(fov_image.shape)
            tmp[nuclei_index] = 1
            cell_contour,_ = cv2.findContours(tmp.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            cell_contour = cell_contour[0]
            area = cv2.contourArea(cell_contour)
            perimeter = cv2.arcLength(cell_contour, True)
            elongation = calculate_elongation(cv2.moments(cell_contour))
            # ac = np.asarray(ac[0][0])
            # ac = ac[:, 0, :]
            #

            # if np.abs(center_x - nuclei_center_x) >10 or  np.abs(center_y - nuclei_center_y) >10:
            #     continue
            # #
            xy_range = [np.max(index[0]) - center_x, center_x - np.min(index[0]), np.max(index[1]) - center_y,center_y - np.min(index[1])]
            cell_tight_crop_size = np.max(xy_range)
            tight_crop_size = cell_tight_crop_size
            # #
            nuclei_xy_range = [np.max(nuclei_index[0]) - nuclei_center_x, nuclei_center_x - np.min(nuclei_index[0]), np.max(nuclei_index[1]) - nuclei_center_y,
                        nuclei_center_y - np.min(nuclei_index[1])]
            nuclei_tight_crop_size = np.max(nuclei_xy_range)
            tight_crop_size = np.max([cell_tight_crop_size,nuclei_tight_crop_size])
            # #
            # #
            x_min, x_max, y_min, y_max = get_pixel_limit(center_x, center_y, tight_crop_size+2,img_height,img_width)
            nuclei_x_min, nuclei_x_max, nuclei_y_min, nuclei_y_max = get_pixel_limit(nuclei_center_x, nuclei_center_y, tight_crop_size, img_height,
                                                         img_width)
            envir_x_min, envir_x_max, envir_y_min, envir_y_max = get_pixel_limit(nuclei_center_x, nuclei_center_y, crop_size, img_height,
                                                         img_width)

            # drawpolygon(boundary, fov_image, colors[i])
            # a.scatter(boundary[0, :],boundary[1, :],color=colors[i],linewidths=0.2)
            a.scatter(cell_contour[::4, 0], cell_contour[::4, 1], color=colors[i], s=30)
            # a.add_patch(
            #     Rectangle((nuclei_y_min, nuclei_x_min), 2 * nuclei_tight_crop_size,
            #               2 * nuclei_tight_crop_size, edgecolor=colors[i], fill=False, linewidth=2))

            # a.add_patch(
            #     Rectangle((y_min, x_min), 2 * tight_crop_size,
            #               2 * tight_crop_size, edgecolor=colors[i], fill=False, linewidth=2))
            # a.scatter(nuclei_center_y, nuclei_center_x, color=colors[i], marker='x')

            #
            # #Visualization
            # print('cell ' +str(inst_cell) + ' is in clust '+ str(clust) +'...')
            # f,a = plt.subplots(2,3)
            # a[0, 0].imshow(env_image)
            # a[0, 0].add_patch(
            #     Rectangle((nuclei_y_min - envir_y_min, nuclei_x_min - envir_x_min), 2 * tight_crop_size,
            #               2 * tight_crop_size, edgecolor='red', fill=False, linewidth=2))
            # a[0, 1].imshow(local_image)
            # a[0, 2].imshow(nuclei_image)
            # a[1, 0].imshow(env_bound)
            # a[1, 0].add_patch(
            #     Rectangle((y_min - envir_y_min, x_min - envir_x_min), 2 * tight_crop_size,
            #               2 * tight_crop_size, edgecolor='red', fill=False, linewidth=2))
            # a[1, 1].imshow(cell_bound_image)
            # a[1, 2].imshow(cell_image)
            #
            # a[0, 0].title.set_text('env_image')
            # a[0, 1].title.set_text('local_image')
            # a[0, 2].title.set_text('nuclei_image')
            # a[1, 0].title.set_text('env_bound')
            # a[1, 1].title.set_text('cell_bound_image')
            # a[1, 2].title.set_text('cell_image')
            # # plt.figure(2)
            # # plt.imshow(seg_image[envir_x_min:envir_x_max,envir_y_min:envir_y_max])
            #
            # plt.show()

            centers.append([center_x,center_y])
            perimeters.append(perimeter)
            areas.append(area)
            elongations.append(elongation)
        cell_center_by_cluster.append(np.asarray(centers))
        perimeter_by_cluster.append(perimeters)
        area_by_cluster.append(areas)
        elongation_by_cluster.append(elongations)
        cancer=0


    plt.show()


    # for i in range(0,len(cell_center_by_cluster)):
    #     xy = cell_center_by_cluster[i]
    #     if not xy.any():
    #         continue
    #     plt.scatter(xy[:,1],xy[:,0],color=colors[i],marker='o')
    #
    # plt.show()





    print('fov '+str(fov)+' done')

        # test = input()
