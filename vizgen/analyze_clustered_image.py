import matplotlib.pyplot as plt
import numpy as np
from vizgen_util import *
from matplotlib.path import Path
from matplotlib.patches import Rectangle
import cv2
import seaborn as sns
import pandas as pd
import scanpy as sc

def get_cell_statistics(path_to_img):
    image = plt.imread(path_to_img)
    image = np.where(image>0,1,0)
    image = image.astype(np.uint8)
    mask_contour,_ = cv2.findContours(image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = mask_contour[0]
    if len(mask_contour)<5:
        return []
    area = cv2.contourArea(mask_contour)
    perimeter = cv2.arcLength(mask_contour,True)
    elongation = calculate_elongation(cv2.moments(mask_contour))
    return [area,perimeter,elongation]


# -----------------  Select Data  ----------------- #
dataset_name = 'HumanColonCancerPatient1'
z_index_number = 0
# -----------------  Paths to Data  ----------------- #
cluster_image_path = "/media/huifang/data/vizgen/HumanColonCancerPatient1/gene_clustered_images/"
cluster_prefix = 'z' + str(z_index_number) + '_ds1_res0.1'

clusts = ['0','1','2']
labels = ['0','1','2']



# # -----------------  generate image list  ----------------- #

def generate_image_list(cluster_image_path, cluster_prefix, image_type, clusts, labels):
    image_path_template = cluster_image_path + cluster_prefix + '/' + image_type + '/cluster_{}/'
    image_list = []

    for clust, label in zip(clusts, labels):
        type_cnt = 0
        image_path = image_path_template.format(clust)
        image_names = os.listdir(image_path)
        for image_name in image_names:
            fov,cell_inst_id,x,y =extract_cell_info_from_cellname(image_name)
            image_list.append((image_path + image_name, label, x, y))
            type_cnt = type_cnt + 1

        print(clust+' has '+str(type_cnt)+' cells.')
    return image_list

def save_image_list(image_file, image_list):
    with open(image_file, 'w') as f:
        for image_path, label,x,y in image_list:
            f.write(image_path + ' ' + str(label) + ' ' + str(x)+ ' ' + str(y)+ '\n')
    f.close()




imagetypes = ['cellpose','vizgen']
for imagetype in imagetypes:
    image_list = generate_image_list(cluster_image_path, cluster_prefix, imagetype, clusts, labels)
    save_image_list('/home/huifang/workspace/data/imagelists/vizgen_colon_z0_ds1_res0.1_' + imagetype + '_3class.txt', image_list)
print("done")

#
#     image_file_all = '/home/huifang/workspace/data/imagelists/vizgen_breast_'+ image_type +'_z0_all_res0.1_all.txt'
#     # save_file_test = '/home/huifang/workspace/data/imagelists/vizgen_gene_clustered_' + image_type + '_z0_all_res0.1_test.txt'
#     f = open(image_file_all, 'w')
#     # f2 = open(save_file_test, 'w')
#
#     for clust,label in zip(clusts,labels):
#         type_cnt=0
#         path_prefix =cluster_image_path + cluster_prefix + '/cluster_' + str(clust)
#         image_path = path_prefix + '/' + image_type + '/'
#         image_names = os.listdir(image_path)
#         i = 0
#         for image_name in image_names:
#             # if clust == '0':
#             #     if i % 10 !=0:
#             #         i=i+1
#             #         continue
#             # cell_stat = get_cell_statistics(path_prefix + '/nuclei_image/'+image_name)
#             # if len(cell_stat) == 0:
#             #     continue
#             # f.write(image_path + image_name + ' ' + str(label) + ' ' + str(cell_stat[0]) + '\n')
#             f.write(image_path + image_name + ' ' + str(label) + '\n')
#             type_cnt = type_cnt+1
#             i = i + 1
#         print(clust+' has '+str(type_cnt)+' cells.')
#     print(image_type + ' done.')
#     f.close()
# print("all done")
test = input()
# # -----------------  generate gene list  ----------------- #
gene_by_cell = pd.read_csv('/media/huifang/data/vizgen/HumanBreastCancerPatient1/cell_by_gene.csv', header=None, sep=',').values
gene_names = gene_by_cell[0]
gene_by_cell = gene_by_cell[1:]
cell_ids = gene_by_cell[:,0]
gene_by_cell = gene_by_cell[:,1:]
for image_type in imagetypes:
    print(image_type)
    image_file_all = '/home/huifang/workspace/data/imagelists/vizgen_breast_'+ image_type +'_z0_all_res0.1_ds_all_with_size.txt'
    genefile = '/home/huifang/workspace/data/imagelists/vizgen_breast_' + image_type + '_z0_all_res0.1_ds_all_gene_by_cell.txt'

    image_f = open(image_file_all,'r')
    images = image_f.readlines()
    rows=[]
    for image in images:
        cellid = image.split(' ')[0]
        cellid = cellid.split('.')[1]
        cellid = cellid.split('_')[-1]
        rows.append(int(cellid))
    genereads = gene_by_cell[rows,:]
    genereads = genereads.astype(float)
    np.savetxt(genefile,genereads,fmt="%.1f")
    print('done')
    # print(genereads)
    # test = input()




    # gene_f = open(genefile)


# # -----------------  calculate statistics  ----------------- #
#
# for image_type in image_types:
#     areas_all = []
#     perimeters_all = []
#     elongations_all = []
#     celldata=[]
#     for clust in clusts:
#         print(clust)
#         path_prefix =cluster_image_path + cluster_prefix + '/cluster_' + str(clust)
#         image_path = path_prefix + '/' + image_type + '/'
#         image_names = os.listdir(image_path)
#         areas=[]
#         perimeters=[]
#         elongations=[]
#         print(len(image_names))
#         if clust== '0':
#             step = 100
#         else:
#             step = 10
#         for i in range(0,len(image_names),step):
#             image_name = image_names[i]
#
#             image = plt.imread(image_path+image_name)
#             image = np.where(image>0,1,0)
#             image = image.astype(np.uint8)
#             mask_contour,_ = cv2.findContours(image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             mask_contour = mask_contour[0]
#             if len(mask_contour)<5:
#                 continue
#             area = cv2.contourArea(mask_contour)
#             perimeter = cv2.arcLength(mask_contour,True)
#             elongation = calculate_elongation(cv2.moments(mask_contour))
#
#             areas.append(area)
#             perimeters.append(perimeter)
#             elongations.append(elongation)
#             celldata.append([area,perimeter,elongation,clust])
#
#         areas_all.append(np.asarray(areas))
#         perimeters_all.append(np.asarray(perimeters))
#         elongations_all.append((np.asarray(elongations)))
#
#     # area_stas = pd.DataFrame(list(zip(areas_all[0],areas_all[1],areas_all[2],areas_all[3])),columns=clusts)
#     # perimeter_stas = pd.DataFrame(list(zip(perimeters_all[0], perimeters_all[1], perimeters_all[2], perimeters_all[3])), columns=clusts)
#     # elongation_stas = pd.DataFrame(list(zip(elongations_all[0], elongations_all[1], elongations_all[2], elongations_all[3])), columns=clusts)
#     stats = pd.DataFrame(data=celldata,columns=['area','perimeter','elongation','clust'])
#     sns.displot(data=stats,x='area',hue='clust',kind='kde',common_norm=False,fill=True, palette=sns.color_palette('bright')[:4])
#     sns.displot(data=stats, x='perimeter', hue='clust', kind='kde',common_norm=False, fill=True, palette=sns.color_palette('bright')[:4])
#     sns.displot(data=stats, x='elongation', hue='clust', kind='kde',common_norm=False,fill=True, palette=sns.color_palette('bright')[:4])
#     plt.show()
#
#     # fig, axes = plt.subplots(1, 3)
#     # for i in range(0,len(areas_all)):
#     #     sns.displot(areas_all[i], rug=True)
#     #     sns.displot(perimeters_all[i], rug=True, ax=axes[1])
#     #     sns.displot(elongations_all[i], rug=True, ax=axes[2])
#     # plt.show()
#
#
#
#
#     print(image_type + ' done.')