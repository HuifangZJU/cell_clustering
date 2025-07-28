import numpy as np
from matplotlib import pyplot as plt


def get_image_path(imagelist,query):
    query_image_path = imagelist[query]
    query_image_path = query_image_path.split(' ')
    query_path = query_image_path[0]
    query_label = query_image_path[1]
    return query_path,query_label







neighbors = np.load('/media/huifang/data/experiment/scan/vizgen/initial/pretext/topk-train-neighbors.npy')



imagelist = '/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_train_with_position.txt'
imagelist = open(imagelist,'r')
imagelist = imagelist.readlines()

for i in range(neighbors.shape[0]):
    query = neighbors[i,0]
    query_neighbor = neighbors[i,1:]

    query_image_path, query_label = get_image_path(imagelist,query)
    print('query label is '+ query_label)
    print('neighbor labels are: ')
    query_image = plt.imread(query_image_path)
    
    # plt.imshow(query_image)
    f,a = plt.subplots(2,3)
    a[0,0].imshow(query_image,cmap='gray')
    for i in range(2):
        for j in range(3):
            id = i * 3 + j -1
            if id <0:
                continue
            temp_neighbor_image_path, knn_label = get_image_path(imagelist,query_neighbor[id])

            print(knn_label+' ')
            temp_neighbor_image = plt.imread(temp_neighbor_image_path)
            a[i][j].imshow(temp_neighbor_image,cmap='gray')

    plt.show()

