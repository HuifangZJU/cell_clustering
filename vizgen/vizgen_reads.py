
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# load field of view transcript data


# Select data
dataset_name = 'HumanBreastCancerPatient1'
z_index_number = 0

# Paths to Data
base_path = '/media/huifang/data/vizgen/' + dataset_name +'/'
transformation_matrix = pd.read_csv(base_path + 'images/micron_to_mosaic_pixel_transform.csv', header=None, sep=' ').values

fov_range = np.load(base_path +'fov_range.npy')
for i in range(0,fov_range.shape[0]):
    Coord = fov_range[i,:]
    fov = Coord[0]
    trans_prefix = 'z' + str(z_index_number) + '_fov' + str(fov)
    # 0: cell id?
    # 2,3,4: globalx, globaly,globalz
    # 5,6:localx,localy
    # 7: fov
    # 8:gene
    transcripts = pd.read_csv(base_path + 'detected_transcripts/' + str(z_index_number) + '/' + trans_prefix+'.txt', header=None, index_col=0)
    # transpose to current field of view coordinate
    temp = transcripts[[2,3]].values
    transcript_positions = np.ones((temp.shape[0], temp.shape[1]+1))
    transcript_positions[:, :-1] = temp
    # Transform coordinates to mosaic pixel coordinates
    transformed_positions = np.matmul(transformation_matrix, np.transpose(transcript_positions))[:-1]
    transcripts.loc[:, 5] = transformed_positions[0, :]
    transcripts.loc[:,6] = transformed_positions[1, :]

    gene_maps = transcripts.groupby(by=8)
    for gene, data in gene_maps:
        plt.scatter(data[5],data[6])
    plt.show()
