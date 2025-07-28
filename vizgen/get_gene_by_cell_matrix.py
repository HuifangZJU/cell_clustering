import matplotlib.pyplot as plt
import numpy as np
from vizgen_util import *
from matplotlib.path import Path
from matplotlib.patches import Rectangle
import cv2
import seaborn as sns
import pandas as pd
import scanpy as sc
import anndata as ad

# # -----------------  generate gene list  ----------------- #
gene_by_cell = pd.read_csv('/media/huifang/data/vizgen/HumanBreastCancerPatient1/cell_by_gene.csv', header=None, sep=',').values
gene_names = gene_by_cell[0]
gene_by_cell = gene_by_cell[1:]
cell_ids = gene_by_cell[:,0]
gene_by_cell = gene_by_cell[:,1:]
gene_by_cell = gene_by_cell.astype(float)

image_file_all = '/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_all_with_size.txt'
genefile = '/home/huifang/workspace/data/imagelists/vizgen_breast_local_image_z0_all_res0.1_ds_all_gene_by_cell.txt'

image_f = open(image_file_all,'r')
images = image_f.readlines()
rows=[]
for image in images:
    cellid = image.split(' ')[0]
    cellid = cellid.split('.')[1]
    cellid = cellid.split('_')[-1]
    rows.append(int(cellid))
genereads = gene_by_cell[rows,:]
print(genereads.shape)
test = input()
adata = ad.AnnData(genereads)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

np.savetxt(genefile,adata.X,fmt="%.5f")
print('done')