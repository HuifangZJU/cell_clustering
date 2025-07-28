import numpy as np
import pandas as pd
import scanpy as sc
from math import cos,sin

marker_genes_dict = {
    'Cancer3': ['FOXM1', 'BIRC5', 'PLK1', 'MYBL2'],
    'Cancer4': ['CDCA7', 'E2F1', 'FOXM1', 'EZH2', 'LRP6'],
    'Cancer1': ['CDH1', 'VCAM1', 'LAMB3', 'TAPBP', 'FZD7'],
    'Cancer2': ['LRP6', 'EPHB3', 'ERBB3', 'FGFR2', 'DKK1'],
    'Cancer5': ['PKM', 'SOD2', 'SPP1', 'CEBPB', 'AKT3'],
    'Dendritic': ['CD207', 'FCER1A', 'HLA-DRA', 'CSF1R', 'HLA-DRB1'],
    'Endothelial': ['COL4A1', 'PLVAP', 'PECAM1', 'ETS1', 'VWF'],
    'Fibroblast1': ['COL1A1', 'FN1', 'COL11A1', 'COL5A1', 'ACTA2'],
    'Fibroblast2': ['COL5A1', 'FN1', 'COL1A1', 'COL11A1'],
    'B/PlasmaB': ['CD79A', 'MZB1', 'POU2AF1', 'XBP1', 'FCRL5'],
    'TAM-C1Q': ['FCGR3A', 'CD14', 'CYBB', 'C1QC', 'CSF1R'],
    'TAM-SPP1': ['SPP1', 'TGFBI', 'MMP9', 'FCGR3A'],
    'T/NK': ['LYZ', 'TRAC', 'CD3E', 'CD2', 'PTPRC', 'ZAP70'],
    'Mast': ['KIT', 'BST2', 'CTSG', 'CD22', 'TGFB1'],
}

def visualize_dotplot(ad_viz,order=False):
    if order:
        ad_viz = ad_viz[ad_viz.obs['leiden'].isin(order), :]
        sc.pl.dotplot(ad_viz, marker_genes_dict, 'leiden', categories_order=order, dendrogram=False)
    else:
        sc.pl.dotplot(ad_viz, marker_genes_dict, 'leiden', dendrogram=True)

def visualize_spatial_distribution(ad_viz,order=False,size_=10,saveflag=False,savename=[]):
    if order:
        ad_viz = ad_viz[ad_viz.obs['leiden'].isin(order), :]
    theta = np.deg2rad(-90)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    ad_viz.obs[['center_x', 'center_y']] = ad_viz.obs[['center_x', 'center_y']].dot(rot)
    coordinate = np.asarray([ad_viz.obs['center_y'], ad_viz.obs['center_x']])
    ad_viz.obsm["spatial"] = coordinate.transpose()
    if saveflag:
        sc.pl.embedding(ad_viz, basis="spatial", color="leiden", size=size_,show=False)
        sc.pl._utils.savefig(savename)
    else:
        sc.pl.embedding(ad_viz, basis="spatial", color="leiden", size=size_)