import os

import pandas as pd
import scanpy as sc
from vizgen_gene_utils import marker_genes_dict,visualize_dotplot,visualize_spatial_distribution


def run(base_path,dataset_name,path_to_save,save_flag=False,cell_downsampling=10,resolution=0.1):
    # -----------------  Read cell data  ----------------- #
    # meta_cell only has one key:featuredata, describe properties of each cell, e.g. location, size
    meta_cell = pd.read_csv(base_path + dataset_name + 'cell_metadata' + '.csv', index_col=0)
    # matrix of cells with gene reads
    cell_by_gene = pd.read_csv(base_path + dataset_name + 'cell_by_gene.csv', index_col=0)

    # -----------------  select cells for clustering  ----------------- #
    # cells = meta_cell.index.tolist()
    cells = cell_by_gene.index.tolist()
    # select cells
    cells = cells[0::cell_downsampling]
    cell_by_gene = cell_by_gene.loc[cells]
    meta_cell = meta_cell.loc[cells]

    # -----------------  prepare gene annodata and select genes for clustering  ----------------- #
    meta_cell['barcodeCount'] = cell_by_gene.sum(axis=1)
    # meta_cell.index = range(len(meta_cell.index.tolist()))
    # initialize meta_gene
    meta_gene = pd.DataFrame(index=cell_by_gene.columns.tolist())
    # drop blanks for single cell analysis
    keep_genes = [x for x in cell_by_gene.columns.tolist() if 'Blank' not in x]


    cell_by_gene = cell_by_gene[keep_genes]
    meta_gene = meta_gene.loc[keep_genes]
    meta_gene['expression'] = cell_by_gene.sum(axis=0)
    ad_viz = sc.AnnData(X=cell_by_gene.values, obs=meta_cell, var=meta_gene)

    # -----------------  prepare gene annodata and select genes for clustering  ----------------- #
    print('beging clustering')
    ad_viz.var_names_make_unique()
    ad_viz.var["mt"] = ad_viz.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ad_viz, qc_vars=["mt"], inplace=True)
    ##filter out data by number_of_counts
    # sc.pp.filter_cells(ad_viz, min_counts=5000)
    # sc.pp.filter_cells(ad_viz, max_counts=35000)
    # sc.pp.filter_cells(ad_viz, min_genes=1)
    # sc.pp.filter_genes(ad_viz, min_cells=int(len(cells)/10))

    print(ad_viz.obs)
    sc.pp.normalize_total(ad_viz)
    sc.pp.log1p(ad_viz)
    sc.pp.scale(ad_viz, max_value=10)
    sc.tl.pca(ad_viz, svd_solver='arpack')
    sc.pp.neighbors(ad_viz, n_neighbors=10, n_pcs=20)
    sc.tl.umap(ad_viz)
    sc.tl.leiden(ad_viz, resolution=resolution)
    print('clustering done')

    # Calculate Leiden Signatures
    #########################################
    ser_counts = ad_viz.obs['leiden'].value_counts()
    ser_counts.name = 'cell counts'
    meta_leiden = pd.DataFrame(ser_counts)
    cat_name = 'leiden'
    sig_leiden = pd.DataFrame(columns=ad_viz.var_names, index=ad_viz.obs[cat_name].cat.categories)
    for clust in ad_viz.obs[cat_name].cat.categories:
        sig_leiden.loc[clust] = ad_viz[ad_viz.obs[cat_name].isin([clust]), :].X.mean(0)
    sig_leiden = sig_leiden.transpose()
    leiden_clusters = ['Leiden-' + str(x) for x in sig_leiden.columns.tolist()]
    sig_leiden.columns = leiden_clusters
    meta_leiden.index = sig_leiden.columns.tolist()
    meta_leiden['leiden'] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())

    # -----------------  save result to file ----------------- #
    if save_flag:
        save_prefix = path_to_save+ 'z' + str(z_index_number) + '_ds'+str(cell_downsampling)+'_res'+str(resolution)
        os.makedirs(save_prefix, exist_ok=True)
        ad_viz.write_h5ad(save_prefix+ '/cluster_mata.hdf5')
        for clust in ad_viz.obs['leiden'].cat.categories:
            gene_expression = sig_leiden['Leiden-' + str(clust)]
            gene_expression = gene_expression.sort_values(axis=0, ascending=False)
            gene_expression.to_csv(save_prefix+ '/cluster' + str(clust) +'_genes.csv')
        return ad_viz

# -----------------  Paths to Data  ----------------- #
base_path = '/media/huifang/data/vizgen/'
dataset_name = 'HumanColonCancerPatient1/'
z_index_number = 0
path_to_save = base_path+dataset_name+'gene_cluster_result/'
# ad_viz = run(base_path,dataset_name,path_to_save,save_flag=True,cell_downsampling=1,resolution=0.1)
ad_viz = sc.read_h5ad('/media/huifang/data/vizgen/HumanBreastCancerPatient1/gene_cluster_result/z0_all_data_res0.1_cluster_mata.hdf5')
visualize_dotplot(ad_viz)
# # -----------------  select & visualization meaningful classes----------------- #
order = ['0', '3', '2', '4']
visualize_dotplot(ad_viz,order)
# # -----------------  spatial visualization----------------- #
visualize_spatial_distribution(ad_viz,order)










