import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.stats import binned_statistic_2d
import seaborn as sns
import pandas as pd



def get_top_coords(signal, coords, percentile):
    threshold = np.percentile(signal, 100 - percentile)
    mask = signal >= threshold
    return coords[mask], signal[mask]

def visualize_marker_overlay():
    # Normalize gene expression to [0, 1]
    norm_xenium = (xenium_gene_expr - xenium_gene_expr.min()) / (xenium_gene_expr.max() - xenium_gene_expr.min())
    norm_codex = (codex_gene_expr - codex_gene_expr.min()) / (codex_gene_expr.max() - codex_gene_expr.min())

    # plot_two_datasets(norm_xenium,warped_xenium_coor,norm_codex,warped_codex_coor)

    f, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Use the same normalization for both "all" and "top" plots
    vmin_x, vmax_x = norm_xenium.min(), norm_xenium.max()
    vmin_c, vmax_c = norm_codex.min(), norm_codex.max()

    # Convert cmap to RGBA with alpha = normalized expression
    cmap_red = cm.get_cmap('Reds')
    colors_xenium = cmap_red(norm_xenium)
    colors_xenium[:, -1] = norm_xenium  # alpha channel

    cmap_blue = cm.get_cmap('Blues')
    colors_codex = cmap_blue(norm_codex)
    colors_codex[:, -1] = norm_codex

    # Full Xenium
    axes[0, 0].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                       color=colors_xenium, s=10, label='Xenium CDKN1A')
    axes[0, 0].set_title("Ds Level in Xenium", fontsize=22)

    # Full CODEX
    axes[0, 1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                       color=colors_codex, s=10, label='CODEX p16')
    axes[0, 1].set_title("P16 Level in CODEX", fontsize=22)

    # Compute top percentile
    ds_percentile = 10
    p16_percentile= 10
    top_coords1, top_signal1 = get_top_coords(norm_xenium, warped_xenium_coor, ds_percentile)
    top_coords2, top_signal2 = get_top_coords(norm_codex, warped_codex_coor, p16_percentile)

    # Xenium: Top-k percentile
    axes[1, 0].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                       c="lightgray", s=5, label="All cells")
    sc1 = axes[1, 0].scatter(top_coords1[:, 0], top_coords1[:, 1],
                             c=top_signal1, cmap="Reds", s=10,
                             vmin=vmin_x, vmax=vmax_x,  # shared color scale
                             label=f"Top {ds_percentile}%")
    axes[1, 0].set_title(f"Xenium: Top {ds_percentile}% ds score", fontsize=22)

    # CODEX: Top-k percentile
    axes[1, 1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                       c="lightgray", s=5, label="All cells")
    sc2 = axes[1, 1].scatter(top_coords2[:, 0], top_coords2[:, 1],
                             c=top_signal2, cmap="Blues", s=10,
                             vmin=vmin_c, vmax=vmax_c,  # shared color scale
                             label=f"Top {p16_percentile}%")
    axes[1, 1].set_title(f"CODEX: Top {p16_percentile}% p16 score", fontsize=22)

    plt.tight_layout()
    plt.show()

def change2xenium_sampleid(s):
    return s[:-1]+'-'+s[-1]

file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
r_values = []
p_values = []

deepscence_file = "/media/huifang/data/sennet/xenium/deepscence/xall_scored_meta.csv"
ds_score = pd.read_csv(deepscence_file)
# Split the "Unnamed: 0" column into sampleid and cell_id
ds_score[['sampleid', 'cell_id']] = ds_score['Unnamed: 0'].str.split('_', n=1, expand=True)
# Keep only needed columns
result = ds_score[['sampleid', 'cell_id', 'ds']]


for i in range(0,len(sennet_pairs)):
    print(i)
    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    sample_id = change2xenium_sampleid(xenium_sampleid)

    subset = result[result['sampleid'] == sample_id]

    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}_registered.h5ad")
    xenium_gene_data.obs = xenium_gene_data.obs.merge(subset[["cell_id", "ds"]], on="cell_id", how="left")

    xenium_gene_data.obs["ds"] = xenium_gene_data.obs["ds"].fillna(0)



    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}_registered.h5ad")


    xenium_coordinate = np.stack([
        xenium_gene_data.obs['x_trans'].values,
        xenium_gene_data.obs['y_trans'].values
    ], axis=1)
    codex_coordinate = np.stack([
        codex_gene_data.obs['x_trans'].values,
        codex_gene_data.obs['y_trans'].values
    ], axis=1)

    warped_xenium_coor = np.stack([
        xenium_gene_data.obs['x_aligned'].values,
        xenium_gene_data.obs['y_aligned'].values
    ], axis=1)

    warped_codex_coor= np.stack([
        codex_gene_data.obs['x_aligned'].values,
        codex_gene_data.obs['y_aligned'].values
    ], axis=1)

    # xenium_gene_expr = xenium_gene_data[:, 'CDKN1A'].to_df()['CDKN1A']
    xenium_ds_value = xenium_gene_data.obs['ds'].values

    # CODEX: use p16 intensity from .obs
    codex_gene_expr = codex_gene_data.obs['p16'].values
    codex_gene_expr = codex_gene_expr - codex_gene_expr.min()

    # xenium_gene_expr = np.log1p(xenium_gene_expr)
    xenium_gene_expr = xenium_ds_value
    codex_gene_expr = np.log1p(codex_gene_expr)

    visualize_marker_overlay()

