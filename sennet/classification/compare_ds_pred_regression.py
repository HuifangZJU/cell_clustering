import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.stats import binned_statistic_2d
import seaborn as sns
import pandas as pd

from matplotlib import cm, colors

def get_top_coords(signal, coords,threshold):
    mask = signal >= threshold
    return coords[mask], signal[mask],mask

def get_bottom_coords_percentile(signal, coords, percentile):
    threshold = np.percentile(signal, percentile)
    mask = signal <= threshold
    return coords[mask], signal[mask],mask
def get_top_bottom_coords(signal, coords, percentile):

    # compute thresholds
    bottom_thr = np.percentile(signal, percentile)
    top_thr = np.percentile(signal, 100 - percentile)

    # mask for bottom or top
    sel_mask = (signal <= bottom_thr) | (signal >= top_thr)

    return coords[sel_mask], signal[sel_mask], sel_mask

def get_top_coords_percentile(signal, coords, percentile):
    threshold = np.percentile(signal, 100 - percentile)
    mask = signal >= threshold
    return coords[mask], signal[mask],mask
def binarize_with_extremes(arr, th):
    arr = np.asarray(arr)
    min_val, max_val = arr.min(), arr.max()
    out = np.where(arr > th, max_val, 0.4)
    return out

def binarize_on_median(arr):
    arr = np.asarray(arr)
    min_val, max_val = arr.min(), arr.max()
    median_val = np.median(arr)  # 50th percentile

    out = np.where(arr > median_val, max_val, min_val)
    return out

def visualize_marker_overlay():
    # Normalize gene expression to [0, 1]
    norm_xenium = (xenium_gene_expr - xenium_gene_expr.min()) / (xenium_gene_expr.max() - xenium_gene_expr.min())
    norm_codex = (codex_gene_expr - codex_gene_expr.min()) / (codex_gene_expr.max() - codex_gene_expr.min())
    norm_pred = (codex2xenium_prediction - codex2xenium_prediction.min())/(codex2xenium_prediction.max() - codex2xenium_prediction.min())


    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    f.suptitle(f"Codex-{codex_sampleid}-{codex_regionid} : Xenium-{xenium_sampleid}-{xenium_regionid}", fontsize=20)

    # Use the same normalization for both "all" and "top" plots
    vmin_x, vmax_x = norm_xenium.min(), norm_xenium.max()
    vmin_c, vmax_c = norm_codex.min(), norm_codex.max()
    vmin_p,vmax_p = norm_pred.min(),norm_pred.max()

    # cmap_blue = cm.get_cmap('Blues')
    # colors_codex = cmap_blue(norm_codex)
    # colors_codex[:, -1] = norm_codex
    cmap_blue = cm.get_cmap('Blues')
    cmap_blue_shifted = colors.LinearSegmentedColormap.from_list(
        "Blues_shifted", cmap_blue(np.linspace(0.3, 1, 256))
    )
    colors_codex = cmap_blue_shifted(norm_codex)
    min_alpha = 0.2
    colors_codex[:, -1] = min_alpha + (1 - min_alpha) * norm_codex

    # Convert cmap to RGBA with alpha = normalized expression
    cmap_red = cm.get_cmap('Reds')
    colors_xenium = cmap_red(norm_xenium)
    colors_xenium[:, -1] = norm_xenium  # alpha channel

    cmap_green = cm.get_cmap('Greens')
    colors_c2x = cmap_green(norm_pred)
    colors_c2x[:, -1] = norm_pred  # alpha channel



    # Full CODEX
    axes[0, 0].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                       color=colors_codex, s=10, label='CODEX p16')
    axes[0, 0].set_title("P16 Level in CODEX", fontsize=22)

    # Full Xenium
    axes[0, 1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                       color=colors_xenium, s=10, label='Xenium DS16')
    axes[0, 1].set_title("Ds Level in Xenium", fontsize=22)

    axes[0, 2].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                       color=colors_c2x, s=10, label='Codex2Xenium Pred')
    axes[0, 2].set_title("Predicted P16 on Xenium", fontsize=22)



    # Compute top percentile

    # top_coords1, top_signal1 = get_top_coords(xenium_gene_data.obs["ds_top"], warped_xenium_coor,0.5)
    # top_coords2, top_signal2 = get_top_coords(codex_gene_data.obs["p16_top"],warped_codex_coor,0.5)
    ds_percentile = 10
    p16_percentile = 10
    top_coords1, top_signal1,top_mask = get_top_bottom_coords(norm_xenium, warped_xenium_coor, ds_percentile)
    top_coords2, top_signal2,_ = get_top_bottom_coords(norm_codex, warped_codex_coor, p16_percentile)
    top_coords3, top_signal3,_ = get_top_bottom_coords(norm_pred, warped_xenium_coor, p16_percentile)
    #
    # plt.hist(top_signal3, bins=100, color="steelblue", edgecolor="black")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of array values")
    # plt.show()
    # top_signal3 = binarize_on_median(top_signal3)


    # CODEX: Top-k percentile
    axes[1, 0].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                       c="lightgray", s=5, label="All cells")
    axes[1, 0].scatter(top_coords2[:, 0], top_coords2[:, 1],
                       c=top_signal2, cmap=cmap_blue_shifted, s=10,
                       vmin=vmin_c, vmax=vmax_c,  # shared color scale
                       label="Global positive senescence cells")
    # axes[1, 0].set_title("CODEX: Global positive senescence cells", fontsize=22)
    axes[1, 0].set_title(f"CODEX: Top & Bottom {p16_percentile}% p16 score", fontsize=22)

    # Xenium: Top-k percentile
    axes[1, 1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                       c="lightgray", s=5, label="All cells")
    axes[1, 1].scatter(top_coords1[:, 0], top_coords1[:, 1],
                             c=top_signal1, cmap="Reds", s=10,
                             vmin=vmin_x, vmax=vmax_x,  # shared color scale
                             label="Global positive senescence cells")
    # axes[1, 1].set_title("Xenium: Global positive senescence cells", fontsize=22)
    axes[1, 1].set_title(f"Xenium: Top & Bottom {ds_percentile}% ds score", fontsize=22)

    # Codex2Xenium
    axes[1, 2].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                       c="lightgray", s=5, label="All cells")
    axes[1, 2].scatter(top_coords3[:, 0], top_coords3[:, 1],
                       c=top_signal3, cmap="Greens", s=10,
                       vmin=vmin_p, vmax=vmax_p,  # shared color scale
                       label="Global positive senescence cells")
    axes[1, 2].set_title(f"C2X: Top & Bottom {p16_percentile}% predicted p16 score", fontsize=22)



    plt.tight_layout()
    plt.show()

def change2xenium_sampleid(s):
    return s[:-1]+'-'+s[-1]


def add_topk_percentile_label(df, value_col, new_col="label_bin", k=5):
    """
    Add a binary column based on percentile ranking in a given column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_col : str
        Column name to rank by.
    new_col : str, optional
        Name of the new binary column. Default "label_bin".
    k : float
        Percentile cutoff (e.g., k=5 means top 5% will be labeled 1).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the new column added.
    """
    # compute threshold for top-k%
    cutoff = np.percentile(df[value_col], 100 - k)

    # assign 1 if value >= cutoff, else 0
    df[new_col] = (df[value_col] >= cutoff).astype(int)

    return df

file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
r_values = []
p_values = []


result_file = "/media/huifang/data/sennet/codex/cell_images/results/regressor_models_predictions.csv"
result = pd.read_csv(result_file)
result = result.rename(columns={
    "sample_id": "sampleid",
    "cellid": "cell_id"
})
codex_result = pd.read_csv("/media/huifang/data/sennet/codex/p16.csv")

result = add_topk_percentile_label(result, value_col="ds", new_col="ds_top", k=10)
# On codex_result: top 10% of p16 as positives
codex_result = add_topk_percentile_label(codex_result, value_col="p16", new_col="p16_top", k=10)



for i in range(0,len(sennet_pairs)):
    print(i)
    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}_registered.h5ad")

    xenium_sampleid = change2xenium_sampleid(xenium_sampleid)
    subset = result[result['sampleid'] == xenium_sampleid]
    xenium_gene_data.obs = xenium_gene_data.obs.merge(subset[["cell_id", "ds", "pred","ds_top"]], on="cell_id", how="left")

    xenium_gene_data.obs["ds"] = xenium_gene_data.obs["ds"].fillna(0)
    xenium_gene_data.obs["pred"] = xenium_gene_data.obs["pred"].fillna(0)



    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}_registered.h5ad")
    codex_subset = codex_result[codex_result["slide_name"] ==codex_sampleid]
    codex_subset = codex_subset[codex_subset["unique_region"] == codex_regionid]

    codex_gene_data.obs = codex_gene_data.obs.merge(codex_subset[["label", "p16_top"]], on="label",
                                                      how="left")


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
    xenium_gene_expr = xenium_gene_data.obs['ds'].values

    # CODEX: use p16 intensity from .obs
    codex_gene_expr = codex_gene_data.obs['p16'].values
    codex_gene_expr = codex_gene_expr - codex_gene_expr.min()
    codex_gene_expr = np.log1p(codex_gene_expr)

    codex2xenium_prediction = xenium_gene_data.obs['pred'].values


    visualize_marker_overlay()

