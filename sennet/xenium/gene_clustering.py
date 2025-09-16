import os
from sklearn.cluster import KMeans
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

def plot_known_clusters(adata, prefix, outdir):
    """
    Plot spatial, UMAP, and t-SNE colored by known cluster labels.
    Adds cell counts to titles and legend entries.

    Parameters
    ----------
    adata : AnnData
        Must contain:
            - adata.obs['Cluster']
            - adata.obs['x_centroid'], adata.obs['y_centroid']
            - adata.obsm['X_umap'], adata.obsm['X_tsne']
    prefix : str
        A label like 'strong' or 'weak' to name output files.
    outdir : str
        Path to directory to save PNG files.
    """
    os.makedirs(outdir, exist_ok=True)
    cluster_col = "Cluster"
    size_pts = 2

    # Build mapping from cluster → 'Cluster X (n=###)'
    cluster_counts = adata.obs[cluster_col].value_counts().sort_index()
    label_map = {
        str(cl): f"Cluster {cl} (n={count})"
        for cl, count in cluster_counts.items()
    }

    # Use this to relabel cluster column for plotting
    adata.obs[f"{cluster_col}_label"] = adata.obs[cluster_col].map(label_map)

    n_cells = adata.n_obs
    title_suffix = f"{prefix} — {n_cells:,} cells"

    # 1) Spatial
    fig, ax = plt.subplots(figsize=(12,10))
    ax.invert_yaxis()
    ax.set_aspect('equal')
    sc.pl.scatter(
        adata,
        x="x_centroid", y="y_centroid",
        color=f"{cluster_col}_label",
        size=size_pts,
        ax=ax,
        show=False,
        title=f"Spatial clusters ({title_suffix})",
    )
    fig.savefig(f"{outdir}/spatial_{prefix}.png", dpi=900, bbox_inches="tight")
    plt.close(fig)

    # # 2) UMAP
    # umap_fig = sc.pl.umap(
    #     adata,
    #     color=f"{cluster_col}_label",
    #     size=size_pts,
    #     show=False,
    #     return_fig=True,
    #     title=f"UMAP clusters ({title_suffix})"
    # )
    # umap_fig.savefig(f"{outdir}/umap_{prefix}.png", dpi=150, bbox_inches="tight")
    # plt.close(umap_fig)
    #
    # # 3) t-SNE
    # tsne_fig = sc.pl.tsne(
    #     adata,
    #     color=f"{cluster_col}_label",
    #     size=size_pts,
    #     show=False,
    #     return_fig=True,
    #     title=f"t-SNE clusters ({title_suffix})"
    # )
    # tsne_fig.savefig(f"{outdir}/tsne_{prefix}.png", dpi=150, bbox_inches="tight")
    # plt.close(tsne_fig)

# ─── 7. Clustering + plotting helper ───────────────────────────────────────
def cluster_and_plot(adata, label, n_neighbors,n_pcs,leiden_res,out_dir):
    # 1) Preprocess & cluster
    #
    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=n_pcs
    )
    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    # Step 2: KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    adata.obs["kmeans"] = kmeans.fit_predict(adata.obsm["X_pca"]).astype(str)

    # ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # 2) Save cluster labels
    adata.obs[["kmeans"]].to_csv(f"{out_dir}/clusters_{label}.csv")

    # 3) Spatial plot
    cluster_col = "kmeans"
    size_pts = 2

    # Build mapping from cluster → 'Cluster X (n=###)'
    cluster_counts = adata.obs[cluster_col].value_counts().sort_index()
    label_map = {
        str(cl): f"Cluster {cl} (n={count})"
        for cl, count in cluster_counts.items()
    }

    # Use this to relabel cluster column for plotting
    adata.obs[f"{cluster_col}_label"] = adata.obs[cluster_col].map(label_map)

    n_cells = adata.n_obs
    title_suffix = f"{label} — {n_cells:,} cells"

    # 1) Spatial
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.invert_yaxis()
    ax.set_aspect('equal')
    sc.pl.scatter(
        adata,
        x="x_centroid", y="y_centroid",
        color=f"{cluster_col}_label",
        size=size_pts,
        ax=ax,
        show=False,
        title=f"Spatial clusters ({title_suffix})",
    )
    fig.savefig(f"{out_dir}/spatial_{label}.png", dpi=900,
                bbox_inches="tight")
    plt.close(fig)

    # # 4) UMAP
    # fig = sc.pl.umap(
    #     adata,
    #     color="kmeans",
    #     size=5,
    #     title=f"UMAP clusters ({label})",
    #     show=False
    # )
    # fig.figure.savefig(f"{out_dir}/umap_{label}.png", dpi=600,
    #                    bbox_inches="tight")
    # plt.close(fig.figure)
    #
    # # 5) t-SNE
    # fig = sc.pl.tsne(
    #     adata,
    #     color="kmeans",
    #     size=5,
    #     title=f"t-SNE clusters ({label})",
    #     show=False
    # )
    # fig.figure.savefig(f"{out_dir}/tsne_{label}.png", dpi=600,
    #                    bbox_inches="tight")
    # plt.close(fig.figure)


def plot_strong_weak_comparison(adata,out_dir):
    # Step 1: Sort cells by combined score
    adata_sorted = adata[adata.obs["combined_score"].sort_values(ascending=False).index]

    # Step 2: Select top and bottom N cells
    N = 7493
    # N = 1000
    top_cells = adata_sorted[:N]
    bottom_cells = adata_sorted[-N:]

    # Step 3: Concatenate with strength labels (optional)
    adata_subset = top_cells.concatenate(bottom_cells, batch_key="strength", batch_categories=["strong", "weak"])

    # Step 4: Extract raw expression for HVGs
    hvg_mask = adata_subset.var["highly_variable"]
    X = adata_subset[:, hvg_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Step 5: Optional clipping (min-max)
    X = np.clip(X, 0, np.percentile(X, 99))  # suppress outlier values
    # Step 7: Plot
    plt.figure(figsize=(8, 14))
    ax = sns.heatmap(
        X,
        cmap="YlGnBu",  # perceptually friendly sequential colormap
        xticklabels=adata_subset[:, hvg_mask].var_names.tolist(),
        yticklabels=False,
        cbar_kws={
            "label": "Raw gene expression",
            "shrink": 0.4,
        },
        linewidths=0,
        linecolor=None
    )
    # Adjust the colorbar width

    # Adjust font size of the colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Raw gene expression", fontsize=12)  # ← Adjust fontsize here
    plt.xticks(rotation=45, fontsize=10)
    # plt.title("Top 1000 Strong vs Bottom 1000 Weak Cells", fontsize=18)
    # plt.xlabel("Highly Variable Genes", fontsize=12)
    plt.xlabel("Senesence Gene Markers", fontsize=12)
    plt.ylabel("Cells (Top = Strong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/senesence_comparison.png", dpi=600,
                       bbox_inches="tight")
    plt.show()

def get_strong_cells(adata,method):
    if method == "threshold":
        strong_thresh = 100
        adata_strong = adata[adata.obs["total_counts"] > strong_thresh].copy()

    elif method == "quantile":
        q = adata.obs["total_counts"].quantile(0.5)
        adata_strong = adata[adata.obs["total_counts"] > q].copy()

    elif method == "mad":
        median = np.median(adata.obs["n_genes"])
        mad = np.median(np.abs(adata.obs["n_genes"] - median))
        cutoff = median + 1 * mad
        adata_strong = adata[adata.obs["n_genes"] > cutoff].copy()

    elif method == "gmm":
        # Step 1: log-transform total counts
        log_counts = np.log1p(adata.obs["total_counts"].values).reshape(-1, 1)

        # Step 2: Fit 2-component GMM
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
        labels = gmm.fit_predict(log_counts)
        means = gmm.means_.flatten()

        # Step 3: identify which component is “strong”
        strong_component = np.argmax(means)

        # Step 4: assign GMM-based label
        adata.obs["gmm_label"] = labels
        adata_strong = adata[adata.obs["gmm_label"] == strong_component].copy()
    elif method =="hvg":
        # Step 1: Normalize and log-transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Step 2: Compute highly variable genes
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
        # Step 4: Get HVG signal per cell
        adata_hvg = adata[:, adata.var["highly_variable"]]
        adata.obs["hvg_total"] = adata_hvg.X.sum(axis=1).A1 if hasattr(adata_hvg.X, "A1") else adata_hvg.X.sum(
            axis=1)

        # Step 5: Compute combined score (e.g., product or weighted sum)
        # Normalize each to [0, 1] range first
        hvg_norm = (adata.obs["hvg_total"] - adata.obs["hvg_total"].min()) / (
                    adata.obs["hvg_total"].max() - adata.obs["hvg_total"].min())
        count_norm = (adata.obs["total_counts"] - adata.obs["total_counts"].min()) / (
                    adata.obs["total_counts"].max() - adata.obs["total_counts"].min())

        # Combined score: weighted sum (can also try product)
        adata.obs["combined_score"] = 0.5 * hvg_norm + 0.5 * count_norm
        # plot_distribution_strong_weak(adata)


        # Step 6: Select top 70% of cells
        n_total = adata.n_obs
        n_keep = int(0.5 * n_total)

        top_cells = adata.obs.sort_values("combined_score", ascending=False).head(n_keep).index
        adata_strong = adata[top_cells].copy()
        # Select all cells not in the top N
        weak_cells = adata.obs.index.difference(top_cells)
        adata_weak = adata[weak_cells].copy()
    elif method =="senesence":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Step 2: Define and filter senescence genes
        senescence_genes = ["IL6", "IGFBP3", "EGFR", "SERPINE1", "IGFBP1", "IGFBP7", "FAS", "FGF2", "VEGFA", "CDKN1A",
                            "CDKN2A", "STAT1", "TNFRSF10C", "PARP1", "CXCL8", "IL1A", "CXCL1", "ICAM1", "CCL2",
                            "IGFBP2",
                            "AXL", "WNT2", "HMGB2", "HMGB1", "IGFBP5", "GDF15", "MDM2", "CDKN2B", "CCNA2", "CDK1",
                            "HELLS",
                            "FOXM1", "BUB1B", "LMNB1", "BRCA1", "IGF1", "JUN", "MIF", "TGFB1"]
        adata.var["highly_variable"] = adata.var_names.isin(senescence_genes)
        # Step 4: Get HVG signal per cell
        adata_hvg = adata[:, adata.var["highly_variable"]]
        adata.obs["hvg_total"] = adata_hvg.X.sum(axis=1).A1 if hasattr(adata_hvg.X, "A1") else adata_hvg.X.sum(
            axis=1)

        # Step 5: Compute combined score (e.g., product or weighted sum)
        # Normalize each to [0, 1] range first
        hvg_norm = (adata.obs["hvg_total"] - adata.obs["hvg_total"].min()) / (
                adata.obs["hvg_total"].max() - adata.obs["hvg_total"].min())
        count_norm = (adata.obs["total_counts"] - adata.obs["total_counts"].min()) / (
                adata.obs["total_counts"].max() - adata.obs["total_counts"].min())

        # Combined score: weighted sum (can also try product)
        adata.obs["combined_score"] = 0.5 * hvg_norm + 0.5 * count_norm
        # plot_distribution_strong_weak(adata)

        # Step 6: Select top 70% of cells
        n_total = adata.n_obs
        n_keep = int(0.5 * n_total)

        top_cells = adata.obs.sort_values("combined_score", ascending=False).head(n_keep).index
        adata_strong = adata[top_cells].copy()
        # Select all cells not in the top N
        weak_cells = adata.obs.index.difference(top_cells)
        adata_weak = adata[weak_cells].copy()
    else:
        raise ValueError(f"Unknown method: {method}")
    return adata,adata_strong,adata_weak

def run(base_path,dataset_name,out_dir):
    # ─── 1. Parameters ─────────────────────────────────────────────────────────
    strong_thresh = 100
    n_neighbors = 10
    n_pcs = 40
    leiden_res = 0.1
    # ─── 2. Load ───────────────────────────────────────────────────────────────
    cell_path = base_path + dataset_name + '/cells.csv.gz'
    cells = pd.read_csv(cell_path)
    cell_centers = cells[['cell_id', 'x_centroid', 'y_centroid']]
    cell_centers = cell_centers.set_index("cell_id")[["x_centroid", "y_centroid"]]

    clusters_file = base_path+dataset_name+'/clustering/gene_expression_kmeans_5_clusters/clusters.csv'
    clusters = pd.read_csv(clusters_file, index_col=0)


    # ─── 4. Positional “join” ──────────────────────────────────────────────────
    adata = sc.read_10x_h5(base_path + dataset_name + 'cell_feature_matrix.h5')
    adata.obs["Cluster"] = (
        clusters
        .reindex(adata.obs_names)["Cluster"]
        .astype(str)
    )
    print(adata.obs["Cluster"].value_counts().head())

    adata.obs.index = adata.obs.index.astype(str)
    cell_centers.index = cell_centers.index.astype(str)

    # 3) Join by index (obs_names ↔ cell_id)
    adata.obs = adata.obs.join(cell_centers, how="left")


    # ─── 5. QC metrics ─────────────────────────────────────────────────────────
    # total counts & gene counts
    X = adata.X
    if hasattr(X, "A1"):
        adata.obs["total_counts"] = X.sum(axis=1).A1
        adata.obs["n_genes"] = (X > 0).sum(axis=1).A1
    else:
        adata.obs["total_counts"] = X.sum(axis=1)
        adata.obs["n_genes"] = (X > 0).sum(axis=1)

    # ─── 6. Subset “strong” vs. “all” ──────────────────────────────────────────
    # method = "quantile"  # choose: "threshold", "quantile", "mad", "knee", "gmm"
    # for method in ["quantile", "threshold", "quantile", "mad",  "gmm","hvg"]:
    method = "hvg"
    adata,adata_strong,adata_weak = get_strong_cells(adata,method)
    plot_strong_weak_comparison(adata,out_dir)
    # # ─── 8. Run for strong & all ───────────────────────────────────────────────
    # print("ploting strong cells")
    plot_known_clusters(adata,"all",out_dir)
    plot_known_clusters(adata_strong, "strong", out_dir)
    plot_known_clusters(adata_weak,"weak",out_dir)
    # print('done')
    # test = input()
    # print("ploting all cells")
    # plot_known_clusters(adata_all, "strong", out_dir)
    # print('done')

    # cluster_and_plot(adata, "all", n_neighbors, n_pcs, leiden_res, out_dir)
    # # print('saved')
    # # test = input()
    # # print("Clustering STRONG cells …")
    # cluster_and_plot(adata_strong, "strong",n_neighbors,n_pcs,leiden_res,out_dir)
    # cluster_and_plot(adata_weak, "weak", n_neighbors, n_pcs, leiden_res, out_dir)
    #
    # #
    # # print("Clustering ALL cells …")





# -----------------  Paths to Data  ----------------- #
base_path = '/media/huifang/data/Xenium/sennet_data/'
dataset_name = '1720/'
path_to_save = base_path+dataset_name+'gene_cluster_result/'
run(base_path,dataset_name,path_to_save)


# visualize_dotplot(ad_viz)
# # # -----------------  select & visualization meaningful classes----------------- #
# order = ['0', '3', '2', '4']
# visualize_dotplot(ad_viz,order)
# # # -----------------  spatial visualization----------------- #
# visualize_spatial_distribution(ad_viz,order)










