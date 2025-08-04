import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
#import pySingleCellNet as cn
from scipy import sparse
import igraph as ig
from typing import Dict, List, Union
#from .adataTools import find_elbow
from anndata import AnnData

def build_gene_knn_graph(
    adata,
    mask_var: str = None,
    mean_cluster: bool = True,
    groupby: str = 'leiden',
    knn: int = 5,
    use_knn: bool = True,
    metric: str = "euclidean",
    key: str = "gene"
):
    """
    Construct a gene–gene k-nearest neighbor (kNN) graph and store the results in `adata.uns`.

    Computes distances and connectivities between genes based on their expression profiles 
    across cells or cell clusters. The result is stored as sparse matrices for use in 
    downstream visualization or network analysis.

    Args:
        adata (AnnData): 
            Annotated data matrix of shape (cells × genes). Internally transposed 
            to (genes × cells) for graph construction.
        mask_var (str, optional): 
            Column name in `adata.var` containing boolean values. Only genes for which 
            `adata.var[mask_var] == True` will be included. If `None`, all genes are used.
        mean_cluster (bool, optional): 
            If `True`, expression values are averaged across clusters defined in 
            `adata.obs[groupby]`, and distances are computed using the gene × cluster 
            matrix instead of gene × cell matrix. Defaults to `True`.
        groupby (str, optional): 
            Column in `adata.obs` specifying cluster labels. Used only if `mean_cluster=True`. 
            Defaults to `'leiden'`.
        knn (int, optional): 
            Number of nearest neighbors to compute for each gene. Passed to `sc.pp.neighbors` 
            as `n_neighbors`. Defaults to `5`.
        use_knn (bool, optional): 
            Whether to use a hard kNN graph (`True`) or a Gaussian-weighted graph (`False`). 
            Passed to `sc.pp.neighbors` as `knn`. Defaults to `True`.
        metric (str, optional): 
            Distance metric for computing pairwise gene distances. Can be any metric 
            accepted by `sc.pp.neighbors`, such as `"euclidean"`, `"manhattan"`, or `"correlation"`.
            Note: If `"correlation"` is used on a sparse matrix, the matrix will be densified. 
            Defaults to `"euclidean"`.
        key (str, optional): 
            Prefix for storing results in `adata.uns`. The following keys will be created:
                - `adata.uns[f"{key}_gene_index"]`: list of included gene names.
                - `adata.uns[f"{key}_connectivities"]`: sparse matrix of kNN connectivities.
                - `adata.uns[f"{key}_distances"]`: sparse matrix of distances between genes.
            Defaults to `"gene"`.

    Returns:
        None: 
            Results are stored in-place in `adata.uns` under the specified `key`.
    """

    # 1) Work on a shallow copy so we don’t overwrite adata.X prematurely
    adata_work = adata.copy()

    # 2) If mask_var is provided, subset to only those genes first
    if mask_var is not None:
        if mask_var not in adata_work.var.columns:
            raise ValueError(f"Column '{mask_var}' not found in adata.var.")
        gene_mask = adata_work.var[mask_var].astype(bool)
        selected_genes = adata_work.var.index[gene_mask].tolist()
        if len(selected_genes) == 0:
            raise ValueError(f"No genes found where var['{mask_var}'] is True.")
        adata_work = adata_work[:, selected_genes].copy()

    # 3) If mean_cluster=True, aggregate by cluster label in `groupby`
    if mean_cluster:
        if groupby not in adata_work.obs.columns:
            raise ValueError(f"Column '{groupby}' not found in adata.obs.")
        # Aggregate each cluster to its mean expression; stored in .layers['mean']
        adata_work = sc.get.aggregate(adata_work, by=groupby, func='mean')
        # Overwrite .X with the mean‑expression matrix
        adata_work.X = adata_work.layers['mean']

    # 4) Transpose so that each gene (or cluster‑mean) is one “observation”
    adata_genes = adata_work.T.copy()

    # 5) If metric=="correlation" and X is sparse, convert to dense
    if metric == "correlation" and sparse.issparse(adata_genes.X):
        adata_genes.X = adata_genes.X.toarray()

    # 6) Compute neighbors on the (genes × [cells or clusters]) matrix.
    #    Pass n_neighbors=knn and knn=use_knn. Default method selection in Scanpy will
    #    use 'umap' if use_knn=True, and 'gauss' if use_knn=False.
    sc.pp.neighbors(
        adata_genes,
        n_neighbors=knn,
        knn=use_knn,
        metric=metric,
        use_rep="X"
    )

    # 7) Extract the two sparse matrices from adata_genes.obsp:
    conn = adata_genes.obsp["connectivities"].copy()  # CSR: gene–gene adjacency weights
    dist = adata_genes.obsp["distances"].copy()       # CSR: gene–gene distances

    # 8) Record the gene‑order (after masking + optional aggregation)
    gene_index = np.array(adata_genes.obs_names)

    adata.uns[f"{key}_gene_index"]      = gene_index
    adata.uns[f"{key}_connectivities"] = conn
    adata.uns[f"{key}_distances"]      = dist


def find_similar_genes(
    adata,
    gene: str,
    n_neighbors: int = 5,
    key: str = "gene",
    use: str = "connectivities"
):
    """
    Find the top `n_neighbors` most similar genes to a given gene using a precomputed gene–gene graph.

    This function retrieves the nearest genes to the query gene using either a 
    connectivity or distance matrix stored in `adata.uns`. The graph must have 
    been generated by `build_gene_knn_graph()`.

    Supports both sparse CSR matrices and dense NumPy arrays.

    Args:
        adata (AnnData): 
            Annotated data matrix. Must contain:
                - `adata.uns[f"{key}_gene_index"]`: Array of gene names.
                - `adata.uns[f"{key}_connectivities"]`: Gene–gene connectivity matrix.
                - `adata.uns[f"{key}_distances"]`: Gene–gene distance matrix.
        gene (str): 
            Gene name for which to find similar genes. Must be present in 
            `adata.uns[f"{key}_gene_index"]`.
        n_neighbors (int, optional): 
            Number of most similar genes to return. Defaults to `5`.
        key (str, optional): 
            Prefix under which the kNN graph is stored in `adata.uns`. For example,
            if `key="gene"`, the function uses:
                - `adata.uns["gene_gene_index"]`
                - `adata.uns["gene_connectivities"]`
                - `adata.uns["gene_distances"]`
            Defaults to `"gene"`.
        use (str, optional): 
            Similarity measure to use. One of:
                - `"connectivities"`: rank by descending connectivity weight.
                - `"distances"`: rank by ascending distance, considering only nonzero entries.
            Defaults to `"connectivities"`.

    Returns:
        List[str]: 
            A list of gene names (length ≤ `n_neighbors`) that are closest to the input gene.
    """

    if use not in ("connectivities", "distances"):
        raise ValueError("`use` must be either 'connectivities' or 'distances'.")

    idx_key = f"{key}_gene_index"
    conn_key = f"{key}_connectivities"
    dist_key = f"{key}_distances"

    if idx_key not in adata.uns:
        raise ValueError(f"Could not find `{idx_key}` in adata.uns.")
    if conn_key not in adata.uns or dist_key not in adata.uns:
        raise ValueError(f"Could not find `{conn_key}` or `{dist_key}` in adata.uns.")

    gene_index = np.array(adata.uns[idx_key])
    if gene not in gene_index:
        raise KeyError(f"Gene '{gene}' not found in {idx_key}.")
    i = int(np.where(gene_index == gene)[0][0])

    # Select the appropriate stored matrix (could be sparse CSR or dense ndarray)
    mat_key = conn_key if use == "connectivities" else dist_key
    stored = adata.uns[mat_key]

    # If stored is a NumPy array, treat it as a dense full matrix:
    if isinstance(stored, np.ndarray):
        row_vec = stored[i].copy()
        # Exclude self
        if use == "connectivities":
            row_vec[i] = -np.inf
            order = np.argsort(-row_vec)  # descending
        else:
            row_vec[i] = np.inf
            order = np.argsort(row_vec)   # ascending
        topk = order[:n_neighbors]
        return [gene_index[j] for j in topk]

    # Otherwise, assume stored is a sparse matrix (CSR or similar):
    if not sparse.issparse(stored):
        raise TypeError(f"Expected CSR or ndarray for `{mat_key}`, got {type(stored)}.")

    row = stored.getrow(i)
    # For connectivities: sort nonzero entries by descending weight
    if use == "connectivities":
        cols = row.indices
        weights = row.data
        mask = cols != i
        cols = cols[mask]
        weights = weights[mask]
        if weights.size == 0:
            return []
        order = np.argsort(-weights)
        topk = cols[order][:n_neighbors]
        return [gene_index[j] for j in topk]

    # For distances: sort nonzero entries by ascending distance
    else:  # use == "distances"
        cols = row.indices
        dists = row.data
        mask = cols != i
        cols = cols[mask]
        dists = dists[mask]
        if dists.size == 0:
            return []
        order = np.argsort(dists)
        topk = cols[order][:n_neighbors]
        return [gene_index[j] for j in topk]


def score_gene_modules(
    adata,
    gene_dict: dict,
    key_added: str = "module_scores"
):

    # Number of cells and clusters
    n_cells = adata.shape[0]
    # Initialize an empty matrix for scores
    # scores_matrix = np.zeros((n_cells, n_clusters))
    scores_df = pd.DataFrame(index=adata.obs_names)
    # For each cluster, calculate the gene scores and store in the DataFrame
    for cluster, genes in gene_dict.items():
        score_name = cluster
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_name, use_raw=False)        
        # Store the scores in the DataFrame
        scores_df[score_name] = adata.obs[score_name].values
        del(adata.obs[score_name])
    # Assign the scores DataFrame to adata.obsm
    obsm_name = key_added
    adata.obsm[obsm_name] = scores_df


def find_knn_modules(
    adata,
    adjacency,
    leiden_resolution: float = 0.5,
    prefix: str = 'gmod_',
    metric='euclidean'
):
    """
    Finds gene modules based on required passed adjacency matrix produced with build_gene_knn_graph 
    clustering with Leiden. Results are written to adata.uns['knn_modules'] in-place.

    Parameters:
    -----------
    adata
        AnnData object to process.
    adjacency
        Adjacency matrix
    leiden_resolution
        Resolution parameter for the Leiden clustering.
    prefix
        Prefix to add to each module name.
    metric
        Distance metric for kNN computation (e.g. 'euclidean', 'manhattan', 'correlation', etc.).
        If metric=='correlation', we force a dense array when the data are sparse.
    """
    # 1) Work on a copy so we don’t modify the user’s adata.X before we’re ready
    adata_subset = adata.copy()

    # 2) subset genes
    # I need to come back to this to get the genes from the {key}_connectivities matrix
    # selected = adata_subset.var.index[gene_mask].tolist()
    # if len(selected) == 0:
    #     raise ValueError(f"No genes found where var['{mask_var}'] is True.")
    # adata_subset = adata_subset[:, selected].copy()

    # 3) Transpose so that genes (or cluster‐means) become observations
    adata_transposed = adata_subset.T.copy()

    # 5) If the metric is 'correlation' and X is sparse, convert to dense
# bnot sure if needed
    if metric == 'correlation' and sparse.issparse(adata_transposed.X):
        adata_transposed.X = adata_transposed.X.toarray()

    # 6) Build the kNN graph directly on .X (no PCA)
    ### sc.pp.neighbors(adata_transposed, n_neighbors=knn, metric=metric, n_pcs=0)

    # 7) Leiden clustering on that graph
    sc.tl.leiden(adata_transposed, adjacency = adjacency, resolution=leiden_resolution)

    # 8) Group by Leiden label to collect modules
    clusters = (
        adata_transposed.obs
        .groupby('leiden', observed=True)['leiden']
        .apply(lambda ser: ser.index.tolist())
        .to_dict()
    )
    modules = {f"{prefix}{cluster_id}": gene_list for cluster_id, gene_list in clusters.items()}

    # 9) Write modules back to the original AnnData (in-place)
    adata.uns['knn_modules'] = modules





def what_module_has_gene(
    adata,
    target_gene,
    mod_slot='knn_modules'
) -> list: 
    if mod_slot not in adata.uns.keys():
        raise ValueError(mod_slot + " have not been identified.")
    genemodules = adata.uns[mod_slot]
    return [key for key, genes in genemodules.items() if target_gene in genes]




