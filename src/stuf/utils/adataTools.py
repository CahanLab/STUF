import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from typing import Dict, List, Optional, Sequence
from scipy.sparse import issparse
import igraph as ig
import scipy.sparse as sp

def get_union_detected_genes(file_list, detection_fraction=0.05):
    """
    Compute the union of genes detected across multiple `.h5ad` files.

    A gene is considered "detected" in a file if it is expressed in at least 
    `detection_fraction` of the cells in that file. The function returns the 
    union of all such genes across the given files.

    Args:
        file_list (List[str]): 
            List of file paths to `.h5ad` files to be processed.
        detection_fraction (float, optional): 
            Minimum fraction of cells in which a gene must be detected to be included 
            for that file. Defaults to `0.05`.

    Returns:
        List[str]: 
            Alphabetically sorted list of gene names that meet the detection threshold 
            in at least one file.
    """

    detected_genes_sets = []

    for file in file_list:
        adata = sc.read_h5ad(file)

        gene_detected = (adata.X > 0).astype(int)
        gene_fraction = np.array(gene_detected.sum(axis=0)).flatten() / adata.n_obs
        detected_genes = adata.var_names[gene_fraction >= detection_fraction]
        
        detected_genes_sets.append(set(detected_genes))

    return sorted(set.union(*detected_genes_sets))



def update_removed_cells(
    adata: AnnData,
    adx: AnnData,
    nickname_value: str,
    obs_cols: Sequence[str],
    update_value: str = 'unassigned'
) -> None:
    """
    Update metadata for cells removed during filtering based on a nickname.

    Identifies cells in `adata` that were assigned `nickname_value` in the `nickname` column 
    but are no longer present in `adx` (a filtered version of `adata`). 
    These "removed" cells are updated in place by:
      - Setting their `nickname` to `NA`
      - Overwriting specified columns in `.obs` with `update_value`

    Args:
        adata (AnnData): 
            The original (unfiltered) AnnData object.
        adx (AnnData): 
            The filtered AnnData object after QC or subsetting.
        nickname_value (str): 
            The value in `adata.obs['nickname']` used to identify cells of interest (e.g., "E12_HL_6").
        obs_cols (Sequence[str]): 
            List of column names in `.obs` to update for the removed cells.
        update_value (str, optional): 
            The value to assign to the specified columns for removed cells. Defaults to `'unassigned'`.

    Raises:
        ValueError: 
            If required columns are missing in `adata.obs` or if `nickname` is not present.
    
    Returns:
        None: 
            The function updates `adata` in place.
    """

    # 0) sanity checks
    if 'nickname' not in adata.obs.columns:
        raise ValueError("Column 'nickname' not found in adata.obs")
    missing = [c for c in obs_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Columns not found in adata.obs: {missing}")

    # 1) Cells that started with that nickname
    mask_orig = adata.obs['nickname'] == nickname_value
    orig_cells = set(adata.obs_names[mask_orig])

    # 2) Cells that remain
    kept_cells = set(adx.obs_names)

    # 3) Those that were removed
    removed = list(orig_cells - kept_cells)
    if not removed:
        return

    # 4) Update nickname → NA
    adata.obs.loc[removed, 'nickname'] = pd.NA

    # 5) Update each specified obs column → update_value
    for col in obs_cols:
        adata.obs.loc[removed, col] = update_value




def summarize_obs_by_group(
    adata: AnnData,
    group_key: str = 'nickname',
    obs_keys: List[str] = ['n_genes_by_counts','total_counts','pct_counts_ribo', 'pct_counts_mt'],
) -> pd.DataFrame:
    """
    Summarize `.obs` metadata by groups in an AnnData object.

    Groups cells by `group_key` and computes the mean of each column in `obs_keys`, 
    along with the number of cells in each group.

    Args:
        adata (AnnData): 
            Annotated data matrix with `.obs` containing both the grouping variable 
            and the observation variables to summarize.
        group_key (str, optional): 
            Column in `adata.obs` to group by (e.g., sample name or condition). 
            Defaults to `'nickname'`.
        obs_keys (List[str], optional): 
            List of column names in `.obs` to average within each group. 
            Defaults to `['n_genes_by_counts','total_counts','pct_counts_ribo', 'pct_counts_mt']`.

    Returns:
        pd.DataFrame: 
            DataFrame with one row per group and columns:
              - `group_key`: the group label
              - `n_cells`: number of cells in the group
              - One column for each `obs_key` with the groupwise mean value

    Raises:
        ValueError: 
            If `group_key` is missing in `.obs`, if `obs_keys` is not a list or is empty, 
            or if any `obs_key` is missing in `.obs`.
    """

    # -- error checking --
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")
    if not isinstance(obs_keys, (list, tuple)) or len(obs_keys) == 0:
        raise ValueError("`obs_keys` must be a non-empty list of column names")
    missing = [k for k in obs_keys if k not in adata.obs]
    if missing:
        raise ValueError(f"obs_keys not found in adata.obs: {missing}")

    # -- assemble DataFrame --
    df = adata.obs[[group_key] + obs_keys].copy()

    # -- group, count, and mean --
    grouped = df.groupby(group_key)
    counts = grouped.size().rename("n_cells")
    means = grouped[obs_keys].mean()

    # -- combine into one result table --
    summary_df = pd.concat([counts, means], axis=1).reset_index()
    return summary_df


def add_nickname(
    adata,
    prefix: str,
    mapping: dict[tuple, str],
    cols: list[str] = ["section", "part"],
    new_col: str = "nickname",
    default: str | None = None,
):
    # build the tuple‐series
    tuples = pd.Series(
        list(zip(*(adata.obs[c] for c in cols))),
        index=adata.obs_names,
    )
    # map to suffixes
    suffix = tuples.map(mapping)
    # only fill if default is not None
    if default is not None:
        suffix = suffix.fillna(default)
    # now build the final nicknames,
    # leaving unmapped cells as None
    adata.obs[new_col] = suffix.map(
        lambda s: f"{prefix}_{s}" if pd.notna(s) else None
    )
    return adata

