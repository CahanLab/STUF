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
    Given a list of .h5ad files, return the union of genes that are detected
    in at least `detection_fraction` of cells in each file.

    Parameters
    ----------
    file_list : list of str
        Paths to .h5ad files to process.
    detection_fraction : float
        Fraction of cells in which a gene must be detected to be included.

    Returns
    -------
    list of str
        Sorted list of gene names detected in at least `detection_fraction` of cells in at least one file.
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
    In `adata`, find cells that had `nickname == nickname_value` but
    were dropped (i.e. not in `adx`), then set their `nickname` to NA
    and their `obs_cols` to `update_value`. Updates in place.

    Parameters
    ----------
    adata
        Original AnnData before subsetting.
    adx
        Filtered AnnData after QC/subsetting.
    nickname_value
        The nickname you originally filtered on (e.g. "E12_HL_6").
    obs_cols
        Which obs columns to overwrite for those removed cells.
    update_value
        What to set those columns to (default "unassigned").

    Raises
    ------
    ValueError
        If required columns are missing.
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
    Group an AnnData's obs by `group_key`, compute per-group averages of `obs_keys`, 
    and count cells per group.

    Parameters
    ----------
    adata
        AnnData with .obs containing the grouping column and the obs_keys.
    group_key
        Column name in adata.obs to group cells by.
    obs_keys
        List of column names in adata.obs to average.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per group, with columns:
          - group_key
          - n_cells
          - <each obs_key> (the mean within that group)

    Raises
    ------
    ValueError
        If group_key missing, obs_keys is empty or not a list, or any obs_key missing.
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

