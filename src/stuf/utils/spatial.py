import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
# from scipy import sparse
# from typing import List, Literal, Sequence,  Tuple 
from anndata import AnnData

def rotate_coordinates(adata, key, degrees):
    """
    Rotate spatial coordinates stored in adata.obsm[key] by the specified number of degrees clockwise.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing spatial coordinates.
    key : str
        Key in adata.obsm where the spatial coordinates are stored (e.g., 'spatial').
        The coordinates should be an array of shape (n_cells, 2).
    degrees : int or float
        The angle in degrees to rotate the coordinates clockwise.
    
    This function modifies adata in place and does not return anything.
    """
    # Retrieve the original coordinates
    coords = adata.obsm[key]
    
    # Convert clockwise degrees to radians (positive rotation is counterclockwise, so negate)
    angle_rad = np.deg2rad(-degrees)
    
    # Build rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    # Apply rotation to each coordinate (assuming coords shape is (n_cells, 2))
    rotated = coords.dot(rotation_matrix.T)
    
    # Update the AnnData object in place
    adata.obsm[key] = rotated



def flip_coordinates(adata, key, axis='x'):
    """
    Flip spatial coordinates stored in adata.obsm[key] along the specified axis.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing spatial coordinates.
    key : str
        Key in adata.obsm where the spatial coordinates are stored (e.g., 'spatial').
        The coordinates should be an array of shape (n_cells, 2) with columns [x, y].
    axis : str
        Axis along which to flip:
        - 'x' (default): Flip vertically (cells at the bottom move to the top). This negates the y-coordinates.
        - 'y': Flip horizontally (cells on the left move to the right). This negates the x-coordinates.

    This function modifies adata in place and does not return anything.
    """
    # Retrieve the original coordinates
    coords = adata.obsm[key]
    
    # Ensure there are two columns
    if coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape (n_cells, 2), got {coords.shape}")

    # Flip based on axis
    if axis == 'x':
        # Flip y-coordinate
        coords[:, 1] = -coords[:, 1]
    elif axis == 'y':
        # Flip x-coordinate
        coords[:, 0] = -coords[:, 0]
    else:
        raise ValueError("Axis must be either 'x' or 'y'")

    # Update the AnnData object in place
    adata.obsm[key] = coords


def annotate_spatially_variable_genes(
    adata: AnnData,
    uns_key: str = 'moranI',
    pval_column: str = 'pval_norm',
    var_pval_name: str = 'moranI_pval_norm',
    var_flag_name: str = 'spatially_variable',
    pval_cutoff: float = 0.01
) -> None:
    """ 
    Annotate `adata.var` with Moran’s I p-values and a Boolean “spatially_variable” flag.

    This function looks in `adata.uns[uns_key]` for a DataFrame whose index is
    gene names (matching `adata.var_names`). It then:
    
      1. Re-indexes that DataFrame to `adata.var_names`.
      2. Copies the specified p-value column into `adata.var[var_pval_name]`.
      3. Creates a boolean column `adata.var[var_flag_name]`, set to True
         wherever `pval < pval_cutoff`, False otherwise (and False if pval is NaN).

    Args:
        adata: Annotated data matrix with `.var_names` matching Moran’s I index.
        uns_key: Key in `adata.uns` where the Moran’s I DataFrame lives.
        pval_column: Name of the column in `adata.uns[uns_key]` holding normalized p-values.
        var_pval_name: Column name to use in `adata.var` for storing p-values.
        var_flag_name: Column name to use in `adata.var` for storing the Boolean flag.
        pval_cutoff: Genes with `pval < pval_cutoff` will be flagged True. Defaults to 0.05.

    Raises:
        KeyError: if `adata.uns[uns_key]` is missing or not a DataFrame, or if `pval_column` is not found.
        ValueError: if `adata.var_names` cannot be aligned with the Moran’s I table index.
    """
    # 1) Validate uns_key and pval_column
    if uns_key not in adata.uns:
        raise KeyError(f"adata.uns['{uns_key}'] not found. Expected a DataFrame with index=genes.")
    moran_df = adata.uns[uns_key]
    if not hasattr(moran_df, 'loc') or pval_column not in moran_df.columns:
        raise KeyError(
            f"Expected `adata.uns['{uns_key}']` to be a DataFrame with a column '{pval_column}'."
        )

    # 2) Re‐index p-values to match adata.var_names
    #    (this will insert NaN for any gene not present in uns index)
    moran_pvals = moran_df[pval_column].reindex(adata.var_names)

    # 3) Store the p-values in adata.var
    adata.var[var_pval_name] = moran_pvals.values

    # 4) Create the Boolean flag
    spatial_flag = (moran_pvals < pval_cutoff).astype(bool).values
    adata.var[var_flag_name] = spatial_flag




