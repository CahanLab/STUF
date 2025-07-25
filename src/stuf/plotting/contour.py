import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from .helpers import _smooth_contour, _temp_plt_axes, make_bivariate_cmap
from matplotlib.colors import to_hex, ListedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Union, Sequence, Callable, Optional, Dict, Tuple
import math
import warnings
from ..config import DEFAULTS_SCATTER, DEFAULTS_CBAR 

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def add_region_hulls(
    ax,
    adata,
    regions_key: str,
    spatial_key: str = 'X_spatial',
    hull_kwargs: dict = None
):
    """
    Outline each region (from adata.obs[regions_key]) with its convex hull.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which you’ve already plotted your scatter.
    adata : AnnData
        Annotated data containing a 2D embedding in adata.obsm[spatial_key].
    regions_key : str
        Column in adata.obs holding region labels (categorical or string).
    spatial_key : str, optional
        Key in adata.obsm for the (n_obs,2) coords. Default 'X_spatial'.
    hull_kwargs : dict, optional
        Passed to matplotlib.patches.Polygon, e.g.
          {
            'edgecolor': 'black',
            'linewidth': 2,
            'alpha': 0.8,
            'facecolor': 'none',
            'zorder': 3
          }
        If None, defaults are used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same Axes, with hulls overlaid.
    """
    # defaults for polygon styling
    if hull_kwargs is None:
        hull_kwargs = {
            'edgecolor': 'black',
            'linewidth': 2,
            'alpha': 0.8,
            'facecolor': 'none',
            'zorder': 3
        }

    # 1) get coords & labels
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be (n_obs,2).")
    x, y = coords[:,0], coords[:,1]
    labels = adata.obs[regions_key].astype(str).values
    unique_labels = np.unique(labels)

    # 2) for each region, compute hull
    for lab in unique_labels:
        mask = labels == lab
        pts = np.column_stack((x[mask], y[mask]))
        # need at least 3 points for a hull
        if pts.shape[0] < 3:
            continue
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]  # in CCW order

        # 3) add the polygon
        poly = Polygon(hull_pts, **hull_kwargs)
        ax.add_patch(poly)

    return ax



def spatial_contours(
    adata: AnnData,
    genes: Union[str, Sequence[str]],
    spatial_key: str = 'spatial',
    summary_func: Callable[[np.ndarray], np.ndarray] = np.mean,
    spot_size: float = 30,
    alpha: float = 0.8,
    log_transform: bool = True,
    clip_percentiles: tuple = (1, 99),
    cmap: str = 'viridis',
    contour_kwargs: dict = None,
    scatter_kwargs: dict = None
) -> None:
    """Scatter spatial expression of one or more genes with smooth contour overlay.

    If multiple genes are provided, each is preprocessed (log1p → clip
    → normalize), then combined per cell via `summary_func` (e.g. mean, sum,
    max) on the normalized values. A smooth contour of the summarized signal
    is overlaid onto the spatial scatter.

    Args:
        adata: AnnData with spatial coordinates in `adata.obsm[spatial_key]`.
        genes: Single gene name or list of gene names to plot (must be in `adata.var_names`).
        spatial_key: Key in `.obsm` for an (n_obs, 2) coords array.
        summary_func: Function to combine multiple normalized gene arrays
            (takes an (n_obs, n_genes) array, returns length-n_obs array).
            Defaults to `np.mean`.
        spot_size: Scatter marker size.
        alpha: Scatter alpha transparency.
        log_transform: If True, apply `np.log1p` to raw expression before clipping.
        clip_percentiles: Tuple `(low_pct, high_pct)` percentiles to clip each gene.
        cmap: Colormap name for the scatter (e.g. 'viridis').
        contour_kwargs: Dict of parameters for smoothing & contouring:
            - levels: int or list of levels (default 6)
            - grid_res: int grid resolution (default 200)
            - smooth_sigma: float Gaussian blur sigma (default 2)
            - contour_kwargs: dict of line style kwargs (default {'colors':'k','linewidths':1})
        scatter_kwargs: Extra kwargs passed to `ax.scatter`.

    Raises:
        ValueError: If any gene is missing or spatial coords are malformed.
    """
    # ensure genes is list
    gene_list = [genes] if isinstance(genes, str) else list(genes)
    for g in gene_list:
        if g not in adata.var_names:
            raise ValueError(f"Gene '{g}' not found in adata.var_names.")

    # helper to extract numpy
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()

    # preprocess each gene: extract, log1p, clip, normalize to [0,1]
    normed = []
    for g in gene_list:
        vals = _get_array(adata[:, g].X)
        if log_transform:
            vals = np.log1p(vals)
        lo, hi = np.percentile(vals, clip_percentiles)
        vals = np.clip(vals, lo, hi)
        normed.append((vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals))
    # stack into (n_obs, n_genes)
    M = np.column_stack(normed)
    # summarize across genes
    summary = summary_func(M, axis=1)

    # fetch spatial coords
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be an (n_obs, 2) array.")
    x, y = coords[:, 0], coords[:, 1]

    # scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    sc_kw = {"c": summary, "cmap": cmap, "s": spot_size, "alpha": alpha}
    if scatter_kwargs:
        sc_kw.update(scatter_kwargs)
    sc = ax.scatter(x, y, **sc_kw)
    ax.set_aspect('equal')
    title = (
        gene_list[0] if len(gene_list) == 1
        else f"{len(gene_list)} genes ({summary_func.__name__})"
    )
    ax.set_title(f"Spatial expression: {title}")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(sc, ax=ax, label="summarized (normalized)")

    # smooth + contour
    # default contour params
    ck = {
        "levels": 6,
        "grid_res": 200,
        "smooth_sigma": 2,
        "contour_kwargs": {"colors": "k", "linewidths": 1}
    }
    if contour_kwargs:
        ck.update(contour_kwargs)
    _smooth_contour(
        x, y, summary,
        levels=ck["levels"],
        grid_res=ck["grid_res"],
        smooth_sigma=ck["smooth_sigma"],
        contour_kwargs=ck["contour_kwargs"]
    )
    plt.tight_layout()
    plt.show()
