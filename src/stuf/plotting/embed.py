import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from .helpers import _smooth_contour, make_bivariate_cmap, annotate_centroids
from matplotlib.colors import to_hex, ListedColormap
from typing import Union, Sequence, Callable, Optional, Dict, Tuple, List, Any
import math
import warnings
from ..config import DEFAULTS_SCATTER, DEFAULTS_CBAR, DEFAULTS_LEGEND, BIMAP_YELLOW
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def embed_bivariate_genes(
    adata: AnnData,
    genes1: List[str],
    genes2: List[str],
    cmap: ListedColormap = BIMAP_YELLOW,
    embedding_key: str = 'X_spatial',
    log_transform: bool = False,
    clip_percentiles: Tuple[float, float] = (0, 100),
    agg_func: Callable = np.mean,
    priority_metric: str = 'sum',
    show_xcoords: bool = False,
    show_ycoords: bool = False,
    show_bbox: bool = False,
    show_legend: bool = True,
    width_ratios: Tuple[float, float] = (10, 1),
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    cbar_kwargs: Optional[Dict[str, Any]] = None,
    cbar_fontsize: str = 'small',
    list_genes: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot two gene lists on a 2D embedding using a bivariate colormap.

    Parameters
    ----------
    adata
        AnnData with spatial coords in `adata.obsm[embedding_key]`.
    genes1, genes2
        Lists of gene names for the two modules.
    ...
    """
    # Determine labels based on number of genes
    module1_label = genes1[0] if len(genes1) == 1 else "List 1"
    module2_label = genes2[0] if len(genes2) == 1 else "List 2"

    # Validate gene presence
    for g in genes1 + genes2:
        if g not in adata.var_names:
            raise ValueError(f"Gene '{g}' not in adata.var_names")

    # Helper to flatten X
    def _arr(x): return x.toarray().ravel() if hasattr(x, 'toarray') else x.ravel()

    # Summarize a list
    def summarize(genelist):
        arrs = []
        for g in genelist:
            v = _arr(adata[:, g].X)
            if log_transform:
                v = np.log1p(v)
            lo, hi = np.percentile(v, clip_percentiles)
            v = np.clip(v, lo, hi)
            arrs.append((v - lo) / (hi - lo) if hi > lo else np.zeros_like(v))
        M = np.vstack(arrs).T
        return agg_func(M, axis=1)

    u, v = summarize(genes1), summarize(genes2)

    # Build LUT
    m = len(cmap.colors)
    n = int(np.sqrt(m))
    C = np.array(cmap.colors).reshape(n, n, 3)

    # Bilinear interp into LUT
    gu, gv = u*(n-1), v*(n-1)
    i0, j0 = np.floor(gu).astype(int), np.floor(gv).astype(int)
    i1, j1 = np.minimum(i0+1, n-1), np.minimum(j0+1, n-1)
    du, dv = gu-i0, gv-j0
    wa, wb = (1-du)*(1-dv), du*(1-dv)
    wc, wd = (1-du)*dv, du*dv
    c00, c10 = C[j0, i0], C[j0, i1]
    c01, c11 = C[j1, i0], C[j1, i1]
    cols = c00*wa[:,None] + c10*wb[:,None] + c01*wc[:,None] + c11*wd[:,None]
    hexcols = [to_hex(c) for c in cols]

    # Determine draw priority
    if priority_metric == 'sum': pr = u + v
    elif priority_metric == 'list1': pr = u
    elif priority_metric == 'list2': pr = v
    else: raise ValueError("priority_metric must be 'sum','list1','list2'")
    order = np.argsort(pr)

    # Coordinates
    coords = adata.obsm.get(embedding_key)
    if coords is None or coords.shape[1] != 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be (n_obs,2)")
    xy = coords[order]

    # Axes setup
    if ax is None:
        fig, (ax, ax_cb) = plt.subplots(
            1, 2, figsize=(8, 4),
            gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.3}
        )
    else:
        ax_cb = None
        fig = ax.figure

    # Scatter kwargs
    skw = DEFAULTS_SCATTER.copy()
    if scatter_kwargs:
        skw.update(scatter_kwargs)

    # Plot scatter
    ax.scatter(xy[:,0], xy[:,1], c=[hexcols[i] for i in order], **skw)
    ax.set_aspect('equal')
    if not show_xcoords: ax.tick_params(bottom=False, labelbottom=False)
    if not show_ycoords: ax.tick_params(left=False, labelleft=False)
    if not show_bbox:
        for sp in ax.spines.values(): sp.set_visible(False)

    # Colorbar
    if show_legend:
        defaults = {'width':'5%','height':'50%','loc':'upper right','borderpad':1}
        if cbar_kwargs: defaults.update(cbar_kwargs)
        if ax_cb is not None:
            ax_cb.imshow(C, origin='lower', extent=[0,1,0,1])
            ax_cb.set_xlabel(module1_label, fontsize=cbar_fontsize)
            ax_cb.set_ylabel(module2_label, fontsize=cbar_fontsize)
            ax_cb.tick_params(labelsize=cbar_fontsize)
            ax_cb.set_xticks([0,1]); ax_cb.set_yticks([0,1])
            ax_cb.set_aspect('equal')
        else:
            cbax = inset_axes(ax, **defaults)
            cbax.imshow(C, origin='lower', extent=[0,1,0,1])
            cbax.set_xlabel(module1_label, fontsize=cbar_fontsize)
            cbax.set_ylabel(module2_label, fontsize=cbar_fontsize)
            cbax.tick_params(labelsize=cbar_fontsize)
            cbax.set_xticks([0,1]); cbax.set_yticks([0,1])
            cbax.set_aspect('equal')
            cbax.patch.set_alpha(0)

    # List genes below plot
    if list_genes:
        fig.subplots_adjust(bottom=0.25)
        def _fmt(genelist):
            x = genelist[:3]
            return ', '.join(x) + ('...' if len(genelist) > 3 else '')
        txt = (
            f"{module1_label}: {_fmt(genes1)}\n"
            f"{module2_label}: {_fmt(genes2)}"
        )
        ax.text(
            0.5, -0.05,
            txt,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize='small', linespacing=1.2
        )

    return ax




def embed_geneset(
    adata: AnnData,
    genes: Sequence[str],
    *,
    embedding_key: str = "X_spatial",
    log1p: bool = False,
    clip_percentiles: Tuple[float, float] = (0, 99.5),
    agg_func: Callable[[np.ndarray], float] = np.mean,
    na_rm: bool = True,
    ax: plt.Axes | None = None,
    scatter_kwargs: dict = None,
    show_colorbar: bool = True,
    colorbar_label: str = 'Geneset expression',
    cbar_kwargs: dict = None,
    show_gene_names: bool = True,
    gene_fontsize: str = 'small',
    gene_y_offset: float = -0.05,
    
    # spot_size: float | np.ndarray = 2.0,
    # cmap: str = "Greens",
    # alpha: float = 0.75,

) -> plt.Axes:
    """
    Scatter‑plot a summary of expression for a set of genes, drawing
    higher‑expressing cells on top.

    Parameters
    ----------
    adata
        AnnData object with expression data and a 2‑D embedding in ``adata.obsm``.
    genes
        Iterable of gene names to summarize.
    embedding_key
        Key in ``adata.obsm`` holding the 2‑D coordinates.
    spot_size
        Marker size (scalar or 1‑D array with length == n_cells).
    cmap
        Matplotlib colormap.
    log1p
        If ``True``, apply log1p to raw counts.
    clip_percentiles
        Percentile range used to clip each gene before rescaling to [0, 1].
    agg_func
        Aggregation function across genes (default: ``np.mean``).
    alpha
        Marker transparency.
    ax
        Matplotlib ``Axes`` to plot on (created if ``None``).
    na_rm
        If ``True``, silently drop genes missing from ``adata.var_names``.
    **scatter_kwargs
        Additional keyword arguments forwarded to ``plt.scatter``.

    Returns
    -------
    ax
        The ``Axes`` containing the plot.
    """
    # ---------- Input validation ----------
    if embedding_key not in adata.obsm_keys():
        raise ValueError(f"Embedding key '{embedding_key}' not present in adata.obsm.")

    coords = adata.obsm[embedding_key]
    if coords.shape[1] != 2:
        raise ValueError(
            f"Embedding '{embedding_key}' must be 2‑D (found shape {coords.shape})."
        )

    genes = list(genes)
    missing = [g for g in genes if g not in adata.var_names]
    if missing and not na_rm:
        raise KeyError(
            f"The following genes are absent from adata.var_names: {missing}"
        )
    if missing and na_rm:
        warnings.warn(
            f"{len(missing)} gene(s) not found and ignored: {', '.join(missing)}",
            RuntimeWarning,
            stacklevel=2,
        )
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        raise ValueError("No valid genes remain after filtering.")

    # set up default scatter and cbar params and update with user-provided ones
    default_scat = DEFAULTS_SCATTER
    if scatter_kwargs:
        default_scat.update(scatter_kwargs)

    default_cbar = DEFAULTS_CBAR
    if cbar_kwargs:
        default_cbar.update(cbar_kwargs)


    # ---------- Extract & preprocess expression ----------
    X = adata[:, genes].X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)

    if log1p:
        np.log1p(X, out=X)

    lo, hi = np.percentile(X, clip_percentiles, axis=0, keepdims=True)
    denom = np.where(hi > lo, hi - lo, 1.0)
    X_rescaled = (np.clip(X, lo, hi) - lo) / denom

    summary = agg_func(X_rescaled, axis=1)  # shape (n_cells,)

    # ---------- Sort so high expression is drawn last ----------
    order = np.argsort(summary)  # ascending: low → high
    coords_ordered = coords[order]
    summary_ordered = summary[order]


    # ---------- Plot ----------
    if ax is None:
        _, ax = plt.subplots()

    sc = ax.scatter(
        coords_ordered[:, 0],
        coords_ordered[:, 1],
        c=summary_ordered,
        **default_scat,
#        **scatter_kwargs
    )
    
    if show_colorbar:
        # cb = plt.colorbar(sc, ax=ax, **default_cbar)
        cb = plt.colorbar(sc, ax=ax, **default_cbar)
        cb.set_label(colorbar_label)    

#     ax.set_xticks([]); ax.set_yticks([])
    # ax.set_aspect("equal", adjustable="box")
    ax.set_aspect("equal")
    ax.axis('off')

    if show_gene_names:
        gene_str = ', '.join(genes)
        ax.text(
            0.5, gene_y_offset, gene_str,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=gene_fontsize
        )

    return ax

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import Patch

def embed_categorical(
    adata,
    category_key: str,
    embedding_key: str = 'X_spatial',
    palette: dict = None,
    ax=None,
    scatter_kwargs: dict = None,
    legend_kwargs: dict = None,
    show_legend: bool = True
):
    """
    Layer‐1: draw every spot colored by adata.obs[category_key].
    If adata.uns['<category_key>_colors'] exists and matches the number
    of categories, use it; otherwise generate a new seaborn palette.

    show_legend : bool
        If False, suppress the legend.
    """
    # 0) sanity checks
    if category_key not in adata.obs:
        raise ValueError(f"Column '{category_key}' not in adata.obs")
    coords = adata.obsm.get(embedding_key)
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be an (n_obs,2) array.")
    x, y = coords[:,0], coords[:,1]

    labels = adata.obs[category_key].astype(str)
    cats   = labels.unique().tolist()

    # 1) build or validate palette
    if palette is None:
        uns_key = f"{category_key}_colors"
        if (
            uns_key in adata.uns
            and isinstance(adata.uns[uns_key], (list, tuple))
            and len(adata.uns[uns_key]) >= len(cats)
        ):
            colors = adata.uns[uns_key][:len(cats)]
        else:
            if uns_key in adata.uns and len(adata.uns[uns_key]) < len(cats):
                warnings.warn(
                    f"{uns_key} in adata.uns has length "
                    f"{len(adata.uns[uns_key])}, but needs {len(cats)}; regenerating."
                )
            colors = sns.color_palette(n_colors=len(cats))
        color_map = dict(zip(cats, colors))
    else:
        missing = set(cats) - set(palette.keys())
        if missing:
            raise ValueError(f"palette dict is missing colors for categories: {missing}")
        color_map = palette

    color_list = [color_map[l] for l in labels]

    # 2) axes
    if ax is None:
        fig, ax = plt.subplots()

    # 3) scatter
    defaults = {
        's': 30,
        'alpha': 0.6,
        'edgecolors': 'none',
        'cmap': None
    }
    if scatter_kwargs:
        defaults.update(scatter_kwargs)
    sc = ax.scatter(
        x, y,
        c=color_list,
        **defaults
    )

    # 4) legend
    if show_legend:
        patches = [
            Patch(facecolor=color_map[cat], label=cat, alpha=defaults.get('alpha', 1.0), edgecolor='none')
            for cat in cats
        ]
        defaults_legend = {
            'title': category_key,
            'loc': 'upper left',
            'bbox_to_anchor': (1.02, 1),
            'ncol': 1,
            'fontsize': 'small'
        }
        if legend_kwargs:
            defaults_legend.update(legend_kwargs)
        ax.legend(handles=patches, **defaults_legend)

    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def old_embed_categorical(
    adata,
    category_key: str,
    embedding_key: str = 'X_spatial',
    palette: dict = None,
    ax=None,
    scatter_kwargs: dict = None,
    legend_kwargs: dict = None,
):
    """
    If adata.uns['<category_key>_colors'] exists and matches the number
    of categories, use it; otherwise generate a new seaborn palette.
    """
    # 0) sanity checks
    if category_key not in adata.obs:
        raise ValueError(f"Column '{category_key}' not in adata.obs")
    coords = adata.obsm.get(embedding_key)
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be an (n_obs,2) array.")
    
    x, y = coords[:,0], coords[:,1]
    labels = adata.obs[category_key].astype(str)
    cats = labels.unique().tolist()

    # 1) build or validate palette
    if palette is None:
        uns_key = f"{category_key}_colors"
        if uns_key in adata.uns and isinstance(adata.uns[uns_key], (list, tuple)):
            uns_colors = adata.uns[uns_key]
            if len(uns_colors) >= len(cats):
                color_map = dict(zip(cats, uns_colors[:len(cats)]))
            else:
                warnings.warn(
                    f"{uns_key} in adata.uns has length {len(uns_colors)}, "
                    f"but there are {len(cats)} categories; generating new palette."
                )
                palette_list = sns.color_palette(n_colors=len(cats))
                color_map = dict(zip(cats, palette_list))
        else:
            palette_list = sns.color_palette(n_colors=len(cats))
            color_map = dict(zip(cats, palette_list))
    else:
        # user-supplied dict
        missing = set(cats) - set(palette.keys())
        if missing:
            raise ValueError(f"palette dict is missing colors for categories: {missing}")
        color_map = palette


    color_list = [color_map[lab] for lab in labels]

    # 2) get or make axes
    if ax is None:
        _, ax = plt.subplots()
    
    # 3) plot each category

    # set up default scatter and cbar params and update with user-provided ones
    default_scat = DEFAULTS_SCATTER
    if scatter_kwargs:
        default_scat.update(scatter_kwargs)

    sc = ax.scatter(x=x, y=y, c=color_list, **default_scat)

    # legend
    patches = [
        Patch(facecolor=clr, label=cat, alpha=default_scat['alpha'], edgecolor='none') for cat, clr in color_map.items()
    ]

    default_legend = DEFAULTS_LEGEND
    if legend_kwargs:
        default_legend.update(legend_kwargs)
    ax.legend(handles=patches, **default_legend)

    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def embed_obsm(
    adata: AnnData,
    feature: str,
    obsm_name: str,
    embedding_key: str  = 'X_spatial',
    alpha: float = 0.75,
    s: float = 10,
    cmap: Union[str, plt.Colormap] = 'Greens',
    ax = None,
    display: bool = True,
    **kwargs
):
    """
    Plot embedding colored by values in .obsm[obsm_name][feature]

    Args:
        adata (AnnData): 
            The AnnData object containing the scRNA-seq data.
        feature (str): 
            A list of columns in .obsm[obs_name] to show
        obsm_name (str): 
            The name of the obsm key containing the values to show
        alpha (float, optional): 
            The transparency level of the points on the UMAP plot. Defaults to 0.75.
        s (int, optional): 
            The size of the points on the UMAP plot. Defaults to 10.
        display (bool, optional): 
            If True, the plot is displayed immediately. If False, the axis object is returned. Defaults to True.

    Returns:
        matplotlib.axes.Axes or None: 
            If `display` is False, returns the matplotlib axes object. Otherwise, returns None.
    """
    # Create a temporary AnnData object with the desired obsm

    # get coordinates
    coords = adata.obsm.get(embedding_key)
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be an (n_obs, 2) array.")
    x_vals, y_vals = coords[:, 0], coords[:, 1]

    if feature not in adata.obsm[obsm_name].columns:
            raise ValueError(f"Feature '{feature}' not found in adata.obsm['{obsm_name}'].")

    fig, ax = plt.subplots()
    # vals = _get_array(adata.obsm[obsm_name][:, feature])
    vals = adata.obsm[obsm_name][:, feature]
    order = np.argsort(norm)
    sc = ax.scatter(
        x_vals[order],
        y_vals[order],
        c = vals[order],
        cmap = cmap,
        s = spot_size,
        alpha = alpha,
    )
    ax.set_title(feature)
    ax.set_xticks([]); ax.set_yticks([])

    # Colorbar axis on the right, spanning full height (15% margin)
    # cbar_ax = fig.add_axes([0.88, 0.05, 0.02, 0.9])
    # cb = fig.colorbar(scatters[0], cax=cbar_ax)
    # cb.set_label('normalized expression')
    return ax


def scatter_genes_oneper(
    adata: AnnData,
    genes: Sequence[str],
    embedding_key: str = "X_spatial",
    spot_size: float = 2,
    alpha: float = 0.9,
    clip_percentiles: tuple = (0, 99.5),
    log_transform: bool = False,
    cmap: Union[str, plt.Colormap] = 'Reds',
    figsize: Optional[tuple] = None,
    panel_width: float = 4.0,
    n_rows: int = 1,
    axes: Optional[Sequence[plt.Axes]] = None
) -> Sequence[plt.Axes]:
    """
    Plot expression of multiple genes on a 2D embedding arranged in a grid.

    Each gene is optionally log-transformed, percentile-clipped, and rescaled to [0,1].
    Cells are plotted on the embedding, colored by expression, with highest values
    drawn on top. A single colorbar is placed to the right of the grid.

    Parameters
    ----------
    adata
        AnnData containing the embedding in `.obsm[embedding_key]`.
    genes
        List of gene names to plot (must be in `adata.var_names`).
    embedding_key
        Key in `.obsm` for an (n_obs, 2) coordinate array.
    spot_size, alpha, clip_percentiles, log_transform, cmap
        Scatter and normalization parameters.
    figsize
        (width, height) of entire figure. If None, computed from
        `panel_width`, `n_rows`, and embedding aspect ratio.
    panel_width
        Width (in inches) of each panel when `figsize` is None.
    n_rows
        Number of rows in the grid. Default 1.
    axes
        Optional sequence of Axes objects matching grid shape; if provided,
        the function will plot into these axes instead of creating new ones.

    Returns
    -------
    axes_flat
        List of Axes (flattened) used for each gene panel.
    """
    # Helper to extract array
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()

    coords = adata.obsm.get(embedding_key)
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be an (n_obs, 2) array.")
    x_vals, y_vals = coords[:, 0], coords[:, 1]

    n_genes = len(genes)
    cols = math.ceil(n_genes / n_rows)

    # Compute figsize if not provided
    if figsize is None and axes is None:
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()
        aspect = x_range / y_range if y_range > 0 else 1.0
        panel_height = panel_width / aspect
        fig_width = panel_width * cols
        fig_height = panel_height * n_rows
    elif figsize is not None and axes is None:
        fig_width, fig_height = figsize
    
    # Create or use existing axes grid
    if axes is None:
        fig, ax_grid = plt.subplots(
            n_rows,
            cols,
            figsize=(fig_width, fig_height),
            squeeze=False
        )
    else:
        # assume axes is sequence of length n_rows*cols
        ax_grid = np.array(axes).reshape(n_rows, cols)
        fig = ax_grid.flatten()[0].figure

    axes_flat = ax_grid.flatten()
    scatters = []

    for idx, gene in enumerate(genes):
        ax = axes_flat[idx]
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
        vals = _get_array(adata[:, gene].X)
        if log_transform:
            vals = np.log1p(vals)
        lo, hi = np.percentile(vals, clip_percentiles)
        clipped = np.clip(vals, lo, hi)
        norm = (clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped)

        order = np.argsort(norm)
        sc = ax.scatter(
            x_vals[order],
            y_vals[order],
            c=norm[order],
            cmap=cmap,
            s=spot_size,
            alpha=alpha,
            vmin=0,
            vmax=1
        )
        ax.set_title(gene)
        ax.set_xticks([])
        ax.set_yticks([])
        scatters.append(sc)

    # Turn off unused axes
    for j in range(n_genes, n_rows * cols):
        axes_flat[j].axis('off')

    # Only create colorbar/figure layout if we made the fig
    if axes is None:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.05, 0.02, 0.9])
        cb = fig.colorbar(scatters[0], cax=cbar_ax)
        cb.set_label('normalized expression')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        return axes_flat
    else:
        # return existing axes for further manipulation
        return axes_flat



def plot_spatial_two_genes_stack(
    adatas, 
    gene1,
    gene2,
    cmap,
    width_ratios: Tuple[float, float] = (3, 1),
    **kwargs
):
    n = len(adatas)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(8, 4 * n),
        gridspec_kw={'width_ratios': width_ratios,
                     'wspace': 0.15, 'hspace': 0.2}
    )
    # make sure axes is 2D
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, ad in enumerate(adatas):
        ax_sc, ax_cb = axes[i]

        # wrap the call to guarantee cleanup
        with _temp_plt_axes(fig, ax_sc, ax_cb):
            try:
                spatial_two_genes(
                    ad, gene1, gene2, cmap, width_ratios=width_ratios, **kwargs
                )
            except Exception as e:
                # you can log or print the error and continue
                print(f"Error plotting {i}-th AnnData: {e}")

        # Optional: add a row title
        title = ad.uns.get('sample', f'Sample {i+1}')
        ax_sc.set_title(f"{title}: {gene1} vs {gene2}")

    plt.tight_layout()
    return fig


def spatial_two_genes(
    adata: AnnData,
    gene1: str,
    gene2: str,
    cmap: ListedColormap,
    spot_size: float = 2,
    alpha: float = 0.9,
    spatial_key: str = 'X_spatial',
    log_transform: bool = False,
    clip_percentiles: tuple = (0, 99.5),
    priority_metric: str = 'sum',
    show_xcoords: bool = False,
    show_ycoords: bool = False,
    show_bbox: bool = False,
    show_legend: bool = True,
    width_ratios: Tuple[float, float] = (10, 1)
) -> None:
    """Plot two‐gene spatial expression with a bivariate colormap.

    Args:
        adata: AnnData with spatial coords in `adata.obsm[spatial_key]`.
        gene1: First gene name (must be in `adata.var_names`).
        gene2: Second gene name.
        cmap: Bivariate colormap from `make_bivariate_cmap` (n×n LUT).
        spot_size: Scatter point size.
        alpha: Point alpha transparency.
        spatial_key: Key in `adata.obsm` for an (n_obs, 2) coords array.
        log_transform: If True, apply `np.log1p` to raw expression.
        clip_percentiles: Tuple `(low_pct, high_pct)` to clip each gene.
        priority_metric: Which metric to sort drawing order by:
            - 'sum': u + v (default)
            - 'gene1': u only
            - 'gene2': v only
        show_xcoords: Whether to display x-axis ticks and labels.
        show_ycoords: Whether to display y-axis ticks and labels.
        show_bbox: Whether to display the bounding box (spines).
        show_legend: Whether to display the legend/colorbar.
        width_ratios: 2‐tuple giving the relative widths of
                  (scatter_panel, legend_panel). Defaults to (3,1).
    
    Raises:
        ValueError: If spatial coords are missing/malformed or
                    if `priority_metric` is invalid.
    """
    # 1) extract raw arrays
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()
    X1 = _get_array(adata[:, gene1].X)
    X2 = _get_array(adata[:, gene2].X)

    # 2) optional log1p
    if log_transform:
        X1 = np.log1p(X1)
        X2 = np.log1p(X2)

    # 3) percentile‐clip
    lo1, hi1 = np.percentile(X1, clip_percentiles)
    lo2, hi2 = np.percentile(X2, clip_percentiles)
    X1 = np.clip(X1, lo1, hi1)
    X2 = np.clip(X2, lo2, hi2)

    # 4) normalize to [0,1]
    u = (X1 - lo1) / (hi1 - lo1) if hi1 > lo1 else np.zeros_like(X1)
    v = (X2 - lo2) / (hi2 - lo2) if hi2 > lo2 else np.zeros_like(X2)

    # 5) prepare LUT
    m = len(cmap.colors)
    n = int(np.sqrt(m))
    C = np.array(cmap.colors).reshape(n, n, 3)

    # 6) bilinear interpolate per‐cell
    gu = u * (n - 1); gv = v * (n - 1)
    i0 = np.floor(gu).astype(int); j0 = np.floor(gv).astype(int)
    i1 = np.minimum(i0 + 1, n - 1); j1 = np.minimum(j0 + 1, n - 1)
    du = gu - i0; dv = gv - j0

    wa = (1 - du) * (1 - dv)
    wb = du * (1 - dv)
    wc = (1 - du) * dv
    wd = du * dv

    c00 = C[j0, i0]; c10 = C[j0, i1]
    c01 = C[j1, i0]; c11 = C[j1, i1]

    cols_rgb = (
        c00 * wa[:, None] +
        c10 * wb[:, None] +
        c01 * wc[:, None] +
        c11 * wd[:, None]
    )
    hex_colors = [to_hex(c) for c in cols_rgb]

    # 7) determine draw order
    if priority_metric == 'sum':
        priority = u + v
    elif priority_metric == 'gene1':
        priority = u
    elif priority_metric == 'gene2':
        priority = v
    else:
        raise ValueError("priority_metric must be 'sum', 'gene1', or 'gene2'")
    order = np.argsort(priority)

    # 8) fetch and sort coords/colors
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be an (n_obs, 2) array")
    coords_sorted = coords[order]
    colors_sorted = [hex_colors[i] for i in order]

    # 9) plot scatter + optional legend
    fig, (ax_sc, ax_cb) = plt.subplots(
        1, 2,
        figsize=(8, 4),
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.3}
    )
    ax_sc.scatter(
        coords_sorted[:, 0],
        coords_sorted[:, 1],
        c=colors_sorted,
        s=spot_size,
        alpha=alpha
    )
    ax_sc.set_aspect('equal')
    ax_sc.set_title(f"{gene1} :: {gene2}")

    # axis display options
    if not show_xcoords:
        ax_sc.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if not show_ycoords:
        ax_sc.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    if not show_bbox:
        for spine in ax_sc.spines.values():
            spine.set_visible(False)

    # legend/colorbar
    if show_legend:
        lut_img = C  # shape (n,n,3)
        ax_cb.imshow(lut_img, origin='lower', extent=[0, 1, 0, 1])
        # ax_cb.set_xlabel(f"{gene1}\nlow → high")
        # ax_cb.set_ylabel(f"{gene2}\nlow → high")
        ax_cb.set_xlabel(f"{gene1}")
        ax_cb.set_ylabel(f"{gene2}")
        ax_cb.set_xticks([0, 1]); ax_cb.set_yticks([0, 1])
        ax_cb.set_aspect('equal')
    else:
        ax_cb.axis('off')

    plt.show()


def embed_bivariate_multi(
    adata: AnnData,
    genes1: Sequence[str],
    genes2: Sequence[str],
    cmap: ListedColormap = None,
    spot_size: float = 2,
    alpha: float = 0.9,
    spatial_key: str = 'X_spatial',
    log_transform: bool = False,
    clip_percentiles: Tuple[float, float] = (0, 99.5),
    priority_metric: str = 'sum',
    n_rows: int = 1,
    show_xcoords: bool = False,
    show_ycoords: bool = False,
    show_bbox: bool = False,
    legend: bool = True,
    width_ratio: Tuple[float, float] = (10, 1),
    panel_size: float = 4.0,
) -> None:
    """Display multiple bivariate‐gene spatial maps in a grid with shared legend.

    Args:
        adata: AnnData with coords in adata.obsm[spatial_key] (n_obs,2).
        genes1: First genes for each panel.
        genes2: Second genes, same length as genes1.
        cmap: Bivariate colormap (ListedColormap).
        spot_size: Marker size.
        alpha: Marker transparency.
        spatial_key: Key in adata.obsm for x,y coords.
        log_transform: If True, log1p each gene.
        clip_percentiles: Percentile clip bounds.
        priority_metric: 'sum', 'gene1', or 'gene2'.
        n_rows: Number of grid rows.
        show_xcoords: Show x-axis ticks/labels.
        show_ycoords: Show y-axis ticks/labels.
        show_bbox: Show axis spines.
        legend: If True, draw shared legend.
        width_ratio: Tuple (scatter_region_ratio, legend_region_ratio),
            controlling relative widths in the GridSpec.
        panel_size: Size (inches) of each scatter panel (both width and height).

    Raises:
        ValueError: If genes1/genes2 lengths differ or coords missing.
    """
    # Validate input lengths
    if len(genes1) != len(genes2):
        raise ValueError("genes1 and genes2 must be the same length.")
    n_plots = len(genes1)
    if n_plots == 0:
        raise ValueError("Must provide at least one gene pair.")

    # Get coordinates
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be shape (n_obs,2).")
    xs, ys = coords[:, 0], coords[:, 1]

    if cmap is None:
        cmap = make_bivariate_cmap()

    # Prepare colormap LUT
    m = len(cmap.colors)
    n = int(math.sqrt(m))
    C = np.array(cmap.colors).reshape(n, n, 3)

    # Layout geometry
    n_cols = math.ceil(n_plots / n_rows)
    w_sc, w_leg = width_ratio
    fig_width = panel_size * (n_cols * w_sc + (w_leg if legend else 0)) / w_sc
    fig_height = panel_size * n_rows

    fig = plt.figure(figsize=(fig_width, fig_height))
    # GridSpec: n_rows x (n_cols + 1 for legend if needed)
    total_cols = n_cols + (1 if legend else 0)
    col_ratios = [w_sc] * n_cols + ([w_leg] if legend else [])
    gs = fig.add_gridspec(n_rows, total_cols, width_ratios=col_ratios, wspace=0.1, hspace=0.1)

    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()

    # Plot panels
    for idx, (g1, g2) in enumerate(zip(genes1, genes2)):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # extract & preprocess
        X1 = _get_array(adata[:, g1].X)
        X2 = _get_array(adata[:, g2].X)
        if log_transform:
            X1 = np.log1p(X1); X2 = np.log1p(X2)
        lo1, hi1 = np.percentile(X1, clip_percentiles)
        lo2, hi2 = np.percentile(X2, clip_percentiles)
        X1 = np.clip(X1, lo1, hi1); X2 = np.clip(X2, lo2, hi2)
        u = (X1 - lo1)/(hi1 - lo1) if hi1>lo1 else np.zeros_like(X1)
        v = (X2 - lo2)/(hi2 - lo2) if hi2>lo2 else np.zeros_like(X2)

        # bilinear LUT mapping
        gu, gv = u*(n-1), v*(n-1)
        i0, j0 = np.floor(gu).astype(int), np.floor(gv).astype(int)
        i1, j1 = np.minimum(i0+1, n-1), np.minimum(j0+1, n-1)
        du, dv = gu-i0, gv-j0
        wa, wb = (1-du)*(1-dv), du*(1-dv)
        wc, wd = (1-du)*dv, du*dv
        c00, c10 = C[j0, i0], C[j0, i1]
        c01, c11 = C[j1, i0], C[j1, i1]
        cols = c00*wa[:,None] + c10*wb[:,None] + c01*wc[:,None] + c11*wd[:,None]
        hexcols = [to_hex(c) for c in cols]

        # draw order
        if priority_metric == 'sum':
            pr = u + v
        elif priority_metric == 'gene1':
            pr = u
        elif priority_metric == 'gene2':
            pr = v
        else:
            raise ValueError("priority_metric must be 'sum', 'gene1', or 'gene2'")
        order = np.argsort(pr)

        ax.scatter(xs[order], ys[order],
                   c=[hexcols[i] for i in order],
                   s=spot_size, alpha=alpha)
        ax.set_aspect('equal')
        ax.set_title(f"{g1} :: {g2}", pad=4)


        if not show_xcoords:
            ax.set_xticks([]); ax.set_xticklabels([])
        if not show_ycoords:
            ax.set_yticks([]); ax.set_yticklabels([])
        if not show_bbox:
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Turn off unused panels
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.add_subplot(gs[row, col]).axis('off')

    # Shared legend
    if legend:
        lg_ax = fig.add_subplot(gs[:, -1])
        lg_ax.imshow(C, origin='lower', extent=[0, 1, 0, 1])
        lg_ax.set_xticks([0, 1]); lg_ax.set_yticks([0, 1])
        lg_ax.set_xlabel("gene1 ↑"); lg_ax.set_ylabel("gene2 ↑")
        if not show_bbox:
            for spine in lg_ax.spines.values():
                spine.set_visible(True)

    plt.tight_layout()
    plt.show()

