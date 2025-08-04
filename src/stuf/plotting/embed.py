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
    Plot two gene modules on a 2D embedding using a bivariate colormap.

    The function summarizes gene expression for two sets of genes and maps
    them onto a bivariate color space. Cells are plotted in 2D using the 
    coordinates in `adata.obsm[embedding_key]`, and colored according to 
    their gene module scores.

    Args:
        adata (AnnData): Annotated data matrix with gene expression values.
        genes1 (List[str]): List of gene names for the first module.
        genes2 (List[str]): List of gene names for the second module.
        cmap (ListedColormap, optional): Colormap used for bivariate color mapping. 
            Defaults to `BIMAP_YELLOW`.
        embedding_key (str, optional): Key in `adata.obsm` containing 2D coordinates. 
            Defaults to `'X_spatial'`.
        log_transform (bool, optional): Whether to log-transform expression values before 
            summarizing. Defaults to `False`.
        clip_percentiles (Tuple[float, float], optional): Percentiles for clipping gene 
            expression values prior to normalization. Defaults to `(0, 100)`.
        agg_func (Callable, optional): Function to summarize expression across genes. 
            Defaults to `np.mean`.
        priority_metric (str, optional): Metric used to order cells in plot.
            Options: `'sum'`, `'list1'`, `'list2'`. Defaults to `'sum'`.
        show_xcoords (bool, optional): Whether to show x-axis tick labels. Defaults to `False`.
        show_ycoords (bool, optional): Whether to show y-axis tick labels. Defaults to `False`.
        show_bbox (bool, optional): Whether to display axis borders. Defaults to `False`.
        show_legend (bool, optional): Whether to display a bivariate color legend. Defaults to `True`.
        width_ratios (Tuple[float, float], optional): Width ratios for scatter vs colorbar subplot. 
            Defaults to `(10, 1)`.
        scatter_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments passed to 
            `ax.scatter()`. Defaults to `None`.
        cbar_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for legend axes 
            when `ax` is not provided. Defaults to `None`.
        cbar_fontsize (str, optional): Font size for the legend labels. Defaults to `'small'`.
        list_genes (bool, optional): Whether to list gene names below the plot. Defaults to `True`.
        ax (Optional[plt.Axes], optional): Existing matplotlib Axes to plot on. If `None`, a new figure 
            and axes will be created. Defaults to `None`.

    Returns:
        plt.Axes: Matplotlib Axes object containing the main scatter plot.
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
) -> plt.Axes:
    """
    Plot a 2D embedding colored by summarized expression of a gene set.

    This function visualizes the expression of a set of genes by aggregating 
    expression values across genes and mapping them to cell coordinates 
    in a 2D embedding (e.g., spatial or UMAP). Higher-expressing cells are 
    drawn on top for clarity.

    Args:
        adata (AnnData): 
            Annotated data matrix with gene expression and embedding coordinates.
        genes (Sequence[str]): 
            List or other iterable of gene names to summarize.
        embedding_key (str, optional): 
            Key in `adata.obsm` containing 2D coordinates. Defaults to `"X_spatial"`.
        log1p (bool, optional): 
            Whether to apply log1p transformation to the expression values. Defaults to `False`.
        clip_percentiles (Tuple[float, float], optional): 
            Percentile range used to clip each gene before min-max scaling to [0, 1]. 
            Defaults to `(0, 99.5)`.
        agg_func (Callable[[np.ndarray], float], optional): 
            Function to aggregate expression across genes for each cell (e.g., `np.mean`). 
            Defaults to `np.mean`.
        na_rm (bool, optional): 
            If `True`, genes not found in `adata.var_names` are silently ignored. 
            Defaults to `True`.
        ax (plt.Axes or None, optional): 
            Existing matplotlib Axes to plot on. If `None`, a new figure and axes are created.
        scatter_kwargs (dict, optional): 
            Additional keyword arguments passed to `plt.scatter()`.
        show_colorbar (bool, optional): 
            Whether to display a colorbar legend. Defaults to `True`.
        colorbar_label (str, optional): 
            Label for the colorbar. Defaults to `"Geneset expression"`.
        cbar_kwargs (dict, optional): 
            Additional keyword arguments passed to the colorbar plotting function.
        show_gene_names (bool, optional): 
            Whether to display the gene list below the plot. Defaults to `True`.
        gene_fontsize (str, optional): 
            Font size for the gene label text. Defaults to `'small'`.
        gene_y_offset (float, optional): 
            Vertical offset (in axes coordinates) for gene list annotation. Defaults to `-0.05`.

    Returns:
        plt.Axes: 
            The matplotlib Axes object containing the plot.
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
    Plot a 2D embedding colored by categorical labels.

    Visualizes cells or spots on a 2D embedding using colors determined 
    by a categorical variable from `adata.obs[category_key]`. If a color 
    palette is stored in `adata.uns['<category_key>_colors']`, it is used;
    otherwise, a new palette is generated or provided.

    Args:
        adata (AnnData): Annotated data matrix with coordinates in `obsm`.
        category_key (str): Key in `adata.obs` containing the categorical variable 
            to visualize (e.g., cell type or cluster ID).
        embedding_key (str, optional): Key in `adata.obsm` for 2D embedding coordinates. 
            Defaults to `'X_spatial'`.
        palette (dict, optional): Dictionary mapping category names to colors. Overrides
            colors in `adata.uns`.
        ax (matplotlib.axes.Axes, optional): Axes object to draw on. If `None`, a new figure 
            and axes will be created.
        scatter_kwargs (dict, optional): Additional keyword arguments passed to `ax.scatter()`.
        legend_kwargs (dict, optional): Additional keyword arguments passed to `ax.legend()`.
        show_legend (bool, optional): Whether to display the legend. Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: Axes object with the categorical embedding plotted.
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
    Plot a 2D embedding colored by a feature value from `adata.obsm`.

    Visualizes the spatial or embedded coordinates in `adata.obsm[embedding_key]`,
    colored by the specified `feature` from a matrix in `adata.obsm[obsm_name]`.

    Args:
        adata (AnnData): 
            The AnnData object containing the data, with relevant `.obsm` entries.
        feature (str): 
            Column name (i.e., feature) within `.obsm[obsm_name]` to use for coloring the plot.
        obsm_name (str): 
            Key in `.obsm` indicating the matrix to extract feature values from.
        embedding_key (str, optional): 
            Key in `.obsm` containing the 2D coordinates for plotting. Defaults to `'X_spatial'`.
        alpha (float, optional): 
            Transparency of points. Ranges from 0 (transparent) to 1 (opaque). Defaults to `0.75`.
        s (float, optional): 
            Size of the scatter plot markers. Defaults to `10`.
        cmap (Union[str, plt.Colormap], optional): 
            Matplotlib colormap or string name for coloring the feature values. Defaults to `'Greens'`.
        ax (matplotlib.axes.Axes, optional): 
            Axes object to plot on. If `None`, a new figure and axes will be created.
        display (bool, optional): 
            If `True`, displays the plot immediately. If `False`, returns the axes object. Defaults to `True`.
        **kwargs: 
            Additional keyword arguments passed to `ax.scatter()`.

    Returns:
        matplotlib.axes.Axes or None: 
            Returns the axes object if `display=False`, otherwise returns `None`.
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

