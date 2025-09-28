#from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
# from scipy import sparse
# from typing import List, Literal, Sequence,  Tuple 
from anndata import AnnData
from typing import Dict, Tuple, Optional, Literal, Union


def pack_spatial_sections(
    adata: ad.AnnData,
    group_key: str = "sample_name",
    obsm_key: str = "X_spatial",
    gutter: Union[int, float] = 50,
    outer_margin: Union[int, float] = 25,
    order: Literal["area", "height", "width", "name"] = "area",
    target_row_width: Optional[Union[int, float]] = None,
    row_align: Literal["left", "center", "right", "space-between", "space-around"] = "left",
    store_backup: bool = True,
    backup_key: str = "X_spatial_backup",
    inplace: bool = True,
    return_layout: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Pack spatial sections (e.g., tissue pieces) to reduce whitespace, with row alignment.

    This function repositions groups of spots/cells (defined by `obs[group_key]`)
    so that each group's bounding box sits with a small, uniform spacing between
    neighboring groups. Groups are placed using a shelf (row-wise) heuristic,
    then rows can be aligned horizontally (left/center/right) or justified.

    Args:
        adata:
            AnnData with coordinates in `obsm[obsm_key]` (shape: n_cells x 2) and
            group labels in `obs[group_key]`.
        group_key:
            `adata.obs` column indicating the sample/section membership (e.g., "sample_name").
        obsm_key:
            Key in `adata.obsm` holding spatial coordinates (n x 2).
        gutter:
            Minimum spacing between adjacent sections (same units as `X_spatial`).
        outer_margin:
            Padding added around the entire packed canvas.
        order:
            Section sort before packing: {"area", "height", "width", "name"}.
        target_row_width:
            Optional target row width used to decide line wraps. If None, a heuristic
            based on total area is used.
        row_align:
            Horizontal alignment of items within each row after packing:
              - "left": rows start at the left margin (default).
              - "center": rows are centered within the widest row’s used width.
              - "right": rows are right-aligned to the widest row’s used width.
              - "space-between": extra space is distributed between items in the row
                (>=2 items). Single-item rows are centered.
              - "space-around": extra space is split around items (>=2 items).
                Single-item rows are centered.
        store_backup:
            If True, store original coordinates in `adata.obsm[backup_key]` if absent.
        backup_key:
            Where to store original coordinates when `store_backup=True`.
        inplace:
            If True, write updated coordinates back to `adata.obsm[obsm_key]`.
        return_layout:
            If True, return a layout DataFrame describing section placements and canvas size.

    Returns:
        DataFrame if `return_layout=True` with columns
        ['group', 'x', 'y', 'width', 'height', 'area', 'row', 'order_idx'] and a
        final row '__canvas__' giving the packed canvas width/height. Otherwise None.

    Raises:
        ValueError: Missing keys/columns or malformed coordinates; invalid options.
        TypeError: Incorrect input types.
    """
    # ---- validations ----
    if not isinstance(adata, ad.AnnData):
        raise TypeError("`adata` must be an AnnData object.")
    if group_key not in adata.obs.columns:
        raise ValueError(f"`adata.obs` missing required column '{group_key}'.")
    if obsm_key not in adata.obsm:
        raise ValueError(f"`adata.obsm` missing required key '{obsm_key}'.")
    coords = adata.obsm[obsm_key]
    if not isinstance(coords, (np.ndarray, pd.DataFrame)):
        raise TypeError(f"`adata.obsm['{obsm_key}']` must be a numpy array or pandas DataFrame.")
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`adata.obsm['{obsm_key}']` must have shape (n, 2). Got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError(f"`adata.obsm['{obsm_key}']` contains non-finite values.")
    if gutter < 0 or outer_margin < 0:
        raise ValueError("`gutter` and `outer_margin` must be non-negative.")

    groups = pd.Series(adata.obs[group_key]).astype("category")
    group_names = [g for g in groups.cat.categories if (groups == g).any()]
    group_to_idx: Dict[str, np.ndarray] = {g: np.where(groups.values == g)[0] for g in group_names}

    # ---- per-group bounding boxes on original coords ----
    stats = []
    for g in group_names:
        ix = group_to_idx[g]
        xy = coords[ix, :]
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        w = float(x_max - x_min)
        h = float(y_max - y_min)
        stats.append(dict(group=g, x_min=x_min, y_min=y_min, width=w, height=h, area=w * h))
    layout = pd.DataFrame(stats)
    eps = 1e-6
    layout["width"] = layout["width"].clip(lower=eps)
    layout["height"] = layout["height"].clip(lower=eps)
    layout["area"] = layout["width"] * layout["height"]

    # ---- ordering ----
    if order == "area":
        layout = layout.sort_values(["area", "height", "width"], ascending=[False, False, False])
    elif order == "height":
        layout = layout.sort_values(["height", "width"], ascending=[False, False])
    elif order == "width":
        layout = layout.sort_values(["width", "height"], ascending=[False, False])
    elif order == "name":
        layout = layout.sort_values("group", ascending=True)
    else:
        raise ValueError("`order` must be one of {'area', 'height', 'width', 'name'}.")
    layout = layout.reset_index(drop=True)
    layout["order_idx"] = np.arange(len(layout))

    # ---- choose target_row_width if not provided ----
    if target_row_width is None:
        total_area = float(layout["area"].sum())
        est_side = np.sqrt(total_area) * 1.25 if total_area > 0 else 0.0
        approx_n = max(1, int(np.sqrt(len(layout))))
        target_row_width = est_side + approx_n * gutter
    if not np.isfinite(target_row_width) or target_row_width <= 0:
        raise ValueError("`target_row_width` must be a positive finite number.")

    # ---- shelf packing (initial left-aligned placement) ----
    placements: Dict[str, Tuple[float, float, int]] = {}
    cursor_x = outer_margin
    cursor_y = outer_margin
    current_row_height = 0.0
    row_id = 0
    max_x_reached = outer_margin

    rows: Dict[int, Dict[str, Union[list, float]]] = {}  # row_id -> {'groups': [..], 'widths': [..], 'heights': [..]}
    rows[row_id] = {"groups": [], "widths": [], "heights": []}

    for _, r in layout.iterrows():
        g = r["group"]
        w = float(r["width"])
        h = float(r["height"])

        # wrap if exceeding target width (ensure at least one per row)
        if cursor_x > outer_margin and (cursor_x + w) > (outer_margin + target_row_width):
            cursor_x = outer_margin
            cursor_y = cursor_y + current_row_height + gutter
            current_row_height = 0.0
            row_id += 1
            rows[row_id] = {"groups": [], "widths": [], "heights": []}

        placements[g] = (cursor_x, cursor_y, row_id)
        rows[row_id]["groups"].append(g)
        rows[row_id]["widths"].append(w)
        rows[row_id]["heights"].append(h)

        cursor_x = cursor_x + w + gutter
        current_row_height = max(current_row_height, h)
        max_x_reached = max(max_x_reached, cursor_x - gutter)

    # ---- compute per-row used widths and reference width ----
    # used_width = sum(widths) + gutter*(n-1) for n>=2, else width for n==1
    row_used_widths: Dict[int, float] = {}
    for rid, info in rows.items():
        n = len(info["widths"])
        if n == 0:
            row_used_widths[rid] = 0.0
        elif n == 1:
            row_used_widths[rid] = info["widths"][0]
        else:
            row_used_widths[rid] = float(sum(info["widths"])) + gutter * (n - 1)

    # choose a reference (maximum) row width so alignment looks consistent
    ref_row_width = max(row_used_widths.values()) if len(row_used_widths) else 0.0

    # ---- horizontal alignment adjustments per row ----
    # We re-assign x positions based on alignment within [outer_margin, outer_margin + ref_row_width]
    # and keep y (row baseline) unchanged.
    aligned_x: Dict[str, float] = {}

    for rid, info in rows.items():
        groups_in_row = info["groups"]
        widths = info["widths"]
        n = len(groups_in_row)
        if n == 0:
            continue

        used = row_used_widths[rid]
        # default left offset
        if row_align == "left":
            start_x = outer_margin
            gaps = [gutter] * max(0, n - 1)
        elif row_align == "right":
            start_x = outer_margin + (ref_row_width - used)
            gaps = [gutter] * max(0, n - 1)
        elif row_align == "center":
            start_x = outer_margin + (ref_row_width - used) / 2.0
            gaps = [gutter] * max(0, n - 1)
        elif row_align == "space-between" and n >= 2:
            # distribute extra space across internal gaps
            extra = max(0.0, ref_row_width - (sum(widths)))
            # n-1 internal gaps
            gapsize = extra / (n - 1) if n > 1 else 0.0
            start_x = outer_margin
            gaps = [gapsize] * (n - 1)
        elif row_align == "space-around" and n >= 2:
            # split extra space around items: equal left/right padding and equal internal gaps
            extra = max(0.0, ref_row_width - (sum(widths)))
            # There are n+1 gaps (including ends); space-around puts equal size on all
            gapsize = extra / (n + 1)
            start_x = outer_margin + gapsize
            gaps = [gapsize] * (n - 1)
        else:
            # fallbacks: single-item rows for space-between/around -> center
            start_x = outer_margin + (ref_row_width - used) / 2.0
            gaps = [gutter] * max(0, n - 1)

        x = float(start_x)
        for i, g in enumerate(groups_in_row):
            aligned_x[g] = x
            x += widths[i]
            if i < len(gaps):
                x += gaps[i]

    # ---- build new coordinates (translate each group to aligned x/y) ----
    new_coords = coords.copy()
    for _, r in layout.iterrows():
        g = r["group"]
        x_min = float(r["x_min"])
        y_min = float(r["y_min"])
        px_old, py, rid = placements[g]
        px = aligned_x.get(g, px_old)  # use aligned x (or original if missing)
        ix = group_to_idx[g]
        new_coords[ix, 0] = (coords[ix, 0] - x_min) + px
        new_coords[ix, 1] = (coords[ix, 1] - y_min) + py

    # ---- canvas size after alignment ----
    total_width = ref_row_width + 2 * outer_margin
    # compute total_height from final row baselines and row heights
    # (same as shelf packing vertical layout)
    # Recompute row baselines and heights from `placements`
    row_baselines = {}
    row_heights = {}
    for _, r in layout.iterrows():
        g = r["group"]
        h = float(r["height"])
        _, py, rid = placements[g]
        row_baselines[rid] = py
        row_heights[rid] = max(row_heights.get(rid, 0.0), h)
    if row_baselines:
        last_row = max(row_baselines)
        total_height = (row_baselines[last_row] - outer_margin) + row_heights[last_row] + 2 * outer_margin
    else:
        total_height = 2 * outer_margin

    # ---- update AnnData (and optional backup) ----
    if store_backup and backup_key not in adata.obsm:
        adata.obsm[backup_key] = coords.copy()
    if inplace:
        adata.obsm[obsm_key] = new_coords

    # ---- layout table (optional) ----
    layout_out = layout[["group", "width", "height", "area", "order_idx"]].copy()
    layout_out["x"] = layout_out["group"].map(aligned_x)
    layout_out["y"] = layout_out["group"].map(lambda g: placements[g][1])
    layout_out["row"] = layout_out["group"].map(lambda g: placements[g][2])
    layout_out = layout_out[["group", "x", "y", "width", "height", "area", "row", "order_idx"]]
    canvas_row = pd.DataFrame(
        [{"group": "__canvas__", "x": 0.0, "y": 0.0,
          "width": float(total_width), "height": float(total_height),
          "area": float(total_width * total_height), "row": -1, "order_idx": -1}]
    )
    layout_out = pd.concat([layout_out, canvas_row], ignore_index=True)

    return layout_out if return_layout else None



def rotate_coordinates(adata, key, degrees):
    """
    Rotate 2D spatial coordinates in `adata.obsm[key]` by a specified angle (clockwise).

    Applies a 2D rotation matrix to coordinates stored in `adata.obsm[key]`, rotating 
    them clockwise by the given number of degrees. The original coordinates are overwritten.

    Args:
        adata (AnnData): 
            Annotated data matrix containing spatial or embedded coordinates.
        key (str): 
            Key in `adata.obsm` pointing to a `(n_cells, 2)` array of 2D coordinates.
        degrees (float or int): 
            Angle in degrees by which to rotate the coordinates clockwise.

    Returns:
        None: 
            The function modifies `adata.obsm[key]` in place.
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
    Flip 2D spatial coordinates in `adata.obsm[key]` along the specified axis.

    This function negates either the x- or y-coordinates, effectively reflecting 
    the data across the specified axis. The transformation is performed in place.

    Args:
        adata (AnnData): 
            Annotated data matrix containing spatial or embedded coordinates.
        key (str): 
            Key in `adata.obsm` containing a `(n_cells, 2)` array of coordinates.
        axis (str, optional): 
            Axis to flip:
                - `'x'`: Flip vertically by negating y-coordinates (default).
                - `'y'`: Flip horizontally by negating x-coordinates.

    Returns:
        None: 
            The function modifies `adata.obsm[key]` in place.
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




