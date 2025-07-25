import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, TextBox, Slider, Button, CheckButtons, RadioButtons
from matplotlib.path import Path
from scipy import sparse

def manual_annotate(
    adata,
    basis: str = 'X_umap',
    dims: tuple[int,int]=(0,1),
    obs_key: str='manual_annotation',
    initial_size: float=5,
    initial_alpha: float=0.7,
):
    """
    Interactive annotator with:
    - Numeric/gene/categorical coloring (categorical palette from .uns if available)
    - Clean, padded control panel
    - Only redraw on submit (textboxes)
    - Active textbox highlight
    """

    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import dash_bootstrap_components as dbc

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # --- Data setup ---
    if 'total_counts' not in adata.obs:
        X = adata.X
        tc = np.array(X.sum(axis=1)).ravel() if sparse.issparse(X) else X.sum(axis=1)
        adata.obs['total_counts'] = tc
    if 'log1p_total_counts' not in adata.obs:
        adata.obs['log1p_total_counts'] = np.log1p(adata.obs['total_counts'])

    pts = adata.obsm[basis][:, dims]
    xs, ys = pts[:,0], pts[:,1]
    n_pts = len(xs)
    if obs_key not in adata.obs:
        adata.obs[obs_key] = 'unlabeled'
    else:
        adata.obs[obs_key] = adata.obs[obs_key].astype(str)

    # --- Figure/axes with gridspec ---
    plt.ion()
    fig = plt.figure(figsize=(12,7))
    gs = plt.GridSpec(1, 2, width_ratios=[7,2.1], wspace=0.4, right=0.98, left=0.05)
    ax = fig.add_subplot(gs[0])
    ax_ctrl = fig.add_subplot(gs[1])
    ax_ctrl.axis('off')

    # --- State holders ---
    current_label = {'name': None}
    point_size    = {'val': initial_size}
    alpha_val     = {'val': initial_alpha}
    cbar = [None]
    color_mode = {'type': 'numeric', 'key': 'log1p_total_counts', 'log': True}

    # --- Helpers ---
    def available_obs():
        import pandas as pd
        numeric = [col for col in adata.obs.columns if pd.api.types.is_numeric_dtype(adata.obs[col])]
        categorical = [col for col in adata.obs.columns
            if (
                pd.api.types.is_categorical_dtype(adata.obs[col]) or
                adata.obs[col].dtype == object
            )
            and adata.obs[col].nunique() <= min(20, n_pts//10)
            and col != obs_key]
        return numeric, categorical


    def get_numeric_data(var, use_log1p):
        if var in adata.var_names:
            raw = adata[:, var].X
            arr = raw.toarray().ravel() if sparse.issparse(raw) else np.array(raw).ravel()
        elif var in adata.obs:
            arr = adata.obs[var].values.astype(float)
        else:
            arr = np.zeros(n_pts)
        arr_show = np.log1p(arr) if use_log1p else arr
        return arr, arr_show

    def get_categorical_data(col):
        series = adata.obs[col].astype(str)
        cats = series.unique().tolist()
        # Try palette from .uns
        palette = None
        color_key = col + '_colors'
        if color_key in adata.uns:
            palette = adata.uns[color_key]
            if isinstance(palette, list) and len(palette) == len(cats):
                pass
            else:
                palette = None
        if palette is None:
            # fallback: matplotlib default color cycle
            base = plt.rcParams['axes.prop_cycle'].by_key()['color']
            palette = base * (len(cats) // len(base) + 1)
            palette = palette[:len(cats)]
        colormap = dict(zip(cats, palette))
        color_arr = [colormap[v] for v in series]
        idx_sort = np.argsort([cats.index(v) for v in series])
        return color_arr, series, cats, colormap, idx_sort

    # --- Initial data and plot
    arr, arr_show = get_numeric_data('log1p_total_counts', True)
    idx_sort = np.argsort(arr_show)
    xs_s, ys_s, arr_s = xs[idx_sort], ys[idx_sort], arr_show[idx_sort]
    scatter = ax.scatter(xs_s, ys_s, c=arr_s, cmap='Reds',
                        s=point_size['val'], alpha=alpha_val['val'], rasterized=True)

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect('equal', adjustable='datalim')

    ax.set_xlabel(f"{basis}[{dims[0]}]"); ax.set_ylabel(f"{basis}[{dims[1]}]")
    ax.set_title("Click ‘Annotate’ then draw polygons to label", fontsize=13)
    vmin, vmax = np.nanmin(arr_show), np.nanmax(arr_show)
    scatter.set_clim(vmin, vmax)
    cbar[0] = fig.colorbar(scatter, ax=ax, label='log1p_total_counts')
    legend_patches = []

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    def redraw(blit=True):
        fig.canvas.restore_region(background)
        ax.draw_artist(scatter)
        if cbar[0]:
            fig.canvas.draw_artist(cbar[0].ax)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

    # --- Coloring logic
    def update_colors(mode=None, key=None, use_log1p=None, full_redraw=False):
        # Remove legend
        for p in legend_patches:
            p.remove()
        legend_patches.clear()

        if mode:
            color_mode['type'] = mode
        if key:
            color_mode['key'] = key
        if use_log1p is not None:
            color_mode['log'] = use_log1p

        if color_mode['type'] == 'numeric':
            arr, arr_show = get_numeric_data(color_mode['key'], color_mode['log'])
            idx_sort = np.argsort(arr_show)
            scatter.set_offsets(np.c_[xs[idx_sort], ys[idx_sort]])
            scatter.set_array(arr_show[idx_sort])
            scatter.set_cmap('Reds')
            scatter.set_clim(np.nanmin(arr_show), np.nanmax(arr_show))
            cbar[0].set_label(f"{color_mode['key']}" + (" (log1p)" if color_mode['log'] else ""))
            cbar[0].update_normal(scatter)
        else:  # categorical
            color_arr, series, cats, colormap, idx_sort = get_categorical_data(color_mode['key'])
            scatter.set_offsets(np.c_[xs[idx_sort], ys[idx_sort]])
            scatter.set_array(None)
            scatter.set_cmap(None)
            scatter.set_facecolors(np.array(color_arr)[idx_sort])
            scatter.set_clim(None)
            cbar[0].ax.cla()
            cbar[0].set_label('')
            # Draw legend
            import matplotlib.patches as mpatches
            for c in cats:
                patch = mpatches.Patch(color=colormap[c], label=c)
                legend_patches.append(patch)
            leg = ax.legend(handles=legend_patches, title=color_mode['key'], loc='upper right', bbox_to_anchor=(1.13, 1))
            legend_patches.append(leg)
        scatter.set_alpha(alpha_val['val'])
        scatter.set_sizes(np.full(n_pts, point_size['val']))
        if full_redraw:
            fig.canvas.draw_idle()
        else:
            redraw()

    # --- Control panel: relative coords in [0,1] of ax_ctrl
    pad = 0.08
    panel_left = ax_ctrl.get_position().x0 + pad*ax_ctrl.get_position().width
    panel_width = ax_ctrl.get_position().width - 2*pad*ax_ctrl.get_position().width
    start = 0.98
    vspace = 0.085
    height = 0.065

    def ctrl_ax(rel_y, rel_height=height):
        return [panel_left, ax_ctrl.get_position().y1 - rel_y,
                panel_width, rel_height*ax_ctrl.get_position().height]

    # Display Options Header
    disp_header = fig.add_axes(ctrl_ax(0.02, 0.045), frameon=False)
    disp_header.set_xticks([]); disp_header.set_yticks([])
    disp_header.text(0, 0.7, "Display options", fontsize=12, fontweight='bold')
    disp_header.axis('off')

    # Coloring type switch (numeric/categorical)
    _, categorical = available_obs()
    col_types = ["Numeric", "Categorical"]
    ax_coltype = fig.add_axes(ctrl_ax(0.10, 0.045))
    radio_coltype = RadioButtons(ax_coltype, col_types, active=0)
    def on_coltype(label):
        if label == "Numeric":
            color_mode['type'] = 'numeric'
            update_colors(mode='numeric', key=color_mode['key'], use_log1p=color_mode['log'], full_redraw=True)
            ax_num.visible = True
            ax_cat.visible = False
        else:
            color_mode['type'] = 'categorical'
            update_colors(mode='categorical', key=color_mode['key'], full_redraw=True)
            ax_num.visible = False
            ax_cat.visible = True
    radio_coltype.on_clicked(on_coltype)

    # Numeric variable textbox
    numeric, categorical_cols = available_obs()
    ax_num = fig.add_axes(ctrl_ax(0.20, rel_height=0.065))
    num_box = TextBox(ax_num, 'Numeric var', initial=color_mode['key'])
    def on_num_submit(txt):
        color_mode['key'] = txt.strip()
        update_colors(mode='numeric', key=txt.strip(), use_log1p=color_mode['log'], full_redraw=True)
    num_box.on_submit(on_num_submit)

    # Categorical variable radio selector
    ax_cat = fig.add_axes(ctrl_ax(0.28, rel_height=0.13))
    cat_radio = RadioButtons(ax_cat, categorical_cols if categorical_cols else [''], active=0)
    if categorical_cols:
        color_mode['key'] = categorical_cols[0]
    else:
        color_mode['key'] = ''
    ax_cat.set_visible(False)
    def on_cat_select(label):
        color_mode['key'] = label
        update_colors(mode='categorical', key=label, full_redraw=True)
    cat_radio.on_clicked(on_cat_select)

    # Log1p checkbox (only for numeric)
    ax_log = fig.add_axes(ctrl_ax(0.41, rel_height=0.045))
    check_log = CheckButtons(ax_log, ['log1p'], [True])
    def on_check(label):
        state = check_log.get_status()[0]
        color_mode['log'] = state
        update_colors(use_log1p=state, full_redraw=True)
    check_log.on_clicked(on_check)

    # Size slider
    ax_size = fig.add_axes(ctrl_ax(0.50))
    size_sl = Slider(ax_size, 'Size', 1, 100, valinit=initial_size)
    def on_size(val):
        point_size['val'] = val
        scatter.set_sizes(np.full(n_pts, val))
        redraw()
    size_sl.on_changed(on_size)

    # Alpha slider
    ax_alpha = fig.add_axes(ctrl_ax(0.60))
    alpha_sl = Slider(ax_alpha, 'Alpha', 0.1, 1.0, valinit=initial_alpha)
    def on_alpha(val):
        alpha_val['val'] = val
        scatter.set_alpha(val)
        redraw()
    alpha_sl.on_changed(on_alpha)

    # --- Annotation widgets ---
    annot_header = fig.add_axes(ctrl_ax(0.70, rel_height=0.045), frameon=False)
    annot_header.set_xticks([]); annot_header.set_yticks([])
    annot_header.text(0, 0.7, "Annotation", fontsize=12, fontweight='bold')
    annot_header.axis('off')

    # Label TextBox
    ax_lab = fig.add_axes(ctrl_ax(0.80, rel_height=0.07))
    lab_box = TextBox(ax_lab, 'Label')
    lab_box.on_submit(lambda t: current_label.update(name=t.strip()))
    lab_box.color = 'white'
    def on_lab_click(event):
        lab_box.set_active(True)
        ax_lab.patch.set_facecolor("#e8f2ff") # Light blue highlight
        fig.canvas.draw_idle()
    def on_lab_focusout(event):
        ax_lab.patch.set_facecolor('white')
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', on_lab_click)
    fig.canvas.mpl_connect('axes_leave_event', on_lab_focusout)

    # Annotate button
    ax_btn = fig.add_axes(ctrl_ax(0.90, rel_height=0.065))
    btn    = Button(ax_btn, 'Annotate')
    fig._selector = None
    def on_button(_):
        if fig._selector is None:
            fig._selector = PolygonSelector(
                ax, onselect, useblit=True,
                props=dict(color='red', linewidth=1.5),
                handle_props=dict(marker='o', markersize=5,
                                  markeredgecolor='red',
                                  markerfacecolor='red', alpha=0.6)
            )
            print("🔹 Polygon mode ON")
        else:
            print("🔹 Already in polygon mode")
    btn.on_clicked(on_button)

    # --- Polygon selection with bbox pre-filter ---
    def onselect(verts):
        lbl = current_label['name']
        if not lbl:
            print("➜  Enter label, then click Annotate before drawing.")
            return
        v = np.array(verts)
        xmin, xmax = v[:,0].min(), v[:,0].max()
        ymin, ymax = v[:,1].min(), v[:,1].max()
        in_box = (xs>=xmin)&(xs<=xmax)&(ys>=ymin)&(ys<=ymax)
        if not in_box.any():
            print("➜  No points in bounding box.")
            return
        pts_box = pts[in_box]
        inside  = Path(verts).contains_points(pts_box)
        mask = np.zeros(n_pts, bool)
        mask[np.where(in_box)[0][inside]] = True
        n = mask.sum()
        if n == 0:
            print("➜  No points inside polygon.")
            return
        adata.obs.loc[mask, obs_key] = lbl
        print(f"➜  Assigned '{lbl}' to {n} cells.")
        redraw()

    plt.show(block=True)

