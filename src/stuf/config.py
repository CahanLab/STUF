# Defaults
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Noto Sans', 'DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

DEFAULTS_SCATTER = {'s':2, 'cmap':'Reds', 'alpha':0.85, 'edgecolors':'none','linewidths':0}
DEFAULTS_CBAR = dict(fraction=0.046, pad=0.04, shrink=0.5)
DEFAULTS_LEGEND = dict(loc = 'upper center', bbox_to_anchor = (0.5, -0.05),ncol=2, fontsize='small', frameon=False)

from .plotting.helpers import make_bivariate_cmap
BIMAP_YELLOW = make_bivariate_cmap()


