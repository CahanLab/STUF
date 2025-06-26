
from .categorize import (
    paga_connectivities_to_igraph,
    graph_from_nodes_and_edges
)

from .comparison import (
    gsea_on_deg,
    collect_gsea_results_from_dict,
    convert_diffExp_to_dict,
    deg
)

# API
__all__ = [
    "paga_connectivities_to_igraph",
    "graph_from_nodes_and_edges",
    "gsea_on_deg",
    "collect_gsea_results_from_dict",
    "convert_diffExp_to_dict",
    "deg"
]

