import scanpy as sc
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point
from shapely.ops import unary_union


# source: https://github.com/gao-lab/GLUE/blob/v0.2.2/scglue/metrics.py
def seurat_alignment_score(
    x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
    n_repeats: int = 4, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = np.random.RandomState(0)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.array(repeat_scores)


def norm_coor(coor: np.ndarray) -> np.ndarray:

    x_min, y_min = coor[:, 0].min(), coor[:, 1].min()

    coor[:, 0] -= x_min
    coor[:, 1] -= y_min

    return coor


def row_argmax(x: np.ndarray) -> np.ndarray:

    mask = 0 == x.max(axis=1)
    result = np.argmax(x, axis=1)
    result[mask] = -1

    return result


def get_recon_iou_metric(
    projecy_coor: np.ndarray, 
    true_coor: np.ndarray, 
) -> float:

    # build the ground truth region
    nbrs = NearestNeighbors(n_neighbors=2).fit(true_coor)
    distance, _ = nbrs.kneighbors(true_coor)
    gt_region = unary_union([
        Point(spot).buffer(distance[:, 1].mean()) 
        for spot in true_coor
    ])

    # find true and false spots
    point_region = unary_union([
        Point(point).buffer(distance[:, 1].mean()) 
        for point in projecy_coor
    ])

    inter = gt_region.intersection(point_region).area
    union = gt_region.union(point_region).area

    return inter / union


def get_iou_per_domain(
    pred_sc_adata: sc.AnnData,
    st_adata: sc.AnnData
) -> List[Tuple[str, float]]:
    result = []
    for label in pred_sc_adata.obs['cluster'].unique():
        if (0 == np.sum(label == st_adata.obs['cluster'])):
            continue    # Reference data do not contain this domain/cell type
        iou = get_recon_iou_metric(
            pred_sc_adata[label == pred_sc_adata.obs['cluster']].obsm['spatial'], 
            st_adata[label == st_adata.obs['cluster']].obsm['spatial']
        )
        result.append((label, iou))
    return result
