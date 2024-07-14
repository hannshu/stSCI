import scanpy as sc
import numpy as np
import pandas as pd
from typing import Dict
import sklearn.neighbors
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from .simulate import get_grid
from .transform_coor import cal_average_distance


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
        nn = sklearn.neighbors.NearestNeighbors(
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


def get_recon_metric(
    sc_adata: sc.AnnData, 
    st_adata: sc.AnnData, 
    grid_width: float = 0.1,
    metric: str = 'iou',
    coor_key: str = 'spatial', 
    cluster_key: str = 'cluster'
) -> Dict[str, float]:
    
    assert (metric in ['iou', 'pcc', 'rmse']), f'>>> ERROR: {metric} not supported. ' + \
        'Please choose from ["iou" (intersection over union), "pcc" (pearson correlation coefficient), ' + \
        '"rmse" (root mean square error)]'

    sc_adata = sc_adata[pd.DataFrame(sc_adata.obsm[coor_key]).dropna().index].copy()
    st_adata = st_adata.copy()

    sc_coor = norm_coor(sc_adata.obsm[coor_key])
    st_coor = norm_coor(st_adata.obsm[coor_key])

    xs, ys = get_grid(st_coor, grid_width)
    canvas = np.hstack((xs.reshape(-1, 1), ys.reshape(-1, 1)))

    label_list = np.unique(st_adata.obs[cluster_key])
    sc_pred = np.zeros((canvas.shape[0], label_list.shape[0]), dtype=int)
    st_truth = np.zeros((canvas.shape[0], label_list.shape[0]), dtype=int)

    for i, label_name in enumerate(label_list):

        cur_sc_coor = sc_coor[label_name == sc_adata.obs[cluster_key]]
        cur_st_coor = st_coor[label_name == st_adata.obs[cluster_key]]

        _, sc_indices = cal_average_distance(canvas, cur_sc_coor, 1)
        _, st_indices = cal_average_distance(canvas, cur_st_coor, 1)

        for ind in np.squeeze(sc_indices):
            sc_pred[ind][i] += 1
        for ind in np.squeeze(st_indices):
            st_truth[ind][i] += 1

    sc_pred_label = row_argmax(sc_pred)
    st_truth_label = row_argmax(st_truth)

    mask = (-1 != sc_pred_label) & (-1 != st_truth_label)
    sc_pred_label = sc_pred_label[mask]
    st_truth_label = st_truth_label[mask]

    if ('pcc' == metric):
        score = pearsonr(st_truth_label, sc_pred_label)[0]
    elif ('rmse' == metric):
        score = np.sqrt(mean_squared_error(st_truth_label, sc_pred_label))

    return score
