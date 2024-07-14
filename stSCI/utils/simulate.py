import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from . import Timer, get_feature


# reference: https://github.com/BayraktarLab/cell2location/issues/204#issuecomment-1272416837
def inverse_lognormalize(adata: sc.AnnData, use_count: str = 'total_counts') -> sc.AnnData:

    adata = adata.copy()

    normalized_data = np.exp(get_feature(adata)) - 1
    data = normalized_data / 10000 * adata.obs[use_count].to_numpy().reshape(-1, 1)
    adata.X = csr_matrix(data.astype(int))

    return adata


def get_grid(coors: np.ndarray, distance: float) -> np.ndarray:

    x_scale = [coors[:, 0].min(), coors[:, 0].max()]
    y_scale = [coors[:, 1].min(), coors[:, 1].max()]

    return np.meshgrid(
        np.linspace(x_scale[0], x_scale[1], int((x_scale[1] - x_scale[0]) // distance)), 
        np.linspace(y_scale[0], y_scale[1], int((y_scale[1] - y_scale[0]) // distance))
    )


@Timer()
def get_simulated_data(
    adata: sc.AnnData, 
    distance: float, 
    label_tag: Optional[str] = 'cluster', 
    loc_style: str = 'visium', 
    used_obsm: str = 'spatial', 
    aggr_method: str = 'sum',
) -> sc.AnnData:

    assert(used_obsm in adata.obsm), '>>> ERROR: No coordinations in adata!'

    xs, ys = get_grid(adata.obsm[used_obsm], distance)
    array_row = np.tile(np.arange(xs.shape[0]), (xs.shape[1], 1)).T.reshape(-1)
    array_col = np.tile(np.arange(xs.shape[1]), (xs.shape[0], 1)).reshape(-1)

    if ('visium' == loc_style):
        xs[list(range(1, xs.shape[0], 2)), :] += distance / 2
    coors = np.array([xs.reshape(-1), ys.reshape(-1)]).T

    nbrs = NearestNeighbors(radius=distance/2).fit(adata.obsm[used_obsm])
    neighbors = nbrs.radius_neighbors(coors, return_distance=False)
    entity_item = np.array([len(item) != 0 for item in neighbors])

    neighbors = neighbors[entity_item]
    coors = coors[entity_item]
    array_row = array_row[entity_item]
    array_col = array_col[entity_item]

    x = get_feature(adata)
    aggr_func = np.sum if ('sum' == aggr_method) else np.mean  
    x = np.array([aggr_func(x[neighbors[i]], axis=0) for i in range(len(neighbors))])

    obs_df = pd.DataFrame(
        data=np.array([array_row, array_col, [len(neighbors[i]) for i in range(len(neighbors))]]).T
        if (not 'total_counts' in adata.obs) else np.array([
            array_row, array_col, 
            [len(neighbors[i]) for i in range(len(neighbors))],
            [int(aggr_func(adata.obs['total_counts'][neighbors[i]], axis=0)) for i in range(len(neighbors))]
        ]).T, 
        index=[
            f'{list(adata.obs.index[neighbors[i]].astype(str))}'
            for i in range(len(neighbors))
        ], 
        columns=['array_row', 'array_col', 'cell_count', 'total_counts'] 
        if ('total_counts' in adata.obs) else ['array_row', 'array_col', 'cell_count']
    )

    sim_adata = sc.AnnData(csr_matrix(x), obs=obs_df, var=adata.var)
    sim_adata.obsm[used_obsm] = coors

    if (label_tag):
        freq = [Counter(adata.obs[label_tag][neighbors[i]]) for i in range(len(neighbors))]

        sim_adata.obs['cluster'] = [item.most_common(1)[0][0] for item in freq]
        sim_adata.obsm['type_count'] = pd.DataFrame(data=freq, index=sim_adata.obs.index).fillna(0).astype(int)
        sim_adata.obsm['deconvolution_result'] = sim_adata.obsm['type_count'].div(sim_adata.obsm['type_count'].sum(axis=1), axis=0)

    return {
        'result': sim_adata,
        'timer_note': f'Generate simulated ST data, simulated data shape {sim_adata.shape}, average {np.mean([len(neighbors[i]) for i in range(len(neighbors))]):.3f} cells in each spot.'
    } 
