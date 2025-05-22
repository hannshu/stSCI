import scanpy as sc
import numpy as np
from typing import Union, Optional
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from scipy.special import softmax
from . import Timer


def cal_average_distance(
    source: np.ndarray, 
    target: Optional[np.ndarray] = None, 
    n_neighbors: int = 6
) -> Union[float, np.ndarray]:

    tree = KDTree(source)
    if (isinstance(target, np.ndarray)):
        distances, indices = tree.query(target, k=n_neighbors)
    else:
        distances, indices = tree.query(source, k=n_neighbors+1)
    avg_distances = np.mean(distances[:, 1:])

    return avg_distances, indices


@Timer(note='Train SC coordination')
def trans_coor(
    sc_adata: sc.AnnData, 
    st_adata: sc.AnnData,
    multi_slice_key: Optional[str] = None,
    distance_threshold: Union[int, float] = 3,
    trans_key: str = 'trans_matrix',
    spatial_key: str = 'spatial',
    coor_save_key: str = 'spatial'
) -> sc.AnnData:

    sc_adata = sc_adata.copy()
    if (multi_slice_key is not None):
        st_adata = st_adata.copy()
        spatial_key_3d = f"{spatial_key}_3d"
        st_adata.obsm[spatial_key_3d] = st_adata.obsm[spatial_key]
        st_adata.obsm[spatial_key] = st_adata.obsm[spatial_key][:, :2]
    
    nan_indices = []
    related_index = [np.nonzero(sim)[0] for sim in sc_adata.obsm[trans_key]]
    related_sim = [sim[index] for sim, index in zip(sc_adata.obsm[trans_key], related_index)]
    related_coor = [st_adata.obsm[spatial_key][index] for index in related_index]

    distance_threshold, _ = cal_average_distance(
        st_adata.obsm[spatial_key], n_neighbors=distance_threshold
    )
    print(f'>>> Set distance threshold to {distance_threshold:.3f}.')

    sc_coors = []
    for i in tqdm(range(len(related_sim)), desc='>>> INFO: Train SC coordination'):

        cur_sim, cur_coor = related_sim[i], related_coor[i]

        if (0 == cur_sim.shape[0]):
            nan_indices.append(i)
            sc_coors.append([np.nan]*st_adata.obsm[spatial_key].shape[1])
            continue
        elif (1 == cur_coor.shape[0]):
            sc_coors.append(cur_coor[0])
            continue

        cluster_result = DBSCAN(eps=distance_threshold, min_samples=1).fit(cur_coor).labels_

        # find best coor set
        best_cluster, best_score = None, -np.inf
        for cluster in np.unique(cluster_result):

            # Skip noise
            if (-1 == cluster):
                continue  

            cluster_mask = (cluster_result == cluster)
            cluster_similarities = cur_sim[cluster_mask]
            cluster_coordinates = cur_coor[cluster_mask]
            similarity_sum = np.sum(cluster_similarities)
            distance = distance_matrix(cluster_coordinates, cluster_coordinates)
            mean_distance = np.mean(distance)

            # Scoring function
            score = similarity_sum - mean_distance  # changeable

            if (score > best_score):
                best_score = score
                best_cluster = cluster

        if (best_cluster):
            used_index = cluster_result == best_cluster
            cur_sim = softmax(cur_sim[used_index])
            cur_coor = cur_coor[used_index]

        sc_coors.append(np.sum(cur_sim.reshape(-1, 1)*cur_coor, axis=0))
    sc_coors = np.array(sc_coors)

    # Handle mismatched spot
    if (0 != len(nan_indices)):

        miss_mask = np.zeros(sc_adata.obsm['embedding'].shape[0], dtype=bool)
        miss_mask[nan_indices] = True
        
        miss_embed = sc_adata.obsm['embedding'][miss_mask]
        matched_embed = sc_adata.obsm['embedding'][~miss_mask]

        # assign coordinate of the most similar spot
        _, indices = cal_average_distance(matched_embed, miss_embed, n_neighbors=1)
        sc_coors[nan_indices] = sc_coors[np.nonzero(~miss_mask)[0][indices].flatten()]

    if (multi_slice_key is not None):
        batch_list = np.sort(st_adata.obs[multi_slice_key].unique())
        z_axis_list = []
        for i in range(len(batch_list)-1):
            slice_0_weight = st_adata[batch_list[i] == st_adata.obs[multi_slice_key]].obsm[trans_key].sum(axis=0)
            nan_mask = ~np.isnan(slice_0_weight)
            slice_0_weight[nan_mask] = np.random.rand(nan_mask.sum())
            slice_0_z_axis = st_adata[batch_list[i] == st_adata.obs[multi_slice_key]].obsm[spatial_key_3d][0, 2]
            slice_1_weight = st_adata[batch_list[i+1] == st_adata.obs[multi_slice_key]].obsm[trans_key].sum(axis=0)
            nan_mask = ~np.isnan(slice_1_weight)
            slice_1_weight[nan_mask] = np.random.rand(nan_mask.shape[0])
            slice_1_z_axis = st_adata[batch_list[i+1] == st_adata.obs[multi_slice_key]].obsm[spatial_key_3d][0, 2]

            overall_weight = np.vstack([slice_0_weight, slice_1_weight])
            overall_weight /= overall_weight.sum(axis=0)
            z_axis = overall_weight[0] * slice_0_z_axis + overall_weight[1] * slice_1_z_axis
            z_axis_list.append(z_axis.reshape(-1, 1))
        sc_coors = np.hstack([sc_coors, np.hstack(z_axis_list).mean(axis=1).reshape(-1, 1)])

    sc_adata.obsm[coor_save_key] = sc_coors
    return sc_adata
