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
    distance_threshold: Union[int, float] = 3,
    trans_key: str = 'trans_matrix',
    spatial_key: str = 'spatial',
    coor_save_key: str = 'spatial'
) -> sc.AnnData:

    sc_adata = sc_adata.copy()
    
    related_index = [np.nonzero(sim)[0] for sim in sc_adata.obsm[trans_key]]
    related_sim = [sim[index] for sim, index in zip(sc_adata.obsm[trans_key], related_index)]
    related_coor = [st_adata.obsm[spatial_key][index] for index in related_index]

    if (isinstance(distance_threshold, int)):
        distance_threshold, _ = cal_average_distance(
            st_adata.obsm[spatial_key], n_neighbors=distance_threshold
        )
        print(f'>>> Set distance threshold to {distance_threshold:.3f}.')

    sc_coors = []
    for i in tqdm(range(len(related_sim)), desc='>>> INFO: Train SC coordination'):

        cur_sim, cur_coor = related_sim[i], related_coor[i]

        if (0 == cur_sim.shape[0]):
            sc_coors.append([np.nan, np.nan])
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

    sc_adata.obsm[coor_save_key] = np.array(sc_coors)
    return sc_adata
