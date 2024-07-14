import scanpy as sc
from typing import List, Tuple, Union, Optional, Literal
import numpy as np
import igraph as ig
import torch
from sklearn.utils.extmath import randomized_svd
from . import Timer, get_feature
from .graph import build_graph


# source: https://github.com/gao-lab/SLAT/blob/main/scSLAT/model/batch.py
@Timer(note='Finish PCA')
def dual_pca(
    X: Union[np.ndarray, torch.FloatTensor], 
    Y: Union[np.ndarray, torch.FloatTensor], 
    dim: int = 30, 
    mode: str = 'dpca'
) -> Union[Tuple[np.ndarray], Tuple[torch.FloatTensor]]:

    assert X.shape[1] == Y.shape[1]

    if (isinstance(X, torch.Tensor)):
        _X, _Y = X.cpu().numpy(), Y.cpu().numpy()
    else:
        _X, _Y = X, Y
    U, S, Vh = randomized_svd(_X @ _Y.T, n_components=dim, random_state=0)

    if ('dpca' != mode):
        return U, Vh.T
    
    Z_x = U @ np.sqrt(np.diag(S))
    Z_y = Vh.T @ np.sqrt(np.diag(S))

    return torch.FloatTensor(Z_x), torch.FloatTensor(Z_y)


@Timer(note='Finish centroid generation')
def get_cluster_centroid(
    feature: torch.FloatTensor, 
    k: int = 15, 
    resolution: float = 1.0
) -> torch.FloatTensor:

    g = ig.Graph(
        n=feature.shape[0], 
        edges=build_graph(feature.numpy(), knears=k, show_timer=False).numpy().T
    )
    partition = list(g.community_multilevel(resolution=resolution))
    centroids = torch.stack([torch.mean(feature[p], dim=0) for p in partition])

    return centroids


def get_k_neighbor(
    source: torch.FloatTensor, 
    target: torch.FloatTensor, 
    k: int, 
    threshold: Optional[float] = None
) -> List[torch.LongTensor]:

    sim = -torch.cdist(source, target)
    sim, indics = torch.topk(torch.sigmoid(sim), k=k, dim=1)
    mask = sim >= (threshold if (None != threshold) else sim.mean())

    source_indices = torch.arange(sim.shape[0]).unsqueeze(1).expand_as(indics).to(mask.device)
    filtered_source_indics = source_indices[mask]
    filtered_target_indics = indics[mask]

    return [filtered_source_indics, filtered_target_indics], sim[mask]


def get_similar_matrix(
    index: List[torch.LongTensor], 
    sim_value: torch.FloatTensor,
    shape: List[int]
) -> torch.LongTensor:

    sim_matrix = torch.zeros(shape)
    sim_matrix[index[0], index[1]] = sim_value.cpu()

    return sim_matrix


@Timer()
def get_mnn_matrix(
    source: torch.FloatTensor, 
    target: torch.FloatTensor, 
    k: Union[int, list] = 100,
    threshold: Optional[float] = None,
    return_type: Literal['edge_index', 'sim_matrix'] = 'edge_index'
) -> torch.LongTensor:
    
    if (isinstance(k, list)):
        s2t_k, t2s_k = k
    else:
        s2t_k = t2s_k = k

    s2t_pairs, s2t_sim = get_k_neighbor(source, target, s2t_k, threshold)
    t2s_pairs, t2s_sim = get_k_neighbor(target, source, t2s_k, threshold)
    s2t_sim_matrix = get_similar_matrix(s2t_pairs, s2t_sim, [source.shape[0], target.shape[0]])
    t2s_sim_matrix = get_similar_matrix(t2s_pairs, t2s_sim, [target.shape[0], source.shape[0]]).T
    mnn_matrix = (0 != s2t_sim_matrix) & (0 != t2s_sim_matrix)

    if ('edge_index' == return_type):
        result = torch.nonzero(mnn_matrix).T
    else:
        result = (s2t_sim_matrix + t2s_sim_matrix) * mnn_matrix / 2

    return {
        'result': result,
        'timer_note': f"Generate {torch.sum(mnn_matrix)} MNN pairs, {(torch.sum(mnn_matrix) / mnn_matrix.shape[0]):.3f} pairs per SC cell; {(torch.sum(mnn_matrix) / mnn_matrix.shape[1]):.3f} pairs per ST spot"
    } 


def nonzero_softmax(matrix: torch.FloatTensor) -> torch.FloatTensor:
    return torch.sparse.softmax(matrix.to_sparse(), dim=1).to_dense()


def get_integration_metric(adata: sc.AnnData, batch_key: str, embedding_key: str, label_key: str = None) -> None:

    from harmonypy import compute_lisi
    from scib.metrics import silhouette_batch

    print(f'>>> LISI(↑): {compute_lisi(adata.obsm[embedding_key], adata.obs, [batch_key]).mean():.3f}')
    if (label_key):
        print(f'>>> Batch ASW(↑): {silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embedding_key):.3f}')


def get_cell_type_centroids(sc_adata: sc.AnnData, cluster_key: str, use_rep: Optional[str] = None) -> Tuple[torch.Tensor, np.ndarray]:

    if (use_rep):
        features = sc_adata.obsm[use_rep]
    else:
        features = get_feature(sc_adata)
    cluster_centroid = []

    for cell_type in np.unique(sc_adata.obs[cluster_key].astype(str)):

        mask = cell_type == sc_adata.obs[cluster_key]
        cluster_centroid.append(np.mean(features[mask], axis=0))

    return torch.FloatTensor(cluster_centroid), np.unique(sc_adata.obs[cluster_key])
