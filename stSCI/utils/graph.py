import torch
import scanpy as sc
import numpy as np
from typing import Union
from sklearn.neighbors import NearestNeighbors
import faiss
from . import Timer, get_feature


@Timer()
def build_graph(
    adata: Union[sc.AnnData, np.ndarray], 
    knears: int, 
    use_rep: str = 'spatial'
) -> torch.LongTensor:

    if (isinstance(adata, sc.AnnData)):
        if ('X' == use_rep):
            coor = get_feature(adata)
        else:
            coor = adata.obsm[use_rep]
    else:
        coor = adata

    if (30 > coor.shape[1]):
        nbrs = NearestNeighbors(n_neighbors=knears+1, metric='l2').fit(coor)
        _, indices = nbrs.kneighbors(coor)
    else:
        index = faiss.IndexFlatL2(coor.shape[1])
        index.add(coor)
        _, indices = index.search(coor, knears+1)

    source_indices = torch.repeat_interleave(
        torch.arange(coor.shape[0]), 
        torch.LongTensor([len(index_list) for index_list in indices])
    )
    target_indices = torch.Tensor([index for index_list in indices for index in index_list])
    edge_list = torch.vstack([source_indices, target_indices]).long()

    return {
        'result': edge_list,
        'timer_note': f"Generate {edge_list.shape[1]} edges, {(edge_list.shape[1] / adata.shape[0]) - 1:.3f} edges per spot"
    } 
