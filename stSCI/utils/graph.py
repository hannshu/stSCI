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


@Timer()
def build_graph_3d(
    adata: Union[sc.AnnData, np.ndarray], 
    multi_slice_key: str,
    knears: int, 
    cross_neigh: int,
    use_rep: str = 'spatial'
) -> torch.LongTensor:

    overall_edge_list = []
    slice_list = [
        adata[slice_name == adata.obs[multi_slice_key]] 
        for slice_name in adata.obs[multi_slice_key].unique()
    ]
    id_offset = 0

    def gen_cross_slice_edge(
        source_adata: sc.AnnData,
        target_adata: sc.AnnData, 
        mode: str
    ) -> torch.LongTensor:

        nbrs = NearestNeighbors(n_neighbors=cross_neigh).fit(target_adata.obsm[use_rep][:, :2])
        _, indices = nbrs.kneighbors(source_adata.obsm[use_rep][:, :2])

        source_indices = torch.repeat_interleave(
            torch.arange(source_adata.shape[0]), 
            torch.LongTensor([len(index_list) for index_list in indices])
        )
        target_indices = torch.Tensor([index for index_list in indices for index in index_list])
        source_indices = source_indices + id_offset
        if ('upper' == mode):
            target_indices = target_indices + id_offset - target_adata.shape[0]
        else:
            target_indices = target_indices + id_offset + source_adata.shape[0]
        edge_list = torch.vstack([source_indices, target_indices]).long()

        return edge_list

    for i in range(len(slice_list)):

        # build inner slice edges
        overall_edge_list.append(build_graph(slice_list[i], knears=knears)+id_offset)

        # build cross slice edges
        if (0 != i and len(slice_list)-1 != i):
            overall_edge_list.append(gen_cross_slice_edge(slice_list[i], slice_list[i-1], 'upper'))  # upper
            overall_edge_list.append(gen_cross_slice_edge(slice_list[i], slice_list[i+1], 'lower'))  # lower

        id_offset += slice_list[i].shape[0]

    edge_list = torch.hstack(overall_edge_list).long()

    return {
        'result': edge_list,
        'timer_note': f"Generate {edge_list.shape[1]} multi-slice edges, {(edge_list.shape[1] / adata.shape[0]) - 1:.3f} edges per spot"
    } 
