import time
import numpy as np
import random
import scanpy as sc
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse import issparse
from typing import List, Callable


class Timer():

    def __init__(self, note: str = '') -> None:
        self.note = note


    def __call__(self, func: Callable) -> Callable:

        def wrapper(*args, **kwargs):

            show_timer = kwargs.pop('show_timer') if ('show_timer' in kwargs) else True

            start_time = time.time()
            ret = func(*args, **kwargs)
            end_time = time.time()

            if (isinstance(ret, dict) and 'timer_note' in ret):
                if (show_timer):
                    print(f'>>> INFO: {ret["timer_note"]} ({end_time - start_time:.2f}s).')
                return ret['result']
            else:
                if (show_timer):
                    print(f'>>> INFO: {self.note} ({end_time - start_time:.2f}s).')

            return ret

        return wrapper


def set_seed(seed: int = 0) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_feature(adata: sc.AnnData, hvg: bool = False) -> np.ndarray:

    def get_x(adata):
        return adata.X.todense().A if (issparse(adata.X)) else adata.X

    if (False == hvg):
        return get_x(adata)
    else:
        return get_x(adata[:, adata.var.highly_variable])
    

def get_overlap_gene(adata_sets: List[sc.AnnData], top_genes: int, min_thershold: int = 3) -> List[sc.AnnData]:

    for i in range(len(adata_sets)):
        adata_sets[i].var_names_make_unique()
        adata_sets[i].obs_names_make_unique()
        if ('highly_variable' in adata_sets[i].var):
            adata_sets[i] = adata_sets[i][:, adata_sets[i].var.highly_variable]

    concat_data = sc.concat(adata_sets, label='batch', keys=list(range(len(adata_sets))))
    sc.pp.highly_variable_genes(concat_data, flavor="seurat_v3", n_top_genes=top_genes)
    concat_data = concat_data[:, concat_data.var.highly_variable]
    sc.pp.filter_genes(concat_data, min_cells=min_thershold)

    for i in range(len(adata_sets)):
        adata_sets[i] = adata_sets[i][:, concat_data.var.index]

    print(f'>>> INFO: Filtered {top_genes - concat_data.shape[1]} genes.')
    print(f'>>> INFO: Find {adata_sets[0].shape[1]} same HVGs, result data shapes: {[item.shape for item in adata_sets]}.')

    return adata_sets
