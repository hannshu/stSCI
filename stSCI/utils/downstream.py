import torch
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, List, Union
from scipy.stats import pearsonr
from . import get_feature
from .decomposition import get_mnn_matrix, nonzero_softmax
from .deconvolution import get_decon_result
from .transform_coor import trans_coor


# source: https://github.com/QIFEIDKN/STAGATE/blob/main/STAGATE/utils.py
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='embedding', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    return mclust_res.astype(int).astype(str)


def downstream_analysis(
    sc_adata: sc.AnnData, 
    st_adata: sc.AnnData, 
    model: torch.nn.Module,
    overall_sim_k: int, 
    label_name: np.ndarray,
    clustering: bool = True,
    cluster_method: str = 'mclust',
    deconvolution: bool = True,
    cluster_key: str = 'cluster',
    coor_reconstruction: bool = True,
) -> Tuple[sc.AnnData]:

    sc_adata = sc_adata.copy()
    st_adata = st_adata.copy()

    # clustering
    if (clustering):
        st_adata.obsm['recon_pca'] = PCA(n_components=30, random_state=0).fit_transform(st_adata.obsm['recon'])
        if ('mclust' == cluster_method and 'cluster' in st_adata.obs):
            st_adata.obs['cluster_result'] = mclust_R(
                st_adata, 
                len(np.unique(st_adata['nan' != st_adata.obs['cluster'], :].obs['cluster'])), 
                used_obsm='recon_pca'
            )
        else:
            sc.pp.neighbors(st_adata, use_rep='recon_pca')
            sc.tl.louvain(st_adata, resolution=0.8, key_added='cluster_result')

    if (deconvolution or coor_reconstruction):
        
        # get transfer matrix
        trans_matrix = get_mnn_matrix(torch.FloatTensor(sc_adata.obsm['embedding']), 
                                    torch.FloatTensor(st_adata.obsm['embedding']), 
                                    overall_sim_k, return_type='mnn_matrix')
        sc_adata.obsm['trans_matrix'] = nonzero_softmax(trans_matrix.float()).numpy()
        st_adata.obsm['trans_matrix'] = nonzero_softmax(trans_matrix.T.float()).numpy()
        
        # get deconvolution result
        if (deconvolution):
            decon_result = model.softmax_func(model.trans_matrix).detach().cpu().numpy()
            deconv_prop = get_decon_result(st_adata.obsm['trans_matrix'], st_adata, sc_adata.obs[cluster_key])
            st_adata.obsm['decon_result'] = pd.DataFrame(
                (decon_result + deconv_prop[label_name].to_numpy()) / 2,
                index=st_adata.obs.index, columns=label_name
            )

        # coor reconstruction
        if (coor_reconstruction):
            sc_adata = trans_coor(sc_adata, st_adata)

    return sc_adata, st_adata


def find_BCD(
    adata: sc.AnnData, 
    ref_gene: str, 
    cluster_key: str = 'cluster_result',
    **visual_kwargs
) -> Union[List, float]:

    # init ref list
    gene_epx = get_feature(adata[:, ref_gene]).reshape(-1)
    ref_list = np.zeros((gene_epx.shape))
    ref_list[gene_epx >= np.quantile(gene_epx, 0.75)] = 1

    # find BCD
    label_list, correlation_score = _find_BCD(adata.obs[cluster_key].to_numpy(), ref_list)

    # visualize
    sc.pl.spatial(adata, color=[ref_gene, cluster_key], groups=label_list, title=[f'Reference gene {ref_gene}', 'Cluster set'], cmap='Blues', **visual_kwargs)

    return label_list, correlation_score


# find the best correlated domain
def _find_BCD(
    label_list: List, 
    ref_list: List, 
    cur_label_list: List = [], 
    cur_max_score: float = 0
) -> Union[List, float]:
    
    # init, filter label
    label = np.unique(label_list)[[
        l not in cur_label_list for l in np.unique(label_list)
    ]]
    max_score = 0
    max_label_list = None

    # find better correlated label list
    for l in label:

        used_label = cur_label_list + [l]

        dist = np.zeros(len(label_list))
        dist[[l in used_label for l in label_list]] = 1
        cur_score = pearsonr(dist, ref_list)[0]

        if (cur_score > max_score):
            max_score = cur_score
            max_label_list = used_label

    if (max_label_list and max_score >= cur_max_score):
        return _find_BCD(label_list, ref_list, max_label_list, max_score)
    else:
        return cur_label_list, cur_max_score
