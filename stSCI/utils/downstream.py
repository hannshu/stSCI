import torch
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, List, Union, Optional
from .decomposition import get_mnn_matrix, nonzero_softmax
from .deconvolution import get_decon_result
from .transform_coor import trans_coor, impute_st


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
    clustering: bool = False,
    cluster_method: str = 'louvain',
    cluster_para: Union[int, float] = 0.8,
    deconvolution: bool = False,
    cluster_key: str = 'cluster',
    coor_reconstruction: bool = False,
    multi_slice_key: Optional[str] = None,
    imputation: bool = False
) -> Tuple[sc.AnnData]:

    sc_adata = sc_adata.copy()
    st_adata = st_adata.copy()

    # do clustering
    if (clustering):
        st_adata.obsm['recon_pca'] = PCA(n_components=30, random_state=0).fit_transform(st_adata.obsm['recon'])
        if ('mclust' == cluster_method):
            assert ('cluster' in st_adata.obs or isinstance(cluster_para, int)), \
                '>>> ERROR: mclust need cluster number as input'
            if ('cluster' in st_adata.obs):
                cluster_para = len(np.unique(st_adata['nan' != st_adata.obs['cluster'], :].obs['cluster']))
            st_adata.obs['cluster_result'] = mclust_R(st_adata, cluster_para, used_obsm='recon_pca')
        else:
            sc.pp.neighbors(st_adata, use_rep='recon_pca')
            sc.tl.louvain(st_adata, resolution=cluster_para, key_added='cluster_result')

    # do deconvolution
    if (deconvolution or coor_reconstruction or imputation):
        
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
            st_adata.obsm['decon_result'] = st_adata.obsm['decon_result'].div(st_adata.obsm['decon_result'].sum(axis=1), axis=0).fillna(0)

    # coor reconstruction
    if (coor_reconstruction or imputation):
        sc_adata = trans_coor(sc_adata, st_adata, multi_slice_key)

    # do imputation
    if (imputation):
        st_adata = impute_st(sc_adata, st_adata, sc_label_key=cluster_key)

    return sc_adata, st_adata
