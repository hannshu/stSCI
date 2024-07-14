import torch
import scanpy as sc
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Union
from .modules.stSCI import stSCI
from .utils import (
    Timer, 
    set_seed, 
    get_feature, 
    get_overlap_gene
)
from .utils.graph import build_graph
from .utils.decomposition import (
    dual_pca, 
    get_mnn_matrix, 
    get_cluster_centroid,
    get_cell_type_centroids
)
from .utils.downstream import downstream_analysis


@Timer(note='Train stSCI model')
def train(
    sc_adata: sc.AnnData,
    st_adata: sc.AnnData,
    hvg_count: int = 3000,
    batch_sim_k: int = 25,
    overall_sim_k: int = 100,
    cluster_key: str = 'cluster',
    init_trans_matrix: bool = False,
    lr: float = 1e-3,
    epochs: int = 500,
    update_iter: int = 50,
    model_dims: List[int] = [512, 30],
    clustering: bool = False,
    cluster_method: str = 'mclust',
    deconvolution: bool = False,
    coor_reconstruction: bool = False,
    model_save_path: Optional[str] = None,
    device: str = 'cuda' if (torch.cuda.is_available()) else 'cpu'
) -> Tuple[sc.AnnData]:
    
    assert(sc_adata.X.shape[0] > st_adata.X.shape[0]), \
        '>>> ERROR: The input must include more SC cells (more than ST spots).'
    set_seed(0) # set random seed

    sc_adata = sc_adata.copy()
    st_adata = st_adata.copy()
    
    # perpare input data
    sc_result, st_result = get_overlap_gene([sc_adata, st_adata], top_genes=hvg_count)
    sc_adata, st_adata = sc_adata[sc_result.obs.index, :], st_adata[st_result.obs.index, :]

    cell_type_centroids, label_name = get_cell_type_centroids(sc_adata[:, sc_result.var.index], cluster_key)
    cell_type_centroids = cell_type_centroids.to(device)
    sc_data = torch.FloatTensor(get_feature(sc_result)).to(device)

    st_data = Data(
        x=torch.FloatTensor(get_feature(st_result)), 
        edge_index=build_graph(st_adata, knears=6)
    ).to(device)

    # reshuffle sc data and generate batch mnn pairs
    reshuffled_sc_data = sc_data[np.random.randint(sc_data.shape[0], size=sc_data.shape[0])]
    sc_pca, st_pca = dual_pca(sc_data.cpu().numpy(), st_data.x.cpu().numpy())
    sc_centroid = get_cluster_centroid(sc_pca).to(device)
    st_centroid = get_cluster_centroid(st_pca).to(device)
    batch_size = st_data.x.shape[0]
    mnn_pairs = []
    p_list = []

    for i in range(int(sc_data.shape[0] / batch_size)):
        sc_pca, st_pca = dual_pca(
            reshuffled_sc_data[i*batch_size: (i+1)*batch_size, :].cpu(), 
            st_data.x.cpu(),
            show_timer=False
        )
        mnn_pairs.append(get_mnn_matrix(sc_pca, st_pca, batch_sim_k, show_timer=False))
        p_list.append([])

    _init_trans_matrix = None
    if (init_trans_matrix):
        sc_adata.obsm['sc_pca'] = sc_pca.detach().cpu().numpy()
        pca_centroid, _ = get_cell_type_centroids(sc_adata, cluster_key, 'sc_pca')
        _init_trans_matrix = torch.mm(st_pca, torch.pinverse(pca_centroid)).to(device)

    # init model
    model = stSCI(
        input_dim=sc_result.X.shape[1], 
        st_count=st_result.X.shape[0],
        cluster_count=cell_type_centroids.shape[0],
        centroids=(sc_centroid, st_centroid), 
        dims=model_dims,
        init_trans_matrix=_init_trans_matrix
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train model
    model.train()
    for epoch in tqdm(range(epochs), desc='>>> Train stSCI'):

        for i in range(int(sc_data.shape[0] / batch_size)):

            cur_sc_data = torch.concat(
                [reshuffled_sc_data[i*batch_size: (i+1)*batch_size, :], cell_type_centroids], dim=0
            )

            optimizer.zero_grad()
            embed, recon, q_list = model(cur_sc_data, st_data, cell_type_centroids.shape[0])

            if (0 == epoch % update_iter):
                # update DEC p distribution
                p_list[i] = [
                    model.get_p_distribution(q_list[0]),
                    model.get_p_distribution(q_list[1])
                ]
                if (0 != epoch):
                    # update mnn pairs
                    mnn_pairs[i] = get_mnn_matrix(
                        embed[0][: -cell_type_centroids.shape[0]], embed[1], batch_sim_k, show_timer=False
                    )

            loss = model.loss(
                cur_sc_data[: -cell_type_centroids.shape[0]], st_data.x, 
                mnn_pairs[i], recon, 
                (embed[0][: -cell_type_centroids.shape[0]], embed[1]), 
                p_list[i], q_list, 
                embed[0][-cell_type_centroids.shape[0]:]
            )
            loss.backward()
            optimizer.step()

    if (model_save_path):
        torch.save(model, model_save_path)

    # generate embedding
    eval_model = model.cpu()
    eval_model.eval()
    embed, recon = eval_model(sc_data.cpu(), st_data.cpu())

    sc_adata.obsm['embedding'] = embed[0].detach().cpu().numpy()
    st_adata.obsm['embedding'] = embed[1].detach().cpu().numpy()
    sc_adata.obsm['recon'] = recon[0].detach().cpu().numpy()
    st_adata.obsm['recon'] = recon[1].detach().cpu().numpy()

    return downstream_analysis(sc_adata, st_adata, eval_model, overall_sim_k, 
                               label_name, clustering, cluster_method, 
                               deconvolution, cluster_key, coor_reconstruction)


@Timer(note='Inference stSCI model')
def eval(
    sc_adata: sc.AnnData,
    st_adata: sc.AnnData,
    model_path: Union[torch.nn.Module, str],
    hvg_count: int = 3000,
    overall_sim_k: int = 100,
    cluster_key: str = 'cluster',
    clustering: bool = False,
    cluster_method: str = 'mclust',
    deconvolution: bool = False,
    coor_reconstruction: bool = False,
) -> Tuple[sc.AnnData]:

    sc_adata = sc_adata.copy()
    st_adata = st_adata.copy()
    
    # perpare input data
    sc_result, st_result = get_overlap_gene([sc_adata, st_adata], top_genes=hvg_count)
    sc_data = torch.FloatTensor(get_feature(sc_result))
    st_data = Data(
        x=torch.FloatTensor(get_feature(st_result)), 
        edge_index=build_graph(st_adata, knears=6)
    )
    label_name = np.unique(sc_adata.obs[cluster_key])

    # generate embedding
    if (isinstance(model_path, str)):
        model = torch.load(model_path).cpu()
    else:
        model = model_path.cpu()

    model.eval()
    embed, recon = model(sc_data, st_data)

    sc_adata.obsm['embedding'] = embed[0].detach().cpu().numpy()
    st_adata.obsm['embedding'] = embed[1].detach().cpu().numpy()
    sc_adata.obsm['recon'] = recon[0].detach().cpu().numpy()
    st_adata.obsm['recon'] = recon[1].detach().cpu().numpy()

    return downstream_analysis(sc_adata, st_adata, model, overall_sim_k, 
                               label_name, clustering, cluster_method, 
                               deconvolution, cluster_key, coor_reconstruction)
