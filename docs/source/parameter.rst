Parameter introduction
======================

stSCI provides two primary functions for its core workflow: 

- ``stSCI.train()``: Train a new stSCI model from your single-cell and spatial transcriptomics data. 
- ``stSCI.eval()``: Load a pre-trained stSCI model and perform inference.

This section provides a detailed explanation of the parameters for each of these two functions.

stSCI.train
-----------

.. code:: python

   def train(
       sc_adata: sc.AnnData,
       st_adata: sc.AnnData,
       multi_slice_key: Optional[str] = None,
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
       cluster_para: Union[int, float] = 0.8,
       deconvolution: bool = False,
       coor_reconstruction: bool = False,
       model_save_path: Optional[str] = None,
       device: str = 'cuda' if (torch.cuda.is_available()) else 'cpu'
   ) -> Tuple[sc.AnnData]:

-  sc_adata: A ``sc.AnnData`` object containing the single-cell transcriptomics data.
-  st_adata: A ``sc.AnnData`` object containing the spatial transcriptomics data.
-  multi_slice_key: A ``str`` specifying the key in ``st_adata.obs`` that indicates different ST batches. This is used for analysising multi-slice spatial datasets, set to ``None`` if you have only a single slice.
-  hvg_count: An ``int`` specifying the number of highly variable genes (HVGs) to select for model training.
-  batch_sim_k: An ``int`` specifying the number of most similar single cells to sample for each spatial spot during the mini-batch training phase.
-  batch_sim_k: An ``int`` specifying the number of most similar single cells to consider for each spatial spot when inferring the final SC-ST relationships after the model is trained.
-  cluster_key: A ``str`` specifying the key in ``sc_adata.obs`` that contains the cell type labels for the single-cell data.
-  lr: A ``float`` specifying the learning rate for the model optimizer during training.
-  epochs: An ``int`` for the total number of training epochs to run.
-  update_iter: An ``int`` specifying the frequency (epochs) to update the MNN pairs used for data integration.
-  model_dims: A\ ``List[int]`` defining the architecture of the neural network. Each integer represents the number of neurons in a hidden layer.
-  clustering: A ``bool`` flag. If True, performs spatial domain clustering.
-  cluster_method: A ``str`` indicating the algorithm to use for clustering, we support ``mclust`` and ``louvain``. This is only used if clustering is ``True``.
-  cluster_para: A numeric input specifying the parameter for the chosen ``cluster_method``. For Louvain, this is the resolution parameter. For mclust, this is the cluster number.
-  deconvolution: A ``bool`` flag. If True, performs deconvolution.
-  coor_reconstruction: A ``bool`` flag. If True, performs coordinate reconstruction.
-  model_save_path: A ``str`` providing the file path where the trained model parameters will be saved. If ``None``, the model is not saved to disk.
-  device: A ``str`` specifying the computational device for model training (e.g., ‘cpu’, ‘cuda’, or ‘cuda:0’).

..

   **NOTE:** stSCI fully supports training on a CPU, ensuring accessibility for users who do not have a dedicated GPU. However, due to the computational intensity of deep learning, using a CUDA-enabled GPU is strongly recommended. If you do not have a compatible GPU, please specify ‘cpu’ for parameter ``device``.

stSCI.eval
----------

.. code:: python

   def eval(
       sc_adata: sc.AnnData,
       st_adata: sc.AnnData,
       model_path: Union[torch.nn.Module, str],
       multi_slice_key: Optional[str] = None,
       hvg_count: int = 3000,
       overall_sim_k: int = 100,
       cluster_key: str = 'cluster',
       clustering: bool = False,
       cluster_method: str = 'mclust',
       cluster_para: Union[int, float] = 0.8,
       deconvolution: bool = False,
       coor_reconstruction: bool = False,
       imputation: bool = False
   ) -> Tuple[sc.AnnData]:

``stSCI.eval`` shares most of its parameters with ``stSCI.train``. The key difference is the ``model_path`` parameter, which is used to specify the pre-trained model for inference. It flexibly accepts either a ``str`` type data pointing to the saved model file or a pre-loaded ``torch.nn.Module`` object directly.
