import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter
from typing import Union
from . import Timer


@Timer(note='Finish generate deconvolution result')
def get_decon_result(
    sim_matrix: np.ndarray,
    st_adata: sc.AnnData, 
    label_list: np.ndarray, 
) -> Union[list, pd.DataFrame]:

    label_per_spot = [list(label_list[item]) for item in sim_matrix.astype(bool)]
    label_freq = [Counter(spot_label_list) for spot_label_list in label_per_spot]

    propertion_matrix = pd.DataFrame(
        [dict(item) for item in label_freq], 
        index=st_adata.obs.index, 
        columns=np.unique(label_list)
    ).fillna(0).astype(int) 
    propertion_matrix = propertion_matrix.div(propertion_matrix.sum(axis=1), axis=0).fillna(0)

    return propertion_matrix
