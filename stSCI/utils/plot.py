import scanpy as sc
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_distribution(
    adata: sc.AnnData,
    use_coor: str = 'spatial',
    spot_size: int = 150,
    figure_size: int = 5,
    label_key: str = 'cluster',
    score: Optional[Dict[str, float]] = None,
    used_label: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    
    if (None == used_label):
        used_label = np.unique(adata.obs[label_key])       
    label_count = len(used_label)

    n_rows = int(np.ceil(np.sqrt(label_count)))
    n_cols = int(np.ceil(label_count / n_rows))
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*figure_size, n_rows*figure_size))
    axes = axes.flatten()

    for i, label in enumerate(used_label):
        title = f'{label} (score={score[label]:.3f})' if (score) else label
        sc.pl.spatial(adata, basis=use_coor, color=label_key, groups=[label], 
                        title=title, spot_size=spot_size, ax=axes[i], show=False)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if (save_path):
        plt.savefig(save_path, dpi=300)


def plot_deconvolution(
    adata: sc.AnnData,
    spot_size: Union[int, float] = 150,
    show_keys: Optional[List[str]] = None,
    spatial_key: str = 'spatial',
    decon_result_key: str = 'decon_result',
    ground_truth_key: Optional[str] = 'deconvolution_result',
    save_path: Optional[str] = None
) -> None:
    
    if (None == show_keys):
        show_keys = adata.obsm[decon_result_key].columns.tolist()

    # plot decon predict distribution
    print('>>> Predicted deconvolution result')
    sc.pl.spatial(
        sc.AnnData(
            obs=adata.obsm[decon_result_key], 
            obsm={'spatial': adata.obsm[spatial_key]}
        ),  
        color=show_keys, 
        spot_size=spot_size,
        cmap='Blues',
        save=f'pred_{save_path}' if (save_path) else None
    )

    # plot ground truth distribution
    if (ground_truth_key in adata.obsm):
        print('>>> Ground truth deconvolution result')
        sc.pl.spatial(
            sc.AnnData(
                obs=adata.obsm[ground_truth_key], 
                obsm={'spatial': adata.obsm[spatial_key]}
            ), 
            color=show_keys, 
            spot_size=spot_size,
            save=f'ground_truth_{save_path}' if (save_path) else None
        )


def plot_volcano(
    adata: sc.AnnData,
    label_name: str,
    pval_threshold: float = 0.05,
    logfc_threshold: float = 1,
    sig_list: Optional[List[str]] = None,
    spot_size: Optional[float] = None,
    save_path: str = None
):
    
    assert ('rank_genes_groups' in adata.uns), '>>> ERROR: MUST run `sc.tl.rank_genes_groups` first!'
    assert (None == sig_list or isinstance(sig_list, list)), f'>>> ERROR: sig_list should be a list type, not a {type(sig_list)}!'

    pvals = adata.uns['rank_genes_groups']['pvals_adj'][label_name]         # p-values
    logfc = adata.uns['rank_genes_groups']['logfoldchanges'][label_name]    # Log2 fold changes
    genes_name = adata.var_names

    # generate dataframe
    df = pd.DataFrame({'gene': genes_name, 'log2FoldChange': logfc, 'pval': pvals})
    df['-log10(pval)'] = -np.log10(df['pval'])
    outliers_df = df[(df['pval'] < pval_threshold) & (abs(df['log2FoldChange']) > logfc_threshold)]

    down = outliers_df[outliers_df['log2FoldChange'] < -logfc_threshold]
    up = outliers_df[outliers_df['log2FoldChange'] > logfc_threshold]
    stable = df[~df.isin(outliers_df).all(axis=1)]

    # draw volcano plot
    plt.scatter(down['log2FoldChange'], down['-log10(pval)'], s=spot_size, color='tab:blue', edgecolors='none')
    plt.scatter(up['log2FoldChange'], up['-log10(pval)'], s=spot_size, color='tab:red', edgecolors='none')
    plt.scatter(stable['log2FoldChange'], stable['-log10(pval)'], s=spot_size, color='tab:gray', edgecolors='none')

    # add significant gene name
    if (None != sig_list):
        for gene in sig_list:
            row = df[df['gene'] == gene]
            plt.text(row['log2FoldChange'], row['-log10(pval)'], gene)

    plt.title(f'Volcano plot for {label_name}')
    plt.xlabel('Log2FoldChange')
    plt.ylabel('-log10(P-value)')
    plt.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--')
    plt.axvline(x=logfc_threshold, color='black', linestyle='--')
    plt.axvline(x=-logfc_threshold, color='black', linestyle='--')
    plt.savefig(f'volcano_{save_path}') if (None != save_path) else None
    plt.show()


def plot_roc_curve(truth_list: List[list], pred_list: List[list], class_name: List[str]) -> None:

    class_count = len(class_name)
    fpr_list = []
    tpr_list = []
    score_list = []

    # calculate auc score
    for i in range(class_count):
        fpr, tpr, _ = roc_curve(truth_list[i], pred_list[i]) 
        roc_auc = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        score_list.append(roc_auc)

    # plot roc curve
    plt.figure()  
    for i in range(class_count):
        plt.plot(fpr_list[i], tpr_list[i], label=f'{class_name[i]} (AUC = {score_list[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
