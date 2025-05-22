Welcome to stSCI's documentation!
===================================

.. image:: https://cdn.jsdelivr.net/gh/hannshu/stSCI/framework.png
   :alt: stSCI overview

Overview
--------

Spatial transcriptomics (ST) data reveals the locations of transcriptomes, offering crucial perspectives but at a compromised quality. Integrating ST data with single-cell transcriptomics (SC) data has proven to be an effective strategy for enhancing the quality of ST data, as demonstrated by previous research. We introduce stSCI (ST-SC Integration), a novel computational method that seamlessly fuses SC and ST data into a unified embedding space by incorporating a newly designed fusion module. Utilizing several simulated and real datasets, stSCI demonstrates its performance in batch correction across the two omics data, in clustering, and deconvolution of ST data, as well as in reconstructing spatial coordinates for SC data. Additionally, stSCI shows its potential for annotating ST data using only marker genes and cell types derived from SC data. Moreover, experiment result shows that stSCI performing well with both sequencing-based and imaging-based ST data, highlighting its capability as a powerful tool for comprehensive biological system analysis.

Contents
--------

.. toctree::

   installation
   section1_clustering
   section2_multi_clustering
   section3_deconvolution_sim
   section3_deconvolution_real
   section3_annotating
   section4_reconstruction
   section5_integration
