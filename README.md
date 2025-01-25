# SVGRN

### Environment setup

- Use the environment.yml to create the anaconda virtual environment for SVGRN:

```
cd SVGRN
conda env create -f environment.yml
```

### Stage 1

- run stage 1 training bash file

```
bash train_main_stage1.sh
```

### Stage 2

- unzip the example sim data for stage 2 training

```
cd in_sim/g110_c2k_0.1
mkdir cell_specific_gt_gene_pair
unzip cell_specific_gt_gene_pair.zip -d cell_specific_gt_gene_pair
```

- run stage 2 training bash file

```
bash train_main_stage2_allcell.sh
```

### scMultiSim configuration for simulation data generation

In our experiments, we used scMultiSim[^1] to generate simulated spatial transcriptomics datasets with dynamic GRNs.

More information about their R package installation can be found in their document: https://zhanglabgt.github.io/scMultiSim/articles/

Here, we provide the R script with hyperparameters we used for running scMultiSim in the folder SVGRN/scMultiSim_DataPrepare. All the experiments we provide in the paper follow these hyperparameters setting with only changing the "num.genes", "num.cells", and "intrinsic.noise".


### CeSpGRN configuration

In our benchmark experiments, we compared our model with CeSpGRN[^2]. 

- CeSpGRN source code can be downloaded from their github repository:
https://github.com/PeterZZQ/CeSpGRN

- We set the hyperparameters as following for running CeSpGRN:

```
bandwidth = 0.1
n_neigh = 30
lamb = 0.1
max_iters = 1000
```

### Reference
[^1] Li, Hechen, et al. "scMultiSim: simulation of multi-modality single cell data guided by cell-cell interactions and gene regulatory networks." Research Square (2023): rs-3.

[^2] Zhang, Ziqi, et al. "CeSpGRN: Inferring cell-specific gene regulatory networks from single cell multi-omics and spatial data." bioRxiv (2022): 2022-03.