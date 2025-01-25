# SVGRN

### Environment setup

Use the environment.yml to create the anaconda virtual environment for SVGRN:

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