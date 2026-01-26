#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=00:20:00
#SBATCH --job-name=stage1
#SBATCH --account=bchx-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=/projects/bcod/yurui/SVGRN_code/SVGRN/log/%x_%j.out
#SBATCH --error=/projects/bcod/yurui/SVGRN_code/SVGRN/log/%x_%j.err

module load python/anaconda3/
source activate base

python main_stage1.py \
    --task simulation_allcell_GRN \
    --setting default \
    --data_file in_sim/example_data/expression_loc_cluster_wlayout.csv \
    --tf_list in_sim/example_data/TF_list.txt \
    --save_name out_stage1/exampledata_train_gpu_b5_a100_ngt \
    --beta 5 \
    --alpha 100 \
    --GPU

# python main_stage1.py \
#     --task simulation_allcell_GRN \
#     --setting default \
#     --data_file in_sim/example_data/expression_loc_cluster_wlayout.csv \
#     --tf_list in_sim/example_data/TF_list.txt \
#     --net_file in_sim/example_data/allcells_GRN.csv \
#     --save_name out_stage1/exampledata_train_gpu_b5_a100 \
#     --beta 5 \
#     --alpha 100 \
#     --GPU

