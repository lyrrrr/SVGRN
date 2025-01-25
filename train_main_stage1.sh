python main_stage1.py \
    --task simulation_allcell_GRN \
    --setting default \
    --data_file in_sim/g110_c2k_0.1/expression_loc_cluster.csv \
    --net_file in_sim/g110_c2k_0.1/allcells_gt_gene_pairs.csv \
    --save_name out_stage1/g110_c2k_0.1_train_gpu_b4_a100 \
    --beta 5 \
    --alpha 100 \
    --GPU

