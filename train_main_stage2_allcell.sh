python main_stage2_allcell.py \
    --setting default \
    --data_file in_sim/example_data/expression_loc_cluster_wlayout.csv \
    --model_file out_stage1/exampledata_train_gpu_b5_a100/stage1.pt \
    --save_path out_stage2/example_data/c10_w6_b5_a100 \
    --cellname_list in_sim/example_data/cellname_list_10.npy \
    --W 6 \
    --beta 5 \
    --alpha 100 \
    --GPU

# python main_stage2_allcell.py \
#     --setting default \
#     --data_file in_sim/example_data/expression_loc_cluster_wlayout.csv \
#     --net_path in_sim/example_data/cell_specific_GRN \
#     --model_file out_stage1/exampledata_train_gpu_b5_a100/stage1.pt \
#     --save_path out_stage2/example_data/c10_w6_b5_a100 \
#     --cellname_list in_sim/example_data/cellname_list_10.npy \
#     --W 6 \
#     --beta 5 \
#     --alpha 100 \
#     --GPU
