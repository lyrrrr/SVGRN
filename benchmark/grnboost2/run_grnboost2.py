import pandas as pd
from arboreto.algo import grnboost2
import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True,
                        help="Simulation dataset name")
    args = parser.parse_args()

    data_name = args.data_name
    data_folder = "in_sim"
    save_folder = "benchmark/grnboost2/output_svgrn"

    # Load expression matrix (cells × genes)
    expr_df = pd.read_csv(f"{data_folder}/{data_name}/normalized_count.csv", index_col=0)  # rows = cells, cols = genes
    print("Expression matrix shape (cells x genes):", expr_df.shape)
    expr_df.columns = expr_df.columns.astype(str)

    #######################
    # tf list for gene 110
    # tf_list = ["2","6","10","19","80","91"]
    # tf list for gene 160
    tf_list = ["5", "8", "14", "18", "25", "32", "57", "62", "147"]
    #######################

    # Run GRNBoost2
    regulons_df = grnboost2(
        expression_data=expr_df,
        tf_names=tf_list,
        verbose=True
    ) # n_jobs=8  # adjust threads

    print("Inferred edges:", regulons_df.shape)
    print(regulons_df.head())

    save_path = os.path.join(save_folder, data_name)
    os.makedirs(save_path, exist_ok=True)

    # Save results
    regulons_df.to_csv(os.path.join(save_path, "grnboost2_edges.csv"), index=False)
    print("Saved GRN edges to grnboost2_edges.csv")
