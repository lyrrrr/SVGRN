import locCSN
import os
import pandas as pd
import numpy as np
import time
from scipy.sparse import issparse

# ----------------------------
# Inputs
# ----------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True,
                    help="Simulation dataset name")
args = parser.parse_args()

data_name = args.data_name
expr_csv = f"in_sim/{data_name}/normalized_count.csv"          # rows=cells, cols=genes, header=gene names, index=cell names
meta_csv = f"in_sim/{data_name}/cell_loc_GRNcluster4.csv"     # must contain cell_type column; also must align by cell names

out_path = f"benchmark/locCSN/output_svgrn/{data_name}/"
os.makedirs(out_path, exist_ok=True)
out_npy = "locCSN_cellGRN.npy" # (n_cells, n_genes, n_genes)

# ----------------------------
# Load data
# ----------------------------
expr = pd.read_csv(expr_csv, index_col=0)          # cells x genes
meta = pd.read_csv(meta_csv, index_col=0)          # rows indexed by cell names

# Align metadata to expression order
meta = meta.loc[expr.index]
assert "ClusterID" in meta.columns, "meta_csv must include a column named 'ClusterID'"

cells = expr.index.to_numpy()
genes = expr.columns.to_numpy()
n_cells, n_genes = expr.shape

print(f"Expr shape: {expr.shape} (cells x genes)")
print(f"Cell types: {meta['ClusterID'].nunique()}")

# ----------------------------
# Allocate output tensor
# ----------------------------
grn_3d = np.zeros((n_cells, n_genes, n_genes), dtype=np.float32)

# ----------------------------
# Run locCSN per cell type
# ----------------------------
for ct, ct_cells in meta.groupby("ClusterID").groups.items():
    ct_cells = list(ct_cells)  # list of cell names in this type
    idx = expr.index.get_indexer(ct_cells)  # positions in global order
    idx = idx[idx >= 0]
    if len(idx) == 0:
        continue

    # Subset: cells x genes → convert to genes x cells for locCSN
    X_ct = expr.iloc[idx, :].to_numpy(dtype=np.float32).T  # genes x cells

    print(f"\n[{ct}] cells={X_ct.shape[1]} genes={X_ct.shape[0]}")
    start = time.time()
    csn_list = locCSN.csn(X_ct, dev=True, ncore = 10)  
    print(f"[{ct}] locCSN time: {time.time() - start:.2f}s")

    if len(csn_list) != X_ct.shape[1]:
        raise RuntimeError(f"[{ct}] Expected {X_ct.shape[1]} networks, got {len(csn_list)}")

    # Place back into global tensor (keep original cell order)
    # csn_list[j] is (genes x genes) for the j-th cell in X_ct column order
    for j, global_i in enumerate(idx):
        Aj = csn_list[j]

        if issparse(Aj):
            A = Aj.toarray().astype(np.float32)
        else:
            A = np.asarray(Aj, dtype=np.float32)
        # safety checks
        if A.shape != (n_genes, n_genes):
            raise RuntimeError(f"[{ct}] Network shape mismatch: {A.shape} vs {(n_genes, n_genes)}")
        np.fill_diagonal(A, 0.0)
        grn_3d[global_i] = A

# ----------------------------
# Save outputs
# ----------------------------
print(f"Shape of first cell GRN: {grn_3d[0].shape}")
print(f"Sample values (first cell GRN):\n{grn_3d[0][:5, :5]}")

np.save(os.path.join(out_path, out_npy), grn_3d)

print(f"\nSaved: {out_npy} with shape {grn_3d.shape}")

