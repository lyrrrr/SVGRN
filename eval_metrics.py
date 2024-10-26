import numpy as np
import pandas as pd
import copy
import math
import os
from sklearn.metrics import average_precision_score


def get_data(GT_path, Adj_path):
    Ground_Truth = pd.read_csv(GT_path, header=0)
    Ground_Truth = Ground_Truth.astype(str)      # change all the value to string (gene names)
    TF = set(Ground_Truth['gene2'])   # "gene2" columns are TF genes
    All_gene = set(Ground_Truth['gene1']) | set(Ground_Truth['gene2'])

    Adj_matrix = pd.read_csv(Adj_path)
    geneNames = [str(i) for i in list(Adj_matrix.columns)]
    Adj_matrix.set_index(Adj_matrix.columns, inplace=True)
    print(Adj_matrix.head(5))
    num_genes = Adj_matrix.shape[0]

    Evaluate_Mask = np.zeros([num_genes, num_genes])
    TF_mask = np.zeros([num_genes, num_genes])
    for i, item in enumerate(Adj_matrix.columns):
        for j, item2 in enumerate(Adj_matrix.columns):
            if i == j:
                continue
            if item2 in TF and item in All_gene:
                Evaluate_Mask[i, j] = 1
            if item2 in TF:
                TF_mask[i, j] = 1
    
    truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=Adj_matrix.columns, columns=Adj_matrix.columns)
    for i in range(Ground_Truth.shape[0]):
        truth_df.loc[Ground_Truth.iloc[i, 0], Ground_Truth.iloc[i, 1]] = 1
    A_truth = truth_df.values
    idx_rec, idx_send = np.where(A_truth)
    # print(f"idx_rec {len(set(idx_rec))}, idx_send {len(set(idx_send))}")
    truth_edges = set(zip(idx_send, idx_rec))

    return Adj_matrix, Evaluate_Mask, truth_edges, TF_mask, geneNames


def evaluate(A, truth_edges, Evaluate_Mask):
    A = A.values
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)

    # print("num_truth_edges: ", num_truth_edges)

    A= abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
    overlap_A = A_edges.intersection(truth_edges)

    # true positive value, EP rate, EPR
    return len(overlap_A), 1. * len(overlap_A) / num_truth_edges, 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))


def extractEdgesFromMatrix(m, geneNames,TFmask):
    geneNames = np.array(geneNames)
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        mat = mat*TFmask
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'TF': geneNames[idx_send], 'Target': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)

    return edges_df

#############################################################################
# the median AUPRC ratio (the AUPRC divided by that of a random predictor). #
#############################################################################

def get_AUPRC(output, GT_path):
    output['EdgeWeight'] = abs(output['EdgeWeight'])
    output = output.sort_values('EdgeWeight',ascending=False)
    label = pd.read_csv(GT_path)
    label = label.astype(str)   # readin all the number is int, so need to change to string
    TFs = set(label['gene2'])
    # Genes = set(label['gene1']) | set(label['gene2'])
    Genes = set(output['Target']) | set(output['TF'])
    print(f"Gene number: {len(Genes)}")

    output = output[output['TF'].apply(lambda x: x in TFs)]

    # print(len(output))
    label_set = set(label['gene2']+label['gene1'])
    preds,labels,randoms = [] ,[],[]
    res_d = {}
    l = []
    p= []
    for item in (output.to_dict('records')):
        res_d[item['TF']+item['Target']] = item['EdgeWeight']
    for item in TFs:
        for item2 in Genes:
            if item+item2 in label_set:
                l.append(1)
            else:
                l.append(0)
            if item+item2 in res_d:
                p.append(res_d[item+item2])
            else:
                p.append(0)
    # print(l)
    # print(p)
    # print(sum(l))
    # print(len(l))
    # return average_precision_score(l,p)/np.mean(l)
    return average_precision_score(l,p), average_precision_score(l,p)/np.mean(l)


if __name__ == "__main__":

    cellname_list = ['cell490', 'cell116', 'cell998', 'cell1674', 'cell706', 'cell1286', 'cell527', 'cell1454', 'cell459', 'cell1258']
    # gene_exp = pd.read_csv("in_sim/g110_c2k_0.1/expression_loc_cluster.csv", index_col=[0])
    # cellname_list = list(gene_exp.index)
    # print(len(cellname_list))

    result_list = ['c10_w6_5_100']

    GT_folder = "in_sim/g110_c2k_0.1/cell_specific_gt_gene_pair"
    result_folder = "out_stage2/g110_c2k_0.1_subset"

    for r in result_list:
        print(r)
        EP_rate_list = []
        EPR_list = []
        auprc_list = []
        auprc_ratio_list = []

        for c in cellname_list:
            print(c)
            GT_path = os.path.join(GT_folder, f"{c}_gt_gene_pairs.csv")
            Adj_path = os.path.join(result_folder, r, c, "RN_150.csv")

            Adj_matrix, Evaluate_Mask, truth_edges, TF_mask, geneNames = get_data(GT_path, Adj_path)

            # get EP
            TP_num, EP_rate, EPR = evaluate(Adj_matrix, truth_edges, TF_mask)
            print("TP_num, EP_rate, EPR: ", TP_num, EP_rate, EPR)
            EP_rate_list.append(EP_rate)
            EPR_list.append(EPR)

            # get AUPRC
            edges_df = extractEdgesFromMatrix(Adj_matrix.values, geneNames, TF_mask)
            auprc, auprc_ratio = get_AUPRC(edges_df, GT_path)
            print("AUPRC, AUPRC_ratio: ", auprc, auprc_ratio)
            auprc_list.append(auprc)
            auprc_ratio_list.append(auprc_ratio)
        
        print(f"EP_rate_list {len(EP_rate_list)}")
        ave_EP_rate = sum(EP_rate_list) / len(EP_rate_list)
        ave_EPR = sum(EPR_list) / len(EPR_list) 
        ave_auprc = sum(auprc_list) / len(auprc_list) 
        ave_auprc_ratio = sum(auprc_ratio_list) / len(auprc_ratio_list) 

        print("ave_EP_rate, ave_EPR, ave_auprc, ave_auprc_ratio", ave_EP_rate, ave_EPR, ave_auprc, ave_auprc_ratio)

        with open(os.path.join(result_folder, r, "eval.txt"), "a") as text_file:
            text_file.write(f"ave_EP_rate: {ave_EP_rate}, ave_EPR: {ave_EPR}\n")
            text_file.write(f"ave_AUPRC: {ave_auprc}, ave_AUPRC_ratio: {ave_auprc_ratio}\n")
