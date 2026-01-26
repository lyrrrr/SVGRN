import sys, os
from os.path import exists

import numpy as np
import pandas as pd
import time
import argparse

import matplotlib.pyplot as plt

import g_admm as CeSpGRN
import kernel
import warnings

from sklearn.decomposition import PCA

plt.rcParams["font.size"] = 20

def preprocess(counts): 
    """\
    Input:
    counts = (ntimes, ngenes)
    
    Description:
    ------------
    Preprocess the dataset
    """
    # normalize according to the library size
    
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
        
    counts = np.log1p(counts)
    return counts

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default="", help='name of the data file to use')
parser.add_argument('--rawcount_path', type=str, default="", help='file of raw count data')
parser.add_argument('--pos_path', type=str, default="", help='file of cell pos')
parser.add_argument('--save_file_name', type=str, default="", help='save file name')
parser.add_argument('--save_folder_path', type=str, default="./results", help='result save folder path')
opt = parser.parse_args()
data_filename = opt.data_file
rawcount_path = opt.rawcount_path
pos_path = opt.pos_path
save_file_name = opt.save_file_name
save_folder_path = opt.save_folder_path
print(f"{rawcount_path} \n {pos_path}",flush=True)

start_time = time.time()

cell_pos = pd.read_csv(pos_path, index_col = 0)[["x", "y"]]
print(cell_pos.head(5),flush=True)
cell_pos = cell_pos.values

counts = pd.read_csv(rawcount_path, index_col = 0).values
# annotation = pd.read_csv(path + "anno.csv", index_col = 0)
ncells, ngenes = counts.shape

print("Raw TimePoints: {}, no.Genes: {}".format(counts.shape[0],counts.shape[1]),flush=True)

libsize = np.median(np.sum(counts, axis = 1))
counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
# the distribution of the original count is log-normal distribution, conduct log transform
counts = np.log1p(counts)

# Estimate cell-specific GRNs
# hyper-parameters
bandwidth = 0.1
n_neigh = 30
lamb = 0.1
max_iters = 1000

# calculate the kernel function
K, K_trun = kernel.calc_kernel_distance(cell_pos, bandwidth = bandwidth, truncate = True, truncate_param = n_neigh)
print("number of neighbor being considered: " + str(np.sum(K_trun[int(ncells/2), :] > 0)),flush=True)

# estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)

print("After est_cov",flush=True)

# estimate cell-specific GRNs
cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
thetas = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)

os.makedirs(f"{save_folder_path}/{data_filename}", exist_ok=True)
np.save(file = f"{save_folder_path}/{data_filename}/{save_file_name}__" + str(bandwidth) + "_" + str(lamb) + "_" + str(n_neigh) + ".npy", arr = thetas)

print("time calculating thetas: {:.2f} sec".format(time.time() - start_time),flush=True)