import os
import sys
import time

import numpy as np
import pandas as pd
# import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.Con_Model_newED import CVAE_EAD_newED
from src.utils import evaluate, extractEdgesFromMatrix, RBF_weights

Tensor = torch.cuda.FloatTensor

def get_adaptive_sigma(pos_df, target_cell_name, k=10, min_sigma=1e-8):
    """
    Set sigma for one target cell as its distance to the kth nearest neighbor.
    pos_df: dataframe with columns ['x', 'y'], index = cell names
    target_cell_name: target cell index/name
    k: kth nearest neighbor, excluding itself
    """
    coords = pos_df[["x", "y"]].values.astype(float)
    target_coord = pos_df.loc[target_cell_name, ["x", "y"]].values.astype(float)

    dists = np.sqrt(((coords - target_coord) ** 2).sum(axis=1))

    # remove self-distance
    dists = dists[dists > 0]

    if len(dists) == 0:
        return min_sigma

    k_eff = min(k, len(dists))

    sigma = np.partition(dists, k_eff - 1)[k_eff - 1]

    sigma = max(float(sigma), min_sigma)

    return sigma


class SC_GRN_model:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def init_data(self):

        All_Data = pd.read_csv(self.opt.data_file, index_col=[0])
        
        pos_df = All_Data[['x','y']]
        
        # calculate weight for loss based on its distance to target cell
        target_pos = pos_df.loc[self.opt.target_cell_name]

        # adaptive RBF bandwidth
        if getattr(self.opt, "adaptive_W", False):
            print("use adaptive W")
            sigma = get_adaptive_sigma(
                pos_df,
                self.opt.target_cell_name,
                k=self.opt.k_neighbor
            )
            print(f"RBF sigma for target cell {self.opt.target_cell_name}: {sigma}")
        else:
            sigma = self.opt.W

        weight_df = pd.DataFrame(columns=['weight'])
        weight_df['weight'] = pos_df.apply(lambda row: RBF_weights(row, target_pos['x'], target_pos['y'], sigma), axis=1)
        # print(f"dist weight: {weight_df.head(5)}")
        total_weight = weight_df['weight'].sum()
        # print(f"sum : {total_weight}")

        # scale the sum of weight to sample number (multiply first to avoid number become so small)
        weight_df['weight'] = (weight_df['weight'] * pos_df.shape[0]) / total_weight
        # print(f"rescaled weight : {weight_df.head(5)}")
        print(f"weight_df describe: {weight_df.describe()}")
        print(f"weight_df num: ", (weight_df['weight'] >= 1).sum())
        print(f"sum now: {weight_df['weight'].sum()}")

        # change all the pos Y to the target pos
        # normalize position x, y
        pos_df = (pos_df-pos_df.min(0))/(pos_df.max(0)-pos_df.min(0))
        target_pos = pos_df.loc[self.opt.target_cell_name]
        target_pos = np.array([[target_pos['x'], target_pos['y']]])
        print(f"target_pos: {target_pos}")
        pos_df = pd.DataFrame(np.repeat(target_pos, pos_df.shape[0], axis=0), 
                    columns=pos_df.columns, index=pos_df.index)
        
        data = All_Data.drop(columns=['x', 'y', 'ClusterID'], errors='ignore')
        gene_name = list(data.columns)     # gene column names are all string
        data_values = data.values
        Dropout_Mask = (data_values != 0).astype(float)
        
        num_genes, num_nodes = data.shape[1], data.shape[0]
        print(f"num_genes {num_genes}, num_nodes {num_nodes}")

        feat_train = torch.FloatTensor(data.values)
        pos_train = torch.FloatTensor(pos_df.values)
        weight_train = torch.FloatTensor(weight_df["weight"].values)

        # add the spatial (x,y) here as pos_train
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask), pos_train, weight_train)

        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)

        if self.opt.net_file is None:
            truth_edges, TF_mask = None, None
        
        else:
            Ground_Truth = pd.read_csv(self.opt.net_file, index_col=0)
            Ground_Truth.index = Ground_Truth.index.astype(str)
            Ground_Truth.columns = Ground_Truth.columns.astype(str)
            TF = set(Ground_Truth.columns)
            All_gene = set(Ground_Truth.index)
            print(f"TF {TF}")
            print(f"TF num {len(TF)}, All_gene num {len(All_gene)}")

            TF_mask = np.zeros([num_genes, num_genes])
            for i, item in enumerate(data.columns):
                for j, item2 in enumerate(data.columns):
                    if i == j:
                        continue
                    if item2 in TF:
                        TF_mask[i, j] = 1

            nonzero_indices = np.where(Ground_Truth.values != 0)
            truth_edges = [(int(Ground_Truth.columns[col])-1, int(Ground_Truth.index[row])-1) for row, col in zip(*nonzero_indices)]
            truth_edges = set(truth_edges)   # idx_send, idx_rec

        return dataloader, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name

    def train_model(self):
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.opt.device)

        opt = self.opt
        if opt.GPU:
            print("Using GPU.....")
        else:
            print("Not using GPU....")
        
        dataloader, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name = self.init_data()
        
        # y_pos_dim = 128
        if opt.GPU:
            cvae = torch.load(opt.model_file).to(opt.device)    # load stage1 model
        else:
            cvae = torch.load(opt.model_file)    # load stage1 model
        print("model loaded")

        optimizer2 = optim.RMSprop([cvae.adj_A], lr=opt.lr * 0.2) # only update
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_Epr = 0
        cvae.train()

        # freeze all other params, only update adj_A
        for param in cvae.parameters():
            param.requires_grad = False
        cvae.adj_A.requires_grad = True

        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            print(f"Epoch: {epoch}")
            
            for i, data_batch in enumerate(dataloader, 0):
                # print(f"epoch: {epoch}, iter: {i}")
                optimizer2.zero_grad()

                # add Y_pos = (x,y) as the corresponding pos for each cell input
                inputs, data_id, dropout_mask, Y_pos, weight = data_batch

                if opt.GPU:
                    inputs = inputs.to(opt.device)
                    Y_pos = Y_pos.to(opt.device)
                    weight = weight.to(opt.device)

                data_ids.append(data_id.numpy())
                temperature = max(0.95 ** epoch, 0.5)

                if opt.dropout_mask:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, weight, dropout_mask=dropout_mask.to(opt.device),
                                                                            temperature=temperature, opt=opt)

                else:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, weight, dropout_mask=None,
                                                                            temperature=temperature, opt=opt)

                sparse_loss = opt.alpha * torch.mean(torch.abs(cvae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_KL.item())
                loss_sparse.append(sparse_loss.item())
                optimizer2.step()  

            scheduler.step()

            if truth_edges is not None and TFmask2 is not None:
                if opt.GPU:
                    Ep, Epr = evaluate(cvae.adj_A.cpu().detach().numpy(), truth_edges, TFmask2)
                else:
                    Ep, Epr = evaluate(cvae.adj_A.detach().numpy(), truth_edges, TFmask2)
                
                best_Epr = max(Epr, best_Epr)
                print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                        np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                        np.mean(loss_sparse))

                # with open(opt.save_name + '/log1.txt', 'a') as f:
                #     # Write the text to the file
                #     f.write(f"Epoch: {epoch}, Ep: {Ep}, Epr: {Epr}, loss: {np.mean(loss_all)} " +
                #         f"mse_loss: {np.mean(mse_rec)}, kl_loss: {np.mean(loss_kl)}, sparse_loss:{np.mean(loss_sparse)}\n")
            
            else:
                print('epoch:', epoch, 'loss:',
                    np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                    np.mean(loss_sparse))

                # with open(opt.save_name + '/log1.txt', 'a') as f:
                #     # Write the text to the file
                #     f.write(f"Epoch: {epoch}, loss: {np.mean(loss_all)} " +
                #         f"mse_loss: {np.mean(mse_rec)}, kl_loss: {np.mean(loss_kl)}, sparse_loss:{np.mean(loss_sparse)}\n")

        if opt.GPU:
            RN_df = pd.DataFrame(cvae.adj_A.cpu().detach().numpy(), columns=list(gene_name))
        else:
            RN_df = pd.DataFrame(cvae.adj_A.detach().numpy(), columns=list(gene_name))
        RN_df.to_csv(opt.save_name + f"/RN_{opt.n_epochs}.csv", index=False)

