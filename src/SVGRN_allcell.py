import os
import sys

import numpy as np
import pandas as pd
# import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.Con_Model_newED import CVAE_EAD_newED
from src.utils import evaluate, extractEdgesFromMatrix


class non_celltype_GRN_model:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def initalize_A(self, data):
        num_genes = data.shape[1]
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0
        return A

    def init_data(self):

        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        Ground_Truth = Ground_Truth.astype(str)      # change all the value to string (gene names)
        TF = set(Ground_Truth['gene2'])   # "gene2" columns are TF genes
        All_gene = set(Ground_Truth['gene1']) | set(Ground_Truth['gene2'])
        print(f"TF {TF}")
        print(f"TF num {len(TF)}, All_gene num {len(All_gene)}")
        #print(f"Ground_Truth data type {Ground_Truth.dtypes}")

        All_Data = pd.read_csv(self.opt.data_file, index_col=[0])
        
        #print(f"if all gene names are string: {all(type(n) == str for n in gene_name)}")
        pos_df = All_Data[['x','y']]
        # normalize position x, y
        pos_df=(pos_df-pos_df.min(0))/(pos_df.max(0)-pos_df.min(0))

        data = All_Data.drop(columns=['x', 'y','ClusterID'])
        gene_name = list(data.columns)     # gene column names are all string
        # print(data.head(5))
        data_values = data.values
        Dropout_Mask = (data_values != 0).astype(float)
        
        num_genes, num_nodes = data.shape[1], data.shape[0]
        print(f"num_genes {num_genes}, num_nodes {num_nodes}")
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:
                    TF_mask[i, j] = 1

        feat_train = torch.FloatTensor(data.values)
        pos_train = torch.FloatTensor(pos_df.values)

        # add the spatial (x,y) here as pos_train
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask), pos_train)

        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 0], Ground_Truth.iloc[i, 1]] = 1
        # print(truth_df.head(5))
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        print(f"idx_rec {len(set(idx_rec))}, idx_send {len(set(idx_send))}")
        truth_edges = set(zip(idx_send, idx_rec))

        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name

    def train_model(self):
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.opt.device)
        opt = self.opt

        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name = self.init_data()
        adj_A_init = self.initalize_A(data)

        y_pos_dim = 128

        if opt.GPU:
            cvae = CVAE_EAD_newED(adj_A_init, 1, opt.n_hidden, opt.K, y_pos_dim).float().cuda()
            print("load model")
        else:
            cvae = CVAE_EAD_newED(adj_A_init, 1, opt.n_hidden, opt.K, y_pos_dim).float()

        optimizer = optim.RMSprop(cvae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([cvae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_Epr = 0
        cvae.train()

        if opt.GPU:
            RN_df = pd.DataFrame(cvae.adj_A.cpu().detach().numpy(), columns=list(gene_name))
        else:
            RN_df = pd.DataFrame(cvae.adj_A.detach().numpy(), columns=list(gene_name))
        RN_df.to_csv(opt.save_name + '/initial_RN.csv', index=False)

        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            if epoch % (opt.K1 + opt.K2) < opt.K1:
                print(f"Epoch: {epoch} Not update adj_A")
                cvae.adj_A.requires_grad = False  # not update adj_A when epoch%3==0
            else:
                print(f"Epoch: {epoch} Only update adj_A")
                cvae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                # print(f"epoch: {epoch}, iter: {i}")
                optimizer.zero_grad()

                # add Y_pos = (x,y) as the corresponding pos for each cell input
                inputs, data_id, dropout_mask, Y_pos = data_batch

                if opt.GPU:
                    inputs = inputs.to(opt.device)
                    Y_pos = Y_pos.to(opt.device)
                # print(f"Y_pos is tensor: {torch.is_tensor(Y_pos)}")
                data_ids.append(data_id.numpy())
                #data_ids.append(data_id.cpu().detach().numpy())
                temperature = max(0.95 ** epoch, 0.5)

                if opt.dropout_mask:
                    print("opt.dropout_mask")
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, dropout_mask=dropout_mask.to(opt.device),
                                                                           temperature=temperature, opt=opt)
                else:
                    loss, loss_rec, loss_KL, dec, hidden = cvae(inputs, Y_pos, dropout_mask=None,
                                                                           temperature=temperature, opt=opt)

                sparse_loss = opt.alpha * torch.mean(torch.abs(cvae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_KL.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (opt.K1 + opt.K2) < opt.K1: # not update adj_A when epoch%3==0
                    optimizer.step()
                else:
                    optimizer2.step()  # only update adj_A
            scheduler.step()

            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                if opt.GPU:
                    Ep, Epr = evaluate(cvae.adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                else:
                    Ep, Epr = evaluate(cvae.adj_A.detach().numpy(), truth_edges, Evaluate_Mask)

                best_Epr = max(Epr, best_Epr)
                print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                      np.mean(loss_sparse))

                with open(opt.save_name + '/log1.txt', 'a') as f:
                    # Write the text to the file
                    f.write(f"Epoch: {epoch}, Ep: {Ep}, Epr: {Epr}, loss: {np.mean(loss_all)} " +
                      f"mse_loss: {np.mean(mse_rec)}, kl_loss: {np.mean(loss_kl)}, sparse_loss:{np.mean(loss_sparse)}\n")
                

        if opt.GPU:
            RN_df = pd.DataFrame(cvae.adj_A.cpu().detach().numpy(), columns=list(gene_name))
        else:
            RN_df = pd.DataFrame(cvae.adj_A.detach().numpy(), columns=list(gene_name))
        RN_df.to_csv(opt.save_name + f"/RN_{opt.n_epochs}.csv", index=False)
        
        with open(opt.save_name + '/log1.txt', 'a') as f:
            # Write the text to the file
            f.write(f"Best EPR: {best_Epr}\n")
        # save model
        torch.save(cvae, opt.save_name + "/stage1.pt")        

