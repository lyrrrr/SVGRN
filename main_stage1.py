import argparse
import sys
import os
import json

from SVGRN_allcell import non_celltype_GRN_model
from SVGRN_singlecell import SC_GRN_model


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=120, help='Number of Epochs for training DeepSEM')
parser.add_argument('--task', type=str, default='celltype_GRN',
                    help='Determine which task to run. Select from (non_celltype_GRN,celltype_GRN,embedding,simulation)')
parser.add_argument('--setting', type=str, default='default', help='Determine whether or not to use the default hyper-parameter')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')
parser.add_argument('--data_file', type=str, help='The input scRNA-seq gene expression file.')
parser.add_argument('--net_file', type=str, default='',
                    help='The ground truth of GRN. Only used in GRN inference task if available. ')
parser.add_argument('--alpha', type=float, default=100, help='The loss coefficient for L1 norm of W, which is same as \\alpha used in our paper.')
parser.add_argument('--beta', type=float, default=1, help='The loss coefficient for KL term (beta-VAE), which is same as \\beta used in our paper.')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of used for RMSprop.')
parser.add_argument('--lr_step_size', type=int, default=0.99, help='The step size of learning rate decay.')
parser.add_argument('--gamma', type=float, default=0.95, help='The decay factor of learning rate')
parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
parser.add_argument('--K', type=int, default=1, help='Number of Gaussian kernel in GMM, default =1')
parser.add_argument('--K1', type=int, default=1, help='The Number of epoch for optimize MLP. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
parser.add_argument('--K2', type=int, default=2, help='The Number of epoch for optimize W. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
parser.add_argument('--save_name', type=str, default='/tmp')

####### params for single cell training (stage 2) ########
parser.add_argument('--model_file', type=str, default='', help='The loaded stage 1 model path')
parser.add_argument('--target_cell_name', type=str, default='', help='The target cell name for GRN in stage 2')

####### if use GPU ################
parser.add_argument('--GPU', type=bool, default=False, help='Use GPU or not')
parser.add_argument('--device', type=str, default='', help='cpu or gpu')

######### if need dropout mask for sparse data #############
parser.add_argument('--dropout_mask', type=bool, default=False, help='if need dropout mask')

opt = parser.parse_args()

if opt.task == 'simulation_allcell_GRN':
    if opt.setting == 'default':
        opt.n_epochs = 150   #120 150
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden = 128
        opt.gamma = 0.95
        opt.lr = 1e-4
        opt.lr_step_size = 0.99
        opt.batch_size = 64      #64
        # opt.GPU = True

    model = non_celltype_GRN_model(opt)

    with open(os.path.join(opt.save_name, 'args.txt'), 'a') as f:
        json.dump(opt.__dict__, f, indent=2)

    model.train_model()

elif opt.task == 'simulation_singlecell_GRN':
    print("simulation_singlecell_GRN")
    if opt.setting == 'default':
        opt.n_epochs = 150   #120
        opt.beta = 1
        opt.alpha = 100
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden = 128
        opt.gamma = 0.95
        opt.lr = 1e-4
        opt.lr_step_size = 0.99
        opt.batch_size = 64 
        opt.GPU = True
    model = SC_GRN_model(opt)
    model.train_model()

