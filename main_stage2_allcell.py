import argparse
import os
import numpy as np
import sys
import json
import time

from src.SVGRN_singlecell import SC_GRN_model


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=120, help='Number of Epochs for training DeepSEM')
parser.add_argument('--setting', type=str, default='default', help='Determine whether or not to use the default hyper-parameter')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')
parser.add_argument('--alpha', type=float, default=100, help='The loss coefficient for L1 norm of W, which is same as \\alpha used in our paper.')
parser.add_argument('--beta', type=float, default=1, help='The loss coefficient for KL term (beta-VAE), which is same as \\beta used in our paper.')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of used for RMSprop.')
parser.add_argument('--lr_step_size', type=int, default=0.99, help='The step size of learning rate decay.')
parser.add_argument('--gamma', type=float, default=0.95, help='The decay factor of learning rate')
parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
parser.add_argument('--K', type=int, default=1, help='Number of Gaussian kernel in GMM, default =1')
parser.add_argument('--K1', type=int, default=1, help='The Number of epoch for optimize MLP. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
parser.add_argument('--K2', type=int, default=2, help='The Number of epoch for optimize W. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
parser.add_argument('--W', type=int, default=5, help='weight for loss')


############## in and out file path/name ###############
parser.add_argument('--save_path', type=str, default='/tmp', help='path to store all the single cell GRN')
parser.add_argument('--save_name', type=str, default='', help='folder in save path to save single cell GRN')
parser.add_argument('--data_file', type=str, help='The input scRNA-seq gene expression file.')
parser.add_argument('--net_file', type=str, default='',
                    help='The ground truth of GRN. Only used in GRN inference task if available. ')
parser.add_argument('--net_path', type=str, default='',
                    help='The folder path of the GT GRN.')                    

####### params for single cell training (stage 2) ########
parser.add_argument('--model_file', type=str, default='', help='The loaded stage 1 model path')
parser.add_argument('--target_cell_name', type=str, default='', help='The target cell name for GRN in stage 2')

####### if use GPU ################
parser.add_argument('--GPU', action='store_true', help='Use GPU or not')
parser.add_argument('--device', type=str, default='', help='cpu or gpu')

###### load cell name .npy #######
parser.add_argument('--cellname_list', type=str, default='', help='cell name ')

######### if need dropout mask for sparse data #############
parser.add_argument('--dropout_mask', action='store_true', help='if need dropout mask')

opt = parser.parse_args()

try:
    os.mkdir(os.path.dirname(opt.save_path))
except:
    print(f'{os.path.dirname(opt.save_path)} exist')

try:
    os.mkdir(opt.save_path)
except:
    print(f'{opt.save_path} exist')

if opt.setting == 'default':
    opt.n_epochs = 150   #120 150
    opt.n_hidden = 128
    opt.gamma = 0.95
    opt.lr_step_size = 0.99
    opt.batch_size = 64 

with open(os.path.join(opt.save_path, 'args.txt'), 'a') as f:
        json.dump(opt.__dict__, f, indent=2)

cell_name_list = np.load(opt.cellname_list)

start_time = time.time()

for cellname in cell_name_list:
    # current GT net file and current target cell name
    opt.net_file = os.path.join(opt.net_path, f"{cellname}_gt_gene_pairs.csv")
    opt.target_cell_name = cellname
    print(cellname, opt.net_file)
    opt.save_name = os.path.join(opt.save_path, cellname)

    try:
        os.mkdir(opt.save_name)
        print(f'Create {opt.save_name}')
    except:
        print(f'{opt.save_name} exist')

    model = SC_GRN_model(opt)
    model.train_model()

end_time = time.time()

with open(os.path.join(opt.save_path, 'args.txt'), 'a') as f:
    f.write(f"\n Run Time: {end_time-start_time} s\n")