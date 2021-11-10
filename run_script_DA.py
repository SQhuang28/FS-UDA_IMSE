from experiment_manage import ExperimentServer
from main import main
import argparse
import make_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1')
parser.add_argument('--metric', default='MELDA')
p_args = parser.parse_args()

parameters_dir = {'metric': [p_args.metric], 'data_name': ['visda'], 'backbone': ['ResNet12'],
                  'shot_num': [1],
                  # 'episode_train_num':[4],
                  'interim': [1],
                  'epochs': [10],

                  'source_domain': ['sketch', 'real', 'quickdraw', 'painting', 'clipart'],
                  'target_domain': ['sketch', 'real', 'quickdraw', 'painting', 'clipart'],
                  # 'source_domain': ['photo', 'sketch'],
                  # 'target_domain': ['photo', 'sketch'],
                  # 'source_domain': ['real', 'sketch'],
                  # 'target_domain': ['real', 'sketch'],
                  # 'source_domain': ['Product', 'Clipart'],
                  # 'target_domain': ['Product', 'Clipart'],
                  # 'source_domain': ['clipart', 'real'],
                  # 'target_domain': ['clipart', 'real'],
                  'loss_weight': [0.95],
                  'soft_weight': [0.5],
                  'softlabel': [True],
                  'cov_align': [True],
                  'cov_weight': [200],
                  'sploss_weight': [5, 50],
                  'DA': [True],
                  'temperature': [1],
                  # 'cluster_num':[64],
                  # 'cluster':[False],
                  'SGD': [False],
                  # 'beta':[0.5],
                  'iter': [1],
                  'lr': [0.0001],
                  'ilr': [0.001],
                  'discri': ['LD'],
                  'kernel_size': [1],
                  'sigma': [0.8],
                  'neighbor_k': [3],
                  'ld_num': [100],
                  'pretrained': [True],
                  # for DSN
                  'on_bn': [False],
                  # 'mask_entropy_weight':[2],
                  # 'template_num': [5],
                  # 'mask_info_weight': [0.5],
                  # 'mask_loss_func': ["mask_mining_lossV6P"],
                  # 'mask_margin': [0.2],
                  'workers':[12],
                  # 'episodeSize':[1],
                  }
exp_server = ExperimentServer(parameters_dir, main)
exp_server.opt.gpu = p_args.gpu
exp_server.run()
