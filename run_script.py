from experiment_manage import ExperimentServer
from main import main
import argparse
import make_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1')
parser.add_argument('--metric', default='MELDA')
p_args = parser.parse_args()

parameters_dir = {'metric': [p_args.metric], 'data_name': ['mini-Imagenet'], 'backbone': ['ResNet12'],
                  'shot_num': [1], 'epochs': [10],
                  'interim': [1],
                  'source_domain': ['photo'],
                  'target_domain': ['photo'],
                  'loss_weight': [1],
                  'soft_weight': [0.0],
                  'softlabel': [False],
                  'cov_align': [True],
                  'cov_weight': [150],
                  'DA': [False],
                  'temperature': [1],
                  # 'cluster_num':[64],
                  # 'cluster':[False],
                  'SGD': [False],
                  # 'beta':[0.5],
                  'iter': [1],
                  'lr': [0.0001],
                  'ilr': [0.001],
                  'sploss_weight': [1],
                  'discri': ['LD'],
                  'kernel_size': [1],
                  # 'template_num': [5],
                  'sigma': [0.8],
                  'neighbor_k': [3],
                  'ld_num': [100],
                  'pretrained': [True],
                  # for DSN
                  'on_bn': [False],
                  'bn_epoch': [10],
                  'workers':[24]
                  }



exp_server = ExperimentServer(parameters_dir, main)
exp_server.opt.gpu = p_args.gpu
exp_server.run()
