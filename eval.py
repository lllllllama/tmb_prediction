from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# 训练设置
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
# 数据根目录
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
# 结果保存目录
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
# 评估结果保存的实验代码
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
# 加载训练模型的实验代码
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
# 数据分割目录
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
# 模型大小选择
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
# 模型类型选择
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
# 交叉验证折数
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
# 起始折
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
# 结束折
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
# 单独评估的折数
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
# 多分类AUC计算方式
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
# 评估数据集划分
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
# 任务类型
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
# Dropout比率
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
# 嵌入维度
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

# 设置设备(GPU/CPU)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置保存路径
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

# 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)

# 如果未指定splits_dir,使用models_dir
if args.splits_dir is None:
    args.splits_dir = args.models_dir

# 确保目录存在
assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

# 实验设置
settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

# 保存实验设置
with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

# 根据任务类型初始化数据集
if args.task == 'task_1_tumor_vs_normal':  # 如果任务为肿瘤与正常组织的二分类
    args.n_classes=2  # 设置类别数为2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/first.csv',  # 加载数据集
                            data_dir= os.path.join(args.data_root_dir, 'MERGED_FEATURES'),  # 数据目录
                            shuffle = False,  # 不打乱数据
                            # seed = args.seed,  # 设置随机种子
                            # print_info = True,  # 打印信息
                            label_dict = {'TMB-H':0, 'TMB-L':1},  # 标签字典
                            patient_strat=False,  # 不进行病人层面分层
                            ignore=[])  # 忽略列表为空

elif args.task == 'task_2_tumor_subtyping':
    # 三分类任务:肿瘤亚型分类
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

# 设置评估的fold范围
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

# 获取每个fold的模型检查点路径
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
# 数据集划分的索引字典
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    # 存储所有fold的评估结果
    all_results = []
    all_auc = []
    all_acc = []
    
    # 对每个fold进行评估
    for ckpt_idx in range(len(ckpt_paths)):
        # 根据split参数选择评估数据集
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
            
        # 评估当前fold
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        # 保存当前fold的详细结果
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    # 汇总所有fold的结果
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    # 根据评估的fold范围确定保存文件名
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    # 保存汇总结果
    final_df.to_csv(os.path.join(args.save_dir, save_name))
