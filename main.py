from __future__ import print_function  # 导入未来版本的print函数

# 导入所需的库
import argparse  # 用于命令行参数解析
import pdb  # 用于调试
import os  # 用于文件和目录操作
import math  # 数学运算

# 导入内部模块
from utils.file_utils import save_pkl, load_pkl  # 用于保存和加载pkl文件
from utils.utils import *  # 导入通用工具函数
from utils.core_utils import train  # 导入训练相关函数
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset  # 导入数据集类

# 导入PyTorch相关库
import torch  # 导入PyTorch深度学习框架
from torch.utils.data import DataLoader, sampler  # 导入数据加载器和采样器
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数库

# 导入数据处理库
import pandas as pd  # 导入数据分析库
import numpy as np  # 导入数值计算库


def main(args):
    # 如果结果目录不存在则创建
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)  # 创建结果目录

    # 设置k折交叉验证的起始和结束fold
    if args.k_start == -1:  # 如果起始fold为-1
        start = 0  # 从第0个fold开始
    else:
        start = args.k_start  # 否则使用指定的起始fold
    if args.k_end == -1:  # 如果结束fold为-1
        end = args.k  # 结束于总的fold数量
    else:
        end = args.k_end  # 否则使用指定的结束fold

    # 存储所有fold的评估指标
    all_test_auc = []  # 存储测试集AUC
    all_val_auc = []   # 存储验证集AUC
    all_test_acc = []  # 存储测试集准确率
    all_val_acc = []   # 存储验证集准确率
    folds = np.arange(start, end)  # 生成fold序列
    
    # 对每个fold进行训练和评估
    for i in folds:
        seed_torch(args.seed)  # 设置随机种子确保可重复性
        # 获取训练集、验证集和测试集
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))  # 从指定路径加载数据集
        
        datasets = (train_dataset, val_dataset, test_dataset)  # 将数据集打包成元组
        print("这是test数据集")  # 打印提示信息
        print(datasets)  # 打印数据集内容
        # 训练模型并获取结果
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)  # 训练模型并获取评估指标
        # 保存评估指标
        all_test_auc.append(test_auc)  # 添加测试集AUC到列表
        all_val_auc.append(val_auc)  # 添加验证集AUC到列表
        all_test_acc.append(test_acc)  # 添加测试集准确率到列表
        all_val_acc.append(val_acc)  # 添加验证集准确率到列表
        # 将结果保存为pkl文件
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))  # 生成文件名
        save_pkl(filename, results)  # 保存结果

    # 创建包含所有fold结果的DataFrame
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})  # 创建结果DataFrame

    # 根据处理的fold数量确定保存文件名
    if len(folds) != args.k:  # 如果处理的fold数量不等于总的fold数量
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)  # 生成部分结果文件名
    else:
        save_name = 'summary.csv'  # 生成完整结果文件名
    # 保存最终结果到CSV文件
    final_df.to_csv(os.path.join(args.results_dir, save_name))  # 保存结果到CSV文件

# 设置命令行参数
parser = argparse.ArgumentParser(description='Configurations for WSI Training')  # 创建参数解析器
# 数据相关参数
parser.add_argument('--data_root_dir', type=str, default='./', 
                    help='data directory')  # 数据目录参数
parser.add_argument('--embed_dim', type=int, default=1024)  # 嵌入维度参数
# 训练相关参数
parser.add_argument('--max_epochs', type=int, default=60,
                    help='maximum number of epochs to train (default: 200)')  # 最大训练轮数参数
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate (default: 0.0001)')  # 学习率参数
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')  # 训练标签比例参数
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')  # 权重衰减参数
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')  # 随机种子参数
# k折交叉验证相关参数
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')  # k折数量参数
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')  # 起始fold参数
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')  # 结束fold参数
# 输出和日志相关参数
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')  # 结果目录参数
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')  # 数据集划分目录参数
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')  # 日志记录参数
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')  # 调试工具参数
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')  # 提前停止参数
# 优化器和模型相关参数
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')  # 优化器选择参数
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')  # dropout参数
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'weighted_ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')  # 包损失函数参数
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')  # 模型类型参数
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')  # 实验代码参数
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')  # 加权采样参数
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')  # 模型大小参数
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])  # 任务类型参数
# CLAM特定参数
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')  # 禁用实例级聚类参数
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'weighted_ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')  # 实例级损失函数参数
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')  # 亚型问题参数
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')  # 包级损失权重参数
parser.add_argument('--B', type=int, default=16, help='numbr of positive/negative patches to sample for clam')  # 采样正负样本的数量参数

# 文本描述相关参数
parser.add_argument('--use_text_desc', action='store_true', default=True,
                    help='whether to use text descriptions')  # 是否使用文本描述
parser.add_argument('--text_desc_col', type=str, default='optimized_description',
                    help='column name for text descriptions')  # 文本描述列名
parser.add_argument('--fusion_method', type=str, choices=['concat', 'add', 'attention'], default='attention',
                    help='method to fuse image and text features')  # 特征融合方法
parser.add_argument('--clip_model_name', type=str, default="ViT-B/32",
                    help='CLIP model name for text feature extraction')  # CLIP模型名称
args = parser.parse_args()  # 解析命令行参数

# 设置计算设备(GPU/CPU)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测可用的计算设备

# 设置随机种子函数
def seed_torch(seed=7):  # 定义设置随机种子的函数
    import random  # 导入随机库
    random.seed(seed)  # 设置Python随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境变量以确保可重复性
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    if device.type == 'cuda':  # 如果使用GPU
        torch.cuda.manual_seed(seed)  # 设置CUDA随机种子
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU，设置所有GPU的随机种子
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的基准模式
    torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性模式

# 设置随机种子
seed_torch(args.seed)  # 调用设置随机种子的函数

# 模型编码维度
encoding_size = 1024  # 设置模型编码维度
# 实验设置字典
settings = {'num_splits': args.k,  # k折数量
            'k_start': args.k_start,  # 起始fold
            'k_end': args.k_end,  # 结束fold
            'task': args.task,  # 任务类型
            'max_epochs': args.max_epochs,  # 最大训练轮数
            'results_dir': args.results_dir,  # 结果目录
            'lr': args.lr,  # 学习率
            'experiment': args.exp_code,  # 实验代码
            'reg': args.reg,  # 权重衰减
            'label_frac': args.label_frac,  # 标签比例
            'bag_loss': args.bag_loss,  # 包损失函数
            'seed': args.seed,  # 随机种子
            'model_type': args.model_type,  # 模型类型
            'model_size': args.model_size,  # 模型大小
            "use_drop_out": args.drop_out,  # 是否使用dropout
            'weighted_sample': args.weighted_sample,  # 是否使用加权采样
            'opt': args.opt}  # 优化器类型

# 如果使用CLAM模型，添加额外设置
if args.model_type in ['clam_sb', 'clam_mb']:  # 如果模型类型为CLAM
   settings.update({'bag_weight': args.bag_weight,  # 添加包级损失权重
                    'inst_loss': args.inst_loss,  # 添加实例级损失函数
                    'B': args.B})  # 添加正负样本数量

print('\nLoad Dataset')  # 打印加载数据集的提示信息

# 根据任务类型加载相应数据集
if args.task == 'task_1_tumor_vs_normal':  # 如果任务为肿瘤与正常组织的二分类
    args.n_classes=2  # 设置类别数为2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/optimized_description.csv',  # 加载数据集
                            data_dir= os.path.join(args.data_root_dir, 'MERGED_FEATURES'),  # 数据目录
                            shuffle = False,  # 不打乱数据
                            seed = args.seed,  # 设置随机种子
                            print_info = True,  # 打印信息
                            label_dict = {'TMB-H':0, 'TMB-L':1},  # 标签字典
                            patient_strat=False,  # 不进行病人层面分层
                            ignore=[],  # 忽略列表为空
                            use_text_desc=args.use_text_desc,  # 是否使用文本描述
                            text_desc_col='optimized_description')  # 文本描述列名

elif args.task == 'task_2_tumor_subtyping':  # 如果任务为肿瘤亚型分类
    args.n_classes=3  # 设置类别数为3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',  # 加载数据集
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),  # 数据目录
                            shuffle = False,  # 不打乱数据
                            seed = args.seed,  # 设置随机种子
                            print_info = True,  # 打印信息
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},  # 标签字典
                            patient_strat= False,  # 不进行病人层面分层
                            ignore=[])  # 忽略列表为空

    if args.model_type in ['clam_sb', 'clam_mb']:  # 如果模型类型为CLAM
        assert args.subtyping  # 确保启用了亚型分类
        
else:
    raise NotImplementedError  # 如果任务类型不支持则抛出异常
    
# 创建结果目录
if not os.path.isdir(args.results_dir):  # 如果结果目录不存在
    os.mkdir(args.results_dir)  # 创建结果目录

# 创建实验特定的结果目录
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))  # 生成实验结果目录
if not os.path.isdir(args.results_dir):  # 如果实验结果目录不存在
    os.mkdir(args.results_dir)  # 创建实验结果目录

# 设置数据集划分目录
if args.split_dir is None:  # 如果没有指定划分目录
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))  # 生成默认划分目录
else:
    args.split_dir = os.path.join('splits', args.split_dir)  # 使用指定的划分目录

print('split_dir: ', args.split_dir)  # 打印划分目录
assert os.path.isdir(args.split_dir)  # 确保划分目录存在

settings.update({'split_dir': args.split_dir})  # 更新设置字典中的划分目录

# 保存实验设置到文本文件
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:  # 打开文件以写入实验设置
    print(settings, file=f)  # 将设置写入文件
f.close()  # 关闭文件

# 打印实验设置
print("################# Settings ###################")  # 打印设置标题
for key, val in settings.items():  # 遍历设置字典
    print("{}:  {}".format(key, val))  # 打印每个设置项及其值

if __name__ == "__main__":  # 如果是主程序
    results = main(args)  # 调用主函数
    print("finished!")  # 打印完成提示
    print("end script")  # 打印结束脚本提示

