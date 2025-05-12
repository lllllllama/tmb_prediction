# 导入所需的库
import pdb  # Python调试器
import os   # 操作系统接口
import pandas as pd  # 数据分析库
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits  # 导入自定义数据集类和保存分割函数
import argparse  # 命令行参数解析
import numpy as np  # 数值计算库

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')  # 标签比例参数
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')  # 随机种子参数
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')  # 分割数量参数
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])  # 任务类型参数
parser.add_argument('--val_frac', type=float, default= 0.3,
                    help='fraction of labels for validation (default: 0.2)')  # 验证集比例参数
parser.add_argument('--test_frac', type=float, default= 0.0,
                    help='fraction of labels for test (default: 0.2)')  # 测试集比例参数

args = parser.parse_args()  # 解析命令行参数

# 根据任务类型初始化数据集
if args.task == 'task_1_tumor_vs_normal':  # 如果是二分类任务
    args.n_classes=2  # 设置类别数为2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/first.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'TMB-H':0, 'TMB-L':1},  # 标签字典:TMB高/低
                            patient_strat=True,  # 启用病人层面分层
                            ignore=[])  # 忽略列表为空

elif args.task == 'task_2_tumor_subtyping':  # 如果是三分类任务
    args.n_classes=3  # 设置类别数为3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},  # 标签字典:三种亚型
                            patient_strat= True,  # 启用病人层面分层
                            patient_voting='maj',  # 使用多数投票策略
                            ignore=[])  # 忽略列表为空

else:
    raise NotImplementedError  # 如果任务类型不支持则抛出异常

# 计算每个类别的验证集和测试集样本数
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])  # 获取每个类别的样本数
val_num = np.round(num_slides_cls * args.val_frac).astype(int)  # 计算验证集样本数
test_num = np.round(num_slides_cls * args.test_frac).astype(int)  # 计算测试集样本数

if __name__ == '__main__':
    # 设置标签比例列表
    if args.label_frac > 0:
        label_fracs = [args.label_frac]  # 如果指定了标签比例,则只使用该比例
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]  # 否则使用预设的比例列表
    
    # 对每个标签比例进行处理
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))  # 创建保存分割结果的目录
        os.makedirs(split_dir, exist_ok=True)  # 确保目录存在
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)  # 创建k折交叉验证的分割
        
        # 对每一折进行处理
        for i in range(args.k):
            dataset.set_splits()
            splits = dataset.return_splits(from_id=True)
            # 在打印之前先进行验证集复制
            if splits[2] is None:
                splits = [splits[0], splits[1], splits[1]]
                # 更新dataset的test_ids
                dataset.test_ids = dataset.val_ids.copy()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            # 保存分割结果(普通格式)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            # 保存分割结果(布尔格式)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            # 保存描述符信息
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))

