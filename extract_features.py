# 导入所需的库
import time  # 用于计时
import os  # 用于文件和目录操作
import argparse  # 用于命令行参数解析
import pdb  # 用于调试

# 导入PyTorch相关库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import DataLoader  # 数据加载器
from PIL import Image  # 图像处理
import h5py  # HDF5文件操作
import openslide  # 病理切片读取

# 导入进度条和数值计算库
from tqdm import tqdm  # 进度条显示
import numpy as np  # 数值计算

# 导入自定义模块
from utils.file_utils import save_hdf5  # HDF5文件保存工具
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag, get_eval_transforms  # 数据集相关类
from models import get_encoder  # 特征提取模型

# 设置计算设备(GPU/CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	使用数据加载器计算特征并保存到h5文件
	args:
		output_path: 保存计算特征的目录(.h5文件)
		model: pytorch模型
		verbose: 反馈级别
	"""
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'  # 文件写入模式,第一次为写入,之后为追加
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	 # 推理模式,不计算梯度
			batch = data['img']  # 获取图像批次
			coords = data['coord'].numpy().astype(np.int32)  # 获取坐标信息
			batch = batch.to(device, non_blocking=True)  # 数据转移到GPU/CPU
			
			features = model(batch)  # 提取特征
			
			features = features.cpu().numpy()  # 特征转回CPU并转为numpy数组

			# 保存特征和坐标信息到h5文件
			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'  # 切换为追加模式
	
	return output_path


# 命令行参数解析器
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str)  # 数据目录
parser.add_argument('--csv_path', type=str)  # CSV文件路径
parser.add_argument('--feat_dir', type=str)  # 特征保存目录
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])  # 模型名称
parser.add_argument('--batch_size', type=int, default=256)  # 批次大小
parser.add_argument('--slide_ext', type=str, default= '.svs')  # 切片文件扩展名
parser.add_argument('--no_auto_skip', default=False, action='store_true')  # 是否自动跳过已处理文件
parser.add_argument('--target_patch_size', type=int, default=224,
					help='the desired size of patches for scaling before feature embedding')  # 目标图像块大小
args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	bags_dataset = Dataset_All_Bags(csv_path)  # 初始化数据集
	
	os.makedirs(args.feat_dir, exist_ok=True)  # 创建特征保存目录
	dest_files = os.listdir(args.feat_dir)  # 获取已存在的特征文件列表

	# 获取编码器模型和图像变换
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)		
	model = model.to(device)  # 模型转移到GPU/CPU
	_ = model.eval()  # 设置为评估模式

	# 设置数据加载器参数
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
	
	total = len(bags_dataset)
	# 遍历处理每个切片
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]  # 获取切片ID
		bag_name = slide_id + '.h5'  # 构建h5文件名
		bag_candidate = os.path.join(args.data_dir, 'patches', bag_name)  # 构建完整文件路径

		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(bag_name)
		# 如果已处理则跳过
		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)  # 特征输出路径
		file_path = bag_candidate
		time_start = time.time()  # 记录开始时间

		# 创建数据集和数据加载器
		dataset = Whole_Slide_Bag(file_path=file_path, img_transforms=img_transforms)
		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		# 计算并显示处理时间
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		
		# 读取并验证特征
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		# 将特征转换为PyTorch张量并保存
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
