# 导入所需的库
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 设置HuggingFace镜像

import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

# 设置设备(GPU/CPU)
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
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

# 命令行参数解析
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)  # h5数据目录
parser.add_argument('--data_slide_dir', type=str, default=None)  # 病理切片目录
parser.add_argument('--slide_ext', type=str, default= '.svs')  # 切片文件扩展名
parser.add_argument('--csv_path', type=str, default=None)  # CSV文件路径
parser.add_argument('--feat_dir', type=str, default=None)  # 特征保存目录
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])  # 模型名称
parser.add_argument('--batch_size', type=int, default=256)  # 批次大小
parser.add_argument('--no_auto_skip', default=False, action='store_true')  # 是否自动跳过已处理文件
parser.add_argument('--target_patch_size', type=int, default=224)  # 目标图像块大小
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	# 初始化数据集
	bags_dataset = Dataset_All_Bags(csv_path)
	
	# 创建特征保存目录
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	# 获取编码器模型和图像变换
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	# 设置数据加载器参数
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	# 遍历处理每个切片
	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		# 如果已处理则跳过
		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		# 创建数据加载器并计算特征
		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		# 读取并验证特征
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		# 保存特征为PyTorch格式
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))


