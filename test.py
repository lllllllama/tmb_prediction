import h5py  
import matplotlib.pyplot as plt  
import os  
import numpy as np  

# 指定你的 .h5 文件路径  
file_path = 'FEATURES_DIRECTORY1/h5_files/TCGA-25-2401-01A-01-TS1.998ebf95-b49a-46c5-b501-b877912cbd44.h5'

# 检查文件路径是否存在  
if not os.path.exists(file_path):  
    print("The specified file path does not exist.")  
else:  
    with h5py.File(file_path, 'r') as f:  
        # 获取坐标数据和特征数据  
        coords = f['coords'][:]  
        features = f['features'][:]  # 假设存在名为 'features' 的数据集  

        # 打印特征数据的形状, 以验证其维度  
        print("Shape of features:", features.shape)  

        # 选择一个特征作为颜色映射，可以选择第一个特征  
        values = features[:, 0]  # 改为您需要的特征列，当前选择第一个特征  

    # 清空当前图形  
    plt.clf()  
    
    # 使用颜色渐变绘制散点图  
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', marker='o', s=5)  
    plt.colorbar(scatter, label='Feature Values')  # 添加颜色条  
  
    plt.title('Coordinate Points Colored by Feature Values')  
    plt.xlabel('X-axis')  
    plt.ylabel('Y-axis')  
    plt.grid()  
    
    # 保存图形  
    plt.savefig('coordinate_points_colored_by_feature_values.png', dpi=300, bbox_inches='tight')  
    plt.close()