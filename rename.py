import os
from pathlib import Path
import re

def rename_tcga_files(directory):
    """
    批量重命名TCGA文件，将长格式文件名改为简短格式
    例如：
    TCGA-13-A5FT-01Z-00-DX1.2B292DC8-7336-4CD9-AB1A-F6F482E6151A.h5 -> TCGA-13-A5FT.h5
    TCGA-13-A5FT-01Z-00-DX1.2B292DC8-7336-4CD9-AB1A-F6F482E6151A.pt -> TCGA-13-A5FT.pt
    """
    # 将路径转换为Path对象
    directory = Path(directory).absolute()
    
    # 定义TCGA文件名的正则表达式模式
    pattern = r'(TCGA-\w{2}-\w{4})-.*\.(h5|pt)'
    
    # 记录处理的文件数量
    count = {'h5': 0, 'pt': 0}
    errors = []
    
    print(f"开始处理目录: {directory}")
    
    # 遍历目录中的所有.h5和.pt文件
    for file_path in directory.glob('*.*'):
        if file_path.suffix not in ['.h5', '.pt']:
            continue
            
        try:
            # 获取文件名
            old_name = file_path.name
            
            # 使用正则表达式匹配并提取所需部分
            match = re.match(pattern, old_name)
            if match:
                # 构建新文件名，保持原始扩展名
                new_name = f"{match.group(1)}{file_path.suffix}"
                new_path = file_path.parent / new_name
                
                # 如果新文件名已存在，则跳过
                if new_path.exists():
                    print(f"警告: 文件已存在，跳过重命名: {new_name}")
                    continue
                
                # 重命名文件
                file_path.rename(new_path)
                count[file_path.suffix[1:]] += 1  # 去掉.后的扩展名
                print(f"重命名: {old_name} -> {new_name}")
                
        except Exception as e:
            errors.append(f"处理文件 {file_path.name} 时出错: {str(e)}")
    
    # 打印统计信息
    print(f"\n处理完成!")
    print(f"成功重命名.h5文件数: {count['h5']}")
    print(f"成功重命名.pt文件数: {count['pt']}")
    print(f"总计重命名文件数: {sum(count.values())}")
    
    if errors:
        print("\n错误信息:")
        for error in errors:
            print(f"- {error}")

if __name__ == "__main__":
    # 指定要处理的目录路径
    directories = [
        "/root/autodl-tmp/CLAM/MERGED_FEATURES/h5_files",  # 修改为你的实际h5文件路径
        "/root/autodl-tmp/CLAM/MERGED_FEATURES/pt_files"   # 修改为你的实际pt文件路径
    ]
    
    # 执行重命名操作
    for directory in directories:
        print(f"\n处理目录: {directory}")
        rename_tcga_files(directory)