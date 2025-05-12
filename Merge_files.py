import os
import shutil
from pathlib import Path

def merge_features_directories(base_path, target_folders, output_base):
    """合并FEATURES_DIRECTORY文件夹"""
    # 将路径转换为绝对路径
    base_path = Path(base_path).absolute()
    output_base = Path(output_base).absolute()
    
    # 创建输出目录
    output_base.mkdir(exist_ok=True, parents=True)
    print(f"创建输出目录: {output_base}")
    
    # 获取所有FEATURES_DIRECTORY*文件夹，使用绝对路径
    feature_dirs = [
        base_path / "FEATURES_DIRECTORY",
        base_path / "FEATURES_DIRECTORY1",
        base_path / "FEATURES_DIRECTORY2",
        base_path / "FEATURES_DIRECTORY3",
        base_path / "FEATURES_DIRECTORY4"
    ]
    
    print(f"准备合并以下目录:")
    for dir in feature_dirs:
        print(f"- {dir}")  # 打印完整路径
    
    # 为每个目标文件夹创建输出目录
    for target in target_folders:
        (output_base / target).mkdir(exist_ok=True)
        print(f"创建子目录: {output_base / target}")
    
    # 遍历每个目录
    total_files = {target: 0 for target in target_folders}
    
    for feature_dir in feature_dirs:
        if not feature_dir.exists():
            print(f"\n警告: 目录不存在: {feature_dir}")
            continue
            
        print(f"\n处理目录: {feature_dir}")
        
        # 遍历需要合并的文件夹类型
        for target in target_folders:
            source_path = feature_dir / target
            if not source_path.exists():
                print(f"  在 {feature_dir} 中未找到 {target}")
                continue
                
            dest_path = output_base / target
            print(f"  合并 {target}:")
            
            # 复制文件
            file_count = 0
            for item in source_path.iterdir():
                dest_item = dest_path / item.name
                try:
                    if item.is_file():
                        shutil.copy2(item, dest_item)
                        file_count += 1
                    else:
                        shutil.copytree(item, dest_item, dirs_exist_ok=True)
                        file_count += len(list(item.rglob('*')))
                except Exception as e:
                    print(f"    错误: 处理 {item} 时出错: {str(e)}")
            
            total_files[target] += file_count
            print(f"    复制了 {file_count} 个文件")

    print("\n合并完成!")
    print("\n最终统计:")
    for target, count in total_files.items():
        print(f"{target} 中的文件总数: {count}")

if __name__ == "__main__":
    # 使用示例 - 使用绝对路径
    base_directory = "/root/autodl-tmp/CLAM"  # 修改为你的实际路径
    target_folders = ["h5_files", "pt_files"]
    output_directory = "/root/autodl-tmp/CLAM/MERGED_FEATURES"  # 修改为你的实际路径

    print("开始合并文件夹...")
    merge_features_directories(base_directory, target_folders, output_directory)