import pandas as pd

# 读取 XLSX 文件
df_xlsx = pd.read_excel('dataset_csv/临床数据.xlsx')  # 替换为你的 XLSX 文件名

# 读取 CSV 文件
df_csv = pd.read_csv('dataset_csv/first.csv')  # 替换为你的 CSV 文件名

# 假设 XLSX 文件有 'Sample ID' 列，CSV 文件有 'case_id' 列
# 这里假设 case_id 的前12个字符与 Sample ID 匹配（如果不同，请调整逻辑）
df_csv['sample_prefix'] = df_csv['case_id'].str[:12]

# 匹配数据（内连接）
df_merged = pd.merge(df_xlsx, df_csv, 
                     left_on='Sample ID', 
                     right_on='sample_prefix', 
                     how='inner')

# 删除辅助列（可选）
df_merged.drop(columns=['sample_prefix'], inplace=True)

# 重新排列列顺序，将case_id和slide_id放在前两列，并删除Sample ID
cols = df_merged.columns.tolist()
cols.remove('case_id')
cols.remove('slide_id')
if 'Sample ID' in cols:
    cols.remove('Sample ID')
new_cols = ['case_id', 'slide_id'] + cols
df_merged = df_merged[new_cols]

# 保存结果到新的 CSV 文件
df_merged.to_csv('output_file.csv', index=False)

# 打印前几行检查结果（可选）
print(df_merged.head())