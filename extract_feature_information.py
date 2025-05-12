import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('output_file.csv')

# 定义需要统计的临床变量及其分类
clinical_variables = {
    "Tumor stage": {
        "Stage I": df['clinical_stage'] == 'Stage I',
        "Stage II": df['clinical_stage'] == 'Stage II',
        "Stage III": df['clinical_stage'] == 'Stage III',
        "Stage IV": df['clinical_stage'] == 'Stage IV',
        "Unknown": df['clinical_stage'].isna() | (df['clinical_stage'] == 'Unknown')
    },
    "Prior malignancy": {
        "Yes": df['other_dx'] == 'Yes, History of Synchronous/Bilateral Malignancy',
        "No": df['other_dx'] == 'No',
        "Unknown": df['other_dx'].isna() | (df['other_dx'] == 'Unknown')
    },
    "AJCC pathologic T": {
        "T1": df['histological_grade'] == 'G1',
        "T2": df['histological_grade'] == 'G2',
        "T3": df['histological_grade'] == 'G3',
        "T4": df['histological_grade'] == 'GB',
        "Unknown": df['histological_grade'].isna() | (df['histological_grade'] == 'Unknown')
    },
    "AJCC pathologic N": {
        "N0": df['lymphatic_invasion'] == 'NO',
        "N1": df['lymphatic_invasion'] == 'YES',
        "N2": np.zeros(len(df), dtype=bool),
        "Nx": np.zeros(len(df), dtype=bool),
        "Unknown": df['lymphatic_invasion'].isna() | (df['lymphatic_invasion'] == 'Unknown')
    },
    "AJCC pathologic M": {
        "M0": df['tumor_status'] == 'TUMOR FREE',
        "M1": df['tumor_status'] == 'WITH TUMOR',
        "Mx": np.zeros(len(df), dtype=bool),
        "Unknown": df['tumor_status'].isna() | (df['tumor_status'] == 'Unknown')
    },
    "Gender": {
        "Women": df['gender'] == 'FEMALE',
        "Men": df['gender'] == 'MALE',
        "Unknown": df['gender'].isna() | (df['gender'] == 'Unknown')
    },
    "Vital status": {
        "Alive": df['vital_status'] == 'Alive',
        "Dead": df['vital_status'] == 'Dead',
        "Unknown": df['vital_status'].isna() | (df['vital_status'] == 'Unknown')
    },
    "Age at index": {
        "≥66": df['age_group'] == '>=60',
        "<66": df['age_group'] == '<60'
    },
    "New tumor event after initial treatment": {
        "Yes": df['new_tumor_event_after_initial_treatment'] == 'YES',
        "No": df['new_tumor_event_after_initial_treatment'] == 'NO',
        "Unknown": df['new_tumor_event_after_initial_treatment'].isna() | (df['new_tumor_event_after_initial_treatment'] == 'Unknown')
    }
}

# 创建结果数据框
results = []
total_patients = len(df)

# 统计每个临床变量及其分类的患者数量
for variable, categories in clinical_variables.items():
    for category, mask in categories.items():
        count = int(mask.sum())
        results.append({
            "Clinical variable": variable,
            "Category": category,
            "Number of patients": count
        })

# 转换为DataFrame
result_df = pd.DataFrame(results)

# 按clinical variable分组，并保持顺序
ordered_variables = list(clinical_variables.keys())
result_df['Clinical variable order'] = result_df['Clinical variable'].apply(lambda x: ordered_variables.index(x))
result_df = result_df.sort_values(by=['Clinical variable order', 'Category'])
result_df = result_df.drop(columns=['Clinical variable order'])

# 保存到CSV文件，使用utf-8-sig编码以支持中文
result_df.to_csv('clinical_statistics.csv', index=False, encoding='utf-8-sig')

# 尝试保存为Excel格式，先检查是否安装了openpyxl
try:
    # 创建 Excel writer
    with pd.ExcelWriter('clinical_statistics.xlsx', engine='openpyxl') as writer:
        # 写入基本统计表
        result_df.to_excel(writer, sheet_name='Clinical Statistics', index=False)
        
        # 设置列宽
        workbook = writer.book
        worksheet = writer.sheets['Clinical Statistics']
        worksheet.column_dimensions['A'].width = 35
        worksheet.column_dimensions['B'].width = 20
        worksheet.column_dimensions['C'].width = 20
        
    print("\n已保存统计结果到 clinical_statistics.xlsx")
except ImportError:
    print("\n警告：未安装openpyxl库，无法保存为Excel格式")
    print("如需保存为Excel格式，请安装openpyxl: pip install openpyxl")

# 打印格式化的表格
print("\n临床特征统计结果：")
print("="*80)
print(f"{'Clinical variable':<35} {'Category':<25} {'Number of patients':<15}")
print("-"*80)

current_var = None
for _, row in result_df.iterrows():
    var = row['Clinical variable']
    category = row['Category']
    count = row['Number of patients']
    
    if var != current_var:
        current_var = var
        print(f"{var:<35} {category:<25} {count:<15}")
    else:
        print(f"{'':<35} {category:<25} {count:<15}")
print("="*80)

# 如果要创建HTML表格
html_output = """
<html>
<head>
<style>
table {
  border-collapse: collapse;
  width: 100%;
}
th, td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}
th {
  background-color: #f2f2f2;
}
</style>
</head>
<body>
<table>
  <tr>
    <th>Clinical variable</th>
    <th>Category</th>
    <th>Number of patients</th>
  </tr>
"""

current_var = None
for _, row in result_df.iterrows():
    var = row['Clinical variable']
    category = row['Category']
    count = row['Number of patients']
    
    if var != current_var:
        current_var = var
        html_output += f"  <tr>\n    <td>{var}</td>\n    <td>{category}</td>\n    <td>{count}</td>\n  </tr>\n"
    else:
        html_output += f"  <tr>\n    <td></td>\n    <td>{category}</td>\n    <td>{count}</td>\n  </tr>\n"

html_output += """
</table>
</body>
</html>
"""

with open('clinical_statistics.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("\n已生成HTML表格文件：clinical_statistics.html")
