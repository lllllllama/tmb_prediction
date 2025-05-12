import pandas as pd
import os

def generate_optimized_description(row):
    """生成优化的临床描述文本（英文版），去除重复信息"""
    text = ""
    
    # 只包含关键差异化信息
    # 1. 年龄 (保留，因为年龄与TMB可能相关)
    text += f"Age {row['age_at_initial_pathologic_diagnosis']} years, "
    
    # 2. 突变计数 (TMB的直接相关指标)
    if pd.notna(row['Maftools_Counts']) and row['Maftools_Counts'] != 'Unknown':
        text += f"mutation count {row['Maftools_Counts']}, "
    
    # 3. HRD评分和标签 (与基因组不稳定性相关)
    if pd.notna(row['HRD_Score']) and row['HRD_Score'] != 'Unknown':
        text += f"HRD score {row['HRD_Score']}"
        if row['HRD_Label'] == 'Positive':
            text += " (positive), "
        elif row['HRD_Label'] == 'Negative':
            text += " (negative), "
        else:
            text += ", "
    
    # 4. 临床分期 (保留，因为晚期肿瘤可能有更高的突变负荷)
    if row['clinical_stage'] != 'Unknown':
        text += f"clinical stage {row['clinical_stage']}, "
    
    # 5. 复发/转移信息 (可能与肿瘤进展和TMB相关)
    recurrence_info = []
    if row['Recurrence or Metastasis within One Years'] == 'YES':
        recurrence_info.append("recurrence within one year")
    elif row['Recurrence or Metastasis within Three Years'] == 'YES' and row['Recurrence or Metastasis within One Years'] != 'YES':
        recurrence_info.append("recurrence within three years")
    elif row['Recurrence or Metastasis within Five Years'] == 'YES' and row['Recurrence or Metastasis within Three Years'] != 'YES':
        recurrence_info.append("recurrence within five years")
    
    if recurrence_info:
        text += f"{', '.join(recurrence_info)}, "
    
    # 6. 肿瘤状态 (可能反映治疗反应，与TMB相关)
    if row['tumor_status'] == 'WITH TUMOR':
        text += "with tumor, "
    elif row['tumor_status'] == 'TUMOR FREE':
        text += "tumor free, "
    
    # 7. 生存状态 (可能与TMB相关)
    if row['vital_status'] == 'Alive':
        text += f"survived {row['futime']} days"
    elif row['vital_status'] == 'Dead':
        text += f"survived {row['futime']} days before death"
    
    # 8. 肿瘤残留 (可能反映肿瘤侵袭性)
    if row['tumor_residual_disease'] != 'Unknown':
        text += f", residual {row['tumor_residual_disease']}"
    
    # 9. 种族 (如果与TMB相关)
    if row['race'] != 'Unknown':
        text += f", {row['race']} race"
    
    text += "."
    
    return text

def main():
    # 读取临床数据
    csv_path = '/root/autodl-tmp/CLAM/output_file.csv'
    output_path = '/root/autodl-tmp/CLAM/optimized_description.csv'
    
    df = pd.read_csv(csv_path)
    
    # 生成优化的描述性文本
    descriptions = []
    for idx, row in df.iterrows():
        description = generate_optimized_description(row)
        descriptions.append(description)
    
    # 创建新的DataFrame，只包含case_id, slide_id, label和描述文本
    result_df = pd.DataFrame({
        'case_id': df['case_id'],
        'slide_id': df['slide_id'],
        'label': df['label'],
        'optimized_description': descriptions
    })
    
    # 保存到CSV文件
    result_df.to_csv(output_path, index=False)
    print(f"Optimized clinical descriptions generated and saved to {output_path}")
    
    # 打印前几个样本的描述文本示例
    print("\nExamples of optimized descriptions for the first 3 samples:")
    for i in range(min(3, len(result_df))):
        print(f"\nSample {i+1} ({result_df.iloc[i]['case_id']}):")
        print(result_df.iloc[i]['optimized_description'])

if __name__ == "__main__":
    main()