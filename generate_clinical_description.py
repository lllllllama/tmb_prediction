import pandas as pd
import os

def generate_clinical_description(row):
    """根据临床数据生成描述性文本"""
    text = f"这是一位{row['age_at_initial_pathologic_diagnosis']}岁的"
    
    # 添加种族和性别信息
    if row['race'] != 'Unknown':
        text += f"{row['race']}人种"
    text += f"{row['gender'].lower()}性患者，"
    
    # 添加肿瘤类型和分级信息
    text += f"被诊断为{row['Primary_Site']}部位的{row['histological_type']}，组织学分级为{row['histological_grade']}，"
    
    # 添加临床分期信息
    if row['clinical_stage'] != 'Unknown':
        text += f"临床分期为{row['clinical_stage']}，"
    
    # 添加诊断方法
    if row['initial_pathologic_diagnosis_method'] != 'Unknown':
        text += f"通过{row['initial_pathologic_diagnosis_method']}进行诊断，"
    
    # 添加HRD评分信息
    if pd.notna(row['HRD_Score']) and row['HRD_Score'] != 'Unknown':
        text += f"HRD评分为{row['HRD_Score']}，"
        if row['HRD_Label'] == 'Positive':
            text += "HRD呈阳性，"
        elif row['HRD_Label'] == 'Negative':
            text += "HRD呈阴性，"
    
    # 添加突变计数信息
    if pd.notna(row['Maftools_Counts']) and row['Maftools_Counts'] != 'Unknown':
        text += f"突变计数为{row['Maftools_Counts']}，"
    
    # 添加肿瘤状态信息
    if row['tumor_status'] == 'WITH TUMOR':
        text += "目前仍有肿瘤，"
    elif row['tumor_status'] == 'TUMOR FREE':
        text += "目前无肿瘤，"
    
    # 添加复发/转移信息
    recurrence_info = []
    if row['Recurrence or Metastasis within One Years'] == 'YES':
        recurrence_info.append("一年内")
    elif row['Recurrence or Metastasis within Three Years'] == 'YES' and row['Recurrence or Metastasis within One Years'] != 'YES':
        recurrence_info.append("三年内")
    elif row['Recurrence or Metastasis within Five Years'] == 'YES' and row['Recurrence or Metastasis within Three Years'] != 'YES':
        recurrence_info.append("五年内")
    
    if recurrence_info:
        text += f"{'、'.join(recurrence_info)}出现复发或转移，"
    elif row['Recurrence or Metastasis within One Years'] == 'NO':
        text += "一年内未出现复发或转移，"
    
    # 添加放疗信息
    if row['radiation_therapy'] == 'YES':
        text += "接受了放射治疗，"
    elif row['radiation_therapy'] == 'NO':
        text += "未接受放射治疗，"
    
    # 添加术后治疗信息
    if row['postoperative_rx_tx'] == 'YES':
        text += "接受了术后辅助治疗，"
    
    # 添加生存状态信息
    if row['vital_status'] == 'Alive':
        text += f"患者存活，随访时间为{row['futime']}天"
        if pd.notna(row['days_to_last_followup']) and row['days_to_last_followup'] != 'Unknown':
            text += f"，最后一次随访时间为{row['days_to_last_followup']}天"
    elif row['vital_status'] == 'Dead':
        text += f"患者已死亡，生存时间为{row['futime']}天"
        if pd.notna(row['days_to_death']) and row['days_to_death'] != 'Unknown':
            text += f"，死亡时间为{row['days_to_death']}天"
    
    text += "。"
    
    # 添加肿瘤残留信息
    if row['tumor_residual_disease'] != 'Unknown':
        text += f"手术后肿瘤残留状态为{row['tumor_residual_disease']}。"
    
    # 添加解剖学位置信息
    if row['anatomic_neoplasm_subdivision'] != 'Unknown':
        text += f"肿瘤位于{row['anatomic_neoplasm_subdivision']}侧。"
    
    # 添加治疗结果信息
    if row['primary_therapy_outcome_success'] != 'Unknown':
        text += f"治疗结果为{row['primary_therapy_outcome_success']}。"
    
    # 添加年龄组信息
    if row['age_group'] != 'Unknown':
        text += f"患者年龄组为{row['age_group']}。"
    
    return text

def main():
    # 读取临床数据
    csv_path = '/root/autodl-tmp/CLAM/output_file.csv'
    output_path = '/root/autodl-tmp/CLAM/clinical_description.csv'
    
    df = pd.read_csv(csv_path)
    
    # 生成描述性文本
    descriptions = []
    for idx, row in df.iterrows():
        description = generate_clinical_description(row)
        descriptions.append(description)
    
    # 创建新的DataFrame，只包含case_id, slide_id, label和描述文本
    result_df = pd.DataFrame({
        'case_id': df['case_id'],
        'slide_id': df['slide_id'],
        'label': df['label'],
        'clinical_description': descriptions
    })
    
    # 保存到CSV文件
    result_df.to_csv(output_path, index=False)
    print(f"已生成临床描述文本，保存到 {output_path}")
    
    # 打印前几个样本的描述文本示例
    print("\n前3个样本的描述文本示例:")
    for i in range(min(3, len(result_df))):
        print(f"\nSample {i+1} ({result_df.iloc[i]['case_id']}):")
        print(result_df.iloc[i]['clinical_description'])

if __name__ == "__main__":
    main()