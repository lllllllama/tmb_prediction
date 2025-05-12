import chardet

# 检测文件的原始编码
with open('/root/autodl-tmp/CLAM/dataset_csv/test1.csv', 'rb') as f:
    result = chardet.detect(f.read())
    original_encoding = result['encoding']

# 读取文件并转换为 UTF-8 编码
with open('/root/autodl-tmp/CLAM/dataset_csv/test1.csv', 'r', encoding=original_encoding) as f:
    content = f.read()

# 将内容写入新的 UTF-8 编码文件
with open('/root/autodl-tmp/CLAM/dataset_csv/test1_utf8.csv', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"文件已成功转换为 UTF-8 编码并保存为 test1_utf8.csv")