import csv

# 定义输入和输出文件路径
input_file = 'evaluation_log.txt'
output_file = 'evaluation_results.csv'

# 初始化数据结构
data = {
    "Semantic Time": [],
    "Technical Time": [],
    "Aesthetic Time": [],
    "Sample Views Time": [],
    "PyTorch Inference Time": []
}

# 读取输入文件
with open(input_file, 'r') as file:
    lines = file.readlines()

# 解析数据
for i in range(0, len(lines), 7):  # 每个视频的记录占用7行
    command_line = lines[i].strip()
    semantic_time = lines[i+1].strip().split(': ')[1].split(' ')[0]
    technical_time = lines[i+2].strip().split(': ')[1].split(' ')[0]
    aesthetic_time = lines[i+3].strip().split(': ')[1].split(' ')[0]
    sample_views_time = lines[i+4].strip().split(': ')[1].split(' ')[0]
    pytorch_inference_time = lines[i+5].strip().split(': ')[1].split(' ')[0]
    
    # 提取视频文件名
    video_file = command_line.split('-v ')[1]
    
    # 提取 NUMA 绑定信息
    if "numactl" in command_line:
        numa_info = command_line.split('-C ')[1].split(' ')[0]
    else:
        numa_info = "没绑核"
    
    # 将数据添加到字典
    data["Semantic Time"].append((video_file, numa_info, semantic_time))
    data["Technical Time"].append((video_file, numa_info, technical_time))
    data["Aesthetic Time"].append((video_file, numa_info, aesthetic_time))
    data["Sample Views Time"].append((video_file, numa_info, sample_views_time))
    data["PyTorch Inference Time"].append((video_file, numa_info, pytorch_inference_time))

# 写入CSV文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入表头
    header = ["Category"] + [f"{video} ({numa})" for video, numa, _ in data["Semantic Time"]]
    writer.writerow(header)
    
    # 写入数据
    for category, values in data.items():
        row = [category] + [time for _, _, time in values]
        writer.writerow(row)

print(f"Data has been written to {output_file}")