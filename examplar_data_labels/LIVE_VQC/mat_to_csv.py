import scipy.io
import pandas as pd

def mat_to_csv(mat_file_path, csv_file_path):
    # 加载 MAT 文件
    mat_data = scipy.io.loadmat(mat_file_path)

    # 提取 video_list 和 mos
    video_list = mat_data['video_list'].flatten()
    mos = mat_data['mos'].flatten()

    # 创建 DataFrame
    df = pd.DataFrame({
        'File': [v[0] for v in video_list],  # 提取视频文件名
        'MOS': mos
    })

    # 保存为 CSV 文件
    df.to_csv(csv_file_path, index=False)

# 使用示例
mat_file_path = '/home/jiehui/VQA/COVER/examplar_data_labels/LIVE_VQC/data.mat'
csv_file_path = './datasets/LIVE_VQC/LIVE_VQC_metadata.csv'
mat_to_csv(mat_file_path, csv_file_path)

# 加载 CSV 文件并使用
datainfo = './datasets/LIVE_VQC/LIVE_VQC_metadata.csv'
df = pd.read_csv(datainfo)
files = df['File'].tolist()
mos = df['MOS'].tolist()

print(files)
print(mos)