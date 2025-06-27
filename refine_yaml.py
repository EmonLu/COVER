from ruamel.yaml import YAML
import sys

def modify_yaml(file_path, seconds):
    yaml = YAML()
    yaml.preserve_quotes = True  # 保留引号

    # 读取 YAML 文件
    with open(file_path, 'r') as file:
        config = yaml.load(file)

    # 修改 val-ytugc 下的相关值
    ytugc = config['data']['val-ytugc']['args']['sample_types']
    
    # 修改 semantic 分支
    ytugc['semantic']['clip_len'] = seconds
    ytugc['semantic']['t_frag'] = seconds
    
    # 修改 technical 分支
    ytugc['technical']['aligned'] = seconds * 2
    ytugc['technical']['clip_len'] = seconds * 2
    ytugc['technical']['t_frag'] = seconds
    
    # 修改 aesthetic 分支
    ytugc['aesthetic']['clip_len'] = seconds * 2
    ytugc['aesthetic']['t_frag'] = seconds
    
    # 修改 val-livevqc 下的相关值
    ytugc = config['data']['val-livevqc']['args']['sample_types']
    
    ytugc['semantic']['clip_len'] = seconds
    ytugc['semantic']['t_frag'] = seconds
    
    ytugc['technical']['aligned'] = seconds * 2
    ytugc['technical']['clip_len'] = seconds * 2
    ytugc['technical']['t_frag'] = seconds
    
    ytugc['aesthetic']['clip_len'] = seconds * 2
    ytugc['aesthetic']['t_frag'] = seconds

    # 将修改后的配置写回 YAML 文件
    with open(file_path, 'w') as file:
        yaml.dump(config, file)


# 示例使用
file_path = 'cover.yml'  # 替换为你的 YAML 文件路径
seconds = int(sys.argv[1])  # 输入秒数
modify_yaml(file_path, seconds)