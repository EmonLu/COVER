import torch

import argparse
import pickle as pkl

import decord
from decord import VideoReader
import numpy as np
import yaml

from cover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from cover.models import COVER
import time
import openvino as ov 

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

mean_clip, std_clip = (
    torch.FloatTensor([122.77, 116.75, 104.09]),
    torch.FloatTensor([68.50, 66.63, 70.32])
)

def fuse_results(results: list):
    x = (results[0] + results[1] + results[2])
    return {
        "semantic" : results[0],
        "technical": results[1],
        "aesthetic": results[2],
        "overall"  : x,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt"   , type=str, default="./cover.yml", help="the option file")
    parser.add_argument("-v", "--video_path", type=str, default="./demo/video_1.mp4" , help='output file to store predict mos value')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    """
    BASIC SETTINGS
    """
    if torch.cuda.is_available():
        torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.opt, "r") as f:
       opt = yaml.safe_load(f)
    
    dopt = opt["data"]["val-ytugc"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        temporal_samplers[stype] = UnifiedFrameSampler(
            sopt["clip_len"] // sopt["t_frag"],
            sopt["t_frag"],
            sopt["frame_interval"],
            sopt["num_clips"],
        )

    """
    LOAD MODEL
    """    
    core = ov.Core()
    compiled_model = core.compile_model("cover.xml")
    infer_request = compiled_model.create_infer_request()
    print("Finish compiling model!")

    """
    VIEW SAMPLE
    """
    t1 = time.time()
    views, _ = spatial_temporal_view_decomposition(
        args.video_path, dopt["sample_types"], temporal_samplers
    )

    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        if k == 'technical' or k == 'aesthetic':
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(device)
            )
        elif k == 'semantic':
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean_clip) / std_clip)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .squeeze(0)
                .to(device)
            )

    semantic_array = np.ascontiguousarray(views["semantic"].numpy())
    print(semantic_array.shape)
    technical_array = np.ascontiguousarray(views["technical"].numpy())
    aesthetic_array = np.ascontiguousarray(views["aesthetic"].numpy())
    semantic_tensor = ov.Tensor(semantic_array, shared_memory=True)
    technical_tensor = ov.Tensor(technical_array, shared_memory=True)
    aethetic_tensor = ov.Tensor(aesthetic_array, shared_memory=True)

    input_data = {
        'semantic': semantic_tensor,
        'technical': technical_tensor,
        'aesthetic': aethetic_tensor
    }

    for input_name, tensor in input_data.items():
        infer_request.set_tensor(input_name, tensor)    
    
    """
    INFERENCE
    """
    t2 = time.time()
    infer_request.start_async()
    infer_request.wait()

    ov_results = []
    for output in compiled_model.outputs:
        output_tensor = infer_request.get_tensor(output)
        output_data = output_tensor.data
        mean_value = output_data.mean().item()
        ov_results.append(mean_value)
    t3 = time.time()
    pred_score = fuse_results(ov_results)
    print(f"path, semantic score, technical score, aesthetic score, overall/final score")
    print(f'{args.video_path.split("/")[-1]},{pred_score["semantic"]:4f},{pred_score["technical"]:4f},{pred_score["aesthetic"]:4f},{pred_score["overall"]:4f}')
    
    print("sample views: {:.3f} seconds".format(t2 - t1))
    print("openvino inference: {:.3f} seconds".format(t3 - t2))
    