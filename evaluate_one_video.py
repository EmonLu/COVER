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
from scipy.special import expit

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

class ScoreNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std if std > 0 else 1e-6  # 防止除以 0

    def normalize(self, score: float) -> float:
        standardized = (score - self.mean) / self.std
        return float(expit(standardized))  # sigmoid 归一化到 0~1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt"   , type=str, default="./cover.yml", help="the option file")
    parser.add_argument('-d', "--device", type=str, default="cuda"       , help='CUDA device id')
    parser.add_argument("-v", "--video_path", type=str, default="./demo/video_1.mp4" , help='output file to store predict mos value')
    parser.add_argument("-m", "--mode", type=str, choices=["inference", "throughput"], default="inference",
                        help="choose to run 'inference' (default) or 'throughput' benchmark")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    normalizer = ScoreNormalizer(-0.46472087, 0.79736321)

    """
    BASIC SETTINGS
    """
    if args.device == "cpu":
        device = torch.device("cpu")
    
    if args.device == "cuda":
        torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")

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
    evaluator = COVER(**opt["model"]["args"]).to(device)
    state_dict = torch.load(opt["test_load_path"], map_location=device, weights_only=False)
    
    # set strict=False here to avoid error of missing
    # weight of prompt_learner in clip-iqa+, cross-gate
    evaluator.load_state_dict(state_dict['state_dict'], strict=False)
    evaluator.eval()

    """
    TESTING
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
                .to(device)
            )
    t2 = time.time()
    print("sample views: {:.3f} seconds".format(t2 - t1))

    semantic = views["semantic"]
    technical = views["technical"]
    aesthetic = views["aesthetic"]
    # torch.Size([1, 3, 20, 512, 512])
    # torch.Size([1, 3, 40, 224, 224])
    # torch.Size([1, 3, 40, 224, 224])

    if args.mode == "inference":
        results = [r.mean().item() for r in evaluator(semantic, technical, aesthetic)]
        pred_score = fuse_results(results)
        regularized = normalizer.normalize(pred_score["overall"])
        print(f"path\tsemantic score\ttechnical score\taesthetic score\toverall/final score\tregularized")
        print(f'{args.video_path.split("/")[-1]}\t{pred_score["semantic"]:4f}\t{pred_score["technical"]:4f}\t{pred_score["aesthetic"]:4f}\t{pred_score["overall"]:4f}\t{regularized:4f}')
    elif args.mode == "throughput":
        warmup = 10
        num_iterations = 100
        with torch.no_grad():
            for _ in range(warmup):
                _ = evaluator(semantic, technical, aesthetic)
    
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = evaluator(semantic, technical, aesthetic)
        torch.cuda.synchronize()
        end = time.time()

        elapsed = end - start

        # 统计总帧数（semantic: 20帧，technical/aesthetic: 40帧）
        semantic_frames = semantic.shape[0] * semantic.shape[2] * num_iterations
        technical_frames = technical.shape[0] * technical.shape[2] * num_iterations
        aesthetic_frames = aesthetic.shape[0] * aesthetic.shape[2] * num_iterations
        total_frames = semantic_frames + technical_frames + aesthetic_frames

        # 输出 throughput
        print(f"[Throughput Test] iterations={num_iterations}, warmup={warmup}")
        print(f"Total time: {elapsed:.3f} seconds")
        print(f"Semantic: {semantic_frames} frames")
        print(f"Technical: {technical_frames} frames")
        print(f"Aesthetic: {aesthetic_frames} frames")
        print(f"Total frames: {total_frames}")
        print(f"Throughput: {num_iterations / elapsed:.2f} samples/sec")

    # ## tuple_input=True 
    # evaluator1 = COVER(**opt["model"]["args"], tuple_input=True).to(device)
    # state_dict = torch.load(opt["test_load_path"], map_location=device, weights_only=False)
    
    # # set strict=False here to avoid error of missing
    # # weight of prompt_learner in clip-iqa+, cross-gate
    # evaluator1.load_state_dict(state_dict['state_dict'], strict=False)
    # evaluator1.eval()
    # results1 = [r.mean().item() for r in evaluator1(views)]
    # pred_score1 = fuse_results(results1)
    # print(f"path, semantic score, technical score, aesthetic score, overall/final score")
    # print(f'{args.video_path.split("/")[-1]},{pred_score1["semantic"]:4f},{pred_score1["technical"]:4f},{pred_score1["aesthetic"]:4f},{pred_score1["overall"]:4f}')
