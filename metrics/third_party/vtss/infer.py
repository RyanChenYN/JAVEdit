import torch
import cv2
import random
import os
import os.path as osp
import glob
from model import DiViDeAddEvaluator
from datasets import FusionDataset

import argparse
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import yaml
import csv
from thop import profile

sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]

def predict_folder(inf_loader, model, device, output_file):
    print(f"Start scoring videos...")
    results = []
    keys = []

    for i, data in enumerate(tqdm(inf_loader, desc="Predicting")):
        result = dict()
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
        
        with torch.no_grad():
            labels = model(video, reduce_scores=False)
            # 获取各个分支的预测分数并求平均
            labels = [np.mean(l.cpu().numpy()) for l in labels]
            result["pr_labels"] = labels
        
        # 获取视频名称
        video_name = data["name"][0]
        if isinstance(video_name, list) or isinstance(video_name, tuple):
            video_name = video_name[0]
            
        result["name"] = os.path.basename(video_name)
        results.append(result)

    # 将结果写入 CSV 文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_name", "score"])
        for r in results:
            # 原代码中对不同分支(resize, fragments等)的分数进行了标准化后相加
            # 在没有GT的纯推理场景下，我们直接将各分支的原始分数相加作为最终得分
            final_score = np.sum(r["pr_labels"])
            writer.writerow([r["name"], final_score])
            
    print(f"Done! Scores saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Directory containing mp4 videos"
    )
    parser.add_argument(
        "-o", "--opt", type=str, default="test.yml", help="the option file"
    )
    parser.add_argument(
        "-t", "--output", type=str, default="scores.csv", help="the output csv file"
    )

    args = parser.parse_args()
    
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)

    # 加上 weights_only=False 解决之前的报错
    state_dict = torch.load(opt["test_load_path"], map_location=device, weights_only=False)["state_dict"]
    
    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(opt["test_load_path_aux"], map_location=device, weights_only=False)["state_dict"]
        
        from collections import OrderedDict
        
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v
            
        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        
        state_dict = fusion_state_dict
        
    model.load_state_dict(state_dict, strict=True)
    model.eval() # 确保模型处于推理模式
    
    # 获取输入文件夹下的所有 mp4 文件
    video_files = glob.glob(os.path.join(args.input, "*.mp4"))
    if len(video_files) == 0:
        print(f"No .mp4 files found in {args.input}")
        return

    # 生成一个临时的伪标注文件，供 FusionDataset 读取
    dummy_anno_path = "temp_dummy_anno.txt"
    with open(dummy_anno_path, "w", encoding="utf-8") as f:
        for vf in video_files:
            # 写入相对路径或绝对路径，并附带一个假的分数 0.0
            f.write(f"{os.path.basename(vf)},0.0\n")

    # 找到配置文件中的测试集配置，并覆盖其数据路径和标注文件路径
    for key in opt["data"].keys(): 
        if "val" in key or "test" in key:
            # 覆盖配置文件中的路径
            opt["data"][key]["args"]["anno_file"] = dummy_anno_path
            opt["data"][key]["args"]["data_prefix"] = args.input
            
            val_dataset = FusionDataset(opt["data"][key]["args"])

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=opt.get("num_workers", 4), pin_memory=True,
            )

            predict_folder(
                val_loader,
                model,
                device, 
                args.output
            )
            break # 只跑一个配置即可

    # 清理临时文件
    if os.path.exists(dummy_anno_path):
        os.remove(dummy_anno_path)

if __name__ == "__main__":
    main()