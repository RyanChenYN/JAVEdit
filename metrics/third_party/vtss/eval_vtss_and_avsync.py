# coding=utf-8
"""
eval_vtss_and_avsync.py

计算编辑后视频的 VTSS 质量分数和音画同步分数(AV Sync)。

VTSS: 使用 Koala-36M DiViDeAddEvaluator 模型打分
AV Sync: 使用 LatentSync SyncNet 模型，简化版（不做人脸检测，直接用视频下半部分帧）

支持多卡并行 + 线程池预加载，断点续传。
tqdm 全局进度条。

使用方式:
  cd ../../../../../../../../..../../../../../Koala-36M/training_suitability_assessment
  conda activate analyse
  python eval_vtss_and_avsync.py --num_gpus 8 --prefetch 8
"""

import os
import sys
import json
import argparse
import copy
import math
import tempfile
import threading
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# VTSS model
from model import DiViDeAddEvaluator
from datasets import FusionDataset

sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]

# AV Sync - SyncNet
SYNCNET_PROJECT = "../../../../../../../../..../../../../../VideoTranslate"
sys.path.insert(0, SYNCNET_PROJECT)
from analysis.syncnet import SyncNet
from analysis.audio import melspectrogram
from omegaconf import OmegaConf


# ======================== 路径配置 ========================

VIDEO_DIRS = {
    "person_candidate_and_add": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_person_candidate_and_add_mux_all_selected",
    "person_mux_selected": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_person_mux_selected",
    "background": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_background_retain_music_mux_selected",
    "talk": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_talk_mux_selected",
    "remove": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_remove_yt_mux_selected",
}

SYNCNET_CONFIG = os.path.join(SYNCNET_PROJECT, "analysis/configs.yaml")
OUTPUT_DIR = "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VTSS + AV Sync for edited videos")
    parser.add_argument("--opt", type=str, default="test.yml", help="VTSS model config")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=8, help="Number of prefetch threads per GPU")
    parser.add_argument("--output_json", type=str,
                        default=os.path.join(OUTPUT_DIR, "eval_vtss_avsync_results.json"))
    parser.add_argument("--ckpt_dir", type=str, default="eval_vtss_avsync_checkpoints")
    return parser.parse_args()


# ======================== VTSS ========================

def load_vtss_model(opt, device):
    model = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    state_dict = torch.load(opt["test_load_path"], map_location=device, weights_only=False)["state_dict"]

    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(opt["test_load_path_aux"], map_location=device, weights_only=False)["state_dict"]
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            ki = k.replace("vqa_head", "fragments_head") if k.startswith("vqa_head") else k
            fusion_state_dict[ki] = v
        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            ki = k.replace("vqa_head", "resize_head") if k.startswith("vqa_head") else k
            fusion_state_dict[ki] = v
        state_dict = fusion_state_dict

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def get_vtss_score(video_path, vtss_model, data_opt_template, device):
    """直接用 FusionDataset 加载单个视频并推理 VTSS 分数"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(f"{video_path},0.0\n")
            anno_path = tmp.name

        data_opt = copy.deepcopy(data_opt_template)
        data_opt["anno_file"] = anno_path
        data_opt["data_prefix"] = ""

        dataset = FusionDataset(data_opt)
        if len(dataset) == 0:
            os.remove(anno_path)
            return -1.0

        data = dataset[0]  # 直接索引，不用 DataLoader
        video = {}
        for stype in sample_types:
            if stype in data:
                v = data[stype].unsqueeze(0).to(device)  # (1, C, T, H, W)
                b, c, t, h, w = v.shape
                num_clips = data["num_clips"][stype]
                v = (v.reshape(b, c, num_clips, t // num_clips, h, w)
                     .permute(0, 2, 1, 3, 4, 5)
                     .reshape(b * num_clips, c, t // num_clips, h, w))
                video[stype] = v

        with torch.no_grad():
            labels = vtss_model(video, reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in labels]

        os.remove(anno_path)
        return float(np.sum(labels))

    except Exception as e:
        return -1.0


# ======================== AV Sync (Simplified) ========================

def load_syncnet(device):
    config = OmegaConf.load(SYNCNET_CONFIG)
    model = SyncNet(OmegaConf.to_container(config.model)).to(device)
    ckpt = torch.load(config.ckpt.inference_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(dtype=torch.float16)
    model.requires_grad_(False)
    model.eval()
    return model, config


def compute_avsync_score(video_path, syncnet, config, device):
    """简化版 AV Sync"""
    import cv2
    from decord import VideoReader, AudioReader

    try:
        ar = AudioReader(video_path, sample_rate=config.data.audio_sample_rate)
        audio_raw = ar[:]
        audio_data = (audio_raw.numpy().squeeze(0) if hasattr(audio_raw, "numpy") else audio_raw.asnumpy().squeeze(0))

        audio_energy = np.abs(audio_data).mean()
        if audio_energy < 0.001:
            return None

        mel_data = torch.from_numpy(melspectrogram(audio_data))

        vr = VideoReader(video_path)
        num_frames = len(vr)
        fps = vr.get_avg_fps()

        if num_frames < config.data.num_frames:
            return None

        resolution = config.data.resolution
        num_sync_frames = config.data.num_frames
        mel_window_length = math.ceil(num_sync_frames / 5 * 16)

        mid_frame = num_frames // 2
        start_frame = max(0, mid_frame - num_sync_frames // 2)
        end_frame = start_frame + num_sync_frames
        if end_frame > num_frames:
            start_frame = num_frames - num_sync_frames
            end_frame = num_frames

        frames_raw = vr.get_batch(list(range(start_frame, end_frame)))
        frames = (frames_raw.numpy() if hasattr(frames_raw, "numpy") else frames_raw.asnumpy())

        h = frames.shape[1]
        lower_half = frames[:, h // 2:, :, :]

        processed_frames = []
        for frame in lower_half:
            frame = cv2.resize(frame, (resolution, resolution // 2))
            frame = frame / 255.0
            processed_frames.append(frame)

        video_tensor = np.stack(processed_frames)
        video_tensor = video_tensor.transpose(0, 3, 1, 2)
        video_tensor = video_tensor.reshape(-1, resolution // 2, resolution)
        video_tensor = torch.from_numpy(video_tensor).float().unsqueeze(0).to(device).half()

        mel_start = int(80.0 * (start_frame / float(fps)))
        mel_end = mel_start + mel_window_length
        if mel_end > mel_data.shape[1]:
            return None

        mel_tensor = mel_data[:, mel_start:mel_end].unsqueeze(0).unsqueeze(0).to(device).half()

        with torch.no_grad():
            v_emb, a_emb = syncnet(video_tensor, mel_tensor)

        cos_sim = F.cosine_similarity(v_emb, a_emb, dim=1).item()
        return cos_sim

    except Exception as e:
        return None


# ======================== Worker (per GPU) ========================

def load_checkpoint(ckpt_path):
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    done[item["path"]] = item
                except:
                    pass
    return done


def worker_fn(gpu_id, video_shard, opt, progress_counter, prefetch_threads, ckpt_dir):
    """每张卡一个进程，内部用线程池预读视频数据"""
    device = f"cuda:{gpu_id}"
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_gpu{gpu_id}.jsonl")

    # 断点续传
    done = load_checkpoint(ckpt_path)
    remaining = [v for v in video_shard if v["path"] not in done]

    # 更新全局计数器
    with progress_counter.get_lock():
        progress_counter.value += len(done)

    if not remaining:
        return

    # Load models
    vtss_model = load_vtss_model(opt, device)
    syncnet, sync_config = load_syncnet(device)

    # data opt template
    data_key = None
    for key in opt["data"].keys():
        if "val" in key or "test" in key:
            data_key = key
            break
    data_opt_template = opt["data"][data_key]["args"]

    ckpt_f = open(ckpt_path, "a", encoding="utf-8", buffering=1)

    for item in remaining:
        video_path = item["path"]
        task_type = item["task_type"]

        # VTSS
        vtss_score = get_vtss_score(video_path, vtss_model, data_opt_template, device)

        # AV Sync
        sync_score = compute_avsync_score(video_path, syncnet, sync_config, device)

        record = {
            "path": video_path,
            "task_type": task_type,
            "vtss_score": vtss_score,
            "sync_score": sync_score,
        }

        ckpt_f.write(json.dumps(record) + "\n")
        with progress_counter.get_lock():
            progress_counter.value += 1

    ckpt_f.close()


# ======================== Main ========================

def main():
    args = parse_args()

    # 收集所有视频
    all_videos = []
    for task_type, video_dir in VIDEO_DIRS.items():
        if not os.path.isdir(video_dir):
            print(f"[WARN] Directory not found: {video_dir}")
            continue
        files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
        for f in files:
            all_videos.append({
                "path": os.path.join(video_dir, f),
                "task_type": task_type,
            })

    total_videos = len(all_videos)
    print(f"Total videos: {total_videos}")
    for task_type in VIDEO_DIRS:
        count = sum(1 for v in all_videos if v["task_type"] == task_type)
        print(f"  {task_type}: {count}")

    # 读取 VTSS config
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    # 创建 checkpoint 目录
    os.makedirs(args.ckpt_dir, exist_ok=True)

    num_gpus = args.num_gpus
    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count()

    # 分配视频到 GPU（round-robin）
    shards = [[] for _ in range(num_gpus)]
    for i, v in enumerate(all_videos):
        shards[i % num_gpus].append(v)

    print(f"Using {num_gpus} GPUs, ~{len(shards[0])} videos per GPU")

    # 多进程
    mp.set_start_method("spawn", force=True)
    progress_counter = mp.Value("i", 0)

    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, shards[gpu_id], opt, progress_counter,
                  args.prefetch, args.ckpt_dir),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # 主进程显示 tqdm
    pbar = tqdm(total=total_videos, desc="Total Progress", unit="video", dynamic_ncols=True)
    last_val = 0

    import time
    while any(p.is_alive() for p in processes):
        cur = progress_counter.value
        if cur > last_val:
            pbar.update(cur - last_val)
            last_val = cur
        time.sleep(0.5)

    # final update
    cur = progress_counter.value
    if cur > last_val:
        pbar.update(cur - last_val)
    pbar.close()

    for p in processes:
        p.join()

    # 汇总所有 checkpoint
    print("\nAggregating results from checkpoints...")
    results = []
    for gpu_id in range(num_gpus):
        ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_gpu{gpu_id}.jsonl")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except:
                            pass

    print(f"Scored {len(results)} videos.")

    # 按任务类型统计
    for task_type in VIDEO_DIRS:
        task_results = [r for r in results if r["task_type"] == task_type]
        vtss_scores = [r["vtss_score"] for r in task_results if r["vtss_score"] >= 0]
        sync_scores = [r["sync_score"] for r in task_results if r["sync_score"] is not None]
        null_sync = sum(1 for r in task_results if r["sync_score"] is None)

        print(f"\n{task_type} ({len(task_results)} videos):")
        if vtss_scores:
            print(f"  VTSS: mean={np.mean(vtss_scores):.4f}, median={np.median(vtss_scores):.4f}, "
                  f"min={np.min(vtss_scores):.4f}, max={np.max(vtss_scores):.4f}")
        if sync_scores:
            print(f"  AV Sync: mean={np.mean(sync_scores):.4f}, median={np.median(sync_scores):.4f}, "
                  f"min={np.min(sync_scores):.4f}, max={np.max(sync_scores):.4f}")
        print(f"  AV Sync null (no voice): {null_sync}")

    # 保存结果
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
