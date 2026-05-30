# coding=utf-8
"""
vtss.py - VTSS 视频质量指标 (Video Training Suitability Score)
基于 Koala-36M DiViDeAddEvaluator 计算视频训练适用性分数。

用法 (单独使用):
    python vtss.py --video_dir <视频目录> --num_gpus 8

作为模块调用:
    from vtss import compute_vtss
    results = compute_vtss(video_dir, bench_csv, device='cuda:0', num_gpus=8, path_cfg=cfg)
"""
import os
import sys
import json
import copy
import argparse
import tempfile
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm
from collections import OrderedDict
import logging

from javbench_utils import (
    load_yaml, collect_videos, build_task_map, assign_tasks,
    compute_statistics, compute_per_task_statistics, save_json
)

logger = logging.getLogger(__name__)

SAMPLE_TYPES = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


def _setup_paths(path_cfg):
    """VTSS paths (kept for interface compatibility)"""
    return path_cfg


def _load_vtss_model(vtss_opt, device, checkpoint_path=None):
    """加载 VTSS 模型"""
    from third_party.vtss.model import DiViDeAddEvaluator

    model = DiViDeAddEvaluator(**vtss_opt["model"]["args"]).to(device)

    load_path = checkpoint_path or vtss_opt.get("test_load_path", vtss_opt.get("load_path"))
    state_dict = torch.load(load_path, map_location=device, weights_only=False)["state_dict"]

    if "test_load_path_aux" in vtss_opt:
        aux_state_dict = torch.load(
            vtss_opt["test_load_path_aux"], map_location=device, weights_only=False)["state_dict"]
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


def _compute_single_video(video_path, vtss_model, data_opt_template, device):
    """计算单个视频的 VTSS 分数"""
    from third_party.vtss.datasets import FusionDataset

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write('%s,0.0\n' % video_path)
            anno_path = tmp.name

        data_opt = copy.deepcopy(data_opt_template)
        data_opt['anno_file'] = anno_path
        data_opt['data_prefix'] = ''
        dataset = FusionDataset(data_opt)

        if len(dataset) == 0:
            os.remove(anno_path)
            return -1.0

        data = dataset[0]
        video = {}
        for stype in SAMPLE_TYPES:
            if stype in data:
                v = data[stype].unsqueeze(0).to(device)
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
        logger.debug(f'VTSS error for {video_path}: {e}')
        return -1.0


def _worker_fn(gpu_id, video_shard, path_cfg, vtss_opt, data_opt_template,
               output_dir, progress_counter):
    """单 GPU Worker 进程"""
    device = 'cuda:%d' % gpu_id
    _setup_paths(path_cfg)

    vtss_model = _load_vtss_model(vtss_opt, device, path_cfg.get('checkpoint'))

    results = []
    for video_path in video_shard:
        video_name = os.path.basename(video_path)
        vtss_score = _compute_single_video(video_path, vtss_model, data_opt_template, device)
        record = {'video': video_name, 'vtss_score': vtss_score}
        results.append(record)

        with progress_counter.get_lock():
            progress_counter.value += 1

    gpu_output = os.path.join(output_dir, 'vtss_gpu%d.jsonl' % gpu_id)
    with open(gpu_output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


def compute_vtss(video_dir, bench_csv, device='cuda:0', num_gpus=8, path_cfg=None, **kwargs):
    """
    计算 VTSS 视频质量指标。

    Args:
        video_dir: 编辑后视频目录 (包含 mp4 文件)
        bench_csv: benchmark CSV 路径
        device: 设备
        num_gpus: 使用的 GPU 数量
        path_cfg: vtss 路径配置 dict (从 path.yml 加载)

    Returns:
        tuple: (overall_stats, per_task_stats, results_list)
    """
    import time

    if path_cfg is None:
        cfg = load_yaml(os.path.join(os.path.dirname(__file__), 'path.yml'))
        path_cfg = cfg['vtss']

    _setup_paths(path_cfg)

    # 加载 VTSS 配置
    vtss_config_path = path_cfg['config']
    with open(vtss_config_path) as f:
        vtss_opt = yaml.safe_load(f)

    # 找到测试数据配置
    data_key = None
    for key in vtss_opt['data'].keys():
        if 'val' in key or 'test' in key:
            data_key = key
            break
    data_opt_template = vtss_opt['data'][data_key]['args']

    videos = collect_videos(video_dir)
    total = len(videos)
    logger.info(f'VTSS evaluation: {total} videos, {num_gpus} GPUs')

    # 临时输出目录
    output_dir = os.path.join(video_dir, '.vtss_tmp')
    os.makedirs(output_dir, exist_ok=True)

    # 分配 GPU
    shards = [[] for _ in range(num_gpus)]
    for i, v in enumerate(videos):
        shards[i % num_gpus].append(v)

    mp.set_start_method('spawn', force=True)
    progress_counter = mp.Value('i', 0)

    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=_worker_fn, args=(
            gpu_id, shards[gpu_id], path_cfg, vtss_opt, data_opt_template,
            output_dir, progress_counter))
        p.start()
        processes.append(p)

    # 进度条
    pbar = tqdm(total=total, desc='VTSS', unit='video', dynamic_ncols=True)
    last_val = 0
    while any(p.is_alive() for p in processes):
        cur = progress_counter.value
        if cur > last_val:
            pbar.update(cur - last_val)
            last_val = cur
        time.sleep(0.5)
    cur = progress_counter.value
    if cur > last_val:
        pbar.update(cur - last_val)
    pbar.close()
    for p in processes:
        p.join()

    # 汇总结果
    results = []
    for gpu_id in range(num_gpus):
        gpu_output = os.path.join(output_dir, 'vtss_gpu%d.jsonl' % gpu_id)
        if os.path.exists(gpu_output):
            with open(gpu_output) as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except:
                            pass
            os.remove(gpu_output)

    # 清理临时目录
    try:
        os.rmdir(output_dir)
    except:
        pass

    # 添加 task 信息
    task_map = build_task_map(bench_csv)
    assign_tasks(results, task_map)

    # 统计 (vtss_score >= 0 为有效)
    overall_stats = compute_statistics(results, 'vtss_score',
                                       valid_fn=lambda x: x is not None and x >= 0)
    per_task_stats = compute_per_task_statistics(results, 'vtss_score',
                                                 valid_fn=lambda x: x is not None and x >= 0)

    logger.info(f'VTSS done: mean={overall_stats["mean"]}, '
                f'valid={overall_stats["valid_count"]}/{len(results)}')

    return overall_stats, per_task_stats, results


def parse_args():
    parser = argparse.ArgumentParser(description='VTSS 视频质量评测')
    parser.add_argument('--video_dir', required=True, help='编辑后视频目录')
    parser.add_argument('--bench_csv', default=None, help='Benchmark CSV 路径')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 路径')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    cfg = load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path.yml'))
    bench_csv = args.bench_csv or cfg['benchmark']['csv_path']

    overall_stats, per_task_stats, results = compute_vtss(
        video_dir=args.video_dir,
        bench_csv=bench_csv,
        num_gpus=args.num_gpus,
        path_cfg=cfg['vtss'],
    )

    output_path = args.output or os.path.join(args.video_dir, 'vtss_results.json')
    output_data = {
        'metric': 'vtss',
        'video_dir': args.video_dir,
        'total': len(results),
        'overall': overall_stats,
        'per_task': per_task_stats,
        'results': results,
    }
    save_json(output_data, output_path)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
