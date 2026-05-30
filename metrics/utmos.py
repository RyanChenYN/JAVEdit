# coding=utf-8
"""
utmos.py - UTMOSv2 音频质量指标
基于 UTMOSv2 模型对视频中的音频进行 MOS (Mean Opinion Score) 评分。

用法 (单独使用):
    python utmos.py --video_dir <视频目录>

作为模块调用:
    from utmos import compute_utmos
    results = compute_utmos(video_dir, bench_csv, device='cuda:0', path_cfg=cfg)
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
import importlib
import numpy as np
import torch
import logging
from pathlib import Path
from collections import defaultdict

from javbench_utils import (
    load_yaml, collect_videos, build_task_map, assign_tasks,
    compute_statistics, compute_per_task_statistics, save_json
)

logger = logging.getLogger(__name__)


def _setup_paths(path_cfg):
    """UTMOSv2 paths (kept for interface compatibility)"""
    return path_cfg


def _extract_audio(video_path, output_wav_path):
    """用 ffmpeg 从视频中提取音频为 16kHz mono wav"""
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        output_wav_path, '-loglevel', 'error'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def compute_utmos(video_dir, bench_csv, device='cuda:0', num_gpus=1, path_cfg=None,
                  fold=0, **kwargs):
    """
    计算 UTMOSv2 音频质量指标。

    Args:
        video_dir: 编辑后视频目录 (包含 mp4 文件)
        bench_csv: benchmark CSV 路径
        device: 设备
        num_gpus: GPU 数量 (UTMOSv2 使用单GPU即可)
        path_cfg: utmos 路径配置 dict (从 path.yml 加载)
        fold: 模型 fold (0-4)

    Returns:
        tuple: (overall_stats, per_task_stats, results_list)
    """
    if path_cfg is None:
        cfg = load_yaml(os.path.join(os.path.dirname(__file__), 'path.yml'))
        path_cfg = cfg['utmos']

    _setup_paths(path_cfg)

    from third_party.utmosv2._settings import configure_defaults, configure_inference_args
    from third_party.utmosv2.runner import run_inference
    from third_party.utmosv2.utils import get_dataloader, get_dataset, get_model

    videos = collect_videos(video_dir)
    total = len(videos)
    logger.info(f'UTMOSv2 evaluation: {total} videos')

    # 提取音频到临时目录
    tmp_dir = tempfile.mkdtemp(prefix='utmos_')
    wav_paths = []
    video_names = []

    logger.info('Extracting audio from videos...')
    for video_path in videos:
        video_name = os.path.basename(video_path)
        wav_path = os.path.join(tmp_dir, Path(video_name).stem + '.wav')
        if _extract_audio(video_path, wav_path):
            wav_paths.append(wav_path)
            video_names.append(video_name)
        else:
            logger.warning(f'Failed to extract audio: {video_name}')

    logger.info(f'Successfully extracted {len(wav_paths)}/{total} audio files')

    if not wav_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {'mean': None, 'valid_count': 0, 'null_count': total}, {}, []

    # 加载 UTMOSv2 模型
    config_name = path_cfg.get('config_name', 'fusion_stage3')
    cfg_module = importlib.import_module(f'third_party.utmosv2.config.{config_name}')

    # 模拟 args 对象
    class FakeArgs:
        pass

    fake_args = FakeArgs()
    fake_args.config = config_name
    fake_args.fold = fold
    fake_args.seed = 42
    fake_args.input_dir = tmp_dir
    fake_args.input_path = None
    fake_args.out_path = None
    fake_args.num_workers = 0  # avoid DataLoader spawn pickling issues after other metrics
    fake_args.val_list_path = None
    fake_args.weight = None
    fake_args.predict_dataset = 'sarulab'
    fake_args.num_repetitions = 1
    fake_args.reproduce = False
    fake_args.final = False

    configure_inference_args(cfg_module, fake_args)
    configure_defaults(cfg_module)

    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    cfg_module.print_config = True
    cfg_module.now_fold = fold
    model = get_model(cfg_module, torch_device)
    cfg_module.print_config = False

    # 构建数据集和 dataloader
    import pandas as pd
    data = pd.DataFrame({
        'file_path': wav_paths,
        'mos': [0.0] * len(wav_paths),
        'sys_id': ['unknown'] * len(wav_paths),
        'predict_dataset': ['sarulab'] * len(wav_paths),
        'dataset': ['sarulab'] * len(wav_paths),
    })

    test_dataset = get_dataset(cfg_module, data, 'test')
    test_dataloader = get_dataloader(cfg_module, test_dataset, 'test')
    test_preds, _ = run_inference(cfg_module, model, test_dataloader, 0, data, torch_device)

    # 清理临时文件
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 整理结果
    results = []
    for i, video_name in enumerate(video_names):
        score = float(test_preds[i])
        results.append({
            'video': video_name,
            'utmos_score': round(score, 4),
        })

    # 添加 task 信息
    task_map = build_task_map(bench_csv)
    assign_tasks(results, task_map)

    # 统计
    overall_stats = compute_statistics(results, 'utmos_score',
                                       valid_fn=lambda x: x is not None)
    per_task_stats = compute_per_task_statistics(results, 'utmos_score',
                                                 valid_fn=lambda x: x is not None)

    logger.info(f'UTMOSv2 done: mean={overall_stats["mean"]}, '
                f'valid={overall_stats["valid_count"]}/{len(results)}')

    return overall_stats, per_task_stats, results


def parse_args():
    parser = argparse.ArgumentParser(description='UTMOSv2 音频质量评测')
    parser.add_argument('--video_dir', required=True, help='编辑后视频目录')
    parser.add_argument('--bench_csv', default=None, help='Benchmark CSV 路径')
    parser.add_argument('--fold', type=int, default=0, help='Model fold (0-4)')
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 路径')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    cfg = load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path.yml'))
    bench_csv = args.bench_csv or cfg['benchmark']['csv_path']

    overall_stats, per_task_stats, results = compute_utmos(
        video_dir=args.video_dir,
        bench_csv=bench_csv,
        path_cfg=cfg['utmos'],
        fold=args.fold,
    )

    output_path = args.output or os.path.join(args.video_dir, 'utmos_results.json')
    output_data = {
        'metric': 'utmos',
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
