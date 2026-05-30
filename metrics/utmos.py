# coding=utf-8
"""
utmos.py - UTMOSv2 audio quality metric.

Uses the UTMOSv2 model to predict a MOS (Mean Opinion Score) for video audio.

Standalone usage:
    python utmos.py --video_dir <video_dir>

As a module:
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
    """Extract audio from a video as 16kHz mono wav using ffmpeg."""
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
    Compute the UTMOSv2 audio quality metric.

    Args:
        video_dir: directory of edited videos (mp4 files)
        bench_csv: benchmark CSV path
        device: device
        num_gpus: number of GPUs (UTMOSv2 runs on a single GPU)
        path_cfg: utmos path config dict (loaded from path.yml)
        fold: model fold (0-4)

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

    # Extract audio into a temporary directory
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

    # Load the UTMOSv2 model
    config_name = path_cfg.get('config_name', 'fusion_stage3')
    cfg_module = importlib.import_module(f'third_party.utmosv2.config.{config_name}')

    # Stub args object
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

    # Build dataset and dataloader
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

    # Clean up temporary files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Collect results
    results = []
    for i, video_name in enumerate(video_names):
        score = float(test_preds[i])
        results.append({
            'video': video_name,
            'utmos_score': round(score, 4),
        })

    # Attach task labels
    task_map = build_task_map(bench_csv)
    assign_tasks(results, task_map)

    # Statistics
    overall_stats = compute_statistics(results, 'utmos_score',
                                       valid_fn=lambda x: x is not None)
    per_task_stats = compute_per_task_statistics(results, 'utmos_score',
                                                 valid_fn=lambda x: x is not None)

    logger.info(f'UTMOSv2 done: mean={overall_stats["mean"]}, '
                f'valid={overall_stats["valid_count"]}/{len(results)}')

    return overall_stats, per_task_stats, results


def parse_args():
    parser = argparse.ArgumentParser(description='UTMOSv2 audio quality evaluation')
    parser.add_argument('--video_dir', required=True, help='Directory of edited videos')
    parser.add_argument('--bench_csv', default=None, help='Benchmark CSV path')
    parser.add_argument('--fold', type=int, default=0, help='Model fold (0-4)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
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
