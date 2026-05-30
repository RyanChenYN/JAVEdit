# coding=utf-8
"""
av_quality.py - Audio-visual quality metric.

Uses the Qwen3-Omni VLM to rate the overall AV quality of edited videos (1-5).

Standalone usage:
    conda activate qwen3omni
    python av_quality.py --video_dir <video_dir>

As a module:
    from av_quality import compute_av_quality
"""
import os
import sys
import csv
import re
import json
import argparse
import logging
from pathlib import Path

from javbench_utils import (
    load_yaml, collect_videos, build_task_map, assign_tasks,
    get_hash_from_filename, compute_statistics, compute_per_task_statistics, save_json
)

logger = logging.getLogger(__name__)
DEFAULT_QWEN_MODEL = str(
    Path(__file__).resolve().parents[6]
    / "huggingface/Qwen/Qwen3-Omni-30B-A3B-Thinking"
)


def _build_entries(video_dir, bench_csv):
    """Build the list of evaluation entries."""
    videos = collect_videos(video_dir)
    task_map = build_task_map(bench_csv)

    entries = []
    for video_path in videos:
        video_name = os.path.basename(video_path)
        video_hash = get_hash_from_filename(video_name)
        task = task_map.get(video_hash, 'unknown')
        entries.append({
            'tgt_path': video_path,
            'video_name': video_name,
            'task': task,
        })
    return entries


def compute_av_quality(video_dir, bench_csv, device='cuda:0', num_gpus=8, path_cfg=None, **kwargs):
    """
    Compute the AV quality metric.

    Args:
        video_dir: directory of edited videos
        bench_csv: benchmark CSV path
        device: device (unused; the model is managed by vLLM)
        num_gpus: number of GPUs (for tensor parallelism)
        path_cfg: config dict (from path.yml)

    Returns:
        tuple: (overall_stats, per_task_stats, results_list)
    """
    if path_cfg is None:
        cfg = load_yaml(os.path.join(os.path.dirname(__file__), 'path.yml'))
        path_cfg = cfg.get('qwen_judge', {})

    from qwen_judge import load_model, run_av_quality_eval

    model_path = path_cfg.get('model_path', DEFAULT_QWEN_MODEL)
    tp_size = path_cfg.get('tensor_parallel_size', 4)
    pp_size = path_cfg.get('pipeline_parallel_size', 2)
    batch_size = path_cfg.get('batch_size', 8)

    entries = _build_entries(video_dir, bench_csv)
    logger.info(f'AV Quality evaluation: {len(entries)} videos')

    llm, processor = load_model(model_path, tp_size, pp_size)
    results = run_av_quality_eval(entries, llm, processor, batch_size=batch_size)

    overall_stats = compute_statistics(results, 'av_quality',
                                       valid_fn=lambda x: x is not None)
    per_task_stats = compute_per_task_statistics(results, 'av_quality',
                                                 valid_fn=lambda x: x is not None)

    logger.info(f'AV Quality done: mean={overall_stats["mean"]}, '
                f'valid={overall_stats["valid_count"]}/{len(results)}')

    return overall_stats, per_task_stats, results


def parse_args():
    parser = argparse.ArgumentParser(description='AV quality evaluation (Qwen3-Omni)')
    parser.add_argument('--video_dir', required=True, help='Directory of edited videos')
    parser.add_argument('--bench_csv', default=None, help='Benchmark CSV path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    cfg = load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path.yml'))
    bench_csv = args.bench_csv or cfg['benchmark']['csv_path']

    path_cfg = cfg.get('qwen_judge', {})
    if args.batch_size:
        path_cfg['batch_size'] = args.batch_size

    overall_stats, per_task_stats, results = compute_av_quality(
        video_dir=args.video_dir,
        bench_csv=bench_csv,
        path_cfg=path_cfg,
    )

    output_path = args.output or os.path.join(args.video_dir, 'av_quality_results.json')
    output_data = {
        'metric': 'av_quality',
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
