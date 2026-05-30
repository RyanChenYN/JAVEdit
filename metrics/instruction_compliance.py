# coding=utf-8
"""
instruction_compliance.py - 指令遵循度指标
基于 Qwen3-Omni VLM 评估编辑结果是否符合指令要求 (1-5分)。

用法 (单独使用):
    conda activate qwen3omni
    python instruction_compliance.py --video_dir <视频目录>

作为模块调用:
    from instruction_compliance import compute_instruction_compliance
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
    load_yaml, collect_videos, build_task_map, get_hash_from_filename,
    compute_statistics, compute_per_task_statistics, save_json
)

logger = logging.getLogger(__name__)
DEFAULT_QWEN_MODEL = str(
    Path(__file__).resolve().parents[6]
    / "huggingface/Qwen/Qwen3-Omni-30B-A3B-Thinking"
)


def _build_entries(video_dir, bench_csv):
    """
    构建评测条目列表。
    需要匹配: 编辑后视频 -> benchmark CSV 中的源视频路径和 prompt。
    """
    videos = collect_videos(video_dir)

    # 从 CSV 构建 hash -> {src_path, prompt, task}
    hash_to_info = {}
    with open(bench_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            src_video = row['video'].strip()
            video_name = os.path.basename(src_video)
            video_hash = Path(video_name).stem
            hash_to_info[video_hash] = {
                'src_path': src_video,
                'prompt': row['prompt'].strip(),
                'task': row['task'].strip(),
            }
            if len(video_hash) >= 32:
                hash_to_info[video_hash[:32]] = hash_to_info[video_hash]

    entries = []
    for video_path in videos:
        video_name = os.path.basename(video_path)
        video_hash = get_hash_from_filename(video_name)
        info = hash_to_info.get(video_hash)
        if info is None:
            continue
        entries.append({
            'src_path': info['src_path'],
            'tgt_path': video_path,
            'prompt': info['prompt'],
            'task': info['task'],
            'video_name': os.path.basename(info['src_path']),
        })
    return entries


def compute_instruction_compliance(video_dir, bench_csv, device='cuda:0', num_gpus=8,
                                    path_cfg=None, **kwargs):
    """
    计算 Instruction Compliance 指令遵循度指标。

    Args:
        video_dir: 编辑后视频目录
        bench_csv: benchmark CSV 路径
        device: 设备 (未使用)
        num_gpus: GPU 数量
        path_cfg: 配置 dict

    Returns:
        tuple: (overall_stats, per_task_stats, results_list)
    """
    if path_cfg is None:
        cfg = load_yaml(os.path.join(os.path.dirname(__file__), 'path.yml'))
        path_cfg = cfg.get('qwen_judge', {})

    from qwen_judge import load_model, run_instruction_eval

    model_path = path_cfg.get('model_path', DEFAULT_QWEN_MODEL)
    tp_size = path_cfg.get('tensor_parallel_size', 4)
    pp_size = path_cfg.get('pipeline_parallel_size', 2)
    batch_size = path_cfg.get('batch_size', 8)

    entries = _build_entries(video_dir, bench_csv)
    logger.info(f'Instruction Compliance evaluation: {len(entries)} videos')

    llm, processor = load_model(model_path, tp_size, pp_size)
    results = run_instruction_eval(entries, llm, processor, batch_size=batch_size)

    # 只保留 instruction_compliance 分数 (video_fidelity 由另一个脚本负责)
    for r in results:
        r.pop('video_fidelity', None)

    overall_stats = compute_statistics(results, 'instruction_compliance',
                                       valid_fn=lambda x: x is not None)
    per_task_stats = compute_per_task_statistics(results, 'instruction_compliance',
                                                 valid_fn=lambda x: x is not None)

    logger.info(f'Instruction Compliance done: mean={overall_stats["mean"]}, '
                f'valid={overall_stats["valid_count"]}/{len(results)}')

    return overall_stats, per_task_stats, results


def parse_args():
    parser = argparse.ArgumentParser(description='Instruction Compliance 指令遵循度评测 (Qwen3-Omni)')
    parser.add_argument('--video_dir', required=True, help='编辑后视频目录')
    parser.add_argument('--bench_csv', default=None, help='Benchmark CSV 路径')
    parser.add_argument('--output', type=str, default=None)
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

    overall_stats, per_task_stats, results = compute_instruction_compliance(
        video_dir=args.video_dir,
        bench_csv=bench_csv,
        path_cfg=path_cfg,
    )

    output_path = args.output or os.path.join(args.video_dir, 'instruction_compliance_results.json')
    save_json({'metric': 'instruction_compliance', 'video_dir': args.video_dir,
               'total': len(results), 'overall': overall_stats,
               'per_task': per_task_stats, 'results': results}, output_path)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
