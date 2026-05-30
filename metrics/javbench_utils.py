# coding=utf-8
"""
javbench_utils.py - JAVEdit Metrics 工具函数
"""
import os
import csv
import json
import yaml
import numpy as np
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent


def load_yaml(path):
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_path(path, base_dir=None):
    """Resolve project-relative paths against the metrics directory."""
    if path is None or not isinstance(path, str) or not path:
        return path
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    base = Path(base_dir) if base_dir is not None else METRICS_DIR
    return str((base / path_obj).resolve())


def resolve_config_paths(cfg, base_dir=None):
    """Resolve known path fields in a nested config dict."""
    path_keys = {
        'path', 'config', 'checkpoint', 'csv_path', 'model_path',
        'config_path', 'det_path', 'p1_path', 'p2_path', 'pts217_path',
        'inference_ckpt_path',
    }
    if isinstance(cfg, dict):
        resolved = {}
        for key, value in cfg.items():
            if isinstance(value, str) and (key in path_keys or key.endswith('_path')):
                resolved[key] = resolve_path(value, base_dir)
            else:
                resolved[key] = resolve_config_paths(value, base_dir)
        return resolved
    if isinstance(cfg, list):
        return [resolve_config_paths(item, base_dir) for item in cfg]
    return cfg


def load_path_config(path=None):
    """Load path.yml and resolve relative paths from the config file directory."""
    if path is None:
        path = METRICS_DIR / 'path.yml'
    path = Path(path).resolve()
    return resolve_config_paths(load_yaml(path), path.parent)


def load_json(path):
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    """保存 JSON 文件，处理 numpy 类型"""
    converted_data = convert_types(data)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)


def convert_types(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_types(item) for item in obj)
    else:
        return obj


def get_hash_from_filename(filename):
    """
    从文件名中提取 32 位 hash 值。
    支持多种命名格式:
      - LTX: 0a16d2122d7b9e9fe6c2a6aa65b658e6_0_edited_121_with_audio.mp4
      - AVI-Edit: 0a16d2122d7b9e9fe6c2a6aa65b658e6_0.mp4
      - Kiwi-Edit: 0a16d2122d7b9e9fe6c2a6aa65b658e6.mp4
    """
    stem = Path(filename).stem
    candidate = stem[:32]
    if len(candidate) == 32 and all(c in '0123456789abcdefABCDEF' for c in candidate):
        return candidate
    return stem.split('_')[0]


def build_task_map(bench_csv):
    """
    从 benchmark CSV 构建 hash -> task 的映射。
    
    Args:
        bench_csv: benchmark_150_v2.csv 的路径
    
    Returns:
        dict: {hash: task_name}
    """
    task_map = {}
    with open(bench_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            video_path = row['video'].strip()
            task = row['task'].strip()
            video_name = os.path.basename(video_path)
            video_hash = Path(video_name).stem
            task_map[video_hash] = task
            if len(video_hash) >= 32:
                task_map[video_hash[:32]] = task
    return task_map


def collect_videos(video_dir):
    """
    收集目录下所有 mp4 视频文件路径。
    
    Args:
        video_dir: 视频目录
    
    Returns:
        list: 排序后的视频完整路径列表
    """
    videos = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith('.mp4')
    ])
    return videos


def assign_tasks(results, task_map):
    """
    给每条结果记录添加 task 字段。
    
    Args:
        results: list of dict, 每条记录需有 'video' 字段(文件名)
        task_map: hash -> task 映射
    
    Returns:
        results (in-place 修改)
    """
    for r in results:
        video_name = os.path.splitext(r['video'])[0]
        video_hash = get_hash_from_filename(r['video'])
        r['task'] = task_map.get(video_hash, 'unknown')
    return results


def compute_statistics(results, score_key, valid_fn=None):
    """
    计算指标统计信息。
    
    Args:
        results: list of dict
        score_key: 分数字段名
        valid_fn: 判断分数是否有效的函数, 默认 lambda x: x is not None and x >= 0
    
    Returns:
        dict: 包含 mean, median, min, max, valid_count, null_count
    """
    if valid_fn is None:
        valid_fn = lambda x: x is not None and x >= 0

    scores = [r[score_key] for r in results if valid_fn(r.get(score_key))]
    null_count = len(results) - len(scores)

    if scores:
        return {
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'std': float(np.std(scores)),
            'valid_count': len(scores),
            'null_count': null_count,
        }
    else:
        return {
            'mean': None,
            'median': None,
            'min': None,
            'max': None,
            'std': None,
            'valid_count': 0,
            'null_count': null_count,
        }


def compute_per_task_statistics(results, score_key, valid_fn=None):
    """
    按任务分类计算统计信息。
    
    Args:
        results: list of dict, 每条需有 'task' 字段
        score_key: 分数字段名
        valid_fn: 同 compute_statistics
    
    Returns:
        dict: {task_name: statistics_dict}
    """
    tasks = sorted(set(r.get('task', 'unknown') for r in results))
    per_task = {}
    for task in tasks:
        task_results = [r for r in results if r.get('task') == task]
        stats = compute_statistics(task_results, score_key, valid_fn)
        stats['count'] = len(task_results)
        per_task[task] = stats
    return per_task
