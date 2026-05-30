# coding=utf-8
"""
javbench.py - JAVEdit Metrics 主评测类
类似 IVEBench 的 VEBench，统一管理所有指标的评测流程。

作为模块调用:
    from javbench import JAVBench
    bench = JAVBench(device='cuda:0', output_path='./results', num_gpus=8)
    bench.evaluate(video_dir, bench_csv, metric_list=['syncnet', 'vtss', 'utmos'])
"""
import os
import json
import csv
import time
import datetime
import importlib
import numpy as np
import logging
from pathlib import Path

from javbench_utils import load_path_config, save_json, convert_types

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{timestamp}_javbench.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


class JAVBench(object):
    """JAVEdit 评测主类"""

    def __init__(self, device, output_path, num_gpus=8, path_yml=None):
        """
        Args:
            device: 默认设备 (如 'cuda:0')
            output_path: 输出目录
            num_gpus: 多 GPU 并行数量
            path_yml: path.yml 配置文件路径 (默认当前目录下)
        """
        self.device = device
        self.output_path = output_path
        self.num_gpus = num_gpus
        os.makedirs(self.output_path, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # 加载路径配置
        if path_yml is None:
            path_yml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path.yml')
        self.path_cfg = load_path_config(path_yml)

        self.logger.info(f"JAVBench initialized with device: {device}, num_gpus: {num_gpus}")
        self.logger.info(f"Output path: {output_path}")

    def build_full_metric_list(self):
        """返回所有支持的指标列表"""
        return ['syncnet', 'vtss', 'utmos', 'av_quality', 'instruction_compliance', 'video_fidelity']

    def evaluate(self, video_dir, bench_csv=None, name=None, metric_list=None):
        """
        执行评测。

        Args:
            video_dir: 编辑后视频目录
            bench_csv: benchmark CSV 路径 (为 None 时从 path.yml 读取)
            name: 输出文件名前缀
            metric_list: 指标列表，为 None 时评测所有指标

        Returns:
            dict: {metric_name: (overall_stats, per_task_stats, results)}
        """
        start_time = time.time()
        self.logger.info(f"Evaluation started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if bench_csv is None:
            bench_csv = self.path_cfg['benchmark']['csv_path']

        if metric_list is None:
            metric_list = self.build_full_metric_list()

        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        if not os.path.exists(bench_csv):
            raise FileNotFoundError(f"Benchmark CSV not found: {bench_csv}")

        self.logger.info(f"Video dir: {video_dir}")
        self.logger.info(f"Bench CSV: {bench_csv}")
        self.logger.info(f"Metrics to evaluate: {metric_list}")

        all_results = {}

        for metric in metric_list:
            try:
                metric_start = time.time()
                self.logger.info(f"{'=' * 40}")
                self.logger.info(f"Evaluating metric: {metric}")

                # 动态导入指标模块
                metric_module = importlib.import_module(metric)
                compute_fn = getattr(metric_module, f'compute_{metric}')

                # 获取该指标的路径配置
                metric_cfg_key = (
                    'qwen_judge'
                    if metric in {'av_quality', 'instruction_compliance', 'video_fidelity'}
                    else metric
                )
                metric_path_cfg = self.path_cfg.get(metric_cfg_key, {})

                # 调用计算函数
                overall_stats, per_task_stats, results = compute_fn(
                    video_dir=video_dir,
                    bench_csv=bench_csv,
                    device=self.device,
                    num_gpus=self.num_gpus,
                    path_cfg=metric_path_cfg,
                )

                all_results[metric] = (overall_stats, per_task_stats, results)

                metric_duration = time.time() - metric_start
                self.logger.info(f"Completed metric: {metric}, "
                               f"mean={overall_stats.get('mean')}, "
                               f"Time: {metric_duration:.2f}s")

            except Exception as e:
                self.logger.error(f'Error in metric {metric}: {e}', exc_info=True)
                all_results[metric] = ({}, {}, [])

        # 保存结果
        if name is None:
            name = 'javbench_eval'
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'{name}_{current_time}'

        self._save_results(all_results, video_dir, bench_csv, output_name)

        # 总结
        end_time = time.time()
        total_seconds = end_time - start_time
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

        self.logger.info("=" * 60)
        self.logger.info("Evaluation Finished.")
        self.logger.info(f"Total Evaluation Time: {time_str} ({total_seconds:.2f} seconds)")
        self.logger.info("=" * 60)

        return all_results

    def _save_results(self, all_results, video_dir, bench_csv, output_name):
        """保存评测结果为 JSON"""
        output_data = {
            'video_dir': video_dir,
            'bench_csv': bench_csv,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {},
        }

        for metric, (overall_stats, per_task_stats, results) in all_results.items():
            output_data['metrics'][metric] = {
                'overall': overall_stats,
                'per_task': per_task_stats,
                'results': results,
            }

        # 保存详细结果
        output_json = os.path.join(self.output_path, f'{output_name}_results.json')
        save_json(output_data, output_json)
        self.logger.info(f"Detailed results saved to: {output_json}")

        # 保存摘要 CSV
        self._save_summary_csv(all_results, output_name)

        return output_json

    def _save_summary_csv(self, all_results, output_name):
        """保存指标摘要为 CSV"""
        output_csv = os.path.join(self.output_path, f'{output_name}_summary.csv')

        rows = []
        for metric, (overall_stats, per_task_stats, results) in all_results.items():
            row = {'metric': metric}
            row['overall_mean'] = overall_stats.get('mean')
            row['overall_median'] = overall_stats.get('median')
            row['valid_count'] = overall_stats.get('valid_count')
            row['null_count'] = overall_stats.get('null_count')

            for task, stats in per_task_stats.items():
                row[f'{task}_mean'] = stats.get('mean')
            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            # 合并所有 row 的 keys
            for row in rows:
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)

            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            self.logger.info(f"Summary CSV saved to: {output_csv}")

    def print_summary(self, all_results):
        """打印评测摘要"""
        print("\n" + "=" * 60)
        print("JAVEdit Benchmark Evaluation Summary")
        print("=" * 60)

        for metric, (overall_stats, per_task_stats, results) in all_results.items():
            print(f"\n--- {metric.upper()} ---")
            mean = overall_stats.get('mean')
            median = overall_stats.get('median')
            valid = overall_stats.get('valid_count', 0)
            total = valid + overall_stats.get('null_count', 0)
            if mean is not None:
                print(f"  Overall: mean={mean:.4f}, median={median:.4f}, valid={valid}/{total}")
            else:
                print(f"  Overall: No valid scores")

            if 'pass_rate_0.2' in overall_stats:
                print(f"  Pass rate (>=0.2): {overall_stats['pass_rate_0.2']:.1%}")

            for task, stats in sorted(per_task_stats.items()):
                t_mean = stats.get('mean')
                t_count = stats.get('count', 0)
                if t_mean is not None:
                    print(f"    {task:20s}: mean={t_mean:.4f} (n={t_count})")
                else:
                    print(f"    {task:20s}: no valid scores (n={t_count})")

        print("=" * 60)
