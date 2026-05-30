# coding=utf-8
"""
evaluate.py - JAVEdit Metrics 总评测入口
统一调用 syncnet, vtss, utmos 指标进行评测。

用法:
    # 运行所有指标
    python evaluate.py --video_dir <视频目录> --num_gpus 8

    # 只运行指定指标
    python evaluate.py --video_dir <视频目录> --metric syncnet vtss

    # 指定输出目录和 benchmark CSV
    python evaluate.py --video_dir <视频目录> --bench_csv <csv路径> --output_path <输出目录>
"""
import torch
import os
import argparse
from datetime import datetime
from javbench import JAVBench


def parse_args():
    parser = argparse.ArgumentParser(
        description='JAVEdit Benchmark - Video Editing Quality Evaluation',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing edited videos (mp4 files)",
    )

    parser.add_argument(
        "--bench_csv",
        type=str,
        default=None,
        help="Path to benchmark_150_v2.csv (default: read from path.yml)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory for results (default: <video_dir>/eval_results/)",
    )

    parser.add_argument(
        "--metric",
        nargs='+',
        default=None,
        choices=['syncnet', 'vtss', 'utmos', 'av_quality', 'instruction_compliance', 'video_fidelity'],
        help="Metrics to evaluate. Default: all (syncnet, vtss, utmos, av_quality, instruction_compliance, video_fidelity)\n"
             "Usage: --metric syncnet vtss",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs for parallel evaluation (default: 8)",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="javbench",
        help="Name prefix for output files (default: javbench)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f'Arguments: {args}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出目录
    if args.output_path is None:
        output_path = os.path.join(args.video_dir, 'eval_results')
    else:
        output_path = args.output_path

    # 创建评测实例
    bench = JAVBench(
        device=device,
        output_path=output_path,
        num_gpus=args.num_gpus,
    )

    print(f'Starting JAVEdit evaluation on device: {device}')
    print(f'Video dir: {args.video_dir}')
    print(f'Output path: {output_path}')
    print(f'Metrics: {args.metric or "all"}')
    print(f'Num GPUs: {args.num_gpus}')
    print()

    # 执行评测
    all_results = bench.evaluate(
        video_dir=args.video_dir,
        bench_csv=args.bench_csv,
        name=args.name,
        metric_list=args.metric,
    )

    # 打印摘要
    bench.print_summary(all_results)

    print('\nEvaluation completed successfully!')


if __name__ == "__main__":
    main()
