# coding=utf-8
"""
evaluate.py - JAVEdit metrics evaluation entry point.

Examples:
    # Run all metrics
    python evaluate.py --video_dir <video_dir> --num_gpus 8

    # Run selected metrics only
    python evaluate.py --video_dir <video_dir> --metric syncnet vtss

    # Specify output directory and benchmark CSV
    python evaluate.py --video_dir <video_dir> --bench_csv <csv> --output_path <out_dir>
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
        help="Path to benchmark_150.csv (default: read from path.yml)",
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

    if args.output_path is None:
        output_path = os.path.join(args.video_dir, 'eval_results')
    else:
        output_path = args.output_path

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

    all_results = bench.evaluate(
        video_dir=args.video_dir,
        bench_csv=args.bench_csv,
        name=args.name,
        metric_list=args.metric,
    )

    bench.print_summary(all_results)

    print('\nEvaluation completed successfully!')


if __name__ == "__main__":
    main()
