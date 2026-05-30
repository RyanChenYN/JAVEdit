"""
针对 VIDGEN 数据集的质量评估流程：
  1. 从 result.json 读取视频路径列表（由 select_720p_25fps.py 生成）
  2. 用 Koala-36M Training Suitability Assessment 模型对每个视频打分（多卡并行）
  3. 过滤 score >= threshold 的视频
  4. 输出结果到单个 VIDGEN_300k_vtss.json

支持断点续传：每张卡的打分结果实时写入 checkpoint_gpu{id}.jsonl，
重启后自动跳过已打分的视频。

用法（自动使用所有可用 GPU）：
  python save_VIDGEN_json.py \
    -i ../../../../../../../../datasets/Fudan-FUXI/VIDGEN-1M/VIDGEN_720p_25fps_sync/result.json \
    -o VIDGEN_300k_vtss.json \
    --num_gpus 8
"""

import os
import json
import argparse
import copy
import numpy as np
import yaml
import csv
import tempfile
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import OrderedDict
from model import DiViDeAddEvaluator
from datasets import FusionDataset

sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


def load_model(opt, device):
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


def load_checkpoint(ckpt_path):
    """从 checkpoint JSONL 文件加载已完成的 {path: score}"""
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    done[item["path"]] = item["score"]
                except Exception:
                    pass
    return done


def worker_fn(gpu_id, video_shard, opt, result_list, num_workers, ckpt_dir):
    """
    子进程：在指定 GPU 上对 video_shard 打分。
    - 结果实时追加写入 ckpt_dir/checkpoint_gpu{gpu_id}.jsonl（断点续传）
    - 最终结果写入 result_list（Manager.list）供主进程汇总

    注意：FusionDataset 按照 anno_file 的行顺序加载，DataLoader shuffle=False，
    因此第 i 条数据对应 remaining[i]，可安全用索引追踪路径。
    """
    device = f"cuda:{gpu_id}"
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_gpu{gpu_id}.jsonl")

    # --- 断点续传：加载已完成的结果 ---
    done = load_checkpoint(ckpt_path)
    for path, score in done.items():
        result_list.append((path, score))

    # 过滤掉已完成的视频
    remaining = [v for v in video_shard if v not in done]
    already_done = len(done)
    total = len(video_shard)

    if not remaining:
        print(f"[GPU {gpu_id}] All {total} videos already done (checkpoint). Skipping.")
        return

    print(f"[GPU {gpu_id}] {already_done} done (from checkpoint), {len(remaining)} remaining, {total} total.")

    # 写临时 anno 文件（只包含 remaining 的视频）
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        for vf in remaining:
            tmp.write(f"{vf},0.0\n")
        anno_path = tmp.name

    try:
        model = load_model(opt, device)

        # 找到 test/val 配置
        data_key = None
        for key in opt["data"].keys():
            if "val" in key or "test" in key:
                data_key = key
                break
        if data_key is None:
            raise RuntimeError("No val/test key found in opt['data']")

        data_opt = copy.deepcopy(opt["data"][data_key]["args"])
        data_opt["anno_file"] = anno_path
        data_opt["data_prefix"] = ""  # 路径已是绝对路径

        # 打开 checkpoint 文件用于追加
        ckpt_f = open(ckpt_path, "a", encoding="utf-8", buffering=1)

        pbar = tqdm(
            total=total,
            desc=f"GPU {gpu_id}",
            position=gpu_id,
            initial=already_done,
            leave=True,
        )

        def make_loader(start_idx):
            """从 remaining[start_idx:] 创建新的 DataLoader 和迭代器"""
            sub_videos = remaining[start_idx:]
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp_sub:
                for vf in sub_videos:
                    tmp_sub.write(f"{vf},0.0\n")
                sub_anno_path = tmp_sub.name
            sub_data_opt = copy.deepcopy(data_opt)
            sub_data_opt["anno_file"] = sub_anno_path
            sub_dataset = FusionDataset(sub_data_opt)
            sub_loader = torch.utils.data.DataLoader(
                sub_dataset,
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
                timeout=8,
            )
            return sub_loader, iter(sub_loader), sub_anno_path

        # 初始化第一个 loader
        data_opt["anno_file"] = anno_path
        loader, loader_iter, current_anno = make_loader(0)

        idx = 0
        while idx < len(remaining):
            try:
                data = next(loader_iter)
            except StopIteration:
                break
            except Exception as e:
                # DataLoader worker 崩溃后迭代器已损坏，需重建
                err_msg = str(e)
                video_path = remaining[idx]
                print(f"\n[GPU {gpu_id}] 跳过视频 {video_path}，DataLoader 错误: {err_msg[:200]}")
                ckpt_f.write(json.dumps({"path": video_path, "score": -1.0}) + "\n")
                result_list.append((video_path, -1.0))
                idx += 1
                pbar.update(1)
                # 清理旧 anno 临时文件
                if current_anno and current_anno != anno_path and os.path.exists(current_anno):
                    try:
                        os.remove(current_anno)
                    except Exception:
                        pass
                if idx < len(remaining):
                    # 重建 DataLoader，从下一个视频开始
                    loader, loader_iter, current_anno = make_loader(idx)
                    loader_start_idx = idx
                continue

            video = {}
            for stype in sample_types:
                if stype in data:
                    video[stype] = data[stype].to(device)
                    b, c, t, h, w = video[stype].shape
                    video[stype] = (
                        video[stype]
                        .reshape(b, c, data["num_clips"][stype], t // data["num_clips"][stype], h, w)
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(b * data["num_clips"][stype], c, t // data["num_clips"][stype], h, w)
                    )

            with torch.no_grad():
                labels = model(video, reduce_scores=False)
                labels = [np.mean(l.cpu().numpy()) for l in labels]

            final_score = float(np.sum(labels))
            video_path = remaining[idx]

            # 实时写 checkpoint
            ckpt_f.write(json.dumps({"path": video_path, "score": final_score}) + "\n")
            result_list.append((video_path, final_score))
            idx += 1
            pbar.update(1)

        pbar.close()
        ckpt_f.close()
        # 清理最后一个 sub anno 临时文件
        if current_anno and current_anno != anno_path and os.path.exists(current_anno):
            try:
                os.remove(current_anno)
            except Exception:
                pass

    finally:
        if os.path.exists(anno_path):
            os.remove(anno_path)


def main():
    parser = argparse.ArgumentParser(description="Score VIDGEN videos and save qualified ones to JSON.")
    parser.add_argument(
        "-i", "--input_json", type=str,
        default="../../../../../../../../datasets/Fudan-FUXI/VIDGEN-1M/VIDGEN_720p_25fps/result_300k_1m.json",
        help="Path to result.json containing filtered video paths."
    )
    parser.add_argument(
        "-o", "--output_json", type=str,
        default="VIDGEN_300k_1m_vtss.json",
        help="Output JSON file path."
    )
    parser.add_argument(
        "--opt", type=str, default="test.yml",
        help="Path to the model config YAML file."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.06,
        help="Score threshold for filtering (default: 0.06)."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=0,
        help="Number of GPUs to use. 0 = auto-detect all available GPUs."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader num_workers per GPU process (default: 4)."
    )
    parser.add_argument(
        "--scores_csv", type=str, default="",
        help="Optional: save all scores to this CSV path."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="vidgen_checkpoints",
        help="Directory to store per-GPU checkpoint JSONL files for resume."
    )

    args = parser.parse_args()

    # 1. 读取 result.json
    print(f"Loading video paths from: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        video_files = json.load(f)
    print(f"Total videos to score: {len(video_files)}")

    # 2. 读取配置
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    # 3. 确定 GPU 数量并切分
    num_gpus = args.num_gpus
    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
        print("Warning: No CUDA GPU detected, falling back to CPU (single process).")

    print(f"Using {num_gpus} GPU(s).")

    # 创建 checkpoint 目录
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # 将视频列表均匀切分给各 GPU
    shards = [[] for _ in range(num_gpus)]
    for i, vf in enumerate(video_files):
        shards[i % num_gpus].append(vf)

    # 4. 多进程打分
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_list = manager.list()

    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, shards[gpu_id], opt, result_list, args.num_workers, ckpt_dir),
            daemon=False,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 5. 汇总
    scores = dict(result_list)
    print(f"Scored {len(scores)} videos.")

    # 可选：保存中间打分 CSV
    if args.scores_csv:
        with open(args.scores_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "score"])
            for path, score in sorted(scores.items()):
                writer.writerow([path, score])
        print(f"All scores saved to: {args.scores_csv}")

    # 6. 按阈值过滤（排除 score=-1 的异常视频）
    error_count = sum(1 for s in scores.values() if s < 0)
    if error_count > 0:
        print(f"Skipped {error_count} videos due to loading errors.")
    passed = [
        {"path": path, "score": score}
        for path, score in scores.items()
        if score >= args.threshold
    ]
    passed.sort(key=lambda x: x["path"])
    print(f"Passed threshold {args.threshold}: {len(passed)} / {len(scores)} videos.")

    # 7. 保存结果
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    print(f"Done! Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
