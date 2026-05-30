# coding=utf-8
"""
syncnet.py - SyncNet audio-visual synchronization metric.

Uses LatentSync's SyncNet to compute AV sync confidence (conf) and offset (av_offset).

Standalone usage:
    python syncnet.py --video_dir <video_dir> --num_gpus 8

As a module:
    from syncnet import compute_syncnet
    results = compute_syncnet(video_dir, bench_csv, device='cuda:0', num_gpus=8, path_cfg=cfg)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import cv2
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
import logging

from javbench_utils import (
    load_yaml, collect_videos, build_task_map, assign_tasks,
    compute_statistics, compute_per_task_statistics, save_json
)

logger = logging.getLogger(__name__)


def _compute_single_video(video_path, align_instance, restorer, lipsync,
                          resize_tf, normalize_tf, device):
    """Compute the SyncNet score for a single video."""
    import subprocess as _sp
    from decord import VideoReader, AudioReader
    from third_party.lipsync.tensor_utils import calc_pdist_cos
    from third_party.lipsync.audio import melspectrogram

    tmp_video = None
    try:
        # Preprocess: bump low frame rate to 25fps and upscale low resolution to 720p
        _vr_check = VideoReader(video_path)
        _fps_check = _vr_check.get_avg_fps()
        _h_check = _vr_check[0].shape[0]
        del _vr_check
        needs_fix = (_fps_check < 20) or (_h_check < 640)
        if needs_fix:
            tmp_video = video_path + '.fixed.mp4'
            cmd = ['ffmpeg', '-y', '-i', video_path]
            if _fps_check < 20:
                cmd += ['-r', '25']
            if _h_check < 640:
                cmd += ['-vf', 'scale=-2:720']
            cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', tmp_video]
            _sp.run(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            video_path = tmp_video

        # Audio
        ar = AudioReader(video_path, sample_rate=lipsync.audio_sample_rate)
        audio_raw = ar[:]
        audio_data = (audio_raw.numpy().squeeze(0) if hasattr(audio_raw, 'numpy')
                      else audio_raw.asnumpy().squeeze(0))
        if np.abs(audio_data).mean() < 0.001:
            return None

        mel_data = torch.from_numpy(melspectrogram(audio_data))
        vr = VideoReader(video_path)
        num_frames = len(vr)
        num_sync_frames = lipsync.num_frames

        if num_frames < num_sync_frames + 4:
            return None

        face_list = []
        for idx in range(num_frames):
            frame_raw = vr[idx]
            frame = (frame_raw.numpy() if hasattr(frame_raw, 'numpy') else frame_raw.asnumpy())
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            landmark_list, score_list, bboxes_list = align_instance(frame_bgr)
            if len(landmark_list) > 0:
                landmark_2d_106 = landmark_list[0]
                lmk3_ = np.zeros((3, 2))
                lmk3_[0] = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
                lmk3_[1] = np.mean(landmark_2d_106[101:106], axis=0)
                lmk3_[2] = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)
                lmk3_ = np.round(lmk3_)
                face, _ = restorer.align_warp_face(
                    frame_bgr.copy(), lmks3=lmk3_, smooth=False, border_mode='constant')
                face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_CUBIC)
                face_list.append(face[:, :, ::-1])
            else:
                face_list.append(None)

        vision_embeds, audio_embeds = [], []
        for start_idx in range(num_frames - num_sync_frames + 1):
            window_faces = face_list[start_idx: start_idx + num_sync_frames]
            if any(f is None for f in window_faces):
                continue
            mel_tensor = lipsync.crop_audio_window(mel_data, start_idx)
            if mel_tensor is None or mel_tensor.shape[1] < lipsync.mel_window_length:
                continue
            mel_tensor = mel_tensor.half().to(device).unsqueeze(0)
            faces = torch.from_numpy(np.stack(window_faces).copy())
            faces = rearrange(faces, 'b h w c -> b c h w')
            faces = resize_tf(faces)
            faces = normalize_tf(faces / 255.0)
            h = faces.shape[2]
            faces = faces[:, :, h // 2:, :]
            faces = faces.contiguous().view(num_sync_frames * 3, h // 2, -1).to(device).half().unsqueeze(0)
            with torch.no_grad():
                v_emb, a_emb = lipsync.syncnet(faces, mel_tensor)
            vision_embeds.append(v_emb)
            audio_embeds.append(a_emb)

        if len(vision_embeds) < 3:
            return None

        vision_embeds = torch.cat(vision_embeds, dim=0)
        audio_embeds = torch.cat(audio_embeds, dim=0)
        dists = calc_pdist_cos(vision_embeds, audio_embeds, vshift=10)
        mean_dists = torch.mean(torch.stack(dists, 1), 1)
        min_dist, minidx = torch.min(mean_dists, 0)
        conf = (torch.median(mean_dists) - min_dist).item()
        av_offset = (10 - minidx).item()
        return {'conf': conf, 'av_offset': av_offset}
    except Exception as e:
        logger.debug(f'SyncNet error for {video_path}: {e}')
        return None
    finally:
        if tmp_video and os.path.exists(tmp_video):
            os.remove(tmp_video)


def _worker_fn(gpu_id, video_shard, path_cfg, output_dir, progress_counter):
    """Single-GPU worker process."""
    device = 'cuda:%d' % gpu_id

    # Ensure the metrics directory is on sys.path (needed in the subprocess)
    metrics_dir = os.path.dirname(os.path.abspath(__file__))
    if metrics_dir not in sys.path:
        sys.path.insert(0, metrics_dir)

    from third_party.lipsync import AlignImage, AlignRestore, LipSync

    align_instance = AlignImage(
        device=device,
        model_root=path_cfg.get('model_root', 'checkpoints/auxiliary'),
    )
    restorer = AlignRestore()
    lipsync = LipSync(device_id=gpu_id, config_path=path_cfg.get('config_path'))
    resize_tf = transforms.Resize((256, 256))
    normalize_tf = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    results = []
    for video_path in video_shard:
        video_name = os.path.basename(video_path)
        sync_result = _compute_single_video(
            video_path, align_instance, restorer, lipsync,
            resize_tf, normalize_tf, device)

        record = {'video': video_name}
        if sync_result is not None:
            record['sync_conf'] = sync_result['conf']
            record['av_offset'] = sync_result['av_offset']
        else:
            record['sync_conf'] = None
            record['av_offset'] = None
        results.append(record)

        with progress_counter.get_lock():
            progress_counter.value += 1

    gpu_output = os.path.join(output_dir, 'syncnet_gpu%d.jsonl' % gpu_id)
    with open(gpu_output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


def compute_syncnet(video_dir, bench_csv, device='cuda:0', num_gpus=8, path_cfg=None, **kwargs):
    """
    Compute the SyncNet audio-visual synchronization metric.

    Args:
        video_dir: directory of edited videos (mp4 files)
        bench_csv: benchmark CSV path
        device: device (base device when using multiple GPUs)
        num_gpus: number of GPUs to use
        path_cfg: syncnet path config dict (loaded from path.yml)

    Returns:
        tuple: (overall_stats, per_task_stats, results_list)
    """
    import time

    if path_cfg is None:
        cfg = load_yaml(os.path.join(os.path.dirname(__file__), 'path.yml'))
        path_cfg = cfg['syncnet']

    videos = collect_videos(video_dir)
    total = len(videos)
    logger.info(f'SyncNet evaluation: {total} videos, {num_gpus} GPUs')

    # Temporary output directory
    output_dir = os.path.join(video_dir, '.syncnet_tmp')
    os.makedirs(output_dir, exist_ok=True)

    # Shard videos across GPUs
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
            gpu_id, shards[gpu_id], path_cfg, output_dir, progress_counter))
        p.start()
        processes.append(p)

    pbar = tqdm(total=total, desc='SyncNet', unit='video', dynamic_ncols=True)
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

    # Merge results
    results = []
    for gpu_id in range(num_gpus):
        gpu_output = os.path.join(output_dir, 'syncnet_gpu%d.jsonl' % gpu_id)
        if os.path.exists(gpu_output):
            with open(gpu_output) as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except:
                            pass
            os.remove(gpu_output)

    # Clean up the temporary directory
    try:
        os.rmdir(output_dir)
    except:
        pass

    # Attach task labels
    task_map = build_task_map(bench_csv)
    assign_tasks(results, task_map)

    # Statistics
    overall_stats = compute_statistics(results, 'sync_conf',
                                       valid_fn=lambda x: x is not None)
    per_task_stats = compute_per_task_statistics(results, 'sync_conf',
                                                 valid_fn=lambda x: x is not None)

    # Extra: pass rate (conf >= 0.2)
    valid_confs = [r['sync_conf'] for r in results if r['sync_conf'] is not None]
    if valid_confs:
        pass_count = sum(1 for c in valid_confs if c >= 0.2)
        overall_stats['pass_rate_0.2'] = pass_count / len(valid_confs)
        overall_stats['pass_count_0.2'] = pass_count

    logger.info(f'SyncNet done: mean={overall_stats["mean"]}, '
                f'valid={overall_stats["valid_count"]}/{len(results)}')

    return overall_stats, per_task_stats, results


def parse_args():
    parser = argparse.ArgumentParser(description='SyncNet audio-visual sync evaluation')
    parser.add_argument('--video_dir', required=True, help='Directory of edited videos')
    parser.add_argument('--bench_csv', default=None, help='Benchmark CSV path')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    cfg = load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path.yml'))
    bench_csv = args.bench_csv or cfg['benchmark']['csv_path']

    overall_stats, per_task_stats, results = compute_syncnet(
        video_dir=args.video_dir,
        bench_csv=bench_csv,
        num_gpus=args.num_gpus,
        path_cfg=cfg['syncnet'],
    )

    output_path = args.output or os.path.join(args.video_dir, 'syncnet_results.json')
    output_data = {
        'metric': 'syncnet',
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
