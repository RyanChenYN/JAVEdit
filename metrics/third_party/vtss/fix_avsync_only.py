# coding=utf-8
"""
fix_avsync_only.py - 用 LatentSync 原版算法计算 sync conf (采样版: 每任务1000个)
"""
import os, sys, json, math, numpy as np, torch, torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
import cv2
from einops import rearrange
from torchvision import transforms

sys.path.insert(0, "../../../../../../../../..../../../../../VideoTranslate")
from face_align.align import AlignImage
from analysis.affine_transform_func import AlignRestore
from analysis.lipsync import LipSync
from analysis.tensor_utils import calc_pdist_cos
from analysis.audio import melspectrogram

VT_ROOT = "../../../../../../../../..../../../../../VideoTranslate"
CKPT_DIR = "../../../../../../../../..../../../../../Koala-36M/training_suitability_assessment/eval_vtss_avsync_checkpoints"
OUTPUT_JSON = "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/eval_vtss_avsync_results.json"

VIDEO_DIRS = {
    "person_candidate_and_add": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_person_candidate_and_add_mux_all_selected",
    "person_mux_selected": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_person_mux_selected",
    "background": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_background_retain_music_mux_selected",
    "talk": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_talk_mux_selected",
    "remove": "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/video_720p_remove_yt_mux_selected",
}


def compute_avsync_conf(video_path, align_instance, restorer, lipsync, resize_tf, normalize_tf, device):
    from decord import VideoReader, AudioReader
    try:
        ar = AudioReader(video_path, sample_rate=lipsync.audio_sample_rate)
        audio_raw = ar[:]
        audio_data = (audio_raw.numpy().squeeze(0) if hasattr(audio_raw, "numpy")
                      else audio_raw.asnumpy().squeeze(0))

        audio_energy = np.abs(audio_data).mean()
        if audio_energy < 0.001:
            return None

        mel_data = torch.from_numpy(melspectrogram(audio_data))

        vr = VideoReader(video_path)
        num_frames = len(vr)
        num_sync_frames = lipsync.num_frames

        if num_frames < num_sync_frames + 4:
            return None

        # 逐帧人脸检测+对齐
        face_list = []
        for idx in range(num_frames):
            frame_raw = vr[idx]
            frame = (frame_raw.numpy() if hasattr(frame_raw, "numpy") else frame_raw.asnumpy())
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pts256_list, score_list, bboxes_list = align_instance(frame_bgr)
            if len(pts256_list) > 0:
                pts256 = pts256_list[0]
                lmk3_ = np.zeros((3, 2))
                lmk3_[0] = np.mean(pts256[[0, 1, 2, 3, 12, 13, 14, 15], 0:2], axis=0)
                lmk3_[1] = np.mean(pts256[[16, 17, 18, 19, 28, 29, 30, 31], 0:2], axis=0)
                lmk3_[2] = np.mean(pts256[[80, 81, 82, 83, 84, 91], 0:2], axis=0)
                face, _ = restorer.align_warp_face(
                    frame_bgr.copy(), lmks3=lmk3_, smooth=False, border_mode="constant"
                )
                face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_CUBIC)
                face_list.append(face[:, :, ::-1])  # BGR->RGB
            else:
                face_list.append(None)

        # 滑动窗口
        vision_embeds = []
        audio_embeds = []
        for start_idx in range(num_frames - num_sync_frames + 1):
            window_faces = face_list[start_idx: start_idx + num_sync_frames]
            if any(f is None for f in window_faces):
                continue
            mel_tensor = lipsync.crop_audio_window(mel_data, start_idx)
            if mel_tensor is None or mel_tensor.shape[1] < lipsync.mel_window_length:
                continue
            mel_tensor = mel_tensor.half().to(device).unsqueeze(0)

            faces = np.stack(window_faces)
            faces = torch.from_numpy(faces.copy())
            faces = rearrange(faces, "b h w c -> b c h w")
            faces = resize_tf(faces)
            faces = normalize_tf(faces / 255.0)
            height = faces.shape[2]
            faces = faces[:, :, height // 2:, :]
            faces = faces.contiguous().view(
                num_sync_frames * 3, height // 2, -1
            ).to(device).half().unsqueeze(0)

            with torch.no_grad():
                v_emb, a_emb = lipsync.syncnet(faces, mel_tensor)
            vision_embeds.append(v_emb)
            audio_embeds.append(a_emb)

        if len(vision_embeds) < 3:
            return None

        vision_embeds = torch.cat(vision_embeds, dim=0)
        audio_embeds = torch.cat(audio_embeds, dim=0)
        vshift = 10
        dists = calc_pdist_cos(vision_embeds, audio_embeds, vshift=vshift)
        mean_dists = torch.mean(torch.stack(dists, 1), 1)
        min_dist, minidx = torch.min(mean_dists, 0)
        conf = (torch.median(mean_dists) - min_dist).item()
        av_offset = (vshift - minidx).item()
        return {"conf": conf, "av_offset": av_offset}

    except Exception as e:
        return None


def worker_fn(gpu_id, video_shard, progress_counter):
    device = f"cuda:{gpu_id}"
    align_instance = AlignImage(
        device=device,
        det_path=f"{VT_ROOT}/data/models/yoloface_v5l.pt",
        p1_path=f"{VT_ROOT}/data/models/p1.pt",
        p2_path=f"{VT_ROOT}/data/models/p2.pt",
        pts217_path=f"{VT_ROOT}/data/models/res101_maxpool_pts217.bin"
    )
    restorer = AlignRestore()
    lipsync = LipSync(device_id=gpu_id, config_path=f"{VT_ROOT}/analysis/configs.yaml")
    resize_tf = transforms.Resize((256, 256))
    normalize_tf = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    results = []
    for item in video_shard:
        result = compute_avsync_conf(
            item["path"], align_instance, restorer, lipsync,
            resize_tf, normalize_tf, device
        )
        if result is not None:
            item["sync_conf"] = result["conf"]
            item["av_offset"] = result["av_offset"]
        else:
            item["sync_conf"] = None
            item["av_offset"] = None
        item.pop("sync_score", None)
        results.append(item)
        with progress_counter.get_lock():
            progress_counter.value += 1

    # 写 checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"sync_conf_addition_gpu{gpu_id}.jsonl")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    # 读取采样文件
    sample_path = os.path.join(CKPT_DIR, "sampled_addition_1000.json")
    with open(sample_path) as f:
        all_records = json.load(f)
    print(f"Loaded {len(all_records)} sampled videos")

    num_gpus = 8
    shards = [[] for _ in range(num_gpus)]
    for i, r in enumerate(all_records):
        shards[i % num_gpus].append(r)

    print(f"Using {num_gpus} GPUs, ~{len(shards[0])} videos per GPU")

    mp.set_start_method("spawn", force=True)
    progress_counter = mp.Value("i", 0)

    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=worker_fn, args=(gpu_id, shards[gpu_id], progress_counter))
        p.start()
        processes.append(p)

    import time
    total = len(all_records)
    pbar = tqdm(total=total, desc="AV Sync (conf)", unit="video", dynamic_ncols=True)
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

    # 汇总
    results = []
    for gpu_id in range(num_gpus):
        ckpt_path = os.path.join(CKPT_DIR, f"sync_conf_addition_gpu{gpu_id}.jsonl")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except:
                            pass

    print(f"\nTotal: {len(results)} videos")

    for task_type in VIDEO_DIRS:
        task_results = [r for r in results if r["task_type"] == task_type]
        vtss_scores = [r["vtss_score"] for r in task_results if r["vtss_score"] >= 0]
        confs = [r["sync_conf"] for r in task_results if r.get("sync_conf") is not None]
        null_count = sum(1 for r in task_results if r.get("sync_conf") is None)
        pass_count = sum(1 for c in confs if c >= 0.2)

        print(f"\n{task_type} ({len(task_results)} videos):")
        if vtss_scores:
            print(f"  VTSS: mean={np.mean(vtss_scores):.4f}, median={np.median(vtss_scores):.4f}")
        if confs:
            print(f"  Sync conf: mean={np.mean(confs):.4f}, median={np.median(confs):.4f}, "
                  f"min={np.min(confs):.4f}, max={np.max(confs):.4f}")
            print(f"  Pass (conf>=0.2): {pass_count}/{len(confs)} ({100*pass_count/len(confs):.1f}%)")
        print(f"  Null (no face/no voice): {null_count}")

    output_path = OUTPUT_JSON.replace(".json", "_sampled_addition_1000.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
