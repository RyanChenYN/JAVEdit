# coding=utf-8
"""eval_vtss_addition.py - 对 4000 个源视频算 VTSS"""
import os, sys, json, copy, tempfile, numpy as np, yaml, torch
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import OrderedDict

sys.path.insert(0, "../../../../../../../../..../../../../../Koala-36M/training_suitability_assessment")
from model import DiViDeAddEvaluator
from datasets import FusionDataset

sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]
CKPT_DIR = "../../../../../../../../..../../../../../Koala-36M/training_suitability_assessment/eval_vtss_avsync_checkpoints"


def load_vtss_model(opt, device):
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


def worker_fn(gpu_id, video_paths, opt, data_opt_template, progress_counter):
    device = "cuda:%d" % gpu_id
    model = load_vtss_model(opt, device)

    results = []
    for video_path in video_paths:
        vtss_score = -1.0
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                tmp.write("%s,0.0\n" % video_path)
                anno_path = tmp.name
            data_opt = copy.deepcopy(data_opt_template)
            data_opt["anno_file"] = anno_path
            data_opt["data_prefix"] = ""
            dataset = FusionDataset(data_opt)
            if len(dataset) > 0:
                data = dataset[0]
                video = {}
                for stype in sample_types:
                    if stype in data:
                        v = data[stype].unsqueeze(0).to(device)
                        b, c, t, h, w = v.shape
                        num_clips = data["num_clips"][stype]
                        v = (v.reshape(b, c, num_clips, t // num_clips, h, w)
                             .permute(0, 2, 1, 3, 4, 5)
                             .reshape(b * num_clips, c, t // num_clips, h, w))
                        video[stype] = v
                with torch.no_grad():
                    labels = model(video, reduce_scores=False)
                    labels = [np.mean(l.cpu().numpy()) for l in labels]
                vtss_score = float(np.sum(labels))
            os.remove(anno_path)
        except:
            vtss_score = -1.0

        results.append({
            "path": video_path,
            "task_type": "source",
            "vtss_score": vtss_score,
        })
        with progress_counter.get_lock():
            progress_counter.value += 1

    ckpt_path = os.path.join(CKPT_DIR, "vtss_addition_gpu%d.jsonl" % gpu_id)
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    paths_file = os.path.join(CKPT_DIR, "sampled_addition_4000_paths.json")
    with open(paths_file) as f:
        all_paths = json.load(f)
    print("Videos to score: %d" % len(all_paths))

    with open("test.yml") as f:
        opt = yaml.safe_load(f)

    data_key = None
    for key in opt["data"].keys():
        if "val" in key or "test" in key:
            data_key = key
            break
    data_opt_template = opt["data"][data_key]["args"]

    num_gpus = 8
    shards = [[] for _ in range(num_gpus)]
    for i, p in enumerate(all_paths):
        shards[i % num_gpus].append(p)

    print("Using %d GPUs, ~%d videos per GPU" % (num_gpus, len(shards[0])))

    mp.set_start_method("spawn", force=True)
    progress_counter = mp.Value("i", 0)

    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=worker_fn, args=(gpu_id, shards[gpu_id], opt, data_opt_template, progress_counter))
        p.start()
        processes.append(p)

    import time
    pbar = tqdm(total=len(all_paths), desc="VTSS Addition", unit="video", dynamic_ncols=True)
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
        p = os.path.join(CKPT_DIR, "vtss_addition_gpu%d.jsonl" % gpu_id)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except:
                            pass

    vtss_scores = [r["vtss_score"] for r in results if r["vtss_score"] >= 0]
    print("\nSource videos VTSS (%d scored):" % len(vtss_scores))
    print("  mean=%.4f, median=%.4f, min=%.4f, max=%.4f" % (
        np.mean(vtss_scores), np.median(vtss_scores), np.min(vtss_scores), np.max(vtss_scores)))

    # 追加到 eval_vtss_results.json
    vtss_out = "../../../../../../../../..../../../../../../../../..../../../../../JAVEdit/data/openhumanvid-06vtss_eng/eval_vtss_results.json"
    with open(vtss_out) as f:
        existing = json.load(f)
    print("Existing records: %d" % len(existing))
    existing.extend(results)
    with open(vtss_out, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print("Updated: %d total records -> %s" % (len(existing), vtss_out))


if __name__ == "__main__":
    main()
