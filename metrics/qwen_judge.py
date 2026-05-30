# coding=utf-8
"""
qwen_judge.py - Qwen3-Omni VLM Judge 共享模块
封装 Qwen3-Omni 模型加载、prompt 模板和推理逻辑，
供 av_quality.py / instruction_compliance.py / video_fidelity.py 调用。

运行环境: conda activate qwen3omni
"""
import os
import sys
import re
import json
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================
# 清理 sys.path 中可能干扰 vllm 的路径
# ============================================================
_cleaned = []
for _p in sys.path:
    _resolved = os.path.abspath(_p) if _p else os.getcwd()
    _candidate = os.path.join(_resolved, 'vllm')
    if os.path.isdir(_candidate) and not os.path.isfile(os.path.join(_candidate, '__init__.py')):
        continue
    _cleaned.append(_p)
sys.path = _cleaned


# ============================================================
# Evaluation Prompts
# ============================================================

STRICT_PREAMBLE = """You must be extremely critical and detail-oriented in your evaluation. Carefully inspect every frame for artifacts, temporal inconsistencies, and unnatural elements. Check physical plausibility (lighting, shadows, perspective, scale). Listen closely for any audio imperfections, unnaturalness, or synchronization issues. Pay attention to fine details that may not be obvious at first glance — zoom in mentally on edges, textures, and transitions.

IMPORTANT scoring guidance:
- A score of 5 means TRULY flawless upon careful frame-by-frame inspection. This is rare.
- A score of 4 means very good with only minor imperfections noticeable on close inspection.
- A score of 3 is the expected baseline for a reasonably acceptable edit with visible but tolerable issues.
- Do NOT default to high scores. Most AI-edited videos will have some imperfections.
- When in doubt between two scores, choose the LOWER one.
"""

INSTRUCTION_PROMPTS = {
    "subject_editing": STRICT_PREAMBLE + """You are given two videos (before and after editing) and the editing instruction. Evaluate the subject/person editing result on a 5-point scale from two perspectives.

Editing instruction: "{instruction}"

**Instruction Compliance** — Has the subject's appearance been changed as specified?
1. No visible change to the subject, or the subject is completely distorted/unrecognizable, or the wrong element was edited.
2. Subject partially changed but most specified attributes are missing or incorrect; result looks nothing like the instruction describes.
3. Some attributes match the instruction (e.g., clothing changed but wrong color/style), or transformation is incomplete (partially applied, inconsistent across frames).
4. Subject appearance mostly matches the instruction with only minor deviations (e.g., slight color mismatch, one small detail missing, minor temporal inconsistency in the edit).
5. Subject appearance exactly matches all aspects of the instruction consistently throughout the entire video duration.

**Video Fidelity** — Are non-edited elements preserved from the source?
1. Background completely different, camera motion changed, or subject's pose/actions are totally different from source; video is incoherent.
2. Significant unwanted changes: background partly replaced, noticeable camera trajectory shift, or subject's body movements clearly altered.
3. Background mostly preserved but with visible modifications (color shifts, object displacement); subject pose roughly maintained but with noticeable jitter or drift.
4. Background and camera motion well preserved; subject pose/actions intact except for minute differences (slight edge artifacts, tiny temporal wobble near edit boundary).
5. All non-edited elements are pixel-perfect: background, camera trajectory, subject pose, body motion, and temporal dynamics are indistinguishable from the source.

Output ONLY a JSON object: {"instruction_compliance": <score>, "video_fidelity": <score>}""",

    "background_editing": STRICT_PREAMBLE + """You are given two videos (before and after editing) and the editing instruction. Evaluate the background editing result on a 5-point scale from two perspectives.

Editing instruction: "{instruction}"

**Instruction Compliance** — Has the background/scene been changed as specified?
1. No background change at all, or background is unrelated to the instruction, or the foreground subject was also replaced/severely distorted.
2. Background partly replaced or has wrong style/content compared to instruction; foreground subject noticeably altered or damaged.
3. Main background replaced but elements are missing or extra compared to instruction; faint spill or artifacts onto subject edges.
4. Requested background fully present with correct content and style; foreground intact except for minute artifacts or small prompt mismatch (e.g., slightly wrong color tone or lighting).
5. Background exactly matches the instruction (content, style, atmosphere, placement); all foreground pixels untouched and naturally composited.

**Video Fidelity** — Are the foreground subject and other non-edited elements preserved?
1. Subject completely distorted, identity lost, or pose/actions radically different from source; large temporal artifacts (tearing, flickering).
2. Subject identity partially preserved but with clear visual damage: obvious cut-out halos, color mismatch, or visible edge instability over time.
3. Subject recognizable and pose roughly maintained; some visible edge artifacts, slight temporal instability (shimmer/wobble), or minor identity drift.
4. Subject appearance, pose, and actions well preserved; edges are stable across motion; only minor issues visible when inspecting closely (tiny edge blur, slight color shift near boundaries).
5. Foreground subject is perfectly preserved: identity, pose, motion, edges, and temporal coherence are indistinguishable from the source throughout the entire video.

Output ONLY a JSON object: {"instruction_compliance": <score>, "video_fidelity": <score>}""",

    "subject_removal": STRICT_PREAMBLE + """You are given two videos (before and after editing) and the editing instruction. Evaluate the subject removal result on a 5-point scale from two perspectives.

Editing instruction: "{instruction}"

**Instruction Compliance** — Has the human subject been successfully removed?
1. Subject is still fully visible with no removal attempt, or the video is severely corrupted/unwatchable.
2. Subject partially removed but major portions remain clearly visible (limbs, torso outline, shadow with human shape); subject's voice still fully audible.
3. Subject mostly removed but noticeable remnants exist: ghosting, visible silhouette edges, blurred human-shaped region, or voice partially audible in audio.
4. Subject removed with only very minor traces: tiny shadow remnant, slight color inconsistency in the removal region, or barely audible voice residue.
5. Subject completely and cleanly removed with zero traces in both video and audio; the region looks as if no person was ever there.

**Video Fidelity** — Is the inpainted region natural and is the rest of the scene preserved?
1. Inpainted region is severely distorted: large holes, extreme blur, or completely wrong content; rest of scene (camera motion, other objects) also corrupted.
2. Inpainted region is clearly artificial: obvious repeating patterns, wrong perspective, or significant temporal flickering; other scene elements partially disrupted.
3. Inpainting is acceptable but visibly imperfect: noticeable texture mismatch, moderate blur in filled area, or minor temporal instability; camera motion and other objects mostly preserved.
4. Inpainting looks natural at normal viewing: textures are well-matched, perspective is correct; very minor imperfections visible only on close inspection; all other scene elements perfectly preserved.
5. Inpainting is flawless: filled region is spatially and temporally seamless with surrounding background; camera motion, lighting, and all other scene elements are perfectly preserved throughout.

Output ONLY a JSON object: {"instruction_compliance": <score>, "video_fidelity": <score>}""",

    "subject_addition": STRICT_PREAMBLE + """You are given two videos (before and after editing) and the editing instruction. Evaluate the subject addition result on a 5-point scale from two perspectives.

Editing instruction: "{instruction}"

**Instruction Compliance** — Has a human subject been successfully added as specified?
1. No subject added at all, or an unrecognizable/severely distorted figure was inserted, or the result is completely corrupted.
2. A figure is inserted but it clearly does not match the instruction (wrong appearance, wrong location), or it looks extremely artificial (flat, static, no motion).
3. A subject is added with some attributes matching the instruction, but with noticeable issues: unnatural pose, wrong scale/perspective, obvious compositing artifacts, or missing specified attributes.
4. Subject is added matching most of the instruction requirements; natural appearance and motion with only minor issues (slight scale mismatch, small temporal wobble, minor attribute deviation).
5. Subject is perfectly added exactly as described: correct appearance, natural motion and pose, proper scale and perspective, with corresponding voice/audio if applicable.

**Video Fidelity** — Are the original background and scene elements preserved?
1. Background completely changed, camera motion altered, or original scene elements displaced/corrupted by the insertion.
2. Background partially disrupted: noticeable spatial distortion around insertion area, color bleeding, or camera motion perturbed.
3. Background mostly preserved but with visible artifacts near the added subject: slight color shift, minor spatial warping, or small temporal instability in surrounding area.
4. Background and scene well preserved; added subject integrates naturally with only minor edge artifacts or barely noticeable local disturbance.
5. Original scene perfectly preserved: background, lighting, camera motion, and ambient elements are untouched; the added subject integrates seamlessly as if originally filmed there.

Output ONLY a JSON object: {"instruction_compliance": <score>, "video_fidelity": <score>}""",

    "speech_editing": STRICT_PREAMBLE + """You are given two videos (before and after editing) and the editing instruction. Evaluate the speech content modification result on a 5-point scale from two perspectives.

Editing instruction: "{instruction}"

**Instruction Compliance** — Has the spoken content been changed as specified, with proper lip-sync?
1. Speech is unchanged from source, or audio is completely corrupted/unintelligible, or no lip movement change is visible.
2. Speech content is mostly wrong (different words than instructed), or severely garbled/distorted; lip movements show no meaningful synchronization with the new audio.
3. Speech content partially matches instruction but with errors (missing/extra words, wrong pronunciation); visible lip-sync offset (>200ms) or intermittent desynchronization.
4. Speech content correctly matches instruction with minor issues: slight mispronunciation, small audio artifacts, or minor lip-sync offset (<200ms) that is noticeable but not distracting.
5. Speech content exactly matches the instruction word-for-word; audio is clear and natural; lip movements are perfectly synchronized with the new speech throughout.

**Video Fidelity** — Are all non-lip visual elements preserved from the source?
1. Person's identity is lost, face is severely distorted, or background/body completely different from source.
2. Person's face is noticeably altered beyond the lip region: eye shape changed, skin texture damaged, or visible warping artifacts across the face; background may have shifted.
3. Face mostly preserved but with visible artifacts near the mouth region spreading to cheeks/chin; minor identity drift; background intact but slight temporal flickering around face.
4. Face and identity well preserved; artifacts confined to immediate lip boundary; background and body perfectly maintained; only minor issues visible on close frame-by-frame inspection.
5. All non-lip elements are pixel-perfect: facial identity, skin texture, expression (beyond lips), background, body pose, and camera motion are indistinguishable from the source.

Output ONLY a JSON object: {"instruction_compliance": <score>, "video_fidelity": <score>}""",
}

AV_QUALITY_PROMPT = STRICT_PREAMBLE + """You are given a single video with its audio track. Evaluate its overall audio-visual quality as a single comprehensive score on a 5-point scale.

Consider ALL of the following aspects in your evaluation:
- Visual quality: sharpness, clarity, temporal stability, absence of artifacts (flickering, blurring, warping, color banding)
- Audio quality: speech clarity and naturalness, music quality, absence of noise/distortion/clipping
- Audio-visual synchronization: lip-sync accuracy, action-sound timing, semantic coherence between what is seen and heard

**Overall AV Quality Score**
1. Severely defective: unwatchable video (extreme distortion, incoherent frames) OR unintelligible audio OR completely out-of-sync audio-video. Any one critical failure in visual, audio, or sync results in this score.
2. Poor quality: major problems in at least one dimension — heavy visual artifacts/blur, OR significantly distorted/robotic audio, OR obvious lip-sync mismatch (200-500ms). Other dimensions may be acceptable but one drags the overall quality down.
3. Acceptable quality: all three dimensions are at least tolerable — some visible artifacts but content understandable, audio has minor issues but is clear enough, sync is imperfect but not distracting. No single dimension is severely broken.
4. Good quality: all three dimensions are good — sharp visuals with only minor imperfections, clear natural audio with minimal artifacts, good sync with only very slight offset. Minor issues may exist but do not distract from the overall experience.
5. Excellent quality: all three dimensions are near-flawless — pristine visuals, crystal-clear natural audio, and perfect synchronization. No noticeable issues even on careful inspection. This score should be rare for AI-generated/edited content.

Output ONLY a JSON object: {"av_quality": <score>}"""


# ============================================================
# Parsing Utilities
# ============================================================

def parse_instruction_scores(text):
    """Extract instruction_compliance and video_fidelity scores from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            ic = parsed.get("instruction_compliance")
            vf = parsed.get("video_fidelity")
            if isinstance(ic, (int, float)) and isinstance(vf, (int, float)):
                return int(ic), int(vf)
        except json.JSONDecodeError:
            pass
    return None, None


def parse_av_scores(text):
    """Extract av_quality score from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            avq = parsed.get("av_quality")
            if isinstance(avq, (int, float)):
                return int(avq)
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
# Model Management
# ============================================================

_global_llm = None
_global_processor = None


def load_model(model_path, tensor_parallel_size=4, pipeline_parallel_size=2):
    """加载 Qwen3-Omni 模型 (全局单例)"""
    global _global_llm, _global_processor

    if _global_llm is not None:
        return _global_llm, _global_processor

    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("vision_process").setLevel(logging.WARNING)

    from vllm import LLM
    from transformers import Qwen3OmniMoeProcessor

    logger.info(f"Loading Qwen3-Omni model from {model_path}...")
    _global_llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
        max_num_seqs=8,
        max_model_len=32768,
        seed=1234,
    )
    _global_processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    logger.info("Model loaded successfully!")

    return _global_llm, _global_processor


# ============================================================
# Batch Inference
# ============================================================

def run_instruction_eval(entries, llm, processor, batch_size=8):
    """
    运行 instruction compliance + video fidelity 评测。

    Args:
        entries: list of dict, 每项需包含 src_path, tgt_path, prompt, task, video_name
        llm: vLLM model instance
        processor: Qwen3OmniMoeProcessor
        batch_size: 批大小

    Returns:
        list of dict with instruction_compliance, video_fidelity, raw_output
    """
    from vllm import SamplingParams
    from qwen_omni_utils import process_mm_info

    results = []

    for batch_start in tqdm(range(0, len(entries), batch_size), desc="Instruction+Fidelity Eval"):
        batch = entries[batch_start: batch_start + batch_size]
        batch_inputs = []
        batch_meta = []

        for entry in batch:
            task = entry["task"]
            if task not in INSTRUCTION_PROMPTS:
                continue

            prompt_text = INSTRUCTION_PROMPTS[task].replace("{instruction}", entry["prompt"])

            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": entry["src_path"]},
                    {"type": "audio", "audio": entry["src_path"]},
                    {"type": "video", "video": entry["tgt_path"]},
                    {"type": "audio", "audio": entry["tgt_path"]},
                    {"type": "text", "text": prompt_text},
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            except Exception as e:
                logger.warning(f"Skipping {entry['video_name']}: {e}")
                continue

            inputs = {
                'prompt': text,
                'multi_modal_data': {},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
            if images is not None:
                inputs['multi_modal_data']['image'] = images
            if videos is not None:
                inputs['multi_modal_data']['video'] = videos
            if audios is not None:
                inputs['multi_modal_data']['audio'] = audios

            batch_inputs.append(inputs)
            batch_meta.append(entry)

        if not batch_inputs:
            continue

        sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=4096, seed=42)
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        for entry, output in zip(batch_meta, outputs):
            raw_text = output.outputs[0].text.strip()
            ic_score, vf_score = parse_instruction_scores(raw_text)
            results.append({
                "video": os.path.basename(entry["tgt_path"]),
                "video_name": entry["video_name"],
                "task": entry["task"],
                "prompt": entry["prompt"],
                "instruction_compliance": ic_score,
                "video_fidelity": vf_score,
                "raw_output": raw_text,
            })

        torch.cuda.empty_cache()

    return results


def run_av_quality_eval(entries, llm, processor, batch_size=8):
    """
    运行 AV quality 评测。

    Args:
        entries: list of dict, 每项需包含 tgt_path, task, video_name
        llm: vLLM model instance
        processor: Qwen3OmniMoeProcessor
        batch_size: 批大小

    Returns:
        list of dict with av_quality, raw_output
    """
    from vllm import SamplingParams
    from qwen_omni_utils import process_mm_info

    results = []

    for batch_start in tqdm(range(0, len(entries), batch_size), desc="AV Quality Eval"):
        batch = entries[batch_start: batch_start + batch_size]
        batch_inputs = []
        batch_meta = []

        for entry in batch:
            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": entry["tgt_path"]},
                    {"type": "audio", "audio": entry["tgt_path"]},
                    {"type": "text", "text": AV_QUALITY_PROMPT},
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            except Exception as e:
                logger.warning(f"Skipping {entry['video_name']}: {e}")
                continue

            inputs = {
                'prompt': text,
                'multi_modal_data': {},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
            if images is not None:
                inputs['multi_modal_data']['image'] = images
            if videos is not None:
                inputs['multi_modal_data']['video'] = videos
            if audios is not None:
                inputs['multi_modal_data']['audio'] = audios

            batch_inputs.append(inputs)
            batch_meta.append(entry)

        if not batch_inputs:
            continue

        sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=4096, seed=42)
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        for entry, output in zip(batch_meta, outputs):
            raw_text = output.outputs[0].text.strip()
            avq = parse_av_scores(raw_text)
            results.append({
                "video": os.path.basename(entry["tgt_path"]),
                "video_name": entry["video_name"],
                "task": entry["task"],
                "av_quality": avq,
                "raw_output": raw_text,
            })

        torch.cuda.empty_cache()

    return results
