import argparse
import gc
import json
import os
import pickle
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from tqdm.auto import tqdm


PLACEHOLDER_RE = re.compile(r"\[([a-zA-Z]+)_\d+\]")
STOP_NOUNS = {
    "thing", "object", "something", "stuff", "someone", "one", "way", "time",
    "kind", "sort", "lot", "piece", "side", "end", "part", "place", "people",
    "left", "right", "top", "bottom", "front", "back",
}
GENERIC_NOUNS = ["person", "car", "vehicle", "object", "animal", "food", "ball", "chair"]


class Config:
    gdino_name = "IDEA-Research/grounding-dino-base"
    deberta_name = "microsoft/deberta-v3-base"
    frcnn_cfg = "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"
    frcnn_score_thresh = 0.05

    n_frames = 16
    box_thresh = 0.20
    text_thresh = 0.25
    focus_thresh_bonus = 0.10
    nms_iou = 0.50
    min_area_ratio = 0.0005
    max_boxes = 12
    max_prompt_nouns = 12

    gdino_max_side = 640
    gdino_frame_chunk = 1
    deberta_batch = 64
    hidden_dim = 2048
    use_decord = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32


cfg = Config()
nlp = None
gdino_processor = None
gdino_model = None
deberta_tokenizer = None
deberta_model = None
frcnn_cfg = None
frcnn_model = None
ImageList = None
Boxes = None
_DECORD_OK = False
_TEXT_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GroundingDINO + Faster R-CNN ROI features for video_id.mp4 files."
    )
    parser.add_argument("--video-dir", required=True, help="Folder containing video_id.mp4 files.")
    parser.add_argument("--output-dir", required=True, help="Folder to write one video_id.pkl per video.")
    parser.add_argument("--qa-root", default=None, help="Optional QA root containing video_id/text.json.")
    parser.add_argument("--split-list", default=None, help="Optional txt file with one video_id per line.")
    parser.add_argument("--processed-file", default="processed.txt", help="Resume txt filename inside output-dir.")
    parser.add_argument("--failed-file", default="failed.txt", help="Failure log filename inside output-dir.")
    parser.add_argument("--detectron2-dir", default=None, help="Optional path to local detectron2 source checkout.")
    parser.add_argument("--n-frames", type=int, default=cfg.n_frames)
    parser.add_argument("--max-side", type=int, default=cfg.gdino_max_side)
    parser.add_argument("--gdino-chunk", type=int, default=cfg.gdino_frame_chunk)
    parser.add_argument("--max-boxes", type=int, default=cfg.max_boxes)
    parser.add_argument("--box-thresh", type=float, default=cfg.box_thresh)
    parser.add_argument("--text-thresh", type=float, default=cfg.text_thresh)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process exactly one video, write it under output-dir/_smoke_test, and exit without updating processed.txt.",
    )
    parser.add_argument(
        "--smoke-video-id",
        default=None,
        help="Optional video_id to use with --smoke-test. Defaults to the first pending/target video.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU. Very slow, but useful for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Re-process videos even if listed in processed.txt.")
    parser.add_argument("--no-decord", action="store_true", help="Use cv2 decoding only.")
    return parser.parse_args()


def apply_args(args):
    cfg.n_frames = args.n_frames
    cfg.gdino_max_side = args.max_side
    cfg.gdino_frame_chunk = max(1, args.gdino_chunk)
    cfg.max_boxes = args.max_boxes
    cfg.box_thresh = args.box_thresh
    cfg.text_thresh = args.text_thresh
    cfg.use_decord = not args.no_decord
    if args.cpu:
        cfg.device = "cpu"
        cfg.dtype = torch.float32


def load_models(detectron2_dir=None):
    global gdino_processor, gdino_model, deberta_tokenizer, deberta_model
    global frcnn_cfg, frcnn_model, ImageList, Boxes, nlp

    try:
        from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. Install it in xla_env, for example: "
            "pip install transformers accelerate huggingface_hub"
        ) from exc

    if detectron2_dir:
        d2_path = str(Path(detectron2_dir).resolve())
        if d2_path not in sys.path:
            sys.path.insert(0, d2_path)

    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.structures import Boxes as D2Boxes
    from detectron2.structures import ImageList as D2ImageList

    ImageList = D2ImageList
    Boxes = D2Boxes

    print(f"Device: {cfg.device} | dtype: {cfg.dtype} | frames/video: {cfg.n_frames}")
    print("Loading GroundingDINO...")
    gdino_processor = AutoProcessor.from_pretrained(cfg.gdino_name)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg.gdino_name, torch_dtype=cfg.dtype
    ).to(cfg.device).eval()

    print("Loading Faster R-CNN R101-C4...")
    frcnn_cfg = get_cfg()
    frcnn_cfg.merge_from_file(model_zoo.get_config_file(cfg.frcnn_cfg))
    frcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.frcnn_cfg)
    frcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.frcnn_score_thresh
    frcnn_cfg.MODEL.DEVICE = cfg.device
    frcnn_model = build_model(frcnn_cfg)
    DetectionCheckpointer(frcnn_model).load(frcnn_cfg.MODEL.WEIGHTS)
    frcnn_model.eval()

    print("Loading DeBERTa-v3-base...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(cfg.deberta_name)
    deberta_model = AutoModel.from_pretrained(
        cfg.deberta_name, torch_dtype=cfg.dtype
    ).to(cfg.device).eval()

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("spaCy en_core_web_sm loaded.")
    except Exception as exc:
        nlp = None
        print(f"spaCy unavailable, using fallback prompts only when needed: {exc}")

    for model in (gdino_model, frcnn_model, deberta_model):
        for param in model.parameters():
            param.requires_grad_(False)
    print("Models ready.")


def setup_decord():
    global _DECORD_OK
    _DECORD_OK = False
    if not cfg.use_decord:
        return
    try:
        import decord

        decord.bridge.set_bridge("native")
        _DECORD_OK = True
        print("decord ready.")
    except Exception as exc:
        print(f"decord unavailable, falling back to cv2: {exc}")


def read_processed(path):
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_line(path, text):
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{text}\n")
        f.flush()
        os.fsync(f.fileno())


def discover_videos(video_dir, split_list=None):
    video_dir = Path(video_dir)
    video_map = {p.stem: p for p in video_dir.rglob("*.mp4")}
    if split_list:
        ids = [Path(line.strip()).stem for line in Path(split_list).read_text(encoding="utf-8").splitlines() if line.strip()]
        return [vid for vid in ids if vid in video_map], video_map
    return sorted(video_map), video_map


def _qa_text_for_video(qa_dir):
    text_path = qa_dir / "text.json"
    if not text_path.exists():
        return ""
    text = json.loads(text_path.read_text(encoding="utf-8"))
    paragraphs = []
    for block in text.values():
        if not isinstance(block, dict):
            continue
        if "question" in block:
            paragraphs.append(str(block["question"]))
        for key in ("answer", "reason"):
            if isinstance(block.get(key), list):
                paragraphs.extend(str(x) for x in block[key])
    return " ".join(paragraphs)


def _clean_word(word):
    word = word.strip().lower()
    if len(word) < 2 or word in STOP_NOUNS or not re.fullmatch(r"[a-zA-Z]+", word):
        return None
    return word


def build_prompt(video_id, qa_root):
    if qa_root is None:
        return " . ".join(GENERIC_NOUNS) + " .", GENERIC_NOUNS[:], []

    raw = _qa_text_for_video(Path(qa_root) / video_id)
    if not raw:
        return " . ".join(GENERIC_NOUNS) + " .", GENERIC_NOUNS[:], []

    focus_nouns = []
    seen_focus = set()
    for match in PLACEHOLDER_RE.finditer(raw):
        noun = _clean_word(match.group(1))
        if noun and noun not in seen_focus:
            seen_focus.add(noun)
            focus_nouns.append(noun)

    raw = PLACEHOLDER_RE.sub(lambda m: m.group(1).lower(), raw)
    nouns = []
    seen = set()
    for noun in focus_nouns:
        seen.add(noun)
        nouns.append(noun)

    if nlp is not None:
        doc = nlp(raw.lower())
        i = 0
        while i < len(doc) and len(nouns) < cfg.max_prompt_nouns:
            if doc[i].pos_ not in ("NOUN", "PROPN"):
                i += 1
                continue
            chunk = []
            while i < len(doc) and doc[i].pos_ in ("NOUN", "PROPN"):
                word = _clean_word(doc[i].lemma_)
                if word:
                    chunk.append(word)
                i += 1
            phrase = " ".join(chunk)
            if phrase and phrase not in seen:
                seen.add(phrase)
                nouns.append(phrase)
        if not nouns:
            nouns = GENERIC_NOUNS[:]
    else:
        words = re.findall(r"[a-zA-Z]+", raw.lower())
        for word in words:
            word = _clean_word(word)
            if word and word not in seen:
                seen.add(word)
                nouns.append(word)
            if len(nouns) >= cfg.max_prompt_nouns:
                break
        if not nouns:
            nouns = GENERIC_NOUNS[:]

    return " . ".join(nouns[: cfg.max_prompt_nouns]) + " .", nouns[: cfg.max_prompt_nouns], focus_nouns


def sample_frames_cv2(video_path, n_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if ok and frame_bgr is not None else None)
    cap.release()
    return frames, indices.tolist(), total


def sample_frames_decord(video_path, n_frames):
    try:
        import decord

        vr = decord.VideoReader(str(video_path), num_threads=1)
        total = len(vr)
        if total == 0:
            return None, None, 0
        indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        batch = vr.get_batch(indices.tolist()).asnumpy()
        return [batch[i] for i in range(batch.shape[0])], indices.tolist(), int(total)
    except Exception:
        return sample_frames_cv2(video_path, n_frames)


def sample_frames(video_path, n_frames):
    if _DECORD_OK:
        return sample_frames_decord(video_path, n_frames)
    return sample_frames_cv2(video_path, n_frames)


def resize_for_gdino(img):
    if not cfg.gdino_max_side or cfg.gdino_max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= cfg.gdino_max_side:
        return img
    scale = cfg.gdino_max_side / float(long_side)
    return cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def cuda_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gdino_inputs(images, prompts):
    try:
        return gdino_processor(images=images, text=prompts, padding=True, truncation=True, return_tensors="pt")
    except TypeError:
        tok = gdino_processor.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        img = gdino_processor.image_processor(images=images, return_tensors="pt")
        return {**img, **tok}


@torch.no_grad()
def run_gdino(frames_rgb, prompt, box_thresh, text_thresh):
    valid_idx = [i for i, frame in enumerate(frames_rgb) if frame is not None]
    results = [None] * len(frames_rgb)
    if not valid_idx:
        return results

    images = [resize_for_gdino(frames_rgb[i]) for i in valid_idx]
    target_sizes = [(frames_rgb[i].shape[0], frames_rgb[i].shape[1]) for i in valid_idx]

    for start in range(0, len(images), cfg.gdino_frame_chunk):
        sub_images = images[start : start + cfg.gdino_frame_chunk]
        sub_sizes = target_sizes[start : start + cfg.gdino_frame_chunk]
        inputs = gdino_inputs(sub_images, [prompt] * len(sub_images))
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=cfg.dtype, enabled=cfg.device == "cuda"):
            outputs = gdino_model(**inputs)
        try:
            post = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=sub_sizes,
            )
        except TypeError:
            post = gdino_processor.post_process_grounded_object_detection(
                outputs,
                threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=sub_sizes,
            )
        for j, res in enumerate(post):
            results[valid_idx[start + j]] = res
        del outputs, inputs, post
        cuda_gc()
    return results


def filter_boxes(res, h, w, allowed_nouns, focus_nouns):
    if res is None or len(res["boxes"]) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    boxes = res["boxes"].detach().cpu().float()
    scores = res["scores"].detach().cpu().float()
    raw_labels = res.get("text_labels", res.get("labels", []))

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep_area = areas >= cfg.min_area_ratio * h * w
    if keep_area.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    boxes = boxes[keep_area]
    scores = scores[keep_area]
    raw_labels = [raw_labels[i] for i, keep in enumerate(keep_area.tolist()) if keep]

    keep = nms(boxes, scores, cfg.nms_iou)
    boxes = boxes[keep]
    scores = scores[keep]
    raw_labels = [raw_labels[i] for i in keep.tolist()]

    allowed_set = set(allowed_nouns)
    focus_set = set(focus_nouns)
    labels = []
    for label in raw_labels:
        words = re.findall(r"[a-zA-Z]+", str(label).lower())
        match = next((word for word in words if word in allowed_set), None)
        labels.append(match or (words[0] if words else "object"))

    focus_thresh = max(0.05, cfg.box_thresh - cfg.focus_thresh_bonus)
    keep_mask = [score >= (focus_thresh if label in focus_set else cfg.box_thresh) for score, label in zip(scores.tolist(), labels)]
    if not any(keep_mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    keep_idx = [i for i, keep in enumerate(keep_mask) if keep]
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = [labels[i] for i in keep_idx]

    if len(boxes) > cfg.max_boxes:
        focus_idx = [i for i, label in enumerate(labels) if label in focus_set]
        non_idx = [i for i, label in enumerate(labels) if label not in focus_set]
        chosen = sorted(focus_idx, key=lambda i: -scores[i].item()) + sorted(non_idx, key=lambda i: -scores[i].item())
        chosen = sorted(chosen[: cfg.max_boxes])
        boxes = boxes[chosen]
        scores = scores[chosen]
        labels = [labels[i] for i in chosen]

    return boxes.numpy().astype(np.float32), scores.numpy().astype(np.float32), labels


@torch.no_grad()
def extract_roi_features(frame_rgb, boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return np.zeros((0, cfg.hidden_dim), dtype=np.float32)

    bgr = frame_rgb[:, :, ::-1].copy()
    img = torch.from_numpy(bgr).permute(2, 0, 1).float().to(cfg.device)
    pixel_mean = torch.tensor(frcnn_cfg.MODEL.PIXEL_MEAN, device=cfg.device).view(-1, 1, 1)
    pixel_std = torch.tensor(frcnn_cfg.MODEL.PIXEL_STD, device=cfg.device).view(-1, 1, 1)
    img = (img - pixel_mean) / pixel_std
    images = ImageList.from_tensors([img], frcnn_model.backbone.size_divisibility)
    features = frcnn_model.backbone(images.tensor)
    boxes_tensor = torch.as_tensor(boxes_xyxy, device=cfg.device, dtype=torch.float32)
    roi_feats = frcnn_model.roi_heads.pooler([features["res4"]], [Boxes(boxes_tensor)])
    res5_feats = frcnn_model.roi_heads.res5(roi_feats)
    feat_vec = torch.mean(res5_feats, dim=[2, 3])
    return feat_vec.float().cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_class_texts(texts):
    out = np.zeros((len(texts), 768), dtype=np.float32)
    todo_idx, todo_txt = [], []
    for i, text in enumerate(texts):
        if text in _TEXT_CACHE:
            out[i] = _TEXT_CACHE[text]
        else:
            todo_idx.append(i)
            todo_txt.append(text)
    for start in range(0, len(todo_txt), cfg.deberta_batch):
        chunk = todo_txt[start : start + cfg.deberta_batch]
        toks = deberta_tokenizer(chunk, padding=True, truncation=True, max_length=8, return_tensors="pt").to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=cfg.dtype, enabled=cfg.device == "cuda"):
            emb = deberta_model(**toks).last_hidden_state[:, 0]
        emb = emb.float().cpu().numpy()
        for j, original_idx in enumerate(todo_idx[start : start + cfg.deberta_batch]):
            _TEXT_CACHE[chunk[j]] = emb[j]
            out[original_idx] = emb[j]
        del toks, emb
        cuda_gc()
    return out


def process_video(video_id, video_path, qa_root):
    prompt, nouns, focus_nouns = build_prompt(video_id, qa_root)
    frames, indices, total = sample_frames(video_path, cfg.n_frames)
    if frames is None:
        raise RuntimeError("cannot decode video")

    h, w = next((frame.shape[:2] for frame in frames if frame is not None), (0, 0))
    if h == 0:
        raise RuntimeError("no valid frames decoded")

    relaxed_box = max(0.05, cfg.box_thresh - cfg.focus_thresh_bonus)
    relaxed_text = max(0.05, cfg.text_thresh - cfg.focus_thresh_bonus)
    gd_results = run_gdino(frames, prompt, relaxed_box, relaxed_text)

    records = []
    unique_labels = OrderedDict()
    for frame_rgb, frame_idx, res in zip(frames, indices, gd_results):
        if frame_rgb is None:
            records.append(
                {
                    "frame_idx": int(frame_idx),
                    "boxes_xyxy_orig": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                    "labels_text": [],
                    "roi_features": np.zeros((0, cfg.hidden_dim), dtype=np.float32),
                }
            )
            continue
        fh, fw = frame_rgb.shape[:2]
        boxes, scores, labels = filter_boxes(res, fh, fw, nouns, focus_nouns)
        roi_features = extract_roi_features(frame_rgb, boxes)
        for label in labels:
            unique_labels.setdefault(label, None)
        records.append(
            {
                "frame_idx": int(frame_idx),
                "boxes_xyxy_orig": boxes,
                "scores": scores,
                "labels_text": labels,
                "roi_features": roi_features,
            }
        )

    labels = list(unique_labels.keys())
    emb_map = dict(zip(labels, encode_class_texts(labels))) if labels else {}
    for rec in records:
        if rec["labels_text"]:
            rec["class_text_embedding"] = np.stack([emb_map[label] for label in rec["labels_text"]]).astype(np.float32)
        else:
            rec["class_text_embedding"] = np.zeros((0, 768), dtype=np.float32)

    return {
        "video_id": video_id,
        "orig_h": int(h),
        "orig_w": int(w),
        "total_frames": int(total),
        "prompt": prompt,
        "nouns": nouns,
        "focus_nouns": focus_nouns,
        "frames": records,
    }


def write_pickle_atomic(package, output_path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(output_path)


def main():
    args = parse_args()
    apply_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / args.processed_file
    failed_path = output_dir / args.failed_file

    target_ids, video_map = discover_videos(args.video_dir, args.split_list)
    processed = set() if args.overwrite else read_processed(processed_path)
    if not args.overwrite:
        for vid in target_ids:
            if vid not in processed and (output_dir / f"{vid}.pkl").exists():
                append_line(processed_path, vid)
                processed.add(vid)
    pending = [vid for vid in target_ids if args.overwrite or vid not in processed]
    if args.smoke_test:
        if args.smoke_video_id:
            if args.smoke_video_id not in video_map:
                raise SystemExit(f"Smoke video not found under --video-dir: {args.smoke_video_id}")
            pending = [args.smoke_video_id]
        else:
            pending = pending[:1] if pending else target_ids[:1]

    print(f"Videos found: {len(video_map)} | targets: {len(target_ids)} | pending: {len(pending)}")
    print(f"Output: {output_dir}")
    print(f"Resume list: {processed_path}")
    if args.smoke_test:
        print("Smoke test mode: processing one video only; processed.txt will not be updated.")
    if not pending:
        print("Nothing to do.")
        return

    setup_decord()
    load_models(args.detectron2_dir)

    ok = 0
    failed = 0
    t0 = time.time()
    for video_id in tqdm(pending, desc="Extracting"):
        if args.smoke_test:
            smoke_dir = output_dir / "_smoke_test"
            smoke_dir.mkdir(parents=True, exist_ok=True)
            output_path = smoke_dir / f"{video_id}.pkl"
        else:
            output_path = output_dir / f"{video_id}.pkl"
        try:
            package = process_video(video_id, video_map[video_id], args.qa_root)
            write_pickle_atomic(package, output_path)
            if args.smoke_test:
                n_boxes = sum(len(frame["labels_text"]) for frame in package["frames"])
                print(
                    f"\nSmoke OK: {video_id} -> {output_path} | "
                    f"frames={len(package['frames'])} boxes={n_boxes}"
                )
            else:
                append_line(processed_path, video_id)
            ok += 1
        except torch.cuda.OutOfMemoryError as exc:
            cuda_gc()
            failed += 1
            append_line(failed_path, f"{video_id}\tOOM\t{exc}")
            print(f"\nOOM on {video_id}; skipped and continued.")
        except Exception as exc:
            cuda_gc()
            failed += 1
            append_line(failed_path, f"{video_id}\tERROR\t{exc}")
            print(f"\nFailed {video_id}: {exc}")
        finally:
            cuda_gc()

    elapsed = time.time() - t0
    print(f"Done. ok={ok} failed={failed} elapsed={elapsed / 3600:.2f}h")


if __name__ == "__main__":
    main()
