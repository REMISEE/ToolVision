#!/usr/bin/env python3
"""Batch-export PaddleOCR results for TextVQA images.

This script reads TextVQA from Hugging Face datasets, deduplicates samples by
image_id within each split, runs PaddleOCR on each unique image, and writes a
JSONL sidecar that can later be consumed by lmms-eval or other pipelines.

Recommended usage on Linux with GPU PaddlePaddle installed:

    python tools/export_textvqa_paddleocr.py \
        --splits train validation \
        --output-root /path/to/textvqa_ocr_ppocrv5 \
        --devices 0 1 \
        --save-vis none
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = "lmms-lab/textvqa"


@dataclass
class ImageTask:
    split: str
    row_index: int
    image_id: str
    question_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PaddleOCR sidecar data for TextVQA.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="HF dataset path.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Dataset splits to process.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--devices", nargs="+", default=["0", "1"], help="GPU ids, e.g. 0 1. Use cpu for CPU mode.")
    parser.add_argument("--processes-per-device", type=int, default=1, help="OCR worker processes per device.")
    parser.add_argument("--image-batch-size", type=int, default=8, help="Number of images per predict() call.")
    parser.add_argument("--limit-images", type=int, default=None, help="Only process the first N unique images after deduplication.")
    parser.add_argument("--text-det-model-name", default="PP-OCRv5_server_det", help="PaddleOCR detection model name.")
    parser.add_argument("--text-rec-model-name", default="PP-OCRv5_server_rec", help="PaddleOCR recognition model name.")
    parser.add_argument("--text-rec-batch-size", type=int, default=32, help="text_recognition_batch_size passed to PaddleOCR.")
    parser.add_argument("--text-det-limit-side-len", type=int, default=960, help="text_det_limit_side_len.")
    parser.add_argument("--text-det-limit-type", default="max", choices=["min", "max"], help="text_det_limit_type.")
    parser.add_argument("--text-det-thresh", type=float, default=0.4, help="text_det_thresh.")
    parser.add_argument("--text-det-box-thresh", type=float, default=0.7, help="text_det_box_thresh.")
    parser.add_argument("--text-rec-score-thresh", type=float, default=0.6, help="text_rec_score_thresh.")
    parser.add_argument("--line-y-threshold", type=float, default=0.6, help="Relative threshold for grouping boxes into lines.")
    parser.add_argument("--save-vis", choices=["none", "all"], default="none", help="Whether to save OCR visualization images.")
    parser.add_argument("--save-raw-json", action="store_true", help="Also save one raw PaddleOCR JSON per image.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing worker outputs.")
    return parser.parse_args()


def load_tasks(dataset_path: str, splits: list[str]) -> list[ImageTask]:
    from datasets import load_dataset

    tasks: list[ImageTask] = []
    for split in splits:
        dataset = load_dataset(dataset_path, split=split)
        by_image: dict[str, dict[str, Any]] = {}
        for row_index, doc in enumerate(dataset):
            image_id = str(doc.get("image_id"))
            question_id = int(doc.get("question_id"))
            if image_id not in by_image:
                by_image[image_id] = {
                    "row_index": row_index,
                    "question_ids": [question_id],
                }
            else:
                by_image[image_id]["question_ids"].append(question_id)
        for image_id, meta in by_image.items():
            tasks.append(
                ImageTask(
                    split=split,
                    row_index=int(meta["row_index"]),
                    image_id=image_id,
                    question_ids=sorted(meta["question_ids"]),
                )
            )
    return tasks


def partition_round_robin(items: list[ImageTask], num_parts: int) -> list[list[ImageTask]]:
    parts = [[] for _ in range(num_parts)]
    for index, item in enumerate(items):
        parts[index % num_parts].append(item)
    return parts


def _box_from_any(box: Any) -> list[int] | None:
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, list):
        return None
    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        return [int(round(v)) for v in box]

    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = point
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        return None
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


def _sorted_ocr_items(texts: list[str], scores: list[float], boxes: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for text, score, box in zip(texts, scores, boxes):
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            continue
        rect = _box_from_any(box)
        if rect is None:
            continue
        items.append({"text": normalized, "score": float(score), "box": rect})
    items.sort(key=lambda item: (item["box"][1], item["box"][0], item["box"][3], item["box"][2]))
    return items


def _group_items_into_lines(items: list[dict[str, Any]], line_y_threshold: float) -> list[str]:
    if not items:
        return []

    line_groups: list[list[dict[str, Any]]] = []
    for item in items:
        x1, y1, x2, y2 = item["box"]
        h = max(1, y2 - y1)
        center_y = (y1 + y2) / 2.0
        attached = False
        for group in line_groups:
            gy1 = min(member["box"][1] for member in group)
            gy2 = max(member["box"][3] for member in group)
            gh = max(1, gy2 - gy1)
            group_center_y = (gy1 + gy2) / 2.0
            tol = max(h, gh) * line_y_threshold
            if abs(center_y - group_center_y) <= tol:
                group.append(item)
                attached = True
                break
        if not attached:
            line_groups.append([item])

    lines: list[str] = []
    line_groups.sort(key=lambda group: min(member["box"][1] for member in group))
    for group in line_groups:
        group.sort(key=lambda member: (member["box"][0], member["box"][1]))
        line = " ".join(member["text"] for member in group).strip()
        if line:
            lines.append(line)
    return lines


def build_sidecar_record(
    *,
    split: str,
    image_id: str,
    question_ids: list[int],
    raw_result: dict[str, Any],
    line_y_threshold: float,
    vis_path: str | None,
    raw_json_path: str | None,
) -> dict[str, Any]:
    if isinstance(raw_result.get("res"), dict):
        raw_result = raw_result["res"]
    
    rec_texts = list(raw_result.get("rec_texts") or [])
    rec_scores = list(raw_result.get("rec_scores") or [])
    rec_boxes = list(raw_result.get("rec_boxes") or raw_result.get("rec_polys") or [])
    ocr_items = _sorted_ocr_items(rec_texts, rec_scores, rec_boxes)
    ocr_lines = _group_items_into_lines(ocr_items, line_y_threshold=line_y_threshold)

    return {
        "split": split,
        "image_id": image_id,
        "question_ids": question_ids,
        "ocr_items": ocr_items,
        "ocr_text": "\n".join(ocr_lines),
        "ocr_lines": ocr_lines,
        "num_ocr_items": len(ocr_items),
        "raw_json_path": raw_json_path,
        "vis_path": vis_path,
    }


def worker_main(
    worker_index: int,
    device: str,
    worker_tasks: list[ImageTask],
    args_dict: dict[str, Any],
) -> None:
    from datasets import load_dataset
    import numpy as np
    from paddleocr import PaddleOCR

    dataset_path = args_dict["dataset_path"]
    output_root = Path(args_dict["output_root"])
    save_vis = args_dict["save_vis"]
    save_raw_json = bool(args_dict["save_raw_json"])
    line_y_threshold = float(args_dict["line_y_threshold"])
    image_batch_size = int(args_dict["image_batch_size"])

    worker_dir = output_root / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = worker_dir / f"worker_{worker_index:02d}.jsonl"
    error_path = worker_dir / f"worker_{worker_index:02d}_errors.jsonl"

    if sidecar_path.exists() and not args_dict["overwrite"]:
        raise FileExistsError(f"{sidecar_path} already exists. Use --overwrite to replace it.")

    split_to_dataset = {split: load_dataset(dataset_path, split=split) for split in sorted({task.split for task in worker_tasks})}

    ocr = PaddleOCR(
        device=device,
        text_detection_model_name=args_dict["text_det_model_name"],
        text_recognition_model_name=args_dict["text_rec_model_name"],
        text_recognition_batch_size=args_dict["text_rec_batch_size"],
        text_det_limit_side_len=args_dict["text_det_limit_side_len"],
        text_det_limit_type=args_dict["text_det_limit_type"],
        text_det_thresh=args_dict["text_det_thresh"],
        text_det_box_thresh=args_dict["text_det_box_thresh"],
        text_rec_score_thresh=args_dict["text_rec_score_thresh"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    raw_json_root = output_root / "raw_json"
    vis_root = output_root / "vis"
    if save_raw_json:
        raw_json_root.mkdir(parents=True, exist_ok=True)
    if save_vis == "all":
        vis_root.mkdir(parents=True, exist_ok=True)

    with sidecar_path.open("w", encoding="utf-8") as out_f, error_path.open("w", encoding="utf-8") as err_f:
        processed = 0
        for batch_start in range(0, len(worker_tasks), image_batch_size):
            batch_tasks = worker_tasks[batch_start : batch_start + image_batch_size]
            batch_inputs: list[Any] = []
            valid_batch_tasks: list[ImageTask] = []

            for task in batch_tasks:
                try:
                    dataset = split_to_dataset[task.split]
                    doc = dataset[task.row_index]
                    image = doc["image"].convert("RGB")
                    batch_inputs.append(np.asarray(image))
                    valid_batch_tasks.append(task)
                except Exception as exc:
                    err_f.write(
                        json.dumps(
                            {
                                "worker_index": worker_index,
                                "device": device,
                                "task": asdict(task),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    err_f.flush()

            if not valid_batch_tasks:
                continue

            try:
                results = list(ocr.predict(batch_inputs))
            except Exception as exc:
                for task in valid_batch_tasks:
                    err_f.write(
                        json.dumps(
                            {
                                "worker_index": worker_index,
                                "device": device,
                                "task": asdict(task),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                err_f.flush()
                continue

            if len(results) != len(valid_batch_tasks):
                err_f.write(
                    json.dumps(
                        {
                            "worker_index": worker_index,
                            "device": device,
                            "error_type": "BatchResultCountMismatch",
                            "error": f"expected {len(valid_batch_tasks)} results, got {len(results)}",
                            "task_image_ids": [task.image_id for task in valid_batch_tasks],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err_f.flush()
                continue

            for task, result in zip(valid_batch_tasks, results):
                try:
                    raw_result = result.json if isinstance(result.json, dict) else json.loads(result.json)

                    raw_json_path = None
                    if save_raw_json:
                        split_raw_dir = raw_json_root / task.split
                        split_raw_dir.mkdir(parents=True, exist_ok=True)
                        raw_json_file = split_raw_dir / f"{task.image_id}.json"
                        with raw_json_file.open("w", encoding="utf-8") as f:
                            json.dump(raw_result, f, ensure_ascii=False)
                        raw_json_path = str(raw_json_file)

                    vis_path = None
                    if save_vis == "all":
                        split_vis_dir = vis_root / task.split
                        split_vis_dir.mkdir(parents=True, exist_ok=True)
                        vis_file = split_vis_dir / f"{task.image_id}.png"
                        img_dict = result.img if isinstance(result.img, dict) else {}
                        vis_img = img_dict.get("ocr_res_img")
                        if vis_img is not None:
                            vis_img.save(vis_file)
                            vis_path = str(vis_file)

                    sidecar = build_sidecar_record(
                        split=task.split,
                        image_id=task.image_id,
                        question_ids=task.question_ids,
                        raw_result=raw_result,
                        line_y_threshold=line_y_threshold,
                        vis_path=vis_path,
                        raw_json_path=raw_json_path,
                    )
                    out_f.write(json.dumps(sidecar, ensure_ascii=False) + "\n")
                    processed += 1
                    if processed % 100 == 0:
                        print(f"[worker {worker_index}][{device}] processed {processed}/{len(worker_tasks)}")
                except Exception as exc:
                    err_f.write(
                        json.dumps(
                            {
                                "worker_index": worker_index,
                                "device": device,
                                "task": asdict(task),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    err_f.flush()


def merge_worker_outputs(output_root: Path, expected_workers: int) -> None:
    worker_dir = output_root / "workers"
    merged_path = output_root / "textvqa_ocr_sidecar.jsonl"
    files = [worker_dir / f"worker_{idx:02d}.jsonl" for idx in range(expected_workers)]
    with merged_path.open("w", encoding="utf-8") as out_f:
        for file_path in files:
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)


def normalize_devices(devices: list[str]) -> list[str]:
    normalized = []
    for device in devices:
        device = str(device).strip()
        if not device:
            continue
        if device.lower() == "cpu":
            normalized.append("cpu")
        elif ":" in device:
            normalized.append(device)
        else:
            normalized.append(f"gpu:{device}")
    if not normalized:
        raise ValueError("No valid devices provided.")
    return normalized


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.dataset_path, args.splits)
    if not tasks:
        raise RuntimeError("No image tasks found.")
    if args.limit_images is not None:
        tasks = tasks[: args.limit_images]

    devices = normalize_devices(args.devices)
    worker_devices: list[str] = []
    for device in devices:
        for _ in range(args.processes_per_device):
            worker_devices.append(device)

    shards = partition_round_robin(tasks, len(worker_devices))
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []

    args_dict = {
        "dataset_path": args.dataset_path,
        "output_root": str(args.output_root),
        "save_vis": args.save_vis,
        "save_raw_json": args.save_raw_json,
        "overwrite": args.overwrite,
        "image_batch_size": args.image_batch_size,
        "text_det_model_name": args.text_det_model_name,
        "text_rec_model_name": args.text_rec_model_name,
        "text_rec_batch_size": args.text_rec_batch_size,
        "text_det_limit_side_len": args.text_det_limit_side_len,
        "text_det_limit_type": args.text_det_limit_type,
        "text_det_thresh": args.text_det_thresh,
        "text_det_box_thresh": args.text_det_box_thresh,
        "text_rec_score_thresh": args.text_rec_score_thresh,
        "line_y_threshold": args.line_y_threshold,
    }

    for worker_index, (device, shard) in enumerate(zip(worker_devices, shards)):
        process = ctx.Process(target=worker_main, args=(worker_index, device, shard, args_dict))
        process.start()
        processes.append(process)

    failed = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed = True

    merge_worker_outputs(args.output_root, expected_workers=len(worker_devices))

    if failed:
        raise SystemExit("One or more workers failed. Check workers/*_errors.jsonl")

    manifest = {
        "dataset_path": args.dataset_path,
        "splits": args.splits,
        "num_unique_images": len(tasks),
        "devices": worker_devices,
        "processes_per_device": args.processes_per_device,
        "image_batch_size": args.image_batch_size,
        "text_det_model_name": args.text_det_model_name,
        "text_rec_model_name": args.text_rec_model_name,
        "text_rec_batch_size": args.text_rec_batch_size,
        "text_det_limit_side_len": args.text_det_limit_side_len,
        "text_det_limit_type": args.text_det_limit_type,
        "text_det_thresh": args.text_det_thresh,
        "text_det_box_thresh": args.text_det_box_thresh,
        "text_rec_score_thresh": args.text_rec_score_thresh,
        "save_vis": args.save_vis,
        "save_raw_json": args.save_raw_json,
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Finished. Wrote merged sidecar to {args.output_root / 'textvqa_ocr_sidecar.jsonl'}")


if __name__ == "__main__":
    main()
