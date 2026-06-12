import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/mnt/cpfs/delinmao"))
VSTAR_BENCH_ROOT = Path(os.getenv("VSTAR_BENCH_ROOT", WORKSPACE_ROOT / "Benchmarks" / "vstar-bench"))


_CHOICE_ANSWER_INSTRUCTION_RE = re.compile(
    r"\s*Answer with the option'?s letter from the given choices(?: directly)?\.\s*$",
    flags=re.IGNORECASE,
)


def _remove_direct_answer_line(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("answer with the option"):
            kept.append(stripped.replace(" directly.", ".").replace(" directly", ""))
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _strip_choice_answer_instruction(text: str) -> str:
    return _CHOICE_ANSWER_INSTRUCTION_RE.sub("", str(text or "").strip()).strip()


def _build_user_prompt(question_text: str, width: int, height: int) -> str:
    normalized = _remove_direct_answer_line(question_text)
    return f"<image>Image size = {width}x{height} pixels.\n\n{normalized}"


def _normalize_question_id(raw_question_id: Any, idx: int) -> str:
    text = str(raw_question_id if raw_question_id is not None else idx).strip()
    return text or str(idx)


def _build_sample_id(question_id: str) -> str:
    return f"vstar__test__{question_id}"


def _build_image_uri(image_path: Path) -> str:
    return f"file://{quote(str(image_path))}"


def _build_record(obj: dict[str, Any], image_root: Path, idx: int) -> dict[str, Any]:
    rel_image = obj["image"]
    image_path = (image_root / rel_image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    question_id = _normalize_question_id(obj.get("question_id"), idx)
    question_text = _remove_direct_answer_line(obj["text"])
    prompt_text = _build_user_prompt(obj["text"], width, height)
    category = str(obj.get("category") or "vstar_bench").strip() or "vstar_bench"
    ground_truth = obj["label"].strip().upper()

    extra_info = {
        "index": idx,
        "question": question_text,
        "category": category,
        "question_id": question_id,
        "image_path": str(image_path),
        "source_benchmark": "vstar_bench",
    }

    return {
        "data_source": category,
        "ability": "mm_qa",
        "prompt": [{"role": "user", "content": prompt_text}],
        "images": [{"image": _build_image_uri(image_path)}],
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": extra_info,
    }


def _build_root_sample(obj: dict[str, Any], image_root: Path, idx: int) -> dict[str, Any]:
    rel_image = obj["image"]
    image_path = (image_root / rel_image).resolve()
    question_id = _normalize_question_id(obj.get("question_id"), idx)
    category = str(obj.get("category") or "vstar_bench").strip() or "vstar_bench"
    question_text = _strip_choice_answer_instruction(obj["text"])
    ground_truth = obj["label"].strip().upper()

    return {
        "sample_id": _build_sample_id(question_id),
        "question": question_text,
        "answer_instruction": "Answer with the option letter only.",
        "images": [
            {
                "image_id": "img0",
                "path": str(image_path),
            }
        ],
        "metadata": {
            "source_dataset": "vstar",
            "source_split": "test",
            "source_sample_id": question_id,
            "source_benchmark": "vstar_bench",
            "category": category,
        },
        "answer": ground_truth,
    }


def _build_eval_annotation(obj: dict[str, Any], idx: int) -> dict[str, Any]:
    question_id = _normalize_question_id(obj.get("question_id"), idx)
    return {
        "sample_id": _build_sample_id(question_id),
        "metric": "exact_match",
        "references": [obj["label"].strip().upper()],
    }


def _load_raw_objects(input_jsonl: Path) -> list[dict[str, Any]]:
    rows = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(records: list[dict[str, Any]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_parquet(records: list[dict[str, Any]], output_parquet: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing parquet requires pandas with a parquet backend such as pyarrow or fastparquet."
        ) from exc

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(output_parquet, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize V*-Bench jsonl into CodeVision eval schema.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=VSTAR_BENCH_ROOT / "test_questions.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=VSTAR_BENCH_ROOT,
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=VSTAR_BENCH_ROOT / "vstar_codevision_eval.jsonl",
    )
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--output-root-samples-jsonl", type=Path, default=None)
    parser.add_argument("--output-eval-annotations-jsonl", type=Path, default=None)
    args = parser.parse_args()

    raw_objects = _load_raw_objects(args.input_jsonl)
    records = [_build_record(obj, image_root=args.image_root, idx=idx) for idx, obj in enumerate(raw_objects)]
    _write_jsonl(records, args.output_jsonl)

    if args.output_parquet is not None:
        _write_parquet(records, args.output_parquet)

    if args.output_root_samples_jsonl is not None:
        root_samples = [
            _build_root_sample(obj, image_root=args.image_root, idx=idx) for idx, obj in enumerate(raw_objects)
        ]
        _write_jsonl(root_samples, args.output_root_samples_jsonl)

    if args.output_eval_annotations_jsonl is not None:
        annotations = [_build_eval_annotation(obj, idx=idx) for idx, obj in enumerate(raw_objects)]
        _write_jsonl(annotations, args.output_eval_annotations_jsonl)

    print(f"Wrote {len(records)} records to {args.output_jsonl}")
    if args.output_parquet is not None:
        print(f"Wrote parquet to {args.output_parquet}")
    if args.output_root_samples_jsonl is not None:
        print(f"Wrote root samples to {args.output_root_samples_jsonl}")
    if args.output_eval_annotations_jsonl is not None:
        print(f"Wrote eval annotations to {args.output_eval_annotations_jsonl}")


if __name__ == "__main__":
    main()
