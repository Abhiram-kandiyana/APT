#!/usr/bin/env python3
"""
One-time migration tool:
- old layout:   prompts/<dataset>_prompts/fold{fold}_round{r}_prompts_<selection>[_corrected].json
- new layout:   prompt_sets/<dataset>/fold-<fold>/...

Also rebuilds:
- global_prompts_<selection>_corrected.json
- class/case_id/image folders from global prompts
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_case_id_from_path(image_path: str) -> str:
    parts = [p for p in Path(str(image_path)).parts if p and p != "."]
    if len(parts) >= 2:
        return str(parts[-2])
    return str(parts[-1]) if parts else ""


def _copy_or_move(src: str, dst: str, move: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if move:
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def _discover_round_files(src_dir: str, fold: int, selection: str) -> List[Tuple[int, str, str]]:
    """
    Returns (round_num, file_kind, abs_path) where file_kind in {"raw","corrected"}.
    """
    pattern = re.compile(
        rf"^fold{int(fold)}_round(\d+)_prompts_{re.escape(selection)}(_corrected)?\.json$",
        flags=re.IGNORECASE,
    )
    out: List[Tuple[int, str, str]] = []
    for name in os.listdir(src_dir):
        m = pattern.match(name)
        if not m:
            continue
        r = int(m.group(1))
        kind = "corrected" if m.group(2) else "raw"
        out.append((r, kind, os.path.join(src_dir, name)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _build_global_corrected(dst_fold_dir: str, selection: str) -> Tuple[str, Dict[str, Any]]:
    pattern = re.compile(
        rf"^fold\d+_round(\d+)_prompts_{re.escape(selection)}_corrected\.json$",
        flags=re.IGNORECASE,
    )
    corrected: List[Tuple[int, str]] = []
    for name in os.listdir(dst_fold_dir):
        m = pattern.match(name)
        if not m:
            continue
        corrected.append((int(m.group(1)), os.path.join(dst_fold_dir, name)))
    corrected.sort(key=lambda x: x[0])

    all_prompts: List[Dict[str, Any]] = []
    total_duration = 0.0
    total_caption_corrections = 0
    accuracies: List[float] = []

    for _, path in corrected:
        payload = _load_json(path)
        prompts = payload.get("prompts", [])
        if isinstance(prompts, list):
            all_prompts.extend(prompts)
        total_duration += _to_float(payload.get("Duration in minutes", 0.0), default=0.0)
        total_caption_corrections += _to_int(payload.get("caption correction count", 0), default=0)
        if "accuracy" in payload:
            accuracies.append(_to_float(payload.get("accuracy", 0.0), default=0.0))

    global_payload = {
        "prompts": all_prompts,
        "caption correction count": int(total_caption_corrections),
        "Duration in minutes": float(total_duration),
        "accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
    }
    global_path = os.path.join(dst_fold_dir, f"global_prompts_{selection}_corrected.json")
    _write_json(global_path, global_payload)
    return global_path, global_payload


def _sync_class_case_dirs(dst_fold_dir: str, global_payload: Dict[str, Any], dry_run: bool) -> None:
    prompts = global_payload.get("prompts", [])
    if not isinstance(prompts, list):
        return

    class_names = sorted({
        str(item.get("class", "")).strip()
        for item in prompts
        if isinstance(item, dict) and str(item.get("class", "")).strip()
    })

    # Rebuild class folders from scratch (only class dirs, leaves JSON files untouched).
    for cls in class_names:
        class_dir = os.path.join(dst_fold_dir, cls)
        if os.path.isdir(class_dir):
            if dry_run:
                print(f"[dry-run] would delete directory: {class_dir}")
            else:
                shutil.rmtree(class_dir)

    for item in prompts:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path", "")).strip()
        cls = str(item.get("class", "")).strip()
        if not image_path or not cls:
            continue
        case_id = _extract_case_id_from_path(image_path)
        if not case_id:
            continue

        dst_case_dir = os.path.join(dst_fold_dir, cls, case_id)
        dst_image = os.path.join(dst_case_dir, os.path.basename(image_path))
        if dry_run:
            print(f"[dry-run] would copy: {image_path} -> {dst_image}")
            continue

        os.makedirs(dst_case_dir, exist_ok=True)
        if os.path.exists(image_path) and not os.path.exists(dst_image):
            shutil.copy2(image_path, dst_image)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time prompts -> prompt_sets migration")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., microscopy_lurcher")
    parser.add_argument("--fold", type=int, required=True, help="Fold number, e.g., 5")
    parser.add_argument("--selection", type=str, default="dts", help="Selection method token in filenames")
    parser.add_argument("--src-root", type=str, default="prompts", help="Old root directory")
    parser.add_argument("--dst-root", type=str, default="prompt_sets", help="New root directory")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, args.src_root, f"{args.dataset}_prompts")
    dst_fold_dir = os.path.join(base_dir, args.dst_root, args.dataset, f"fold-{int(args.fold)}")

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    round_files = _discover_round_files(src_dir, args.fold, args.selection)
    if not round_files:
        raise RuntimeError(
            f"No matching files found in {src_dir} for fold={args.fold}, selection={args.selection}"
        )

    if args.dry_run:
        print(f"[dry-run] source: {src_dir}")
        print(f"[dry-run] destination: {dst_fold_dir}")
    else:
        os.makedirs(dst_fold_dir, exist_ok=True)

    for round_num, _, src_path in round_files:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_fold_dir, fname)
        if args.dry_run:
            action = "move" if args.move else "copy"
            print(f"[dry-run] would {action}: {src_path} -> {dst_path}")
        else:
            _copy_or_move(src_path, dst_path, move=args.move)

    if args.dry_run:
        print("[dry-run] would rebuild global corrected file and class/case directories")
        return

    global_path, global_payload = _build_global_corrected(dst_fold_dir, args.selection)
    _sync_class_case_dirs(dst_fold_dir, global_payload, dry_run=False)

    print(f"Migration completed for dataset={args.dataset}, fold={args.fold}, selection={args.selection}")
    print(f"New fold directory: {dst_fold_dir}")
    print(f"Global file: {global_path}")


if __name__ == "__main__":
    main()

