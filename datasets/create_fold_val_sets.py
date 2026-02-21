#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train.jsonl, val.jsonl, test.jsonl, and val_selected.jsonl for "
            "fold directories using splits.json and dataset directories."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to dataset root containing splits.json and fold-* directories.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        required=True,
        help="Fold numbers to process, e.g. --folds 5 6 10",
    )
    parser.add_argument(
        "--validation-round-key",
        type=str,
        default="round-1",
        help="Split key to use for validation IDs (default: round-1).",
    )
    parser.add_argument(
        "--val-percent-samples-per-case",
        type=float,
        default=25.0,
        help="Percent of val images to select per case for val_selected.jsonl (default: 25).",
    )
    parser.add_argument(
        "--image-extensions",
        type=str,
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Allowed image extensions (default: .jpg .jpeg .png .tif .tiff).",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def normalize_class(raw: str) -> str:
    value = raw.strip().lower()
    if value.startswith("c:"):
        value = value.split(":", 1)[1].strip()
    return value


def resolve_existing_dataset_root(
    configured_path: str,
    all_splits: Mapping[str, dict],
    path_key: str,
) -> Path:
    """
    Resolve dataset root path from splits.json.

    If configured path does not exist locally (e.g., /content/... path from Colab),
    fallback to any existing path under the same key that shares the same leaf folder.
    """
    configured = Path(configured_path)
    if configured.exists():
        return configured

    target_leaf = configured.name
    for fold_data in all_splits.values():
        candidate_raw = fold_data.get(path_key)
        if not isinstance(candidate_raw, str):
            continue
        candidate = Path(candidate_raw)
        if candidate.exists() and candidate.name == target_leaf:
            return candidate

    raise FileNotFoundError(
        f"Could not resolve existing path for {path_key}={configured_path!r}. "
        "Update splits.json with a local path or add a fold that provides one."
    )


def list_case_images(
    dataset_root: Path,
    class_name: str,
    case_id: str,
    allowed_extensions: Sequence[str],
) -> List[Path]:
    case_dir = dataset_root / class_name / str(case_id)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions}
    image_paths = [
        p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed
    ]
    return sorted(image_paths)


def build_rows_from_case_map(
    dataset_root: Path,
    case_map: Mapping[str, Sequence[int]],
    allowed_extensions: Sequence[str],
) -> List[dict]:
    rows: List[dict] = []
    for class_name in sorted(case_map):
        normalized = normalize_class(class_name)
        case_ids = sorted(str(case_id) for case_id in case_map[class_name])
        for case_id in case_ids:
            image_paths = list_case_images(
                dataset_root=dataset_root,
                class_name=normalized,
                case_id=case_id,
                allowed_extensions=allowed_extensions,
            )
            for image_path in image_paths:
                rows.append({"image_path": str(image_path), "class": normalized})
    return rows


def list_case_ids_by_class(dataset_root: Path) -> Dict[str, List[str]]:
    """
    Enumerate class/case IDs from dataset directory layout:
    <dataset_root>/<class_name>/<case_id>/*
    """
    case_ids_by_class: Dict[str, List[str]] = {}
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        class_name = normalize_class(class_dir.name)
        case_ids = sorted(p.name for p in class_dir.iterdir() if p.is_dir())
        case_ids_by_class[class_name] = case_ids
    return case_ids_by_class


def build_train_case_map_excluding_holdouts(
    dataset_root: Path,
    val_case_map: Mapping[str, Sequence[int]],
    test_case_map: Mapping[str, Sequence[int]],
) -> Dict[str, List[str]]:
    """
    Build train case map as:
    all case IDs in dataset_root - (validation round case IDs U test case IDs),
    computed per class.
    """
    all_cases_by_class = list_case_ids_by_class(dataset_root)
    train_case_map: Dict[str, List[str]] = {}

    for class_name, all_case_ids in all_cases_by_class.items():
        holdout_ids = set()
        for raw_case_id in val_case_map.get(class_name, []):
            holdout_ids.add(str(raw_case_id))
        for raw_case_id in test_case_map.get(class_name, []):
            holdout_ids.add(str(raw_case_id))
        train_case_map[class_name] = [
            case_id for case_id in all_case_ids if case_id not in holdout_ids
        ]

    return train_case_map


def extract_case_id(row: dict) -> str:
    return Path(str(row.get("image_path", ""))).parent.name


def extract_slide_id(row: dict) -> str:
    """
    Parse slide ID from image filename.

    Example:
    6480_lurcher_10x_lurcher_3_1_Image_03.jpg -> slide_id = "3"
    """
    stem = Path(str(row.get("image_path", ""))).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return "unknown"
    return parts[-4]


def target_count_from_percent(total_count: int, percent: float) -> int:
    if total_count <= 0 or percent <= 0:
        return 0
    return min(total_count, max(1, math.ceil((percent / 100.0) * total_count)))


def select_case_slide_balanced(
    case_rows: List[dict],
    target_count: int,
    rng: random.Random,
) -> List[dict]:
    if target_count >= len(case_rows):
        return list(case_rows)

    rows_by_slide: Dict[str, List[dict]] = defaultdict(list)
    for row in case_rows:
        rows_by_slide[extract_slide_id(row)].append(row)

    for slide_id in rows_by_slide:
        rng.shuffle(rows_by_slide[slide_id])

    selected: List[dict] = []
    selected_count_by_slide: Dict[str, int] = defaultdict(int)
    slide_order = sorted(rows_by_slide)

    while len(selected) < target_count:
        available_slides = [
            slide_id
            for slide_id in slide_order
            if selected_count_by_slide[slide_id] < len(rows_by_slide[slide_id])
        ]
        if not available_slides:
            break

        min_selected = min(selected_count_by_slide[slide_id] for slide_id in available_slides)
        least_selected = [
            slide_id
            for slide_id in available_slides
            if selected_count_by_slide[slide_id] == min_selected
        ]

        chosen_slide = rng.choice(least_selected)
        chosen_idx = selected_count_by_slide[chosen_slide]
        selected.append(rows_by_slide[chosen_slide][chosen_idx])
        selected_count_by_slide[chosen_slide] += 1

    return selected


def select_val_rows_by_case_percent(
    val_rows: List[dict],
    val_percent_samples_per_case: float,
    random_seed: int,
) -> List[dict]:
    rows_by_case: Dict[str, List[dict]] = defaultdict(list)
    for row in val_rows:
        rows_by_case[extract_case_id(row)].append(row)

    rng = random.Random(random_seed)
    selected_rows: List[dict] = []

    for case_id in sorted(rows_by_case):
        case_rows = rows_by_case[case_id]
        target_count = target_count_from_percent(len(case_rows), val_percent_samples_per_case)
        selected_rows.extend(
            select_case_slide_balanced(
                case_rows=case_rows,
                target_count=target_count,
                rng=rng,
            )
        )

    return selected_rows


def summarize_rows(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    class_counts = Counter(normalize_class(str(row.get("class", ""))) for row in rows)
    case_counts = Counter(extract_case_id(row) for row in rows)
    return {
        "class_counts": dict(sorted(class_counts.items())),
        "case_counts": dict(sorted(case_counts.items())),
    }


def main() -> None:
    args = parse_args()

    if args.val_percent_samples_per_case < 0 or args.val_percent_samples_per_case > 100:
        raise ValueError("--val-percent-samples-per-case must be in [0, 100].")

    dataset_dir = args.dataset_dir
    splits_path = dataset_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found at {splits_path}")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    for fold in args.folds:
        fold_key = str(fold)
        if fold_key not in splits:
            raise KeyError(f"Fold {fold_key} not found in {splits_path}")

        fold_splits = splits[fold_key]
        required_keys = [args.validation_round_key, "test", "dataset_path", "test_dataset_path"]
        missing_keys = [key for key in required_keys if key not in fold_splits]
        if missing_keys:
            raise KeyError(f"Fold {fold_key} is missing keys: {missing_keys}")

        prompt_dataset_root = resolve_existing_dataset_root(
            configured_path=str(fold_splits["dataset_path"]),
            all_splits=splits,
            path_key="dataset_path",
        )
        eval_dataset_root = resolve_existing_dataset_root(
            configured_path=str(fold_splits["test_dataset_path"]),
            all_splits=splits,
            path_key="test_dataset_path",
        )

        val_case_map = {
            normalize_class(str(class_name)): list(case_ids)
            for class_name, case_ids in fold_splits[args.validation_round_key].items()
        }
        test_case_map = {
            normalize_class(str(class_name)): list(case_ids)
            for class_name, case_ids in fold_splits["test"].items()
        }
        train_case_map = build_train_case_map_excluding_holdouts(
            dataset_root=prompt_dataset_root,
            val_case_map=val_case_map,
            test_case_map=test_case_map,
        )

        train_rows = build_rows_from_case_map(
            dataset_root=prompt_dataset_root,
            case_map=train_case_map,
            allowed_extensions=args.image_extensions,
        )
        val_rows = build_rows_from_case_map(
            dataset_root=eval_dataset_root,
            case_map=val_case_map,
            allowed_extensions=args.image_extensions,
        )
        test_rows = build_rows_from_case_map(
            dataset_root=eval_dataset_root,
            case_map=test_case_map,
            allowed_extensions=args.image_extensions,
        )

        fold_seed = int(fold_splits.get("random seed", 0) or 0)
        val_selected_rows = select_val_rows_by_case_percent(
            val_rows=val_rows,
            val_percent_samples_per_case=args.val_percent_samples_per_case,
            random_seed=fold_seed,
        )

        fold_dir = dataset_dir / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_path = fold_dir / "train.jsonl"
        val_path = fold_dir / "val.jsonl"
        test_path = fold_dir / "test.jsonl"
        val_selected_path = fold_dir / "val_selected.jsonl"

        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)
        write_jsonl(test_path, test_rows)
        write_jsonl(val_selected_path, val_selected_rows)

        train_summary = summarize_rows(train_rows)
        val_summary = summarize_rows(val_rows)
        test_summary = summarize_rows(test_rows)
        val_selected_summary = summarize_rows(val_selected_rows)

        print(f"fold-{fold}: dataset roots -> train={prompt_dataset_root}, eval={eval_dataset_root}")
        print(f"fold-{fold}: wrote {train_path} ({len(train_rows)} rows)")
        print(f"fold-{fold}: train class counts {train_summary['class_counts']}")
        print(f"fold-{fold}: wrote {val_path} ({len(val_rows)} rows)")
        print(f"fold-{fold}: val class counts {val_summary['class_counts']}")
        print(f"fold-{fold}: wrote {test_path} ({len(test_rows)} rows)")
        print(f"fold-{fold}: test class counts {test_summary['class_counts']}")
        print(f"fold-{fold}: wrote {val_selected_path} ({len(val_selected_rows)} rows)")
        print(f"fold-{fold}: val_selected class counts {val_selected_summary['class_counts']}")
        print(f"fold-{fold}: val_selected case counts {val_selected_summary['case_counts']}")


if __name__ == "__main__":
    # Example:
    # python /Users/abhiramkandiyana/Microscopy/Microscopy/APT-MDL/datasets/create_fold_val_sets.py \
    #   --dataset-dir /Users/abhiramkandiyana/Microscopy/Microscopy/APT-MDL/datasets/microscopy_lurcher \
    #   --folds 5 6 10 \
    #   --validation-round-key round-1 \
    #   --val-percent-samples-per-case 25
    main()
