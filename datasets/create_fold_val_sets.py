#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create val.jsonl and val_selected.jsonl for fold directories by using "
            "case IDs from splits.json (e.g., round-1)."
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
        help="Fold numbers to process, e.g. --folds 6 10",
    )
    parser.add_argument(
        "--round-key",
        type=str,
        default="round-1",
        help="Round key under each fold in splits.json (default: round-1).",
    )
    parser.add_argument(
        "--samples-per-case",
        type=int,
        default=25,
        help="Number of samples per case to write to val_selected.jsonl (default: 25).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def normalize_class(raw: str) -> str:
    value = raw.strip().lower()
    if value.startswith("c:"):
        value = value.split(":", 1)[1].strip()
    return value


def split_train_to_val(
    train_rows: List[dict],
    case_ids_by_class: Dict[str, Set[str]],
) -> Tuple[List[dict], List[dict]]:
    remaining: List[dict] = []
    val_rows: List[dict] = []

    for row in train_rows:
        image_path = row.get("image_path")
        class_name = normalize_class(str(row.get("class", "")))

        if not image_path:
            remaining.append(row)
            continue

        case_id = Path(str(image_path)).parent.name
        target_ids = case_ids_by_class.get(class_name, set())

        if case_id in target_ids:
            val_rows.append(row)
        else:
            remaining.append(row)

    return remaining, val_rows


def extract_case_id(row: dict) -> str:
    return Path(str(row.get("image_path", ""))).parent.name


def extract_slide_id(row: dict) -> str:
    """
    Parse slide ID from image filename:
    6480_lurcher_10x_lurcher_3_1_Image_03.jpg -> slide_id = "3"
    """
    stem = Path(str(row.get("image_path", ""))).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return "unknown"
    return parts[-4]


def select_case_slide_balanced(
    case_rows: List[dict],
    samples_per_case: int,
    rng: random.Random,
) -> List[dict]:
    if len(case_rows) <= samples_per_case:
        return list(case_rows)

    by_slide: Dict[str, List[dict]] = defaultdict(list)
    for row in case_rows:
        by_slide[extract_slide_id(row)].append(row)

    # Shuffle rows within each slide once, then pick in round-robin across slides
    # to keep slide representation as uniform as possible.
    for slide_id in by_slide:
        rng.shuffle(by_slide[slide_id])

    slide_order = sorted(by_slide.keys())
    selected: List[dict] = []
    selected_count_by_slide: Dict[str, int] = defaultdict(int)

    while len(selected) < samples_per_case:
        candidates = [
            slide_id
            for slide_id in slide_order
            if selected_count_by_slide[slide_id] < len(by_slide[slide_id])
        ]
        if not candidates:
            break

        # Prioritize slides with fewer selections so far; use RNG for deterministic
        # tie-breaking under the fold seed.
        min_selected = min(selected_count_by_slide[slide_id] for slide_id in candidates)
        least_selected = [
            slide_id
            for slide_id in candidates
            if selected_count_by_slide[slide_id] == min_selected
        ]
        chosen_slide = rng.choice(least_selected)
        idx = selected_count_by_slide[chosen_slide]
        selected.append(by_slide[chosen_slide][idx])
        selected_count_by_slide[chosen_slide] += 1

    return selected


def select_balanced_by_case(
    val_rows: List[dict],
    samples_per_case: int,
    seed: int,
) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in val_rows:
        grouped[extract_case_id(row)].append(row)

    rng = random.Random(seed)
    selected: List[dict] = []
    for case_id in sorted(grouped):
        selected.extend(
            select_case_slide_balanced(
                case_rows=grouped[case_id],
                samples_per_case=samples_per_case,
                rng=rng,
            )
        )
    return selected


def main() -> None:
    args = parse_args()
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
        if args.round_key not in fold_splits:
            raise KeyError(f"{args.round_key!r} not found under fold {fold_key}")

        round_cases = fold_splits[args.round_key]
        case_ids_by_class: Dict[str, Set[str]] = {
            normalize_class(class_name): {str(case_id) for case_id in case_ids}
            for class_name, case_ids in round_cases.items()
        }

        fold_dir = dataset_dir / f"fold-{fold}"
        train_path = fold_dir / "train.jsonl"
        val_path = fold_dir / "val.jsonl"
        val_selected_path = fold_dir / "val_selected.jsonl"

        if not train_path.exists():
            raise FileNotFoundError(f"train.jsonl not found at {train_path}")

        train_rows = load_jsonl(train_path)
        _, val_rows = split_train_to_val(train_rows, case_ids_by_class)

        fold_seed = int(fold_splits.get("random seed", 0))
        selected_rows = select_balanced_by_case(
            val_rows=val_rows,
            samples_per_case=args.samples_per_case,
            seed=fold_seed,
        )

        write_jsonl(val_path, val_rows)
        write_jsonl(val_selected_path, selected_rows)

        class_counts = defaultdict(int)
        selected_counts = defaultdict(int)
        selected_case_counts = defaultdict(int)
        selected_slide_counts = defaultdict(int)
        for row in val_rows:
            class_counts[normalize_class(str(row.get("class", "")))] += 1
        for row in selected_rows:
            selected_counts[normalize_class(str(row.get("class", "")))] += 1
            selected_case_counts[extract_case_id(row)] += 1
            selected_slide_counts[f"{extract_case_id(row)}:slide-{extract_slide_id(row)}"] += 1

        print(f"fold-{fold}: wrote {val_path} ({len(val_rows)} rows)")
        print(f"fold-{fold}: val class counts {dict(sorted(class_counts.items()))}")
        print(
            f"fold-{fold}: wrote {val_selected_path} ({len(selected_rows)} rows)"
        )
        print(
            f"fold-{fold}: val_selected class counts {dict(sorted(selected_counts.items()))}"
        )
        print(
            f"fold-{fold}: val_selected case counts {dict(sorted(selected_case_counts.items()))}"
        )


if __name__ == "__main__":

#     python3 /Users/abhiramkandiyana/Microscopy/Microscopy/APT-MDL/datasets/create_fold_val_sets.py \
#   --dataset-dir /Users/abhiramkandiyana/Microscopy/Microscopy/APT-MDL/datasets/microscopy_lurcher \
#   --folds 6 10 \
#   --round-key round-1 \
#   --samples-per-case 25

    main()
