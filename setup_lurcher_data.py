#!/usr/bin/env python3
"""Download and place the Lurcher dataset files used by APT."""

import argparse
import shutil
from pathlib import Path
from typing import Optional


DEFAULT_REPO_ID = "USF-CS-Microscopy-Image-Analysis/Lurcher_10x"
DATASET_KEY = "microscopy_lurcher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Lurcher 10x Hugging Face dataset assets and arrange "
            "the fold manifests for this repository."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, or commit hash.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="APT repository root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Download only manifests, metadata, and prompt bank; useful for quick checks.",
    )
    return parser.parse_args()


def snapshot_download(repo_id: str, revision: Optional[str], repo_root: Path, skip_images: bool) -> None:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install project requirements first:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    allow_patterns = [
        f"folds/{DATASET_KEY}/**",
        "metadata/**",
        f"prompt_banks/{DATASET_KEY}.json",
    ]
    if not skip_images:
        allow_patterns.append(f"images/{DATASET_KEY}/**")

    hf_snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(repo_root),
        allow_patterns=allow_patterns,
    )


def copy_fold_manifests(repo_root: Path) -> None:
    source_root = repo_root / "folds" / DATASET_KEY
    target_root = repo_root / "datasets" / DATASET_KEY

    if not source_root.exists():
        raise FileNotFoundError(
            f"Downloaded fold manifest directory not found: {source_root}"
        )

    target_root.mkdir(parents=True, exist_ok=True)
    for fold_dir in sorted(source_root.glob("fold-*")):
        if not fold_dir.is_dir():
            continue
        out_dir = target_root / fold_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            src = fold_dir / split_name
            if src.exists():
                shutil.copy2(src, out_dir / split_name)


def verify_layout(repo_root: Path, skip_images: bool) -> None:
    required = [
        repo_root / "prompt_banks" / f"{DATASET_KEY}.json",
        repo_root / "datasets" / DATASET_KEY / "fold-5" / "train.jsonl",
        repo_root / "datasets" / DATASET_KEY / "fold-5" / "val.jsonl",
        repo_root / "datasets" / DATASET_KEY / "fold-5" / "test.jsonl",
    ]
    if not skip_images:
        required.append(repo_root / "images" / DATASET_KEY)

    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Setup did not create expected paths:\n{formatted}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    repo_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from Hugging Face dataset: {args.repo_id}")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        repo_root=repo_root,
        skip_images=args.skip_images,
    )

    print("Copying fold manifests into datasets/microscopy_lurcher/")
    copy_fold_manifests(repo_root)
    verify_layout(repo_root, skip_images=args.skip_images)

    print("Lurcher data setup complete.")
    if args.skip_images:
        print("Images: skipped")
    else:
        print(f"Images: {repo_root / 'images' / DATASET_KEY}")
    print(f"Folds: {repo_root / 'datasets' / DATASET_KEY}")
    print(f"Prompt bank: {repo_root / 'prompt_banks' / (DATASET_KEY + '.json')}")


if __name__ == "__main__":
    main()
