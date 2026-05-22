#!/usr/bin/env python3
"""
Re-query only invalid (-1) test predictions for a finished/incomplete fold run.

This script:
1) Loads existing test predictions for a fold + selection artifact token.
2) Finds rows with invalid predictions (typically pred_label_raw == -1).
3) Re-queries only those images in batches via vlm_query.
4) Updates test_results payload incrementally (resume-safe).
5) Recomputes and updates results/.../results_selection=<token>.json test summary.
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main import APT, _selection_artifact_token, vlm_query, with_selection_suffix


def _resolve_root(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(os.path.dirname(__file__), path_value)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _backup_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_requery_invalid_{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def _maybe_relpath(path: str) -> str:
    try:
        cwd = Path.cwd().resolve()
        rel = Path(path).resolve().relative_to(cwd)
        return str(rel)
    except Exception:
        return str(path)


def _selection_base_from_token(selection_token: str) -> str:
    token = str(selection_token).strip().lower()
    if token.startswith("dts_") or token == "dts":
        return "dts"
    for base in ("mdl", "entropy", "random"):
        if token == base or token.startswith(f"{base}_"):
            return base
    return token.split("_", 1)[0]


def _discover_selection_tokens(test_results_root: str, dataset: str, fold: int) -> List[str]:
    root = Path(test_results_root) / dataset
    if not root.exists():
        return []
    matches: List[str] = []
    for child in sorted(root.glob("selection_method=*")):
        if not child.is_dir():
            continue
        fold_file = child / f"fold-{int(fold)}.json"
        if fold_file.exists():
            name = child.name
            token = name.split("selection_method=", 1)[-1]
            if token:
                matches.append(token)
    return matches


def _resolve_selection_token(
    selection_method: str,
    explicit_token: Optional[str],
    dataset: str,
    fold: int,
    test_results_root: str,
    initial_batch_size: int,
    candidate_pool_size: Optional[int],
    dts_clip_model_name: Optional[str],
) -> str:
    if explicit_token:
        return str(explicit_token).strip()

    method_raw = str(selection_method).strip()
    lower = method_raw.lower()
    if "candidate-size=" in lower or "_b=" in lower or lower.startswith("dts_"):
        return method_raw

    available = _discover_selection_tokens(test_results_root, dataset, fold)
    prefix_matches = [tok for tok in available if tok.lower().startswith(lower)]
    exact_matches = [tok for tok in available if tok.lower() == lower]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ValueError(
            f"Multiple selection artifacts matched '{selection_method}' for fold-{fold}: "
            f"{prefix_matches}. Pass --selection_artifact_token explicitly."
        )

    return _selection_artifact_token(
        selection_method=selection_method,
        dts_clip_model_name=dts_clip_model_name,
        active_set_batch_size=initial_batch_size,
        candidate_pool_size=candidate_pool_size,
    )


def _expand_fold_placeholder(path_template: str, fold: int) -> str:
    return str(path_template).replace("{fold}", str(int(fold)))


def _resolve_candidate_paths(base_path: str, selection_token: str) -> List[str]:
    cands: List[str] = []
    base = str(base_path).strip()
    cands.append(base)
    cands.append(with_selection_suffix(base, selection_token))
    out: List[str] = []
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _load_prompt_set_from_file(path: str) -> List[Tuple[str, str]]:
    data = _read_json(path)
    prompt_set: List[Tuple[str, str]] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                prompt_set.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                image_path = str(item.get("image_path", "")).strip()
                if not image_path:
                    continue
                caption = item.get("caption")
                if caption is None:
                    rationale = item.get("rationale")
                    if rationale is None:
                        rationale = item.get("explanation", "")
                    label = str(item.get("class", "")).strip()
                    caption = (f"{rationale} C: {label}").strip() if label else str(rationale)
                prompt_set.append((image_path, str(caption)))
        return prompt_set

    if isinstance(data, dict):
        items = data.get("prompt")
        if not isinstance(items, list):
            items = data.get("prompts", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                image_path = str(item.get("image_path", "")).strip()
                if not image_path:
                    continue
                rationale = item.get("rationale")
                if rationale is None:
                    rationale = item.get("explanation", "")
                label = str(item.get("class", "")).strip()
                caption = (f"{rationale} C: {label}").strip() if label else str(rationale)
                prompt_set.append((image_path, caption))

    return prompt_set


def _looks_invalid_prediction(record: Dict[str, Any]) -> bool:
    pred_raw = record.get("pred_label_raw")
    pred_idx = record.get("pred_label_idx")
    pred_label = str(record.get("pred_label", "")).strip().lower()
    caption = str(record.get("caption", "")).lower()

    if isinstance(pred_raw, Integral) and int(pred_raw) == -1:
        return True
    if isinstance(pred_raw, str) and pred_raw.strip() == "-1":
        return True
    if isinstance(pred_idx, Integral) and int(pred_idx) < 0:
        return True
    if pred_idx is None and pred_label == "invalid":
        return True
    if "marked invalid" in caption and "excluded from accuracy metrics" in caption:
        return True
    return False


def _infer_label_map(records: Sequence[Dict[str, Any]], fallback_dataset: str, provided: Optional[List[str]]) -> List[str]:
    if provided:
        labels = [str(x).strip() for x in provided if str(x).strip()]
        if labels:
            return labels

    idx_to_name: Dict[int, str] = {}
    for rec in records:
        idx = rec.get("ground_truth_label_idx")
        name = rec.get("ground_truth_label")
        if isinstance(idx, Integral) and name is not None:
            idx_to_name[int(idx)] = str(name)
    if idx_to_name:
        return [idx_to_name[k] for k in sorted(idx_to_name.keys())]

    ordered: List[str] = []
    for rec in records:
        name = str(rec.get("ground_truth_label", "")).strip()
        if name and name not in ordered:
            ordered.append(name)
    if ordered:
        return ordered

    if str(fallback_dataset).strip().lower() == "microscopy_lurcher":
        return ["wild", "lurcher"]
    return ["class0", "class1"]


def _resolve_target_round_and_records(
    test_payload: Dict[str, Any],
    requested_round: Optional[int],
) -> Tuple[int, str, List[Dict[str, Any]]]:
    round_predictions = test_payload.get("round_predictions", {})
    key_to_round: Dict[str, int] = {}
    if isinstance(round_predictions, dict):
        for key in round_predictions.keys():
            m = re.match(r"^round_(\d+)$", str(key))
            if m:
                key_to_round[str(key)] = int(m.group(1))

    if requested_round is not None:
        target_round = int(requested_round)
        round_key = f"round_{target_round:02d}"
        records = round_predictions.get(round_key, []) if isinstance(round_predictions, dict) else []
        if not isinstance(records, list) or not records:
            fallback = test_payload.get("predictions", [])
            if (
                isinstance(fallback, list)
                and int(test_payload.get("round", -1)) == target_round
                and fallback
            ):
                records = fallback
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"No prediction records found for round {target_round}. "
                f"Available round keys: {sorted(key_to_round.keys())}"
            )
        return target_round, round_key, records

    # Auto-resolve latest round from round_predictions, then fallback to payload["round"].
    if key_to_round:
        round_key = max(key_to_round.keys(), key=lambda k: key_to_round[k])
        target_round = key_to_round[round_key]
        records = round_predictions.get(round_key, [])
        if isinstance(records, list) and records:
            return target_round, round_key, records

    payload_round = test_payload.get("round")
    if isinstance(payload_round, Integral):
        target_round = int(payload_round)
        round_key = f"round_{target_round:02d}"
        records = test_payload.get("predictions", [])
        if isinstance(records, list) and records:
            return target_round, round_key, records

    raise ValueError("Could not resolve target round/predictions from test payload.")


def _persist_test_payload(
    test_payload: Dict[str, Any],
    test_predictions_path: str,
    round_num: int,
    round_key: str,
    records: List[Dict[str, Any]],
) -> None:
    test_payload.setdefault("round_predictions", {})
    test_payload["round_predictions"][round_key] = records
    test_payload["round"] = int(round_num)
    test_payload["predictions"] = records
    _write_json(test_predictions_path, test_payload)


def _build_test_metrics(
    apt: APT,
    records: Sequence[Dict[str, Any]],
    label_map: List[str],
    round_num: int,
    test_predictions_path: str,
) -> Dict[str, Any]:
    eval_paths: List[str] = []
    y_true_labels: List[int] = []
    y_pred_labels: List[Optional[int]] = []

    sampled_per_class: Dict[str, int] = {str(lbl): 0 for lbl in label_map}
    predicted_per_class: Dict[str, int] = {str(lbl): 0 for lbl in label_map}
    predicted_per_class["unknown"] = 0

    for rec in records:
        image_path = str(rec.get("image_path", "")).strip()
        if not image_path:
            continue

        y_true = rec.get("ground_truth_label_idx")
        if not isinstance(y_true, Integral):
            y_true = apt._infer_label_from_path(image_path, label_map)
        if y_true is None or int(y_true) < 0 or int(y_true) >= len(label_map):
            continue
        y_true_idx = int(y_true)

        pred_raw = rec.get("pred_label_raw")
        if pred_raw is None:
            pred_raw = rec.get("pred_label_idx")
        if pred_raw is None:
            pred_raw = rec.get("pred_label")
        y_pred = apt._normalize_prediction_label(pred_raw, label_map)

        eval_paths.append(image_path)
        y_true_labels.append(y_true_idx)
        y_pred_labels.append(int(y_pred) if y_pred is not None else None)
        sampled_per_class[str(label_map[y_true_idx])] += 1

        if y_pred is None or int(y_pred) < 0 or int(y_pred) >= len(label_map):
            predicted_per_class["unknown"] += 1
        else:
            predicted_per_class[str(label_map[int(y_pred)])] += 1

    class_metrics = apt._summarize_class_metrics(
        image_paths=eval_paths,
        y_true_labels=y_true_labels,
        y_pred_labels=y_pred_labels,
        label_map=list(label_map),
        ignore_invalid_predictions=True,
    )

    return {
        "round": int(round_num),
        "test_subset_size": int(len(eval_paths)),
        "sampled_per_class": sampled_per_class,
        "predicted_per_class": predicted_per_class,
        "class_totals": class_metrics["class_totals"],
        "class_correct": class_metrics["class_correct"],
        "case_ids_by_class": class_metrics["case_ids_by_class"],
        "case_ids_correct_by_class": class_metrics["case_ids_correct_by_class"],
        "class_accuracy_pct": class_metrics["class_accuracy_pct"],
        "avg_class_accuracy_pct": class_metrics["avg_class_accuracy_pct"],
        "test_predictions_path": _maybe_relpath(test_predictions_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-query only invalid (-1) test predictions and refresh summary metrics."
    )
    parser.add_argument("--dataset", type=str, default="microscopy_lurcher")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--selection_method",
        type=str,
        required=True,
        help="Base method (mdl/entropy/dts/random) or full artifact token.",
    )
    parser.add_argument(
        "--selection_artifact_token",
        type=str,
        default=None,
        help="Optional explicit artifact token (e.g., random_b=10_candidate-size=100).",
    )
    parser.add_argument("--round", type=int, default=None, help="Round to patch; default is latest available.")

    parser.add_argument("--test_results_root", type=str, default="test_results")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint_fold={fold}.json",
        help="Checkpoint template; can include {fold}.",
    )
    parser.add_argument(
        "--prompt_set_path",
        type=str,
        default="final_prompt_set_fold={fold}.json",
        help="Prompt-set template fallback; can include {fold}.",
    )

    parser.add_argument("--system_prompt_1", type=str, default="system_prompts/microscopy_lurcher_1.md")
    parser.add_argument("--system_prompt_2", type=str, default="system_prompts/microscopy_lurcher_2.md")
    parser.add_argument("--label_map", nargs="*", type=str, default=None)

    parser.add_argument("--initial_batch_size", type=int, default=10)
    parser.add_argument("--candidate_pool_size", type=int, default=None)
    parser.add_argument("--dts_clip_model_name", type=str, default=None)

    parser.add_argument("--vlm_query_batch_size", type=int, default=5)
    parser.add_argument("--vlm_timeout_s", type=float, default=120.0)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--vlm_log_path", type=str, default=None)

    parser.add_argument("--max_requery", type=int, default=None, help="Optional cap on invalid rows to re-query.")
    parser.add_argument("--dry_run", action="store_true", help="Inspect only, no API calls and no writes.")
    parser.add_argument("--no_backup", action="store_true", help="Skip creating .bak files before writes.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    test_results_root = _resolve_root(args.test_results_root)
    results_root = _resolve_root(args.results_root)

    selection_token = _resolve_selection_token(
        selection_method=args.selection_method,
        explicit_token=args.selection_artifact_token,
        dataset=args.dataset,
        fold=args.fold,
        test_results_root=test_results_root,
        initial_batch_size=int(args.initial_batch_size),
        candidate_pool_size=args.candidate_pool_size,
        dts_clip_model_name=args.dts_clip_model_name,
    )
    selection_base = _selection_base_from_token(selection_token)

    test_predictions_path = os.path.join(
        test_results_root,
        args.dataset,
        f"selection_method={selection_token}",
        f"fold-{int(args.fold)}.json",
    )
    if not os.path.exists(test_predictions_path):
        print(f"ERROR: test predictions file not found: {test_predictions_path}")
        return 1

    results_summary_path = os.path.join(
        results_root,
        args.dataset,
        f"fold-{int(args.fold)}",
        f"results_selection={selection_token}.json",
    )

    test_payload_raw = _read_json(test_predictions_path)
    if not isinstance(test_payload_raw, dict):
        print(f"ERROR: invalid JSON shape in {test_predictions_path} (expected object).")
        return 1
    test_payload: Dict[str, Any] = test_payload_raw

    target_round, round_key, records = _resolve_target_round_and_records(
        test_payload=test_payload,
        requested_round=args.round,
    )
    records = [dict(x) for x in records if isinstance(x, dict)]
    if not records:
        print(f"ERROR: no records found for round {target_round} in {test_predictions_path}")
        return 1

    label_map = _infer_label_map(records, args.dataset, args.label_map)
    print(
        f"Resolved fold={args.fold}, selection_token={selection_token}, round={target_round}, "
        f"records={len(records)}, label_map={label_map}"
    )

    apt = APT(
        system_prompt_1_path=args.system_prompt_1,
        system_prompt_2_path=args.system_prompt_2,
        selection_method=selection_base,
        candidate_pool_size=(args.candidate_pool_size if args.candidate_pool_size is not None else -1),
        active_set_batch_size=args.initial_batch_size,
        fold=int(args.fold),
    )

    checkpoint_template = _expand_fold_placeholder(args.checkpoint_path, args.fold)
    prompt_set_template = _expand_fold_placeholder(args.prompt_set_path, args.fold)
    checkpoint_candidates = _resolve_candidate_paths(checkpoint_template, selection_token)
    prompt_set_candidates = _resolve_candidate_paths(prompt_set_template, selection_token)

    loaded_from = ""
    for ckpt_path in checkpoint_candidates:
        if not os.path.exists(ckpt_path):
            continue
        try:
            apt.load_checkpoint(ckpt_path)
            if apt.prompt_set:
                loaded_from = f"checkpoint:{ckpt_path}"
                break
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")

    if not apt.prompt_set:
        for prompt_path in prompt_set_candidates:
            if not os.path.exists(prompt_path):
                continue
            try:
                prompt_set = _load_prompt_set_from_file(prompt_path)
                if prompt_set:
                    apt.prompt_set = prompt_set
                    loaded_from = f"prompt_set:{prompt_path}"
                    break
            except Exception as e:
                print(f"Warning: failed to load prompt set {prompt_path}: {e}")

    if not apt.prompt_set:
        print(
            "ERROR: could not load prompt_set from checkpoint or final prompt set. "
            f"Checked checkpoints={checkpoint_candidates}, prompt_sets={prompt_set_candidates}"
        )
        return 1
    print(f"Using prompt_set size={len(apt.prompt_set)} from {loaded_from}")

    invalid_indices = [i for i, rec in enumerate(records) if _looks_invalid_prediction(rec)]
    if args.max_requery is not None:
        invalid_indices = invalid_indices[: max(0, int(args.max_requery))]
    print(f"Invalid predictions detected for re-query: {len(invalid_indices)}")

    if not args.dry_run and not args.no_backup:
        test_backup = _backup_file(test_predictions_path)
        results_backup = _backup_file(results_summary_path) if os.path.exists(results_summary_path) else None
        if test_backup:
            print(f"Backup written: {test_backup}")
        if results_backup:
            print(f"Backup written: {results_backup}")

    if invalid_indices and not args.dry_run:
        qbatch = max(1, int(args.vlm_query_batch_size))
        for start in range(0, len(invalid_indices), qbatch):
            batch_indices = invalid_indices[start:start + qbatch]
            batch_paths = [str(records[i].get("image_path", "")).strip() for i in batch_indices]
            if any(not p for p in batch_paths):
                print(f"Warning: skipping batch with empty image_path at indices {batch_indices}")
                continue

            batch_num = (start // qbatch) + 1
            batch_total = (len(invalid_indices) + qbatch - 1) // qbatch
            print(f"Re-query batch {batch_num}/{batch_total} ({len(batch_paths)} images)")

            batch_preds = vlm_query(
                batch_paths,
                apt.S_1,
                apt.S_2_template,
                apt.prompt_set,
                stochastic=False,
                label_map=list(label_map),
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                vlm_timeout_s=args.vlm_timeout_s,
                vlm_log_path=args.vlm_log_path,
            )
            if not isinstance(batch_preds, list):
                batch_preds = [batch_preds]
            if len(batch_preds) != len(batch_indices):
                print(
                    f"ERROR: prediction length mismatch in batch {batch_num}: "
                    f"{len(batch_preds)} vs {len(batch_indices)}"
                )
                return 1

            for idx, pred_pair in zip(batch_indices, batch_preds):
                rec = records[idx]
                pred_raw = pred_pair[0] if isinstance(pred_pair, tuple) and len(pred_pair) > 0 else None
                rationale = pred_pair[1] if isinstance(pred_pair, tuple) and len(pred_pair) > 1 else ""

                image_path = str(rec.get("image_path", "")).strip()
                y_true_raw = rec.get("ground_truth_label_idx")
                y_true_idx: Optional[int]
                if isinstance(y_true_raw, Integral):
                    y_true_idx = int(y_true_raw)
                else:
                    y_true_idx = apt._infer_label_from_path(image_path, label_map)

                y_pred = apt._normalize_prediction_label(pred_raw, label_map)
                y_pred_idx = int(y_pred) if y_pred is not None else None
                pred_label_name = (
                    str(label_map[y_pred_idx])
                    if y_pred_idx is not None and 0 <= y_pred_idx < len(label_map)
                    else "invalid"
                )

                rec["round"] = int(target_round)
                rec["pred_label_raw"] = pred_raw
                rec["pred_label_idx"] = y_pred_idx
                rec["pred_label"] = pred_label_name
                rec["caption"] = "" if rationale is None else str(rationale)
                rec["is_correct"] = bool(y_pred_idx == y_true_idx) if y_pred_idx is not None else False

                if y_true_idx is not None and 0 <= y_true_idx < len(label_map):
                    rec["ground_truth_label_idx"] = int(y_true_idx)
                    rec["ground_truth_label"] = str(label_map[y_true_idx])

            _persist_test_payload(
                test_payload=test_payload,
                test_predictions_path=test_predictions_path,
                round_num=target_round,
                round_key=round_key,
                records=records,
            )
    elif invalid_indices and args.dry_run:
        print("Dry-run enabled: skipped VLM calls and file writes.")

    # Always recompute summary from latest records if writing is enabled.
    if not args.dry_run:
        # Ensure payload references the in-memory records.
        _persist_test_payload(
            test_payload=test_payload,
            test_predictions_path=test_predictions_path,
            round_num=target_round,
            round_key=round_key,
            records=records,
        )

        test_metrics = _build_test_metrics(
            apt=apt,
            records=records,
            label_map=label_map,
            round_num=target_round,
            test_predictions_path=test_predictions_path,
        )

        summary_payload: Dict[str, Any] = {}
        if os.path.exists(results_summary_path):
            raw = _read_json(results_summary_path)
            if isinstance(raw, dict):
                summary_payload = raw

        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_payload.setdefault("dataset", args.dataset)
        summary_payload.setdefault("fold", int(args.fold))
        summary_payload.setdefault("selection_method", selection_base)
        summary_payload.setdefault("created_at", now_ts)
        summary_payload["updated_at"] = now_ts
        summary_payload["test"] = test_metrics
        _write_json(results_summary_path, summary_payload)

        remaining_invalid = sum(1 for rec in records if _looks_invalid_prediction(rec))
        print(
            f"Updated test predictions: {test_predictions_path}\n"
            f"Updated test summary: {results_summary_path}\n"
            f"Remaining invalid predictions: {remaining_invalid}\n"
            f"New avg_class_accuracy_pct: {test_metrics['avg_class_accuracy_pct']:.4f}"
        )
    else:
        remaining_invalid = sum(1 for rec in records if _looks_invalid_prediction(rec))
        print(f"Dry-run summary: remaining invalid predictions (unchanged) = {remaining_invalid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
